"""
train.py — SigLIP-B/16 NaFlex (frozen) + per-block MLP head with 3D position.

Key differences from viblock_run2 (per-patch linear):
  - Per-BLOCK prediction: each visible block queries its own screen patch
  - 3D position embedding: (x,y,z)/16 concatenated to patch feature disambiguates
    blocks at the same UV but different depths
  - MLP head (512 hidden) instead of linear
  - CosineAnnealingLR instead of OneCycleLR (smoother for a tiny head)

Architecture per block:
  SigLIP patch feature at (row, col)     (768,)
  Block position (xi-16, yi-16, zi-16)/16  (3,)
  → cat → Linear(771, 512) → GELU → Dropout → Linear(512, 91)

Ground truth: precomputed viblock2_data.npz (run precompute_data.py first).
Loss: CrossEntropyLoss, ignore_index=-1 (OOV and air).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SIGLIP_W = 40
SIGLIP_H = 22
DATA_STEM = "viblock2_data.npz"


# ── Vocab ─────────────────────────────────────────────────────────────────────

def load_vocab(vocab_path: Path) -> tuple[np.ndarray, int, list[str]]:
    with open(vocab_path) as f:
        v = json.load(f)
    lut = np.full(65536, -1, dtype=np.int64)
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        if 0 < sid < 65536:
            lut[sid] = int(cls)
    return lut, int(v["n_classes"]), v["classes"]


def get_session_names(split: str, n_train: int) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / DATA_STEM).exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ViBlock2Dataset(IterableDataset):
    """
    Fast dataset reading precomputed viblock2_data.npz (no world_states.bin).

    Yields (frame_rgb_np, vis_uvs, vis_sids, vis_xyz_rel) per tick.
      frame_rgb_np : (360, 640, 3) uint8
      vis_uvs      : (N, 2)        float32  screen UVs in [0,1]
      vis_sids     : (N,)          int32    block-state IDs
      vis_xyz_rel  : (N, 3)        float32  (xi-16,yi-16,zi-16)/16
    """

    def __init__(self, session_names: list[str], shuffle: bool = True):
        self.session_names = session_names
        self.shuffle = shuffle

    def __iter__(self):
        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per = int(np.ceil(len(sessions) / worker_info.num_workers))
            start = worker_info.id * per
            sessions = sessions[start: start + per]

        for session in sessions:
            data_path = SMELTED_ROOT / session / "labels" / DATA_STEM
            video_path = RAW_ROOT / session / "video.mp4"
            if not data_path.exists() or not video_path.exists():
                continue

            data          = np.load(data_path)
            tick_indices  = data["tick_indices"]   # (K,) int32
            offsets       = data["block_offsets"]  # (K+1,) int32
            block_uvs     = data["block_uvs"]      # (N, 2) float32
            block_sids    = data["block_sids"]     # (N,) uint16
            block_xyz     = data["block_xyz"]      # (N, 3) float32

            cap     = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            order = list(range(len(tick_indices)))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                tick_idx = int(tick_indices[k])
                if tick_idx >= n_frames:
                    continue

                s, e = int(offsets[k]), int(offsets[k + 1])
                if s >= e:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, tick_idx)
                ok, bgr = cap.read()
                if not ok:
                    continue

                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield (frame_rgb,
                       block_uvs[s:e],
                       block_sids[s:e].astype(np.int32),
                       block_xyz[s:e])

            cap.release()


def collate_fn(batch):
    frames_np = [b[0] for b in batch]
    uvs_list = [b[1] for b in batch]
    sids_list = [b[2] for b in batch]
    xyz_list = [torch.from_numpy(b[3]) for b in batch]
    return frames_np, uvs_list, sids_list, xyz_list


# ── Fast dataset: precomputed SigLIP features (no video, no backbone) ─────────

FEAT_STEM = "viblock2_feats.npz"


class ViBlock2FeatDataset(IterableDataset):
    """
    Ultra-fast dataset using precomputed SigLIP block features.
    Run precompute_siglip_feats.py first.

    Yields (block_feats, block_sids, block_xyz) per tick.
      block_feats : (N, 768) float32
      block_sids  : (N,)     int32
      block_xyz   : (N, 3)   float32
    """

    def __init__(self, session_names: list[str], shuffle: bool = True):
        self.session_names = session_names
        self.shuffle = shuffle

    def __iter__(self):
        sessions = list(self.session_names)
        if self.shuffle:
            np.random.shuffle(sessions)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per = int(np.ceil(len(sessions) / worker_info.num_workers))
            start = worker_info.id * per
            sessions = sessions[start: start + per]

        for session in sessions:
            labels    = SMELTED_ROOT / session / "labels"
            data_path = labels / DATA_STEM
            feat_path = labels / FEAT_STEM
            if not data_path.exists() or not feat_path.exists():
                continue

            data        = np.load(data_path)
            offsets     = data["block_offsets"]   # (K+1,)
            block_sids  = data["block_sids"]      # (N,) uint16
            block_xyz   = data["block_xyz"]        # (N, 3) float32

            feat_data   = np.load(feat_path)
            block_feats = feat_data["block_feats"] # (N, 768) float16

            K     = len(offsets) - 1
            order = list(range(K))
            if self.shuffle:
                np.random.shuffle(order)

            for k in order:
                s, e = int(offsets[k]), int(offsets[k + 1])
                if s >= e:
                    continue
                yield (block_feats[s:e].astype(np.float32),
                       block_sids[s:e].astype(np.int32),
                       block_xyz[s:e])

    @staticmethod
    def available(session_names: list[str]) -> bool:
        """True iff all sessions have precomputed feature files."""
        return all(
            (SMELTED_ROOT / s / "labels" / FEAT_STEM).exists()
            for s in session_names
        )


def feat_collate_fn(batch):
    feats_list = [torch.from_numpy(b[0]) for b in batch]
    sids_list  = [b[1] for b in batch]
    xyz_list   = [torch.from_numpy(b[2]) for b in batch]
    return feats_list, sids_list, xyz_list


# ── SigLIP ────────────────────────────────────────────────────────────────────

def load_siglip(backbone_name: str, device: torch.device):
    from transformers import AutoModel, AutoProcessor
    log.info("Loading SigLIP: %s", backbone_name)
    model = AutoModel.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    vision = model.vision_model.eval().to(device)
    for p in vision.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(backbone_name, cache_dir="/data/vvm33/hf_cache")
    log.info("SigLIP loaded (%d params frozen)", sum(p.numel() for p in vision.parameters()))
    return vision, proc


@torch.no_grad()
def siglip_patch_grid(vision, proc, frames_np: list, device: torch.device) -> torch.Tensor:
    """List of (360,640,3) uint8 → (B, 22, 40, 768) patch feature grid."""
    from PIL import Image
    pil_images = [Image.fromarray(f) for f in frames_np]
    inputs = proc(images=pil_images, return_tensors="pt", max_num_patches=880)
    pixel_values = inputs["pixel_values"].to(device)
    attention_mask = inputs["pixel_attention_mask"].to(device)
    spatial_shapes = inputs["spatial_shapes"].to(device)
    outputs = vision(pixel_values=pixel_values,
                     attention_mask=attention_mask,
                     spatial_shapes=spatial_shapes)
    h = outputs.last_hidden_state   # (B, 880, 768)
    return h.view(len(frames_np), SIGLIP_H, SIGLIP_W, 768)


def uv_to_patch(uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) screen UVs → (row, col) into the 22×40 SigLIP patch grid."""
    col = np.floor(uvs[:, 0] * SIGLIP_W).astype(np.int32).clip(0, SIGLIP_W - 1)
    row = np.floor(uvs[:, 1] * SIGLIP_H).astype(np.int32).clip(0, SIGLIP_H - 1)
    return row, col


def extract_block_feats(patch_grid: torch.Tensor,
                        uvs_list: list) -> torch.Tensor:
    """
    Look up SigLIP patch features for each visible block.
    Returns concatenated (N_total, 768) tensor.
    """
    all_feats = []
    for b, uvs in enumerate(uvs_list):
        rows, cols = uv_to_patch(uvs)
        all_feats.append(patch_grid[b, rows, cols, :])
    return torch.cat(all_feats, dim=0)   # (N_total, 768)


# ── Model ─────────────────────────────────────────────────────────────────────

class ViBlock2Head(nn.Module):
    """Per-block MLP: (patch_feat ∥ xyz_rel) → class logits."""

    def __init__(self, feat_dim: int, xyz_dim: int, hidden_dim: int,
                 n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + xyz_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feats, xyz], dim=-1))


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(vision, proc, head: ViBlock2Head,
             val_loader: DataLoader, class_lut: np.ndarray,
             device: torch.device) -> dict:
    head.eval()
    correct1 = correct5 = n_valid = n_total = 0

    for frames_np, uvs_list, sids_list, xyz_list in val_loader:
        grid = siglip_patch_grid(vision, proc, frames_np, device)
        feats = extract_block_feats(grid, uvs_list)
        xyz = torch.cat(xyz_list, dim=0).to(device)

        all_sids = np.concatenate([s for s in sids_list])
        all_classes = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
        valid = all_classes >= 0
        n_total += len(all_classes)

        if valid.sum() == 0:
            continue

        targets = torch.from_numpy(all_classes[valid]).to(device)
        logits = head(feats[valid], xyz[valid])

        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        correct1 += (top5[:, :1] == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        n_valid += int(valid.sum())

    return {
        "top1": correct1 / max(n_valid, 1),
        "top5": correct5 / max(n_valid, 1),
        "n_valid": n_valid,
        "oov_rate": 1 - n_valid / max(n_total, 1),
    }


@torch.no_grad()
def evaluate_cached(head: ViBlock2Head, val_loader: DataLoader,
                    class_lut: np.ndarray, device: torch.device) -> dict:
    """Fast eval using precomputed features — no SigLIP needed."""
    head.eval()
    correct1 = correct5 = n_valid = n_total = 0

    for feats_list, sids_list, xyz_list in val_loader:
        feats = torch.cat(feats_list, dim=0).to(device)
        xyz   = torch.cat(xyz_list,   dim=0).to(device)
        all_sids    = np.concatenate(sids_list)
        all_classes = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
        valid = all_classes >= 0
        n_total += len(all_classes)
        if valid.sum() == 0:
            continue
        targets = torch.from_numpy(all_classes[valid]).to(device)
        logits  = head(feats[valid], xyz[valid])
        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        correct1 += (top5[:, :1] == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        n_valid  += int(valid.sum())

    return {
        "top1":     correct1 / max(n_valid, 1),
        "top5":     correct5 / max(n_valid, 1),
        "n_valid":  n_valid,
        "oov_rate": 1 - n_valid / max(n_total, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_path = Path(cfg["data"]["vocab_path"])
    class_lut, n_classes, class_names = load_vocab(vocab_path)
    log.info("Vocab: %d classes", n_classes)

    n_train = cfg["data"]["n_train"]
    train_sessions = get_session_names("train", n_train)
    val_sessions = get_session_names("val", n_train)
    log.info("Train: %d sessions  Val: %d sessions",
             len(train_sessions), len(val_sessions))

    # ── Choose fast (precomputed feats) or slow (live SigLIP) path ───────────
    use_cached = (ViBlock2FeatDataset.available(train_sessions) and
                  ViBlock2FeatDataset.available(val_sessions))

    if use_cached:
        log.info("Using precomputed SigLIP features (fast path)")
        train_ds    = ViBlock2FeatDataset(train_sessions, shuffle=True)
        val_ds      = ViBlock2FeatDataset(val_sessions,   shuffle=False)
        vision, proc = None, None
    else:
        log.info("Precomputed features not found — running live SigLIP (slow path). "
                 "Run precompute_siglip_feats.py to speed up training.")
        train_ds    = ViBlock2Dataset(train_sessions, shuffle=True)
        val_ds      = ViBlock2Dataset(val_sessions,   shuffle=False)
        vision, proc = load_siglip(cfg["model"]["backbone"], device)

    bs = cfg["training"]["batch_size"]
    if use_cached:
        train_loader = DataLoader(train_ds, batch_size=bs,
                                  collate_fn=feat_collate_fn, num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=bs,
                                  collate_fn=feat_collate_fn, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs,
                                  collate_fn=collate_fn, num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=bs,
                                  collate_fn=collate_fn, num_workers=2)

    m = cfg["model"]
    head = ViBlock2Head(
        feat_dim=m["feat_dim"],
        xyz_dim=m["xyz_dim"],
        hidden_dim=m["hidden_dim"],
        n_classes=n_classes,
        dropout=m["dropout"],
    ).to(device)
    log.info("ViBlock2Head params: %d  (feat %d + xyz %d → hidden %d → %d classes)",
             sum(p.numel() for p in head.parameters()),
             m["feat_dim"], m["xyz_dim"], m["hidden_dim"], n_classes)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["wandb"]["run_name"],
        config={**cfg, "use_cached_feats": use_cached},
    )

    best_top1 = 0.0

    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        n_steps = 0

        if use_cached:
            for feats_list, sids_list, xyz_list in train_loader:
                feats = torch.cat(feats_list, dim=0).to(device)  # (N, 768)
                xyz   = torch.cat(xyz_list,   dim=0).to(device)  # (N, 3)
                all_sids    = np.concatenate(sids_list)
                all_classes = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
                valid = all_classes >= 0
                if valid.sum() == 0:
                    continue
                targets = torch.from_numpy(all_classes[valid]).to(device)
                logits  = head(feats[valid], xyz[valid])
                loss    = criterion(logits, targets)
                if torch.isnan(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_steps    += 1
        else:
            for frames_np, uvs_list, sids_list, xyz_list in train_loader:
                with torch.no_grad():
                    grid = siglip_patch_grid(vision, proc, frames_np, device)
                feats = extract_block_feats(grid, uvs_list)
                xyz   = torch.cat(xyz_list, dim=0).to(device)
                all_sids    = np.concatenate([s for s in sids_list])
                all_classes = class_lut[np.clip(all_sids, 0, len(class_lut) - 1)]
                valid = all_classes >= 0
                if valid.sum() == 0:
                    continue
                targets = torch.from_numpy(all_classes[valid]).to(device)
                logits  = head(feats[valid], xyz[valid])
                loss    = criterion(logits, targets)
                if torch.isnan(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_steps    += 1

        scheduler.step()
        train_loss /= max(n_steps, 1)
        if use_cached:
            metrics = evaluate_cached(head, val_loader, class_lut, device)
        else:
            metrics = evaluate(vision, proc, head, val_loader, class_lut, device)

        log.info("Epoch %d/%d  loss=%.4f  top1=%.4f  top5=%.4f  oov=%.3f  n=%d",
                 epoch, epochs, train_loss,
                 metrics["top1"], metrics["top5"], metrics["oov_rate"], metrics["n_valid"])

        run.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/top1": metrics["top1"],
            "val/top5": metrics["top5"],
            "val/oov_rate": metrics["oov_rate"],
            "val/n_valid": metrics["n_valid"],
            "lr": scheduler.get_last_lr()[0],
        })

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save({
                "epoch": epoch,
                "model": head.state_dict(),
                "top1": best_top1,
                "top5": metrics["top5"],
                "n_classes": n_classes,
                "vocab": str(vocab_path),
                "backbone": cfg["model"]["backbone"],
            }, ckpt_dir / "best.pt")
            log.info("  → best top1=%.4f saved", best_top1)

        if epoch % cfg["checkpointing"]["save_every_epochs"] == 0:
            torch.save({"epoch": epoch, "model": head.state_dict()},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

    log.info("Done. Best top1: %.4f", best_top1)
    run.finish()


if __name__ == "__main__":
    main()
