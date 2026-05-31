"""
train.py — DINOv2 (frozen) + BlockTypeHead for visible-block classification.

Task: given a video frame and the screen positions of visible non-air blocks
(ground truth visibility from Furnace), predict the block-state ID at each
visible position.

Ground truth: block state IDs from world_states.bin at positions where m=1.
Loss: cross-entropy over a compact vocab of the top-K most common visible block
states (built offline by build_vocab.py).

Metrics: top-1 accuracy, top-5 accuracy (per visible block instance).

Run build_vocab.py first to generate vocab.json, then:
    python train.py [--config config.yaml] [--device cuda:0]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pum root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from data.vis_dataset import (
    VisibleBlockDataset, visible_block_collate,
    uv_to_patch_idx, PATCHES_H, PATCHES_W,
    SMELTED_ROOT,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DINO_DIM = 768


def get_session_names(split: str, n_train: int = 73) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


def load_vocab(vocab_path: str | Path) -> tuple[np.ndarray, int, list[str]]:
    """
    Load vocab JSON. Returns (class_lut, n_classes, class_names).
    class_lut is a (65536,) int64 array: lut[state_id] = class_idx or -1.
    """
    with open(vocab_path) as f:
        v = json.load(f)
    lut = np.full(65536, -1, dtype=np.int64)
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        if sid < 65536:
            lut[sid] = cls
    return lut, v["n_classes"], v["classes"]


def sids_to_classes(sids: np.ndarray, class_lut: np.ndarray) -> np.ndarray:
    """Vectorised block-state ID → class index lookup. Non-scatter → -1."""
    return class_lut[np.clip(sids, 0, len(class_lut) - 1).astype(np.int32)]


def load_dino(device: torch.device) -> nn.Module:
    log.info("Loading DINOv2 ViT-B/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("DINOv2 loaded")
    return model


@torch.no_grad()
def dino_patch_grids(dino, frames: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) → (B, PATCHES_H, PATCHES_W, 768)."""
    out   = dino.forward_features(frames)
    patch = out["x_norm_patchtokens"]
    return patch.reshape(frames.shape[0], PATCHES_H, PATCHES_W, DINO_DIM)


def extract_vis_feats(patch_grids: torch.Tensor,
                      screen_uvs_list: list,
                      device: torch.device) -> torch.Tensor:
    """
    Look up DINOv2 patch features at each visible block's screen UV.
    Returns (N_total, 768) float32 on device.
    """
    all_feats = []
    for b, uvs in enumerate(screen_uvs_list):
        rows, cols = uv_to_patch_idx(uvs)
        all_feats.append(patch_grids[b, rows, cols, :])
    return torch.cat(all_feats, dim=0)


class BlockTypeHead(nn.Module):
    """
    Predicts visible block type from DINOv2 patch feature.

    No block-state embedding as input — the block state ID IS the target.
    The model must learn to recognise block textures from visual features alone.
    """
    def __init__(self, feat_dim: int, n_classes: int,
                 hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.net = nn.Linear(feat_dim, n_classes)

    def forward(self, dino_feats: torch.Tensor) -> torch.Tensor:
        return self.net(dino_feats)   # (N, n_classes) logits


@torch.no_grad()
def evaluate(dino, model: nn.Module, val_loader: DataLoader,
             class_lut: np.ndarray, device: torch.device) -> dict:
    model.eval()
    total = correct1 = correct5 = n_valid = 0

    for frames, uvs_list, sid_list in val_loader:
        frames = frames.to(device)
        grids  = dino_patch_grids(dino, frames)
        feats  = extract_vis_feats(grids, uvs_list, device)

        # Build class targets, skipping OOV blocks
        all_classes = np.concatenate([
            sids_to_classes(s, class_lut) for s in sid_list
        ])
        valid = all_classes >= 0
        if valid.sum() == 0:
            continue

        targets = torch.from_numpy(all_classes[valid]).to(device)
        logits  = model(feats[valid])

        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        top1 = top5[:, :1]

        correct1 += (top1 == targets.unsqueeze(1)).any(dim=1).sum().item()
        correct5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        n_valid  += len(targets)
        total    += len(all_classes)

    if n_valid == 0:
        return {"top1": 0.0, "top5": 0.0, "n_valid": 0, "oov_rate": 1.0}

    return {
        "top1":     correct1 / n_valid,
        "top5":     correct5 / n_valid,
        "n_valid":  n_valid,
        "oov_rate": 1.0 - (n_valid / max(total, 1)),
    }


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
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. Run build_vocab.py first.")
    class_lut, n_classes, class_names = load_vocab(vocab_path)
    log.info("Vocab loaded: %d classes from %s", n_classes, vocab_path)

    train_sessions = get_session_names("train", cfg["data"]["n_train"])
    val_sessions   = get_session_names("val",   cfg["data"]["n_train"])
    log.info("Train: %d sessions  Val: %d sessions",
             len(train_sessions), len(val_sessions))

    train_ds = VisibleBlockDataset(train_sessions, shuffle=True)
    val_ds   = VisibleBlockDataset(val_sessions,   shuffle=False)

    frame_bs = cfg["training"].get("frame_batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=frame_bs,
                              collate_fn=visible_block_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=frame_bs,
                              collate_fn=visible_block_collate, num_workers=2)

    dino = load_dino(device)

    m_cfg = cfg["model"]
    model = BlockTypeHead(
        feat_dim   = DINO_DIM,
        n_classes  = n_classes,
        hidden_dim = m_cfg.get("hidden_dim", 0),
        dropout    = m_cfg.get("dropout", 0.0),
    ).to(device)
    log.info("BlockTypeHead params: %d  n_classes: %d",
             sum(p.numel() for p in model.parameters()), n_classes)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    epochs          = cfg["training"]["epochs"]
    steps_per_epoch = (len(train_sessions) * 30) // frame_bs  # ~30 vis ticks/session avg
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["training"]["lr"],
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=cfg["training"]["lr_warmup_epochs"] / epochs,
    )

    ckpt_dir = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(
        project = cfg["wandb"]["project"],
        entity  = cfg["wandb"]["entity"],
        name    = cfg["wandb"]["run_name"],
        config  = cfg,
    )

    best_top1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_steps    = 0

        for frames, uvs_list, sid_list in train_loader:
            frames = frames.to(device)
            with torch.no_grad():
                grids = dino_patch_grids(dino, frames)
            feats = extract_vis_feats(grids, uvs_list, device)

            all_classes = np.concatenate([
                sids_to_classes(s, class_lut) for s in sid_list
            ])
            valid_mask = all_classes >= 0
            if valid_mask.sum() == 0:
                continue
            feats_valid = feats[valid_mask]
            targets     = torch.from_numpy(all_classes[valid_mask]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats_valid)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if n_steps < steps_per_epoch:
                scheduler.step()

            train_loss += loss.item()
            n_steps    += 1

        train_loss /= max(n_steps, 1)
        metrics = evaluate(dino, model, val_loader, class_lut, device)

        log.info(
            "Epoch %d/%d  loss=%.4f  val_top1=%.4f  val_top5=%.4f  oov=%.3f",
            epoch, epochs, train_loss,
            metrics["top1"], metrics["top5"], metrics["oov_rate"],
        )

        run.log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "val/top1":       metrics["top1"],
            "val/top5":       metrics["top5"],
            "val/oov_rate":   metrics["oov_rate"],
            "val/n_valid":    metrics["n_valid"],
            "lr":             scheduler.get_last_lr()[0],
        })

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "val_top1":  best_top1,
                "n_classes": n_classes,
                "vocab":     cfg["data"]["vocab_path"],
            }, ckpt_dir / "best.pt")

        if epoch % cfg["checkpointing"]["save_every_epochs"] == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

    log.info("Done. Best val_top1: %.4f", best_top1)
    run.finish()


if __name__ == "__main__":
    main()
