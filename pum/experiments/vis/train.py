"""
train.py — DINOv2 (frozen) + binary BlockHead for Furnace visibility prediction.

Task: given a video frame, predict which non-air voxels in the 32³ window are
visible (m=1) vs occluded/behind-camera (m=0), matching Furnace's raycaster.

Ground truth is loaded from per-tick npz files via load_tick_npz.

Ship criterion: val F1 >= 0.70 on in-frustum visibility prediction.

Usage:
    python train.py [--config config.yaml] [--device cuda:0]
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pum root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from data.vis_dataset import (
    OnlineFrameDataset, online_collate,
    uv_to_patch_idx, PATCHES_H, PATCHES_W,
    SMELTED_ROOT, RAW_ROOT,
    project_nonair_visibility, preprocess_frame_np,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DINO_DIM            = 768
N_PROBE_PER_SESSION = 3


def get_session_names(split: str, n_train: int = 73) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


def load_dino(device: torch.device) -> torch.nn.Module:
    log.info("Loading DINOv2 ViT-B/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("DINOv2 loaded")
    return model


@torch.no_grad()
def dino_patch_grids(dino, frames: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) frames → (B, PATCHES_H, PATCHES_W, 768)."""
    out   = dino.forward_features(frames)
    patch = out["x_norm_patchtokens"]
    return patch.reshape(frames.shape[0], PATCHES_H, PATCHES_W, DINO_DIM)


MAX_BLOCK_STATES = 65536   # covers all uint16 block-state IDs


def extract_block_feats(patch_grids: torch.Tensor,
                        screen_uvs_list: list,
                        sid_list: list,
                        device: torch.device) -> tuple:
    """
    Look up DINOv2 patch features at each voxel's screen UV.
    Returns (feats (N_total, 768), sids (N_total,) int64).
    """
    all_feats = []
    all_sids  = []
    for b, (uvs, sids) in enumerate(zip(screen_uvs_list, sid_list)):
        rows, cols = uv_to_patch_idx(uvs)
        feats = patch_grids[b, rows, cols, :]
        all_feats.append(feats)
        all_sids.append(torch.from_numpy(sids.astype(np.int64)).to(device))
    return torch.cat(all_feats, dim=0), torch.cat(all_sids, dim=0)


class BlockHead(nn.Module):
    """
    Binary visibility classifier: DINOv2 patch feature + block-state embedding → logit.

    Block-state embedding lets the model learn per-block-type appearance patterns:
    "this patch looks like stone, but we're querying a grass block → probably occluded."
    """
    def __init__(self, feat_dim: int, block_embed_dim: int = 64,
                 hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.block_embed = nn.Embedding(MAX_BLOCK_STATES, block_embed_dim, padding_idx=0)
        in_dim = feat_dim + block_embed_dim
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, dino_feats: torch.Tensor, block_sids: torch.Tensor):
        embed = self.block_embed(block_sids.clamp(0, MAX_BLOCK_STATES - 1))
        x = torch.cat([dino_feats, embed], dim=1)
        return self.net(x).squeeze(-1)   # (N,) logits


def compute_pos_weight(session_names: list[str],
                       device: torch.device) -> torch.Tensor:
    """Estimate BCE pos_weight = n_neg / n_pos from precomputed visibility files."""
    from io_helpers import load_world_states
    VIS_STEM = "visibility_s20.npz"
    log.info("Estimating pos_weight from visibility files …")
    n_pos = n_neg = 0
    for session in session_names[:5]:
        labels = SMELTED_ROOT / session / "labels"
        vis_path = labels / VIS_STEM
        if not vis_path.exists():
            continue
        px_arr, py_arr, pz_arr, blocks = load_world_states(labels)
        vis_data    = np.load(vis_path)
        tick_indices = vis_data["tick_indices"]
        visibility   = vis_data["visibility"]
        for k, tick_idx in enumerate(tick_indices[:50]):   # sample first 50 ticks
            nonair = blocks[tick_idx] > 0
            m = visibility[k]
            n_pos += int(m[nonair].sum())
            n_neg += int((~m.astype(bool))[nonair].sum())
    if n_pos == 0:
        log.warning("pos_weight estimation failed (no visibility files?), defaulting to 1.0")
        return torch.tensor(1.0, device=device)
    pw = n_neg / n_pos
    log.info("pos_weight = %.2f  (n_pos=%d  n_neg=%d)", pw, n_pos, n_neg)
    return torch.tensor(pw, dtype=torch.float32, device=device)


def save_probes(val_sessions: list[str], dino,
                probe_dir: Path, device: torch.device):
    """
    Create eval probe npz files for visualization.
    Each probe stores a preprocessed frame + metadata for a single tick.
    DINOv2 features are NOT pre-extracted; eval_vis runs DINOv2 live.
    Requires visibility_s20.npz from precompute_visibility.py.
    """
    from io_helpers import load_world_states
    VIS_STEM = "visibility_s20.npz"

    probe_dir.mkdir(parents=True, exist_ok=True)
    existing = list(probe_dir.glob("*.npz"))
    if len(existing) >= N_PROBE_PER_SESSION * min(len(val_sessions), 5):
        log.info("Probes already present (%d), skipping creation", len(existing))
        return

    log.info("Creating probes for val sessions …")
    for session in val_sessions[:5]:
        labels = SMELTED_ROOT / session / "labels"
        vis_path = labels / VIS_STEM
        if not vis_path.exists():
            log.warning("No visibility file for %s, skipping probes", session)
            continue

        raw    = RAW_ROOT / session
        px_arr, py_arr, pz_arr, blocks = load_world_states(labels)
        vis_data     = np.load(vis_path)
        tick_indices = vis_data["tick_indices"].tolist()
        visibility   = vis_data["visibility"]

        with open(raw / "ticks.jsonl") as f:
            ticks = [json.loads(line) for line in f]

        video_path = str(raw / "video.mp4")
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(1, len(tick_indices) // (N_PROBE_PER_SESSION + 1))
        for i in range(1, N_PROBE_PER_SESSION + 1):
            k = i * step
            if k >= len(tick_indices):
                continue
            tick_idx = tick_indices[k]
            if tick_idx >= len(ticks) or tick_idx >= n_frames:
                continue

            m      = visibility[k]
            tick   = ticks[tick_idx]
            player = tick.get("player", tick)
            yaw    = float(player["yaw"])
            pitch  = float(player["pitch"])
            fov    = float(player.get("fov", 70.0))
            px     = int(px_arr[tick_idx])
            py     = int(py_arr[tick_idx])
            pz     = int(pz_arr[tick_idx])

            screen_uvs, vis_gt, xi, yi, zi = project_nonair_visibility(
                blocks[tick_idx], m, px, py, pz, yaw, pitch, fov
            )
            if len(vis_gt) == 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, min(tick_idx, n_frames - 1))
            ok, bgr = cap.read()
            if not ok:
                continue
            frame_t = preprocess_frame_np(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            out_path = probe_dir / f"{session}_tick{tick_idx:05d}.npz"
            np.savez_compressed(
                out_path,
                frame_t     = frame_t,
                screen_uvs  = screen_uvs,
                vis_gt      = vis_gt,
                xi          = xi,
                yi          = yi,
                zi          = zi,
                block_array = blocks[tick_idx].astype(np.int32),
                m_gt        = m,
                cam_x=px, cam_y=py, cam_z=pz,
                yaw=yaw, pitch=pitch, fov=fov,
            )
            log.info("  saved probe %s", out_path.name)

        cap.release()
    log.info("Probe creation done")


@torch.no_grad()
def evaluate(dino, model: nn.Module, val_loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    all_logits = []
    all_gt     = []

    for frames, uvs_list, vis_list, sid_list in val_loader:
        frames = frames.to(device)
        grids  = dino_patch_grids(dino, frames)
        feats, sids_t = extract_block_feats(grids, uvs_list, sid_list, device)
        vis_t  = torch.from_numpy(
            np.concatenate(vis_list).astype(np.float32)).to(device)
        logits = model(feats, sids_t)
        all_logits.append(logits.cpu())
        all_gt.append(vis_t.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_gt     = torch.cat(all_gt).numpy().astype(bool)
    probs      = 1 / (1 + np.exp(-all_logits))   # sigmoid

    # Sweep thresholds to find best F1
    best_f1 = best_thresh = 0.0
    best_stats = {}
    for thresh in np.linspace(0.05, 0.95, 37):
        preds = probs > thresh
        tp = int(( preds &  all_gt).sum())
        fp = int(( preds & ~all_gt).sum())
        fn = int((~preds &  all_gt).sum())
        tn = int((~preds & ~all_gt).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = thresh
            best_stats  = dict(precision=prec, recall=rec, f1=f1,
                               iou=tp/(tp+fp+fn+1e-8),
                               acc=(tp+tn)/(len(all_gt)+1e-8),
                               thresh=thresh,
                               tp=tp, fp=fp, fn=fn, tn=tn,
                               total=len(all_gt))

    return best_stats


@torch.no_grad()
def eval_vis(dino, model: nn.Module, probe_dir: Path,
             epoch: int, device: torch.device, wandb_run, vis_out_dir: Path):
    try:
        from visualize import render_perspective_view, make_side_by_side, save_render
    except ImportError:
        log.warning("eval_vis: Furnace visualize.py not importable, skipping")
        return

    model.eval()
    probe_files = sorted(probe_dir.glob("*.npz"))
    if not probe_files:
        log.warning("eval_vis: no probe files in %s", probe_dir)
        return

    vis_out_dir.mkdir(parents=True, exist_ok=True)
    wandb_images = []

    for probe_path in probe_files:
        try:
            d = np.load(probe_path)
            frame_t    = d["frame_t"]               # (3, 350, 630)
            screen_uvs = d["screen_uvs"]            # (N, 2)
            xi         = d["xi"].astype(np.int32)
            yi         = d["yi"].astype(np.int32)
            zi         = d["zi"].astype(np.int32)
            block_arr  = d["block_array"].astype(np.int32)
            m_gt       = d["m_gt"].astype(np.uint8)
            pose = {
                "x": float(d["cam_x"]), "y": float(d["cam_y"]), "z": float(d["cam_z"]),
                "yaw": float(d["yaw"]), "pitch": float(d["pitch"]), "fov": float(d["fov"]),
            }

            frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).to(device)
            grids  = dino_patch_grids(dino, frame_tensor)          # (1, 25, 45, 768)
            sids_probe = block_arr[xi, yi, zi].astype(np.int32)
            feats, sids_t = extract_block_feats(
                grids, [screen_uvs], [sids_probe], device)         # (N, 768), (N,)
            logits = model(feats, sids_t)
            pred_vis = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

            m_pred = np.zeros((32, 32, 32), dtype=np.uint8)
            m_pred[xi, yi, zi] = pred_vis

            render_gt   = render_perspective_view(m_gt,   block_arr, pose, fullbright=True)
            render_pred = render_perspective_view(m_pred, block_arr, pose, fullbright=True)
            side_by_side = make_side_by_side(render_gt, render_pred)

            out_path = vis_out_dir / f"epoch_{epoch:03d}_{probe_path.stem}.png"
            save_render(out_path, side_by_side)
            try:
                import wandb
                wandb_images.append(wandb.Image(str(out_path), caption=probe_path.stem))
            except Exception:
                pass
        except Exception as e:
            log.warning("eval_vis probe %s failed: %s", probe_path.name, e, exc_info=True)

    if wandb_images and wandb_run is not None:
        wandb_run.log({"vis/renders": wandb_images, "epoch": epoch})
    log.info("eval_vis: %d renders logged for epoch %d", len(wandb_images), epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_sessions = get_session_names("train", cfg["data"]["n_train"])
    val_sessions   = get_session_names("val",   cfg["data"]["n_train"])
    log.info("Train sessions: %d  Val sessions: %d",
             len(train_sessions), len(val_sessions))

    train_ds = OnlineFrameDataset(train_sessions, shuffle=True)
    val_ds   = OnlineFrameDataset(val_sessions,   shuffle=False)

    frame_bs = cfg["training"].get("frame_batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=frame_bs,
                              collate_fn=online_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=frame_bs,
                              collate_fn=online_collate, num_workers=2)

    dino = load_dino(device)

    m_cfg = cfg["model"]
    model = BlockHead(
        feat_dim       = DINO_DIM,
        block_embed_dim = m_cfg.get("block_embed_dim", 64),
        hidden_dim     = m_cfg.get("hidden_dim", 0),
        dropout        = m_cfg.get("dropout", 0.0),
    ).to(device)
    log.info("BlockHead params: %d", sum(p.numel() for p in model.parameters()))

    pw_override = cfg["training"].get("pos_weight")
    if pw_override is not None:
        pos_weight = torch.tensor(float(pw_override), device=device)
        log.info("pos_weight = %.2f  (from config override)", float(pos_weight))
    else:
        pos_weight = compute_pos_weight(train_sessions, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    epochs           = cfg["training"]["epochs"]
    steps_per_epoch  = (len(train_sessions) * 280) // frame_bs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["training"]["lr"],
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=cfg["training"]["lr_warmup_epochs"] / epochs,
    )

    ckpt_dir    = Path(cfg["checkpointing"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    probe_dir   = ckpt_dir / "probes"
    vis_out_dir = ckpt_dir / "renders"
    vis_every   = cfg.get("vis", {}).get("render_every_epochs", 5)

    save_probes(val_sessions, dino, probe_dir, device)

    import wandb
    run = wandb.init(
        project = cfg["wandb"]["project"],
        entity  = cfg["wandb"]["entity"],
        name    = cfg["wandb"]["run_name"],
        config  = cfg,
    )

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_steps    = 0

        for frames, uvs_list, vis_list, sid_list in train_loader:
            frames = frames.to(device)
            with torch.no_grad():
                grids = dino_patch_grids(dino, frames)

            feats, sids_t = extract_block_feats(grids, uvs_list, sid_list, device)
            vis_t  = torch.from_numpy(
                np.concatenate(vis_list).astype(np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats, sids_t)
            loss   = criterion(logits, vis_t)
            loss.backward()
            optimizer.step()
            if n_steps < steps_per_epoch:
                scheduler.step()

            train_loss += loss.item()
            n_steps    += 1

        train_loss /= max(n_steps, 1)
        metrics = evaluate(dino, model, val_loader, device)

        log.info(
            "Epoch %d/%d  loss=%.4f  val_f1=%.4f  val_prec=%.4f  val_rec=%.4f"
            "  val_iou=%.4f  val_acc=%.4f  thresh=%.2f",
            epoch, epochs, train_loss,
            metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["iou"], metrics["acc"], metrics.get("thresh", 0.5),
        )

        run.log({
            "epoch":          epoch,
            "train/loss":     train_loss,
            "val/f1":         metrics["f1"],
            "val/precision":  metrics["precision"],
            "val/recall":     metrics["recall"],
            "val/iou":        metrics["iou"],
            "val/acc":        metrics["acc"],
            "val/thresh":     metrics.get("thresh", 0.5),
            "lr":             scheduler.get_last_lr()[0],
        })

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_f1": best_f1}, ckpt_dir / "best.pt")

        if epoch % cfg["checkpointing"]["save_every_epochs"] == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / f"epoch_{epoch:03d}.pt")

        if epoch % vis_every == 0 and probe_dir.exists():
            eval_vis(dino, model, probe_dir, epoch, device, run, vis_out_dir)

    log.info("Done. Best val_f1: %.4f", best_f1)
    run.finish()


if __name__ == "__main__":
    main()
