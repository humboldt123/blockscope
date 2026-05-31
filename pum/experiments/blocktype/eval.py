"""
eval.py — Render GT vs predicted block types for the blocktype model.

For each val session probe tick, renders two perspective views of the 32³ grid:
  Left:  ground truth block types at visible positions
  Right: predicted block types at those same positions

Block positions that are OOV (not in SCATTER_BLOCKS) are excluded from both renders.
Saves side-by-side PNGs and logs to wandb.

Usage:
    python eval.py [--config config.yaml] [--ckpt /path/to/best.pt] [--device cuda:0]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from data.vis_dataset import (
    SMELTED_ROOT, RAW_ROOT, VIS_STEM,
    uv_to_patch_idx, PATCHES_H, PATCHES_W,
    project_nonair_visibility, preprocess_frame_np,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DINO_DIM        = 768
N_PROBES        = 3   # ticks per val session
N_VAL_SESSIONS  = 5


def get_session_names(split: str, n_train: int = 73) -> list[str]:
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    return all_sessions[:n_train] if split == "train" else all_sessions[n_train:]


def load_vocab(vocab_path: Path) -> tuple[np.ndarray, int, list[str], np.ndarray]:
    """
    Returns (class_lut, n_classes, class_names, class_to_rep_sid).
    class_to_rep_sid[class_idx] = a representative block-state ID for rendering.
    """
    with open(vocab_path) as f:
        v = json.load(f)
    lut = np.full(65536, -1, dtype=np.int64)
    class_to_rep = np.zeros(v["n_classes"], dtype=np.int32)
    for sid_str, cls in v["sid_to_class"].items():
        sid = int(sid_str)
        if sid < 65536:
            lut[sid] = cls
            if class_to_rep[cls] == 0:   # first SID seen for this class
                class_to_rep[cls] = sid
    return lut, v["n_classes"], v["classes"], class_to_rep


def load_dino(device):
    log.info("Loading DINOv2 ViT-B/14 …")
    import torch
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_model(ckpt_path: Path, n_classes: int, device):
    import torch.nn as nn
    import torch

    hidden_dim = 512
    model = nn.Sequential(
        nn.Linear(DINO_DIM, hidden_dim), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    # strip "net." prefix if present (BlockTypeHead wraps in self.net)
    state = {k.replace("net.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    log.info("Loaded checkpoint %s (epoch %s)", ckpt_path, ckpt.get("epoch", "?"))
    return model


@torch.no_grad()
def render_probe(dino, model, session: str, tick_idx: int,
                 class_lut: np.ndarray, class_to_rep: np.ndarray,
                 device) -> np.ndarray | None:
    """Run model on one tick and return side-by-side render (GT | pred)."""
    try:
        from io_helpers import load_world_states
        from visualize import render_perspective_view, make_side_by_side
    except ImportError as e:
        log.warning("Import failed: %s", e)
        return None

    import torch

    labels = SMELTED_ROOT / session / "labels"
    vis_path = labels / VIS_STEM
    if not vis_path.exists():
        return None

    px_arr, py_arr, pz_arr, blocks = load_world_states(labels)
    vis_data     = np.load(vis_path)
    tick_indices = vis_data["tick_indices"].tolist()
    visibility   = vis_data["visibility"]

    if tick_idx not in tick_indices:
        tick_idx = tick_indices[0]
    k = tick_indices.index(tick_idx)

    m      = visibility[k]
    raw    = RAW_ROOT / session
    with open(raw / "ticks.jsonl") as f:
        ticks = [json.loads(line) for line in f]
    tick   = ticks[tick_idx]
    player = tick.get("player", tick)
    yaw, pitch, fov = float(player["yaw"]), float(player["pitch"]), float(player.get("fov", 70.0))
    px, py, pz = int(px_arr[tick_idx]), int(py_arr[tick_idx]), int(pz_arr[tick_idx])
    pose = {"x": px, "y": py, "z": pz, "yaw": yaw, "pitch": pitch, "fov": fov}

    screen_uvs, vis_gt, xi, yi, zi, _ = project_nonair_visibility(
        blocks[tick_idx], m, px, py, pz, yaw, pitch, fov
    )
    if len(vis_gt) == 0:
        log.warning("  [%s tick=%d] no in-frustum blocks", session[:12], tick_idx)
        return None

    vis_mask = vis_gt == 1.0
    if vis_mask.sum() == 0:
        log.warning("  [%s tick=%d] no visible blocks (all occluded)", session[:12], tick_idx)
        return None
    log.info("  [%s tick=%d] %d visible blocks", session[:12], tick_idx, int(vis_mask.sum()))
    vis_uvs = screen_uvs[vis_mask]
    vxi = xi[vis_mask].astype(np.int32)
    vyi = yi[vis_mask].astype(np.int32)
    vzi = zi[vis_mask].astype(np.int32)

    true_sids   = blocks[tick_idx][vxi, vyi, vzi].astype(np.int32)
    true_classes = class_lut[np.clip(true_sids, 0, len(class_lut) - 1)]
    valid = true_classes >= 0
    log.info("  [%s tick=%d] %d/%d in-vocab blocks (OOV=%.0f%%)",
             session[:12], tick_idx, int(valid.sum()), len(valid),
             100.0 * (1 - valid.sum() / max(len(valid), 1)))
    if valid.sum() == 0:
        log.warning("  [%s tick=%d] all visible blocks are OOV — no scatter blocks visible",
                    session[:12], tick_idx)
        return None
    vxi, vyi, vzi = vxi[valid], vyi[valid], vzi[valid]
    vis_uvs     = vis_uvs[valid]
    true_classes = true_classes[valid]

    # Load frame and run DINOv2
    video_path = raw / "video.mp4"
    if not video_path.exists():
        log.warning("  [%s tick=%d] video.mp4 not found", session[:12], tick_idx)
        return None
    cap      = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(tick_idx, n_frames - 1))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        log.warning("  [%s tick=%d] video read failed (n_frames=%d)", session[:12], tick_idx, n_frames)
        return None
    frame_t = preprocess_frame_np(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).to(device)

    out   = dino.forward_features(frame_tensor)
    patch = out["x_norm_patchtokens"].reshape(1, PATCHES_H, PATCHES_W, DINO_DIM)
    rows, cols = uv_to_patch_idx(vis_uvs)
    feats = patch[0, rows, cols, :]           # (N, 768)

    logits = model(feats)
    pred_classes = logits.argmax(dim=1).cpu().numpy().astype(np.int32)

    # Build 32³ block arrays for rendering
    block_arr = blocks[tick_idx].astype(np.int32)

    gt_arr   = np.zeros((32, 32, 32), dtype=np.int32)
    pred_arr = np.zeros((32, 32, 32), dtype=np.int32)
    m_render = np.zeros((32, 32, 32), dtype=np.uint8)

    gt_arr[vxi, vyi, vzi]   = class_to_rep[true_classes]
    pred_arr[vxi, vyi, vzi] = class_to_rep[pred_classes]
    m_render[vxi, vyi, vzi] = 1

    render_gt   = render_perspective_view(m_render, gt_arr,   pose, fullbright=True)
    render_pred = render_perspective_view(m_render, pred_arr, pose, fullbright=True)
    return make_side_by_side(render_gt, render_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--ckpt",   type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out",    type=Path, default=None)
    args = ap.parse_args()

    import torch
    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir  = Path(cfg["checkpointing"]["dir"])
    ckpt_path = args.ckpt or ckpt_dir / "best.pt"
    out_dir   = args.out  or ckpt_dir / "renders_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_lut, n_classes, class_names, class_to_rep = load_vocab(
        Path(cfg["data"]["vocab_path"]))
    log.info("%d classes loaded", n_classes)

    dino  = load_dino(device)
    model = load_model(ckpt_path, n_classes, device)

    val_sessions = get_session_names("val", cfg["data"]["n_train"])[:N_VAL_SESSIONS]
    log.info("Rendering %d sessions × %d ticks …", len(val_sessions), N_PROBES)

    import wandb
    run = wandb.init(
        project = cfg["wandb"]["project"],
        entity  = cfg["wandb"]["entity"],
        name    = cfg["wandb"]["run_name"] + "_eval",
        config  = cfg,
    )
    wandb_images = []

    from io_helpers import load_world_states
    for session in val_sessions:
        labels   = SMELTED_ROOT / session / "labels"
        vis_path = labels / VIS_STEM
        if not vis_path.exists():
            continue
        vis_data    = np.load(vis_path)
        tick_indices = vis_data["tick_indices"].tolist()
        step = max(1, len(tick_indices) // (N_PROBES + 1))
        probe_ticks = [tick_indices[min((i + 1) * step, len(tick_indices) - 1)]
                       for i in range(N_PROBES)]

        for tick_idx in probe_ticks:
            render = render_probe(dino, model, session, tick_idx,
                                  class_lut, class_to_rep, device)
            if render is None:
                continue
            stem     = f"{session}_tick{tick_idx:05d}"
            out_path = out_dir / f"{stem}.png"
            import cv2 as _cv2
            _cv2.imwrite(str(out_path), render)
            log.info("Saved %s", out_path.name)
            try:
                wandb_images.append(
                    wandb.Image(str(out_path),
                                caption=f"{session[:12]}…  tick={tick_idx}  [GT | pred]"))
            except Exception:
                pass

    if wandb_images:
        run.log({"blocktype/renders": wandb_images})
    run.finish()
    log.info("Done. %d renders saved to %s", len(wandb_images), out_dir)


if __name__ == "__main__":
    main()
