"""
extract_features.py — Phase 1: offline DINOv2 feature extraction.

For each session in train/val, sample ticks at TICK_STRIDE (1 fps), decode the
video frame, run DINOv2 ViT-B/14, compute frustum-visible blocks, look up the
nearest patch token for each visible block, and save:

    /data/vvm33/pum_vis_features/{split}/{session}.npz
        patch_feats : (N, 768) float32
        token_ids   : (N,)     int32
        screen_uvs  : (N, 2)   float32   [debug; u,v in [0,1]]

Usage:
    python extract_features.py [--split train] [--device cuda:0]
    python extract_features.py --split val   --device cuda:0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pum root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from data.vis_dataset import (
    IMG_H, IMG_W, WINDOW,
    SMELTED_ROOT, RAW_ROOT,
    frustum_project, read_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VOCAB_JSON    = Path(__file__).parents[2] / "vocab" / "stage1_vocab.json"
FURNACE_PYTHON = "/home/vvm33/blockscope/furnace/pipeline/src/python"
OUT_ROOT       = Path("/data/vvm33/pum_vis_features")

TICK_STRIDE      = 20    # sample every 20 ticks ≈ 1 fps (20 tps server)
DINO_PATCH       = 14    # ViT-B/14 patch size in pixels
DINO_DIM         = 768   # feature dimension
N_PROBE_PER_SESSION = 4  # probe ticks saved per val session for eval_vis

# DINOv2 expected input size — we feed full 640×360 frame
# Patch grid: (640//14) × (360//14) = 45 × 25 = 1125 patches (with some overlap handling)
PATCHES_W = IMG_W // DINO_PATCH   # 45
PATCHES_H = IMG_H // DINO_PATCH   # 25


def load_vocab():
    with open(VOCAB_JSON) as f:
        raw = json.load(f)
    s2t = {int(k): v for k, v in raw["state_to_token"].items()}
    return s2t


def get_session_names(split: str, n_train: int = 73):
    """Return list of session names for the given split, matching stage1 convention."""
    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    train_sessions = all_sessions[:n_train]
    val_sessions   = all_sessions[n_train:]
    return train_sessions if split == "train" else val_sessions


def load_dino(device: torch.device) -> torch.nn.Module:
    log.info("Loading DINOv2 ViT-B/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("DINOv2 loaded")
    return model


def preprocess_frame(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, 3) uint8 RGB → (1, 3, H', W') float32 normalised tensor.

    DINOv2 expects input divisible by patch_size=14. Crop the bottom few pixels:
    360 → 350 (= 25 × 14), 640 stays (= 45 × 14 + 10 → crop to 630).
    We centre-crop to (350, 630) so the patch grid is exactly 25 × 45.
    """
    h_crop = PATCHES_H * DINO_PATCH   # 350
    w_crop = PATCHES_W * DINO_PATCH   # 630

    # Centre crop
    top  = (IMG_H - h_crop) // 2   # 5
    left = (IMG_W - w_crop) // 2   # 5
    frame = frame_rgb[top:top+h_crop, left:left+w_crop]  # (350, 630, 3)

    t = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0  # (3, H', W')
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    t = (t.to(device) - mean) / std
    return t.unsqueeze(0)  # (1, 3, H', W')


@torch.no_grad()
def extract_patch_features(dino: torch.nn.Module,
                            frame_tensor: torch.Tensor) -> np.ndarray:
    """Run DINOv2 and return patch features (PATCHES_H, PATCHES_W, 768) float32."""
    out = dino.forward_features(frame_tensor)
    # patch_tokens: (1, n_patches, 768) — excludes [CLS]
    patch_tokens = out["x_norm_patchtokens"]   # (1, 1125, 768)
    feat = patch_tokens[0].cpu().float().numpy()   # (1125, 768)
    return feat.reshape(PATCHES_H, PATCHES_W, DINO_DIM)  # (25, 45, 768)


def uv_to_patch_feat(screen_uvs: np.ndarray,
                     patch_grid: np.ndarray) -> np.ndarray:
    """
    Map (N, 2) normalised screen UV coords → (N, 768) patch features.

    screen_uvs: u in [0,1] = horizontal, v in [0,1] = vertical (top=0).
    patch_grid: (PATCHES_H, PATCHES_W, 768).

    The DINOv2 input was centre-cropped to (350, 630) — so the UV mapping must
    account for the 5px top/left offset relative to the original 640×360 frame.
    """
    top_frac  = 5.0 / IMG_H   # 5/360 ≈ 0.0139
    left_frac = 5.0 / IMG_W   # 5/640 ≈ 0.0078
    h_frac    = (PATCHES_H * DINO_PATCH) / IMG_H   # 350/360
    w_frac    = (PATCHES_W * DINO_PATCH) / IMG_W   # 630/640

    # Remap u,v into cropped frame coordinates
    u_crop = np.clip((screen_uvs[:, 0] - left_frac) / w_frac, 0.0, 1.0)
    v_crop = np.clip((screen_uvs[:, 1] - top_frac)  / h_frac, 0.0, 1.0)

    col_idx = np.clip((u_crop * PATCHES_W).astype(np.int32), 0, PATCHES_W - 1)
    row_idx = np.clip((v_crop * PATCHES_H).astype(np.int32), 0, PATCHES_H - 1)

    return patch_grid[row_idx, col_idx]   # (N, 768)


def _cam_pose(tick: dict, px: int, py: int, pz: int) -> dict:
    """Extract camera world-space pose dict for Furnace render_perspective_view."""
    player = tick.get("player", tick)
    cam    = player.get("camera") or player
    return {
        "x":     float(cam["x"]),
        "y":     float(cam["y"]),
        "z":     float(cam["z"]),
        "yaw":   float(cam["yaw"]),
        "pitch": float(cam["pitch"]),
        "fov":   float(player.get("fov", 70.0)),
    }


def _vis_ijk(screen_uv: np.ndarray, token_ids: np.ndarray,
             blocks_u16: np.ndarray, s2t: dict,
             px: int, py: int, pz: int,
             yaw: float, pitch: float, fov: float) -> tuple:
    """Return (xi, yi, zi) voxel grid indices for frustum-visible non-air blocks."""
    from data.vis_dataset import WINDOW
    max_state = int(blocks_u16.max()) if blocks_u16.size > 0 else 0
    lut = np.zeros(max_state + 1, dtype=np.int32)
    for sid in range(max_state + 1):
        lut[sid] = s2t.get(sid, 0)
    tokens_grid = lut[blocks_u16]
    xi, yi, zi = np.where(tokens_grid > 0)
    # reapply the same frustum mask used in frustum_project to get consistent indices
    # (frustum_project already returned screen_uv & token_ids for the visible subset)
    # We reconstruct by re-running the frustum filter on all non-air positions.
    import numpy as _np
    from data.vis_dataset import EYE_HEIGHT, MIN_DIST, _camera_basis, IMG_W, IMG_H
    wx = (xi - WINDOW // 2 + px + 0.5).astype(np.float64)
    wy = (yi - WINDOW // 2 + py + 0.5).astype(np.float64)
    wz = (zi - WINDOW // 2 + pz + 0.5).astype(np.float64)
    cx, cy, cz = px + 0.5, py + EYE_HEIGHT, pz + 0.5
    dx, dy, dz = wx - cx, wy - cy, wz - cz
    fwd, right, up = _camera_basis(yaw, pitch)
    dot_f = dx*fwd[0] + dy*fwd[1] + dz*fwd[2]
    dot_r = dx*right[0] + dy*right[1] + dz*right[2]
    dot_u = dx*up[0]   + dy*up[1]   + dz*up[2]
    fov_x = _np.radians(fov)
    fov_y = fov_x * (IMG_H / IMG_W)
    tan_x, tan_y = _np.tan(fov_x / 2), _np.tan(fov_y / 2)
    in_front = dot_f > MIN_DIST
    safe_f   = _np.where(dot_f > 0, dot_f, 1e-9)
    ndc_x = _np.where(in_front, (dot_r / safe_f) / tan_x, 2.0)
    ndc_y = _np.where(in_front, -(dot_u / safe_f) / tan_y, 2.0)
    mask = in_front & (_np.abs(ndc_x) <= 1.0) & (_np.abs(ndc_y) <= 1.0)
    return xi[mask].astype(np.int32), yi[mask].astype(np.int32), zi[mask].astype(np.int32)


def save_probes(session_name: str, s2t: dict, dino: torch.nn.Module,
                device: torch.device, probe_dir: Path,
                px_arr, py_arr, pz_arr, blocks, ticks, video_path: str,
                n_frames: int):
    """Save N_PROBE_PER_SESSION probe npz files for eval_vis during training."""
    import cv2
    T = len(px_arr)
    sampled = list(range(0, min(T, len(ticks), n_frames), TICK_STRIDE))
    # Pick evenly spaced probe ticks across the session
    step = max(1, len(sampled) // (N_PROBE_PER_SESSION + 1))
    probe_indices = [sampled[i * step] for i in range(1, N_PROBE_PER_SESSION + 1)
                     if i * step < len(sampled)]

    for tick_idx in probe_indices:
        tick   = ticks[tick_idx]
        player = tick.get("player", tick)
        yaw    = float(player["yaw"])
        pitch  = float(player["pitch"])
        fov    = float(player.get("fov", 70.0))
        px     = int(px_arr[tick_idx])
        py     = int(py_arr[tick_idx])
        pz     = int(pz_arr[tick_idx])

        blocks_u16 = blocks[tick_idx]
        screen_uv, token_ids = frustum_project(blocks_u16, s2t, px, py, pz, yaw, pitch, fov)
        if len(token_ids) == 0:
            continue

        xi, yi, zi = _vis_ijk(screen_uv, token_ids, blocks_u16, s2t,
                               px, py, pz, yaw, pitch, fov)

        frame_rgb    = read_frame(video_path, min(tick_idx, n_frames - 1))
        frame_tensor = preprocess_frame(frame_rgb, device)
        patch_grid   = extract_patch_features(dino, frame_tensor)
        feats        = uv_to_patch_feat(screen_uv, patch_grid)

        pose = _cam_pose(tick, px, py, pz)
        out  = probe_dir / f"{session_name}_{tick_idx:05d}.npz"
        np.savez_compressed(out,
                            patch_feats  = feats.astype(np.float32),
                            token_ids_gt = token_ids.astype(np.int32),
                            block_array  = blocks_u16.astype(np.uint16),
                            vis_xi=xi, vis_yi=yi, vis_zi=zi,
                            cam_x=pose["x"], cam_y=pose["y"], cam_z=pose["z"],
                            yaw=pose["yaw"], pitch=pose["pitch"], fov=pose["fov"],
                            px=px, py=py, pz=pz)
        log.info("    probe saved: %s (%d visible blocks)", out.name, len(token_ids))


def process_session(session_name: str, s2t: dict, dino: torch.nn.Module,
                    device: torch.device, out_path: Path,
                    probe_dir: Path | None = None):
    from io_helpers import load_world_states

    labels = SMELTED_ROOT / session_name / "labels"
    px_arr, py_arr, pz_arr, blocks = load_world_states(labels)
    T = len(px_arr)

    raw = RAW_ROOT / session_name
    with open(raw / "ticks.jsonl") as f:
        ticks = [json.loads(line) for line in f]

    video_path = str(raw / "video.mp4")
    import cv2
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    all_feats  = []
    all_tokens = []
    all_uvs    = []

    tick_indices = range(0, min(T, len(ticks), n_frames), TICK_STRIDE)
    log.info("  %s: %d ticks → %d sampled", session_name, T, len(tick_indices))

    for tick_idx in tick_indices:
        tick = ticks[tick_idx]
        player = tick.get("player", tick)
        yaw   = float(player["yaw"])
        pitch = float(player["pitch"])
        fov   = float(player.get("fov", 70.0))
        px    = int(px_arr[tick_idx])
        py    = int(py_arr[tick_idx])
        pz    = int(pz_arr[tick_idx])

        blocks_u16 = blocks[tick_idx]   # (32, 32, 32)

        screen_uv, token_ids = frustum_project(
            blocks_u16, s2t, px, py, pz, yaw, pitch, fov
        )
        if len(token_ids) == 0:
            continue

        frame_idx = min(tick_idx, n_frames - 1)
        frame_rgb = read_frame(video_path, frame_idx)

        frame_tensor = preprocess_frame(frame_rgb, device)
        patch_grid   = extract_patch_features(dino, frame_tensor)   # (25, 45, 768)

        feats = uv_to_patch_feat(screen_uv, patch_grid)   # (N, 768)

        all_feats.append(feats)
        all_tokens.append(token_ids)
        all_uvs.append(screen_uv)

    if not all_feats:
        log.warning("  %s: no visible blocks found, skipping", session_name)
        return 0

    patch_feats = np.concatenate(all_feats,  axis=0).astype(np.float32)
    token_ids   = np.concatenate(all_tokens, axis=0).astype(np.int32)
    screen_uvs  = np.concatenate(all_uvs,    axis=0).astype(np.float32)

    np.savez_compressed(out_path,
                        patch_feats=patch_feats,
                        token_ids=token_ids,
                        screen_uvs=screen_uvs)
    log.info("  %s: saved %d observations → %s", session_name, len(token_ids), out_path)

    if probe_dir is not None:
        save_probes(session_name, s2t, dino, device, probe_dir,
                    px_arr, py_arr, pz_arr, blocks, ticks, video_path, n_frames)

    return len(token_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="train", choices=["train", "val"])
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--n-train", type=int, default=73)
    ap.add_argument("--resume",  action="store_true",
                    help="Skip sessions whose .npz already exists")
    args = ap.parse_args()

    device = torch.device(args.device)
    s2t = load_vocab()
    dino = load_dino(device)

    out_dir = OUT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe frames saved only for val — used by eval_vis in train.py
    probe_dir = None
    if args.split == "val":
        probe_dir = OUT_ROOT / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)

    sessions = get_session_names(args.split, n_train=args.n_train)
    log.info("Processing %d %s sessions → %s", len(sessions), args.split, out_dir)

    total_obs = 0
    for i, session in enumerate(sessions):
        out_path = out_dir / f"{session}.npz"
        if args.resume and out_path.exists():
            log.info("[%d/%d] %s — already done, skipping",
                     i + 1, len(sessions), session)
            continue
        log.info("[%d/%d] %s", i + 1, len(sessions), session)
        try:
            n = process_session(session, s2t, dino, device, out_path,
                                probe_dir=probe_dir)
            total_obs += n
        except Exception as e:
            log.error("  FAILED: %s", e, exc_info=True)

    log.info("Done. Total observations: %d", total_obs)


if __name__ == "__main__":
    main()
