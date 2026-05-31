"""
precompute_patch_labels.py — Build per-patch block-type labels for the SigLIP NaFlex head.

For each session and every tick in visibility_s20.npz:
  - Project visible blocks (Furnace m=1) to screen UVs
  - Map UVs to the 40×22 SigLIP patch grid (16px patches, 640×352 center crop)
  - For each patch assign the SID of the closest visible block projecting into it
  - Patches with no visible block get SID = 0 (sky)

Saves per-session: {session}/labels/patch_labels_s20.npz
  tick_indices  (K,)          int32   — same indices as visibility_s20.npz
  patch_labels  (K, 22, 40)  uint16  — block-state IDs; 0 = sky/none

Run on CPU before training (no GPU needed).
Usage:
    python precompute_patch_labels.py [--sessions s1 s2 ...] [--overwrite]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import (
    SMELTED_ROOT, RAW_ROOT, VIS_STEM,
    project_nonair_visibility, _camera_basis,
    EYE_HEIGHT, WINDOW,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMG_W, IMG_H      = 640, 360
SIGLIP_PATCH      = 16
SIGLIP_W          = 40   # patches per row  (640/16)
SIGLIP_H          = 22   # patches per col  (processor resizes 360→352=22×16)

PATCH_LABELS_STEM = "patch_labels_s20.npz"


def uvs_to_siglip_patches(screen_uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map (N, 2) screen UVs (in [0,1]×[0,1] for 640×360) to SigLIP patch indices.

    The Siglip2ImageProcessorFast resizes the 640×360 frame to 640×352 (22 rows
    of 16px patches), with no center crop.  UV space maps linearly:
      col = floor(u * 40),  row = floor(v * 22)

    Returns row (N,) int32, col (N,) int32.  All blocks are valid (no crop region).
    """
    col = np.floor(screen_uvs[:, 0] * SIGLIP_W).astype(np.int32).clip(0, SIGLIP_W - 1)
    row = np.floor(screen_uvs[:, 1] * SIGLIP_H).astype(np.int32).clip(0, SIGLIP_H - 1)
    return row, col


def compute_forward_depth(xi, yi, zi, px, py, pz, yaw, pitch) -> np.ndarray:
    """Return camera-forward depth for grid positions."""
    wx = (xi - WINDOW // 2 + px + 0.5).astype(np.float64)
    wy = (yi - WINDOW // 2 + py + 0.5).astype(np.float64)
    wz = (zi - WINDOW // 2 + pz + 0.5).astype(np.float64)
    cx, cy, cz = px + 0.5, py + EYE_HEIGHT, pz + 0.5
    dx, dy, dz = wx - cx, wy - cy, wz - cz
    fwd, _, _ = _camera_basis(yaw, pitch)
    return dx * fwd[0] + dy * fwd[1] + dz * fwd[2]


def build_patch_labels(blocks_t, m, px, py, pz, yaw, pitch, fov) -> np.ndarray:
    """
    Build (SIGLIP_H, SIGLIP_W) uint16 patch label array for one tick.

    Label = block-state ID of closest visible block projecting to that patch.
    Label = 0 if no visible block projects to that patch (sky/empty).
    """
    patch_labels = np.zeros((SIGLIP_H, SIGLIP_W), dtype=np.uint16)

    screen_uvs, vis_labels, xi, yi, zi, sids = project_nonair_visibility(
        blocks_t, m, px, py, pz, yaw, pitch, fov)

    if len(vis_labels) == 0:
        return patch_labels

    vis_mask = vis_labels == 1.0
    if vis_mask.sum() == 0:
        return patch_labels

    vis_uvs = screen_uvs[vis_mask]
    vis_sids = sids[vis_mask]
    vis_xi = xi[vis_mask]
    vis_yi = yi[vis_mask]
    vis_zi = zi[vis_mask]

    row, col = uvs_to_siglip_patches(vis_uvs)

    depth = compute_forward_depth(vis_xi, vis_yi, vis_zi, px, py, pz, yaw, pitch)

    # Sort descending by depth (farthest first) so closer blocks overwrite farther ones.
    # Numpy fancy indexing: last write to each (r,c) wins.
    sort_order = np.argsort(-depth)
    patch_labels[row[sort_order], col[sort_order]] = vis_sids[sort_order].astype(np.uint16)

    return patch_labels


def get_all_sessions() -> list[str]:
    return sorted(d.name for d in SMELTED_ROOT.iterdir()
                  if d.is_dir() and (d / "labels" / "world_states.bin").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from io_helpers import load_world_states

    sessions = args.sessions or get_all_sessions()
    log.info("Processing %d sessions — SigLIP grid %d×%d (processor resizes 360→352, patch=%dpx)",
             len(sessions), SIGLIP_W, SIGLIP_H, SIGLIP_PATCH)

    total_sky = total_cells = 0

    for session in sessions:
        labels_dir = SMELTED_ROOT / session / "labels"
        vis_path   = labels_dir / VIS_STEM
        out_path   = labels_dir / PATCH_LABELS_STEM

        if out_path.exists() and not args.overwrite:
            log.info("  %s  already exists, skipping", session)
            continue

        if not vis_path.exists():
            log.warning("  %s  no visibility file", session)
            continue

        raw        = RAW_ROOT / session
        ticks_path = raw / "ticks.jsonl"
        if not ticks_path.exists():
            log.warning("  %s  no ticks.jsonl", session)
            continue

        px_arr, py_arr, pz_arr, blocks = load_world_states(labels_dir)
        vis_data     = np.load(vis_path)
        tick_indices = vis_data["tick_indices"]   # (K,) int
        visibility   = vis_data["visibility"]     # (K, 32, 32, 32) uint8
        K            = len(tick_indices)

        with open(ticks_path) as f:
            ticks = [json.loads(line) for line in f]

        all_labels = np.zeros((K, SIGLIP_H, SIGLIP_W), dtype=np.uint16)

        for k, tick_idx in enumerate(tick_indices.tolist()):
            tick_idx = int(tick_idx)
            if tick_idx >= len(ticks):
                continue
            tick   = ticks[tick_idx]
            player = tick.get("player", tick)
            yaw    = float(player["yaw"])
            pitch  = float(player["pitch"])
            fov    = float(player.get("fov", 70.0))
            px     = int(px_arr[tick_idx])
            py     = int(py_arr[tick_idx])
            pz     = int(pz_arr[tick_idx])

            all_labels[k] = build_patch_labels(
                blocks[tick_idx], visibility[k],
                px, py, pz, yaw, pitch, fov)

        n_sky   = int((all_labels == 0).sum())
        n_total = all_labels.size
        total_sky   += n_sky
        total_cells += n_total
        log.info("  %s  K=%d ticks  sky=%.1f%%  scatter=%.1f%%",
                 session, K, 100 * n_sky / max(n_total, 1),
                 100 * (1 - n_sky / max(n_total, 1)))

        np.savez_compressed(out_path,
                            tick_indices=tick_indices.astype(np.int32),
                            patch_labels=all_labels)

    if total_cells > 0:
        log.info("All done. Overall sky=%.1f%% scatter=%.1f%%",
                 100 * total_sky / total_cells,
                 100 * (1 - total_sky / total_cells))


if __name__ == "__main__":
    main()
