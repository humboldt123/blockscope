"""
Precompute per-block visibility data for viblock2 training.

For each session, loads world_states.bin + visibility_s20.npz + ticks.jsonl
ONCE and saves compact viblock2_data.npz containing:

  tick_indices  (K,)      int32   - indices into video/ticks for valid ticks
  block_offsets (K+1,)    int32   - CSR offsets into flat arrays
  block_uvs     (N, 2)    float32 - screen UV [0,1] of each visible block
  block_sids    (N,)      uint16  - block-state IDs (0 = air / sky)
  block_xyz     (N, 3)    float32 - (xi-16,yi-16,zi-16)/16 in [-1,1]

Run once on Brev before training:
  PYTHONPATH=/home/vvm33/blockscope python precompute_data.py
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

LOADING_SCREEN_STD = 8.0  # frames with pixel std below this are loading screens / black

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import (
    SMELTED_ROOT, RAW_ROOT, VIS_STEM,
    project_nonair_visibility, WINDOW,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STEM = "viblock2_data.npz"


def get_valid_frame_ticks(session: str, tick_indices: np.ndarray) -> set[int]:
    """Sequential video pass — returns tick indices whose frame has real visual content."""
    video_path = RAW_ROOT / session / "video.mp4"
    if not video_path.exists():
        return set(tick_indices.tolist())

    needed = set(int(x) for x in tick_indices)
    valid  = set()
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            if bgr.std() >= LOADING_SCREEN_STD:
                valid.add(frame_idx)
        frame_idx += 1
    cap.release()
    return valid


def precompute_session(session: str, force: bool = False) -> bool:
    labels = SMELTED_ROOT / session / "labels"
    out_path = labels / STEM
    vis_path = labels / VIS_STEM
    ws_path  = labels / "world_states.bin"
    raw      = RAW_ROOT / session
    ticks_path = raw / "ticks.jsonl"

    if out_path.exists() and not force:
        log.info("skip %s (exists)", session)
        return True

    for p in (vis_path, ws_path, ticks_path):
        if not p.exists():
            log.warning("skip %s (missing %s)", session, p.name)
            return False

    from io_helpers import load_world_states
    px_arr, py_arr, pz_arr, blocks = load_world_states(labels)

    vis_data     = np.load(vis_path)
    tick_indices = vis_data["tick_indices"]
    visibility   = vis_data["visibility"]

    with open(ticks_path) as f:
        ticks = [json.loads(line) for line in f]

    valid_frame_ticks = get_valid_frame_ticks(session, tick_indices)
    log.info("%s: %d/%d ticks have valid frames", session, len(valid_frame_ticks), len(tick_indices))

    valid_tick_indices = []
    all_uvs  = []
    all_sids = []
    all_xyz  = []

    for k in range(len(tick_indices)):
        tick_idx = int(tick_indices[k])
        if tick_idx >= len(ticks) or tick_idx >= len(blocks):
            continue
        if tick_idx not in valid_frame_ticks:
            continue

        m      = visibility[k]
        tick   = ticks[tick_idx]
        player = tick.get("player", tick)
        yaw    = float(player["yaw"])
        pitch  = float(player["pitch"])
        fov    = float(player.get("fov", 70.0))
        px = int(px_arr[tick_idx])
        py = int(py_arr[tick_idx])
        pz = int(pz_arr[tick_idx])

        screen_uvs, vis_labels, xi, yi, zi, sids = project_nonair_visibility(
            blocks[tick_idx], m, px, py, pz, yaw, pitch, fov)

        mask = vis_labels == 1.0
        if mask.sum() == 0:
            continue

        uvs   = screen_uvs[mask].astype(np.float32)
        sids_v = sids[mask].astype(np.uint16)
        xyz   = np.stack([
            (xi[mask].astype(np.float32) - WINDOW // 2) / (WINDOW // 2),
            (yi[mask].astype(np.float32) - WINDOW // 2) / (WINDOW // 2),
            (zi[mask].astype(np.float32) - WINDOW // 2) / (WINDOW // 2),
        ], axis=1)

        valid_tick_indices.append(tick_idx)
        all_uvs.append(uvs)
        all_sids.append(sids_v)
        all_xyz.append(xyz)

    if not valid_tick_indices:
        log.warning("skip %s (no valid ticks)", session)
        return False

    counts  = np.array([len(u) for u in all_uvs], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

    np.savez_compressed(
        out_path,
        tick_indices  = np.array(valid_tick_indices, dtype=np.int32),
        block_offsets = offsets,
        block_uvs     = np.concatenate(all_uvs,  axis=0),
        block_sids    = np.concatenate(all_sids, axis=0),
        block_xyz     = np.concatenate(all_xyz,  axis=0),
    )
    log.info("%s: %d ticks, %d blocks total", session, len(valid_tick_indices), int(offsets[-1]))
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Reprocess sessions even if viblock2_data.npz already exists")
    args = ap.parse_args()

    sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    log.info("Processing %d sessions (force=%s)", len(sessions), args.force)
    ok = sum(precompute_session(s, force=args.force) for s in sessions)
    log.info("Done: %d/%d succeeded", ok, len(sessions))


if __name__ == "__main__":
    main()
