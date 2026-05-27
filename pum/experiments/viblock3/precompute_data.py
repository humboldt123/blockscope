"""
Precompute per-tick metadata for viblock3 training.

Generates viblock3_data.npz per session:
  tick_indices  (K,)   int32   - valid tick indices (non-loading-screen filtered)
  tick_yaws     (K,)   float32 - player yaw in degrees at each tick

Much lighter than viblock2_data.npz — no block projection, just yaw + valid frames.
The world block grids come from world_states.bin at training time and are rotated
on-the-fly by each tick's yaw into camera-relative coordinates.

Run once on Brev:
  PYTHONPATH=/home/vvm33/blockscope python precompute_data.py
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

LOADING_SCREEN_STD = 8.0

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/home/vvm33/blockscope/furnace/pipeline/src/python")

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT, VIS_STEM

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STEM = "viblock3_data.npz"


def get_valid_frame_ticks(session: str, tick_indices: np.ndarray) -> set[int]:
    """Sequential video pass — returns tick indices with real visual content."""
    video_path = RAW_ROOT / session / "video.mp4"
    if not video_path.exists():
        return set(int(x) for x in tick_indices)

    needed = set(int(x) for x in tick_indices)
    valid  = set()
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx in needed and bgr.std() >= LOADING_SCREEN_STD:
            valid.add(frame_idx)
        frame_idx += 1
    cap.release()
    return valid


def precompute_session(session: str, force: bool = False) -> bool:
    labels     = SMELTED_ROOT / session / "labels"
    out_path   = labels / STEM
    vis_path   = labels / VIS_STEM
    ws_path    = labels / "world_states.bin"
    ticks_path = RAW_ROOT / session / "ticks.jsonl"

    if out_path.exists() and not force:
        log.info("skip %s (exists)", session)
        return True

    for p in (vis_path, ws_path, ticks_path):
        if not p.exists():
            log.warning("skip %s (missing %s)", session, p.name)
            return False

    vis_data     = np.load(vis_path)
    tick_indices = vis_data["tick_indices"]

    with open(ticks_path) as f:
        ticks = [json.loads(line) for line in f]

    valid_frame_ticks = get_valid_frame_ticks(session, tick_indices)
    log.info("%s: %d/%d ticks have valid frames", session, len(valid_frame_ticks), len(tick_indices))

    valid_tick_indices = []
    valid_yaws         = []

    for k in range(len(tick_indices)):
        tick_idx = int(tick_indices[k])
        if tick_idx >= len(ticks) or tick_idx not in valid_frame_ticks:
            continue
        tick   = ticks[tick_idx]
        player = tick.get("player", tick)
        yaw    = float(player.get("yaw", 0.0))
        valid_tick_indices.append(tick_idx)
        valid_yaws.append(yaw)

    if not valid_tick_indices:
        log.warning("skip %s (no valid ticks)", session)
        return False

    np.savez_compressed(
        out_path,
        tick_indices = np.array(valid_tick_indices, dtype=np.int32),
        tick_yaws    = np.array(valid_yaws,         dtype=np.float32),
    )
    log.info("%s: saved %d ticks", session, len(valid_tick_indices))
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",   action="store_true",
                    help="Recompute even if viblock3_data.npz already exists")
    ap.add_argument("--session", default=None,
                    help="Process a single session only")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(
            d.name for d in SMELTED_ROOT.iterdir()
            if d.is_dir() and (d / "labels" / "world_states.bin").exists()
        )
    log.info("Processing %d sessions (force=%s)", len(sessions), args.force)
    ok = sum(precompute_session(s, force=args.force) for s in sessions)
    log.info("Done: %d/%d succeeded", ok, len(sessions))


if __name__ == "__main__":
    main()
