"""
precompute_frames.py — Decode video frames needed for viblock2/3 training to JPEGs.

Uses frame_mapping.jsonl to find the correct video frame number for each game tick,
then does one sequential pass through the video to extract them.

Saves to: SMELTED_ROOT/{session}/labels/frames/{tick_idx:06d}.jpg
  where tick_idx is the world_states.bin index (game tick), and the JPEG content
  is the video frame that actually corresponds to that tick via frame_mapping.jsonl.

Uses PyAV for video decoding (consistent with the furnace pipeline).

Usage:
  PYTHONPATH=/home/vvm33/blockscope python precompute_frames.py [--workers 8] [--force]
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Accept sessions with either viblock2 or viblock3 data stems
_DATA_STEMS = ("viblock3_data.npz", "viblock2_data.npz")
JPEG_QUALITY = 95


def load_frame_mapping(session_dir: Path) -> dict:
    """Returns corrected_tick -> video_frame_number from frame_mapping.jsonl."""
    mapping = {}
    fm_path = session_dir / "frame_mapping.jsonl"
    if not fm_path.exists():
        return mapping
    for line in fm_path.read_text().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        corrected_tick = max(0, obj["tick"] - 1)
        mapping[corrected_tick] = obj["frame"]
    return mapping


def process_session(session: str, force: bool = False) -> tuple[str, int, int]:
    labels_dir = SMELTED_ROOT / session / "labels"
    frames_dir = labels_dir / "frames"

    # Find which data stem exists for this session
    data_path = None
    for stem in _DATA_STEMS:
        p = labels_dir / stem
        if p.exists():
            data_path = p
            break
    if data_path is None:
        return session, 0, 0

    tick_indices = [int(x) for x in np.load(data_path)["tick_indices"]]

    if not force:
        already = sum(1 for t in tick_indices if (frames_dir / f"{t:06d}.jpg").exists())
        if already == len(tick_indices):
            return session, 0, len(tick_indices)

    video_path = RAW_ROOT / session / "video.mp4"
    if not video_path.exists():
        return session, 0, 0

    tick_to_vframe = load_frame_mapping(RAW_ROOT / session)
    if not tick_to_vframe:
        log.warning("%s: no frame_mapping.jsonl, skipping", session)
        return session, 0, 0

    vframe_to_ticks: dict[int, list[int]] = {}
    skipped = 0
    for tick_idx in tick_indices:
        vframe = tick_to_vframe.get(tick_idx)
        if vframe is None:
            skipped += 1
            continue
        vframe_to_ticks.setdefault(vframe, []).append(tick_idx)

    if skipped:
        log.warning("%s: %d tick_indices had no frame_mapping entry", session, skipped)

    needed_vframes = set(vframe_to_ticks.keys())
    frames_dir.mkdir(exist_ok=True)

    # One sequential pass through the video using PyAV
    container = av.open(str(video_path))
    saved = 0
    frame_idx = 0
    remaining = len(needed_vframes)

    for frame in container.decode(video=0):
        if remaining == 0:
            break
        if frame_idx in needed_vframes:
            rgb = frame.to_ndarray(format="rgb24")
            img = Image.fromarray(rgb)
            for tick_idx in vframe_to_ticks[frame_idx]:
                out_path = frames_dir / f"{tick_idx:06d}.jpg"
                img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)
                saved += 1
            remaining -= 1
        frame_idx += 1

    container.close()
    return session, saved, len(tick_indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force",   action="store_true",
                    help="Re-extract even if JPEGs already exist")
    ap.add_argument("--session", default=None,
                    help="Process a single session only")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(
            d.name for d in SMELTED_ROOT.iterdir()
            if d.is_dir() and (d / "labels" / DATA_STEM).exists()
        )
    log.info("Found %d sessions (force=%s)", len(sessions), args.force)

    total_saved = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_session, s, args.force): s for s in sessions}
        for fut in as_completed(futs):
            session, saved, needed = fut.result()
            total_saved += saved
            if saved or args.force:
                log.info("  %-40s  saved=%d / needed=%d", session, saved, needed)

    log.info("Done. Total frames saved: %d", total_saved)


if __name__ == "__main__":
    main()
