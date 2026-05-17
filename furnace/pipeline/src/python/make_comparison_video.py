"""
make_comparison_video.py — render a side-by-side comparison video.

Usage:
    python make_comparison_video.py [session_dir] [max_seconds]

Drives output by VIDEO FRAMES (not ticks) so the comparison is always
locked to the actual recording timeline.  For each video frame F the
script finds the most-recent tick whose frame-mapping is ≤ F, renders
the voxel view for that tick, and puts video frame F on the left.
"""

import json
import logging
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PIPELINE_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_ROOT      = Path(os.environ.get("DATA_DIR", Path.home() / "blockscope_data"))
PREFER_SESSION = "session_1777852157"

sys.path.insert(0, str(PIPELINE_ROOT / "src" / "python"))

from visualize import (render_perspective_view, make_side_by_side, IMG_W, IMG_H)
from io_helpers import load_tick_npz


def find_session() -> Path:
    sessions = sorted(DATA_ROOT.glob("session_*"))
    if not sessions:
        raise FileNotFoundError(f"No session_* dirs in {DATA_ROOT}")
    preferred = DATA_ROOT / PREFER_SESSION
    return preferred if preferred.exists() else sessions[0]


def load_ticks(session_dir: Path) -> dict:
    """Returns dict tick_num → tick_dict."""
    result = {}
    for line in (session_dir / "ticks.jsonl").read_text().splitlines():
        if line:
            t = json.loads(line)
            result[t["tick"]] = t
    return result


def build_frame_index(session_dir: Path, ticks: dict) -> list:
    """
    Returns a list indexed by VIDEO FRAME NUMBER.
    Each entry is the tick_num of the most recent tick whose frame_mapping ≤ that frame.
    Entries before the first mapped frame are None.
    """
    # Raw mapping: corrected_tick → video_frame
    tick_to_frame = {}
    for line in (session_dir / "frame_mapping.jsonl").read_text().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        corrected = max(0, obj["tick"] - 1)
        if corrected in ticks:
            tick_to_frame[corrected] = obj["frame"]

    if not tick_to_frame:
        return []

    max_frame = max(tick_to_frame.values())

    # Build reverse: for each frame, which tick is latest tick with mapping ≤ frame?
    # Sort ticks by their mapped frame
    sorted_pairs = sorted(tick_to_frame.items(), key=lambda kv: kv[1])  # (tick, frame)

    frame_to_tick = [None] * (max_frame + 1)
    for tick_num, frame_num in sorted_pairs:
        # This tick is valid from frame_num onward (until a later tick takes over)
        frame_to_tick[frame_num] = tick_num

    # Forward-fill: frames with no direct mapping use the last known tick
    last = None
    for i in range(len(frame_to_tick)):
        if frame_to_tick[i] is not None:
            last = frame_to_tick[i]
        else:
            frame_to_tick[i] = last

    return frame_to_tick


class SequentialVideoReader:
    """Decode a video file exactly once in forward order — no seeking, no re-opens."""

    def __init__(self, video_path: str):
        import av as _av
        self._container = _av.open(video_path)
        self._iter      = self._container.decode(video=0)
        self._pos       = -1          # index of last decoded frame
        self._last      = None        # last decoded array (uint8 H×W×3)

    def get(self, frame_num: int):
        """Return the frame at index frame_num (must be called in non-decreasing order)."""
        while self._pos < frame_num:
            try:
                frm = next(self._iter)
                self._pos  += 1
                self._last  = frm.to_ndarray(format="rgb24")
            except StopIteration:
                break
        return self._last

    def close(self):
        self._container.close()


def main():
    import av
    import numpy as np
    from PIL import Image

    session_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_session()
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    log.info("Session: %s  max_seconds=%.0f", session_dir, max_seconds)

    labels_dir  = session_dir / "labels"
    renders_dir = labels_dir / "test_renders"
    renders_dir.mkdir(exist_ok=True)
    out_path    = renders_dir / "comparison.mp4"

    ticks = load_ticks(session_dir)
    log.info("Loaded %d ticks", len(ticks))

    frame_index = build_frame_index(session_dir, ticks)
    if not frame_index:
        log.error("No frame mapping found")
        return

    fps = 20
    max_frames = min(len(frame_index), int(max_seconds * fps))
    log.info("Video frames to render: %d  (%.0f s)", max_frames, max_frames / fps)

    last_tick_num = None
    last_voxel    = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    out_container = av.open(str(out_path), mode='w')
    stream        = out_container.add_stream('h264', rate=Fraction(fps, 1))
    stream.width   = IMG_W * 2
    stream.height  = IMG_H
    stream.pix_fmt = 'yuv420p'

    video_path = str(session_dir / "video.mp4")
    vid_reader = SequentialVideoReader(video_path)
    t0 = time.time()

    try:
        for frame_num in range(max_frames):
            if frame_num % (fps * 5) == 0:
                log.info("[%d/%d]  %.0fs  elapsed=%.0fs",
                         frame_num, max_frames, frame_num / fps, time.time() - t0)

            tick_num = frame_index[frame_num]

            # Re-render voxel only when the tick changes
            if tick_num != last_tick_num and tick_num is not None:
                try:
                    m, b, s, inv, pose = load_tick_npz(labels_dir, tick_num)
                    last_voxel = render_perspective_view(m, b, pose, fullbright=True)
                except Exception as e:
                    log.warning("Failed tick %s: %s", tick_num, e)
                last_tick_num = tick_num

            # Left panel: video frame (sequential decode — no file-reopen)
            video_arr = vid_reader.get(frame_num)
            if video_arr is not None and video_arr.shape[:2] != (IMG_H, IMG_W):
                video_arr = np.array(Image.fromarray(video_arr).resize((IMG_W, IMG_H)))

            composite = make_side_by_side(video_arr, last_voxel)
            av_frame  = av.VideoFrame.from_ndarray(composite, format='rgb24')
            for pkt in stream.encode(av_frame):
                out_container.mux(pkt)
    finally:
        vid_reader.close()

    for pkt in stream.encode():
        out_container.mux(pkt)
    out_container.close()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    log.info("Done: %s  (%.1f MB, %.0fs)", out_path, size_mb, elapsed)


if __name__ == "__main__":
    main()
