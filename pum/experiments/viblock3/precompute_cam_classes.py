"""
Precompute camera-relative class label arrays for all viblock3 ticks.

For each tick saves (32, 32, 32) int8 cam_classes to:
  SMELTED_ROOT/<session>/labels/cam_classes/<tick_idx:06d>.npy

Eliminates the 375MB world_states.bin load per session during training.
Disk cost: ~32KB/tick × ~20k ticks ≈ 660MB total.

Yaw is snapped to the nearest 90° before rotation so that:
  - every block maps to an exact integer cam-space position (no aliasing)
  - the 32³ capture window fully covers the rotated cam grid (no corner gaps)
  - the model sees only 4 canonical orientations (N/E/S/W)

Run on CPU (no GPU needed):
  PYTHONPATH=/home/vvm33/blockscope python precompute_cam_classes.py --force
"""

import logging
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pum.data.vis_dataset import SMELTED_ROOT
from pum.experiments.viblock3.dataset import STEM, build_class_lut, rotate_blocks_yaw, snap_yaw

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CAM_CLASSES_DIR = "cam_classes"
VOCAB_PATH      = Path("/data/vvm33/pum_checkpoints/blocktype/vocab.json")
_MAGIC = b"WSBIN001"


def load_world_states(labels_dir: Path):
    """Load world_states.bin — verbatim from furnace/pipeline/labeler.py."""
    path = labels_dir / "world_states.bin"
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError(f"Bad magic in world_states.bin: {magic!r}")
        tick_count = struct.unpack("<I", f.read(4))[0]
        px = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        py = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        pz = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        raw = f.read(tick_count * 32 ** 3 * 2)
        blocks = np.frombuffer(raw, dtype=np.uint16).reshape(
            tick_count, 32, 32, 32
        ).copy()
    return px, py, pz, blocks


def precompute_session(session: str, lut: np.ndarray, force: bool = False) -> int:

    labels_dir   = SMELTED_ROOT / session / "labels"
    data_path    = labels_dir / STEM
    cam_cls_dir  = labels_dir / CAM_CLASSES_DIR
    ws_path      = labels_dir / "world_states.bin"

    if not data_path.exists() or not ws_path.exists():
        return 0

    d            = np.load(data_path)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]

    pending = [
        i for i in range(len(tick_indices))
        if force or not (cam_cls_dir / f"{int(tick_indices[i]):06d}.npy").exists()
    ]
    if not pending:
        log.info("skip %s (all %d cached)", session, len(tick_indices))
        return 0

    cam_cls_dir.mkdir(exist_ok=True)

    _, _, _, blocks = load_world_states(labels_dir)

    n_saved = 0
    for i in pending:
        tick_idx  = int(tick_indices[i])
        raw_yaw   = float(tick_yaws[i])
        snapped, _ = snap_yaw(raw_yaw)
        if tick_idx >= len(blocks):
            continue
        cam_u16  = rotate_blocks_yaw(blocks[tick_idx], snapped)
        flat        = cam_u16.reshape(-1)
        cam_classes = lut[np.clip(flat, 0, len(lut) - 1)].reshape(32, 32, 32).astype(np.int8)
        np.save(str(cam_cls_dir / f"{tick_idx:06d}.npy"), cam_classes)
        n_saved += 1

    log.info("%s: saved %d/%d cam_classes", session, n_saved, len(pending))
    return n_saved


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lut = build_class_lut(VOCAB_PATH)

    sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / STEM).exists()
    )
    log.info("Processing %d sessions", len(sessions))

    total = sum(precompute_session(s, lut, args.force) for s in sessions)
    log.info("Done: %d cam_classes saved", total)


if __name__ == "__main__":
    main()
