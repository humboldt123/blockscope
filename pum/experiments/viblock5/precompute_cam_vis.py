"""
precompute_cam_vis.py — generate the per-tick visibility masks viblock5 needs (E10).

This is the enabler for viblock5's headline fix.  Furnace already computes a
32³ visibility mask `m` per tick (renderer.compute_visibility), but the
viblock3/4 label path regenerated block grids straight from world_states.bin and
never carried the mask through — so viblock4 supervised all 32,768 voxels,
including the ~half behind the camera, and its occ_iou partly measured a terrain
prior rather than perception.

For each valid tick this writes a camera-relative visibility mask:
  SMELTED_ROOT/<session>/labels/cam_vis/<tick_idx:06d>.npy   (32,32,32) uint8
    1 = block at this cell was visible from the camera this tick
    0 = occluded / behind the camera / off-grid

The mask is in the SAME camera-relative (yaw-snapped) frame as cam_classes/, so
they line up cell-for-cell.  cam_classes/ is left untouched — viblock5 trains on
the exact same labels viblock4 did, only the loss masking changes, keeping the
re-baseline comparison clean.

Two sources, in priority order (per session):
  1. PREFERRED — Furnace's per-tick `tick_<idx:05d>.npz` (key "m"), written by
     labeler.py.  If the smelt ran with --localize (the default; see
     manifest.json), `m` is already rotated to camera space and is used as-is;
     otherwise it is rotated here by the snapped yaw.  CPU only, fast.
  2. FALLBACK — recompute via the GPU renderer from world_states.bin + ticks.jsonl
     poses (mirrors labeler.compute_visibility), then rotate by the snapped yaw.
     Used only when the per-tick npz is absent.  Needs moderngl + the baker cache.

Run on Brev:
  PYTHONPATH=/home/vvm33/pum python -m pum.experiments.viblock5.precompute_cam_vis [--force] [--session NAME]
"""

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_FURNACE_DIR = Path(__file__).resolve().parents[3] / "furnace"
if str(_FURNACE_DIR) not in sys.path:
    sys.path.insert(0, str(_FURNACE_DIR))

from pum.data.vis_dataset import SMELTED_ROOT, RAW_ROOT
from pum.experiments.viblock3.dataset import STEM, snap_yaw, rotate_blocks_yaw

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CAM_VIS_DIR = "cam_vis"
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
        blocks = np.frombuffer(raw, dtype=np.uint16).reshape(tick_count, 32, 32, 32).copy()
    return px, py, pz, blocks


def _resolve_cache_dir() -> Path | None:
    for cand in [
        Path("/home/vvm33/blockscope/furnace/pipeline/cache"),
        Path(__file__).resolve().parents[3] / "furnace" / "pipeline" / "cache",
    ]:
        if cand.exists():
            return cand
    return None


class _RendererFallback:
    """Lazily loads world_states.bin + ticks.jsonl + GPU renderer for one session.

    Only constructed when a session has no per-tick npz masks on disk.
    """

    def __init__(self, session: str, version: str):
        self.session = session
        self.version = version
        self._renderer = None
        self._blocks = None
        self._px = self._py = self._pz = None
        self._ticks = None

    def _ensure(self):
        if self._renderer is not None:
            return
        labels_dir = SMELTED_ROOT / self.session / "labels"
        cache_dir = _resolve_cache_dir()
        if cache_dir is None:
            raise RuntimeError("no baker cache found — cannot recompute visibility")
        self._px, self._py, self._pz, self._blocks = load_world_states(labels_dir)
        ticks_path = RAW_ROOT / self.session / "ticks.jsonl"
        self._ticks = [json.loads(l) for l in ticks_path.read_text().splitlines() if l.strip()]
        from pipeline.renderer import Renderer
        self._renderer = Renderer(cache_dir, self.version)
        log.info("%s: renderer fallback initialised (%d ticks)", self.session, len(self._ticks))

    def cam_vis(self, tick_idx: int, yaw: float) -> np.ndarray | None:
        self._ensure()
        if tick_idx >= len(self._blocks) or tick_idx >= len(self._ticks):
            return None
        pose = self._ticks[tick_idx]["player"]
        pbp = (int(self._px[tick_idx]), int(self._py[tick_idx]), int(self._pz[tick_idx]))
        m_world = self._renderer.compute_visibility(self._blocks[tick_idx], pose, pbp)
        snap, _ = snap_yaw(yaw)
        m_cam = rotate_blocks_yaw(m_world.astype(np.uint16), snap)
        return (m_cam > 0).astype(np.uint8)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


def precompute_session(session: str, force: bool = False) -> int:
    labels_dir  = SMELTED_ROOT / session / "labels"
    data_path   = labels_dir / STEM
    cam_vis_dir = labels_dir / CAM_VIS_DIR

    if not data_path.exists():
        return 0

    d            = np.load(data_path)
    tick_indices = d["tick_indices"]
    tick_yaws    = d["tick_yaws"]

    # manifest tells us whether the per-tick npz masks are already cam-rotated.
    localize = True
    manifest_path = labels_dir / "manifest.json"
    if manifest_path.exists():
        try:
            localize = bool(json.loads(manifest_path.read_text()).get("localize", True))
        except Exception:
            pass

    version = "1.19.4"
    if manifest_path.exists():
        try:
            version = json.loads(manifest_path.read_text()).get("version", version)
        except Exception:
            pass

    cam_vis_dir.mkdir(exist_ok=True)
    fallback = _RendererFallback(session, version)

    n_saved = n_skipped = n_missing = 0
    for k in range(len(tick_indices)):
        tick_idx = int(tick_indices[k])
        out_path = cam_vis_dir / f"{tick_idx:06d}.npy"
        if out_path.exists() and not force:
            n_skipped += 1
            continue

        cam_vis = None

        # 1. preferred: per-tick npz "m" from the labeler
        tick_npz = labels_dir / f"tick_{tick_idx:05d}.npz"
        if tick_npz.exists():
            try:
                m = np.load(str(tick_npz))["m"]
                if not localize:
                    snap, _ = snap_yaw(float(tick_yaws[k]))
                    m = rotate_blocks_yaw(m.astype(np.uint16), snap)
                cam_vis = (m > 0).astype(np.uint8)
            except Exception as e:
                log.warning("%s tick %d: bad per-tick npz (%s) — trying renderer", session, tick_idx, e)

        # 2. fallback: recompute on GPU
        if cam_vis is None:
            try:
                cam_vis = fallback.cam_vis(tick_idx, float(tick_yaws[k]))
            except Exception as e:
                log.warning("%s tick %d: renderer fallback failed: %s", session, tick_idx, e)
                cam_vis = None

        if cam_vis is None:
            n_missing += 1
            continue

        np.save(str(out_path), cam_vis)
        n_saved += 1

    fallback.close()
    log.info("%s: saved=%d skipped=%d missing=%d (of %d ticks)",
             session, n_saved, n_skipped, n_missing, len(tick_indices))
    if n_missing:
        log.warning("%s: %d ticks have NO visibility source (no per-tick npz and no "
                    "renderer result). Those ticks will be skipped in training. "
                    "Re-smelt with labeler --localize to regenerate masks.",
                    session, n_missing)
    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",   action="store_true", help="Recompute even if cam_vis exists")
    ap.add_argument("--session", default=None,        help="Process a single session only")
    args = ap.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(
            dd.name for dd in SMELTED_ROOT.iterdir()
            if dd.is_dir() and (dd / "labels" / STEM).exists()
        )
    log.info("Processing %d sessions (force=%s)", len(sessions), args.force)

    total = sum(precompute_session(s, force=args.force) for s in sessions)
    log.info("Done: %d cam_vis masks saved", total)


if __name__ == "__main__":
    main()
