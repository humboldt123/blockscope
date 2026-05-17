"""
pipeline.py — top-level orchestrator.

Usage:
  python -m pipeline.src.python.pipeline <session_dir>

Runs Stage 1 (JS: .mcpr → world_states.bin) then Stage 2 (Python: raycaster
+ seen_before → per-tick .npz files) and writes labels/manifest.json.

Idempotent: if manifest.json exists and is current, exits immediately.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Resolve pipeline root regardless of working directory
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent   # .../pipeline/
JS_PARSE      = PIPELINE_ROOT / "src" / "js" / "parse_mcpr.js"


def run_stage1(session_dir: Path, labels_dir: Path):
    """Shell out to parse_mcpr.js to produce world_states.bin."""
    world_bin = labels_dir / "world_states.bin"
    if world_bin.exists():
        # Regenerate only if .mcpr is newer
        mcpr_files = list(session_dir.glob("*.mcpr"))
        if mcpr_files and mcpr_files[0].stat().st_mtime <= world_bin.stat().st_mtime:
            log.info("Stage 1: world_states.bin is current — skipping")
            return
    log.info("Stage 1: parsing .mcpr → world_states.bin …")
    t0 = time.time()
    result = subprocess.run(
        ["node", str(JS_PARSE), str(session_dir), str(labels_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"parse_mcpr.js exited with code {result.returncode}")
    log.info("Stage 1 done in %.1f s", time.time() - t0)


def run_stage2(session_dir: Path, labels_dir: Path):
    """Raycaster + seen_before → per-tick .npz files."""
    import sys
    sys.path.insert(0, str(PIPELINE_ROOT / "src" / "python"))
    from io_helpers import (
        load_blockstate_table, load_world_states, save_tick_npz, write_manifest
    )
    from raycaster import compute_visibility
    from seen_before import compute_seen_before

    log.info("Stage 2: loading data …")
    t0 = time.time()

    ticks = [json.loads(l) for l in
             (session_dir / "ticks.jsonl").read_text().splitlines() if l]

    opaque, aabb_start, aabb_count, flat_aabbs = load_blockstate_table(PIPELINE_ROOT)
    px_arr, py_arr, pz_arr, blocks = load_world_states(labels_dir)

    T = len(ticks)
    assert len(px_arr) == T, f"tick count mismatch: {len(px_arr)} vs {T}"

    # ── Raycaster pass ───────────────────────────────────────────────────────
    log.info("Stage 2: raycasting %d ticks …", T)
    t_ray = time.time()

    m_all = np.zeros((T, 32, 32, 32), dtype=np.uint8)
    b_all = blocks.astype(np.int32)   # (T,32,32,32)

    for t_idx, tick in enumerate(ticks):
        m = compute_visibility(
            blocks[t_idx],
            tick,
            (int(px_arr[t_idx]), int(py_arr[t_idx]), int(pz_arr[t_idx])),
            opaque, aabb_start, aabb_count, flat_aabbs,
        )
        m_all[t_idx] = m

        if t_idx % 100 == 0:
            log.info("  raycaster tick %d/%d  visible_cells=%d",
                     t_idx, T, int(m.sum()))

    log.info("Raycasting done in %.1f s", time.time() - t_ray)

    # ── seen_before pass ─────────────────────────────────────────────────────
    log.info("Stage 2: computing seen_before …")
    t_sb = time.time()
    s_all = compute_seen_before(px_arr, py_arr, pz_arr, m_all, b_all)
    log.info("seen_before done in %.1f s", time.time() - t_sb)

    # ── Write per-tick npz files ─────────────────────────────────────────────
    log.info("Stage 2: writing per-tick .npz files …")
    for t_idx, tick in enumerate(ticks):
        # camera is absent on the first few ticks before GameRenderer.getCamera() is ready;
        # fall back to the player entity position so downstream code always has a valid pose.
        cam = tick["player"].get("camera") or tick["player"]
        pose = {
            "x": cam["x"],
            "y": cam["y"],
            "z": cam["z"],
            "yaw":   cam["yaw"],
            "pitch": cam["pitch"],
            "fov":   tick["player"].get("fov", 70.0),
            "perspective": tick["player"].get("cameraPerspective", "first_person"),
        }
        save_tick_npz(
            labels_dir,
            tick["tick"],
            m_all[t_idx],
            b_all[t_idx],
            s_all[t_idx],
            tick.get("inventory", {}),
            pose,
        )

    write_manifest(labels_dir, ticks, session_dir)
    log.info("Stage 2 done in %.1f s total", time.time() - t0)


def process_session(session_dir: Path):
    session_dir = Path(session_dir).resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    labels_dir = session_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    from io_helpers import manifest_is_current
    import sys
    sys.path.insert(0, str(PIPELINE_ROOT / "src" / "python"))

    if manifest_is_current(labels_dir, session_dir):
        log.info("Manifest is current — nothing to do for %s", session_dir.name)
        return

    t0 = time.time()
    run_stage1(session_dir, labels_dir)
    run_stage2(session_dir, labels_dir)
    log.info("Session %s processed in %.1f s", session_dir.name, time.time() - t0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <session_dir>")
        sys.exit(1)
    process_session(Path(sys.argv[1]))
