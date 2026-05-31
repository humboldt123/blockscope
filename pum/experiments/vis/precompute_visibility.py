"""
precompute_visibility.py — Compute Furnace raycaster visibility masks for all sessions.

For each session, loads world_states.bin + ticks.jsonl and runs compute_visibility
at every TICK_STRIDE tick, saving results to:
  /data/vvm33/SMELTED_DATA/{session}/labels/visibility_s20.npz

Output format:
  tick_indices : (K,)           int32  — tick indices computed
  visibility   : (K, 32, 32, 32) uint8  — Furnace m arrays, one per tick

Estimated runtime: ~40-60 min for all 88 sessions on this server.
Run as: nohup python precompute_visibility.py > /data/vvm33/precompute_vis.log 2>&1 &
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

FURNACE_PYTHON  = "/home/vvm33/blockscope/furnace/pipeline/src/python"
PIPELINE_ROOT   = Path("/home/vvm33/blockscope/furnace/pipeline")
SMELTED_ROOT    = Path("/data/vvm33/SMELTED_DATA")
RAW_ROOT        = Path("/data/vvm33/BLOCKSCOPE_DATA")
TICK_STRIDE     = 20
OUTPUT_STEM     = "visibility_s20.npz"

sys.path.insert(0, FURNACE_PYTHON)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    from io_helpers import load_blockstate_table, load_world_states

    log.info("Loading blockstate table from %s", PIPELINE_ROOT)
    opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs = \
        load_blockstate_table(PIPELINE_ROOT)
    log.info("Blockstate table loaded")

    # Numba JIT warm-up: import raycaster so the decorator runs
    from raycaster import compute_visibility

    all_sessions = sorted(
        d.name for d in SMELTED_ROOT.iterdir()
        if d.is_dir() and (d / "labels" / "world_states.bin").exists()
    )
    log.info("Found %d sessions to process", len(all_sessions))

    total_ticks = 0
    t0_all = time.time()

    for sess_i, session in enumerate(all_sessions):
        labels = SMELTED_ROOT / session / "labels"
        out_path = labels / OUTPUT_STEM
        if out_path.exists():
            log.info("[%d/%d] %s — already done, skipping",
                     sess_i + 1, len(all_sessions), session)
            continue

        raw = RAW_ROOT / session
        ticks_path = raw / "ticks.jsonl"
        if not ticks_path.exists():
            log.warning("[%d/%d] %s — no ticks.jsonl, skipping",
                        sess_i + 1, len(all_sessions), session)
            continue

        t0 = time.time()
        px_arr, py_arr, pz_arr, blocks = load_world_states(labels)
        T = len(px_arr)

        with open(ticks_path) as f:
            ticks = [json.loads(line) for line in f]

        tick_indices = list(range(0, min(T, len(ticks)), TICK_STRIDE))
        vis_out = np.zeros((len(tick_indices), 32, 32, 32), dtype=np.uint8)

        n_computed = 0
        for k, tick_idx in enumerate(tick_indices):
            tick   = ticks[tick_idx]
            player = tick.get("player", tick)

            # Ensure tick has the structure compute_visibility expects
            if "player" not in tick:
                tick = {"player": tick}

            px = int(px_arr[tick_idx])
            py = int(py_arr[tick_idx])
            pz = int(pz_arr[tick_idx])

            try:
                m = compute_visibility(
                    blocks[tick_idx],
                    tick,
                    (px, py, pz),
                    opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs,
                )
                vis_out[k] = m
                n_computed += 1
            except Exception as e:
                log.debug("tick %d failed: %s", tick_idx, e)

        np.savez_compressed(
            out_path,
            tick_indices = np.array(tick_indices, dtype=np.int32),
            visibility   = vis_out,
        )
        elapsed = time.time() - t0
        total_ticks += n_computed
        log.info(
            "[%d/%d] %s — %d ticks in %.1fs (%.0f ms/tick)",
            sess_i + 1, len(all_sessions), session,
            n_computed, elapsed, 1000 * elapsed / max(n_computed, 1),
        )

    log.info("Done. %d total ticks in %.0fs", total_ticks, time.time() - t0_all)


if __name__ == "__main__":
    main()
