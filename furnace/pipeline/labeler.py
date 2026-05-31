"""
pipeline.py — batch label generation orchestrator.

For each tick in a session, computes the GPU visibility mask and saves
tick_NNNNN.npz to session/labels/.

Usage:
  uv run python -m pipeline.labeler --session path/to/session [--version 1.19.4] [--jar path]

--- Attributions ---
load_world_states():
    Copied verbatim from furnace/pipeline/src/python/io_helpers.py

JS replay parse stage (world_states.bin generation):
    furnace/pipeline/src/js/parse_mcpr.js — uses PrismarineJS packages:
      prismarine-chunk, prismarine-nbt, prismarine-world (PrismarineJS org)
      yauzl (zip reading, © Gilbert Roncoli)
      minecraft-data (PrismarineJS org)
"""

import argparse
import json
import logging
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent          # furnace/pipeline/
DEFAULT_JAR = (
    _ROOT.parent / "visualizer" / ".cache" / "minecraft-1.19.4.jar"
)
MAGIC = b"WSBIN001"
WINDOW = 32


# ── world_states.bin loader (verbatim from furnace/pipeline/src/python/io_helpers.py) ──

def load_world_states(labels_dir: Path):
    path = labels_dir / "world_states.bin"
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Bad magic in world_states.bin: {magic!r}")
        tick_count = struct.unpack("<I", f.read(4))[0]
        px = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        py = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        pz = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        raw = f.read(tick_count * WINDOW ** 3 * 2)
        blocks = np.frombuffer(raw, dtype=np.uint16).reshape(
            tick_count, WINDOW, WINDOW, WINDOW
        ).copy()
    return px, py, pz, blocks


def _snap_yaw(yaw_deg: float):
    """Snap yaw to nearest 90° cardinal. Returns (snapped, local_residual)."""
    s = float(round(yaw_deg / 90) * 90)
    return s, yaw_deg - s


def _rotate_grid_yaw(grid: np.ndarray, yaw_deg: float) -> np.ndarray:
    """
    Rotate a (32,32,32) array around the Y axis by yaw_deg (must be 90° multiple).
    Works for any dtype (block IDs, visibility masks, etc).
    Non-zero values are moved to their rotated positions; zeros fill the rest.
    """
    yaw_rad = np.radians(yaw_deg)
    cos_y, sin_y = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    xi, yi, zi = np.nonzero(grid)
    if len(xi) == 0:
        return np.zeros_like(grid)
    vals  = grid[xi, yi, zi]
    x_rel = xi.astype(np.float32) - 15.0
    z_rel = zi.astype(np.float32) - 15.0
    xi_cam = np.round(x_rel * cos_y + z_rel * sin_y).astype(np.int32) + 15
    zi_cam = np.round(-x_rel * sin_y + z_rel * cos_y).astype(np.int32) + 15
    valid  = (0 <= xi_cam) & (xi_cam < 32) & (0 <= zi_cam) & (zi_cam < 32)
    result = np.zeros_like(grid)
    result[xi_cam[valid], yi[valid], zi_cam[valid]] = vals[valid]
    return result


def run(session_dir: Path, version: str = "1.19.4", jar_path: Path = None,
        max_ticks: int = None, fast_graphics: bool = False,
        localize: bool = False) -> None:
    if jar_path is None:
        jar_path = DEFAULT_JAR

    labels_dir = session_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # ── JS stage: parse_mcpr → world_states.bin ───────────────────────────────
    ws_bin = labels_dir / "world_states.bin"
    if not ws_bin.exists():
        log.info("Running JS stage: parse_mcpr.js ...")
        result = subprocess.run(
            ["node",
             str(_ROOT / "src" / "js" / "parse_mcpr.js"),
             str(session_dir),
             str(labels_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"parse_mcpr.js failed (exit {result.returncode}):\n{result.stderr}"
            )
        log.info("JS stage complete.")

    # ── Baker ──────────────────────────────────────────────────────────────────
    from .baker import bake, _ROOT as NR
    cache_version = f"{version}-fast" if fast_graphics else version
    cache_dir = NR / "cache"
    geo_cache = cache_dir / cache_version / f"geometry_{cache_version}.npz"
    if not geo_cache.exists():
        log.info("Running baker ...")
        bake(jar_path, version, fast_graphics=fast_graphics)
    version = cache_version

    # ── Load world states ──────────────────────────────────────────────────────
    px, py, pz, blocks = load_world_states(labels_dir)
    tick_count = len(px)
    log.info("Loaded %d ticks from world_states.bin", tick_count)

    # ── Load tick poses ────────────────────────────────────────────────────────
    ticks_path = session_dir / "ticks.jsonl"
    ticks = []
    for line in ticks_path.read_text().splitlines():
        if line.strip():
            ticks.append(json.loads(line))
    log.info("Loaded %d tick poses from ticks.jsonl", len(ticks))

    # ── Create renderer ────────────────────────────────────────────────────────
    from .renderer import Renderer
    renderer = Renderer(cache_dir, version)

    # ── Process ticks ──────────────────────────────────────────────────────────
    limit = min(tick_count, max_ticks) if max_ticks is not None else tick_count
    processed = 0
    try:
        for tick_idx in range(limit):
            out_path = labels_dir / f"tick_{tick_idx:05d}.npz"
            if out_path.exists():
                continue

            if tick_idx >= len(ticks):
                log.warning("No pose for tick %d (ticks.jsonl has %d entries), skipping",
                            tick_idx, len(ticks))
                continue

            pose = ticks[tick_idx]["player"]
            world_u16 = blocks[tick_idx]
            pbp = (int(px[tick_idx]), int(py[tick_idx]), int(pz[tick_idx]))

            m = renderer.compute_visibility(world_u16, pose, pbp)
            b = world_u16.astype(np.int32)

            save_kwargs: dict = {
                "inventory": np.array(json.dumps(pose.get("inventory", [])), dtype=object),
                "pose":      np.array(json.dumps(pose), dtype=object),
            }

            if localize:
                cam     = pose.get("camera", pose)
                raw_yaw = float(cam.get("yaw", pose.get("yaw", 0.0)))
                snapped, local_yaw = _snap_yaw(raw_yaw)
                b = _rotate_grid_yaw(world_u16, snapped).astype(np.int32)
                m = _rotate_grid_yaw(m, snapped)
                save_kwargs["local_yaw"] = np.float32(local_yaw)

            save_kwargs.update({"m": m, "b": b})
            np.savez(str(out_path), **save_kwargs)   # no compression — small arrays, speed > space
            processed += 1

            if tick_idx % 100 == 0:
                log.info("  Processed %d / %d ticks", tick_idx, tick_count)

    finally:
        renderer.close()

    log.info("Done. Wrote %d tick NPZ files to %s", processed, labels_dir)

    manifest = {
        "session": session_dir.name,
        "tick_count": tick_count,
        "processed_ticks": processed,
        "version": version,
        "fast_graphics": fast_graphics,
        "localize": localize,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (labels_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch visibility label generation.")
    parser.add_argument("--session", required=True, type=Path,
                        help="Path to session directory")
    parser.add_argument("--version", default="1.19.4",
                        help="Minecraft version tag")
    parser.add_argument("--jar", type=Path, default=None,
                        help="Path to Minecraft JAR (for baker)")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Stop after processing this many ticks (for testing)")
    parser.add_argument("--fast-graphics", action="store_true",
                        help="Use fast-graphics leaf cache (leaves fully opaque)")
    parser.add_argument("--localize", action="store_true",
                        help="Snap yaw to nearest 90° and rotate b+m to camera-relative space "
                             "before saving (player always faces +Z in the output grid)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args.session, args.version, args.jar, args.max_ticks, args.fast_graphics,
        args.localize)


if __name__ == "__main__":
    main()
