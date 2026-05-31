"""
smelt.py — on-demand b-only smelting for Stage 1 tokenizer training.

Stage 1 needs only the b (block-state ID) arrays — no visibility raycasting,
no seen-before, no inventory. This module runs only the JS stage of Furnace
(parse_mcpr.js → world_states.bin) and exposes the b arrays directly.

Full smelting (m, s, inventory, pose) is done by furnace/pipeline/src/python/pipeline.py
and is needed from Stage 2 onward.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FURNACE_ROOT = Path(os.environ.get(
    "FURNACE_ROOT", "/home/vvm33/blockscope/furnace/pipeline"
))
JS_PARSE = FURNACE_ROOT / "src" / "js" / "parse_mcpr.js"
SMELTED_ROOT = Path(os.environ.get(
    "SMELTED_ROOT", "/data/vvm33/SMELTED_DATA"
))


def _io_helpers():
    sys.path.insert(0, str(FURNACE_ROOT / "src" / "python"))
    from io_helpers import load_world_states
    return load_world_states


def smelt_session_b(session_dir: Path) -> Path:
    """
    Ensure world_states.bin exists for session_dir.
    Runs parse_mcpr.js if needed. Returns path to world_states.bin.
    """
    session_dir = Path(session_dir)
    labels_dir = SMELTED_ROOT / session_dir.name / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    world_bin = labels_dir / "world_states.bin"
    mcpr_files = sorted(session_dir.glob("*.mcpr"))

    if not mcpr_files:
        raise FileNotFoundError(f"No .mcpr file found in {session_dir}")

    if world_bin.exists():
        # Skip if world_states.bin is newer than the .mcpr
        if mcpr_files[0].stat().st_mtime <= world_bin.stat().st_mtime:
            log.debug("world_states.bin current for %s", session_dir.name)
            return world_bin

    log.info("Smelting (b-only) %s ...", session_dir.name)
    result = subprocess.run(
        ["node", str(JS_PARSE), str(session_dir), str(labels_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"parse_mcpr.js failed (code {result.returncode}) for {session_dir.name}"
        )
    log.info("Smelted %s → %s", session_dir.name, world_bin)
    return world_bin


def load_b_arrays(session_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load b arrays for a session, smelting first if necessary.
    Returns (px, py, pz, blocks) where blocks is uint16 (T, 32, 32, 32).
    """
    world_bin = smelt_session_b(session_dir)
    load_world_states = _io_helpers()
    return load_world_states(world_bin.parent)


def session_tick_count(session_dir: Path) -> int:
    """Return number of ticks in a session (fast, uses ticks.jsonl line count)."""
    ticks_file = Path(session_dir) / "ticks.jsonl"
    if not ticks_file.exists():
        return 0
    with open(ticks_file) as f:
        return sum(1 for line in f if line.strip())
