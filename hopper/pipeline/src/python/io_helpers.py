"""
io_helpers.py — load/save world_states.bin, blockstate table, tick npz files, manifest.
"""

import json
import logging
import math
import struct
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAGIC = b"WSBIN001"
WINDOW = 32  # 32³ block window


# ---------- blockstate table -------------------------------------------------

def load_blockstate_table(pipeline_root: Path):
    """
    Load pipeline/data/blockstate_table_1.19.4.json and return numpy arrays
    suitable for the Numba raycaster:
      opaque     : uint8[max_state_id+1]   — 1=opaque, 0=transparent
      aabb_start : int32[max_state_id+1]   — start index into flat_aabbs
      aabb_count : int32[max_state_id+1]   — number of AABBs
      flat_aabbs : float32[N, 6]           — all AABBs packed
    """
    json_path = pipeline_root / "data" / "blockstate_table_1.19.4.json"
    log.info("Loading blockstate table from %s", json_path)
    with open(json_path, "r") as f:
        table = json.load(f)

    states = table["states"]
    n = len(states)

    opaque     = np.zeros(n, dtype=np.uint8)
    aabb_start = np.zeros(n, dtype=np.int32)
    aabb_count = np.zeros(n, dtype=np.int32)

    aabb_list = []
    offset = 0
    for sid, entry in enumerate(states):
        opaque[sid]     = 1 if entry["opaque"] else 0
        aabbs           = entry["aabbs"]          # [[x0,y0,z0,x1,y1,z1], ...]
        aabb_start[sid] = offset
        aabb_count[sid] = len(aabbs)
        for box in aabbs:
            aabb_list.extend(box)
        offset += len(aabbs)

    flat_aabbs = np.array(aabb_list, dtype=np.float32).reshape(-1, 6) if aabb_list else np.zeros((0, 6), dtype=np.float32)
    log.info("Blockstate table: %d states, %d total AABBs", n, len(aabb_list) // 6)
    return opaque, aabb_start, aabb_count, flat_aabbs


# ---------- world_states.bin -------------------------------------------------

def load_world_states(labels_dir: Path):
    """
    Read <labels_dir>/world_states.bin. Returns:
      px, py, pz : int32[tick_count]  — player block coords per tick
      blocks     : uint16[tick_count, 32, 32, 32]  — block state IDs
    """
    path = labels_dir / "world_states.bin"
    log.info("Loading world states from %s  (%s)", path, _human_size(path.stat().st_size))
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Bad magic in world_states.bin: {magic!r}")
        tick_count = struct.unpack("<I", f.read(4))[0]
        px = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        py = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        pz = np.frombuffer(f.read(tick_count * 4), dtype=np.int32).copy()
        raw = f.read(tick_count * WINDOW**3 * 2)
        blocks = np.frombuffer(raw, dtype=np.uint16).reshape(tick_count, WINDOW, WINDOW, WINDOW).copy()
    log.info("Loaded %d tick world states", tick_count)
    return px, py, pz, blocks


# ---------- per-tick npz output ----------------------------------------------

def save_tick_npz(labels_dir: Path, tick_idx: int, m, b, s, inventory, pose):
    """Save per-tick label arrays to labels/tick_NNNNN.npz."""
    path = labels_dir / f"tick_{tick_idx:05d}.npz"
    np.savez_compressed(
        path,
        m=m,          # uint8 (32,32,32)
        b=b,          # int32 (32,32,32)
        s=s,          # uint8 (32,32,32)
        inventory=np.array(json.dumps(inventory), dtype=object),
        pose=np.array(json.dumps(pose), dtype=object),
    )


def load_tick_npz(labels_dir: Path, tick_idx: int):
    path = labels_dir / f"tick_{tick_idx:05d}.npz"
    data = np.load(path, allow_pickle=True)
    return (
        data["m"],
        data["b"],
        data["s"],
        json.loads(str(data["inventory"])),
        json.loads(str(data["pose"])),
    )


# ---------- manifest ---------------------------------------------------------

def write_manifest(labels_dir: Path, ticks, session_dir: Path):
    manifest = {
        "session": session_dir.name,
        "tick_count": len(ticks),
        "tick_indices": [t["tick"] for t in ticks],
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = labels_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: %d ticks", len(ticks))
    return path


def manifest_is_current(labels_dir: Path, session_dir: Path) -> bool:
    """Return True if manifest.json exists and world_states.bin is not newer."""
    manifest = labels_dir / "manifest.json"
    ws_bin   = labels_dir / "world_states.bin"
    if not manifest.exists():
        return False
    mtime = manifest.stat().st_mtime
    # Regenerate if input data is newer than the manifest
    for src in (session_dir / "ticks.jsonl", session_dir / "frame_mapping.jsonl"):
        if src.exists() and src.stat().st_mtime > mtime:
            return False
    return True


# ---------- camera math ------------------------------------------------------

def camera_vectors(yaw_deg: float, pitch_deg: float):
    """
    Return (fwd, right, up) unit vectors for the Minecraft camera.

    Minecraft conventions:
      yaw=0   → facing south (+Z)
      yaw=90  → facing west  (-X)
      yaw=180 → facing north (-Z)
      pitch>0 → looking down

    right_z = +sin(yaw), NOT -sin — confirmed by strafing direction.
    up = cross(fwd, right).
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)

    fwd = np.array([
        -math.sin(y) * math.cos(p),
        -math.sin(p),
         math.cos(y) * math.cos(p),
    ], dtype=np.float64)

    right = np.array([math.cos(y), 0.0, math.sin(y)], dtype=np.float64)

    up = np.cross(fwd, right)  # = (-sin(y)*sin(p), cos(p), cos(y)*sin(p))

    return fwd, right, up


def player_screen_bbox(
    tick: dict, width: int, height: int,
    fwd, right, up, half_w: float, half_h: float,
) -> tuple[int, int, int, int]:
    """
    Project the player model AABB onto the screen and return
    (col_min, col_max, row_min, row_max) in pixel coords.

    Used to mask out the player model in F5 third-person mode.
    The player model occupies [x±0.3, y_feet..y_feet+1.8, z±0.3].
    player.*  values in tick data are eye height (camera.y ≈ player.y in first-person),
    so feet_y = player.y - 1.62.
    """
    cam = tick["player"]["camera"]
    player = tick["player"]
    cam_pos = np.array([cam["x"], cam["y"], cam["z"]])
    # Player feet position
    feet_y = player["y"] - 1.62
    px, pz = player["x"], player["z"]

    # 8 corners of the player AABB
    corners = []
    for dx in (-0.35, 0.35):
        for dy in (0.0, 1.85):       # slight extra margin
            for dz in (-0.35, 0.35):
                corners.append(np.array([px + dx, feet_y + dy, pz + dz]))

    cols, rows = [], []
    for corner in corners:
        rel = corner - cam_pos
        fwd_dist = float(np.dot(rel, fwd))
        if fwd_dist <= 0.01:
            continue  # behind camera
        screen_x = float(np.dot(rel, right)) / (fwd_dist * half_w)  # NDC
        screen_y = -float(np.dot(rel, up))   / (fwd_dist * half_h)  # NDC (inverted Y)
        col = int((screen_x + 1.0) * 0.5 * width)
        row = int((screen_y + 1.0) * 0.5 * height)
        cols.append(col)
        rows.append(row)

    if not cols:
        return (0, -1, 0, -1)  # empty (no valid corners projected)

    margin = 4  # extra pixel margin
    c0 = max(0, min(cols) - margin)
    c1 = min(width - 1, max(cols) + margin)
    r0 = max(0, min(rows) - margin)
    r1 = min(height - 1, max(rows) + margin)
    return c0, c1, r0, r1


# ---------- misc -------------------------------------------------------------

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
