"""
lighting.py — approximate Beta 1.7 lighting for the 32³ visibility window.

Sky light:   BFS from top layer (y=31), blocked by opaque blocks.
Block light: BFS from emitters (glowstone, torches, etc.), blocked by opaque blocks.
Brightness:  Beta 1.7 curve from lightmap.fsh — 0.0281 * 1.25^(lv*1.06-0.03)*16 + 0.05
"""

import json
import logging
from pathlib import Path

import numpy as np
from numba import njit

log = logging.getLogger(__name__)

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent

_EMIT_MAP:   np.ndarray = None   # uint8[max_state_id+1]
_OPAQUE_LUT: np.ndarray = None   # bool[max_state_id+1]


def _init_tables() -> None:
    global _EMIT_MAP, _OPAQUE_LUT
    if _EMIT_MAP is not None:
        return

    bst_path = _PIPELINE_ROOT / "data" / "blockstate_table_1.19.4.json"
    bst_states = json.loads(bst_path.read_text())["states"]
    _OPAQUE_LUT = np.array([s.get("opaque", False) for s in bst_states], dtype=np.bool_)

    candidates = [
        _PIPELINE_ROOT / "node_modules" / "minecraft-data" / "data" / "pc" / "1.19.4" / "blocks.json",
        _PIPELINE_ROOT / "node_modules" / "minecraft-data" / "minecraft-data" / "data" / "pc" / "1.19.4" / "blocks.json",
    ]
    mc_path = next((p for p in candidates if p.exists()), None)

    max_id = len(bst_states) - 1
    _EMIT_MAP = np.zeros(max_id + 1, dtype=np.uint8)

    if mc_path is None:
        log.warning("minecraft-data blocks.json not found — block light disabled")
        return

    for b in json.loads(mc_path.read_text()):
        lv = b.get("emitLight", 0)
        if lv > 0:
            for sid in range(b["minStateId"], min(b["maxStateId"] + 1, max_id + 1)):
                _EMIT_MAP[sid] = lv

    log.info("Lighting tables: %d opaque states, %d emitting states",
             int(_OPAQUE_LUT.sum()), int((_EMIT_MAP > 0).sum()))


def beta17_brightness(level: float) -> float:
    """Beta 1.7 brightness curve. Input: [0,1]. Output: [~0.075, 1.0]."""
    lv = float(level) * 1.06 - 0.03
    return min(1.0, max(0.0, 0.0281 * (1.25 ** (lv * 16)) + 0.05))


# ── Numba BFS kernel ──────────────────────────────────────────────────────────

@njit(cache=True)
def _bfs_sky_nb(light_map, is_opaque, W):
    """
    Sky-light BFS: transparent cells transmit full strength (no decrease),
    opaque cells block propagation entirely. This matches Minecraft's sky
    light model where sunlight doesn't attenuate through air.
    """
    MAXQ = W * W * W * 6
    qx = np.zeros(MAXQ, dtype=np.int32)
    qy = np.zeros(MAXQ, dtype=np.int32)
    qz = np.zeros(MAXQ, dtype=np.int32)
    head = 0
    tail = 0

    for x in range(W):
        for y in range(W):
            for z in range(W):
                if light_map[x, y, z] > 0:
                    qx[tail] = x; qy[tail] = y; qz[tail] = z; tail += 1

    ddx = np.array([1, -1, 0, 0, 0,  0], dtype=np.int32)
    ddy = np.array([0,  0, 1, -1, 0,  0], dtype=np.int32)
    ddz = np.array([0,  0, 0,  0, 1, -1], dtype=np.int32)

    while head < tail:
        cx = qx[head]; cy = qy[head]; cz = qz[head]; head += 1
        lv = int(light_map[cx, cy, cz])
        for d in range(6):
            nx = cx + ddx[d]; ny = cy + ddy[d]; nz = cz + ddz[d]
            if 0 <= nx < W and 0 <= ny < W and 0 <= nz < W:
                if not is_opaque[nx, ny, nz] and lv > int(light_map[nx, ny, nz]):
                    light_map[nx, ny, nz] = np.uint8(lv)
                    if tail < MAXQ:
                        qx[tail] = nx; qy[tail] = ny; qz[tail] = nz; tail += 1


@njit(cache=True)
def _bfs_flood_nb(light_map, is_opaque, W):
    """Block-light BFS: decreases by 1 per step (torch/lava model)."""
    MAXQ = W * W * W * 6
    qx = np.zeros(MAXQ, dtype=np.int32)
    qy = np.zeros(MAXQ, dtype=np.int32)
    qz = np.zeros(MAXQ, dtype=np.int32)
    head = 0
    tail = 0

    for x in range(W):
        for y in range(W):
            for z in range(W):
                if light_map[x, y, z] > 0:
                    qx[tail] = x; qy[tail] = y; qz[tail] = z; tail += 1

    ddx = np.array([1, -1,  0,  0,  0,  0], dtype=np.int32)
    ddy = np.array([0,  0,  1, -1,  0,  0], dtype=np.int32)
    ddz = np.array([0,  0,  0,  0,  1, -1], dtype=np.int32)

    while head < tail:
        cx = qx[head]; cy = qy[head]; cz = qz[head]; head += 1
        lv  = int(light_map[cx, cy, cz])
        nlv = lv - 1
        if nlv <= 0:
            continue
        for d in range(6):
            nx = cx + ddx[d]; ny = cy + ddy[d]; nz = cz + ddz[d]
            if 0 <= nx < W and 0 <= ny < W and 0 <= nz < W:
                if not is_opaque[nx, ny, nz] and nlv > int(light_map[nx, ny, nz]):
                    light_map[nx, ny, nz] = np.uint8(nlv)
                    if tail < MAXQ:
                        qx[tail] = nx; qy[tail] = ny; qz[tail] = nz; tail += 1


@njit(cache=True)
def _fill_opaque_nb(light_map, is_opaque, W):
    """
    BFS skips opaque cells, leaving them at 0. For face-lighting we sample
    the adjacent cell in the face-normal direction — if that adjacent cell is
    also opaque it would read 0, making side-faces very dark even outdoors.
    Fix: set each opaque cell to the max light of its non-opaque neighbours.
    """
    ddx = np.array([1, -1,  0,  0,  0,  0], dtype=np.int32)
    ddy = np.array([0,  0,  1, -1,  0,  0], dtype=np.int32)
    ddz = np.array([0,  0,  0,  0,  1, -1], dtype=np.int32)
    for x in range(W):
        for y in range(W):
            for z in range(W):
                if is_opaque[x, y, z]:
                    mx = np.uint8(0)
                    for d in range(6):
                        nx = x + ddx[d]
                        ny = y + ddy[d]
                        nz = z + ddz[d]
                        if 0 <= nx < W and 0 <= ny < W and 0 <= nz < W:
                            if not is_opaque[nx, ny, nz]:
                                if light_map[nx, ny, nz] > mx:
                                    mx = light_map[nx, ny, nz]
                    light_map[x, y, z] = mx


def compute_lightmap(b_arr: np.ndarray) -> np.ndarray:
    """
    Approximate per-cell light levels (0-15) for a 32³ window.
    Returns uint8 (32,32,32) — max(sky_light, block_light) per cell.
    """
    _init_tables()
    W = 32

    b_safe     = np.clip(b_arr.astype(np.int64), 0, len(_OPAQUE_LUT) - 1).astype(np.int32)
    is_opaque  = _OPAQUE_LUT[b_safe]

    # Sky light — seed top layer, then BFS (full-strength propagation through air)
    sky = np.zeros((W, W, W), dtype=np.uint8)
    sky[:, W - 1, :] = np.where(~is_opaque[:, W - 1, :], np.uint8(15), np.uint8(0))
    _bfs_sky_nb(sky, is_opaque, W)

    # Block light — seed emitters, then BFS
    emit_safe = np.clip(b_arr.astype(np.int64), 0, len(_EMIT_MAP) - 1).astype(np.int32)
    bl = _EMIT_MAP[emit_safe].astype(np.uint8)
    _bfs_flood_nb(bl, is_opaque, W)

    combined = np.maximum(sky, bl)
    # Fill opaque cells so side-face lighting samples read correct neighbour levels
    _fill_opaque_nb(combined, is_opaque, W)
    return combined
