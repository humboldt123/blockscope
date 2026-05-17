"""
raycaster.py — Numba-parallel DDA raycaster for the 32³ visibility window.

No Taichi available in this environment; Numba @njit(parallel=True) is used
instead. With 16 CPU cores the throughput is sufficient for offline processing
(≈100–300 ms per tick after JIT warm-up).

V1 limitations (see README):
  - Entities not modelled as occluders.
  - Particles (snow, rain, smoke) ignored.
  - Binary opacity only — no Beer-Lambert.
  - Lighting ignored.
"""

import logging
import math

import numpy as np
from numba import njit, prange

log = logging.getLogger(__name__)

IMG_W = 640
IMG_H = 360
WINDOW = 32

# ── Numba JIT helpers ────────────────────────────────────────────────────────

@njit(cache=True)
def _ray_aabb_hit(ox, oy, oz, dx, dy, dz, ax0, ay0, az0, ax1, ay1, az1):
    """Slab-method ray-AABB intersection; returns True if hit with t >= 0."""
    EPS = 1e-12
    if abs(dx) > EPS:
        t1 = (ax0 - ox) / dx;  t2 = (ax1 - ox) / dx
        if t1 > t2: t1, t2 = t2, t1
    else:
        if ox < ax0 or ox > ax1: return False
        t1 = -1e30;  t2 = 1e30

    if abs(dy) > EPS:
        t3 = (ay0 - oy) / dy;  t4 = (ay1 - oy) / dy
        if t3 > t4: t3, t4 = t4, t3
    else:
        if oy < ay0 or oy > ay1: return False
        t3 = -1e30;  t4 = 1e30

    if abs(dz) > EPS:
        t5 = (az0 - oz) / dz;  t6 = (az1 - oz) / dz
        if t5 > t6: t5, t6 = t6, t5
    else:
        if oz < az0 or oz > az1: return False
        t5 = -1e30;  t6 = 1e30

    tmin = max(t1, t3, t5)
    tmax = min(t2, t4, t6)
    return tmax >= tmin and tmax >= 0.0


@njit(cache=True)
def _cast_ray(
    cam_wx, cam_wy, cam_wz,   # camera in window coords
    dx, dy, dz,                # normalised ray direction
    world,                     # uint16 (32,32,32)
    opaque,                    # uint8  (N,)  — unused here, kept for symmetry
    terminates_ray,            # uint8  (N,)  — 1 if ray stops (opaque OR leaf)
    aabb_start,                # int32  (N,)
    aabb_count,                # int32  (N,)
    flat_aabbs,                # float32(M,6)
    visibility,                # uint8  (32,32,32)  — shared, written atomically
):
    EPS = 1e-12

    # Starting cell (camera position floored)
    cx = int(math.floor(cam_wx))
    cy = int(math.floor(cam_wy))
    cz = int(math.floor(cam_wz))

    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    if abs(dx) > EPS:
        tx = (cx + (1 if dx >= 0 else 0) - cam_wx) / dx
        dtx = abs(1.0 / dx)
    else:
        tx = 1e30;  dtx = 1e30

    if abs(dy) > EPS:
        ty = (cy + (1 if dy >= 0 else 0) - cam_wy) / dy
        dty = abs(1.0 / dy)
    else:
        ty = 1e30;  dty = 1e30

    if abs(dz) > EPS:
        tz = (cz + (1 if dz >= 0 else 0) - cam_wz) / dz
        dtz = abs(1.0 / dz)
    else:
        tz = 1e30;  dtz = 1e30

    for _ in range(100):   # max traversal steps (32√3 ≈ 56)
        if cx < 0 or cx >= WINDOW or cy < 0 or cy >= WINDOW or cz < 0 or cz >= WINDOW:
            break

        sid = int(world[cx, cy, cz])
        if sid > 0:
            n = aabb_count[sid]
            if n > 0:
                start = aabb_start[sid]
                hit = False
                for k in range(n):
                    ax0 = cx + flat_aabbs[start + k, 0]
                    ay0 = cy + flat_aabbs[start + k, 1]
                    az0 = cz + flat_aabbs[start + k, 2]
                    ax1 = cx + flat_aabbs[start + k, 3]
                    ay1 = cy + flat_aabbs[start + k, 4]
                    az1 = cz + flat_aabbs[start + k, 5]
                    if _ray_aabb_hit(cam_wx, cam_wy, cam_wz, dx, dy, dz,
                                     ax0, ay0, az0, ax1, ay1, az1):
                        hit = True
                        break
                if hit:
                    visibility[cx, cy, cz] = 1
                    if terminates_ray[sid]:
                        return    # opaque or leaf (Fast graphics) — ray terminates

        # Advance DDA
        if tx <= ty and tx <= tz:
            cx += step_x;  tx += dtx
        elif ty <= tz:
            cy += step_y;  ty += dty
        else:
            cz += step_z;  tz += dtz


@njit(parallel=True, cache=True)
def _cast_all_rays(
    world,
    cam_wx, cam_wy, cam_wz,
    fwd_x, fwd_y, fwd_z,
    right_x, right_y, right_z,
    up_x, up_y, up_z,
    half_w, half_h,
    width, height,
    opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs,
    occ_c0, occ_c1, occ_r0, occ_r1,   # F5 player-model occlusion rect (inclusive)
):
    visibility = np.zeros((WINDOW, WINDOW, WINDOW), dtype=np.uint8)
    inv_w = 1.0 / width
    inv_h = 1.0 / height

    for py in prange(height):
        ndc_y = (py + 0.5) * inv_h * 2.0 - 1.0
        for px in range(width):
            # Skip pixels covered by the player model in F5 mode
            if occ_c0 <= px <= occ_c1 and occ_r0 <= py <= occ_r1:
                continue

            ndc_x = (px + 0.5) * inv_w * 2.0 - 1.0
            rdx = fwd_x + ndc_x * half_w * right_x - ndc_y * half_h * up_x
            rdy = fwd_y + ndc_x * half_w * right_y - ndc_y * half_h * up_y
            rdz = fwd_z + ndc_x * half_w * right_z - ndc_y * half_h * up_z

            # Normalise
            rlen = math.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
            if rlen < 1e-12:
                continue
            rdx /= rlen;  rdy /= rlen;  rdz /= rlen

            _cast_ray(cam_wx, cam_wy, cam_wz, rdx, rdy, rdz,
                      world, opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs,
                      visibility)

    return visibility


# ── Public API ───────────────────────────────────────────────────────────────

_warmed_up = False

def compute_visibility(
    world_u16,         # uint16 (32,32,32) — block state IDs, window coords
    tick,              # tick dict from ticks.jsonl
    player_block_pos,  # (px, py, pz) integer block coords (window origin = pos-15)
    opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs,  # blockstate table arrays
) -> np.ndarray:
    """
    Cast 640×360 rays from the camera and return uint8 (32,32,32) visibility mask.
    m[i,j,k] = 1 if any ray hit cell (i,j,k)'s collision geometry.
    """
    global _warmed_up
    from io_helpers import camera_vectors, player_screen_bbox

    # camera is absent on early ticks before GameRenderer.getCamera() is ready;
    # fall back to the player entity position so rays still originate somewhere sane.
    cam = tick["player"].get("camera") or tick["player"]
    fov = tick["player"].get("fov", 70.0)
    perspective = tick["player"].get("cameraPerspective", "first_person")

    fwd, right, up = camera_vectors(cam["yaw"], cam["pitch"])

    aspect = IMG_W / IMG_H
    half_h = math.tan(math.radians(fov / 2))
    half_w = aspect * half_h

    # Camera position in window coords: world_coord - (px-15, py-15, pz-15)
    px, py, pz = player_block_pos
    cam_wx = cam["x"] - (px - 15)
    cam_wy = cam["y"] - (py - 15)
    cam_wz = cam["z"] - (pz - 15)

    # F5 player-model occlusion (only in third-person modes)
    occ_c0 = occ_r0 = 1
    occ_c1 = occ_r1 = -1   # c0 > c1 → empty rectangle → no occlusion
    if "third_person" in perspective:
        occ_c0, occ_c1, occ_r0, occ_r1 = player_screen_bbox(
            tick, IMG_W, IMG_H, fwd, right, up, half_w, half_h
        )

    if not _warmed_up:
        log.info("Numba JIT warm-up — first tick will be slow (~10-30 s)…")
        _warmed_up = True

    vis = _cast_all_rays(
        world_u16.astype(np.uint16),
        cam_wx, cam_wy, cam_wz,
        float(fwd[0]), float(fwd[1]), float(fwd[2]),
        float(right[0]), float(right[1]), float(right[2]),
        float(up[0]), float(up[1]), float(up[2]),
        half_w, half_h,
        IMG_W, IMG_H,
        opaque, terminates_ray, aabb_start, aabb_count, flat_aabbs,
        int(occ_c0), int(occ_c1), int(occ_r0), int(occ_r1),
    )
    return vis
