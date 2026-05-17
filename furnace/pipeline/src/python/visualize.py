"""
visualize.py — side-by-side renders for the integration test.

Left panel: actual video frame (extracted via PyAV).
Right panel: perspective-correct textured voxel render with Beta 1.7 lighting.

Renderer path:
  1. Texture atlas built once at startup (all 16×16 tiles in one RGBA array).
  2. Per-render: project all face corners in numpy, look up atlas UVs.
  3. Numba @njit kernel rasterises all faces via barycentric interpolation.

This replaces the old per-face homography solve + numpy meshgrid approach and
is ~50-200× faster.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit
from PIL import Image

log = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 360


# ── Video frame extraction ────────────────────────────────────────────────────

def _load_frame_mapping(session_dir: Path) -> dict:
    """Returns dict: tick_index -> frame_index.

    The mod writes {"frame": F, "tick": T} where T is the completed tick whose
    world state is shown in frame F.
    """
    mapping = {}
    for line in (session_dir / "frame_mapping.jsonl").read_text().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        corrected_tick = max(0, obj["tick"] - 1)
        mapping[corrected_tick] = obj["frame"]
    return mapping


def extract_video_frame(session_dir: Path, tick_index: int) -> Optional[np.ndarray]:
    """
    Extract the video frame corresponding to tick_index.
    Returns uint8 (H, W, 3) RGB or None.
    """
    try:
        import av
        fm = _load_frame_mapping(session_dir)
        frame_num = fm.get(tick_index)
        if frame_num is None:
            log.warning("No frame mapping for tick %d", tick_index)
            return None
        video_path = str(session_dir / "video.mp4")
        container = av.open(video_path)
        stream = container.streams.video[0]
        target_pts = int(frame_num * stream.duration / stream.frames) if stream.frames else 0
        container.seek(target_pts, stream=stream, any_frame=True)
        current = 0
        for packet in container.demux(stream):
            for frm in packet.decode():
                if current == frame_num:
                    img = frm.to_ndarray(format="rgb24")
                    container.close()
                    return img.astype(np.uint8)
                current += 1
                if current > frame_num + 5:
                    break
            if current > frame_num + 5:
                break
        container.close()
        container = av.open(video_path)
        for idx, frm in enumerate(container.decode(video=0)):
            if idx == frame_num:
                img = frm.to_ndarray(format="rgb24")
                container.close()
                return img.astype(np.uint8)
        container.close()
        return None
    except Exception as e:
        log.warning("Could not extract video frame for tick %d: %s", tick_index, e)
        return None


# ── Voxel render (legacy orthographic) ───────────────────────────────────────

def _state_color(state_id: int) -> tuple:
    hue = (state_id * 137.508) % 360
    h = hue / 60.0
    c = 0.7 * 0.85
    x = c * (1 - abs(h % 2 - 1))
    m = 0.85 - c
    if   h < 1: r, g, b = c, x, 0
    elif h < 2: r, g, b = x, c, 0
    elif h < 3: r, g, b = 0, c, x
    elif h < 4: r, g, b = 0, x, c
    elif h < 5: r, g, b = x, 0, c
    else:       r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def render_voxel_view(m, b, cam_yaw, cam_pitch, out_w=IMG_W, out_h=IMG_H):
    yaw   = math.radians(cam_yaw)
    pitch = math.radians(cam_pitch)
    fwd   = np.array([-math.sin(yaw)*math.cos(pitch), -math.sin(pitch),
                        math.cos(yaw)*math.cos(pitch)])
    right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    up    = np.cross(fwd, right)
    vis_idx = np.argwhere(m)
    if vis_idx.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    centres = vis_idx.astype(np.float32) + 0.5 - 16.0
    depth   = centres @ fwd
    x_proj  = centres @ right
    y_proj  = centres @ up
    order   = np.argsort(-depth)
    vis_idx = vis_idx[order]; x_proj = x_proj[order]; y_proj = y_proj[order]
    scale   = out_w / 36.0
    img     = np.full((out_h, out_w, 3), 30, dtype=np.uint8)
    cell_px = max(1, int(scale))
    for n in range(len(vis_idx)):
        i, j, k = vis_idx[n]
        col = _state_color(int(b[i, j, k]))
        px_ = int(out_w/2 + x_proj[n]*scale)
        py_ = int(out_h/2 - y_proj[n]*scale)
        x0  = max(0, px_-cell_px); x1 = min(out_w, px_+cell_px+1)
        y0  = max(0, py_-cell_px); y1 = min(out_h, py_+cell_px+1)
        img[y0:y1, x0:x1] = col
    return img


# ── Perspective voxel render ─────────────────────────────────────────────────

_FACE_BRIGHTNESS = {
    ( 0, 1, 0): 1.00,
    ( 0,-1, 0): 0.45,
    ( 1, 0, 0): 0.75,
    (-1, 0, 0): 0.65,
    ( 0, 0, 1): 0.70,
    ( 0, 0,-1): 0.60,
}

_RENDER_BRIGHTNESS = 1.25  # global boost on top of Beta 1.7 curve

_NORMAL_TO_DIR = {
    ( 0, 1, 0): 'up',
    ( 0,-1, 0): 'down',
    ( 0, 0,-1): 'north',
    ( 0, 0, 1): 'south',
    ( 1, 0, 0): 'east',
    (-1, 0, 0): 'west',
}

_BST_DATA:     Optional[dict] = None
_TEX_MAP_DATA: Optional[list] = None
_TEX_CACHE:    dict = {}

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_TEX_DIR  = Path(os.environ.get("MC_TEX_DIR",  _REPO_ROOT / "visualizer" / "assets" / "minecraft" / "textures" / "block"))
_JAR_PATH = Path(os.environ.get("MC_JAR_PATH", _REPO_ROOT / "visualizer" / ".cache" / "minecraft-1.19.4.jar"))

_GRASS_TINT:   Optional[np.ndarray] = None
_FOLIAGE_TINT: Optional[np.ndarray] = None


def _load_colormaps() -> None:
    global _GRASS_TINT, _FOLIAGE_TINT
    if _GRASS_TINT is not None:
        return
    try:
        import zipfile, io as _io
        with zipfile.ZipFile(str(_JAR_PATH)) as zf:
            def _sample(name, row, col):
                arr = np.array(Image.open(_io.BytesIO(zf.read(name))).convert('RGB'))
                return arr[min(row, arr.shape[0]-1), min(col, arr.shape[1]-1)].astype(np.float32) / 255.0
            _GRASS_TINT   = _sample('assets/minecraft/textures/colormap/grass.png',   173, 51)
            _FOLIAGE_TINT = _sample('assets/minecraft/textures/colormap/foliage.png', 173, 51)
    except Exception as e:
        log.warning("Could not load colormaps: %s", e)
        _GRASS_TINT   = np.array([145, 189,  89], dtype=np.float32) / 255.0
        _FOLIAGE_TINT = np.array([119, 171,  47], dtype=np.float32) / 255.0


def _load_bst() -> dict:
    global _BST_DATA
    if _BST_DATA is not None:
        return _BST_DATA
    p = Path(__file__).resolve().parent.parent.parent / "data" / "blockstate_table_1.19.4.json"
    if not p.exists():
        _BST_DATA = {}
        return _BST_DATA
    raw = json.loads(p.read_text())
    _BST_DATA = {i: s for i, s in enumerate(raw["states"])}
    return _BST_DATA


def _get_tex_map() -> list:
    global _TEX_MAP_DATA
    if _TEX_MAP_DATA is not None:
        return _TEX_MAP_DATA
    p = Path(__file__).resolve().parent.parent.parent / "data" / "texture_map_1.19.4.json"
    if not p.exists():
        _TEX_MAP_DATA = []
        return _TEX_MAP_DATA
    _TEX_MAP_DATA = json.loads(p.read_text())["states"]
    return _TEX_MAP_DATA


def _load_texture(name: str, tinted: bool = False) -> Optional[np.ndarray]:
    """Load texture as RGBA uint8 (16×16×4)."""
    key = (name, tinted)
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    tex_path = _TEX_DIR / f"{name}.png"
    if not tex_path.exists():
        _TEX_CACHE[key] = None
        return None
    try:
        arr = np.array(Image.open(str(tex_path)).convert('RGBA'))[:16, :16]
        if tinted:
            _load_colormaps()
            n = name.lower()
            if n == 'grass_block_side':
                ov_path = _TEX_DIR / 'grass_block_side_overlay.png'
                if ov_path.exists():
                    ov    = np.array(Image.open(str(ov_path)).convert('RGBA'))[:16, :16]
                    ov_a  = ov[:, :, 3:4].astype(np.float32) / 255.0
                    ov_rgb = np.clip(ov[:, :, :3].astype(np.float32) * _GRASS_TINT, 0, 255).astype(np.uint8)
                    base  = arr[:, :, :3].astype(np.float32)
                    comp  = np.clip(ov_a * ov_rgb + (1.0 - ov_a) * base, 0, 255).astype(np.uint8)
                    arr   = np.concatenate([comp, arr[:, :, 3:4]], axis=2)
            else:
                tint = _FOLIAGE_TINT if ('leaves' in n or n in ('vine', 'lily_pad')) else _GRASS_TINT
                rgb  = arr[:, :, :3].astype(np.float32) * tint
                arr  = np.concatenate([np.clip(rgb, 0, 255).astype(np.uint8), arr[:, :, 3:4]], axis=2)
        _TEX_CACHE[key] = arr
        return arr
    except Exception:
        _TEX_CACHE[key] = None
        return None


# ── Texture atlas ─────────────────────────────────────────────────────────────

_ATLAS_DATA: Optional[np.ndarray] = None  # (H, W, 4) uint8
_ATLAS_MAP:  dict = {}                     # (name, tinted) -> atlas_id int
_ATLAS_COLS = 64

# UV corners for each rotation (standard: BL, BR, TR, TL in [0,1] UV space)
_ROT_UV = {
    0:   np.array([(0., 1.), (1., 1.), (1., 0.), (0., 0.)], dtype=np.float32),
    90:  np.array([(1., 1.), (1., 0.), (0., 0.), (0., 1.)], dtype=np.float32),
    180: np.array([(1., 0.), (0., 0.), (0., 1.), (1., 1.)], dtype=np.float32),
    270: np.array([(0., 0.), (0., 1.), (1., 1.), (1., 0.)], dtype=np.float32),
}


def _get_atlas() -> np.ndarray:
    global _ATLAS_DATA
    if _ATLAS_DATA is not None:
        return _ATLAS_DATA

    tex_map = _get_tex_map()
    dirs    = ('up', 'down', 'north', 'south', 'east', 'west')
    needed  = set()
    for entry in tex_map:
        if entry is None:
            continue
        for d in dirs:
            name = entry.get(d)
            if name:
                needed.add((name, bool(entry.get(d + '_t'))))

    needed = sorted(needed)
    n      = len(needed)
    n_rows = max(1, (n + _ATLAS_COLS - 1) // _ATLAS_COLS)
    atlas  = np.zeros((n_rows * 16, _ATLAS_COLS * 16, 4), dtype=np.uint8)

    for idx, (name, tinted) in enumerate(needed):
        col = idx % _ATLAS_COLS
        row = idx // _ATLAS_COLS
        tex = _load_texture(name, tinted)
        if tex is not None:
            atlas[row*16:(row+1)*16, col*16:(col+1)*16] = tex
        _ATLAS_MAP[(name, tinted)] = idx

    _ATLAS_DATA = atlas
    log.info("Texture atlas built: %d textures, %dx%d px", n, atlas.shape[1], atlas.shape[0])
    return _ATLAS_DATA


def _face_atlas_uvs(tex_name: str, tinted: bool, uv_rot: int,
                    uv_rect) -> tuple[np.ndarray, np.ndarray]:
    """Return (atlas_u[4], atlas_v[4]) float32 pixel coords for 4 face corners."""
    atlas_id = _ATLAS_MAP.get((tex_name, tinted))
    if atlas_id is None:
        return None, None

    ac = atlas_id % _ATLAS_COLS        # atlas column
    ar = atlas_id // _ATLAS_COLS       # atlas row

    uvs = _ROT_UV.get(uv_rot, _ROT_UV[0]).copy()
    if uv_rect is not None:
        r_u1, r_v1, r_u2, r_v2 = uv_rect
        uvs[:, 0] = r_u1 + uvs[:, 0] * (r_u2 - r_u1)
        uvs[:, 1] = r_v1 + uvs[:, 1] * (r_v2 - r_v1)

    au = (ac * 16.0 + uvs[:, 0] * 16.0).astype(np.float32)
    av = (ar * 16.0 + uvs[:, 1] * 16.0).astype(np.float32)
    return au, av


# ── Numba rasterizer ─────────────────────────────────────────────────────────

@njit(cache=True)
def _rasterize_tri_nb(img,
                      p0x, p0y, p0w,
                      p1x, p1y, p1w,
                      p2x, p2y, p2w,
                      u0, v0, u1, v1, u2, v2,
                      br, atlas):
    """Rasterise one triangle with perspective-correct UV sampling from atlas."""
    H = img.shape[0]
    W = img.shape[1]
    AH = atlas.shape[0]
    AW = atlas.shape[1]

    min_x = max(0, int(min(p0x, p1x, p2x)))
    max_x = min(W - 1, int(max(p0x, p1x, p2x)) + 1)
    min_y = max(0, int(min(p0y, p1y, p2y)))
    max_y = min(H - 1, int(max(p0y, p1y, p2y)) + 1)
    if min_x > max_x or min_y > max_y:
        return

    denom = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
    if abs(denom) < 0.5:
        return
    inv_d = 1.0 / denom

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            pfx = float(px) + 0.5
            pfy = float(py) + 0.5

            lam0 = ((p1y - p2y) * (pfx - p2x) + (p2x - p1x) * (pfy - p2y)) * inv_d
            lam1 = ((p2y - p0y) * (pfx - p2x) + (p0x - p2x) * (pfy - p2y)) * inv_d
            lam2 = 1.0 - lam0 - lam1

            if lam0 < 0.0 or lam1 < 0.0 or lam2 < 0.0:
                continue

            wi = lam0 * p0w + lam1 * p1w + lam2 * p2w
            if wi < 1e-10:
                continue

            au = (lam0 * u0 * p0w + lam1 * u1 * p1w + lam2 * u2 * p2w) / wi
            av = (lam0 * v0 * p0w + lam1 * v1 * p1w + lam2 * v2 * p2w) / wi

            ia = max(0, min(AW - 1, int(au)))
            ja = max(0, min(AH - 1, int(av)))

            r = int(atlas[ja, ia, 0])
            g = int(atlas[ja, ia, 1])
            b = int(atlas[ja, ia, 2])
            a = int(atlas[ja, ia, 3])

            if a < 4:
                continue

            rb = min(255, int(r * br))
            gb = min(255, int(g * br))
            bb = min(255, int(b * br))

            af = a / 255.0
            img[py, px, 0] = min(255, int(af * rb + (1.0 - af) * img[py, px, 0]))
            img[py, px, 1] = min(255, int(af * gb + (1.0 - af) * img[py, px, 1]))
            img[py, px, 2] = min(255, int(af * bb + (1.0 - af) * img[py, px, 2]))


@njit(cache=True)
def _rasterize_all_nb(img, sx, sy, pw, au, av, br, order, n_faces, atlas):
    """Rasterise all faces back-to-front (painter's algorithm)."""
    for fi in range(n_faces):
        f  = order[fi]
        b_ = br[f]
        # Triangle 0: corners 0, 1, 2
        _rasterize_tri_nb(img,
            sx[f, 0], sy[f, 0], pw[f, 0],
            sx[f, 1], sy[f, 1], pw[f, 1],
            sx[f, 2], sy[f, 2], pw[f, 2],
            au[f, 0], av[f, 0], au[f, 1], av[f, 1], au[f, 2], av[f, 2],
            b_, atlas)
        # Triangle 1: corners 0, 2, 3
        _rasterize_tri_nb(img,
            sx[f, 0], sy[f, 0], pw[f, 0],
            sx[f, 2], sy[f, 2], pw[f, 2],
            sx[f, 3], sy[f, 3], pw[f, 3],
            au[f, 0], av[f, 0], au[f, 2], av[f, 2], au[f, 3], av[f, 3],
            b_, atlas)


# ── AABB face geometry ────────────────────────────────────────────────────────

def _aabb_faces(aabb: list) -> list:
    """
    Return 6 (normal_tuple, corners_list) for AABB [x0,y0,z0,x1,y1,z1].
    Corner winding: bottom-left → bottom-right → top-right → top-left viewed
    from outside, so UV corners (0,1),(1,1),(1,0),(0,0) map correctly.
    """
    ax0, ay0, az0, ax1, ay1, az1 = aabb
    return [
        (( 1, 0, 0), [(ax1,ay0,az1),(ax1,ay0,az0),(ax1,ay1,az0),(ax1,ay1,az1)]),
        ((-1, 0, 0), [(ax0,ay0,az0),(ax0,ay0,az1),(ax0,ay1,az1),(ax0,ay1,az0)]),
        (( 0, 1, 0), [(ax0,ay1,az0),(ax1,ay1,az0),(ax1,ay1,az1),(ax0,ay1,az1)]),
        (( 0,-1, 0), [(ax1,ay0,az0),(ax0,ay0,az0),(ax0,ay0,az1),(ax1,ay0,az1)]),
        (( 0, 0, 1), [(ax1,ay0,az1),(ax0,ay0,az1),(ax0,ay1,az1),(ax1,ay1,az1)]),
        (( 0, 0,-1), [(ax0,ay0,az0),(ax1,ay0,az0),(ax1,ay1,az0),(ax0,ay1,az0)]),
    ]


# ── Main perspective renderer ─────────────────────────────────────────────────

def render_perspective_view(
    m: np.ndarray,
    b: np.ndarray,
    pose: dict,
    out_w: int = IMG_W,
    out_h: int = IMG_H,
    fullbright: bool = False,
) -> np.ndarray:
    """
    Perspective-correct textured block render.
    Uses Numba barycentric rasterizer + texture atlas (~50-200× faster than
    the old per-face homography approach).
    fullbright=True skips the light level and renders everything at max brightness.
    """
    from lighting import compute_lightmap, beta17_brightness

    cam_x = float(pose["x"])
    cam_y = float(pose["y"])
    cam_z = float(pose["z"])
    yaw   = math.radians(float(pose["yaw"]))
    pitch = math.radians(float(pose["pitch"]))
    fov_v = math.radians(float(pose.get("fov", 70.0)))

    fwd   = np.array([-math.sin(yaw)*math.cos(pitch),
                       -math.sin(pitch),
                        math.cos(yaw)*math.cos(pitch)], dtype=np.float64)
    right = np.array([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)
    up    = np.cross(fwd, right)

    aspect = out_w / out_h
    tan_hv = math.tan(fov_v / 2)
    tan_hh = tan_hv * aspect
    NEAR   = 0.05   # matches Minecraft's actual near plane
    half_w = out_w / 2.0
    half_h = out_h / 2.0

    px_int = math.floor(cam_x)
    py_int = math.floor(cam_y)
    pz_int = math.floor(cam_z)
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    vis_idx = np.argwhere(m)
    if len(vis_idx) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    bst      = _load_bst()
    tex_map  = _get_tex_map()
    atlas    = _get_atlas()
    lightmap = compute_lightmap(b)

    MAX_FACES = len(vis_idx) * 24    # generous upper bound
    fsx = np.empty((MAX_FACES, 4), dtype=np.float32)
    fsy = np.empty((MAX_FACES, 4), dtype=np.float32)
    fpw = np.empty((MAX_FACES, 4), dtype=np.float32)
    fau = np.empty((MAX_FACES, 4), dtype=np.float32)
    fav = np.empty((MAX_FACES, 4), dtype=np.float32)
    fbr = np.empty(MAX_FACES,      dtype=np.float32)
    fdp = np.empty(MAX_FACES,      dtype=np.float32)

    # Solid-colour fallback faces collected separately (unmapped textures, rare)
    fallback_faces = []

    n_faces = 0

    for gi, gj, gk in vis_idx:
        sid   = int(b[gi, gj, gk])
        state = bst.get(sid, {})
        aabbs = state.get("aabbs") or [[0, 0, 0, 1, 1, 1]]
        entry = tex_map[sid] if (tex_map and sid < len(tex_map)) else None

        # Block origin relative to camera (world space)
        bx = px_int - 15 + gi - cam_x
        by = py_int - 15 + gj - cam_y
        bz = pz_int - 15 + gk - cam_z
        block_orig = np.array([bx, by, bz], dtype=np.float64)

        for aabb in aabbs:
            for (nx, ny, nz), corners in _aabb_faces(aabb):
                # Back-face cull: face centre must face camera
                cs  = np.array(corners, dtype=np.float64)  # (4,3) local AABB coords
                fc  = block_orig + cs.mean(axis=0)
                fn  = np.array([nx, ny, nz], dtype=np.float64)
                if np.dot(fn, -fc) <= 0:
                    continue

                # Project 4 corners to screen
                verts = block_orig + cs          # (4,3) world-relative
                vd    = verts @ fwd              # (4,) view depths
                if np.any(vd < NEAR):
                    continue

                inv_vd = 1.0 / vd               # (4,) perspective weights
                sx_ = -( verts @ right) * inv_vd * half_w / tan_hh + half_w
                sy_ = -( verts @ up)    * inv_vd * half_h / tan_hv + half_h

                # Lighting: sample adjacent cell
                if fullbright:
                    br_ = min(1.0, _RENDER_BRIGHTNESS * float(_FACE_BRIGHTNESS[(nx, ny, nz)]))
                else:
                    li = gi + nx;  lj = gj + ny;  lk = gk + nz
                    if 0 <= li < 32 and 0 <= lj < 32 and 0 <= lk < 32:
                        lv = int(lightmap[li, lj, lk])
                    else:
                        lv = int(lightmap[gi, gj, gk])
                    br_ = min(1.0, _RENDER_BRIGHTNESS * float(_FACE_BRIGHTNESS[(nx, ny, nz)]) * beta17_brightness(lv / 15.0))

                # Atlas UV
                face_dir    = _NORMAL_TO_DIR[(nx, ny, nz)]
                tex_name    = (entry or {}).get(face_dir)
                tinted      = bool((entry or {}).get(face_dir + '_t'))
                uv_rot      = int((entry or {}).get(face_dir + '_r', 0))
                uv_rect_raw = (entry or {}).get(face_dir + '_uv')
                uv_rect     = tuple(uv_rect_raw) if uv_rect_raw else None

                if tex_name:
                    au_, av_ = _face_atlas_uvs(tex_name, tinted, uv_rot, uv_rect)
                else:
                    au_ = av_ = None

                if au_ is not None:
                    if n_faces < MAX_FACES:
                        fsx[n_faces] = sx_.astype(np.float32)
                        fsy[n_faces] = sy_.astype(np.float32)
                        fpw[n_faces] = inv_vd.astype(np.float32)
                        fau[n_faces] = au_
                        fav[n_faces] = av_
                        fbr[n_faces] = np.float32(br_)
                        fdp[n_faces] = np.float32(float(fc @ fwd))
                        n_faces += 1
                else:
                    # Rare fallback: solid colour via PIL
                    fallback_faces.append((float(fc @ fwd), list(zip(sx_.tolist(), sy_.tolist())), sid, br_))

    img = np.full((out_h, out_w, 3), 20, dtype=np.uint8)

    if n_faces > 0:
        order = np.argsort(-fdp[:n_faces]).astype(np.int32)
        _rasterize_all_nb(img,
                          fsx[:n_faces], fsy[:n_faces], fpw[:n_faces],
                          fau[:n_faces], fav[:n_faces], fbr[:n_faces],
                          order, n_faces, atlas)

    # Solid-colour fallbacks (unmapped states)
    if fallback_faces:
        from PIL import ImageDraw as _ID
        fallback_faces.sort(key=lambda x: -x[0])
        tmp = Image.fromarray(img)
        drw = _ID.Draw(tmp)
        for _, pts, sid, br_ in fallback_faces:
            base = _state_color(sid)
            col  = tuple(int(min(255, c * br_)) for c in base)
            drw.polygon(pts, fill=col)
        img = np.array(tmp)

    return img


# ── Side-by-side compositor ───────────────────────────────────────────────────

def make_side_by_side(
    video_frame: Optional[np.ndarray],
    voxel_frame: np.ndarray,
    label: str = "",
) -> np.ndarray:
    h, w = IMG_H, IMG_W
    if video_frame is None:
        left = np.zeros((h, w, 3), dtype=np.uint8)
        left[h // 2 - 10 : h // 2 + 10, :] = 60
    else:
        if video_frame.shape[:2] != (h, w):
            video_frame = np.array(Image.fromarray(video_frame).resize((w, h)))
        left = video_frame
    if voxel_frame.shape[:2] != (h, w):
        voxel_frame = np.array(Image.fromarray(voxel_frame).resize((w, h)))
    combined = np.concatenate([left, voxel_frame], axis=1)
    combined[:4, :] = (255, 200, 0)
    return combined


def save_render(out_path: Path, img: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(out_path))
    log.info("Saved render: %s", out_path)
