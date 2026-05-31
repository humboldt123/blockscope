"""
renderer.py — GPU visibility computation using ModernGL offscreen rendering.

Public API:
  Renderer(cache_dir, version)
  .compute_visibility(world_u16, pose, player_block_pos) → uint8 (32,32,32)
  .render_rgb(world_u16, pose, player_block_pos, mask=None, orbit_cam=None) → uint8 (360,640,3)
  .close()

--- Attributions ---
Camera math (_camera_vectors, view matrix construction):
    Adapted from furnace/pipeline/src/python/io_helpers.py
    Originally written for the V1 furnace pipeline; yaw/pitch/fov conventions
    confirmed correct against Minecraft's GameRenderer.getCamera().

FBO readback workaround (color_tex.read() instead of fbo.read()):
    fbo.read() silently reads framebuffer 0 on macOS/M2 with CGL backend.
    Using the color attachment texture's read() method bypasses this.

ID-buffer visibility technique:
    Standard G-buffer / deferred rendering pattern; each fragment writes
    its block (i,j,k) index as an RGB color, decoded on readback.
"""

import json
import math
import platform
from pathlib import Path
from typing import Optional

import moderngl
import numpy as np
from PIL import Image

from .baker import CULL_DIRS, CULL_DIR_KEYS

IMG_W, IMG_H = 640, 360

# World-space offset of each cull direction's neighbor
CULL_NEIGHBOR = {
    "north": (0, 0, -1),
    "south": (0, 0,  1),
    "east":  (1, 0,  0),
    "west":  (-1, 0, 0),
    "up":    (0,  1,  0),
    "down":  (0, -1,  0),
}

_VERT_SRC = """\
#version 330 core

in vec3 a_pos;
in vec2 a_uv;
in vec3 a_id;

uniform mat4 u_mvp;

out vec2 v_uv;
out vec3 v_id;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_uv = a_uv;
    v_id = a_id;
}
"""

_ID_FRAG_SRC = """\
#version 330 core

uniform sampler2D u_atlas;

in vec2 v_uv;
in vec3 v_id;

out vec4 frag_color;

void main() {
    vec4 t = texture(u_atlas, v_uv);
    if (t.a < 0.5) discard;
    frag_color = vec4(v_id, 1.0);
}
"""

_RGB_FRAG_SRC = """\
#version 330 core

uniform sampler2D u_atlas;
uniform float u_alpha;

in vec2 v_uv;
in vec3 v_id;

out vec4 frag_color;

void main() {
    vec4 t = texture(u_atlas, v_uv);
    if (t.a < 0.1) discard;
    frag_color = vec4(t.rgb, u_alpha);
}
"""


class Renderer:
    def __init__(self, cache_dir: Path, version: str = "1.19.4"):
        self._version = version
        self._ver_dir = Path(cache_dir) / version
        self._load_cache()
        self._init_gl()

    # ── Cache loading ──────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        v = self._version
        geo_path   = self._ver_dir / f"geometry_{v}.npz"
        atlas_path = self._ver_dir / f"atlas_{v}.png"
        meta_path  = self._ver_dir / f"meta_{v}.json"

        data = np.load(str(geo_path))
        self._transparency: np.ndarray = data["transparency"]   # uint8 (N,)
        self._fluid_type:   np.ndarray = data.get("fluid_type",
            np.zeros(len(self._transparency), dtype=np.uint8))
        self._fluid_level:  np.ndarray = data.get("fluid_level",
            np.zeros(len(self._transparency), dtype=np.uint8))
        self._verts: dict[str, np.ndarray] = {}
        self._offsets: dict[str, np.ndarray] = {}
        for key in CULL_DIR_KEYS:
            self._verts[key] = data[f"verts_{key}"]
            self._offsets[key] = data[f"offsets_{key}"]

        atlas_img = Image.open(str(atlas_path)).convert("RGBA")
        self._atlas_rgba = np.array(atlas_img, dtype=np.uint8)
        self._atlas_size = (atlas_img.width, atlas_img.height)

        # Fluid UV rects from meta
        self._fluid_uvs: dict[str, tuple] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            for name, rect in meta.get("fluid_uvs", {}).items():
                self._fluid_uvs[name] = tuple(rect)

    # ── GL initialisation ──────────────────────────────────────────────────────

    def _init_gl(self) -> None:
        if platform.system() == "Linux":
            try:
                self._ctx = moderngl.create_standalone_context(backend="egl")
            except Exception:
                self._ctx = moderngl.create_standalone_context()
        else:
            self._ctx = moderngl.create_standalone_context()

        ctx = self._ctx
        self._id_prog = ctx.program(
            vertex_shader=_VERT_SRC,
            fragment_shader=_ID_FRAG_SRC,
        )
        self._rgb_prog = ctx.program(
            vertex_shader=_VERT_SRC,
            fragment_shader=_RGB_FRAG_SRC,
        )

        aw, ah = self._atlas_size
        self._atlas_tex = ctx.texture((aw, ah), 4, data=self._atlas_rgba.tobytes())
        self._atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._atlas_tex.use(location=0)
        self._id_prog["u_atlas"] = 0
        self._rgb_prog["u_atlas"] = 0

        self._color_tex = ctx.texture((IMG_W, IMG_H), 4)
        self._depth_buf = ctx.depth_renderbuffer((IMG_W, IMG_H))
        self._fbo = ctx.framebuffer(
            color_attachments=[self._color_tex],
            depth_attachment=self._depth_buf,
        )

    # ── Camera helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _camera_vectors(yaw_deg: float, pitch_deg: float):
        y = math.radians(yaw_deg)
        p = math.radians(pitch_deg)
        fwd = (
            -math.sin(y) * math.cos(p),
            -math.sin(p),
             math.cos(y) * math.cos(p),
        )
        # Negated vs io_helpers: east(+X) is to player's LEFT when facing south,
        # so the camera-right vector points west when yaw=0.
        right = (-math.cos(y), 0.0, -math.sin(y))
        # up = cross(right, fwd)  — order matters with negated right
        up = (
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        )
        return fwd, right, up

    def _mvp_bytes(self, cx, cy, cz, yaw_deg, pitch_deg, fov_deg) -> bytes:
        fwd, right, up = self._camera_vectors(yaw_deg, pitch_deg)
        cam = np.array([cx, cy, cz], dtype=np.float64)
        r = np.array(right, dtype=np.float64)
        u = np.array(up, dtype=np.float64)
        f = np.array(fwd, dtype=np.float64)

        # View matrix (row-major), as specified:
        #   row0 = [right,  -dot(right, eye)]
        #   row1 = [up,     -dot(up,    eye)]
        #   row2 = [-fwd,    dot(fwd,   eye)]
        #   row3 = [0, 0, 0,              1]
        view_rm = np.array([
            [r[0],   r[1],   r[2],  -float(r @ cam)],
            [u[0],   u[1],   u[2],  -float(u @ cam)],
            [-f[0], -f[1],  -f[2],   float(f @ cam)],
            [0,      0,      0,      1              ],
        ], dtype=np.float32)

        # Standard OpenGL perspective (row-major)
        fov_y = math.radians(fov_deg)
        aspect = IMG_W / IMG_H
        near, far = 0.05, 500.0
        t = math.tan(fov_y / 2.0)
        proj_rm = np.array([
            [1.0 / (aspect * t), 0,         0,                         0                        ],
            [0,                  1.0 / t,   0,                         0                        ],
            [0,                  0,         -(far + near) / (far-near), -2*far*near / (far-near) ],
            [0,                  0,         -1,                         0                        ],
        ], dtype=np.float32)

        # MVP in row-major; transpose to column-major for GLSL
        mvp = (proj_rm @ view_rm).T
        return mvp.tobytes()

    def _mvp_from_pose(self, pose: dict) -> bytes:
        cam = pose.get("camera", pose)
        cx = float(cam["x"])
        cy = float(cam["y"])
        cz = float(cam["z"])
        yaw   = float(cam.get("yaw",   pose.get("yaw",   0.0)))
        pitch = float(cam.get("pitch", pose.get("pitch", 0.0)))
        fov   = float(pose.get("fov", 70.0))
        return self._mvp_bytes(cx, cy, cz, yaw, pitch, fov)

    def _mvp_from_orbit(self, orbit_cam: dict) -> bytes:
        return self._mvp_bytes(
            float(orbit_cam["x"]),
            float(orbit_cam["y"]),
            float(orbit_cam["z"]),
            float(orbit_cam["yaw"]),
            float(orbit_cam["pitch"]),
            float(orbit_cam.get("fov", 70.0)),
        )

    # ── Vertex buffer construction ─────────────────────────────────────────────

    @staticmethod
    def _neighbor_slices(d: int):
        """
        For offset d ∈ {-1, 0, 1}, return (src_start, src_end), (dst_start, dst_end)
        such that neighbor_arr[dst] = world[src] satisfies:
          neighbor_arr[i] = world[i + d]
        with boundary cells staying 0.
        """
        s0 = max(0, d)
        s1 = 32 + min(0, d)
        d0 = max(0, -d)
        d1 = 32 - max(0, d)
        return (s0, s1), (d0, d1)

    # ── Fluid geometry ─────────────────────────────────────────────────────────

    @staticmethod
    def _fluid_surface_height(level: int) -> float:
        """
        Top-face Y of a fluid block.
        Level 0 = source (full-ish height).  Levels 1-7 = flowing (decreasing).
        Level 8 = falling (treated same as source).
        """
        if level == 0 or level == 8:
            return 14.0 / 16.0      # source block height in Minecraft (14/16)
        return max(2.0 / 16.0, (8 - level) / 8.0 * (14.0 / 16.0))

    def _build_fluid_vertices(
        self,
        world_u16: np.ndarray,
        player_block_pos: tuple,
    ) -> np.ndarray:
        """
        Synthesise geometry for water and lava blocks.
        Only surface faces are emitted (faces adjacent to non-fluid blocks).
        Returns float32 (N, 8) in the same layout as the regular vertex buffer.
        """
        if not self._fluid_uvs:
            return np.zeros((0, 8), dtype=np.float32)

        px, py, pz = player_block_pos
        world_origin = np.array([px - 15, py - 15, pz - 15], dtype=np.float32)
        n_states = len(self._fluid_type)

        # UV rects for each face type
        water_top_uv  = self._fluid_uvs.get("water_still",  (0, 0, 1, 1))
        water_side_uv = self._fluid_uvs.get("water_flow",   water_top_uv)
        lava_top_uv   = self._fluid_uvs.get("lava_still",   (0, 0, 1, 1))
        lava_side_uv  = self._fluid_uvs.get("lava_flow",    lava_top_uv)

        def _is_fluid(sid: int) -> bool:
            return sid > 0 and sid < n_states and self._fluid_type[sid] > 0

        def _is_opaque(sid: int) -> bool:
            return sid > 0 and sid < n_states and self._transparency[sid] == 0

        rows: list[np.ndarray] = []

        def _quad(verts4: np.ndarray, uv: tuple, i: int, j: int, k: int) -> None:
            """Emit two triangles (6 vertices) for a quad."""
            u0, v0, u1, v1 = uv
            uvs4 = np.array([[u0,v0],[u1,v0],[u1,v1],[u0,v1]], dtype=np.float32)
            id_rgb = np.array([i / 255.0, j / 255.0, k / 255.0], dtype=np.float32)
            for tri_vi in [(0,1,2),(0,2,3)]:
                for vi in tri_vi:
                    row = np.empty(8, dtype=np.float32)
                    row[:3] = verts4[vi]
                    row[3:5] = uvs4[vi]
                    row[5:8] = id_rgb
                    rows.append(row)

        # ── Vectorised replacement for the Python triple loop ──────────────────
        sid_safe = np.minimum(world_u16.astype(np.int32), n_states - 1)
        cell_ftypes  = self._fluid_type[sid_safe]
        cell_flevels = self._fluid_level[sid_safe]
        fluid_mask   = (world_u16 > 0) & (cell_ftypes > 0)
        coords = np.argwhere(fluid_mask)   # (N, 3): i,j,k indices
        if len(coords) == 0:
            return np.zeros((0, 8), dtype=np.float32)

        ci, cj, ck = coords[:, 0], coords[:, 1], coords[:, 2]
        ftypes_n  = cell_ftypes[ci, cj, ck]
        flevels_n = cell_flevels[ci, cj, ck].astype(np.float32)

        # Surface heights (vectorised _fluid_surface_height)
        h_n = np.where(
            (flevels_n == 0) | (flevels_n == 8),
            np.float32(14.0 / 16.0),
            np.maximum(np.float32(2.0 / 16.0),
                       (8.0 - flevels_n) / 8.0 * np.float32(14.0 / 16.0)),
        )

        wx_n = world_origin[0] + ci.astype(np.float32)
        wy_n = world_origin[1] + cj.astype(np.float32)
        wz_n = world_origin[2] + ck.astype(np.float32)
        id_rgb_n = coords.astype(np.float32) / 255.0   # (N, 3)

        is_water = ftypes_n == 1

        wt  = np.array(water_top_uv,  dtype=np.float32)
        ws  = np.array(water_side_uv, dtype=np.float32)
        lt  = np.array(lava_top_uv,   dtype=np.float32)
        ls  = np.array(lava_side_uv,  dtype=np.float32)

        def _emit_faces_batch(
            corners_n4: np.ndarray,   # (K, 4, 3) world-space corners
            uv_n4:      np.ndarray,   # (K, 4, 2) atlas UVs
            id_rgb:     np.ndarray,   # (K, 3)
        ) -> np.ndarray:
            """Convert K quads → 6K vertex rows of shape (6K, 8)."""
            # Two triangles per quad: indices (0,1,2) and (0,2,3)
            tri = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
            pos  = corners_n4[:, tri]               # (K, 6, 3)
            uv   = uv_n4[:, tri]                    # (K, 6, 2)
            ids  = np.repeat(id_rgb[:, None, :], 6, axis=1)  # (K, 6, 3)
            return np.concatenate([pos, uv, ids], axis=2).reshape(-1, 8).astype(np.float32)

        def _uv4(uv_rect: np.ndarray) -> np.ndarray:
            """(4,) [u0,v0,u1,v1] → (4,2) corner UVs."""
            u0, v0, u1, v1 = uv_rect
            return np.array([[u0,v0],[u1,v0],[u1,v1],[u0,v1]], dtype=np.float32)

        # ── Top faces ──────────────────────────────────────────────────────────
        above_j   = np.minimum(cj + 1, 31)
        above_sid = world_u16[ci, above_j, ck]
        above_ft  = self._fluid_type[np.minimum(above_sid.astype(np.int32), n_states - 1)]
        needs_top = (above_ft == 0) | (cj == 31)

        if needs_top.any():
            idx   = np.where(needs_top)[0]
            h_i   = h_n[idx];  wx_i = wx_n[idx];  wy_i = wy_n[idx];  wz_i = wz_n[idx]
            top_y = wy_i + h_i
            c4 = np.stack([
                np.stack([wx_i,   top_y, wz_i  ], axis=1),
                np.stack([wx_i+1, top_y, wz_i  ], axis=1),
                np.stack([wx_i+1, top_y, wz_i+1], axis=1),
                np.stack([wx_i,   top_y, wz_i+1], axis=1),
            ], axis=1)
            uv_w = _uv4(wt);  uv_l = _uv4(lt)
            uv4 = np.where(is_water[idx, None, None], uv_w[None], uv_l[None])
            rows.append(_emit_faces_batch(c4, uv4, id_rgb_n[idx]))

        # ── Side faces (4 horizontal directions) ───────────────────────────────
        for (di, dk), (cx0, cx1, cz0, cz1) in [
            # (di, dk): offset.  Corners: (dx, [wy+h, wy, wy, wy+h]) × z-extent
            ((+1,  0), (1, 1, 0, 1)),
            ((-1,  0), (0, 0, 1, 0)),
            (( 0, +1), (1, 0, 0, 1)),
            (( 0, -1), (0, 1, 1, 0)),
        ]:
            ni = ci + di;  nk = ck + dk
            in_bounds = (0 <= ni) & (ni < 32) & (0 <= nk) & (nk < 32)
            ni_c = np.clip(ni, 0, 31);  nk_c = np.clip(nk, 0, 31)
            nbr_sid  = np.where(in_bounds, world_u16[ni_c, cj, nk_c], np.uint16(0))
            nbr_safe = np.minimum(nbr_sid.astype(np.int32), n_states - 1)
            nbr_ft   = self._fluid_type[nbr_safe]
            nbr_op   = (nbr_sid > 0) & (self._transparency[nbr_safe] == 0)
            need     = in_bounds & (nbr_ft == 0) & ~nbr_op

            if not need.any():
                continue

            idx   = np.where(need)[0]
            h_i   = h_n[idx];  wx_i = wx_n[idx];  wy_i = wy_n[idx];  wz_i = wz_n[idx]
            top_y = wy_i + h_i

            # Four corners: encoded as (wx+cx, [top_y, wy_i, wy_i, top_y], wz+cz)
            c4 = np.stack([
                np.stack([wx_i + cx0, top_y, wz_i + cz0], axis=1),
                np.stack([wx_i + cx0, wy_i,  wz_i + cz0], axis=1),
                np.stack([wx_i + cx1, wy_i,  wz_i + cz1], axis=1),
                np.stack([wx_i + cx1, top_y, wz_i + cz1], axis=1),
            ], axis=1)
            uv_w = _uv4(ws);  uv_l = _uv4(ls)
            uv4 = np.where(is_water[idx, None, None], uv_w[None], uv_l[None])
            rows.append(_emit_faces_batch(c4, uv4, id_rgb_n[idx]))

        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 8), dtype=np.float32)

    def _build_vertex_buffers(
        self,
        world_u16: np.ndarray,
        player_block_pos: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build opaque and translucent vertex arrays for one tick.
        Returns (opaque float32 (N,8), trans float32 (M,8)).
        Each row: (x, y, z, atlas_u, atlas_v, cell_i/255, cell_j/255, cell_k/255).
        """
        px, py, pz = player_block_pos
        # World-space origin of cell (0,0,0) in the 32³ window
        world_origin = np.array([px - 15, py - 15, pz - 15], dtype=np.float32)

        n_states = len(self._transparency)
        opaque_list: list[np.ndarray] = []
        trans_list:  list[np.ndarray] = []

        for cull_dir, key in zip(CULL_DIRS, CULL_DIR_KEYS):
            # ── Determine which cells have a face in this cull direction ──────
            if cull_dir is None:
                valid = world_u16 > 0
            else:
                di, dj, dk = CULL_NEIGHBOR[cull_dir]
                neighbor_sid = np.zeros((32, 32, 32), dtype=np.uint16)

                (si0, si1), (oi0, oi1) = self._neighbor_slices(di)
                (sj0, sj1), (oj0, oj1) = self._neighbor_slices(dj)
                (sk0, sk1), (ok0, ok1) = self._neighbor_slices(dk)
                neighbor_sid[oi0:oi1, oj0:oj1, ok0:ok1] = (
                    world_u16[si0:si1, sj0:sj1, sk0:sk1]
                )

                # Cull only when neighbor is FullOpaque (== 0) and exists
                nsid_safe = np.minimum(neighbor_sid, n_states - 1)
                neighbor_opaque = (self._transparency[nsid_safe] == 0) & (neighbor_sid > 0)
                valid = (world_u16 > 0) & ~neighbor_opaque

            cells = np.argwhere(valid)   # (M, 3) with i,j,k
            if len(cells) == 0:
                continue

            cell_sids = world_u16[cells[:, 0], cells[:, 1], cells[:, 2]].astype(np.int32)

            for sid in np.unique(cell_sids):
                sid = int(sid)
                if sid == 0 or sid >= n_states:
                    continue

                s = int(self._offsets[key][sid])
                e = int(self._offsets[key][sid + 1])
                if s == e:
                    continue

                template = self._verts[key][s:e]  # (V, 5): x,y,z,au,av
                V = len(template)

                these = cells[cell_sids == sid]   # (K, 3)
                K = len(these)

                # World-space origins for these K cells
                origins = these.astype(np.float32) + world_origin  # (K, 3)

                # Build (K*V, 8) vertex array vectorised
                # Positions: tile template V times, add per-cell origins
                tmpl_pos = np.tile(template[:, :3], (K, 1))        # (K*V, 3)
                tmpl_uv  = np.tile(template[:, 3:5], (K, 1))       # (K*V, 2)
                origin_rep = np.repeat(origins, V, axis=0)          # (K*V, 3)
                tmpl_pos += origin_rep

                # ID encoding: cell index / 255
                id_vals = np.repeat(these / 255.0, V, axis=0).astype(np.float32)  # (K*V, 3)

                flat = np.empty((K * V, 8), dtype=np.float32)
                flat[:, :3]  = tmpl_pos
                flat[:, 3:5] = tmpl_uv
                flat[:, 5:8] = id_vals

                # Only fluids (water) go to the translucent pass (0.45 alpha).
                # Leaves/glass are FullTranslucent but render at full opacity
                # with alpha-cutout — opaque pass is correct for them.
                if int(self._transparency[sid]) == 1 and int(self._fluid_type[sid]) > 0:
                    trans_list.append(flat)
                else:
                    opaque_list.append(flat)

        # Fluid geometry (water = translucent pass, lava = opaque pass)
        fluid_verts = self._build_fluid_vertices(world_u16, player_block_pos)
        if len(fluid_verts) > 0:
            # Separate water (fluid_type=1) from lava (fluid_type=2) by checking
            # the cell ID embedded in columns 5-7: look up fluid_type per cell
            n_states = len(self._fluid_type)
            cell_i = np.round(fluid_verts[:, 5] * 255).astype(np.int32)
            cell_j = np.round(fluid_verts[:, 6] * 255).astype(np.int32)
            cell_k = np.round(fluid_verts[:, 7] * 255).astype(np.int32)
            # Clamp to valid range
            ci = np.clip(cell_i, 0, 31)
            cj = np.clip(cell_j, 0, 31)
            ck = np.clip(cell_k, 0, 31)
            sids = world_u16[ci, cj, ck].astype(np.int32)
            sids_safe = np.minimum(sids, n_states - 1)
            is_water = self._fluid_type[sids_safe] == 1
            if is_water.any():
                trans_list.append(fluid_verts[is_water])
            if (~is_water).any():
                opaque_list.append(fluid_verts[~is_water])

        opaque = (np.concatenate(opaque_list, axis=0) if opaque_list
                  else np.zeros((0, 8), dtype=np.float32))
        trans  = (np.concatenate(trans_list,  axis=0) if trans_list
                  else np.zeros((0, 8), dtype=np.float32))
        return opaque, trans

    # ── Pixel decode ──────────────────────────────────────────────────────────

    @staticmethod
    def _decode_visibility(pixels: np.ndarray) -> np.ndarray:
        """Extract unique (i, j, k) cell indices from RGBA readback where alpha > 0.
        Returns int32 (K, 3) array, empty shape (0, 3) if nothing visible."""
        mask = pixels[:, :, 3] > 0
        if not mask.any():
            return np.empty((0, 3), dtype=np.int32)
        ys, xs = np.where(mask)
        rgb = pixels[ys, xs, :3].astype(np.int32)   # (M, 3)
        return np.unique(rgb, axis=0)                 # (K, 3)

    # ── Render passes helper ──────────────────────────────────────────────────

    def _render_passes(
        self,
        prog,
        mvp_bytes: bytes,
        opaque_v: np.ndarray,
        trans_v: np.ndarray,
        clear_rgba=(0.0, 0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Run opaque + translucent render passes into self._fbo.
        Returns the RGBA readback after both passes (H, W, 4) uint8.
        """
        ctx = self._ctx
        self._atlas_tex.use(location=0)
        self._fbo.use()
        ctx.viewport = (0, 0, IMG_W, IMG_H)
        self._fbo.clear(*clear_rgba, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        prog["u_mvp"].write(mvp_bytes)

        is_rgb = "u_alpha" in prog
        if is_rgb:
            prog["u_alpha"].value = 1.0

        # Vertex format: 3f pos + 2f uv + 3f cell_id.
        # The RGB program optimises out a_id (unused in fragment shader), so we
        # pad those 12 bytes instead of trying to bind a missing attribute.
        has_id = "a_id" in prog
        vfmt   = "3f 2f 3f" if has_id else "3f 2f 12x"
        vattrs = ("a_pos", "a_uv", "a_id") if has_id else ("a_pos", "a_uv")

        def _draw(data, depth_func, depth_write):
            vbo = ctx.buffer(data.tobytes(), dynamic=True)
            vao = ctx.vertex_array(prog, [(vbo, vfmt, *vattrs)])
            ctx.depth_func = depth_func
            ctx.depth_write_mask = depth_write
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()

        # Pass 1 — opaque/cutout (depth write ON, depth func LESS)
        if len(opaque_v) > 0:
            _draw(opaque_v, "<", True)

        # Readback after opaque pass.
        # fbo.read() is broken on macOS/M2 (reads from default FB 0).
        # Read directly from the color texture attachment instead.
        raw1 = self._color_tex.read()

        # Pass 2 — translucent (depth test ON/LEQUAL, depth write OFF, blending ON for RGB)
        if len(trans_v) > 0:
            if is_rgb:
                prog["u_alpha"].value = 0.45
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            _draw(trans_v, "<=", False)
            if is_rgb:
                ctx.disable(moderngl.BLEND)
            ctx.depth_write_mask = True

        raw2 = self._color_tex.read()
        return raw1, raw2

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_visibility(
        self,
        world_u16: np.ndarray,
        pose: dict,
        player_block_pos: tuple,
    ) -> np.ndarray:
        """
        Render the 32³ window and return uint8 (32,32,32) visibility mask.
        m[i,j,k] = 1 if cell (i,j,k) was visible in the camera frame.
        """
        opaque_v, trans_v = self._build_vertex_buffers(world_u16, player_block_pos)
        mvp = self._mvp_from_pose(pose)

        raw1, raw2 = self._render_passes(
            self._id_prog, mvp, opaque_v, trans_v,
            clear_rgba=(0.0, 0.0, 0.0, 0.0),
        )

        px1 = np.frombuffer(raw1, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
        px2 = np.frombuffer(raw2, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)

        v1 = self._decode_visibility(px1)   # (K1, 3) int32
        v2 = self._decode_visibility(px2)   # (K2, 3) int32

        m = np.zeros((32, 32, 32), dtype=np.uint8)
        parts = [a for a in (v1, v2) if len(a)]
        if parts:
            vis = np.unique(np.concatenate(parts, axis=0), axis=0)  # (K, 3)
            ok  = np.all((vis >= 0) & (vis < 32), axis=1)
            vis = vis[ok]
            if len(vis):
                m[vis[:, 0], vis[:, 1], vis[:, 2]] = 1
        return m

    def render_rgb(
        self,
        world_u16: np.ndarray,
        pose: dict,
        player_block_pos: tuple,
        mask: Optional[np.ndarray] = None,
        orbit_cam: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Render a textured RGB view of the world.
        mask=None renders all blocks; pass visibility mask to render only visible ones.
        orbit_cam overrides the camera (for orbit video).
        Returns uint8 (360, 640, 3).
        """
        if mask is not None:
            filtered = world_u16.copy()
            filtered[mask == 0] = 0
        else:
            filtered = world_u16

        opaque_v, trans_v = self._build_vertex_buffers(filtered, player_block_pos)
        mvp = (self._mvp_from_orbit(orbit_cam) if orbit_cam is not None
               else self._mvp_from_pose(pose))

        _, raw2 = self._render_passes(
            self._rgb_prog, mvp, opaque_v, trans_v,
            clear_rgba=(0.529, 0.808, 0.922, 1.0),  # Minecraft sky blue
        )

        # OpenGL framebuffer is bottom-to-top; flip to get normal image orientation
        img = np.frombuffer(raw2, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
        return np.flipud(img[:, :, :3])

    def close(self) -> None:
        for attr in ("_fbo", "_color_tex", "_depth_buf",
                     "_atlas_tex", "_id_prog", "_rgb_prog", "_ctx"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
