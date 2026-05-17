"""
Builds chunk mesh vertex data from a sparse block dict.

Vertex format per vertex: 9 floats
  x, y, z          (float32) - world position
  tex_id            (float32, used as int in shader)
  packed_face_ao    (float32, used as int in shader)
                    = face_id * 16 + ao_id * 2 + flip_id
                    face_id 0-5 = normal shading, 6-11 = unshaded
  tint_r, tint_g, tint_b  (float32) - tint color (1,1,1 = no tint)
  alpha             (float32) - opacity (1.0 = opaque, <1 = transparent)
"""

import numpy as np

FLOATS_PER_VERTEX = 9
RENDER_ELEMENTS = 2

# Blocks that render without AO or face shading
_UNSHADED_BLOCKS = frozenset({'minecraft:water', 'minecraft:lava'})

# Map element face names to mesh builder face_id and neighbor offset for cullface
_ELEM_FACE_INFO = {
    'up':    (0, (0, 1, 0)),
    'down':  (1, (0, -1, 0)),
    'east':  (2, (1, 0, 0)),
    'west':  (3, (-1, 0, 0)),
    'north': (4, (0, 0, -1)),
    'south': (5, (0, 0, 1)),
}

# Map face name to tint_faces index (matches FACE_NAMES order in block_registry)
_FACE_NAME_TO_TINT_IDX = {'up': 0, 'down': 1, 'north': 2, 'south': 3, 'west': 4, 'east': 5}

# Face normals: (dx, dy, dz) for neighbor check
_FACE_NORMALS = np.array([
    [0, 1, 0],   # 0 top
    [0, -1, 0],  # 1 bottom
    [1, 0, 0],   # 2 right +x
    [-1, 0, 0],  # 3 left -x
    [0, 0, -1],  # 4 back -z
    [0, 0, 1],   # 5 front +z
], dtype='i4')

# Quad corner offsets per face (4 corners each)
_FACE_VERTS = np.array([
    [[0,1,0], [1,1,0], [1,1,1], [0,1,1]],   # top
    [[0,0,0], [1,0,0], [1,0,1], [0,0,1]],   # bottom
    [[1,0,0], [1,1,0], [1,1,1], [1,0,1]],   # right +x
    [[0,0,0], [0,1,0], [0,1,1], [0,0,1]],   # left -x
    [[0,0,0], [0,1,0], [1,1,0], [1,0,0]],   # back -z
    [[0,0,1], [0,1,1], [1,1,1], [1,0,1]],   # front +z
], dtype='f4')

# AO neighbor offsets: for each face type, 8 neighbors around the face plane
# Top/bottom (y-plane): neighbors in xz
_AO_Y = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
# Left/right (x-plane): neighbors in yz
_AO_X = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]
# Front/back (z-plane): neighbors in xy
_AO_Z = [(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]

# Winding orders per face_id, per flip_id
_WINDING = [
    [[0,3,2,0,2,1], [1,0,3,1,3,2]],  # top
    [[0,2,3,0,1,2], [1,3,0,1,2,3]],  # bottom
    [[0,1,2,0,2,3], [3,0,1,3,1,2]],  # right
    [[0,2,1,0,3,2], [3,1,0,3,2,1]],  # left
    [[0,1,2,0,2,3], [3,0,1,3,1,2]],  # back
    [[0,2,1,0,3,2], [3,1,0,3,2,1]],  # front
]

# Pre-compute cross geometry template (4 quads x 6 verts x 3 floats)
_d = 0.854
_e = 1.0 - _d
_CROSS_PLANES = [
    [(_e,0,_e), (_e,1,_e), (_d,1,_d), (_d,0,_d)],
    [(_d,0,_d), (_d,1,_d), (_e,1,_e), (_e,0,_e)],
    [(_d,0,_e), (_d,1,_e), (_e,1,_d), (_e,0,_d)],
    [(_e,0,_d), (_e,1,_d), (_d,1,_e), (_d,0,_e)],
]
# Build flat cross template: 24 vertices x 3 position floats
_CROSS_POS = np.empty((24, 3), dtype='f4')
_ci = 0
for _plane in _CROSS_PLANES:
    _v0, _v1, _v2, _v3 = _plane
    for _vi in [_v0, _v1, _v2, _v0, _v2, _v3]:
        _CROSS_POS[_ci] = _vi
        _ci += 1
_CROSS_PACKED = float(0 * 16 + 3 * 2 + 0)  # face_id=0, ao=3 (full bright), flip=0


def _compute_ao_corner(solid, bx, by, bz, face_id, corner):
    """Compute AO value for one corner of one face."""
    i = corner * 2
    if face_id <= 1:
        ny = by + _FACE_NORMALS[face_id][1]
        a = (bx + _AO_Y[i % 8][0], ny, bz + _AO_Y[i % 8][1]) in solid
        b = (bx + _AO_Y[(i+1) % 8][0], ny, bz + _AO_Y[(i+1) % 8][1]) in solid
        c = (bx + _AO_Y[(i+2) % 8][0], ny, bz + _AO_Y[(i+2) % 8][1]) in solid
    elif face_id <= 3:
        nx = bx + _FACE_NORMALS[face_id][0]
        a = (nx, by + _AO_X[i % 8][0], bz + _AO_X[i % 8][1]) in solid
        b = (nx, by + _AO_X[(i+1) % 8][0], bz + _AO_X[(i+1) % 8][1]) in solid
        c = (nx, by + _AO_X[(i+2) % 8][0], bz + _AO_X[(i+2) % 8][1]) in solid
    else:
        nz = bz + _FACE_NORMALS[face_id][2]
        a = (bx + _AO_Z[i % 8][0], by + _AO_Z[i % 8][1], nz) in solid
        b = (bx + _AO_Z[(i+1) % 8][0], by + _AO_Z[(i+1) % 8][1], nz) in solid
        c = (bx + _AO_Z[(i+2) % 8][0], by + _AO_Z[(i+2) % 8][1], nz) in solid
    return int(a) + int(b) + int(c)


def _element_face_verts(f, t, face_name):
    """Generate 4 corner vertices for a face of a sub-cube element.
    f = (x0, y0, z0), t = (x1, y1, z1) in 0-1 scale."""
    x0, y0, z0 = f
    x1, y1, z1 = t
    if face_name == 'up':
        return ((x0,y1,z0), (x1,y1,z0), (x1,y1,z1), (x0,y1,z1))
    elif face_name == 'down':
        return ((x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1))
    elif face_name == 'east':   # +x
        return ((x1,y0,z0), (x1,y1,z0), (x1,y1,z1), (x1,y0,z1))
    elif face_name == 'west':   # -x
        return ((x0,y0,z0), (x0,y1,z0), (x0,y1,z1), (x0,y0,z1))
    elif face_name == 'north':  # -z
        return ((x0,y0,z0), (x0,y1,z0), (x1,y1,z0), (x1,y0,z0))
    else:  # south, +z
        return ((x0,y0,z1), (x0,y1,z1), (x1,y1,z1), (x1,y0,z1))


def build_chunk_mesh(blocks, solid_occupied, block_face_tex_map, block_render_types,
                     block_tint_colors, block_tint_faces, block_elements=None,
                     block_alpha=None, liquid_occupied=None):
    """
    Build mesh for a set of blocks.

    Args:
        blocks: dict[(x,y,z)] -> block_id (just this chunk's blocks)
        solid_occupied: set of (x,y,z) for ALL full opaque blocks in the world
                        (used for face culling and AO across chunk boundaries)
        block_face_tex_map: dict[block_id] -> 6-tuple of tex indices
        block_render_types: dict[block_id] -> 0 (cube) or 1 (cross)
        block_tint_colors: dict[block_id] -> (r, g, b) floats 0-1
        block_tint_faces: dict[block_id] -> 6-tuple of bool per face
        block_elements: dict[block_id] -> list of (from, to, faces) for RENDER_ELEMENTS
        block_alpha: dict[block_id] -> float (opacity, default 1.0)
        liquid_occupied: set of (x,y,z) for liquid blocks (water/lava) — used to
                         cull internal faces between adjacent liquid blocks
    Returns:
        (opaque_data, transparent_data) — two numpy float32 arrays.
        Opaque should be rendered first, transparent second with depth write off.
    """
    _empty = np.array([], dtype='f4')
    if not blocks:
        return _empty, _empty

    if block_elements is None:
        block_elements = {}
    if block_alpha is None:
        block_alpha = {}
    if liquid_occupied is None:
        liquid_occupied = set()

    # Pre-allocate generous buffers (opaque and transparent)
    buf = np.empty(len(blocks) * 6 * 6 * FLOATS_PER_VERTEX, dtype='f4')
    idx = 0
    buf_t = np.empty(len(blocks) * 2 * 6 * FLOATS_PER_VERTEX, dtype='f4')
    idx_t = 0

    # Group blocks by render type
    cross_blocks = []
    cube_blocks = []
    cube_blocks_trans = []
    element_blocks = []
    for pos, bid in blocks.items():
        rt = block_render_types.get(bid, 0)
        if rt == 1:
            cross_blocks.append(pos)
        elif rt == RENDER_ELEMENTS:
            element_blocks.append((pos, bid))
        else:
            if block_alpha.get(bid, 1.0) < 1.0:
                cube_blocks_trans.append((pos, bid))
            else:
                cube_blocks.append((pos, bid))

    # --- Cross blocks (batch-friendly) ---
    if cross_blocks:
        # Group by block_id for fewer dict lookups
        cross_by_id = {}
        for pos in cross_blocks:
            bid = blocks[pos]
            if bid not in cross_by_id:
                cross_by_id[bid] = []
            cross_by_id[bid].append(pos)

        for bid, positions in cross_by_id.items():
            tex_id = float(block_face_tex_map.get(bid, (0,0,0,0,0,0))[0])
            tr, tg, tb = block_tint_colors.get(bid, (1.0, 1.0, 1.0))
            packed = _CROSS_PACKED
            alpha = block_alpha.get(bid, 1.0)

            for bx, by, bz in positions:
                # 24 vertices per cross block
                need = 24 * FLOATS_PER_VERTEX
                if idx + need > len(buf):
                    buf = np.resize(buf, max(len(buf) * 2, idx + need))

                for v in range(24):
                    base = idx + v * FLOATS_PER_VERTEX
                    buf[base]     = bx + _CROSS_POS[v, 0]
                    buf[base + 1] = by + _CROSS_POS[v, 1]
                    buf[base + 2] = bz + _CROSS_POS[v, 2]
                    buf[base + 3] = tex_id
                    buf[base + 4] = packed
                    buf[base + 5] = tr
                    buf[base + 6] = tg
                    buf[base + 7] = tb
                    buf[base + 8] = alpha
                idx += need

    # --- Cube blocks ---
    # Pre-fetch face data per unique block_id
    cube_data_cache = {}
    for _, bid in cube_blocks:
        if bid not in cube_data_cache:
            # Check if this block should be unshaded (water, lava)
            base_id = bid.split('[')[0] if '[' in bid else bid
            unshaded = base_id in _UNSHADED_BLOCKS
            cube_data_cache[bid] = (
                block_face_tex_map.get(bid, (0,0,0,0,0,0)),
                block_tint_colors.get(bid, (1.0, 1.0, 1.0)),
                block_tint_faces.get(bid, (False,)*6),
                unshaded,
                block_alpha.get(bid, 1.0),
            )

    so_contains = solid_occupied.__contains__  # micro-opt: avoid attribute lookup in loop
    lq_contains = liquid_occupied.__contains__

    for (bx, by, bz), bid in cube_blocks:
        face_textures, tint_color, tint_faces_mask, unshaded, alpha = cube_data_cache[bid]

        for face_id in range(6):
            dx, dy, dz = int(_FACE_NORMALS[face_id][0]), int(_FACE_NORMALS[face_id][1]), int(_FACE_NORMALS[face_id][2])
            nb = (bx + dx, by + dy, bz + dz)
            if so_contains(nb):
                continue
            # Liquids cull internal faces against other liquids
            if unshaded and lq_contains(nb):
                continue

            tex_id = face_textures[face_id]
            if tint_faces_mask[face_id]:
                tr, tg, tb = tint_color
            else:
                tr, tg, tb = 1.0, 1.0, 1.0

            if unshaded:
                # No AO, no face shading — use face_id + 6 for unshaded shading lookup
                packed_fid = face_id + 6
                packed = float(packed_fid * 16 + 3 * 2 + 0)  # ao=3 (full bright), flip=0
                c0 = (bx + _FACE_VERTS[face_id][0,0], by + _FACE_VERTS[face_id][0,1], bz + _FACE_VERTS[face_id][0,2],
                       tex_id, packed, tr, tg, tb, alpha)
                c1 = (bx + _FACE_VERTS[face_id][1,0], by + _FACE_VERTS[face_id][1,1], bz + _FACE_VERTS[face_id][1,2],
                       tex_id, packed, tr, tg, tb, alpha)
                c2 = (bx + _FACE_VERTS[face_id][2,0], by + _FACE_VERTS[face_id][2,1], bz + _FACE_VERTS[face_id][2,2],
                       tex_id, packed, tr, tg, tb, alpha)
                c3 = (bx + _FACE_VERTS[face_id][3,0], by + _FACE_VERTS[face_id][3,1], bz + _FACE_VERTS[face_id][3,2],
                       tex_id, packed, tr, tg, tb, alpha)
                corners = (c0, c1, c2, c3)
                order = _WINDING[face_id][0]
            else:
                # Compute AO for 4 corners
                ao0 = _compute_ao_corner(solid_occupied, bx, by, bz, face_id, 0)
                ao1 = _compute_ao_corner(solid_occupied, bx, by, bz, face_id, 1)
                ao2 = _compute_ao_corner(solid_occupied, bx, by, bz, face_id, 2)
                ao3 = _compute_ao_corner(solid_occupied, bx, by, bz, face_id, 3)

                flip_id = 1 if (ao1 + ao3 > ao0 + ao2) else 0
                ao_ids = (3 - ao0, 3 - ao1, 3 - ao2, 3 - ao3)

                # Build 4 corner vertex data
                verts = _FACE_VERTS[face_id]
                c0 = (bx + verts[0,0], by + verts[0,1], bz + verts[0,2], tex_id, face_id * 16 + ao_ids[0] * 2 + flip_id, tr, tg, tb, alpha)
                c1 = (bx + verts[1,0], by + verts[1,1], bz + verts[1,2], tex_id, face_id * 16 + ao_ids[1] * 2 + flip_id, tr, tg, tb, alpha)
                c2 = (bx + verts[2,0], by + verts[2,1], bz + verts[2,2], tex_id, face_id * 16 + ao_ids[2] * 2 + flip_id, tr, tg, tb, alpha)
                c3 = (bx + verts[3,0], by + verts[3,1], bz + verts[3,2], tex_id, face_id * 16 + ao_ids[3] * 2 + flip_id, tr, tg, tb, alpha)
                corners = (c0, c1, c2, c3)
                order = _WINDING[face_id][flip_id]

            need = 6 * FLOATS_PER_VERTEX
            if idx + need > len(buf):
                buf = np.resize(buf, max(len(buf) * 2, idx + need))

            for vi in order:
                buf[idx:idx + FLOATS_PER_VERTEX] = corners[vi]
                idx += FLOATS_PER_VERTEX

    # --- Transparent cube blocks (water, etc.) → into buf_t ---
    if cube_blocks_trans:
        trans_data_cache = {}
        for _, bid in cube_blocks_trans:
            if bid not in trans_data_cache:
                base_id = bid.split('[')[0] if '[' in bid else bid
                trans_data_cache[bid] = (
                    block_face_tex_map.get(bid, (0,0,0,0,0,0)),
                    block_tint_colors.get(bid, (1.0, 1.0, 1.0)),
                    block_tint_faces.get(bid, (False,)*6),
                    base_id in _UNSHADED_BLOCKS,
                    block_alpha.get(bid, 1.0),
                )

        for (bx, by, bz), bid in cube_blocks_trans:
            face_textures, tint_color, tint_faces_mask, unshaded, alpha = trans_data_cache[bid]

            for face_id in range(6):
                dx, dy, dz = int(_FACE_NORMALS[face_id][0]), int(_FACE_NORMALS[face_id][1]), int(_FACE_NORMALS[face_id][2])
                nb = (bx + dx, by + dy, bz + dz)
                if so_contains(nb):
                    continue
                if unshaded and lq_contains(nb):
                    continue

                tex_id = face_textures[face_id]
                if tint_faces_mask[face_id]:
                    tr, tg, tb = tint_color
                else:
                    tr, tg, tb = 1.0, 1.0, 1.0

                packed_fid = face_id + 6 if unshaded else face_id
                packed = float(packed_fid * 16 + 3 * 2 + 0)

                verts = _FACE_VERTS[face_id]
                c0 = (bx + verts[0,0], by + verts[0,1], bz + verts[0,2], tex_id, packed, tr, tg, tb, alpha)
                c1 = (bx + verts[1,0], by + verts[1,1], bz + verts[1,2], tex_id, packed, tr, tg, tb, alpha)
                c2 = (bx + verts[2,0], by + verts[2,1], bz + verts[2,2], tex_id, packed, tr, tg, tb, alpha)
                c3 = (bx + verts[3,0], by + verts[3,1], bz + verts[3,2], tex_id, packed, tr, tg, tb, alpha)
                corners = (c0, c1, c2, c3)
                order = _WINDING[face_id][0]

                need = 6 * FLOATS_PER_VERTEX
                if idx_t + need > len(buf_t):
                    buf_t = np.resize(buf_t, max(len(buf_t) * 2, idx_t + need))

                for vi in order:
                    buf_t[idx_t:idx_t + FLOATS_PER_VERTEX] = corners[vi]
                    idx_t += FLOATS_PER_VERTEX

    # --- Element blocks (stairs, slabs, anvils, etc.) ---
    if element_blocks:
        so_contains_e = solid_occupied.__contains__

        for (bx, by, bz), bid in element_blocks:
            elements = block_elements.get(bid)
            if not elements:
                continue

            tint_color = block_tint_colors.get(bid, (1.0, 1.0, 1.0))
            tint_faces_mask = block_tint_faces.get(bid, (False,) * 6)
            alpha = block_alpha.get(bid, 1.0)

            for f, t, faces in elements:
                for face_name, (tex_idx, has_cullface) in faces.items():
                    fi = _ELEM_FACE_INFO.get(face_name)
                    if fi is None:
                        continue
                    face_id, (dx, dy, dz) = fi

                    # Cullface: only cull if the model says so AND neighbor is solid
                    if has_cullface and so_contains_e((bx + dx, by + dy, bz + dz)):
                        continue

                    # Tint
                    tint_idx = _FACE_NAME_TO_TINT_IDX.get(face_name, 0)
                    if tint_faces_mask[tint_idx]:
                        tr, tg, tb = tint_color
                    else:
                        tr, tg, tb = 1.0, 1.0, 1.0

                    # Full brightness, no AO flip for elements
                    packed = float(face_id * 16 + 3 * 2 + 0)

                    verts = _element_face_verts(f, t, face_name)
                    c0 = (bx + verts[0][0], by + verts[0][1], bz + verts[0][2],
                           tex_idx, packed, tr, tg, tb, alpha)
                    c1 = (bx + verts[1][0], by + verts[1][1], bz + verts[1][2],
                           tex_idx, packed, tr, tg, tb, alpha)
                    c2 = (bx + verts[2][0], by + verts[2][1], bz + verts[2][2],
                           tex_idx, packed, tr, tg, tb, alpha)
                    c3 = (bx + verts[3][0], by + verts[3][1], bz + verts[3][2],
                           tex_idx, packed, tr, tg, tb, alpha)
                    corners = (c0, c1, c2, c3)

                    order = _WINDING[face_id][0]
                    need = 6 * FLOATS_PER_VERTEX
                    if idx + need > len(buf):
                        buf = np.resize(buf, max(len(buf) * 2, idx + need))

                    for vi in order:
                        buf[idx:idx + FLOATS_PER_VERTEX] = corners[vi]
                        idx += FLOATS_PER_VERTEX

    return buf[:idx], buf_t[:idx_t]
