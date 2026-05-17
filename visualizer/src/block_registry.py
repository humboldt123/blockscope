"""
Parses Minecraft block model JSONs and builds a texture atlas from individual PNGs.
Produces:
  - A texture array (OpenGL sampler2DArray) where each layer is one 16x16 texture.
  - A mapping: minecraft block id string -> (top_tex, bottom_tex, north_tex, south_tex, west_tex, east_tex)
    where each tex is an index into the texture array.
  - Metadata about block render type (cube vs cross) and tinting.
"""

import json
import os

import numpy as np
import pygame as pg

from settings import ROOT_DIR

ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
MODELS_DIR = os.path.join(ASSETS_DIR, 'models', 'block')
TEXTURES_DIR = os.path.join(ASSETS_DIR, 'textures', 'block')
COLORMAP_DIR = os.path.join(ASSETS_DIR, 'textures', 'colormap')
BLOCKSTATES_DIR = os.path.join(ASSETS_DIR, 'blockstates')

TEX_SIZE = 16  # pixels per texture tile

# Face order we use everywhere: top, bottom, north, south, west, east
FACE_NAMES = ['up', 'down', 'north', 'south', 'west', 'east']

# Render types
RENDER_CUBE = 0
RENDER_CROSS = 1
RENDER_ELEMENTS = 2

# Tint types
TINT_NONE = 0
TINT_GRASS = 1    # uses grass colormap
TINT_FOLIAGE = 2  # uses foliage colormap

# Minecraft biome -> (temperature, downfall) for colormap lookup
# The colormap is indexed as: x = clamp(temp, 0, 1), y = clamp(downfall, 0, 1) * clamp(temp, 0, 1)
# Then pixel at (255 * (1-temp), 255 * (1 - downfall*temp))
BIOME_CLIMATE = {
    'minecraft:ocean':               (0.5,  0.5),
    'minecraft:plains':              (0.8,  0.4),
    'minecraft:desert':              (2.0,  0.0),
    'minecraft:mountains':           (0.2,  0.3),
    'minecraft:forest':              (0.7,  0.8),
    'minecraft:taiga':               (0.25, 0.8),
    'minecraft:swamp':               (0.8,  0.9),
    'minecraft:river':               (0.5,  0.5),
    'minecraft:frozen_ocean':        (0.0,  0.5),
    'minecraft:frozen_river':        (0.0,  0.5),
    'minecraft:snowy_tundra':        (0.0,  0.5),
    'minecraft:snowy_mountains':     (0.0,  0.5),
    'minecraft:mushroom_fields':     (0.9,  1.0),
    'minecraft:beach':               (0.8,  0.4),
    'minecraft:wooded_hills':        (0.7,  0.8),
    'minecraft:birch_forest':        (0.6,  0.6),
    'minecraft:dark_forest':         (0.7,  0.8),
    'minecraft:snowy_taiga':         (-0.5, 0.4),
    'minecraft:giant_tree_taiga':    (0.3,  0.8),
    'minecraft:savanna':             (2.0,  0.0),
    'minecraft:savanna_plateau':     (2.0,  0.0),
    'minecraft:badlands':            (2.0,  0.0),
    'minecraft:jungle':              (0.95, 0.9),
    'minecraft:bamboo_jungle':       (0.95, 0.9),
    'minecraft:warm_ocean':          (0.5,  0.5),
    'minecraft:lukewarm_ocean':      (0.5,  0.5),
    'minecraft:cold_ocean':          (0.5,  0.5),
    'minecraft:deep_ocean':          (0.5,  0.5),
    'minecraft:sunflower_plains':    (0.8,  0.4),
    'minecraft:flower_forest':       (0.7,  0.8),
}

# Blocks that use foliage colormap (not grass) for tinting
FOLIAGE_TINTED_BLOCKS = {
    'minecraft:oak_leaves', 'minecraft:jungle_leaves', 'minecraft:acacia_leaves',
    'minecraft:dark_oak_leaves',
}
# Blocks with hardcoded leaf colors (not from colormap)
HARDCODED_LEAF_COLORS = {
    'minecraft:birch_leaves': (128, 167, 85),
    'minecraft:spruce_leaves': (97, 153, 97),
}
# Hardcoded grass plant tint (short_grass, tall_grass)
HARDCODED_GRASS_PLANT_COLOR = (0x7C, 0xBD, 0x6B)
# Hardcoded water color
WATER_COLOR = (0x3F / 255.0, 0x76 / 255.0, 0xE4 / 255.0)


def _resolve_texture_ref(ref: str) -> str:
    """Strip 'minecraft:' prefix and 'block/' prefix to get bare texture name."""
    ref = ref.replace('minecraft:', '')
    if ref.startswith('block/'):
        ref = ref[6:]
    return ref


def _load_model_json(model_name: str, cache: dict) -> dict | None:
    """Load and cache a model JSON, resolving parent chain."""
    if model_name in cache:
        return cache[model_name]

    bare = model_name.replace('minecraft:', '')
    if bare.startswith('block/'):
        bare = bare[6:]

    path = os.path.join(MODELS_DIR, f'{bare}.json')
    if not os.path.exists(path):
        cache[model_name] = None
        return None

    with open(path) as f:
        data = json.load(f)

    # Track parent chain for render type detection
    parent_chain = []
    if 'parent' in data:
        parent_name = data['parent']
        parent_chain.append(parent_name)
        parent = _load_model_json(parent_name, cache)
        if parent:
            if 'textures' in parent:
                # Merge parent textures as base
                merged = dict(parent.get('textures', {}))
                merged.update(data.get('textures', {}))
                data['textures'] = merged
            if 'elements' not in data and 'elements' in parent:
                data['elements'] = parent['elements']
            parent_chain.extend(parent.get('_parent_chain', []))

    data['_parent_chain'] = parent_chain

    # Resolve texture variable references (#variable -> actual texture path)
    textures = data.get('textures', {})
    resolved = True
    iterations = 0
    while resolved and iterations < 10:
        resolved = False
        iterations += 1
        for key, val in list(textures.items()):
            if isinstance(val, str) and val.startswith('#'):
                ref_key = val[1:]
                if ref_key in textures and isinstance(textures[ref_key], str) and not textures[ref_key].startswith('#'):
                    textures[key] = textures[ref_key]
                    resolved = True

    cache[model_name] = data
    return data


def _is_cross_model(model_data: dict) -> bool:
    """Check if the model is a cross-shaped plant model."""
    chain = model_data.get('_parent_chain', [])
    for p in chain:
        bare = p.replace('minecraft:', '').replace('block/', '')
        if bare in ('cross', 'tinted_cross'):
            return True
    # Also check if the model name itself suggests cross
    textures = model_data.get('textures', {})
    if 'cross' in textures and len(textures) <= 2:  # cross + particle
        return True
    return False


def _has_tintindex(model_data: dict) -> bool:
    """Check if any face in the model has tintindex."""
    chain = model_data.get('_parent_chain', [])
    for p in chain:
        bare = p.replace('minecraft:', '').replace('block/', '')
        if bare == 'tinted_cross':
            return True
        if bare == 'leaves':
            return True
    elements = model_data.get('elements', [])
    for element in elements:
        faces = element.get('faces', {})
        for face_data in faces.values():
            if 'tintindex' in face_data:
                return True
    return False


def _get_face_textures(model_data: dict) -> dict:
    """From a resolved model, determine which texture each face uses."""
    textures = model_data.get('textures', {})
    face_map = {}

    if 'elements' in model_data:
        for element in model_data['elements']:
            faces = element.get('faces', {})
            for face_name in FACE_NAMES:
                if face_name in faces and face_name not in face_map:
                    tex_ref = faces[face_name].get('texture', '')
                    if tex_ref.startswith('#'):
                        ref_key = tex_ref[1:]
                        tex_ref = textures.get(ref_key, tex_ref)
                    if isinstance(tex_ref, str) and not tex_ref.startswith('#'):
                        face_map[face_name] = _resolve_texture_ref(tex_ref)

    if not face_map:
        if 'all' in textures:
            all_tex = _resolve_texture_ref(textures['all'])
            for face in FACE_NAMES:
                face_map[face] = all_tex

        elif 'cross' in textures:
            cross_tex = _resolve_texture_ref(textures['cross'])
            for face in FACE_NAMES:
                face_map[face] = cross_tex

        elif 'end' in textures and 'side' in textures:
            end_tex = _resolve_texture_ref(textures['end'])
            side_tex = _resolve_texture_ref(textures['side'])
            face_map['up'] = end_tex
            face_map['down'] = end_tex
            for f in ['north', 'south', 'west', 'east']:
                face_map[f] = side_tex

        elif 'top' in textures and 'side' in textures:
            top_tex = _resolve_texture_ref(textures['top'])
            side_tex = _resolve_texture_ref(textures['side'])
            bottom_tex = _resolve_texture_ref(textures.get('bottom', textures.get('top', '')))
            face_map['up'] = top_tex
            face_map['down'] = bottom_tex
            for f in ['north', 'south', 'west', 'east']:
                face_map[f] = side_tex

        elif 'top' in textures and 'front' in textures:
            top_tex = _resolve_texture_ref(textures['top'])
            front_tex = _resolve_texture_ref(textures['front'])
            side_tex = _resolve_texture_ref(textures.get('side', front_tex))
            face_map['up'] = top_tex
            face_map['down'] = _resolve_texture_ref(textures.get('bottom', top_tex))
            face_map['north'] = front_tex
            face_map['south'] = front_tex
            face_map['west'] = side_tex
            face_map['east'] = side_tex

    fallback = None
    if face_map:
        fallback = next(iter(face_map.values()))
    elif textures:
        for v in textures.values():
            if isinstance(v, str) and not v.startswith('#'):
                fallback = _resolve_texture_ref(v)
                break

    if fallback:
        for face in FACE_NAMES:
            if face not in face_map:
                face_map[face] = fallback

    return face_map


def _load_blockstate_json(block_id: str) -> dict | None:
    """Load and return the blockstate JSON for a block."""
    bare = block_id.replace('minecraft:', '')
    path = os.path.join(BLOCKSTATES_DIR, f'{bare}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _match_blockstate_variant(blockstate: dict, properties: dict) -> dict | None:
    """Match a set of properties to a blockstate variant. Returns model info dict."""
    variants = blockstate.get('variants', {})
    for variant_key, variant_data in variants.items():
        # Parse variant key: "facing=east,half=bottom,shape=straight"
        variant_props = {}
        for prop in variant_key.split(','):
            if '=' in prop:
                k, v = prop.split('=', 1)
                variant_props[k.strip()] = v.strip()
        # Check if all variant properties match
        if all(properties.get(k) == v for k, v in variant_props.items()):
            if isinstance(variant_data, list):
                return variant_data[0]
            return variant_data
    return None


_Y_FACE_ROT = {
    90:  {'north': 'east', 'east': 'south', 'south': 'west', 'west': 'north'},
    180: {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'},
    270: {'north': 'west', 'west': 'south', 'south': 'east', 'east': 'north'},
}
_X_FACE_ROT = {
    180: {'up': 'down', 'down': 'up', 'north': 'south', 'south': 'north'},
}


def _rotate_elements(elements: list, rot_x: int, rot_y: int) -> list:
    """Apply X then Y rotation to element geometry. Rotations in degrees (0/90/180/270)."""
    if rot_x == 0 and rot_y == 0:
        return elements

    result = elements

    # Apply X rotation first
    if rot_x == 180:
        face_map = _X_FACE_ROT[180]
        rotated = []
        for f, t, faces in result:
            x0, y0, z0 = f
            x1, y1, z1 = t
            nf = (x0, 1.0 - y1, 1.0 - z1)
            nt = (x1, 1.0 - y0, 1.0 - z0)
            new_faces = {face_map.get(fn, fn): fd for fn, fd in faces.items()}
            rotated.append((nf, nt, new_faces))
        result = rotated

    # Apply Y rotation
    if rot_y in (90, 180, 270):
        face_map = _Y_FACE_ROT[rot_y]
        rotated = []
        for f, t, faces in result:
            x0, y0, z0 = f
            x1, y1, z1 = t
            if rot_y == 90:
                nf = (1.0 - z1, y0, x0)
                nt = (1.0 - z0, y1, x1)
            elif rot_y == 180:
                nf = (1.0 - x1, y0, 1.0 - z1)
                nt = (1.0 - x0, y1, 1.0 - z0)
            else:  # 270
                nf = (z0, y0, 1.0 - x1)
                nt = (z1, y1, 1.0 - x0)
            new_faces = {face_map.get(fn, fn): fd for fn, fd in faces.items()}
            rotated.append((nf, nt, new_faces))
        result = rotated

    return result


def get_biome_color(biome: str, colormap_surf: pg.Surface) -> tuple[int, int, int]:
    """Sample the colormap for a given biome to get the tint color."""
    temp, downfall = BIOME_CLIMATE.get(biome, (0.8, 0.4))
    temp = max(0.0, min(1.0, temp))
    downfall = max(0.0, min(1.0, downfall)) * temp  # adjusted downfall

    # Colormap coordinates
    cx = int(255 * (1.0 - temp))
    cy = int(255 * (1.0 - downfall))
    cx = max(0, min(255, cx))
    cy = max(0, min(255, cy))

    color = colormap_surf.get_at((cx, cy))
    return (color.r, color.g, color.b)


class BlockRegistry:
    def __init__(self):
        self.tex_name_to_index: dict[str, int] = {}
        self.block_face_textures: dict[str, tuple] = {}
        self.block_render_type: dict[str, int] = {}   # RENDER_CUBE or RENDER_CROSS
        self.block_tint_type: dict[str, int] = {}     # TINT_NONE, TINT_GRASS, TINT_FOLIAGE
        self.block_tint_faces: dict[str, tuple] = {}   # 6-tuple of bool
        self.block_is_full: dict[str, bool] = {}       # whether block is a full opaque cube
        self.block_elements: dict[str, list] = {}      # element geometry for RENDER_ELEMENTS blocks
        self._tex_surfaces: list[pg.Surface] = []
        self._model_cache: dict = {}

        # Load colormaps
        self._grass_colormap = None
        self._foliage_colormap = None
        self._load_colormaps()

        # Always have index 0 = missing texture (magenta/black checkerboard)
        self._add_missing_texture()

    def _load_colormaps(self):
        grass_path = os.path.join(COLORMAP_DIR, 'grass.png')
        foliage_path = os.path.join(COLORMAP_DIR, 'foliage.png')
        if os.path.exists(grass_path):
            self._grass_colormap = pg.image.load(grass_path).convert()
        if os.path.exists(foliage_path):
            self._foliage_colormap = pg.image.load(foliage_path).convert()

    def get_tint_color(self, block_id: str, biome: str) -> tuple[float, float, float]:
        """Get the tint color for a block in a given biome. Returns (r, g, b) in 0-1 range."""
        # # Water tint disabled — using pre-tinted water texture
        # if block_id == 'minecraft:water':
        #     return WATER_COLOR

        if block_id in HARDCODED_LEAF_COLORS:
            r, g, b = HARDCODED_LEAF_COLORS[block_id]
            return (r / 255.0, g / 255.0, b / 255.0)

        # Hardcoded grass plant tint
        if block_id in ('minecraft:short_grass', 'minecraft:tall_grass', 'minecraft:grass'):
            # r, g, b = HARDCODED_GRASS_PLANT_COLOR
            # return (r / 255.0, g / 255.0, b / 255.0)
            r, g, b = get_biome_color(biome, self._grass_colormap)
            return (r / 255.0, g / 255.0, b / 255.0)

        # # lets try doing that with grass too
        # if block_id in ('minecraft:grass_block'):
        #     r, g, b = HARDCODED_GRASS_PLANT_COLOR
        #     return (r / 255.0, g / 255.0, b / 255.0)

        tint_type = self.block_tint_type.get(block_id, TINT_NONE)
        if tint_type == TINT_NONE:
            return (1.0, 1.0, 1.0)

        if tint_type == TINT_FOLIAGE and self._foliage_colormap:
            r, g, b = get_biome_color(biome, self._foliage_colormap)
            return (r / 255.0, g / 255.0, b / 255.0)

        # # Grass block biome tint disabled — using plain texture
        if tint_type == TINT_GRASS and self._grass_colormap:
            r, g, b = get_biome_color(biome, self._grass_colormap)
            return (r / 255.0, g / 255.0, b / 255.0)

        return (1.0, 1.0, 1.0)

    def _add_missing_texture(self):
        surf = pg.Surface((TEX_SIZE, TEX_SIZE), pg.SRCALPHA)
        for y in range(TEX_SIZE):
            for x in range(TEX_SIZE):
                if (x // 4 + y // 4) % 2 == 0:
                    surf.set_at((x, y), (255, 0, 255, 255))
                else:
                    surf.set_at((x, y), (0, 0, 0, 255))
        self.tex_name_to_index['__missing__'] = 0
        self._tex_surfaces.append(surf)

    def _get_or_load_texture(self, tex_name: str) -> int:
        if tex_name in self.tex_name_to_index:
            return self.tex_name_to_index[tex_name]

        png_path = os.path.join(TEXTURES_DIR, f'{tex_name}.png')
        if not os.path.exists(png_path):
            base = os.path.basename(tex_name)
            png_path = os.path.join(TEXTURES_DIR, f'{base}.png')
            if not os.path.exists(png_path):
                self.tex_name_to_index[tex_name] = 0
                return 0

        try:
            surf = pg.image.load(png_path).convert_alpha()
            if surf.get_height() > TEX_SIZE:
                frame = pg.Surface((TEX_SIZE, TEX_SIZE), pg.SRCALPHA)
                frame.blit(surf, (0, 0), (0, 0, TEX_SIZE, TEX_SIZE))
                surf = frame
            elif surf.get_width() != TEX_SIZE or surf.get_height() != TEX_SIZE:
                surf = pg.transform.scale(surf, (TEX_SIZE, TEX_SIZE))
        except Exception:
            self.tex_name_to_index[tex_name] = 0
            return 0

        idx = len(self._tex_surfaces)
        self.tex_name_to_index[tex_name] = idx
        self._tex_surfaces.append(surf)
        return idx

    def _get_uv_cropped_texture(self, tex_name: str, uv: list, rotation: int = 0) -> int:
        """Get a texture index, cropping to the UV region and optionally rotating.
        UV is [u1, v1, u2, v2] in 0-16 texel coordinates.
        rotation is 0/90/180/270 degrees clockwise (Minecraft UV rotation)."""
        u1, v1, u2, v2 = int(uv[0]), int(uv[1]), int(uv[2]), int(uv[3])
        # Normalize (handle mirrored UVs)
        if u1 > u2:
            u1, u2 = u2, u1
        if v1 > v2:
            v1, v2 = v2, v1

        # If it's the full texture with no rotation, no crop needed
        if u1 == 0 and v1 == 0 and u2 == 16 and v2 == 16 and rotation == 0:
            return self._get_or_load_texture(tex_name)

        crop_key = f"{tex_name}__uv{u1}_{v1}_{u2}_{v2}_r{rotation}"
        if crop_key in self.tex_name_to_index:
            return self.tex_name_to_index[crop_key]

        # Load the full texture first
        full_idx = self._get_or_load_texture(tex_name)
        if full_idx == 0:
            return 0

        full_surf = self._tex_surfaces[full_idx]
        w = max(1, u2 - u1)
        h = max(1, v2 - v1)

        # Crop the UV region
        cropped = pg.Surface((w, h), pg.SRCALPHA)
        cropped.blit(full_surf, (0, 0), (u1, v1, w, h))

        # Apply UV rotation (Minecraft CW → pygame CCW)
        if rotation == 90:
            cropped = pg.transform.rotate(cropped, -90)
        elif rotation == 180:
            cropped = pg.transform.rotate(cropped, 180)
        elif rotation == 270:
            cropped = pg.transform.rotate(cropped, -270)

        # Scale to 16x16
        sz = cropped.get_size()
        if sz[0] != TEX_SIZE or sz[1] != TEX_SIZE:
            cropped = pg.transform.scale(cropped, (TEX_SIZE, TEX_SIZE))

        idx = len(self._tex_surfaces)
        self.tex_name_to_index[crop_key] = idx
        self._tex_surfaces.append(cropped)
        return idx

    def _get_rotated_texture(self, tex_idx: int, rotation: int) -> int:
        """Create a rotated copy of an existing texture layer.
        rotation is degrees CW (90/180/270). Returns new texture index."""
        if rotation == 0 or tex_idx == 0:
            return tex_idx
        rot_key = f"__bsrot_{tex_idx}_r{rotation}"
        if rot_key in self.tex_name_to_index:
            return self.tex_name_to_index[rot_key]
        surf = self._tex_surfaces[tex_idx]
        # pygame rotate is CCW, Minecraft blockstate is CW
        rotated = pg.transform.rotate(surf, -rotation)
        if rotated.get_size() != (TEX_SIZE, TEX_SIZE):
            rotated = pg.transform.scale(rotated, (TEX_SIZE, TEX_SIZE))
        idx = len(self._tex_surfaces)
        self.tex_name_to_index[rot_key] = idx
        self._tex_surfaces.append(rotated)
        return idx

    def _extract_elements(self, model_data: dict) -> list | None:
        """Extract element geometry from model. Returns None if all elements are full cubes."""
        elements = model_data.get('elements', [])
        if not elements:
            return None

        # Check if ALL elements are full cubes (e.g. grass_block has 2 overlapping full cubes)
        all_full = all(
            e.get('from', [0, 0, 0]) == [0, 0, 0] and e.get('to', [16, 16, 16]) == [16, 16, 16]
            for e in elements
        )
        if all_full:
            return None

        textures = model_data.get('textures', {})
        result = []
        for element in elements:
            from_pos = element.get('from', [0, 0, 0])
            to_pos = element.get('to', [16, 16, 16])

            # Convert from 0-16 scale to 0-1 scale
            f = (from_pos[0] / 16.0, from_pos[1] / 16.0, from_pos[2] / 16.0)
            t = (to_pos[0] / 16.0, to_pos[1] / 16.0, to_pos[2] / 16.0)

            faces = {}
            for face_name, face_data in element.get('faces', {}).items():
                tex_ref = face_data.get('texture', '')
                if tex_ref.startswith('#'):
                    ref_key = tex_ref[1:]
                    tex_ref = textures.get(ref_key, tex_ref)

                if isinstance(tex_ref, str) and not tex_ref.startswith('#'):
                    tex_name = _resolve_texture_ref(tex_ref)
                    # Use UV-cropped texture if model specifies a sub-region or rotation
                    uv = face_data.get('uv')
                    uv_rotation = face_data.get('rotation', 0)
                    if (uv and uv != [0, 0, 16, 16]) or uv_rotation:
                        if not uv:
                            uv = [0, 0, 16, 16]
                        tex_idx = self._get_uv_cropped_texture(tex_name, uv, uv_rotation)
                    else:
                        tex_idx = self._get_or_load_texture(tex_name)
                else:
                    tex_idx = 0

                has_cullface = 'cullface' in face_data
                faces[face_name] = (tex_idx, has_cullface)

            result.append((f, t, faces))

        return result

    def register_block(self, block_id: str, biome: str = 'minecraft:plains'):
        if block_id in self.block_face_textures:
            return

        bare = block_id.replace('minecraft:', '')

        # Try direct model first, then common variants (_bottom, _floor0, etc.)
        model = _load_model_json(f'minecraft:block/{bare}', self._model_cache)
        if model is None:
            for suffix in ('_bottom', '_floor0', '_0'):
                model = _load_model_json(f'minecraft:block/{bare}{suffix}', self._model_cache)
                if model is not None:
                    break
        if model is None:
            tex_idx = self._get_or_load_texture(bare)
            self.block_face_textures[block_id] = (tex_idx,) * 6
            self.block_render_type[block_id] = RENDER_CUBE
            self.block_tint_type[block_id] = TINT_NONE
            self.block_tint_faces[block_id] = (False,) * 6
            self.block_is_full[block_id] = True
            return

        # Determine render type
        is_cross = _is_cross_model(model)

        # Check for element-based geometry (stairs, slabs, anvils, etc.)
        elem_data = None
        if not is_cross:
            elem_data = self._extract_elements(model)

        if elem_data is not None:
            self.block_render_type[block_id] = RENDER_ELEMENTS
            self.block_elements[block_id] = elem_data
        elif is_cross:
            self.block_render_type[block_id] = RENDER_CROSS
        else:
            self.block_render_type[block_id] = RENDER_CUBE

        # Determine if this is a full opaque cube
        is_full = not is_cross and elem_data is None
        if block_id == 'minecraft:water':
            is_full = False
        # Leaves have transparency
        chain = model.get('_parent_chain', [])
        for p in chain:
            p_bare = p.replace('minecraft:', '').replace('block/', '')
            if p_bare == 'leaves':
                is_full = False
                break
        self.block_is_full[block_id] = is_full

        # Determine tint type
        has_tint = _has_tintindex(model)
        if has_tint:
            if block_id in FOLIAGE_TINTED_BLOCKS:
                self.block_tint_type[block_id] = TINT_FOLIAGE
            else:
                self.block_tint_type[block_id] = TINT_GRASS
        else:
            self.block_tint_type[block_id] = TINT_NONE

        # # Water tint disabled — using pre-tinted water texture
        # if block_id == 'minecraft:water':
        #     self.block_tint_type[block_id] = TINT_NONE
        #     self.block_tint_faces[block_id] = (True,) * 6

        # Determine which faces get tinted
        if block_id == 'minecraft:grass_block':
            # Top gets runtime tint; sides have tint baked into overlay; bottom (dirt) untinted
            self.block_tint_faces[block_id] = (True, False, False, False, False, False)
        elif has_tint:
            self.block_tint_faces[block_id] = (True,) * 6
        else:
            self.block_tint_faces[block_id] = (False,) * 6

        face_map = _get_face_textures(model)
        if not face_map:
            tex_idx = self._get_or_load_texture(bare)
            self.block_face_textures[block_id] = (tex_idx,) * 6
            return

        face_indices = []
        for face_name in FACE_NAMES:
            tex_name = face_map.get(face_name)
            if tex_name:
                idx = self._get_or_load_texture(tex_name)
            else:
                idx = 0
            face_indices.append(idx)

        self.block_face_textures[block_id] = tuple(face_indices)

        # Grass block: bake tinted overlay onto side texture so dirt stays untinted
        if block_id == 'minecraft:grass_block':
            self._bake_grass_block_sides(block_id, biome)

    def _bake_grass_block_sides(self, block_id: str, biome: str):
        """Composite tinted grass_block_side_overlay onto grass_block_side."""
        overlay_path = os.path.join(TEXTURES_DIR, 'grass_block_side_overlay.png')
        if not os.path.exists(overlay_path):
            return

        base_idx = self._get_or_load_texture('grass_block_side')
        base_surf = self._tex_surfaces[base_idx]

        overlay_surf = pg.image.load(overlay_path).convert_alpha()
        if overlay_surf.get_size() != (TEX_SIZE, TEX_SIZE):
            overlay_surf = pg.transform.scale(overlay_surf, (TEX_SIZE, TEX_SIZE))

        # Tint the overlay with grass color
        r, g, b = self.get_tint_color(block_id, biome)
        tinted_overlay = overlay_surf.copy()
        for y in range(TEX_SIZE):
            for x in range(TEX_SIZE):
                pr, pg_, pb, pa = tinted_overlay.get_at((x, y))
                if pa > 0:
                    tinted_overlay.set_at((x, y), (int(pr * r), int(pg_ * g), int(pb * b), pa))

        # Composite: base side + tinted overlay on top
        baked = base_surf.copy()
        baked.blit(tinted_overlay, (0, 0))

        baked_key = f'__grass_block_side_baked_{biome}'
        if baked_key in self.tex_name_to_index:
            baked_idx = self.tex_name_to_index[baked_key]
            self._tex_surfaces[baked_idx] = baked
        else:
            baked_idx = len(self._tex_surfaces)
            self.tex_name_to_index[baked_key] = baked_idx
            self._tex_surfaces.append(baked)

        # Replace side face textures (north, south, west, east = indices 2,3,4,5)
        top, bottom = self.block_face_textures[block_id][:2]
        self.block_face_textures[block_id] = (top, bottom, baked_idx, baked_idx, baked_idx, baked_idx)

    def register_block_variant(self, block_id: str, properties_str: str,
                               biome: str = 'minecraft:plains') -> str:
        """Register a block variant with specific state properties.
        Returns the variant key (or plain block_id if no variant needed)."""
        # Ensure base block is registered first
        if block_id not in self.block_face_textures:
            self.register_block(block_id, biome)

        # Only create variants for element blocks
        if self.block_render_type.get(block_id) != RENDER_ELEMENTS:
            return block_id

        variant_key = f"{block_id}[{properties_str}]"
        if variant_key in self.block_face_textures:
            return variant_key  # Already registered

        # Parse properties
        props = {}
        for prop in properties_str.split(','):
            if '=' in prop:
                k, v = prop.split('=', 1)
                props[k.strip()] = v.strip()

        # Load blockstate JSON
        blockstate = _load_blockstate_json(block_id)
        if not blockstate:
            return block_id

        # Match variant to get model + rotation
        variant_info = _match_blockstate_variant(blockstate, props)
        if not variant_info:
            return block_id

        model_name = variant_info.get('model', '')
        rot_x = variant_info.get('x', 0)
        rot_y = variant_info.get('y', 0)

        # Load the variant's model
        model = _load_model_json(model_name, self._model_cache)
        if not model:
            return block_id

        elements = self._extract_elements(model)
        if not elements:
            return block_id

        # Apply rotation to geometry
        elements = _rotate_elements(elements, rot_x, rot_y)

        # Rotate top/bottom face textures to match blockstate Y rotation.
        # When Y rotation swaps X/Z axes, the texture must rotate too.
        if rot_y in (90, 180, 270):
            rotated = []
            for f, t, faces in elements:
                new_faces = {}
                for fn, (tidx, cull) in faces.items():
                    if fn in ('up', 'down'):
                        tidx = self._get_rotated_texture(tidx, rot_y)
                    new_faces[fn] = (tidx, cull)
                rotated.append((f, t, new_faces))
            elements = rotated

        # Register variant: copy base block metadata, override elements
        self.block_face_textures[variant_key] = self.block_face_textures[block_id]
        self.block_render_type[variant_key] = RENDER_ELEMENTS
        self.block_tint_type[variant_key] = self.block_tint_type.get(block_id, TINT_NONE)
        self.block_tint_faces[variant_key] = self.block_tint_faces.get(block_id, (False,) * 6)
        self.block_is_full[variant_key] = False
        self.block_elements[variant_key] = elements

        return variant_key

    def get_face_textures(self, block_id: str) -> tuple:
        if block_id not in self.block_face_textures:
            self.register_block(block_id)
        return self.block_face_textures.get(block_id, (0, 0, 0, 0, 0, 0))

    def build_texture_array(self) -> tuple:
        num_layers = len(self._tex_surfaces)
        data = bytearray()
        for surf in self._tex_surfaces:
            flipped = pg.transform.flip(surf, flip_x=True, flip_y=False)
            data.extend(pg.image.tostring(flipped, 'RGBA'))
        return num_layers, bytes(data)

    @property
    def num_textures(self):
        return len(self._tex_surfaces)
