"""
baker.py — one-time geometry cache builder for pipeline.

Extracts Minecraft JAR assets, loads Minecraft-Model-Reader, bakes
per-state-ID vertex geometry (with atlas UVs) and a texture atlas PNG.

Run: uv run python -m pipeline.baker [--jar path/to/jar] [--version 1.19.4]

--- Attributions ---
Block model loading:
    Minecraft-Model-Reader — vendored in pipeline/vendor/minecraft_model_reader/
    © Amulet MC / James Clare — https://github.com/Amulet-Team/Minecraft-Model-Reader (MIT)
    Bedrock support stripped; only the 19 Java-path files are kept.
    Used for: JavaResourcePackManager, BlockMesh geometry extraction.

amulet-nbt replacement:
    pipeline/vendor/amulet_nbt.py — 50-line shim
    Replaces the amulet-nbt PyPI package (© Amulet MC, MIT).
    Only TAG_String is needed at runtime; other tags are stub-only.

State ID → properties mapping:
    minecraft-data (pipeline/node_modules/minecraft-data)
    © PrismarineJS — https://github.com/PrismarineJS/minecraft-data (MIT)
    Used via: scripts/gen_state_properties.js

Texture atlas packing approach:
    Adapted from furnace/visualizer/src/block_registry.py
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
# JS stage and data/ live alongside us now

# Canonical cull direction ordering used everywhere in the pipeline
CULL_DIRS = [None, "north", "south", "east", "west", "up", "down"]
CULL_DIR_KEYS = ["none", "north", "south", "east", "west", "up", "down"]

AIR_NAMES = {"minecraft:air", "minecraft:void_air", "minecraft:cave_air"}


# ── JAR extraction ─────────────────────────────────────────────────────────────

def _extract_jar(jar_path: Path, extracted_dir: Path) -> None:
    if extracted_dir.exists() and (extracted_dir / "assets").exists():
        log.info("JAR already extracted at %s", extracted_dir)
        return

    extracted_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting JAR %s → %s", jar_path, extracted_dir)

    with zipfile.ZipFile(str(jar_path)) as zf:
        for name in zf.namelist():
            if name.startswith("assets/minecraft/"):
                zf.extract(name, str(extracted_dir))

    # The client JAR has no pack.mcmeta; JavaResourcePack requires it.
    meta = {"pack": {"pack_format": 12, "description": "Minecraft 1.19.4"}}
    (extracted_dir / "pack.mcmeta").write_text(json.dumps(meta))
    log.info("Extraction complete.")


# ── State properties generation ────────────────────────────────────────────────

def _gen_state_properties(cache_dir: Path, version: str) -> Path:
    out_path = cache_dir / f"state_properties_{version}.json"
    if out_path.exists():
        log.info("state_properties already at %s", out_path)
        return out_path

    script = _ROOT / "scripts" / "gen_state_properties.js"
    result = subprocess.run(
        ["node", str(script), version, str(out_path)],
        cwd=str(_ROOT),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gen_state_properties.js failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )
    log.info("Generated state_properties: %s", out_path)
    return out_path


# ── Minecraft-Model-Reader loader ─────────────────────────────────────────────

def _load_mmr(extracted_dir: Path, cache_dir: Path):
    vendor = _ROOT / "vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    # Must be set before importing the library
    os.environ["CACHE_DIR"] = str(cache_dir)

    from minecraft_model_reader.api.resource_pack.java import (
        JavaResourcePack, JavaResourcePackManager,
    )

    pack = JavaResourcePack(str(extracted_dir))
    if not pack.valid_pack:
        raise RuntimeError(f"JavaResourcePack not valid at {extracted_dir}. "
                           "Is pack.mcmeta present?")

    log.info("Loading JavaResourcePackManager (scanning %s) ...", extracted_dir)
    manager = JavaResourcePackManager(pack)
    log.info("Resource pack loaded.")
    return manager


# ── Biome tint (fixed plains biome defaults) ──────────────────────────────────

_GRASS_TINT   = np.array([146, 189,  89], dtype=np.float32) / 255.0
_FOLIAGE_TINT = np.array([119, 171,  47], dtype=np.float32) / 255.0
_BIRCH_TINT   = np.array([128, 167,  85], dtype=np.float32) / 255.0
_SPRUCE_TINT  = np.array([ 97, 153,  97], dtype=np.float32) / 255.0
# Overworld ocean/river water colour
_WATER_TINT   = np.array([ 63, 118, 228], dtype=np.float32) / 255.0

# Exact texture stems → tint (checked with frozensets for O(1) lookup)
_FOLIAGE_STEMS = frozenset({
    "oak_leaves", "jungle_leaves", "acacia_leaves", "dark_oak_leaves",
    "mangrove_leaves", "azalea_leaves", "flowering_azalea_leaves",
    "vine", "lily_pad", "mangrove_roots",
})
_BIRCH_STEMS   = frozenset({"birch_leaves"})
_SPRUCE_STEMS  = frozenset({"spruce_leaves"})
_GRASS_STEMS   = frozenset({
    "grass",                     # minecraft:grass plant (short_grass pre-1.20)
    "short_grass",               # renamed in 1.20+
    "grass_block_top",
    "grass_block_side_overlay",
    "fern",
    "large_fern_top", "large_fern_bottom",
    "tall_grass_top", "tall_grass_bottom",
    "seagrass",
    "tall_seagrass_top", "tall_seagrass_bottom",
})
_WATER_STEMS   = frozenset({"water_still", "water_flow", "water_overlay"})


def _texture_tint(path: str) -> np.ndarray | None:
    stem = Path(path).stem.lower()
    if stem in _BIRCH_STEMS:   return _BIRCH_TINT
    if stem in _SPRUCE_STEMS:  return _SPRUCE_TINT
    if stem in _FOLIAGE_STEMS: return _FOLIAGE_TINT
    if stem in _GRASS_STEMS:   return _GRASS_TINT
    if stem in _WATER_STEMS:   return _WATER_TINT
    return None


# ── Texture atlas ─────────────────────────────────────────────────────────────

_LEAF_STEMS = _FOLIAGE_STEMS | _BIRCH_STEMS | _SPRUCE_STEMS | frozenset({
    "azalea_leaves", "flowering_azalea_leaves",
    "bamboo_large_leaves", "bamboo_small_leaves", "cherry_leaves",
})


def _build_atlas(
    texture_paths: list[str],
    atlas_cols: int = 64,
    fast_graphics: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]:
    """
    Build a packed RGBA texture atlas.
    Biome tints are baked in for leaves, grass, and cross-shaped plants.
    fast_graphics=True: leaf textures are made fully opaque (alpha=255).
    Returns (atlas uint8 H×W×4, path → (u0, v0, u1, v1) in 0-1 atlas coords).
    """
    n = len(texture_paths)
    n_rows = max(1, math.ceil(n / atlas_cols))
    atlas_w = atlas_cols * 16
    atlas_h = n_rows * 16
    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)

    uv_rects: dict[str, tuple[float, float, float, float]] = {}
    for idx, path in enumerate(texture_paths):
        col = idx % atlas_cols
        row = idx // atlas_cols
        try:
            img = Image.open(path).convert("RGBA")
            w, h = img.size
            if h > w:
                img = img.crop((0, 0, w, w))
            if img.size != (16, 16):
                img = img.resize((16, 16), Image.NEAREST)
            arr = np.array(img, dtype=np.uint8)

            tint = _texture_tint(path)
            if tint is not None:
                rgb = arr[:, :, :3].astype(np.float32) * tint
                arr[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

            # Fast graphics: leaf textures become fully opaque (no transparent gaps)
            if fast_graphics and Path(path).stem.lower() in _LEAF_STEMS:
                arr[:, :, 3] = 255

            atlas[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = arr
        except Exception as e:
            log.debug("Texture load failed for %s: %s", path, e)

        u0 = col * 16 / atlas_w
        v0 = row * 16 / atlas_h
        u1 = (col + 1) * 16 / atlas_w
        v1 = (row + 1) * 16 / atlas_h
        uv_rects[path] = (u0, v0, u1, v1)

    return atlas, uv_rects


# ── BER geometry injection ────────────────────────────────────────────────────

def _mc_box_quads(u0, v0, ox, oy, oz, w, h, d, tw, th,
                  pose_offset=(0.0, 0.0, 0.0)):
    """
    Convert a Minecraft model addBox call to 6 face quads.
    Returns list of (4_positions, 4_uvs) where each position/uv is [x,y,z]/[u,v].
    Positions are in [0,1] block space (raw model [0,16] / 16).
    UV formula extracted from ModelPart$Cube.java and ModelPart$Polygon.java.
    """
    # Apply pose offset and convert to [0,1] space
    px, py, pz = pose_offset
    x0 = (ox + px) / 16.0;  x1 = x0 + w / 16.0
    y0 = (oy + py) / 16.0;  y1 = y0 + h / 16.0
    z0 = (oz + pz) / 16.0;  z1 = z0 + d / 16.0

    # 8 corners of the box (named as in Cube.java)
    v1 = [x0, y0, z0];  v2 = [x1, y0, z0]
    v3 = [x1, y1, z0];  v4 = [x0, y1, z0]
    v5 = [x0, y0, z1];  v6 = [x1, y0, z1]
    v7 = [x1, y1, z1];  v8 = [x0, y1, z1]

    # UV edge offsets (in texels, from Cube.java)
    f4 = u0;          f5 = u0+d;       f6 = u0+d+w
    f7 = u0+d+w+w;    f8 = u0+d+w+d;   f9 = u0+d+w+d+w
    f10 = v0;         f11 = v0+d;      f12 = v0+d+h

    def _quad(verts4, us, vs, ue, ve):
        # From Polygon.java: vertex[0]→(ue,vs), [1]→(us,vs), [2]→(us,ve), [3]→(ue,ve)
        uvs = [[ue/tw, vs/th], [us/tw, vs/th], [us/tw, ve/th], [ue/tw, ve/th]]
        return (verts4, uvs)

    return [
        _quad([v6,v5,v1,v2], f5,f10,f6,f11),   # DOWN  (-Y)
        _quad([v3,v4,v8,v7], f6,f11,f7,f10),   # UP    (+Y)  V-flipped
        _quad([v1,v5,v8,v4], f4,f11,f5,f12),   # WEST  (-X)
        _quad([v2,v1,v4,v3], f5,f11,f6,f12),   # NORTH (-Z)
        _quad([v6,v2,v3,v7], f6,f11,f8,f12),   # EAST  (+X)
        _quad([v5,v6,v7,v8], f8,f11,f9,f12),   # SOUTH (+Z)
    ]


def _quads_to_tris(quads, tex_idx):
    """Split quads to baker (pos (3,3), uv (3,2), tex_idx) triples."""
    tris = []
    for verts, uvs in quads:
        for a, b, c in [(0,1,2), (0,2,3)]:
            pos = np.array([verts[a], verts[b], verts[c]], dtype=np.float32)
            uv  = np.array([uvs[a],  uvs[b],  uvs[c]],  dtype=np.float32)
            tris.append((pos, uv, tex_idx))
    return tris


def _rotate_y_verts(quads, facing):
    """Rotate quad positions around Y at (0.5,0.5,0.5) for block facing."""
    # ChestRenderer: mulPose(Axis.YP.rotationDegrees(-toYRot(facing)))
    # toYRot: SOUTH=0, WEST=90, NORTH=180, EAST=270
    angle = {"south": 0.0, "west": -90.0, "north": -180.0, "east": -270.0,
             "up": 0.0, "down": 0.0}.get(facing, 0.0)
    if angle == 0.0:
        return quads
    rad = math.radians(angle)
    ca, sa = math.cos(rad), math.sin(rad)
    def rot(v):
        dx, dz = v[0] - 0.5, v[2] - 0.5
        return [0.5 + ca*dx + sa*dz, v[1], 0.5 - sa*dx + ca*dz]
    return [([rot(p) for p in verts], uvs) for verts, uvs in quads]


def _inject_ber_geometry(state_data, bst_states, state_properties,
                          ber_tex_dir, texture_to_idx, all_textures_ordered):
    """
    Inject hardcoded geometry for Block Entity Renderer blocks that have no
    block model (chest, sign, bed, shulker box, ender chest).
    Geometry extracted from deobfuscated 1.19.4 source via Mojang mappings.
    """

    def _add_tex(rel):
        p = str(ber_tex_dir / rel)
        if not Path(p).exists():
            return None
        if p not in texture_to_idx:
            texture_to_idx[p] = len(all_textures_ordered)
            all_textures_ordered.append(p)
        return texture_to_idx[p]

    def _set(sid, quads, tex_idx):
        if tex_idx is None or not quads:
            return
        tris = _quads_to_tris(quads, tex_idx)
        state_data[sid]["faces"].setdefault(None, []).extend(tris)  # None = no culling
        state_data[sid]["transparency"] = 1   # partial / BER

    # ── Chest boxes (createSingleBodyLayer) ───────────────────────────────────
    # bottom: texOffs(0,19).addBox(1,0,1, 14,10,14)  tw=64 th=64
    # lid:    texOffs(0,0).addBox(1,0,0, 14,5,14) pose_offset=(0,9,1)
    # lock:   texOffs(0,0).addBox(7,-2,14, 2,4,1) pose_offset=(0,9,1)
    def _chest_single_quads(facing):
        q  = _mc_box_quads(0,19, 1,0,1, 14,10,14, 64,64)
        q += _mc_box_quads(0, 0, 1,0,0, 14, 5,14, 64,64, pose_offset=(0,9,1))
        q += _mc_box_quads(0, 0, 7,-2,14, 2,4,1,  64,64, pose_offset=(0,9,1))
        return _rotate_y_verts(q, facing)

    # double right: texOffs(0,19).addBox(1,0,1, 15,10,14) / lid (1,0,0,15,5,14)
    def _chest_right_quads(facing):
        q  = _mc_box_quads(0,19, 1,0,1, 15,10,14, 64,64)
        q += _mc_box_quads(0, 0, 1,0,0, 15, 5,14, 64,64, pose_offset=(0,9,1))
        return _rotate_y_verts(q, facing)

    # double left: texOffs(0,19).addBox(0,0,1, 15,10,14) / lid (0,0,0,15,5,14)
    def _chest_left_quads(facing):
        q  = _mc_box_quads(0,19, 0,0,1, 15,10,14, 64,64)
        q += _mc_box_quads(0, 0, 0,0,0, 15, 5,14, 64,64, pose_offset=(0,9,1))
        return _rotate_y_verts(q, facing)

    # ── Sign boxes (createSignLayer, tw=64 th=32) ─────────────────────────────
    # sign board: texOffs(0,0).addBox(-12,-14,-1, 24,12,2)
    # stick:      texOffs(0,14).addBox(-1,-2,-1, 2,14,2)
    # Sign model origin is at block center (0.5,0.5,0.5)
    def _sign_board_quads(rotation_deg):
        q  = _mc_box_quads(0, 0, -12,-14,-1, 24,12,2, 64,32)
        q += _mc_box_quads(0,14,  -1, -2,-1,  2,14,2, 64,32)
        # Shift from model-center space to block space
        q = [([[ v[0]+0.5, v[1]+0.5, v[2]+0.5] for v in verts], uvs) for verts, uvs in q]
        # Apply Y rotation for standing sign orientation
        rad = math.radians(rotation_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        def rot(v):
            dx, dz = v[0] - 0.5, v[2] - 0.5
            return [0.5 + ca*dx + sa*dz, v[1], 0.5 - sa*dx + ca*dz]
        return [([rot(p) for p in verts], uvs) for verts, uvs in q]

    def _wall_sign_quads(facing):
        q  = _mc_box_quads(0, 0, -12,-14,-1, 24,12,2, 64,32)
        q = [([[v[0]+0.5, v[1]+0.5, v[2]+0.5] for v in verts], uvs) for verts, uvs in q]
        return _rotate_y_verts(q, facing)

    # ── Bed boxes (tw=64 th=64) ───────────────────────────────────────────────
    # Head: texOffs(0,0).addBox(0,0,0, 16,16,6)  — Y-rotated 90° to lie flat
    # Foot: texOffs(0,22).addBox(0,0,0, 16,16,6) — Y-rotated 90°
    # Simplified: render as a flat slab on the ground, 9/16 tall
    def _bed_head_quads(facing):
        # Bed lies flat: the model box (w=16,h=16,d=6) is rotated 90° around X.
        # Simplified: map model Y→world Z, model Z→world Y scaled to 9/16 height.
        q = _mc_box_quads(0, 0, 0,0,0, 16,16,6, 64,64)
        def transform(v):
            return [v[0], v[2] * (9.0/16.0), v[1]]   # head occupies z=[0,1]
        q = [([transform(p) for p in verts], uvs) for verts, uvs in q]
        return _rotate_y_verts(q, facing)

    def _bed_foot_quads(facing):
        # Foot occupies its own block [0,1]³ — same flat slab, different texture region
        q = _mc_box_quads(0,22, 0,0,0, 16,16,6, 64,64)
        def transform(v):
            return [v[0], v[2] * (9.0/16.0), v[1]]
        q = [([transform(p) for p in verts], uvs) for verts, uvs in q]
        return _rotate_y_verts(q, facing)

    # ── Shulker box (tw=64 th=64) ────────────────────────────────────────────
    # lid:  texOffs(0, 0).addBox(-8,-16,-8, 16,12,16)  pose origin at block center
    # base: texOffs(0,28).addBox(-8,-8,-8, 16,8,16)
    def _shulker_quads(facing_up=True):
        # In closed state, lid sits on base. Model origin at bottom-center of block.
        q  = _mc_box_quads(0, 0, -8,-16,-8, 16,12,16, 64,64)
        q += _mc_box_quads(0,28, -8, -8,-8, 16, 8,16, 64,64)
        # Shift: model origin at (8,16,8) in [0,16] = (0.5, 1.0, 0.5) in block space
        # i.e., at the TOP-center. Translate so block occupies [0,1]^3
        def shift(v):
            return [v[0]+0.5, v[1]+1.0, v[2]+0.5]
        q = [([[v[0]+0.5, v[1]+1.0, v[2]+0.5] for v in verts], uvs) for verts, uvs in q]
        return q

    # ── Map block names to handler ─────────────────────────────────────────────
    sign_wood_types = {
        "oak", "spruce", "birch", "acacia", "cherry", "jungle",
        "dark_oak", "mangrove", "bamboo", "crimson", "warped",
    }
    bed_colors = {
        "white", "orange", "magenta", "light_blue", "yellow", "lime",
        "pink", "gray", "light_gray", "cyan", "purple", "blue",
        "brown", "green", "red", "black",
    }
    shulker_colors = bed_colors | {""}   # "" = undyed shulker_box

    for sid, sd in enumerate(state_data):
        sp = state_properties.get(str(sid))
        if sp is None:
            continue
        name = bst_states[sid]["name"].removeprefix("minecraft:")
        props = sp.get("properties", {})

        # ── Chests ────────────────────────────────────────────────────────────
        if name in ("chest", "trapped_chest"):
            tex_key = "trapped" if name == "trapped_chest" else "normal"
            chest_type = props.get("type", "single")
            facing     = props.get("facing", "south")
            if chest_type == "single":
                tex = _add_tex(f"chest/{tex_key}.png")
                _set(sid, _chest_single_quads(facing), tex)
            elif chest_type == "right":
                tex = _add_tex(f"chest/{tex_key}_right.png")
                _set(sid, _chest_right_quads(facing), tex)
            elif chest_type == "left":
                tex = _add_tex(f"chest/{tex_key}_left.png")
                _set(sid, _chest_left_quads(facing), tex)

        elif name == "ender_chest":
            facing = props.get("facing", "south")
            tex = _add_tex("chest/ender.png")
            _set(sid, _chest_single_quads(facing), tex)

        # ── Standing signs ────────────────────────────────────────────────────
        elif any(name == f"{wt}_sign" for wt in sign_wood_types):
            wt = name.replace("_sign", "")
            rotation = int(props.get("rotation", 0)) * (-22.5)  # 0-15 → degrees
            tex = _add_tex(f"signs/{wt}.png")
            _set(sid, _sign_board_quads(rotation), tex)

        # ── Wall signs ────────────────────────────────────────────────────────
        elif any(name == f"{wt}_wall_sign" for wt in sign_wood_types):
            wt = name.replace("_wall_sign", "")
            facing = props.get("facing", "south")
            tex = _add_tex(f"signs/{wt}.png")
            _set(sid, _wall_sign_quads(facing), tex)

        # ── Hanging signs ─────────────────────────────────────────────────────
        elif any(name == f"{wt}_hanging_sign" for wt in sign_wood_types):
            wt = name.replace("_hanging_sign", "")
            rotation = int(props.get("rotation", 0)) * (-22.5)
            tex = _add_tex(f"signs/hanging/{wt}.png")
            if tex:
                _set(sid, _sign_board_quads(rotation), tex)

        # ── Beds ──────────────────────────────────────────────────────────────
        elif any(name == f"{c}_bed" for c in bed_colors):
            color  = name.replace("_bed", "")
            facing = props.get("facing", "south")
            part   = props.get("part", "head")
            tex    = _add_tex(f"bed/{color}.png")
            if part == "head":
                _set(sid, _bed_head_quads(facing), tex)
            else:
                _set(sid, _bed_foot_quads(facing), tex)

        # ── Shulker boxes ─────────────────────────────────────────────────────
        elif name == "shulker_box":
            tex = _add_tex("shulker/shulker.png")
            _set(sid, _shulker_quads(), tex)
        elif any(name == f"{c}_shulker_box" for c in bed_colors):
            color = name.replace("_shulker_box", "")
            tex   = _add_tex(f"shulker/shulker_{color}.png")
            _set(sid, _shulker_quads(), tex)


# ── Main bake function ─────────────────────────────────────────────────────────

def bake(jar_path: Path, version: str = "1.19.4",
         fast_graphics: bool = False) -> None:
    jar_path = Path(jar_path)
    cache_version = f"{version}-fast" if fast_graphics else version
    cache_dir = _ROOT / "cache" / cache_version
    cache_dir.mkdir(parents=True, exist_ok=True)

    geo_cache = cache_dir / f"geometry_{cache_version}.npz"
    if geo_cache.exists():
        log.info("Geometry cache already exists: %s  (delete to re-bake)", geo_cache)
        return

    # ── Step 1: Extract JAR ────────────────────────────────────────────────────
    # Re-use the regular version's extracted assets (no need to re-extract)
    base_cache = _ROOT / "cache" / version
    extracted_dir = base_cache / f"extracted_{version}"
    if not extracted_dir.exists():
        _extract_jar(jar_path, extracted_dir)
    else:
        log.info("Reusing extracted JAR from %s", extracted_dir)

    # ── Step 2: State properties JSON (re-use from base version) ──────────────
    base_props = base_cache / f"state_properties_{version}.json"
    if not base_props.exists():
        _gen_state_properties(base_cache, version)
    state_props_path = base_props
    with open(state_props_path) as f:
        state_properties: dict[str, dict] = json.load(f)

    # ── Step 3: Blockstate table (name / opaque) ───────────────────────────────
    bst_path = _ROOT / "data" / f"blockstate_table_{version}.json"
    with open(bst_path) as f:
        bst = json.load(f)
    bst_states: list[dict] = bst["states"]
    num_states = len(bst_states)

    # ── Step 4: Minecraft-Model-Reader ─────────────────────────────────────────
    # MMR cache goes in base_cache so it's shared between fancy/fast bakes
    manager = _load_mmr(extracted_dir, base_cache)

    from minecraft_model_reader.api import Block
    import amulet_nbt

    missing_no_path: str = manager.missing_no

    # ── Step 5: Extract mesh data per state ID ─────────────────────────────────
    log.info("Processing %d state IDs ...", num_states)

    all_textures_ordered: list[str] = []
    texture_to_idx: dict[str, int] = {}

    # Per-state geometry: list of dicts
    # state_data[sid] = {
    #   "transparency": int,
    #   "faces": { cull_dir → list of (pos (3,3), uv (3,2), tex_idx int) }
    # }
    state_data: list[dict] = []

    for sid in range(num_states):
        name = bst_states[sid]["name"]

        if name in AIR_NAMES:
            state_data.append({"transparency": 2, "faces": {}})
            continue

        sp = state_properties.get(str(sid))
        if sp is None:
            state_data.append({"transparency": 2, "faces": {}})
            continue

        raw_name: str = sp["name"]
        if ":" in raw_name:
            ns, base = raw_name.split(":", 1)
        else:
            ns, base = "minecraft", raw_name

        props = {
            k: amulet_nbt.TAG_String(v)
            for k, v in sp["properties"].items()
        }
        block = Block(namespace=ns, base_name=base, properties=props)

        try:
            mesh = manager._get_model(block)
        except Exception as e:
            log.debug("Model load failed for sid %d (%s): %s", sid, name, e)
            state_data.append({"transparency": 2, "faces": {}})
            continue

        transparency = int(mesh.is_transparent)
        faces_data: dict = {}

        for cull_dir, faces_arr in mesh.faces.items():
            if len(faces_arr) == 0:
                continue

            verts_arr = mesh.verts[cull_dir]
            tc_arr = mesh.texture_coords[cull_dir]
            ti_arr = mesh.texture_index[cull_dir]

            n_tris = len(faces_arr) // 3
            faces_r = faces_arr.reshape(-1, 3)
            verts_r = verts_arr.reshape(-1, 3)
            uvs_r = tc_arr.reshape(-1, 2)

            tris = []
            for t in range(n_tris):
                vi = faces_r[t]
                tex_idx_raw = int(ti_arr[t])
                if tex_idx_raw < len(mesh.textures):
                    tex_path = mesh.textures[tex_idx_raw]
                else:
                    tex_path = missing_no_path

                if tex_path not in texture_to_idx:
                    texture_to_idx[tex_path] = len(all_textures_ordered)
                    all_textures_ordered.append(tex_path)

                pos = verts_r[vi].astype(np.float32)   # (3, 3)
                uv = uvs_r[vi].astype(np.float32)       # (3, 2)
                tris.append((pos, uv, texture_to_idx[tex_path]))

            if tris:
                faces_data[cull_dir] = tris

        state_data.append({"transparency": transparency, "faces": faces_data})

        if sid % 1000 == 0:
            log.info("  %d / %d state IDs processed", sid, num_states)

    log.info("Collected %d unique textures.", len(all_textures_ordered))

    # ── Step 5b: Classify fluid states ────────────────────────────────────────
    # fluid_type[sid]: 0=not fluid, 1=water, 2=lava
    # fluid_level[sid]: 0-15 (water/lava level from blockstate properties)
    fluid_type  = np.zeros(num_states, dtype=np.uint8)
    fluid_level = np.zeros(num_states, dtype=np.uint8)
    for sid in range(num_states):
        name = bst_states[sid]["name"]
        sp   = state_properties.get(str(sid))
        if name == "minecraft:water" and sp is not None:
            fluid_type[sid]  = 1
            fluid_level[sid] = int(sp["properties"].get("level", 0))
        elif name == "minecraft:lava" and sp is not None:
            fluid_type[sid]  = 2
            fluid_level[sid] = int(sp["properties"].get("level", 0))

    # ── Step 5c: Force-add fluid textures (not in any block model) ────────────
    tex_dir = extracted_dir / "assets" / "minecraft" / "textures" / "block"
    _fluid_tex_names = [
        "water_still", "water_flow",
        "lava_still",  "lava_flow",
    ]
    fluid_tex_paths: dict[str, str] = {}
    for name in _fluid_tex_names:
        p = str(tex_dir / f"{name}.png")
        if Path(p).exists():
            if p not in texture_to_idx:
                texture_to_idx[p] = len(all_textures_ordered)
                all_textures_ordered.append(p)
            fluid_tex_paths[name] = p
    log.info("Fluid textures added: %s", list(fluid_tex_paths.keys()))

    # ── Step 5d: BER geometry injection ──────────────────────────────────────
    ber_tex_dir = extracted_dir / "assets" / "minecraft" / "textures" / "entity"
    _inject_ber_geometry(
        state_data, bst_states, state_properties, ber_tex_dir,
        texture_to_idx, all_textures_ordered,
    )
    log.info("BER geometry injected.")

    # ── Step 5e: Fast-graphics leaf override ──────────────────────────────────
    if fast_graphics:
        for sid in range(num_states):
            if "leaves" in bst_states[sid]["name"]:
                state_data[sid]["transparency"] = 0   # FullOpaque → opaque pass

    # ── Step 6: Build texture atlas ────────────────────────────────────────────
    atlas, uv_rects = _build_atlas(all_textures_ordered,
                                   fast_graphics=fast_graphics)

    # ── Step 7: Remap tile UVs → atlas UVs, build geometry arrays ─────────────
    log.info("Building per-cull-dir geometry arrays ...")

    transparency_arr = np.zeros(num_states, dtype=np.uint8)

    verts_accum: dict[str, list[np.ndarray]] = {k: [] for k in CULL_DIR_KEYS}
    offsets: dict[str, list[int]] = {k: [0] for k in CULL_DIR_KEYS}

    for sid in range(num_states):
        sd = state_data[sid]
        transparency_arr[sid] = sd["transparency"]

        for cull_dir, key in zip(CULL_DIRS, CULL_DIR_KEYS):
            tris = sd["faces"].get(cull_dir, [])

            if not tris:
                offsets[key].append(offsets[key][-1])
                continue

            rows = []
            for pos, uv, tex_idx in tris:
                tex_path = all_textures_ordered[tex_idx]
                u0, v0, u1, v1 = uv_rects[tex_path]
                for vi in range(3):
                    x = float(pos[vi, 0])
                    y = float(pos[vi, 1])
                    z = float(pos[vi, 2])
                    # Clamp tile UV to [0,1] before remapping to atlas space
                    tu = max(0.0, min(1.0, float(uv[vi, 0])))
                    tv = max(0.0, min(1.0, float(uv[vi, 1])))
                    au = u0 + tu * (u1 - u0)
                    av = v0 + tv * (v1 - v0)
                    rows.append((x, y, z, au, av))

            arr = np.array(rows, dtype=np.float32)  # (N*3, 5)
            verts_accum[key].append(arr)
            offsets[key].append(offsets[key][-1] + len(arr))

    # ── Step 8: Save cache files ───────────────────────────────────────────────
    log.info("Saving geometry cache to %s ...", geo_cache)

    save_dict: dict[str, np.ndarray] = {
        "transparency": transparency_arr,
        "fluid_type":   fluid_type,
        "fluid_level":  fluid_level,
    }
    for key in CULL_DIR_KEYS:
        arrs = verts_accum[key]
        save_dict[f"verts_{key}"] = (
            np.concatenate(arrs, axis=0) if arrs
            else np.zeros((0, 5), dtype=np.float32)
        )
        save_dict[f"offsets_{key}"] = np.array(offsets[key], dtype=np.int32)

    np.savez_compressed(str(geo_cache), **save_dict)
    log.info("Geometry cache saved.")

    atlas_path = cache_dir / f"atlas_{cache_version}.png"
    Image.fromarray(atlas).save(str(atlas_path))
    log.info("Atlas saved: %s (%dx%d)", atlas_path, atlas.shape[1], atlas.shape[0])

    # Fluid UV rects for the renderer
    fluid_uvs: dict[str, list[float]] = {}
    for name, path in fluid_tex_paths.items():
        if path in uv_rects:
            fluid_uvs[name] = list(uv_rects[path])

    meta = {
        "version": version,
        "state_count": num_states,
        "atlas_width": int(atlas.shape[1]),
        "atlas_height": int(atlas.shape[0]),
        "texture_count": len(all_textures_ordered),
        "fluid_uvs": fluid_uvs,   # name → [u0,v0,u1,v1]
    }
    (cache_dir / f"meta_{cache_version}.json").write_text(json.dumps(meta, indent=2))
    log.info("Bake complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bake pipeline geometry cache.")
    default_jar = (
        _ROOT.parent.parent / "visualizer" / ".cache" / "minecraft-1.19.4.jar"
    )
    parser.add_argument("--jar", type=Path, default=default_jar,
                        help="Path to Minecraft JAR (default: visualizer/.cache/minecraft-1.19.4.jar)")
    parser.add_argument("--version", default="1.19.4",
                        help="Version tag for cache subdirectory")
    parser.add_argument("--fast-graphics", action="store_true",
                        help="Bake leaves as fully opaque (Fast graphics mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    bake(args.jar, args.version, fast_graphics=args.fast_graphics)


if __name__ == "__main__":
    main()
