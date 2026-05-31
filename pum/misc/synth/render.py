"""
synth/render.py — Render inventory icons for every item in the vocab.

Uses Minecraft's model JSONs to determine rendering type:
  - item/generated  → flat 2D sprite (torches, food, arrows, etc.)
  - item/handheld   → flat 2D sprite (tools, swords — same as generated in GUI)
  - block models    → isometric 3-face composite (cobblestone, dirt, etc.)

Outputs 16×16 RGBA PNGs to synth/icons/<item_name>.png.
Run once after extract.py:
  python synth/render.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import vocab as V

ASSETS_DIR = Path(__file__).parent / "assets"
ICONS_DIR  = Path(__file__).parent / "icons"
ICON_SIZE  = 16   # pixels — will be scaled up at generation time

# Isometric face shading (top / left / right)
SHADE_TOP   = 1.00
SHADE_LEFT  = 0.80
SHADE_RIGHT = 0.60

# Biome tint colours applied to tintindex faces.
# Sampled from Minecraft's default temperate biome colormap.
GRASS_TINT   = np.array([117, 176,  73], dtype=np.float32) / 255.0
FOLIAGE_TINT = np.array([ 89, 132,  33], dtype=np.float32) / 255.0
WATER_TINT   = np.array([ 63, 118, 228], dtype=np.float32) / 255.0


# ─── Texture loading ──────────────────────────────────────────────────────────

def _load_tex(rel_path: str) -> Image.Image | None:
    """Load a texture PNG, returning RGBA or None if not found."""
    for base in (ASSETS_DIR / "textures",):
        p = base / rel_path
        if p.exists():
            img = Image.open(p).convert("RGBA")
            # Animated textures are tall strips — take first frame (top square).
            if img.height > img.width:
                img = img.crop((0, 0, img.width, img.width))
            return img
    return None


def _tex_from_ref(ref: str) -> Image.Image | None:
    """Resolve a Minecraft texture reference like 'block/stone' → PNG."""
    ref = ref.lstrip("#")
    if ":" in ref:
        ref = ref.split(":", 1)[1]
    for subdir in ("item", "block"):
        img = _load_tex(f"{subdir}/{ref}.png")
        if img:
            return img
    img = _load_tex(f"{ref}.png")
    return img


# ─── Model JSON parsing ────────────────────────────────────────────────────────

def _find_model_path(name: str):
    """Resolve a model name to a Path, handling 'block/foo' and 'item/foo' prefixes."""
    # name may be bare ("cobblestone") or prefixed ("block/cobblestone", "item/handheld")
    if "/" in name:
        subdir, fname = name.split("/", 1)
        p = ASSETS_DIR / "models" / subdir / f"{fname}.json"
        if p.exists():
            return p
        # fallback: treat the whole thing as a filename under item/ or block/
    for subdir in ("item", "block"):
        p = ASSETS_DIR / "models" / subdir / f"{name}.json"
        if p.exists():
            return p
    return None


def _load_model(name: str, _seen: frozenset = frozenset()) -> dict:
    """Load and flatten a model JSON (follows 'parent' chain, cycle-safe)."""
    if name in _seen:
        return {}
    p = _find_model_path(name)
    if p is None:
        return {}
    with open(p) as f:
        model = json.load(f)
    parent = model.get("parent", "")
    if parent and not parent.startswith("builtin"):
        # Strip namespace prefix ("minecraft:block/foo" → "block/foo")
        parent_name = parent.split(":")[-1]
        parent_model = _load_model(parent_name, _seen | {name})
        merged = {**parent_model, **model}
        merged["textures"] = {**parent_model.get("textures", {}),
                               **model.get("textures", {})}
        return merged
    return model


# ─── Renderers ────────────────────────────────────────────────────────────────

def _tint_for_layer(model: dict, layer_key: str) -> np.ndarray | None:
    """Return RGB tint array [3] for a layer, or None if untinted.
    Detects tintindex by scanning element face definitions."""
    layer_idx = int(layer_key.replace("layer", "")) if layer_key.startswith("layer") else -1
    for elem in model.get("elements", []):
        for face_data in elem.get("faces", {}).values():
            if face_data.get("tintindex", -1) == layer_idx:
                # Determine tint type from texture name.
                textures = model.get("textures", {})
                ref = textures.get(layer_key, "")
                if any(k in ref for k in ("grass", "fern", "vine", "lily_pad", "leaves")):
                    return GRASS_TINT if "grass" in ref else FOLIAGE_TINT
                return GRASS_TINT  # default tint for unknown tinted faces
    return None


def _apply_tint(img: Image.Image, tint: np.ndarray) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    arr[..., :3] *= tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


def _render_flat(model: dict, size: int) -> Image.Image | None:
    """item/generated or item/handheld — composite all layers, apply tints."""
    textures = model.get("textures", {})
    result = None
    for key in sorted(k for k in textures if k.startswith("layer")):
        ref = textures[key]
        if not ref:
            continue
        layer_img = _tex_from_ref(ref)
        if layer_img is None:
            continue
        layer_img = layer_img.resize((size, size), Image.NEAREST)
        tint = _tint_for_layer(model, key)
        if tint is not None:
            layer_img = _apply_tint(layer_img, tint)
        if result is None:
            result = layer_img
        else:
            result = Image.alpha_composite(result, layer_img)
    return result


def _render_handheld(model: dict, size: int) -> Image.Image | None:
    """item/handheld — in the inventory GUI these are flat 2D sprites, no rotation."""
    return _render_flat(model, size)


def _resolve_tex_ref(ref: str, textures: dict) -> Image.Image | None:
    """Resolve a texture reference like '#side' or 'block/cobblestone' to an image."""
    ref = ref.lstrip("#")
    # Follow #-reference chains through the textures map.
    for _ in range(8):
        if not ref.startswith("#") and ref in textures:
            ref = textures[ref].lstrip("#")
        else:
            break
    resolved = textures.get(ref, ref)
    if resolved.startswith("#"):
        resolved = textures.get(resolved.lstrip("#"), resolved)
    return _tex_from_ref(resolved)


def _face_tex_and_tint(model: dict, face_name: str):
    """
    Return (PIL image | None, tint | None) for a block face by reading
    element face definitions with UV cropping.  Falls back to texture-map keys.
    """
    textures = model.get("textures", {})
    elements = model.get("elements", [])

    # Use the LARGEST element that has this face (best approximation for the
    # dominant surface — avoids picking tiny detail elements like buttons).
    best_img, best_tint, best_vol = None, None, -1
    for elem in elements:
        faces = elem.get("faces", {})
        if face_name not in faces:
            continue
        face_data = faces[face_name]
        tex_ref = face_data.get("texture", "")
        img = _resolve_tex_ref(tex_ref, textures)
        if img is None:
            continue
        # Apply UV crop (in 0-16 texture space).
        uv = face_data.get("uv")
        if uv and len(uv) == 4:
            w, h = img.size
            x0, y0, x1, y1 = (int(v / 16 * w) for v in uv[:4])
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))
        tint = GRASS_TINT if face_data.get("tintindex", -1) >= 0 else None
        # Volume of this element.
        frm = elem.get("from", [0, 0, 0])
        to  = elem.get("to",   [16, 16, 16])
        vol = (to[0]-frm[0]) * (to[1]-frm[1]) * (to[2]-frm[2])
        if vol > best_vol:
            best_img, best_tint, best_vol = img, tint, vol

    if best_img is not None:
        return best_img, best_tint

    # Fallback: look in texture key map directly.
    FACE_KEYS = {
        "up":    ["top", "end", "up",   "all", "particle", "texture", "side"],
        "north": ["north", "side", "end", "all", "particle", "texture"],
        "east":  ["east",  "side", "end", "all", "particle", "texture"],
    }
    for key in FACE_KEYS.get(face_name, ["all", "particle", "texture"]):
        ref = textures.get(key)
        if ref:
            img = _resolve_tex_ref(ref, textures)
            if img:
                return img, None
    return None, None


def _render_isometric_block(model: dict, size: int) -> Image.Image | None:
    """
    Isometric block icon: top + left (north) + right (east) faces composited.
    Uses element face definitions with UV mapping for correct texture selection.
    """
    top_tex,   top_tint   = _face_tex_and_tint(model, "up")
    left_tex,  left_tint  = _face_tex_and_tint(model, "north")
    right_tex, right_tint = _face_tex_and_tint(model, "east")

    if top_tex is None:
        return None
    if left_tex  is None: left_tex,  left_tint  = top_tex,  top_tint
    if right_tex is None: right_tex, right_tint = left_tex, left_tint

    s    = size
    half = s // 2

    def shade(pil_img, factor, w, h, tint=None):
        arr = np.array(pil_img.resize((w, h), Image.NEAREST), dtype=np.float32)
        if tint is not None:
            arr[..., :3] *= tint
        arr[..., :3] *= factor
        return np.clip(arr, 0, 255).astype(np.uint8)

    out = np.zeros((s, s, 4), dtype=np.uint8)

    # Top face: parallelogram across upper half, skewed left-to-right.
    top_arr = shade(top_tex, SHADE_TOP, s, half, top_tint)
    for y in range(half):
        shift = half - 1 - y
        src   = top_arr[y]
        avail = s - shift
        if avail > 0:
            out[y, shift: shift + avail] = src[:avail]

    # Left face: bottom-left quadrant.
    left_arr = shade(left_tex, SHADE_LEFT, half, half, left_tint)
    out[half:, :half] = left_arr

    # Right face: bottom-right quadrant.
    right_arr = shade(right_tex, SHADE_RIGHT, half, half, right_tint)
    out[half:, half:] = right_arr

    return Image.fromarray(out, "RGBA")


def render_item_icon(item_name: str, size: int = ICON_SIZE) -> Image.Image | None:
    """
    Render an inventory icon for `item_name` (e.g. 'cobblestone', 'torch').
    Returns a PIL RGBA image of `size×size` pixels, or None on failure.
    """
    model = _load_model(item_name)
    if not model:
        return None

    parent = model.get("parent", "")

    if "builtin/entity" in parent:
        # Chests, beds, banners, shields, shulker boxes etc. use entity rendering
        # which is outside the model JSON system entirely. Fall back to particle texture.
        textures = model.get("textures", {})
        ref = textures.get("particle") or next(iter(textures.values()), None)
        img = _tex_from_ref(ref).resize((size, size), Image.NEAREST) if ref else None
    elif "handheld" in parent:
        img = _render_handheld(model, size)
    elif "generated" in parent:
        img = _render_flat(model, size)
    elif model.get("elements") or "block/" in parent:
        img = _render_isometric_block(model, size)
    else:
        img = _render_flat(model, size)

    return img


# ─── Batch render all items in vocab ─────────────────────────────────────────

def render_all(size: int = ICON_SIZE, force: bool = False):
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    items = [(V.item_id_to_name(i), i) for i in range(V.vocab_size())]
    ok = skipped = failed = 0

    for name, iid in items:
        if name == "air":
            continue
        dest = ICONS_DIR / f"{name}.png"
        if dest.exists() and not force:
            skipped += 1
            continue
        img = render_item_icon(name, size=size)
        if img is None:
            print(f"  [MISS]    {name}")
            failed += 1
        else:
            img.save(dest)
            ok += 1

    print(f"\nIcons: {ok} rendered  {skipped} cached  {failed} failed")
    return ok, failed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--size",  type=int,  default=ICON_SIZE)
    p.add_argument("--force", action="store_true")
    p.add_argument("--item",  type=str,  default=None,
                   help="Render a single item and save preview.png")
    args = p.parse_args()

    if args.item:
        img = render_item_icon(args.item, size=max(args.size, 64))
        if img:
            out = Path(f"synth_{args.item}.png")
            img.save(out)
            print(f"Saved {out}")
        else:
            print(f"Failed to render {args.item}")
    else:
        render_all(size=args.size, force=args.force)
