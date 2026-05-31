"""
synth/generate.py — Synthetic Minecraft inventory image generator.

Player slot IDs follow the Minecraft Java Edition protocol (wiki.vg/Inventory).
All layouts expose two separate slot arrays:

  player_ids[46]    — indexed by protocol slot ID:
    0      crafting output   (survival/survival_recipe only)
    1-4    crafting 2×2 grid (survival/survival_recipe only)
    5-8    armor: head, chest, legs, feet (survival/survival_recipe only)
    9-35   main inventory (row-major, 9 = top-left of top row)
    36-44  hotbar (36 = slot 0 / leftmost)
    45     offhand (survival/survival_recipe only)

  container_ids[N]  — indexed 0..N-1, container-specific:
    chest           N=27  (3×9)
    large_chest     N=54  (6×9)
    furnace/smoker  N=3   (ingredient, fuel, output)
    dispenser       N=9   (3×3 grid)
    hopper          N=5   (5-slot row)
    brewing_stand   N=5   (fuel, ingredient, bottle×3)
    enchanting_table N=2  (item, lapis)
    anvil           N=3   (input1, input2, output)
    cartography_table N=3 (map, paper, output)
    grindstone      N=3   (input1, input2, output)
    smithing        N=3   (input, material, output)
    stonecutter     N=2   (input, output)

Slot positions measured from 1.19.2 jar GUI textures via pixel analysis.
1/3 of samples place the GUI at a random valid screen position.
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import IterableDataset

sys.path.insert(0, str(Path(__file__).parents[1]))
import vocab as V

SYNTH_DIR = Path(__file__).parent
ICONS_DIR = SYNTH_DIR / "icons" / "1.16.4"
GUI_DIR   = SYNTH_DIR / "gui"
FONT_PATH = SYNTH_DIR / "minecraft.ttf"

IMG_W, IMG_H = 640, 360
GUI_SCALE    = 1      # VPT recordings at 640×360 use GUI scale 1
SLOT_GUI     = 18     # each slot is 18×18 GUI pixels
SLOT_PX      = SLOT_GUI * GUI_SCALE   # 18 screen pixels per slot

# At GUI scale 1: icon is 16×16 inside an 18×18 slot → 1px border each side.
SLOT_ICON_INSET  = 1
SLOT_ICON_OFFSET = -1
TEXT_ICON_OFFSET = -1   # nudge count text position
ICON_PX          = SLOT_PX - 2 * SLOT_ICON_INSET   # 16px

# Hotbar / offhand HUD (always visible behind dark overlay at fixed position).
HOTBAR_X         = 229     # top-left of hotbar widget in 640×360 canvas
HOTBAR_Y         = 338
HOTBAR_SLOT_W    = 20      # hotbar slot cell is 20px wide (vs 18 in inventory)
HOTBAR_SLOT_INSET = 2      # icon inset within the 20px hotbar cell
OFFHAND_X        = 188     # top-left of offhand widget
OFFHAND_Y        = 338
OFFHAND_ICON_OFF = 16      # icon x offset in offhand widget (12px padding + 4px inset)
OFFHAND_ICON_Y   = 4       # icon y offset in offhand widget
INVENTORY_DIM    = 0.65    # fraction to darken background when inventory is open

_PANEL_NUDGE_X = 0
_PANEL_NUDGE_Y = 0

RECIPE_BTN_X_GUI    = 336 - 232
RECIPE_BTN_Y_GUI    = 158 - 97
RECIPE_ICON_CROP    = (178, 0, 198, 18)
RECIPE_OPEN_PANEL_X = 309

_EXCLUDED_IDS = {
    1056,   # knowledge_book — looks like recipe book, confusing
}

N_PLAYER_SLOTS   = 46   # protocol slot IDs 0-45
_CRAFTING_PROTOS = frozenset((0, 1, 2, 3, 4))  # output + 2×2 grid

# Protocol IDs that show an armor/offhand placeholder when empty.
_ARMOR_SLOT_TYPE = {
    5: "helmet",
    6: "chestplate",
    7: "leggings",
    8: "boots",
    45: "offhand",
}

_ARMOR_PLACEHOLDER_FILES = {
    "helmet":     "empty_armor_slot_helmet.png",
    "chestplate": "empty_armor_slot_chestplate.png",
    "leggings":   "empty_armor_slot_leggings.png",
    "boots":      "empty_armor_slot_boots.png",
    "offhand":    "empty_armor_slot_shield.png",
}


# ── Slot position builders ─────────────────────────────────────────────────────

def _row(n_cols, x_gui, y_gui, scale=GUI_SCALE):
    return [(int((x_gui + c * SLOT_GUI) * scale), int(y_gui * scale))
            for c in range(n_cols)]

def _grid(n_rows, n_cols, x_gui, y_start_gui, scale=GUI_SCALE):
    slots = []
    for r in range(n_rows):
        slots += _row(n_cols, x_gui, y_start_gui + r * SLOT_GUI, scale)
    return slots

def _s(x, y, scale=GUI_SCALE):
    return (int(x * scale), int(y * scale))


# ── 2×2 crafting recipe table ─────────────────────────────────────────────────

def _recipe_placements(inShape, r_id, r_count, valid_set):
    """Expand a shaped recipe to all valid 2×2 grid placements + horizontal mirror.

    Returns list of (grid[4], r_id, r_count) where grid[i] is item_id (0=empty).
    grid indices: 0=TL, 1=TR, 2=BL, 3=BR (row-major, proto slots 1-4).
    """
    rows = len(inShape)
    cols = max((len(row) for row in inShape), default=0)
    if rows > 2 or cols > 2:
        return []
    padded = [list(row) + [None] * (cols - len(row)) for row in inShape]
    entries, seen = [], set()
    for mirror in (False, True):
        shape = [list(reversed(r)) for r in padded] if mirror else padded
        for dr in range(2 - rows + 1):
            for dc in range(2 - cols + 1):
                grid = [0, 0, 0, 0]
                ok = True
                for ri, row in enumerate(shape):
                    for ci, cell in enumerate(row):
                        if cell is not None:
                            if cell not in valid_set:
                                ok = False
                                break
                            grid[(ri + dr) * 2 + (ci + dc)] = cell
                    if not ok:
                        break
                if not ok or r_id not in valid_set:
                    continue
                key = tuple(grid)
                if key not in seen:
                    seen.add(key)
                    entries.append((grid, r_id, r_count))
    return entries


class _CraftingRecipes:
    """Pre-computed 2×2 crafting recipe table for the survival screen."""
    def __init__(self):
        self.entries: list = []   # populated lazily in _init_once

    def _init_once(self, valid_set: set):
        if self.entries:
            return
        import minecraft_data as mcdata
        mc = mcdata(V.CANONICAL_VERSION)
        for _, recipes in mc.recipes.items():
            for rec in recipes:
                if 'inShape' not in rec:
                    continue
                self.entries += _recipe_placements(
                    rec['inShape'],
                    rec['result']['id'],
                    rec['result']['count'],
                    valid_set,
                )

    def random_recipe(self, rng: random.Random, valid_set: set):
        """Return (grid[4], result_id, result_count) for a random 2×2 recipe."""
        self._init_once(valid_set)
        return rng.choice(self.entries) if self.entries else None


_CRAFTING = _CraftingRecipes()


def _player_coords_full(scale=GUI_SCALE):
    """Survival: craft(0-4) + armor(5-8) + main(9-35) + hotbar(36-44) + offhand(45)."""
    c = {}
    c[0] = _s(154, 28, scale)                              # crafting output
    for i, pos in enumerate(_grid(2, 2, 98, 18, scale)):
        c[1 + i] = pos                                     # crafting 2×2 (1-4)
    for i, y in enumerate([8, 26, 44, 62]):
        c[5 + i] = _s(8, y, scale)                        # armor (5-8)
    for i, pos in enumerate(_grid(3, 9, 8, 84, scale)):
        c[9 + i] = pos                                     # main inv (9-35)
    for i, pos in enumerate(_row(9, 8, 142, scale)):
        c[36 + i] = pos                                    # hotbar (36-44)
    c[45] = _s(77, 62, scale)                              # offhand
    return c

def _player_coords_standard(scale=GUI_SCALE):
    """Standard containers: main(9-35) + hotbar(36-44), no armor/crafting."""
    c = {}
    for i, pos in enumerate(_grid(3, 9, 8, 84, scale)):
        c[9 + i] = pos
    for i, pos in enumerate(_row(9, 8, 142, scale)):
        c[36 + i] = pos
    return c

def _player_coords_hopper(scale=GUI_SCALE):
    """Hopper: main(9-35) + hotbar(36-44), y shifted up for shorter panel."""
    c = {}
    for i, pos in enumerate(_grid(3, 9, 8, 51, scale)):
        c[9 + i] = pos
    for i, pos in enumerate(_row(9, 8, 109, scale)):
        c[36 + i] = pos
    return c

def _player_coords_large_chest(scale=GUI_SCALE):
    """Large chest: main(9-35) + hotbar(36-44), y shifted down for 6-row chest."""
    c = {}
    for i, pos in enumerate(_grid(3, 9, 8, 140, scale)):
        c[9 + i] = pos
    for i, pos in enumerate(_row(9, 8, 198, scale)):
        c[36 + i] = pos
    return c


# ── Container layout definition ───────────────────────────────────────────────

@dataclass
class Layout:
    name: str
    gui_file: str
    panel_w_gui: int
    panel_h_gui: int
    container_slots: list    # [(x_rel_px, y_rel_px), ...] indexed 0..N-1
    player_slot_coords: dict # {protocol_id: (x_rel_px, y_rel_px)}
    weight: float = 1.0
    empty_container_slots: frozenset = frozenset()  # container slot indices that are always empty


LAYOUTS: dict[str, Layout] = {
    "survival": Layout(
        name="survival", gui_file="inventory.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[],
        player_slot_coords=_player_coords_full(),
        weight=4.0,
    ),
    "survival_recipe": Layout(
        name="survival_recipe", gui_file="inventory.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[],
        player_slot_coords=_player_coords_full(),
        weight=1.0,
    ),
    "chest": Layout(
        name="chest", gui_file="shulker_box.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=_grid(3, 9, 8, 18),
        player_slot_coords=_player_coords_standard(),
        weight=1.5,
    ),
    "large_chest": Layout(
        name="large_chest", gui_file="generic_54.png",
        panel_w_gui=176, panel_h_gui=222,
        container_slots=_grid(6, 9, 8, 18),
        player_slot_coords=_player_coords_large_chest(),
        weight=1.0,
    ),
    "furnace": Layout(
        name="furnace", gui_file="furnace.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(56,17), _s(56,53), _s(112,31)],  # ingredient, fuel, output
        player_slot_coords=_player_coords_standard(),
        weight=0.5,
        empty_container_slots=frozenset({2}),  # output slot always empty (oversized, would misalign)
    ),
    "smoker": Layout(
        name="smoker", gui_file="smoker.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(56,17), _s(56,53), _s(112,31)],
        player_slot_coords=_player_coords_standard(),
        weight=0.3,
        empty_container_slots=frozenset({2}),
    ),
    "dispenser": Layout(
        name="dispenser", gui_file="dispenser.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=_grid(3, 3, 62, 17),
        player_slot_coords=_player_coords_standard(),
        weight=0.4,
    ),
    "hopper": Layout(
        name="hopper", gui_file="hopper.png",
        panel_w_gui=176, panel_h_gui=133,
        container_slots=_row(5, 44, 20),
        player_slot_coords=_player_coords_hopper(),
        weight=0.4,
    ),
    "brewing_stand": Layout(
        name="brewing_stand", gui_file="brewing_stand.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(17,17), _s(79,17), _s(56,51), _s(102,51), _s(79,58)],
        player_slot_coords=_player_coords_standard(),
        weight=0.3,
    ),
    "enchanting_table": Layout(
        name="enchanting_table", gui_file="enchanting_table.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(15,47), _s(35,47)],
        player_slot_coords=_player_coords_standard(),
        weight=0.4,
    ),
    "anvil": Layout(
        name="anvil", gui_file="anvil.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(27,47), _s(76,47), _s(134,47)],
        player_slot_coords=_player_coords_standard(),
        weight=0.4,
    ),
    "cartography_table": Layout(
        name="cartography_table", gui_file="cartography_table.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(15,15), _s(15,52), _s(141,35)],
        player_slot_coords=_player_coords_standard(),
        weight=0.3,
    ),
    "grindstone": Layout(
        name="grindstone", gui_file="grindstone.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(49,19), _s(49,40), _s(129,34)],
        player_slot_coords=_player_coords_standard(),
        weight=0.4,
    ),
    "smithing": Layout(
        name="smithing", gui_file="smithing.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(27,47), _s(76,47), _s(134,47)],
        player_slot_coords=_player_coords_standard(),
        weight=0.4,
    ),
    "stonecutter": Layout(
        name="stonecutter", gui_file="stonecutter.png",
        panel_w_gui=176, panel_h_gui=166,
        container_slots=[_s(20,33), _s(143,33)],
        player_slot_coords=_player_coords_standard(),
        weight=0.3,
    ),
    # "loom":   output slot detection unreliable, re-enable once measured
    # "beacon": unusual panel size, re-enable later
}

_LAYOUT_NAMES   = list(LAYOUTS.keys())
_LAYOUT_WEIGHTS = [LAYOUTS[k].weight for k in _LAYOUT_NAMES]


# ── Count text constants ───────────────────────────────────────────────────────
# All digits are 5×7 in the Minecraft bitmap font (including '1', which has a
# 5px-wide bottom serif).  Character widths come from actual bitmap size.
_CHAR_H     = 7
_CHAR_SEP   = 1    # 1px gap between characters
_SHADOW_RGB = (63, 63, 63)   # = white >> 2, Minecraft's exact shadow formula


# ── Asset cache ────────────────────────────────────────────────────────────────

class _Assets:
    def __init__(self):
        self._panels:       dict[str, Image.Image] = {}
        self._icons:        dict[int, Optional[Image.Image]] = {}
        self._placeholders: dict[str, Optional[Image.Image]] = {}
        self._recipe_btn:   Optional[Image.Image] = None
        self._hotbar:       Optional[Image.Image] = None
        self._offhand:      Optional[Image.Image] = None

        self._char_bitmaps: dict[str, Image.Image] = {}
        self._char_bitmaps = self._load_bitmap_font()

        self.valid_ids = [
            i for i in range(1, V.vocab_size())
            if i not in _EXCLUDED_IDS
            and (ICONS_DIR / f"{V.item_id_to_name(i)}.png").exists()
        ]

    def _load_bitmap_font(self) -> dict:
        """Extract digit glyphs from Minecraft's ascii.png bitmap font."""
        ascii_path = GUI_DIR / "ascii.png"
        if not ascii_path.exists():
            return {}
        sheet = np.array(Image.open(ascii_path).convert("RGBA"))
        CELL  = sheet.shape[1] // 16
        cache = {}
        for ch in '0123456789':
            code     = ord(ch)
            row, col = code // 16, code % 16
            cell     = sheet[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL]
            ys, xs   = np.where(cell[..., 3] > 10)
            if len(xs) == 0:
                cache[ch] = Image.new("RGBA", (5, _CHAR_H), (0, 0, 0, 0))
                continue
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            glyph   = Image.fromarray(cell[y0:y1, x0:x1])
            arr     = np.array(glyph)
            arr[arr[..., 3] >  10, :3] = 255
            arr[arr[..., 3] <= 10,  3] =   0
            cache[ch] = Image.fromarray(arr, "RGBA")
        return cache

    def recipe_book_btn(self) -> Optional[Image.Image]:
        if self._recipe_btn is None:
            full = Image.open(GUI_DIR / "inventory.png").convert("RGBA")
            self._recipe_btn = full.crop(RECIPE_ICON_CROP)
        return self._recipe_btn

    def panel(self, layout: Layout) -> Image.Image:
        if layout.gui_file not in self._panels:
            full  = Image.open(GUI_DIR / layout.gui_file).convert("RGBA")
            panel = full.crop((0, 0, layout.panel_w_gui, layout.panel_h_gui))
            panel = panel.resize(
                (layout.panel_w_gui * GUI_SCALE, layout.panel_h_gui * GUI_SCALE),
                Image.NEAREST,
            )
            self._panels[layout.gui_file] = panel
        return self._panels[layout.gui_file]

    def icon(self, item_id: int) -> Optional[Image.Image]:
        if item_id not in self._icons:
            name = V.item_id_to_name(item_id)
            p = ICONS_DIR / f"{name}.png"
            if p.exists():
                self._icons[item_id] = (
                    Image.open(p).convert("RGBA")
                    .resize((ICON_PX, ICON_PX), Image.LANCZOS)
                )
            else:
                self._icons[item_id] = None
        return self._icons[item_id]

    def armor_placeholder(self, slot_type: str) -> Optional[Image.Image]:
        if slot_type not in self._placeholders:
            fname = _ARMOR_PLACEHOLDER_FILES.get(slot_type, "")
            p = GUI_DIR / fname
            if p.exists():
                self._placeholders[slot_type] = (
                    Image.open(p).convert("RGBA").resize((ICON_PX, ICON_PX), Image.LANCZOS)
                )
            else:
                self._placeholders[slot_type] = None
        return self._placeholders[slot_type]

    def hotbar_widget(self) -> Image.Image:
        if self._hotbar is None:
            img = Image.open(GUI_DIR / "widgets.png").convert("RGBA")
            self._hotbar = img.crop((0, 0, 182, 22))
        return self._hotbar

    def offhand_widget(self) -> Image.Image:
        if self._offhand is None:
            img = Image.open(GUI_DIR / "widgets.png").convert("RGBA")
            self._offhand = img.crop((48, 22, 96, 46))
        return self._offhand

    def draw_count(self, canvas: Image.Image, sx: int, sy: int, count: int,
                   slot_px: int = SLOT_PX):
        """Render stack count anchored to bottom-right of slot."""
        if not self._char_bitmaps:
            return
        text    = str(count)
        bitmaps = [self._char_bitmaps.get(ch) for ch in text]
        widths  = [bm.size[0] if bm is not None else 5 for bm in bitmaps]

        anchor_x = sx + slot_px - 2 + TEXT_ICON_OFFSET
        anchor_y = sy + slot_px - 2 + TEXT_ICON_OFFSET
        paste_y  = anchor_y - _CHAR_H + 1

        positions = []
        cx = anchor_x
        for w in reversed(widths):
            positions.insert(0, cx - w + 1)
            cx = cx - w + 1 - _CHAR_SEP - 1

        for bm, px in zip(bitmaps, positions):
            if bm is None:
                continue
            arr = np.array(bm)
            arr[..., :3] = _SHADOW_RGB
            canvas.paste(Image.fromarray(arr, "RGBA"), (px + 1, paste_y + 1),
                         Image.fromarray(arr, "RGBA"))
            canvas.paste(bm, (px, paste_y), bm)


_ASSETS = _Assets()
_HAS_RECIPE_BOOK = True   # False for versions < 1.12 (recipe book added in 1.12)


def set_icons_dir(path) -> None:
    """Switch the icon set (e.g. '1.16.4' or '1.9') and rebuild the asset cache."""
    global ICONS_DIR, _ASSETS, _HAS_RECIPE_BOOK
    ICONS_DIR = SYNTH_DIR / "icons" / path if not Path(path).is_absolute() else Path(path)
    _ASSETS   = _Assets()
    try:
        parts = [int(x) for x in Path(path).name.split(".")]
        _HAS_RECIPE_BOOK = (parts[0] > 1) or (len(parts) > 1 and parts[1] >= 12)
    except ValueError:
        _HAS_RECIPE_BOOK = True


# ── Background generation ──────────────────────────────────────────────────────

def _random_bg(rng: random.Random) -> Image.Image:
    """Random-shape background: compressible but visually diverse."""
    r = lambda: rng.randint(0, 255)
    color = lambda: (r(), r(), r())

    canvas = Image.new("RGBA", (IMG_W, IMG_H), color())
    draw   = ImageDraw.Draw(canvas)

    n_shapes = rng.randint(10, 40)
    for _ in range(n_shapes):
        x0 = rng.randint(-IMG_W // 4, IMG_W)
        y0 = rng.randint(-IMG_H // 4, IMG_H)
        w  = rng.randint(20, IMG_W // 2)
        h  = rng.randint(20, IMG_H // 2)
        x1, y1 = x0 + w, y0 + h
        c  = color()
        kind = rng.randint(0, 2)
        if kind == 0:
            draw.rectangle([x0, y0, x1, y1], fill=c)
        elif kind == 1:
            draw.ellipse([x0, y0, x1, y1], fill=c)
        else:
            # polygon (triangle or quad)
            pts = [(rng.randint(0, IMG_W), rng.randint(0, IMG_H))
                   for _ in range(rng.randint(3, 5))]
            draw.polygon(pts, fill=c)

    return canvas


# ── Image generation ───────────────────────────────────────────────────────────

def generate_sample(
    layout_name: Optional[str] = None,
    empty_prob:  float = 0.25,
    mask_frac:   float = 0.04,
    rng: random.Random = None,
    show_hotbar: bool = True,
) -> tuple:
    """
    Returns (image, layout_name,
             player_ids[46], player_counts[46],
             container_ids[N], container_counts[N]).

    player_ids indexed by Minecraft protocol slot ID (0-45); 0 = empty.
    Slots not visible in this layout remain 0 in player_ids/player_counts.
    container_ids indexed 0..N-1 per container slot order; N=0 for survival.
    """
    if rng is None:
        rng = random.Random()
    if layout_name is None:
        layout_name = rng.choices(_LAYOUT_NAMES, weights=_LAYOUT_WEIGHTS)[0]

    layout      = LAYOUTS[layout_name]
    recipe_open = layout_name == "survival_recipe"

    valid = _ASSETS.valid_ids

    # ── Phase 1: Decide all items (no drawing yet) ────────────────────────────

    # Container slots
    container_ids    = []
    container_counts = []
    for c_idx in range(len(layout.container_slots)):
        if c_idx in layout.empty_container_slots or not valid or rng.random() < empty_prob:
            container_ids.append(0)
            container_counts.append(0)
        else:
            container_ids.append(rng.choice(valid))
            container_counts.append(rng.randint(1, 64))

    # Player slots
    player_ids    = [0] * N_PLAYER_SLOTS
    player_counts = [0] * N_PLAYER_SLOTS

    # Crafting area: 2/3 empty | 1/6 random junk (no result) | 1/6 valid recipe
    craft_override: dict[int, tuple[int, int]] = {}
    if 0 in layout.player_slot_coords and valid:
        roll = rng.random()
        if roll < 1 / 6:
            result = _CRAFTING.random_recipe(rng, set(valid))
            if result:
                grid, r_id, r_count = result
                craft_override[0] = (r_id, r_count)
                for i, iid in enumerate(grid):
                    if iid:
                        craft_override[1 + i] = (iid, rng.randint(1, 64))
        elif roll < 2 / 6:
            for pid in range(1, 5):
                if rng.random() > 0.4:
                    craft_override[pid] = (rng.choice(valid), rng.randint(1, 64))

    for proto_id in layout.player_slot_coords:
        if proto_id in _CRAFTING_PROTOS:
            iid, count = craft_override.get(proto_id, (0, 0))
        elif not valid or rng.random() < empty_prob:
            iid, count = 0, 0
        else:
            iid   = rng.choice(valid)
            count = rng.randint(1, 64)
        player_ids[proto_id]    = iid
        player_counts[proto_id] = count

    # ── Phase 2: Draw ─────────────────────────────────────────────────────────

    # Shape background (compressible, diverse).
    canvas = _random_bg(rng)

    if show_hotbar:
        # Hotbar widget at fixed position.
        hotbar = _ASSETS.hotbar_widget()
        canvas.paste(hotbar, (HOTBAR_X, HOTBAR_Y), hotbar)
        for i in range(9):
            iid   = player_ids[36 + i]
            count = player_counts[36 + i]
            if iid:
                sx = HOTBAR_X + 1 + i * HOTBAR_SLOT_W
                sy = HOTBAR_Y + 1
                icon = _ASSETS.icon(iid)
                if icon is not None:
                    canvas.paste(icon, (sx + HOTBAR_SLOT_INSET, sy + HOTBAR_SLOT_INSET), icon)
                if count > 1:
                    _ASSETS.draw_count(canvas, sx, sy, count, slot_px=HOTBAR_SLOT_W)

        # Offhand widget (only when offhand slot has an item).
        if player_ids[45]:
            offhand = _ASSETS.offhand_widget()
            canvas.paste(offhand, (OFFHAND_X, OFFHAND_Y), offhand)
            icon = _ASSETS.icon(player_ids[45])
            if icon is not None:
                canvas.paste(icon, (OFFHAND_X + OFFHAND_ICON_OFF, OFFHAND_Y + OFFHAND_ICON_Y), icon)

    # Darken background (Minecraft dims the world when any inventory is open).
    arr = np.array(canvas)
    arr[:, :, :3] = (arr[:, :, :3] * (1.0 - INVENTORY_DIM)).astype(np.uint8)
    canvas = Image.fromarray(arr, "RGBA")

    # Panel position.
    panel_w = layout.panel_w_gui * GUI_SCALE
    panel_h = layout.panel_h_gui * GUI_SCALE
    base_x  = RECIPE_OPEN_PANEL_X if recipe_open else (IMG_W - panel_w) // 2
    panel_x = base_x + _PANEL_NUDGE_X
    panel_y = (IMG_H - panel_h) // 2 + _PANEL_NUDGE_Y

    # 1/3 chance: place GUI at a random fully-on-screen position.
    if rng.random() < 1 / 3:
        max_x = IMG_W - panel_w
        max_y = IMG_H - panel_h
        if max_x >= 0 and max_y >= 0:
            panel_x = rng.randint(0, max_x)
            panel_y = rng.randint(0, max_y)

    # GUI panel.
    canvas.paste(_ASSETS.panel(layout), (panel_x, panel_y), _ASSETS.panel(layout))

    # Recipe book button (survival only, added in 1.12).
    if _HAS_RECIPE_BOOK and layout_name in ("survival", "survival_recipe"):
        btn = _ASSETS.recipe_book_btn()
        if btn:
            canvas.paste(btn,
                         (panel_x + RECIPE_BTN_X_GUI * GUI_SCALE,
                          panel_y + RECIPE_BTN_Y_GUI * GUI_SCALE), btn)

    def _paste_item(sx, sy, iid, count):
        icon = _ASSETS.icon(iid)
        if icon is not None:
            canvas.paste(icon,
                         (sx + SLOT_ICON_INSET + SLOT_ICON_OFFSET,
                          sy + SLOT_ICON_INSET + SLOT_ICON_OFFSET), icon)
        if count > 1:
            _ASSETS.draw_count(canvas, sx, sy, count)

    # Container items.
    for c_idx, (cx, cy) in enumerate(layout.container_slots):
        if container_ids[c_idx]:
            _paste_item(panel_x + cx, panel_y + cy,
                        container_ids[c_idx], container_counts[c_idx])

    # Player items (inside the GUI).
    for proto_id, (px_rel, py_rel) in layout.player_slot_coords.items():
        sx, sy     = panel_x + px_rel, panel_y + py_rel
        iid        = player_ids[proto_id]
        count      = player_counts[proto_id]
        armor_type = _ARMOR_SLOT_TYPE.get(proto_id)
        if iid:
            _paste_item(sx, sy, iid, count)
        elif armor_type:
            ph = _ASSETS.armor_placeholder(armor_type)
            if ph:
                canvas.paste(ph,
                             (sx + SLOT_ICON_INSET + SLOT_ICON_OFFSET,
                              sy + SLOT_ICON_INSET + SLOT_ICON_OFFSET), ph)

    # Random pixel masking (cursor / tooltip occlusion).
    if mask_frac > 0:
        arr  = np.array(canvas)
        n    = int(IMG_H * IMG_W * mask_frac)
        rows = np.random.randint(0, IMG_H, n)
        cols = np.random.randint(0, IMG_W, n)
        arr[rows, cols, :3] = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        canvas = Image.fromarray(arr)

    return (canvas.convert("RGB"), layout_name,
            player_ids, player_counts,
            container_ids, container_counts)


# ── IterableDataset ────────────────────────────────────────────────────────────

class SyntheticInventoryDataset(IterableDataset):
    """Infinite stream of synthetic inventory screenshots with per-slot labels.

    Each sample is a dict:
      image            : PIL RGB image (640×360)
      layout           : str layout name
      player_ids       : LongTensor[46]  — protocol slot IDs 0-45 (0 = empty)
      player_counts    : LongTensor[46]  — stack counts
      container_ids    : LongTensor[N]   — container slot items (N varies by layout)
      container_counts : LongTensor[N]
    """
    def __init__(self, layout_name=None, empty_prob=0.25, mask_frac=0.04, seed=42):
        self.layout_name = layout_name
        self.empty_prob  = empty_prob
        self.mask_frac   = mask_frac
        self.seed        = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = self.seed + (info.id if info else 0)
        rng  = random.Random(seed)
        while True:
            img, lname, p_ids, p_counts, c_ids, c_counts = generate_sample(
                layout_name=self.layout_name,
                empty_prob=self.empty_prob,
                mask_frac=self.mask_frac,
                rng=rng,
            )
            yield {
                "image":            img,
                "layout":           lname,
                "player_ids":       torch.tensor(p_ids,    dtype=torch.long),
                "player_counts":    torch.tensor(p_counts, dtype=torch.long),
                "container_ids":    torch.tensor(c_ids,    dtype=torch.long),
                "container_counts": torch.tensor(c_counts, dtype=torch.long),
            }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int, default=4)
    p.add_argument("--out",    default="test_samples")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--layout", default=None, choices=list(LAYOUTS) + [None])
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Valid icons: {len(_ASSETS.valid_ids)} / {V.vocab_size()-1}")
    rng = random.Random(args.seed)
    for i in range(args.n):
        img, lname, p_ids, p_counts, c_ids, c_counts = generate_sample(
            layout_name=args.layout, rng=rng)
        path = Path(args.out) / f"sample_{i:03d}_{lname}.png"
        img.save(path)

        filled_p = [(pid, V.item_id_to_name(iid), cnt)
                    for pid, (iid, cnt) in enumerate(zip(p_ids, p_counts)) if iid]
        filled_c = [(ci, V.item_id_to_name(iid), cnt)
                    for ci, (iid, cnt) in enumerate(zip(c_ids, c_counts)) if iid]

        print(f"\n[{i}] {path}  layout={lname}")
        if filled_p:
            print(f"  player ({len(filled_p)} slots):")
            for pid, name, cnt in filled_p[:6]:
                print(f"    proto_id={pid:>2}: {name:<28} x{cnt}")
            if len(filled_p) > 6:
                print(f"    ... +{len(filled_p)-6} more")
        if filled_c:
            print(f"  container ({len(filled_c)} slots):")
            for ci, name, cnt in filled_c[:4]:
                print(f"    slot {ci}: {name:<28} x{cnt}")
            if len(filled_c) > 4:
                print(f"    ... +{len(filled_c)-4} more")
