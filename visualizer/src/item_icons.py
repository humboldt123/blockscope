"""
Load pre-rendered item icons from synth.
"""

from pathlib import Path
from PIL import Image

ICONS_DIR = Path(__file__).parent.parent / "item_icons"
ICON_SIZE = 32  # pixels

# Cache loaded icons
_icon_cache = {}


def get_item_icon(item_id: str, size: int = ICON_SIZE) -> Image.Image | None:
    """
    Load a pre-rendered item icon.
    Args:
        item_id: Item ID like 'minecraft:diamond_sword' or 'diamond_sword'
        size: Output size in pixels
    Returns:
        PIL RGBA image or None if not found
    """
    cache_key = (item_id, size)
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    # Strip minecraft: namespace
    item_name = item_id.split(":")[-1]

    # Load pre-rendered icon
    icon_path = ICONS_DIR / f"{item_name}.png"
    if not icon_path.exists():
        print(f"[ItemIcons] Missing icon: {item_name}")
        _icon_cache[cache_key] = None
        return None

    try:
        img = Image.open(icon_path).convert("RGBA")
        if img.size != (size, size):
            img = img.resize((size, size), Image.NEAREST)
        _icon_cache[cache_key] = img
        return img
    except Exception as e:
        print(f"[ItemIcons] Failed to load {item_name}: {e}")
        _icon_cache[cache_key] = None
        return None
