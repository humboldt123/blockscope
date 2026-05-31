"""
synth/download_1.9_icons.py — Download 1.9 item icons from minecraft-ids.grahamedgecombe.com
and map them to canonical 1.19.2 names.

Usage:
    python synth/download_1.9_icons.py

Output:
    synth/icons/1.9/{canonical_name}.png   — icons for matched items
    synth/icons/1.9/_unmatched.tsv         — items we couldn't auto-map (review manually)
"""

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
import vocab as V

OUT_DIR      = Path(__file__).parent / "icons" / "1.9"
API_JSON     = "https://minecraft-ids.grahamedgecombe.com/items.json"
ICONS_ZIP    = "https://minecraft-ids.grahamedgecombe.com/items.zip"

# ── Name normalization ────────────────────────────────────────────────────────

# Display-name fragments that differ between the 1.9 site names and 1.19.2
# canonical names.  Applied in order (first match wins).
_SUBSTITUTIONS = [
    # Wool colors — "Blue Wool" → "blue_wool" works fine, no subs needed
    # Tool/weapon material names changed in some cases:
    ("gold ",        "golden "),
    ("gold_",        "golden_"),
    # "Wooden X" — already correct
    # Specific renames:
    # Food renames
    ("raw beef",     "beef"),
    ("steak",        "cooked_beef"),
    ("raw chicken",  "chicken"),
    ("cooked chicken", "cooked_chicken"),
    ("raw rabbit",   "rabbit"),
    ("raw mutton",   "mutton"),
    # Item renames
    ("eye of ender", "ender_eye"),
    ("glistering melon", "glistering_melon_slice"),
    ("redstone repeater", "repeater"),
    ("redstone comparator", "comparator"),
    ("nether quartz", "quartz"),
    ("book and quill", "writable_book"),
    ("bottle o enchanting", "experience_bottle"),
    ("dragon breath", "dragon_breath"),
    ("empty map",    "map"),
    ("rabbit foot",  "rabbit_foot"),
    ("minecart with tnt",     "tnt_minecart"),
    ("minecart with hopper",  "hopper_minecart"),
    ("minecart with command block", "command_block_minecart"),
    ("moss stone",   "mossy_cobblestone"),
    ("banner",       "white_banner"),
    # Block name fixes
    ("grass ",       "grass_block "),  # "Grass" → "grass_block"
    ("nether brick", "nether_bricks"),
    ("ender stone",  "end_stone"),
    ("web",          "cobweb"),
    ("snow layer",   "snow"),
    ("snow block",   "snow_block"),
    ("red rose",     "poppy"),
    ("yellow flower","dandelion"),
    ("pumpkin lantern","jack_o_lantern"),
    ("workbench",    "crafting_table"),
    ("stone bricks", "stone_bricks"),
    ("huge brown mushroom", "brown_mushroom_block"),
    ("huge red mushroom",   "red_mushroom_block"),
    ("iron bars",    "iron_bars"),
    ("lily pad",     "lily_pad"),
    ("nether wart block", "nether_wart_block"),
    ("brick block",  "bricks"),
    ("bookshelf",    "bookshelf"),
    ("mossy cobblestone", "mossy_cobblestone"),
    ("glowstone",    "glowstone"),
    ("stone pressure plate", "stone_pressure_plate"),
    ("wooden pressure plate", "oak_pressure_plate"),
    ("redstone ore", "redstone_ore"),
    ("ice",          "ice"),
    ("clay block",   "clay"),
    ("sugar canes",  "sugar_cane"),
    ("jukebox",      "jukebox"),
    ("fence",        "oak_fence"),          # bare "fence" → oak_fence
    ("pumpkin",      "pumpkin"),
    ("netherrack",   "netherrack"),
    ("soul sand",    "soul_sand"),
    ("glowing redstone ore", "redstone_ore"),
    ("stone slab",   "stone_slab"),
    ("brick slab",   "brick_slab"),
    ("stone brick slab", "stone_brick_slab"),
    ("wooden slab",  "oak_slab"),
    ("cobblestone slab", "cobblestone_slab"),
    ("sandstone slab",   "sandstone_slab"),
    ("nether brick slab","nether_brick_slab"),
    ("torch",        "torch"),
    ("ladder",       "ladder"),
    ("lever",        "lever"),
    ("redstone torch (off)", "redstone_torch"),
    ("redstone torch (on)",  "redstone_torch"),
    ("button",       "stone_button"),
    ("wooden button","oak_button"),
    ("trapdoor",     "oak_trapdoor"),
    ("iron trapdoor","iron_trapdoor"),
    ("wooden door",  "oak_door"),
    ("birch door",   "birch_door"),
    ("spruce door",  "spruce_door"),
    ("jungle door",  "jungle_door"),
    ("acacia door",  "acacia_door"),
    ("dark oak door","dark_oak_door"),
    ("iron door",    "iron_door"),
    ("sign",         "oak_sign"),
    ("oak wood plank", "oak_planks"),
    ("spruce wood plank", "spruce_planks"),
    ("birch wood plank",  "birch_planks"),
    ("jungle wood plank", "jungle_planks"),
    ("acacia wood plank", "acacia_planks"),
    ("dark oak wood plank", "dark_oak_planks"),
    ("oak wood",     "oak_log"),
    ("spruce wood",  "spruce_log"),
    ("birch wood",   "birch_log"),
    ("jungle wood",  "jungle_log"),
    ("acacia wood",  "acacia_log"),
    ("dark oak wood","dark_oak_log"),
    ("sapling",      "oak_sapling"),
    ("leaves",       "oak_leaves"),
    # Ores / mineral blocks
    ("lapis lazuli ore",   "lapis_ore"),
    ("lapis lazuli block", "lapis_block"),
    ("block of coal",      "coal_block"),
    ("pillar quartz block","quartz_pillar"),
    # Renamed blocks
    ("dead shrub",         "dead_bush"),
    ("melon block",        "melon"),
    ("vines",              "vine"),
    ("enchantment table",  "enchanting_table"),
    ("redstone lamp (inactive)", "redstone_lamp"),
    ("redstone lamp (active)",   "redstone_lamp"),
    ("hay bale",           "hay_block"),
    ("grass path",         "dirt_path"),
    ("cake block",         "cake"),
    ("daylight sensor",    "daylight_detector"),
    ("weighted pressure plate (light)", "light_weighted_pressure_plate"),
    ("weighted pressure plate (heavy)", "heavy_weighted_pressure_plate"),
    # Hardened clay → terracotta (1.12→1.13 rename)
    (" hardened clay",     "_terracotta"),  # "white hardened clay" → "white_terracotta"
    ("hardened clay",      "terracotta"),
    # Wood stairs/slabs: "X wood stairs" → "X_stairs", "X wood slab" → "X_slab"
    (" wood stairs",       "_stairs"),
    (" wood slab",         "_slab"),
    ("wooden trapdoor",    "oak_trapdoor"),
    ("wooden button",      "oak_button"),
    # Doors (block form → item name)
    ("oak door block",     "oak_door"),
    ("spruce door block",  "spruce_door"),
    ("birch door block",   "birch_door"),
    ("jungle door block",  "jungle_door"),
    ("acacia door block",  "acacia_door"),
    ("dark oak door block","dark_oak_door"),
    ("iron door block",    "iron_door"),
    # Food / item renames (post-1.13)
    ("raw porkchop",       "porkchop"),
    ("raw fish",           "cod"),
    ("raw salmon",         "salmon"),
    ("clownfish",          "tropical_fish"),
    ("cooked fish",        "cooked_cod"),
    ("slimeball",          "slime_ball"),
    ("ink sack",           "ink_sac"),
    ("rose red",           "red_dye"),
    ("cactus green",       "green_dye"),
    ("coco beans",         "cocoa_beans"),
    ("dandelion yellow",   "yellow_dye"),
    ("leather tunic",      "leather_chestplate"),
    ("leather pants",      "leather_leggings"),
    # Minecarts
    ("minecart with chest",   "chest_minecart"),
    ("minecart with furnace", "furnace_minecart"),
    # Apostrophe-normalized names (apostrophes stripped before subs are applied)
    ("jack olantern",      "jack_o_lantern"),
    ("rabbits foot",       "rabbit_foot"),
    ("dragons breath",     "dragon_breath"),
    ("bottle o enchanting","experience_bottle"),
    # Infested stone (Monster Egg blocks)
    ("stone monster egg",             "infested_stone"),
    ("cobblestone monster egg",       "infested_cobblestone"),
    ("stone brick monster egg",       "infested_stone_bricks"),
    ("mossy stone brick monster egg", "infested_mossy_stone_bricks"),
    ("cracked stone brick monster egg", "infested_cracked_stone_bricks"),
    ("chiseled stone brick monster egg","infested_chiseled_stone_bricks"),
    # Bed — default was red in pre-1.12
    ("bed",                "red_bed"),
]


def _canonical(display_name: str) -> list[str]:
    """Return candidate canonical names to try (in order) for a display name."""
    raw    = display_name.strip().lower()
    normed = raw.replace("'", "")   # strip apostrophes once; used for subs + direct
    candidates = []

    # Spawn eggs: "Spawn X" → "x_spawn_egg"
    if normed.startswith("spawn "):
        mob = normed[6:].replace(" ", "_")
        candidates.append(f"{mob}_spawn_egg")
        candidates.append(mob.replace("zombie_pigman", "zombified_piglin") + "_spawn_egg")
        candidates.append(mob.replace("ocelot", "cat") + "_spawn_egg")

    # Music discs: "13 Disc" → "music_disc_13"
    if normed.endswith(" disc"):
        label = normed[:-5].strip().replace(" ", "_")
        candidates.append(f"music_disc_{label}")

    # Mob heads: "Mob Head (X)" → "x_skull" / "x_head"
    if normed.startswith("mob head ("):
        mob = normed[10:-1]
        mob_map = {
            "skeleton": "skeleton_skull",  "wither skeleton": "wither_skeleton_skull",
            "zombie":   "zombie_head",     "human":           "player_head",
            "creeper":  "creeper_head",    "dragon":          "dragon_head",
        }
        if mob in mob_map:
            candidates.append(mob_map[mob])

    # Direct: replace spaces with underscores
    candidates.append(normed.replace(" ", "_"))

    # Try substitutions (applied on apostrophe-stripped name)
    subbed = normed
    for old, new in _SUBSTITUTIONS:
        if old in subbed:
            subbed = subbed.replace(old, new, 1)
    if subbed != normed:
        candidates.append(subbed.replace(" ", "_"))

    return candidates


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_names = set(V.item_id_to_name(i) for i in range(1, V.vocab_size()))
    canonical_names.discard(None)

    # ── Download metadata ─────────────────────────────────────────────────────
    print("Fetching items.json …")
    with urllib.request.urlopen(API_JSON) as r:
        items = json.load(r)
    print(f"  {len(items)} items")

    # ── Download icons zip ────────────────────────────────────────────────────
    print("Downloading items.zip …")
    with urllib.request.urlopen(ICONS_ZIP) as r:
        zip_data = r.read()
    print(f"  {len(zip_data)//1024} KB")

    # ── Map and extract ───────────────────────────────────────────────────────
    matched   = 0
    skipped   = 0
    unmatched = []  # (type, meta, display_name, tried_names)

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        zip_names = set(zf.namelist())

        for item in items:
            t, m, name = item["type"], item["meta"], item["name"]
            png_key = f"{t}-{m}.png"
            if png_key not in zip_names:
                skipped += 1
                continue

            candidates = _canonical(name)
            found = next((c for c in candidates if c in canonical_names), None)

            if found:
                dest = OUT_DIR / f"{found}.png"
                if not dest.exists():           # don't overwrite (keep meta=0 as primary)
                    dest.write_bytes(zf.read(png_key))
                matched += 1
            else:
                unmatched.append((t, m, name, candidates))

    print(f"\nMatched:   {matched}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Skipped (no png): {skipped}")

    # Write unmatched report
    report = OUT_DIR / "_unmatched.tsv"
    with open(report, "w") as f:
        f.write("type\tmeta\tdisplay_name\ttried\n")
        for t, m, name, tried in sorted(unmatched):
            f.write(f"{t}\t{m}\t{name}\t{', '.join(tried)}\n")
    print(f"\nUnmatched items written to {report}")
    print("Review and add substitutions to _SUBSTITUTIONS if needed.")


if __name__ == "__main__":
    main()
