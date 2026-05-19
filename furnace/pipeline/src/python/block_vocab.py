"""
block_vocab.py — canonical block token vocabulary for the Blockscope pipeline.

Token 0  = UNKNOWN / air (air, cave_air, void_air all collapse here)
Tokens 1+ = every known block, sorted alphabetically for determinism

NATURAL  = blocks that appear in the world without player action
DERIVED  = blocks reachable via crafting / smelting / placing by a player
KNOWN    = NATURAL | DERIVED  (~284 tokens)
UNKNOWN  = everything else → token 0  (deepslate, copper, nether/end, 1.20+, …)

SCATTER_BLOCKS = solid, item-holdable subset used for void-world scatter generation
                 and void_scatter session inventory. Kept in sync with the Java
                 Material list in VoidScatterGenerator.java.
"""

# ── Natural blocks (exist in world without player) ──────────────────────────

NATURAL = frozenset({
    # Air (collapsed to token 0 — listed here for completeness)
    "air", "cave_air", "void_air",

    # Stone
    "stone", "cobblestone",

    # Dirt family
    "dirt", "grass_block", "coarse_dirt", "mycelium", "dirt_path", "farmland",

    # Sand / gravel / clay
    "sand", "gravel", "clay",

    # Oak wood
    "oak_log", "oak_wood", "oak_leaves", "oak_sapling",

    # Ores & mineral blocks (gold_block / diamond_block appear in custom structures)
    "coal_ore", "iron_ore", "gold_ore", "gold_block",
    "diamond_ore", "diamond_block",
    "lapis_ore", "redstone_ore", "emerald_ore",

    # Water / lava / ice
    "water", "lava", "obsidian", "ice", "packed_ice", "snow", "snow_block",

    # Vegetation
    "grass", "tall_grass", "fern", "large_fern", "dead_bush",
    "vine", "lily_pad", "sugar_cane", "cactus",

    # Flowers
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet",
    "red_tulip", "white_tulip", "orange_tulip", "pink_tulip",
    "oxeye_daisy", "cornflower", "lily_of_the_valley", "sunflower",

    # Mushrooms
    "brown_mushroom", "red_mushroom",

    # Crops (growing forms — not the item forms)
    "wheat", "pumpkin", "melon",
    "pumpkin_stem", "melon_stem", "attached_pumpkin_stem", "attached_melon_stem",

    # Wool (from sheep)
    "white_wool", "orange_wool", "magenta_wool", "light_blue_wool", "yellow_wool",
    "lime_wool", "pink_wool", "gray_wool", "light_gray_wool", "cyan_wool",
    "purple_wool", "blue_wool", "brown_wool", "green_wool", "red_wool", "black_wool",

    # Beds
    "white_bed", "orange_bed", "magenta_bed", "light_blue_bed", "yellow_bed",
    "lime_bed", "pink_bed", "gray_bed", "light_gray_bed", "cyan_bed",
    "purple_bed", "blue_bed", "brown_bed", "green_bed", "red_bed", "black_bed",

    # Naturally-occurring structures / dungeon blocks
    "mossy_cobblestone", "cobweb", "spawner", "rail",  # rail from mineshafts
    "fire", "bedrock",

    # Herobrine shrine (intentionally placed netherrack)
    "netherrack",
})

# ── Derived blocks (crafted / smelted / placed by player) ───────────────────

DERIVED = frozenset({
    # Stone family
    "stone_slab", "stone_stairs",
    "stone_bricks", "stone_brick_slab", "stone_brick_stairs", "stone_brick_wall",
    "cracked_stone_bricks", "mossy_stone_bricks",
    "cobblestone_slab", "cobblestone_stairs", "cobblestone_wall",
    "mossy_cobblestone_wall",
    "smooth_stone", "smooth_stone_slab",
    "infested_stone", "infested_cobblestone",

    # Sandstone (from sand)
    "sandstone", "cut_sandstone", "chiseled_sandstone",
    "sandstone_slab", "sandstone_stairs", "sandstone_wall",
    "smooth_sandstone", "smooth_sandstone_slab", "smooth_sandstone_stairs",
    "glass", "glass_pane",

    # Terracotta (from clay)
    "terracotta",
    "white_terracotta", "orange_terracotta", "magenta_terracotta", "light_blue_terracotta",
    "yellow_terracotta", "lime_terracotta", "pink_terracotta", "gray_terracotta",
    "light_gray_terracotta", "cyan_terracotta", "purple_terracotta", "blue_terracotta",
    "brown_terracotta", "green_terracotta", "red_terracotta", "black_terracotta",

    # Bricks (from clay balls → bricks → brick blocks)
    "bricks", "brick_slab", "brick_stairs", "brick_wall",

    # Oak derived
    "oak_planks", "oak_slab", "oak_stairs", "oak_door", "oak_trapdoor",
    "oak_fence", "oak_fence_gate", "oak_button", "oak_pressure_plate",
    "oak_sign", "oak_wall_sign",
    "stripped_oak_log", "stripped_oak_wood",
    "crafting_table", "bookshelf", "chest", "trapped_chest", "jukebox", "note_block",

    # Coal derived
    "coal_block", "torch", "wall_torch", "campfire",

    # Iron derived
    "iron_block", "iron_door", "iron_trapdoor", "iron_bars",
    "heavy_weighted_pressure_plate",
    "anvil", "chipped_anvil", "damaged_anvil",
    "powered_rail", "activator_rail", "detector_rail",
    "cauldron", "lantern",

    # Gold derived
    "light_weighted_pressure_plate",

    # Diamond derived
    "enchanting_table",

    # Redstone derived
    "redstone_wire", "redstone_torch", "redstone_wall_torch",
    "redstone_block", "redstone_lamp",
    "piston", "sticky_piston", "piston_head",
    "observer", "repeater", "comparator",
    "dispenser", "dropper", "hopper",
    "lever", "stone_button", "stone_pressure_plate",
    "daylight_detector", "target", "tripwire", "tripwire_hook",

    # Lapis derived
    "lapis_block",

    # Emerald derived
    "emerald_block",

    # Mushroom blocks (bonemeal on mushroom)
    "brown_mushroom_block", "red_mushroom_block", "mushroom_stem",

    # Carpet (from wool)
    "white_carpet", "orange_carpet", "magenta_carpet", "light_blue_carpet", "yellow_carpet",
    "lime_carpet", "pink_carpet", "gray_carpet", "light_gray_carpet", "cyan_carpet",
    "purple_carpet", "blue_carpet", "brown_carpet", "green_carpet", "red_carpet", "black_carpet",

    # Stained glass (from glass + dye)
    "white_stained_glass", "orange_stained_glass", "magenta_stained_glass",
    "light_blue_stained_glass", "yellow_stained_glass", "lime_stained_glass",
    "pink_stained_glass", "gray_stained_glass", "light_gray_stained_glass",
    "cyan_stained_glass", "purple_stained_glass", "blue_stained_glass",
    "brown_stained_glass", "green_stained_glass", "red_stained_glass", "black_stained_glass",
    "white_stained_glass_pane", "orange_stained_glass_pane", "magenta_stained_glass_pane",
    "light_blue_stained_glass_pane", "yellow_stained_glass_pane", "lime_stained_glass_pane",
    "pink_stained_glass_pane", "gray_stained_glass_pane", "light_gray_stained_glass_pane",
    "cyan_stained_glass_pane", "purple_stained_glass_pane", "blue_stained_glass_pane",
    "brown_stained_glass_pane", "green_stained_glass_pane", "red_stained_glass_pane",
    "black_stained_glass_pane",

    # Functional / crafted
    "furnace", "ladder", "brewing_stand", "hay_block", "tnt",
    "cake", "sponge", "wet_sponge", "flower_pot",
})

# ── Combined vocabulary ──────────────────────────────────────────────────────

KNOWN = NATURAL | DERIVED

# Token 0 = unknown/air; tokens 1..N = known non-air blocks (alphabetical)
_AIR = {"air", "cave_air", "void_air"}
_TOKENED = sorted(KNOWN - _AIR)
BLOCK_TO_ID: dict[str, int] = {b: i + 1 for i, b in enumerate(_TOKENED)}
# Air variants all map to 0
for _a in _AIR:
    BLOCK_TO_ID[_a] = 0
ID_TO_BLOCK: dict[int, str] = {v: k for k, v in BLOCK_TO_ID.items() if v != 0}
ID_TO_BLOCK[0] = "air"

UNKNOWN_ID = 0
VOCAB_SIZE  = len(_TOKENED) + 1  # +1 for token-0 slot


def block_to_id(name: str) -> int:
    """Return token ID for a block name string, 0 if unknown."""
    return BLOCK_TO_ID.get(name, UNKNOWN_ID)


# ── Scatter blocks ───────────────────────────────────────────────────────────
# Solid blocks with corresponding items, used for void_scatter world generation
# and player inventory. Kept in sync with VoidScatterGenerator.SCATTER_MATERIALS.

SCATTER_BLOCKS = frozenset({
    # Terrain
    "stone", "cobblestone", "dirt", "grass_block", "sand", "gravel", "clay",
    "sandstone", "mossy_cobblestone", "stone_bricks", "bricks", "obsidian",
    "coarse_dirt", "packed_ice", "snow_block",

    # Ores
    "coal_ore", "iron_ore", "gold_ore", "diamond_ore",
    "lapis_ore", "redstone_ore", "emerald_ore",

    # Mineral blocks
    "coal_block", "iron_block", "gold_block", "diamond_block",
    "lapis_block", "emerald_block",

    # Wood
    "oak_log", "oak_planks", "oak_leaves", "bookshelf",

    # Wool (all 16)
    "white_wool", "orange_wool", "magenta_wool", "light_blue_wool", "yellow_wool",
    "lime_wool", "pink_wool", "gray_wool", "light_gray_wool", "cyan_wool",
    "purple_wool", "blue_wool", "brown_wool", "green_wool", "red_wool", "black_wool",

    # Terracotta (all 17)
    "terracotta",
    "white_terracotta", "orange_terracotta", "magenta_terracotta", "light_blue_terracotta",
    "yellow_terracotta", "lime_terracotta", "pink_terracotta", "gray_terracotta",
    "light_gray_terracotta", "cyan_terracotta", "purple_terracotta", "blue_terracotta",
    "brown_terracotta", "green_terracotta", "red_terracotta", "black_terracotta",

    # Organic / structural
    "hay_block", "pumpkin", "melon",
    "brown_mushroom_block", "red_mushroom_block", "mushroom_stem",

    # Glass
    "glass",
    "white_stained_glass", "orange_stained_glass", "magenta_stained_glass",
    "light_blue_stained_glass", "yellow_stained_glass", "lime_stained_glass",
    "pink_stained_glass", "gray_stained_glass", "cyan_stained_glass",
    "purple_stained_glass", "blue_stained_glass", "brown_stained_glass",
    "green_stained_glass", "red_stained_glass", "black_stained_glass",

    # Functional
    "crafting_table", "chest", "furnace",

    # Rare
    "netherrack",
})

assert SCATTER_BLOCKS <= KNOWN, "SCATTER_BLOCKS contains blocks not in KNOWN"
