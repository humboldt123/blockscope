"""
Generates 1.19.4 datapacks restricting world decoration to a period-accurate
block palette for Beta 1.7.3 or Release 1.5.2.

Strategy:
  Override CONFIGURED_FEATURE JSONs (not placed_features) with no_op.
  The placed_features still exist in biome feature lists with their original
  resource locations and step positions, so Paper 1.19.4's ordering graph
  is unchanged — no feature-order-cycle crashes.

  IMPORTANT: placed_feature names often differ from configured_feature names.
  All names here are configured_feature names (verified from minecraft-1.19.4.jar).

Usage:
    python generate.py --target beta   -> period_accurate_beta/
    python generate.py --target 1.5.2  -> period_accurate_152/
"""

import argparse, json, os, shutil, zipfile

# ---------------------------------------------------------------------------
# CONFIGURED_FEATURES TO DISABLE
# Names are CONFIGURED_FEATURE resource names (may differ from placed_feature names).
# Mismatches found from jar analysis:
#   placed ore_andesite_upper + ore_andesite_lower -> configured ore_andesite
#   placed ore_granite_upper/lower -> configured ore_granite
#   placed ore_diorite_upper/lower -> configured ore_diorite
#   placed ore_copper -> configured ore_copper_small
#   placed bamboo -> configured bamboo_some_podzol
#   placed bamboo_light -> configured bamboo_no_podzol
#   placed kelp_cold, kelp_warm -> configured kelp
#   placed seagrass_cold/normal/warm -> configured seagrass_short
#   placed seagrass_river -> configured seagrass_slightly_less_short
#   placed flower_warm -> configured flower_default
#   placed patch_tall_grass_2 -> configured patch_tall_grass
#   placed patch_berry_common -> configured patch_berry_bush
#   placed trees_meadow -> configured meadow_trees
#   placed trees_cherry -> configured cherry_bees_005
#   placed trees_badlands -> configured oak  (oak is period-accurate, NOT disabled)
#   placed trees_swamp -> configured swamp_oak (swamp oaks are period-accurate, NOT disabled)
# ---------------------------------------------------------------------------

DISABLE_FEATURES_BETA = [
    # ── 1.8 stone variants ────────────────────────────────────────────────────
    "minecraft:ore_andesite",        # placed as ore_andesite_upper + _lower
    "minecraft:ore_granite",
    "minecraft:ore_diorite",

    # ── 1.17+ underground ─────────────────────────────────────────────────────
    "minecraft:amethyst_geode",
    "minecraft:ore_copper_small",    # placed as ore_copper (small) + ore_copper_small
    "minecraft:ore_copper_large",    # placed as ore_copper_large
    "minecraft:ore_tuff",
    "minecraft:dripstone_cluster",
    "minecraft:pointed_dripstone",
    "minecraft:large_dripstone",
    "minecraft:cave_vine",
    "minecraft:cave_vine_in_cave",
    "minecraft:glow_lichen",
    "minecraft:moss_patch",
    "minecraft:moss_patch_bonemeal",
    "minecraft:rooted_azalea_tree",
    "minecraft:spore_blossom",
    "minecraft:hanging_roots",
    "minecraft:sculk_patch_ancient_city",
    "minecraft:sculk_patch_deep_dark",
    "minecraft:sculk_vein",
    "minecraft:freeze_top_layer",    # powder snow top layer (1.17)

    # ── 1.19 mangrove ─────────────────────────────────────────────────────────
    "minecraft:mangrove_fallen_log",
    "minecraft:frogspawn",
    "minecraft:mud_layer",

    # ── 1.13+ ocean ───────────────────────────────────────────────────────────
    "minecraft:kelp",                # placed as kelp_cold + kelp_warm
    "minecraft:sea_pickle",
    "minecraft:warm_ocean_vegetation",
    "minecraft:seagrass_short",             # placed as seagrass_cold/normal/warm
    "minecraft:seagrass_slightly_less_short",  # placed as seagrass_river
    "minecraft:seagrass_tall",
    "minecraft:coral_claw",
    "minecraft:coral_mushroom",
    "minecraft:coral_tree",
    "minecraft:coral_tree_inhabited",

    # ── 1.16 nether ───────────────────────────────────────────────────────────
    "minecraft:nether_sprouts",
    "minecraft:twisting_vines",
    "minecraft:twisting_vines_cave",
    "minecraft:weeping_vines",
    "minecraft:weeping_vines_cave",
    "minecraft:warped_forest_vegetation",
    "minecraft:crimson_forest_vegetation",
    "minecraft:nether_forest_vegetation",
    "minecraft:ore_ancient_debris_large",
    "minecraft:ore_ancient_debris_small",
    "minecraft:ore_nether_gold",
    "minecraft:ore_blackstone",
    "minecraft:ore_gravel_nether",
    "minecraft:ore_soul_sand",
    "minecraft:ore_magma",
    "minecraft:basalt_pillar",
    "minecraft:basalt_blobs",
    "minecraft:blackstone_blobs",
    "minecraft:delta",
    "minecraft:small_basalt_columns",
    "minecraft:large_basalt_columns",
    "minecraft:spring_lava_nether",
    "minecraft:patch_soul_sand_vegetation",
    "minecraft:patch_crimson_roots",
    "minecraft:patch_warped_roots",

    # ── Nether quartz — added in 1.5, NOT in beta ─────────────────────────────
    "minecraft:ore_quartz",

    # ── 1.14 bamboo ───────────────────────────────────────────────────────────
    "minecraft:bamboo_some_podzol",  # placed as bamboo
    "minecraft:bamboo_no_podzol",    # placed as bamboo_light
    "minecraft:bamboo_vegetation",

    # ── 1.14 sweet berry bushes ───────────────────────────────────────────────
    "minecraft:patch_berry_bush",    # placed as patch_berry_bush + patch_berry_common
    "minecraft:patch_berry_decorated",

    # ── 1.7.2 flowers (post-beta AND post-1.5.2) ─────────────────────────────
    "minecraft:flower_allium",
    "minecraft:flower_azure_bluet",
    "minecraft:flower_blue_orchid",
    "minecraft:flower_cornflower",
    "minecraft:flower_lily_of_the_valley",
    "minecraft:flower_oxeye_daisy",
    "minecraft:flower_tulip_orange",
    "minecraft:flower_tulip_pink",
    "minecraft:flower_tulip_red",
    "minecraft:flower_tulip_white",
    "minecraft:flower_wither_rose",
    "minecraft:flower_plain",
    "minecraft:flower_meadow",
    "minecraft:flower_swamp",
    "minecraft:flower_default",     # placed as flower_warm (warm biome flowers)

    # ── 1.7.2 double-tall plants ──────────────────────────────────────────────
    "minecraft:patch_sunflower",
    "minecraft:patch_lilac",
    "minecraft:patch_rose_bush",
    "minecraft:patch_peony",
    "minecraft:patch_large_fern",
    "minecraft:patch_tall_grass",    # placed as patch_tall_grass + patch_tall_grass_2

    # ── 1.7.2 acacia / new biome trees ───────────────────────────────────────
    "minecraft:trees_savanna",
    "minecraft:trees_water",
    "minecraft:meadow_trees",        # placed as trees_meadow (1.18 meadow biome)
    "minecraft:cherry_bees_005",     # placed as trees_cherry (1.20 cherry grove)
    "minecraft:trees_grove",

    # ── 1.7.2 jungle content ──────────────────────────────────────────────────
    "minecraft:trees_jungle",
    "minecraft:jungle_bush",
    "minecraft:patch_melon",

    # ── Vines (added Beta 1.3, present in 1.5.2 but NOT in pre-beta) ─────────
    "minecraft:vines",

    # NOT disabled (period-accurate for both beta and 1.5.2):
    #   swamp_oak (swamp oaks exist since beta)
    #   oak (placed by trees_badlands — oak trees are period-accurate)
    #   patch_waterlily (lily pads added Beta 1.4, before both target eras)
]

# 1.5.2 palette: same but re-enable quartz ore and vines (both exist in 1.5.2)
DISABLE_FEATURES_152 = [
    f for f in DISABLE_FEATURES_BETA
    if f not in ("minecraft:ore_quartz", "minecraft:vines")
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def make_no_op():
    # Override the CONFIGURED_FEATURE with no_op.
    # Placed_features still exist in biome feature lists unchanged —
    # the ordering graph is unaffected, so no cycle crashes.
    return {"type": "minecraft:no_op", "config": {}}

def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def generate_datapack(target, out_dir):
    assert target in ("beta", "1.5.2")
    label = "beta_1_7_3" if target == "beta" else "1_5_2"
    pack_dir = os.path.join(out_dir, f"period_accurate_{label}")

    if os.path.exists(pack_dir):
        shutil.rmtree(pack_dir)
    os.makedirs(pack_dir)

    disable = DISABLE_FEATURES_BETA if target == "beta" else DISABLE_FEATURES_152

    write_json(os.path.join(pack_dir, "pack.mcmeta"), {
        "pack": {"pack_format": 12, "description": f"Period-accurate block palette: {target}"}
    })

    # Override configured_features with no_op
    cfg_dir = os.path.join(pack_dir, "data", "minecraft", "worldgen", "configured_feature")
    for feature in disable:
        name = feature.split(":")[1]
        write_json(os.path.join(cfg_dir, f"{name}.json"), make_no_op())

    write_text(os.path.join(pack_dir, "README.txt"), f"""\
Period-accurate block palette: {target}  |  pack_format 12 (Minecraft 1.19.4)

Overrides CONFIGURED_FEATURES (not placed_features) with no_op.
This preserves the placed_feature ordering graph so Paper 1.19.4
does not throw feature-order-cycle crashes.

{len(disable)} configured_features disabled.

Disabled highlights:
  Ores:     andesite, granite, diorite (1.8+), copper, tuff (1.17+)
  Plants:   bamboo (1.14+), sweet berries (1.14+), 1.7.2+ flowers,
            tall plants (sunflower, lilac, etc.), kelp, seagrass (1.13+)
  Trees:    acacia (savanna), jungle content, meadow/grove/cherry trees
  Nether:   ancient debris, 1.16 nether vegetation, basalt features
  Ocean:    coral, sea pickle, warm ocean vegetation (1.13+)
  Misc:     amethyst geode, dripstone, sculk, freeze_top_layer (1.17+)
{"  Nether:   quartz ore disabled (added 1.5, not in beta)" if target == "beta" else "  Nether:   quartz ore KEPT (added 1.5, present in 1.5.2)"}
{"  Vines:    disabled (added Beta 1.3, not in pre-beta)" if target == "beta" else "  Vines:    KEPT (added Beta 1.3, present in 1.5.2)"}

NOT disabled (period-accurate):
  Oak trees, swamp oaks, lily pads, dandelions, poppies,
  mushrooms, sugar cane, pumpkins, cacti, vines (1.5.2 only)

Not handled (surface rules, not placed_features):
  Deepslate (y<0), podzol in mega-taiga, terracotta in mesa —
  these require noise_settings overrides or the Undeepslated datapack.
""")

    zip_path = os.path.join(out_dir, f"period_accurate_{label}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(pack_dir):
            for file in files:
                fp = os.path.join(root, file)
                zf.write(fp, os.path.relpath(fp, pack_dir))

    print(f"Written: {zip_path}  ({len(disable)} configured_features disabled)")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["beta", "1.5.2"], required=True)
    parser.add_argument("--out", default="./output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    generate_datapack(args.target, args.out)
