#!/usr/bin/env python3
"""
gen_worldgen_datapack.py — generate a beta-style Minecraft 1.19.4 worldgen datapack.

Extracts vanilla 1.19.4 biome/noise/structure JSON from the minecraft jar and
produces a self-contained datapack with:
  - Plain oak-tree overworld: plains, forest, river, beach, ocean, desert, swamp
  - Snowy/taiga/jungle/badlands/mushroom biomes redirected to plain equivalents
  - No deepslate, no aquifers, no ore veins
  - Granite/diorite/andesite/copper/tuff ores stripped; classic ores kept
  - Glow lichen, seagrass, kelp, coral, amethyst, sculk stripped
  - Oak trees only (birch_and_oak replaced with trees_plains)
  - Villages: plains only; mineshafts kept; all modern structures disabled
  - World height y=0 (bedrock) to y=255

Usage:
    cd <repo_root>
    python3 scripts/gen_worldgen_datapack.py

Output: scripts/beta_world.zip — drop into <world>/datapacks/ and reload/create world.
"""

import copy
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JAR_PATH  = REPO_ROOT / "visualizer" / ".cache" / "minecraft-1.19.4.jar"
OUT_ZIP   = Path(__file__).resolve().parent / "beta_world.zip"

# ---------------------------------------------------------------------------
# Overworld biomes to redirect: source_biome → target_biome (copy target's JSON)
# The biome source still generates these areas; we just swap their contents.
# ---------------------------------------------------------------------------
BIOME_REDIRECTS = {
    # Badlands family → desert
    "badlands":               "desert",
    "eroded_badlands":        "desert",
    "wooded_badlands":        "desert",
    # Jungle family → forest
    "bamboo_jungle":          "forest",
    "jungle":                 "forest",
    "sparse_jungle":          "forest",
    # Birch / dark / conifer family → forest
    "birch_forest":           "forest",
    "old_growth_birch_forest":"forest",
    "flower_forest":          "forest",
    "dark_forest":            "forest",
    "grove":                  "forest",
    "old_growth_pine_taiga":  "forest",
    "old_growth_spruce_taiga":"forest",
    "taiga":                  "forest",
    "snowy_taiga":            "forest",
    # Snowy / mushroom / savanna / meadow → plains
    "snowy_plains":           "plains",
    "ice_spikes":             "plains",
    "meadow":                 "plains",
    "sunflower_plains":       "plains",
    "mushroom_fields":        "plains",
    "savanna":                "plains",
    "savanna_plateau":        "plains",
    "deep_dark":              "plains",
    # Mountain peaks → windswept_hills
    "frozen_peaks":           "windswept_hills",
    "jagged_peaks":           "windswept_hills",
    "snowy_slopes":           "windswept_hills",
    "stony_peaks":            "windswept_hills",
    "windswept_forest":       "windswept_hills",
    "windswept_gravelly_hills":"windswept_hills",
    "windswept_savanna":      "windswept_hills",
    # Rivers / beaches / oceans normalised
    "frozen_river":           "river",
    "snowy_beach":            "beach",
    "stony_shore":            "beach",
    "cold_ocean":             "ocean",
    "frozen_ocean":           "ocean",
    "lukewarm_ocean":         "ocean",
    "warm_ocean":             "ocean",
    "deep_cold_ocean":        "deep_ocean",
    "deep_frozen_ocean":      "deep_ocean",
    "deep_lukewarm_ocean":    "deep_ocean",
    # Swamp / mangrove
    "mangrove_swamp":         "swamp",
    # Cave biomes (underground only — use surface counterpart for surface features)
    "dripstone_caves":        "plains",
    "lush_caves":             "forest",
}

# Features stripped from every biome regardless of phase
STRIP_FEATURES = {
    # Modern stone variants
    "minecraft:ore_granite_upper",
    "minecraft:ore_granite_lower",
    "minecraft:ore_diorite_upper",
    "minecraft:ore_diorite_lower",
    "minecraft:ore_andesite_upper",
    "minecraft:ore_andesite_lower",
    "minecraft:ore_tuff",
    "minecraft:ore_copper",
    # Modern vegetation
    "minecraft:glow_lichen",
    "minecraft:underwater_magma",
    # Seagrass / kelp / coral
    "minecraft:seagrass_cold",
    "minecraft:seagrass_normal",
    "minecraft:seagrass_warm",
    "minecraft:seagrass_river",
    "minecraft:seagrass_swamp",
    "minecraft:seagrass_deep",
    "minecraft:seagrass_simple",
    "minecraft:seagrass_slightly_less_seagrass",
    "minecraft:seagrass_tall",
    "minecraft:cold_kelp",
    "minecraft:kelp_cold",
    "minecraft:kelp",
    "minecraft:sea_pickle",
    "minecraft:coral_tree_warm",
    "minecraft:coral_mushroom_warm",
    "minecraft:coral_fan_warm",
    "minecraft:coral_claw_warm",
    # Amethyst / dripstone / lush cave features
    "minecraft:amethyst_geode",
    "minecraft:pointed_dripstone_ceiling",
    "minecraft:pointed_dripstone_floor",
    "minecraft:dripstone_clump",
    "minecraft:large_dripstone",
    "minecraft:small_dripstone",
    "minecraft:cave_vine",
    "minecraft:cave_vine_in_cave",
    "minecraft:spore_blossom",
    "minecraft:moss_patch",
    "minecraft:moss_patch_ceiling",
    # Sculk
    "minecraft:sculk_vein",
    "minecraft:sculk_patch_deep_dark",
    "minecraft:sculk_patch_ancient_city",
    "minecraft:sculk_patch",
    # Nether-like
    "minecraft:patch_fire",
    "minecraft:patch_soul_fire",
    # Bamboo
    "minecraft:bamboo_light",
    "minecraft:bamboo",
}

# Phases emptied entirely for all biomes
EMPTY_PHASES = {2, 10}  # 2=amethyst_geode  10=freeze_top_layer

# Tree feature replacements (None = drop entirely)
TREE_REPLACEMENTS = {
    "minecraft:trees_birch_and_oak":          "minecraft:trees_plains",
    "minecraft:trees_birch":                  "minecraft:trees_plains",
    "minecraft:trees_taiga":                  "minecraft:trees_plains",
    "minecraft:trees_old_growth_pine_taiga":  "minecraft:trees_plains",
    "minecraft:trees_old_growth_spruce_taiga":"minecraft:trees_plains",
    "minecraft:trees_grove":                  "minecraft:trees_plains",
    "minecraft:trees_meadow":                 "minecraft:trees_plains",
    "minecraft:trees_windswept_hills":        "minecraft:trees_plains",
    "minecraft:trees_savanna":                "minecraft:trees_plains",
    "minecraft:trees_jungle":                 "minecraft:trees_plains",
    "minecraft:trees_sparse_jungle":          "minecraft:trees_plains",
    "minecraft:trees_bamboo_jungle":          "minecraft:trees_plains",
    "minecraft:trees_snowy":                  None,   # snowy biomes → plains, no trees needed
}

# Structure sets to disable (empty structures list)
DISABLE_STRUCTURE_SETS = {
    "ancient_cities",
    "buried_treasures",
    "desert_pyramids",
    "end_cities",
    "igloos",
    "jungle_temples",
    "nether_complexes",
    "nether_fossils",
    "ocean_monuments",
    "ocean_ruins",
    "pillager_outposts",
    "ruined_portals",
    "shipwrecks",
    "strongholds",
    "swamp_huts",
    "woodland_mansions",
}

# ---------------------------------------------------------------------------
# Overworld biome names (skip nether/end when iterating)
# ---------------------------------------------------------------------------
NETHER_END_BIOMES = {
    "basalt_deltas", "crimson_forest", "nether_wastes", "soul_sand_valley", "warped_forest",
    "end_barrens", "end_highlands", "end_midlands", "small_end_islands", "the_end", "the_void",
}


def clean_biome(biome: dict) -> dict:
    """Strip/replace features in a biome JSON in-place copy."""
    b = copy.deepcopy(biome)
    features = b.get("features", [])
    new_features = []
    for i, phase in enumerate(features):
        if i in EMPTY_PHASES:
            new_features.append([])
            continue
        new_phase = []
        for feat in phase:
            if feat in STRIP_FEATURES:
                continue
            if feat in TREE_REPLACEMENTS:
                replacement = TREE_REPLACEMENTS[feat]
                if replacement is not None:
                    new_phase.append(replacement)
            else:
                new_phase.append(feat)
        new_features.append(new_phase)
    b["features"] = new_features
    return b


def make_noise_settings(vanilla: dict) -> dict:
    """No aquifers, no ore veins, no deepslate, classic y=0..255 world height."""
    ns = copy.deepcopy(vanilla)
    ns["aquifers_enabled"]  = False
    ns["ore_veins_enabled"] = False

    # Classic world height: bedrock at y=0, max y=255
    ns["noise"]["min_y"]  = 0
    ns["noise"]["height"] = 256

    # Surface rule: drop entry [2] which is the deepslate vertical_gradient
    sr = ns["surface_rule"]
    if sr.get("type") == "minecraft:sequence":
        sr["sequence"] = sr["sequence"][:2]  # keep [0]=bedrock floor, [1]=surface layers

    return ns


def disable_structure_set(vanilla: dict) -> dict:
    """Return a copy with an empty structures list so nothing spawns."""
    ss = copy.deepcopy(vanilla)
    ss["structures"] = []
    return ss


def villages_plains_only(vanilla: dict) -> dict:
    """Keep only village_plains in the villages structure set."""
    ss = copy.deepcopy(vanilla)
    ss["structures"] = [s for s in ss["structures"] if s["structure"] == "minecraft:village_plains"]
    return ss


def mineshafts_no_mesa(vanilla: dict) -> dict:
    """Remove mineshaft_mesa (mesa = badlands = modern feel) from mineshafts."""
    ss = copy.deepcopy(vanilla)
    ss["structures"] = [s for s in ss["structures"] if s["structure"] != "minecraft:mineshaft_mesa"]
    return ss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not JAR_PATH.exists():
        raise FileNotFoundError(f"Minecraft jar not found at {JAR_PATH}")

    print(f"Reading vanilla data from {JAR_PATH}")

    with zipfile.ZipFile(JAR_PATH) as jar:
        # --- load all overworld biomes ---
        biome_files = [
            n for n in jar.namelist()
            if n.startswith("data/minecraft/worldgen/biome/") and n.endswith(".json")
        ]
        vanilla_biomes = {}
        for path in biome_files:
            name = path.split("/")[-1].replace(".json", "")
            vanilla_biomes[name] = json.loads(jar.read(path))

        # --- load noise settings ---
        vanilla_noise = json.loads(jar.read("data/minecraft/worldgen/noise_settings/overworld.json"))

        # --- load structure sets ---
        ss_files = [
            n for n in jar.namelist()
            if n.startswith("data/minecraft/worldgen/structure_set/") and n.endswith(".json")
        ]
        vanilla_ss = {}
        for path in ss_files:
            name = path.split("/")[-1].replace(".json", "")
            vanilla_ss[name] = json.loads(jar.read(path))

    # --- resolve redirects: build a map biome_name → clean JSON ---
    # First, clean all "target" biomes that are kept as-is
    kept_biomes = set(vanilla_biomes.keys()) - set(BIOME_REDIRECTS.keys()) - NETHER_END_BIOMES
    cleaned: dict[str, dict] = {}
    for name in kept_biomes:
        cleaned[name] = clean_biome(vanilla_biomes[name])

    # Then, resolve redirects (copy cleaned target)
    for src, tgt in BIOME_REDIRECTS.items():
        if src in NETHER_END_BIOMES:
            continue
        if tgt not in cleaned:
            # Target itself might not be in cleaned yet (e.g. deep_ocean)
            if tgt in vanilla_biomes:
                cleaned[tgt] = clean_biome(vanilla_biomes[tgt])
            else:
                print(f"  WARNING: redirect target '{tgt}' not in jar, skipping {src}")
                continue
        cleaned[src] = copy.deepcopy(cleaned[tgt])

    # --- build structure set overrides ---
    out_ss: dict[str, dict] = {}
    for name, data in vanilla_ss.items():
        if name in DISABLE_STRUCTURE_SETS:
            out_ss[name] = disable_structure_set(data)
        elif name == "villages":
            out_ss[name] = villages_plains_only(data)
        elif name == "mineshafts":
            out_ss[name] = mineshafts_no_mesa(data)
        else:
            out_ss[name] = copy.deepcopy(data)

    # --- build noise settings ---
    out_noise = make_noise_settings(vanilla_noise)

    # --- write zip ---
    print(f"Writing datapack to {OUT_ZIP}")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        # pack.mcmeta
        pack_meta = {
            "pack": {
                "pack_format": 12,
                "description": "Beta-style worldgen: plains/forest/ocean/desert/swamp, oak trees, no deepslate, no modern blocks"
            }
        }
        zout.writestr("pack.mcmeta", json.dumps(pack_meta, indent=2))

        # biomes
        for name, data in cleaned.items():
            zout.writestr(
                f"data/minecraft/worldgen/biome/{name}.json",
                json.dumps(data, indent=2),
            )

        # noise settings
        zout.writestr(
            "data/minecraft/worldgen/noise_settings/overworld.json",
            json.dumps(out_noise, indent=2),
        )

        # structure sets
        for name, data in out_ss.items():
            zout.writestr(
                f"data/minecraft/worldgen/structure_set/{name}.json",
                json.dumps(data, indent=2),
            )

    print(f"Done. Drop {OUT_ZIP.name} into <world>/datapacks/ and create a new world.")
    print(f"  Biomes written : {len(cleaned)}")
    print(f"  Structure sets : {len(out_ss)}  ({len(DISABLE_STRUCTURE_SETS)} disabled)")


if __name__ == "__main__":
    main()
