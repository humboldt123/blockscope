#!/usr/bin/env python3
"""
gen_worldgen_datapack.py — single beta-style Minecraft 1.19.4 worldgen datapack.

Sources:
  - minecraft-1.19.4.jar      vanilla biome/noise/structure JSON
  - OldVillages_1.0.1.zip     classic village structures (old_villages namespace)
  - Cascades-v1.0-1.19.4.zip  terrain density-function + noise + biome-source overrides

Produces scripts/beta_world.zip with:
  - Simple overworld: plains/forest/river/beach/ocean/desert/swamp/windswept_hills
  - Biome source overridden (via dimension/overworld.json) so snowy/frozen/peak
    biomes never appear — remapped at noise-source level using Cascades' parameter list
  - No deepslate, no aquifers, no ore veins, world height y=0..255
  - Classic ores kept; granite/diorite/andesite/copper/tuff stripped
  - Oak trees only, small (no fancy oak, no bees); dense forests preserved
  - forest_flowers (lilacs/peonies) stripped; flower_plains → flower_default
  - Classic mob spawners: horses/donkeys/glow squids/foxes/bees etc. removed
  - Old-style village structures from OldVillages (terrain_adaptation: none)
  - Vanilla villages disabled; all modern structures disabled
  - Cascades terrain density functions + noise parameters for beta-like terrain

Usage:
    cd <repo_root>
    python3 scripts/gen_worldgen_datapack.py

Output: scripts/beta_world.zip — drag into the Data Packs screen when creating a new world.
"""

import copy
import json
import zipfile
from pathlib import Path

REPO_ROOT        = Path(__file__).resolve().parent.parent
JAR_PATH         = REPO_ROOT / "visualizer" / ".cache" / "minecraft-1.19.4.jar"
OLD_VILLAGES_ZIP = Path.home() / "Downloads" / "OldVillages_1.0.1.zip"
CASCADES_ZIP     = Path.home() / "Downloads" / "Cascades-v1.0-1.19.4.zip"
OUT_ZIP          = Path(__file__).resolve().parent / "beta_world.zip"

# ---------------------------------------------------------------------------
# Biome redirects applied to both biome JSON content AND the biome source
# source_name (no namespace) → target_name (no namespace), both minecraft:
# ---------------------------------------------------------------------------
BIOME_REDIRECTS = {
    "badlands": "desert", "eroded_badlands": "desert", "wooded_badlands": "desert",
    "bamboo_jungle": "forest", "jungle": "forest", "sparse_jungle": "forest",
    "birch_forest": "forest", "old_growth_birch_forest": "forest",
    "flower_forest": "forest", "dark_forest": "forest", "grove": "forest",
    "old_growth_pine_taiga": "forest", "old_growth_spruce_taiga": "forest",
    "taiga": "forest", "snowy_taiga": "forest",
    "snowy_plains": "plains", "ice_spikes": "plains", "meadow": "plains",
    "sunflower_plains": "plains", "mushroom_fields": "plains",
    "savanna": "plains", "savanna_plateau": "plains", "deep_dark": "plains",
    "frozen_peaks": "windswept_hills", "jagged_peaks": "windswept_hills",
    "snowy_slopes": "windswept_hills", "stony_peaks": "windswept_hills",
    "windswept_forest": "windswept_hills", "windswept_gravelly_hills": "windswept_hills",
    "windswept_savanna": "windswept_hills",
    "frozen_river": "river",
    "snowy_beach": "beach", "stony_shore": "beach",
    "cold_ocean": "ocean", "frozen_ocean": "ocean",
    "lukewarm_ocean": "ocean", "warm_ocean": "ocean",
    "deep_cold_ocean": "deep_ocean", "deep_frozen_ocean": "deep_ocean",
    "deep_lukewarm_ocean": "deep_ocean",
    "mangrove_swamp": "swamp",
    "dripstone_caves": "plains", "lush_caves": "forest",
}

# Cascades custom biomes → our equivalents (for biome source remapping)
CASCADES_BIOME_MAP = {
    "hybrid_beta:seasonal_forest":     "minecraft:forest",
    "hybrid_beta:rainforest":          "minecraft:forest",
    "hybrid_beta:autumnal_forest":     "minecraft:forest",
    "hybrid_beta:temperate_rainforest":"minecraft:forest",
}

NETHER_END_BIOMES = {
    "basalt_deltas", "crimson_forest", "nether_wastes", "soul_sand_valley", "warped_forest",
    "end_barrens", "end_highlands", "end_midlands", "small_end_islands", "the_end", "the_void",
}

# ---------------------------------------------------------------------------
# Feature stripping / replacement
# ---------------------------------------------------------------------------
STRIP_FEATURES = {
    "minecraft:ore_granite_upper", "minecraft:ore_granite_lower",
    "minecraft:ore_diorite_upper", "minecraft:ore_diorite_lower",
    "minecraft:ore_andesite_upper", "minecraft:ore_andesite_lower",
    "minecraft:ore_tuff", "minecraft:ore_copper",
    "minecraft:glow_lichen", "minecraft:underwater_magma",
    "minecraft:seagrass_cold", "minecraft:seagrass_normal", "minecraft:seagrass_warm",
    "minecraft:seagrass_river", "minecraft:seagrass_swamp", "minecraft:seagrass_deep",
    "minecraft:seagrass_simple", "minecraft:seagrass_slightly_less_seagrass",
    "minecraft:seagrass_tall", "minecraft:cold_kelp", "minecraft:kelp_cold", "minecraft:kelp",
    "minecraft:sea_pickle",
    "minecraft:coral_tree_warm", "minecraft:coral_mushroom_warm",
    "minecraft:coral_fan_warm", "minecraft:coral_claw_warm",
    "minecraft:amethyst_geode",
    "minecraft:pointed_dripstone_ceiling", "minecraft:pointed_dripstone_floor",
    "minecraft:dripstone_clump", "minecraft:large_dripstone", "minecraft:small_dripstone",
    "minecraft:cave_vine", "minecraft:cave_vine_in_cave",
    "minecraft:spore_blossom", "minecraft:moss_patch", "minecraft:moss_patch_ceiling",
    "minecraft:sculk_vein", "minecraft:sculk_patch_deep_dark",
    "minecraft:sculk_patch_ancient_city", "minecraft:sculk_patch",
    "minecraft:patch_fire", "minecraft:patch_soul_fire",
    "minecraft:bamboo_light", "minecraft:bamboo",
    "minecraft:forest_flowers",   # lilac, peony, rose bush, lily of the valley
}

EMPTY_PHASES = {2, 10}  # amethyst_geode, freeze_top_layer

# Tree placed-feature ID replacements (only exotic/conifer types — NOT birch_and_oak,
# which we keep for density and fix via configured_feature override instead)
TREE_REPLACEMENTS = {
    "minecraft:trees_birch":                  "minecraft:trees_birch_and_oak",  # keep density
    "minecraft:trees_taiga":                  "minecraft:trees_birch_and_oak",
    "minecraft:trees_old_growth_pine_taiga":  "minecraft:trees_birch_and_oak",
    "minecraft:trees_old_growth_spruce_taiga":"minecraft:trees_birch_and_oak",
    "minecraft:trees_grove":                  "minecraft:trees_birch_and_oak",
    "minecraft:trees_meadow":                 "minecraft:trees_plains",
    "minecraft:trees_windswept_hills":        "minecraft:trees_plains",
    "minecraft:trees_savanna":                "minecraft:trees_plains",
    "minecraft:trees_jungle":                 "minecraft:trees_birch_and_oak",
    "minecraft:trees_sparse_jungle":          "minecraft:trees_plains",
    "minecraft:trees_bamboo_jungle":          "minecraft:trees_birch_and_oak",
    "minecraft:trees_snowy":                  None,
}

FEATURE_REPLACEMENTS = {
    "minecraft:flower_plains": "minecraft:flower_default",
}

# ---------------------------------------------------------------------------
# Mob spawner stripping
# ---------------------------------------------------------------------------
STRIP_SPAWNERS = {
    "minecraft:horse", "minecraft:donkey", "minecraft:mule",
    "minecraft:goat", "minecraft:fox", "minecraft:panda", "minecraft:parrot",
    "minecraft:polar_bear", "minecraft:llama", "minecraft:trader_llama",
    "minecraft:bee", "minecraft:cat", "minecraft:frog", "minecraft:tadpole",
    "minecraft:allay", "minecraft:sniffer", "minecraft:camel", "minecraft:armadillo",
    "minecraft:glow_squid", "minecraft:axolotl",
    "minecraft:tropical_fish", "minecraft:pufferfish", "minecraft:cod",
    "minecraft:salmon", "minecraft:dolphin", "minecraft:turtle",
    "minecraft:pillager", "minecraft:ravager", "minecraft:evoker",
    "minecraft:vindicator", "minecraft:vex", "minecraft:illusioner",
    "minecraft:warden", "minecraft:phantom", "minecraft:wandering_trader",
}

# ---------------------------------------------------------------------------
# Structure sets to fully disable
# ---------------------------------------------------------------------------
DISABLE_STRUCTURE_SETS = {
    "ancient_cities", "buried_treasures", "desert_pyramids", "end_cities",
    "igloos", "jungle_temples", "nether_complexes", "nether_fossils",
    "ocean_monuments", "ocean_ruins", "pillager_outposts", "ruined_portals",
    "shipwrecks", "strongholds", "swamp_huts", "woodland_mansions",
    "villages",  # replaced by old_villages namespace
}

# Cascades files to copy verbatim (DFs + noise params); excludes biomes and
# noise_settings (we generate our own) and dimension (we generate our own)
CASCADES_INCLUDE_PREFIXES = (
    "data/minecraft/worldgen/density_function/",
    "data/hybrid_beta/worldgen/density_function/",
    "data/minecraft/worldgen/noise/",
    "data/hybrid_beta/worldgen/noise/",
)
CASCADES_EXCLUDE_EXACT = {
    "data/minecraft/worldgen/noise_settings/amplified.json",
    "data/minecraft/worldgen/noise_settings/caves.json",
    "data/minecraft/worldgen/noise_settings/floating_islands.json",
    "data/minecraft/worldgen/noise_settings/large_biomes.json",
    "data/minecraft/worldgen/noise_settings/overworld.json",
}

# Small oak configured_feature: no fancy oak, no bees
SMALL_OAK_CF = {
    "type": "minecraft:tree",
    "config": {
        "decorators": [],
        "dirt_provider": {"type": "minecraft:simple_state_provider",
                          "state": {"Name": "minecraft:dirt"}},
        "foliage_placer": {"type": "minecraft:blob_foliage_placer",
                           "height": 3, "offset": 0, "radius": 2},
        "foliage_provider": {"type": "minecraft:simple_state_provider", "state": {
            "Name": "minecraft:oak_leaves",
            "Properties": {"distance": "7", "persistent": "false", "waterlogged": "false"},
        }},
        "force_dirt": False,
        "ignore_vines": True,
        "minimum_size": {"type": "minecraft:two_layers_feature_size",
                         "limit": 1, "lower_size": 0, "upper_size": 1},
        "trunk_placer": {"type": "minecraft:straight_trunk_placer",
                         "base_height": 4, "height_rand_a": 2, "height_rand_b": 0},
        "trunk_provider": {"type": "minecraft:simple_state_provider", "state": {
            "Name": "minecraft:oak_log", "Properties": {"axis": "y"},
        }},
    },
}


# ---------------------------------------------------------------------------
# Biome cleaning
# ---------------------------------------------------------------------------

def clean_biome(biome: dict) -> dict:
    b = copy.deepcopy(biome)

    features = b.get("features", [])
    new_features = []
    for i, phase in enumerate(features):
        if i in EMPTY_PHASES:
            new_features.append([])
            continue
        new_phase = []
        seen = set()
        for feat in phase:
            if feat in STRIP_FEATURES:
                continue
            if feat in TREE_REPLACEMENTS:
                replacement = TREE_REPLACEMENTS[feat]
                if replacement is None:
                    continue
            elif feat in FEATURE_REPLACEMENTS:
                replacement = FEATURE_REPLACEMENTS[feat]
            else:
                replacement = feat
            if replacement not in seen:
                new_phase.append(replacement)
                seen.add(replacement)
        new_features.append(new_phase)
    b["features"] = new_features

    spawners = b.get("spawners", {})
    for cat in list(spawners.keys()):
        spawners[cat] = [e for e in spawners[cat] if e.get("type") not in STRIP_SPAWNERS]
    b["spawners"] = spawners

    return b


# ---------------------------------------------------------------------------
# Noise settings
# ---------------------------------------------------------------------------

def make_noise_settings(vanilla: dict, cascades_ns: dict) -> dict:
    ns = copy.deepcopy(vanilla)
    ns["aquifers_enabled"]  = False
    ns["ore_veins_enabled"] = False
    ns["noise"]["min_y"]    = 0
    ns["noise"]["height"]   = 256
    sr = ns["surface_rule"]
    if sr.get("type") == "minecraft:sequence":
        sr["sequence"] = sr["sequence"][:2]
    ns["noise_router"] = copy.deepcopy(cascades_ns["noise_router"])
    return ns


# ---------------------------------------------------------------------------
# Dimension / biome source override
# ---------------------------------------------------------------------------

def remap_biome_id(biome_id: str) -> str:
    """Apply BIOME_REDIRECTS and CASCADES_BIOME_MAP to a fully-qualified biome ID."""
    if biome_id in CASCADES_BIOME_MAP:
        return CASCADES_BIOME_MAP[biome_id]
    if biome_id.startswith("minecraft:"):
        short = biome_id[len("minecraft:"):]
        if short in BIOME_REDIRECTS:
            return f"minecraft:{BIOME_REDIRECTS[short]}"
    return biome_id


def make_dimension_override(cascades_dim: dict) -> dict:
    """
    Take Cascades' dimension/overworld.json (which has a full explicit multi_noise
    biome source with ~2000 entries) and remap every biome ID through our
    BIOME_REDIRECTS so snowy/frozen/peak/birch biomes never appear.
    """
    dim = copy.deepcopy(cascades_dim)
    bs = dim["generator"]["biome_source"]
    bs.pop("preset", None)
    bs["type"] = "minecraft:multi_noise"
    bs["biomes"] = [
        {**entry, "biome": remap_biome_id(entry["biome"])}
        for entry in bs.get("biomes", [])
    ]
    return dim


# ---------------------------------------------------------------------------
# Structure set helpers
# ---------------------------------------------------------------------------

def disable_ss(vanilla: dict) -> dict:
    ss = copy.deepcopy(vanilla)
    ss["structures"] = []
    return ss


def mineshafts_no_mesa(vanilla: dict) -> dict:
    ss = copy.deepcopy(vanilla)
    ss["structures"] = [s for s in ss["structures"] if s["structure"] != "minecraft:mineshaft_mesa"]
    return ss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p, label in [(JAR_PATH, "MC jar"), (OLD_VILLAGES_ZIP, "OldVillages zip"),
                     (CASCADES_ZIP, "Cascades zip")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found at {p}")

    print(f"Reading vanilla data from {JAR_PATH.name}")
    with zipfile.ZipFile(CASCADES_ZIP) as zca:
        cascades_noise = json.loads(zca.read("data/minecraft/worldgen/noise_settings/overworld.json"))
        cascades_dim   = json.loads(zca.read("data/minecraft/dimension/overworld.json"))

    with zipfile.ZipFile(JAR_PATH) as jar:
        vanilla_biomes = {
            p.split("/")[-1].replace(".json", ""): json.loads(jar.read(p))
            for p in jar.namelist()
            if p.startswith("data/minecraft/worldgen/biome/") and p.endswith(".json")
        }
        vanilla_noise = json.loads(jar.read("data/minecraft/worldgen/noise_settings/overworld.json"))
        vanilla_ss = {
            p.split("/")[-1].replace(".json", ""): json.loads(jar.read(p))
            for p in jar.namelist()
            if p.startswith("data/minecraft/worldgen/structure_set/") and p.endswith(".json")
        }

    # --- clean biomes ---
    kept = set(vanilla_biomes) - set(BIOME_REDIRECTS) - NETHER_END_BIOMES
    cleaned: dict[str, dict] = {name: clean_biome(vanilla_biomes[name]) for name in kept}
    for src, tgt in BIOME_REDIRECTS.items():
        if src in NETHER_END_BIOMES:
            continue
        if tgt not in cleaned and tgt in vanilla_biomes:
            cleaned[tgt] = clean_biome(vanilla_biomes[tgt])
        if tgt in cleaned:
            cleaned[src] = copy.deepcopy(cleaned[tgt])

    # --- structure sets ---
    out_ss = {}
    for name, data in vanilla_ss.items():
        if name in DISABLE_STRUCTURE_SETS:
            out_ss[name] = disable_ss(data)
        elif name == "mineshafts":
            out_ss[name] = mineshafts_no_mesa(data)
        else:
            out_ss[name] = copy.deepcopy(data)

    print(f"Writing datapack to {OUT_ZIP.name}")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:

        zout.writestr("pack.mcmeta", json.dumps({
            "pack": {
                "pack_format": 12,
                "description": "Beta-style worldgen: plain oak world, classic terrain, old villages",
            }
        }, indent=2))

        # biomes
        for name, data in cleaned.items():
            zout.writestr(f"data/minecraft/worldgen/biome/{name}.json",
                          json.dumps(data, indent=2))

        # noise settings
        zout.writestr("data/minecraft/worldgen/noise_settings/overworld.json",
                      json.dumps(make_noise_settings(vanilla_noise, cascades_noise), indent=2))

        # dimension override — replaces snowy/frozen/peak biomes in the noise source
        dim_override = make_dimension_override(cascades_dim)
        unique_biomes = set(e["biome"] for e in dim_override["generator"]["biome_source"]["biomes"])
        print(f"  Biome source: {len(dim_override['generator']['biome_source']['biomes'])} entries, "
              f"{len(unique_biomes)} unique biomes")
        zout.writestr("data/minecraft/dimension/overworld.json",
                      json.dumps(dim_override, indent=2))

        # structure sets
        for name, data in out_ss.items():
            zout.writestr(f"data/minecraft/worldgen/structure_set/{name}.json",
                          json.dumps(data, indent=2))

        # configured_feature overrides: both trees_plains AND trees_birch_and_oak → small oak
        # (birch_and_oak keeps its placed-feature density; only the tree type changes)
        for cf_name in ("trees_plains", "trees_birch_and_oak"):
            zout.writestr(f"data/minecraft/worldgen/configured_feature/{cf_name}.json",
                          json.dumps(SMALL_OAK_CF, indent=2))

        # OldVillages
        print(f"  Merging {OLD_VILLAGES_ZIP.name}")
        ov_count = 0
        with zipfile.ZipFile(OLD_VILLAGES_ZIP) as zov:
            for item in zov.infolist():
                if item.filename.startswith("data/old_villages/") and not item.filename.endswith("/"):
                    zout.writestr(item.filename, zov.read(item.filename))
                    ov_count += 1
        print(f"  OldVillages files merged: {ov_count}")

        # Cascades density functions + noise parameters
        print(f"  Merging {CASCADES_ZIP.name} terrain data")
        cascades_count = 0
        with zipfile.ZipFile(CASCADES_ZIP) as zca:
            for item in zca.infolist():
                if item.filename.endswith("/") or item.filename in CASCADES_EXCLUDE_EXACT:
                    continue
                if any(item.filename.startswith(pfx) for pfx in CASCADES_INCLUDE_PREFIXES):
                    zout.writestr(item.filename, zca.read(item.filename))
                    cascades_count += 1
        print(f"  Cascades files merged: {cascades_count}")

    print(f"\nDone → {OUT_ZIP}")
    print(f"  Biomes            : {len(cleaned)}")
    print(f"  Structure sets    : {len(out_ss)}  ({len(DISABLE_STRUCTURE_SETS)} disabled)")
    print(f"  OldVillages files : {ov_count}")
    print(f"  Cascades files    : {cascades_count}")
    print(f"  Unique biomes in source: {len(unique_biomes)}")


if __name__ == "__main__":
    main()
