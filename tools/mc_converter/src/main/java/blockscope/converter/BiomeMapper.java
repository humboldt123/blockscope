package blockscope.converter;

import java.util.HashMap;
import java.util.Map;

/** Maps legacy 1.8 biome IDs (0–255) to 1.19.4 resource locations. */
public class BiomeMapper {

    private static final Map<Integer, String> MAP = new HashMap<>();

    static {
        MAP.put(0,   "minecraft:ocean");
        MAP.put(1,   "minecraft:plains");
        MAP.put(2,   "minecraft:desert");
        MAP.put(3,   "minecraft:windswept_hills");
        MAP.put(4,   "minecraft:forest");
        MAP.put(5,   "minecraft:taiga");
        MAP.put(6,   "minecraft:swamp");
        MAP.put(7,   "minecraft:river");
        MAP.put(8,   "minecraft:nether_wastes");
        MAP.put(9,   "minecraft:the_end");
        MAP.put(10,  "minecraft:frozen_ocean");
        MAP.put(11,  "minecraft:frozen_river");
        MAP.put(12,  "minecraft:snowy_plains");
        MAP.put(13,  "minecraft:snowy_mountains");  // removed in 1.18; treated as snowy_plains
        MAP.put(14,  "minecraft:mushroom_fields");
        MAP.put(15,  "minecraft:mushroom_field_shore"); // removed; treat as mushroom_fields
        MAP.put(16,  "minecraft:beach");
        MAP.put(17,  "minecraft:desert_hills"); // removed; treat as desert
        MAP.put(18,  "minecraft:wooded_hills"); // removed; treat as forest
        MAP.put(19,  "minecraft:taiga_hills");  // removed; treat as taiga
        MAP.put(20,  "minecraft:mountain_edge"); // removed
        MAP.put(21,  "minecraft:jungle");
        MAP.put(22,  "minecraft:jungle_hills"); // removed; treat as jungle
        MAP.put(23,  "minecraft:sparse_jungle");
        MAP.put(24,  "minecraft:deep_ocean");
        MAP.put(25,  "minecraft:stony_shore");
        MAP.put(26,  "minecraft:snowy_beach");
        MAP.put(27,  "minecraft:birch_forest");
        MAP.put(28,  "minecraft:birch_forest_hills"); // removed; treat as birch_forest
        MAP.put(29,  "minecraft:dark_forest");
        MAP.put(30,  "minecraft:snowy_taiga");
        MAP.put(31,  "minecraft:snowy_taiga_hills"); // removed; treat as snowy_taiga
        MAP.put(32,  "minecraft:old_growth_pine_taiga");
        MAP.put(33,  "minecraft:old_growth_spruce_taiga");
        MAP.put(34,  "minecraft:windswept_forest");
        MAP.put(35,  "minecraft:savanna");
        MAP.put(36,  "minecraft:savanna_plateau");
        MAP.put(37,  "minecraft:badlands");
        MAP.put(38,  "minecraft:wooded_badlands");
        MAP.put(39,  "minecraft:badlands_plateau"); // removed; treat as badlands
        MAP.put(127, "minecraft:the_void");
        MAP.put(129, "minecraft:sunflower_plains");
        MAP.put(130, "minecraft:desert_lakes"); // removed; treat as desert
        MAP.put(131, "minecraft:windswept_gravelly_hills");
        MAP.put(132, "minecraft:flower_forest");
        MAP.put(133, "minecraft:taiga_mountains"); // removed; treat as taiga
        MAP.put(134, "minecraft:swamp_hills"); // removed; treat as swamp
        MAP.put(140, "minecraft:ice_spikes");
        MAP.put(149, "minecraft:modified_jungle"); // removed; treat as jungle
        MAP.put(151, "minecraft:modified_jungle_edge"); // removed
        MAP.put(155, "minecraft:old_growth_birch_forest");
        MAP.put(156, "minecraft:tall_birch_hills"); // removed
        MAP.put(157, "minecraft:dark_forest_hills"); // removed; treat as dark_forest
        MAP.put(158, "minecraft:snowy_taiga_mountains"); // removed
        MAP.put(160, "minecraft:old_growth_pine_taiga");
        MAP.put(161, "minecraft:old_growth_spruce_taiga");
        MAP.put(162, "minecraft:windswept_forest");
        MAP.put(163, "minecraft:shattered_savanna");
        MAP.put(164, "minecraft:shattered_savanna_plateau"); // removed
        MAP.put(165, "minecraft:eroded_badlands");
        MAP.put(166, "minecraft:modified_wooded_badlands_plateau"); // removed
        MAP.put(167, "minecraft:modified_badlands_plateau"); // removed
    }

    public static String get(int id) {
        return MAP.getOrDefault(id, "minecraft:plains");
    }
}
