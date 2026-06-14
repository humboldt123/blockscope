package blockscope.converter;

import java.util.HashMap;
import java.util.Map;

/** Remaps pre-1.11 unnamespaced block entity IDs to 1.19.4 namespaced IDs. */
public class BlockEntityMapper {

    private static final Map<String, String> MAP = new HashMap<>();

    static {
        MAP.put("Chest",         "minecraft:chest");
        MAP.put("LargeChest",    "minecraft:chest");
        MAP.put("TrappedChest",  "minecraft:trapped_chest");
        MAP.put("Furnace",       "minecraft:furnace");
        MAP.put("Sign",          "minecraft:sign");
        MAP.put("MobSpawner",    "minecraft:mob_spawner");
        MAP.put("Music",         "minecraft:note_block");
        MAP.put("Trap",          "minecraft:dispenser");
        MAP.put("Hopper",        "minecraft:hopper");
        MAP.put("Comparator",    "minecraft:comparator");
        MAP.put("FlowerPot",     "minecraft:flower_pot");
        MAP.put("Beacon",        "minecraft:beacon");
        MAP.put("Skull",         "minecraft:skull");
        MAP.put("DaylightDetector", "minecraft:daylight_detector");
        MAP.put("Banner",        "minecraft:banner");
        MAP.put("EnchantTable",  "minecraft:enchanting_table");
        MAP.put("Cauldron",      "minecraft:brewing_stand");
        MAP.put("UNKNOWN",       "minecraft:air");
    }

    public static String remap(String oldId) {
        String remapped = MAP.get(oldId);
        if (remapped != null) return remapped;
        // Convert "SomeId" style to "minecraft:some_id" as a best-effort guess
        String snake = oldId.replaceAll("([A-Z])", "_$1").toLowerCase().replaceFirst("^_", "");
        return "minecraft:" + snake;
    }
}
