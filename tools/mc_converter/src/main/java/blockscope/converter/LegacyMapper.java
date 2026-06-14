package blockscope.converter;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

/**
 * Loads WorldEdit's legacy.json and maps (blockId:meta) → namespaced block state string.
 * Format: {"blocks": {"1:0": "minecraft:stone", "2:0": "minecraft:grass_block[snowy=false]", ...}}
 */
public class LegacyMapper {

    private final Map<String, String> blocks;

    private LegacyMapper(Map<String, String> blocks) {
        this.blocks = blocks;
    }

    public static LegacyMapper load() throws Exception {
        try (InputStream in = LegacyMapper.class.getResourceAsStream("/legacy.json")) {
            if (in == null) throw new IllegalStateException(
                "legacy.json not found in classpath. Run `./gradlew downloadLegacyJson` first.");
            Gson gson = new Gson();
            Type mapType = new TypeToken<Map<String, Map<String, String>>>() {}.getType();
            Map<String, Map<String, String>> root = gson.fromJson(new InputStreamReader(in), mapType);
            return new LegacyMapper(root.getOrDefault("blocks", new HashMap<>()));
        }
    }

    /** Returns the 1.13+ block state string for a given 1.8 block ID + metadata. */
    public String getBlockState(int blockId, int meta) {
        String key = blockId + ":" + meta;
        String state = blocks.get(key);
        if (state != null) return state;
        // Fallback: try meta=0
        state = blocks.get(blockId + ":0");
        if (state != null) return state;
        // Last resort
        if (blockId == 0) return "minecraft:air";
        System.err.println("Unknown block " + blockId + ":" + meta + " → air");
        return "minecraft:air";
    }
}
