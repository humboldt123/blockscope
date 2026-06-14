package blockscope.converter;

import net.querz.nbt.tag.*;

import java.util.*;

public class ChunkConverter {

    private static final int TARGET_DATA_VERSION = 3337; // 1.19.4
    // 1.19.4 world height: sections Y = -4 .. 19 (24 sections total)
    private static final int MIN_SECTION_Y = -4;
    private static final int MAX_SECTION_Y = 19;
    private static final int TOTAL_SECTIONS = MAX_SECTION_Y - MIN_SECTION_Y + 1; // 24

    private final LegacyMapper mapper;

    public ChunkConverter(LegacyMapper mapper) {
        this.mapper = mapper;
    }

    public CompoundTag convert(CompoundTag oldData, int absoluteChunkX, int absoluteChunkZ) {
        CompoundTag level = oldData.getCompoundTag("Level");
        if (level == null) {
            // Already in newer format — set DataVersion and return as-is
            oldData.putInt("DataVersion", TARGET_DATA_VERSION);
            return oldData;
        }

        int chunkX = level.containsKey("xPos") ? level.getInt("xPos") : absoluteChunkX;
        int chunkZ = level.containsKey("zPos") ? level.getInt("zPos") : absoluteChunkZ;
        long lastUpdate   = level.getLong("LastUpdate");
        long inhabitedTime = level.getLong("InhabitedTime");

        // Sections
        ListTag<?> rawSections = level.getListTag("Sections");
        @SuppressWarnings("unchecked")
        ListTag<CompoundTag> oldSections = rawSections != null
            ? (ListTag<CompoundTag>) rawSections
            : new ListTag<>(CompoundTag.class);

        byte[] biomes = level.getByteArray("Biomes"); // 256 bytes, one per column

        ListTag<CompoundTag> newSections = convertSections(oldSections, biomes);

        // Block entities
        ListTag<?> rawTEs = level.getListTag("TileEntities");
        ListTag<CompoundTag> blockEntities = convertBlockEntities(rawTEs);

        // structures field: must have References and starts sub-compounds (even if empty)
        CompoundTag structuresTag = new CompoundTag();
        structuresTag.put("References", new CompoundTag());
        structuresTag.put("starts", new CompoundTag());

        // Build 1.19.4 root compound — no Level wrapper
        CompoundTag out = new CompoundTag();
        out.putInt("DataVersion", TARGET_DATA_VERSION);
        out.putInt("xPos", chunkX);
        out.putInt("zPos", chunkZ);
        out.putInt("yPos", MIN_SECTION_Y); // 1.19.4 minimum section Y = -4
        out.putString("Status", "features"); // DFU uses "features" not "minecraft:full"
        out.putLong("LastUpdate", lastUpdate);
        out.putLong("InhabitedTime", inhabitedTime);
        out.put("sections", newSections);
        out.put("block_entities", blockEntities);
        out.put("block_ticks", new ListTag<>(CompoundTag.class));
        out.put("fluid_ticks", new ListTag<>(CompoundTag.class));
        out.put("Lights", emptyNestedList(TOTAL_SECTIONS));
        out.put("PostProcessing", emptyNestedList(TOTAL_SECTIONS));
        out.put("Heightmaps", new CompoundTag()); // server recomputes on load
        out.put("structures", structuresTag);
        out.put("entities", new ListTag<>(CompoundTag.class)); // required in 1.19.4
        out.put("CarvingMasks", new CompoundTag()); // present (empty) in DFU output
        // blending_data intentionally omitted — prevents deepslate cave generation
        // below_zero_retrogen intentionally omitted

        return out;
    }

    // -----------------------------------------------------------------------

    private ListTag<CompoundTag> convertSections(ListTag<CompoundTag> oldSections, byte[] biomes) {
        // Index old sections by Y value
        Map<Integer, CompoundTag> oldByY = new LinkedHashMap<>();
        for (CompoundTag sec : oldSections) {
            int y = sec.getByte("Y") & 0xFF;
            // 1.8 sections use Y 0..15; stored as byte so Y=255 means -1 (light-only section)
            // Treat signed: values >= 128 are negative (section Y -128..-1 in signed byte)
            byte yByte = sec.getByte("Y");
            oldByY.put((int) yByte, sec);
        }

        ListTag<CompoundTag> result = new ListTag<>(CompoundTag.class);

        // Emit ALL 24 sections for the 1.19.4 height range (-4 to 19)
        for (int y = MIN_SECTION_Y; y <= MAX_SECTION_Y; y++) {
            CompoundTag oldSec = oldByY.get(y);

            CompoundTag blockStates;
            if (oldSec != null) {
                byte[] blocks  = oldSec.getByteArray("Blocks");
                byte[] add     = oldSec.getByteArray("Add");
                byte[] dataNib = oldSec.getByteArray("Data");

                if (blocks != null && blocks.length >= 4096) {
                    blockStates = convertBlockSection(blocks, add, dataNib);
                } else {
                    blockStates = airBlockStates();
                }
            } else {
                blockStates = airBlockStates();
            }

            // biomes compound (4×4×4 per section)
            CompoundTag biomesTag = buildSectionBiomes(biomes, y);

            CompoundTag newSec = new CompoundTag();
            newSec.putByte("Y", (byte) y);
            newSec.put("block_states", blockStates);
            newSec.put("biomes", biomesTag);
            // Omit BlockLight/SkyLight — server recalculates lighting

            result.add(newSec);
        }

        return result;
    }

    private CompoundTag airBlockStates() {
        // Single-entry palette = air; no data array needed
        ListTag<CompoundTag> palette = new ListTag<>(CompoundTag.class);
        CompoundTag air = new CompoundTag();
        air.putString("Name", "minecraft:air");
        palette.add(air);
        CompoundTag bs = new CompoundTag();
        bs.put("palette", palette);
        return bs;
    }

    private CompoundTag convertBlockSection(byte[] blocks, byte[] add, byte[] dataNib) {
        // Build palette + index array
        Map<String, Integer> paletteMap = new LinkedHashMap<>();
        int[] indices = new int[4096];

        for (int i = 0; i < 4096; i++) {
            int addNibble = 0;
            if (add != null && add.length > 0) {
                addNibble = (add[i / 2] >> ((i % 2) * 4)) & 0xF;
            }
            int blockId = (addNibble << 8) | (blocks[i] & 0xFF);
            int meta    = 0;
            if (dataNib != null && dataNib.length > 0) {
                meta = (dataNib[i / 2] >> ((i % 2) * 4)) & 0xF;
            }

            String state = mapper.getBlockState(blockId, meta);
            indices[i] = paletteMap.computeIfAbsent(state, k -> paletteMap.size());
        }

        // Palette
        String[] stateNames = new String[paletteMap.size()];
        paletteMap.forEach((k, v) -> stateNames[v] = k);
        ListTag<CompoundTag> palette = new ListTag<>(CompoundTag.class);
        for (String s : stateNames) palette.add(parseBlockState(s));

        // block_states compound
        CompoundTag blockStates = new CompoundTag();
        blockStates.put("palette", palette);
        if (paletteMap.size() > 1) {
            blockStates.put("data", packIndices(indices, paletteMap.size(), 4096));
        }
        return blockStates;
    }

    private CompoundTag parseBlockState(String blockState) {
        CompoundTag tag = new CompoundTag();
        int bracket = blockState.indexOf('[');
        if (bracket == -1) {
            tag.putString("Name", blockState);
        } else {
            tag.putString("Name", blockState.substring(0, bracket));
            String propsStr = blockState.substring(bracket + 1, blockState.length() - 1);
            CompoundTag props = new CompoundTag();
            for (String prop : propsStr.split(",")) {
                String[] kv = prop.split("=", 2);
                if (kv.length == 2) props.putString(kv[0].trim(), kv[1].trim());
            }
            tag.put("Properties", props);
        }
        return tag;
    }

    private LongArrayTag packIndices(int[] indices, int paletteSize, int count) {
        int bpe = Math.max(4, 32 - Integer.numberOfLeadingZeros(paletteSize - 1));
        int epl = 64 / bpe;
        long[] longs = new long[(count + epl - 1) / epl];
        for (int i = 0; i < count; i++) {
            longs[i / epl] |= ((long) indices[i]) << ((i % epl) * bpe);
        }
        return new LongArrayTag(longs);
    }

    private CompoundTag buildSectionBiomes(byte[] oldBiomes, int sectionY) {
        CompoundTag tag = new CompoundTag();
        ListTag<StringTag> palette = new ListTag<>(StringTag.class);

        if (oldBiomes == null || oldBiomes.length < 256) {
            palette.add(new StringTag("minecraft:plains"));
            tag.put("palette", palette);
            return tag;
        }

        Map<String, Integer> biomeMap = new LinkedHashMap<>();
        int[] bIndices = new int[64];

        for (int by = 0; by < 4; by++) {
            for (int bz = 0; bz < 4; bz++) {
                for (int bx = 0; bx < 4; bx++) {
                    int colX = Math.min(bx * 4 + 2, 15);
                    int colZ = Math.min(bz * 4 + 2, 15);
                    int id   = oldBiomes[colZ * 16 + colX] & 0xFF;
                    String name = BiomeMapper.get(id);
                    bIndices[by * 16 + bz * 4 + bx] =
                        biomeMap.computeIfAbsent(name, k -> biomeMap.size());
                }
            }
        }

        String[] names = new String[biomeMap.size()];
        biomeMap.forEach((k, v) -> names[v] = k);
        for (String n : names) palette.add(new StringTag(n));
        tag.put("palette", palette);

        if (biomeMap.size() > 1) {
            int bpe = Math.max(1, 32 - Integer.numberOfLeadingZeros(biomeMap.size() - 1));
            int epl = 64 / bpe;
            long[] longs = new long[(64 + epl - 1) / epl];
            for (int i = 0; i < 64; i++) {
                longs[i / epl] |= ((long) bIndices[i]) << ((i % epl) * bpe);
            }
            tag.put("data", new LongArrayTag(longs));
        }

        return tag;
    }

    private ListTag<CompoundTag> convertBlockEntities(ListTag<?> oldTEs) {
        ListTag<CompoundTag> result = new ListTag<>(CompoundTag.class);
        if (oldTEs == null) return result;
        for (Object te : oldTEs) {
            if (te instanceof CompoundTag) {
                CompoundTag entry = (CompoundTag) ((CompoundTag) te).clone();
                String id = entry.getString("id");
                if (id != null && !id.contains(":")) {
                    entry.putString("id", BlockEntityMapper.remap(id));
                }
                result.add(entry);
            }
        }
        return result;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ListTag emptyNestedList(int count) {
        ListTag outer = new ListTag(ListTag.class);
        for (int i = 0; i < count; i++) {
            outer.add(new ListTag<>(ShortTag.class));
        }
        return outer;
    }
}
