package blockscope.converter;

import net.querz.nbt.io.NBTUtil;
import net.querz.nbt.io.NamedTag;
import net.querz.nbt.tag.ByteTag;
import net.querz.nbt.tag.CompoundTag;
import net.querz.nbt.tag.IntTag;
import net.querz.nbt.tag.ListTag;
import net.querz.nbt.tag.LongTag;
import net.querz.nbt.tag.StringTag;

import java.io.File;
import java.nio.file.Files;

public class LevelDatUpdater {

    private static final int TARGET_DATA_VERSION = 3337; // 1.19.4

    public static void update(File input, File output) throws Exception {
        if (!input.exists()) {
            System.err.println("Warning: level.dat not found at " + input);
            return;
        }

        NamedTag named = NBTUtil.read(input);
        CompoundTag root = (CompoundTag) named.getTag();
        CompoundTag data = root.getCompoundTag("Data");

        if (data == null) {
            System.err.println("Warning: level.dat has no Data compound, skipping");
            Files.copy(input.toPath(), output.toPath());
            return;
        }

        // ----------------------------------------------------------------
        // 1. Update version fields
        // ----------------------------------------------------------------
        data.putInt("DataVersion", TARGET_DATA_VERSION);

        CompoundTag version = new CompoundTag();
        version.putInt("Id", TARGET_DATA_VERSION);
        version.putString("Name", "1.19.4");
        version.putString("Series", "main");
        version.putByte("Snapshot", (byte) 0);
        data.put("Version", version);

        // ----------------------------------------------------------------
        // 2. Read old seed (Data.RandomSeed in 1.8) before removing it
        // ----------------------------------------------------------------
        long seed = data.getLong("RandomSeed"); // 0 if absent, which is fine

        // ----------------------------------------------------------------
        // 3. Remove old fields that conflict with 1.19.4
        // ----------------------------------------------------------------
        data.remove("generatorName");    // replaced by WorldGenSettings
        data.remove("generatorVersion");
        data.remove("generatorOptions");
        data.remove("MapFeatures");
        data.remove("RandomSeed");       // moved into WorldGenSettings.seed
        data.remove("SizeOnDisk");       // obsolete

        // ----------------------------------------------------------------
        // 4. Build WorldGenSettings
        //    The original 1.8 world used generator "flat" with no options,
        //    which DFU converts to a minecraft:flat overworld with default
        //    flat layers (air/bedrock/dirt/grass_block).
        // ----------------------------------------------------------------
        CompoundTag wgs = new CompoundTag();
        wgs.put("seed", new LongTag(seed));
        wgs.put("generate_features", new ByteTag((byte) 1));
        wgs.put("bonus_chest", new ByteTag((byte) 0));

        CompoundTag dimensions = new CompoundTag();

        // --- minecraft:overworld (flat) ---
        CompoundTag flatSettings = new CompoundTag();
        flatSettings.put("features", new ByteTag((byte) 0));
        flatSettings.putString("biome", "minecraft:plains");
        flatSettings.put("lakes", new ByteTag((byte) 0));

        @SuppressWarnings("unchecked")
        ListTag<CompoundTag> layers = new ListTag<>(CompoundTag.class);
        CompoundTag layerAir = new CompoundTag();
        layerAir.putString("block", "minecraft:air");
        layerAir.put("height", new IntTag(64));
        layers.add(layerAir);
        CompoundTag layerBedrock = new CompoundTag();
        layerBedrock.putString("block", "minecraft:bedrock");
        layerBedrock.put("height", new IntTag(1));
        layers.add(layerBedrock);
        CompoundTag layerDirt = new CompoundTag();
        layerDirt.putString("block", "minecraft:dirt");
        layerDirt.put("height", new IntTag(2));
        layers.add(layerDirt);
        CompoundTag layerGrass = new CompoundTag();
        layerGrass.putString("block", "minecraft:grass_block");
        layerGrass.put("height", new IntTag(1));
        layers.add(layerGrass);
        flatSettings.put("layers", layers);

        CompoundTag overworldGen = new CompoundTag();
        overworldGen.putString("type", "minecraft:flat");
        overworldGen.put("settings", flatSettings);

        CompoundTag overworldDim = new CompoundTag();
        overworldDim.putString("type", "minecraft:overworld");
        overworldDim.put("generator", overworldGen);
        dimensions.put("minecraft:overworld", overworldDim);

        // --- minecraft:the_nether (noise) ---
        CompoundTag netherBiomeSource = new CompoundTag();
        netherBiomeSource.putString("preset", "minecraft:nether");
        netherBiomeSource.putString("type", "minecraft:multi_noise");

        CompoundTag netherGen = new CompoundTag();
        netherGen.putString("type", "minecraft:noise");
        netherGen.put("settings", new StringTag("minecraft:nether"));
        netherGen.put("biome_source", netherBiomeSource);

        CompoundTag netherDim = new CompoundTag();
        netherDim.putString("type", "minecraft:the_nether");
        netherDim.put("generator", netherGen);
        dimensions.put("minecraft:the_nether", netherDim);

        // --- minecraft:the_end (noise) ---
        CompoundTag endBiomeSource = new CompoundTag();
        endBiomeSource.putString("type", "minecraft:the_end");

        CompoundTag endGen = new CompoundTag();
        endGen.putString("type", "minecraft:noise");
        endGen.put("settings", new StringTag("minecraft:end"));
        endGen.put("biome_source", endBiomeSource);

        CompoundTag endDim = new CompoundTag();
        endDim.putString("type", "minecraft:the_end");
        endDim.put("generator", endGen);
        dimensions.put("minecraft:the_end", endDim);

        wgs.put("dimensions", dimensions);
        data.put("WorldGenSettings", wgs);

        // ----------------------------------------------------------------
        // 5. Add other fields required/expected by 1.19.4 that the old
        //    level.dat lacked.
        // ----------------------------------------------------------------
        if (!data.containsKey("CustomBossEvents")) {
            data.put("CustomBossEvents", new CompoundTag());
        }

        if (!data.containsKey("DataPacks")) {
            CompoundTag dataPacks = new CompoundTag();
            @SuppressWarnings("unchecked")
            ListTag<StringTag> enabled = new ListTag<>(StringTag.class);
            enabled.add(new StringTag("vanilla"));
            @SuppressWarnings("unchecked")
            ListTag<StringTag> disabled = new ListTag<>(StringTag.class);
            dataPacks.put("Enabled", enabled);
            dataPacks.put("Disabled", disabled);
            data.put("DataPacks", dataPacks);
        }

        if (!data.containsKey("DragonFight")) {
            CompoundTag dragonFight = new CompoundTag();
            dragonFight.put("DragonKilled", new ByteTag((byte) 1));
            dragonFight.put("PreviouslyKilled", new ByteTag((byte) 1));
            dragonFight.put("NeedsStateScanning", new ByteTag((byte) 1));
            data.put("DragonFight", dragonFight);
        }

        if (!data.containsKey("ScheduledEvents")) {
            @SuppressWarnings("unchecked")
            ListTag<CompoundTag> scheduledEvents = new ListTag<>(CompoundTag.class);
            data.put("ScheduledEvents", scheduledEvents);
        }

        if (!data.containsKey("ServerBrands")) {
            @SuppressWarnings("unchecked")
            ListTag<StringTag> brands = new ListTag<>(StringTag.class);
            brands.add(new StringTag("vanilla"));
            data.put("ServerBrands", brands);
        }

        if (!data.containsKey("SpawnAngle")) {
            data.putFloat("SpawnAngle", 0.0f);
        }

        if (!data.containsKey("WanderingTraderSpawnChance")) {
            data.putInt("WanderingTraderSpawnChance", 25);
        }

        if (!data.containsKey("WanderingTraderSpawnDelay")) {
            data.putInt("WanderingTraderSpawnDelay", 24000);
        }

        if (!data.containsKey("WasModded")) {
            data.put("WasModded", new ByteTag((byte) 0));
        }

        // ----------------------------------------------------------------
        // 6. Write output
        //
        // NOTE: querz NBTUtil.write() wraps the FileOutputStream in a
        // GZIPOutputStream but never calls finish()/close() on it — only the
        // underlying FileOutputStream is closed, so the gzip final block and
        // trailer are never written. The result is a truncated gzip stream
        // (eof never reached) that querz's lenient reader accepts but strict
        // tools — and potentially Mojang's NBT reader — reject. We gzip the
        // NBT ourselves and close the GZIPOutputStream explicitly so finish()
        // runs and the stream is well-formed.
        // ----------------------------------------------------------------
        output.getParentFile().mkdirs();
        try (java.util.zip.GZIPOutputStream gz =
                 new java.util.zip.GZIPOutputStream(new java.io.FileOutputStream(output))) {
            new net.querz.nbt.io.NBTSerializer(false).toStream(named, gz);
        }
        System.out.println("level.dat updated to DataVersion " + TARGET_DATA_VERSION);
        System.out.println("  WorldGenSettings.seed = " + seed);
    }
}
