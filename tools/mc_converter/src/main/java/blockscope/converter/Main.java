package blockscope.converter;

import net.querz.nbt.io.NBTDeserializer;
import net.querz.nbt.io.NBTSerializer;
import net.querz.nbt.io.NamedTag;
import net.querz.nbt.tag.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;
import java.util.zip.*;

public class Main {

    private static final int TARGET_DATA_VERSION = 3337; // 1.19.4

    public static void main(String[] args) throws Exception {
        String inputPath = null, outputPath = null;
        for (int i = 0; i < args.length; i++) {
            if ("--input".equals(args[i])  && i + 1 < args.length) inputPath  = args[++i];
            if ("--output".equals(args[i]) && i + 1 < args.length) outputPath = args[++i];
        }
        if (inputPath == null || outputPath == null) {
            System.err.println("Usage: mc-converter --input <world_dir> --output <out_dir>");
            System.exit(1);
        }

        Path input  = Paths.get(inputPath);
        Path output = Paths.get(outputPath);

        LegacyMapper   mapper = LegacyMapper.load();
        ChunkConverter conv   = new ChunkConverter(mapper);

        // ---- Overworld region + entities ----
        processRegionDir(input, output, conv);

        // ---- DIM-1 (Nether) ----
        Path dim1In  = input.resolve("DIM-1");
        Path dim1Out = output.resolve("DIM-1");
        if (Files.isDirectory(dim1In)) {
            if (Files.isDirectory(dim1In.resolve("region"))) {
                processRegionDir(dim1In, dim1Out, conv);
            }
            // Copy dim-level data files (data/, forcedchunks.dat) regardless
            copyIfExists(dim1In.resolve("data"),            dim1Out.resolve("data"));
            copyFileIfExists(dim1In.resolve("forcedchunks.dat"), dim1Out.resolve("forcedchunks.dat"));
        }

        // ---- DIM1 (End) ----
        Path dim2In  = input.resolve("DIM1");
        Path dim2Out = output.resolve("DIM1");
        if (Files.isDirectory(dim2In)) {
            if (Files.isDirectory(dim2In.resolve("region"))) {
                processRegionDir(dim2In, dim2Out, conv);
            }
            // Copy dim-level data files
            copyIfExists(dim2In.resolve("data"),            dim2Out.resolve("data"));
            copyFileIfExists(dim2In.resolve("forcedchunks.dat"), dim2Out.resolve("forcedchunks.dat"));
        }

        // ---- level.dat ----
        LevelDatUpdater.update(input.resolve("level.dat").toFile(),
                               output.resolve("level.dat").toFile());

        // ---- Copy unchanged directories/files ----
        copyIfExists(input.resolve("data"),       output.resolve("data"));
        copyIfExists(input.resolve("datapacks"),  output.resolve("datapacks"));
        copyIfExists(input.resolve("playerdata"), output.resolve("playerdata"));
        copyIfExists(input.resolve("stats"),      output.resolve("stats"));
        copyFileIfExists(input.resolve("forcedchunks.dat"), output.resolve("forcedchunks.dat"));
        copyFileIfExists(input.resolve("uid.dat"),           output.resolve("uid.dat"));

        System.out.println("Done. Output: " + output.toAbsolutePath());
    }

    // -----------------------------------------------------------------------
    // Process a dimension root: converts region/*.mca and writes entities/*.mca
    // -----------------------------------------------------------------------

    /**
     * Map from entity region key "rX,rZ" → list of (chunkX, chunkZ, entities-list).
     * We accumulate entities from every chunk in a region pass, then flush per entity-region.
     */
    static record EntityChunk(int chunkX, int chunkZ, ListTag<CompoundTag> entities) {}

    private static void processRegionDir(Path dimIn, Path dimOut, ChunkConverter conv) throws Exception {
        Path inRegion  = dimIn.resolve("region");
        Path outRegion = dimOut.resolve("region");
        Path outEntities = dimOut.resolve("entities");
        Files.createDirectories(outRegion);

        // regionKey → list of entity chunks to write
        Map<String, List<EntityChunk>> entityMap = new LinkedHashMap<>();

        try (Stream<Path> files = Files.list(inRegion)) {
            files.filter(p -> p.toString().endsWith(".mca"))
                 .sorted()
                 .forEach(mcaPath -> {
                     try {
                         convertRegion(mcaPath, outRegion.resolve(mcaPath.getFileName()), conv, entityMap);
                     } catch (Exception e) {
                         System.err.println("Error on " + mcaPath.getFileName() + ": " + e.getMessage());
                         e.printStackTrace();
                     }
                 });
        }

        // Write entity region files if there are any
        if (!entityMap.isEmpty()) {
            Files.createDirectories(outEntities);
            for (Map.Entry<String, List<EntityChunk>> entry : entityMap.entrySet()) {
                String key = entry.getKey(); // "rX,rZ"
                String[] parts = key.split(",");
                int rX = Integer.parseInt(parts[0]);
                int rZ = Integer.parseInt(parts[1]);
                Path entityFile = outEntities.resolve("r." + rX + "." + rZ + ".mca");
                writeEntityRegion(entityFile, rX, rZ, entry.getValue());
            }
            System.out.println("  Wrote " + entityMap.size() + " entity region file(s) to " + outEntities);
        }
    }

    // -----------------------------------------------------------------------
    // Raw .mca (Anvil region) reader/writer — avoids Querz's Chunk API.
    //
    // Format:
    //   [0..4095]    chunk offsets: 1024 × (3-byte sector offset | 1-byte sector count)
    //   [4096..8191] chunk timestamps: 1024 × 4-byte int
    //   [8192..]     4KB sectors, each chunk: 4-byte length | 1-byte compression | zlib data
    // -----------------------------------------------------------------------

    private static void convertRegion(Path in, Path out, ChunkConverter conv,
                                      Map<String, List<EntityChunk>> entityMap) throws Exception {
        // Parse region X/Z from filename r.X.Z.mca
        String name = in.getFileName().toString(); // e.g. r.-1.2.mca
        String[] parts = name.split("\\.");        // ["r", "-1", "2", "mca"]
        int regionX = Integer.parseInt(parts[1]);
        int regionZ = Integer.parseInt(parts[2]);

        System.out.println("Converting " + name + " ...");
        byte[] regionBytes = Files.readAllBytes(in);

        // Header: 1024 location entries (4 bytes each)
        int[] offsets      = new int[1024];
        int[] sectorCounts = new int[1024];
        int[] timestamps   = new int[1024];

        for (int i = 0; i < 1024; i++) {
            int base = i * 4;
            offsets[i]      = ((regionBytes[base] & 0xFF) << 16)
                             | ((regionBytes[base + 1] & 0xFF) << 8)
                             |  (regionBytes[base + 2] & 0xFF);
            sectorCounts[i] =   regionBytes[base + 3] & 0xFF;
            timestamps[i]   = ((regionBytes[4096 + base] & 0xFF) << 24)
                             | ((regionBytes[4096 + base + 1] & 0xFF) << 16)
                             | ((regionBytes[4096 + base + 2] & 0xFF) << 8)
                             |  (regionBytes[4096 + base + 3] & 0xFF);
        }

        // Collect converted chunks: index → bytes (zlib-compressed 1.19.4 NBT)
        byte[][] convertedChunks = new byte[1024][];
        int converted = 0;

        for (int i = 0; i < 1024; i++) {
            if (offsets[i] == 0 && sectorCounts[i] == 0) continue; // empty slot

            int byteOffset = offsets[i] * 4096;
            int length     = ((regionBytes[byteOffset] & 0xFF) << 24)
                           | ((regionBytes[byteOffset + 1] & 0xFF) << 16)
                           | ((regionBytes[byteOffset + 2] & 0xFF) << 8)
                           |  (regionBytes[byteOffset + 3] & 0xFF);
            int compression = regionBytes[byteOffset + 4] & 0xFF; // 2 = zlib

            byte[] chunkData = new byte[length - 1];
            System.arraycopy(regionBytes, byteOffset + 5, chunkData, 0, chunkData.length);

            // Decompress
            byte[] raw = decompress(chunkData, compression);

            // Parse NBT (already decompressed above — pass false for compressed)
            NamedTag named = new NBTDeserializer(false).fromStream(new ByteArrayInputStream(raw));
            CompoundTag oldTag = (CompoundTag) named.getTag();

            // Extract entities before conversion (from Level.Entities in 1.8 format)
            extractEntities(oldTag, regionX, regionZ, i, entityMap);

            // Convert
            int chunkX = regionX * 32 + (i & 31);
            int chunkZ = regionZ * 32 + (i >> 5);
            CompoundTag newTag = conv.convert(oldTag, chunkX, chunkZ);

            // Re-serialize (uncompressed — region writer handles compression)
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            new NBTSerializer(false).toStream(new NamedTag("", newTag), baos);
            convertedChunks[i] = compress(baos.toByteArray());
            converted++;
        }

        System.out.println("  → " + converted + " chunks");

        // Write output region file
        writeRegion(out, convertedChunks, timestamps);
    }

    // -----------------------------------------------------------------------
    // Entity extraction from 1.8 Level.Entities → 1.17+ entities/ region format
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static void extractEntities(CompoundTag oldChunkTag, int regionX, int regionZ,
                                        int slotIndex, Map<String, List<EntityChunk>> entityMap) {
        CompoundTag level = oldChunkTag.getCompoundTag("Level");
        if (level == null) return; // already new format, no Level wrapper

        ListTag<?> rawEntities = level.getListTag("Entities");
        if (rawEntities == null || rawEntities.size() == 0) return;

        int chunkX = regionX * 32 + (slotIndex & 31);
        int chunkZ = regionZ * 32 + (slotIndex >> 5);

        // Entity region file is the same region as the chunk
        String key = regionX + "," + regionZ;

        ListTag<CompoundTag> entityList = new ListTag<>(CompoundTag.class);
        for (Object obj : rawEntities) {
            if (obj instanceof CompoundTag) {
                CompoundTag entity = (CompoundTag) ((CompoundTag) obj).clone();
                // Remap entity id to namespaced form (1.8: "Sheep" → 1.19.4: "minecraft:sheep")
                String oldId = entity.getString("id");
                entity.putString("id", remapEntityId(oldId));
                // Ensure UUID exists — 1.8 entities use "UUIDMost"/"UUIDLeast" longs
                ensureEntityUUID(entity);
                entityList.add(entity);
            }
        }

        if (entityList.size() > 0) {
            entityMap.computeIfAbsent(key, k -> new ArrayList<>())
                     .add(new EntityChunk(chunkX, chunkZ, entityList));
        }
    }

    private static String remapEntityId(String id) {
        if (id == null) return "minecraft:pig"; // fallback
        if (id.contains(":")) return id;        // already namespaced
        // 1.8 entity IDs are PascalCase — just lowercase and namespace
        return "minecraft:" + id.toLowerCase(java.util.Locale.ROOT);
    }

    private static void ensureEntityUUID(CompoundTag entity) {
        // 1.19.4 expects UUID as int[4]; 1.8 stores UUIDMost/UUIDLeast as longs
        if (!entity.containsKey("UUID")) {
            long most  = entity.getLong("UUIDMost");
            long least = entity.getLong("UUIDLeast");
            if (most != 0 || least != 0) {
                int[] uuid = new int[]{
                    (int) (most  >> 32),
                    (int)  most,
                    (int) (least >> 32),
                    (int)  least
                };
                entity.put("UUID", new IntArrayTag(uuid));
            }
            entity.remove("UUIDMost");
            entity.remove("UUIDLeast");
        }
    }

    /**
     * Write an entity region file (.mca in the entities/ folder).
     * Each chunk's NBT root: { DataVersion, Position:[chunkX,chunkZ], Entities:[...] }
     */
    private static void writeEntityRegion(Path out, int regionX, int regionZ,
                                          List<EntityChunk> chunks) throws Exception {
        System.out.println("  Writing entity region " + out.getFileName()
                           + " (" + chunks.size() + " chunks with entities)");

        byte[][] slotData = new byte[1024][];

        for (EntityChunk ec : chunks) {
            int localX = ec.chunkX() - regionX * 32;
            int localZ = ec.chunkZ() - regionZ * 32;
            int slot   = localZ * 32 + localX;
            if (slot < 0 || slot >= 1024) continue;

            // Build entity chunk root
            CompoundTag root = new CompoundTag();
            root.putInt("DataVersion", TARGET_DATA_VERSION);
            IntArrayTag position = new IntArrayTag(new int[]{ec.chunkX(), ec.chunkZ()});
            root.put("Position", position);
            root.put("Entities", ec.entities());

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            new NBTSerializer(false).toStream(new NamedTag("", root), baos);
            slotData[slot] = compress(baos.toByteArray());
        }

        int[] timestamps = new int[1024]; // all zero is fine
        writeRegion(out, slotData, timestamps);
    }

    // -----------------------------------------------------------------------
    // File copy utilities (issue 3)
    // -----------------------------------------------------------------------

    private static void copyIfExists(Path src, Path dst) throws IOException {
        if (!Files.exists(src)) return;
        copyRecursive(src, dst);
        System.out.println("  Copied " + src.getFileName() + "/");
    }

    private static void copyFileIfExists(Path src, Path dst) throws IOException {
        if (!Files.exists(src)) return;
        Files.createDirectories(dst.getParent());
        Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("  Copied " + src.getFileName());
    }

    private static void copyRecursive(Path src, Path dst) throws IOException {
        if (Files.isDirectory(src)) {
            Files.createDirectories(dst);
            try (Stream<Path> children = Files.list(src)) {
                for (Path child : (Iterable<Path>) children::iterator) {
                    copyRecursive(child, dst.resolve(child.getFileName()));
                }
            }
        } else {
            Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // -----------------------------------------------------------------------
    // Compression helpers
    // -----------------------------------------------------------------------

    private static byte[] decompress(byte[] data, int type) throws IOException {
        if (type == 2) { // zlib
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            try (InflaterInputStream iis = new InflaterInputStream(new ByteArrayInputStream(data))) {
                iis.transferTo(out);
            }
            return out.toByteArray();
        } else if (type == 1) { // gzip
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            try (GZIPInputStream gis = new GZIPInputStream(new ByteArrayInputStream(data))) {
                gis.transferTo(out);
            }
            return out.toByteArray();
        }
        throw new IOException("Unknown compression type: " + type);
    }

    private static byte[] compress(byte[] data) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (DeflaterOutputStream dos = new DeflaterOutputStream(baos)) {
            dos.write(data);
        }
        return baos.toByteArray();
    }

    private static void writeRegion(Path out, byte[][] chunks, int[] timestamps) throws IOException {
        // Two-pass: figure out sector assignments, then write
        int sector = 2; // sectors 0 and 1 are the header
        int[] newOffsets      = new int[1024];
        int[] newSectorCounts = new int[1024];

        for (int i = 0; i < 1024; i++) {
            if (chunks[i] == null) continue;
            int payloadLen  = 5 + chunks[i].length; // 4-byte length + 1-byte type + data
            int sectorsNeeded = (payloadLen + 4095) / 4096;
            newOffsets[i]      = sector;
            newSectorCounts[i] = sectorsNeeded;
            sector += sectorsNeeded;
        }

        int totalBytes = sector * 4096;
        byte[] output = new byte[totalBytes];

        // Write header
        for (int i = 0; i < 1024; i++) {
            int base = i * 4;
            output[base]     = (byte) (newOffsets[i] >> 16);
            output[base + 1] = (byte) (newOffsets[i] >> 8);
            output[base + 2] = (byte)  newOffsets[i];
            output[base + 3] = (byte)  newSectorCounts[i];

            int ts = timestamps[i];
            output[4096 + base]     = (byte) (ts >> 24);
            output[4096 + base + 1] = (byte) (ts >> 16);
            output[4096 + base + 2] = (byte) (ts >> 8);
            output[4096 + base + 3] = (byte)  ts;
        }

        // Write chunk data
        for (int i = 0; i < 1024; i++) {
            if (chunks[i] == null) continue;
            int byteOffset  = newOffsets[i] * 4096;
            int payloadLen  = 1 + chunks[i].length;
            output[byteOffset]     = (byte) (payloadLen >> 24);
            output[byteOffset + 1] = (byte) (payloadLen >> 16);
            output[byteOffset + 2] = (byte) (payloadLen >> 8);
            output[byteOffset + 3] = (byte)  payloadLen;
            output[byteOffset + 4] = 2; // zlib
            System.arraycopy(chunks[i], 0, output, byteOffset + 5, chunks[i].length);
        }

        Files.write(out, output);
    }
}
