package com.blockscope.lodestone;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.generator.ChunkGenerator;
import org.bukkit.generator.WorldInfo;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Void world with a connected branching walkway, sky splatters, and no floor.
 * Falling off the path teleports you back to spawn (handled by SessionManager).
 *
 * Three block palettes (set via config: void_scatter_palette):
 *   scatter_classic — the existing ~91-class model vocabulary (default)
 *   classic_152     — every block that existed in Minecraft Java 1.5.2
 *   all_1_19        — every placeable block in 1.19.4 (incl. water, lava, stairs, etc.)
 */
public class VoidScatterGenerator extends ChunkGenerator {

    // ── Palettes ──────────────────────────────────────────────────────────────

    /** Current model vocabulary — mirror of block_vocab.py :: SCATTER_BLOCKS. */
    static final Material[] PALETTE_SCATTER_CLASSIC = {
        Material.STONE, Material.COBBLESTONE, Material.DIRT, Material.GRASS_BLOCK,
        Material.SAND, Material.GRAVEL, Material.CLAY, Material.SANDSTONE,
        Material.MOSSY_COBBLESTONE, Material.STONE_BRICKS, Material.BRICKS, Material.OBSIDIAN,
        Material.COARSE_DIRT, Material.PACKED_ICE, Material.SNOW_BLOCK,
        Material.COAL_ORE, Material.IRON_ORE, Material.GOLD_ORE, Material.DIAMOND_ORE,
        Material.LAPIS_ORE, Material.REDSTONE_ORE, Material.EMERALD_ORE,
        Material.COAL_BLOCK, Material.IRON_BLOCK, Material.GOLD_BLOCK,
        Material.DIAMOND_BLOCK, Material.LAPIS_BLOCK, Material.EMERALD_BLOCK,
        Material.OAK_LOG, Material.OAK_PLANKS, Material.OAK_LEAVES, Material.BOOKSHELF,
        Material.WHITE_WOOL, Material.ORANGE_WOOL, Material.MAGENTA_WOOL, Material.LIGHT_BLUE_WOOL,
        Material.YELLOW_WOOL, Material.LIME_WOOL, Material.PINK_WOOL, Material.GRAY_WOOL,
        Material.LIGHT_GRAY_WOOL, Material.CYAN_WOOL, Material.PURPLE_WOOL, Material.BLUE_WOOL,
        Material.BROWN_WOOL, Material.GREEN_WOOL, Material.RED_WOOL, Material.BLACK_WOOL,
        Material.TERRACOTTA,
        Material.WHITE_TERRACOTTA, Material.ORANGE_TERRACOTTA, Material.MAGENTA_TERRACOTTA,
        Material.LIGHT_BLUE_TERRACOTTA, Material.YELLOW_TERRACOTTA, Material.LIME_TERRACOTTA,
        Material.PINK_TERRACOTTA, Material.GRAY_TERRACOTTA, Material.LIGHT_GRAY_TERRACOTTA,
        Material.CYAN_TERRACOTTA, Material.PURPLE_TERRACOTTA, Material.BLUE_TERRACOTTA,
        Material.BROWN_TERRACOTTA, Material.GREEN_TERRACOTTA, Material.RED_TERRACOTTA,
        Material.BLACK_TERRACOTTA,
        Material.HAY_BLOCK, Material.PUMPKIN, Material.MELON,
        Material.BROWN_MUSHROOM_BLOCK, Material.RED_MUSHROOM_BLOCK, Material.MUSHROOM_STEM,
        Material.GLASS,
        Material.WHITE_STAINED_GLASS, Material.ORANGE_STAINED_GLASS, Material.MAGENTA_STAINED_GLASS,
        Material.LIGHT_BLUE_STAINED_GLASS, Material.YELLOW_STAINED_GLASS, Material.LIME_STAINED_GLASS,
        Material.PINK_STAINED_GLASS, Material.GRAY_STAINED_GLASS, Material.CYAN_STAINED_GLASS,
        Material.PURPLE_STAINED_GLASS, Material.BLUE_STAINED_GLASS, Material.BROWN_STAINED_GLASS,
        Material.GREEN_STAINED_GLASS, Material.RED_STAINED_GLASS, Material.BLACK_STAINED_GLASS,
        Material.CRAFTING_TABLE, Material.CHEST, Material.FURNACE,
        Material.NETHERRACK,
    };

    /**
     * Every block that existed in Minecraft Java Edition 1.5.2.
     * 1.5 added: quartz family, dropper, hopper, comparator, daylight sensor,
     *            weighted pressure plates, activator rail, redstone block.
     * Excludes post-1.5.2 blocks: coal_block (1.6), hay_bale (1.6),
     *   terracotta (1.6), packed_ice (1.7), stained_glass (1.7),
     *   acacia/dark_oak (1.7), granite/diorite/andesite (1.8), etc.
     */
    static final Material[] PALETTE_CLASSIC_152 = {
        // Terrain
        Material.STONE, Material.COBBLESTONE, Material.MOSSY_COBBLESTONE,
        Material.GRASS_BLOCK, Material.DIRT, Material.SAND, Material.GRAVEL,
        Material.BEDROCK, Material.SPONGE, Material.OBSIDIAN, Material.CLAY,
        // Fluids
        Material.WATER, Material.LAVA,
        // Ores
        Material.COAL_ORE, Material.IRON_ORE, Material.GOLD_ORE, Material.DIAMOND_ORE,
        Material.LAPIS_ORE, Material.REDSTONE_ORE, Material.EMERALD_ORE,
        Material.NETHER_QUARTZ_ORE,
        // Mineral blocks (no coal_block — that's 1.6)
        Material.IRON_BLOCK, Material.GOLD_BLOCK, Material.DIAMOND_BLOCK,
        Material.LAPIS_BLOCK, Material.EMERALD_BLOCK, Material.REDSTONE_BLOCK,
        // Quartz (added in 1.5)
        Material.QUARTZ_BLOCK, Material.CHISELED_QUARTZ_BLOCK, Material.QUARTZ_PILLAR,
        Material.QUARTZ_STAIRS, Material.QUARTZ_SLAB,
        // Wood — only 4 types existed in 1.5.2
        Material.OAK_LOG, Material.SPRUCE_LOG, Material.BIRCH_LOG, Material.JUNGLE_LOG,
        Material.OAK_PLANKS, Material.SPRUCE_PLANKS, Material.BIRCH_PLANKS, Material.JUNGLE_PLANKS,
        Material.OAK_LEAVES, Material.SPRUCE_LEAVES, Material.BIRCH_LEAVES, Material.JUNGLE_LEAVES,
        Material.OAK_SAPLING, Material.SPRUCE_SAPLING, Material.BIRCH_SAPLING, Material.JUNGLE_SAPLING,
        // Stairs
        Material.OAK_STAIRS, Material.COBBLESTONE_STAIRS, Material.BRICK_STAIRS,
        Material.STONE_BRICK_STAIRS, Material.NETHER_BRICK_STAIRS, Material.SANDSTONE_STAIRS,
        Material.SPRUCE_STAIRS, Material.BIRCH_STAIRS, Material.JUNGLE_STAIRS,
        // Slabs
        Material.STONE_SLAB, Material.SANDSTONE_SLAB, Material.OAK_SLAB,
        Material.COBBLESTONE_SLAB, Material.BRICK_SLAB, Material.STONE_BRICK_SLAB,
        Material.NETHER_BRICK_SLAB, Material.SPRUCE_SLAB, Material.BIRCH_SLAB, Material.JUNGLE_SLAB,
        // Stone variants
        Material.STONE_BRICKS, Material.MOSSY_STONE_BRICKS,
        Material.CRACKED_STONE_BRICKS, Material.CHISELED_STONE_BRICKS,
        // Sandstone variants
        Material.SANDSTONE, Material.CHISELED_SANDSTONE, Material.CUT_SANDSTONE,
        // Bricks & special stone
        Material.BRICKS, Material.BOOKSHELF,
        // Nether
        Material.NETHERRACK, Material.SOUL_SAND, Material.GLOWSTONE,
        Material.NETHER_BRICKS, Material.NETHER_BRICK_FENCE, Material.NETHER_BRICK_STAIRS,
        // End
        Material.END_STONE, Material.END_PORTAL_FRAME, Material.DRAGON_EGG,
        // Wool (16 colors)
        Material.WHITE_WOOL, Material.ORANGE_WOOL, Material.MAGENTA_WOOL, Material.LIGHT_BLUE_WOOL,
        Material.YELLOW_WOOL, Material.LIME_WOOL, Material.PINK_WOOL, Material.GRAY_WOOL,
        Material.LIGHT_GRAY_WOOL, Material.CYAN_WOOL, Material.PURPLE_WOOL, Material.BLUE_WOOL,
        Material.BROWN_WOOL, Material.GREEN_WOOL, Material.RED_WOOL, Material.BLACK_WOOL,
        // Glass
        Material.GLASS, Material.GLASS_PANE, Material.IRON_BARS,
        // Functional blocks
        Material.CRAFTING_TABLE, Material.FURNACE, Material.CHEST, Material.TRAPPED_CHEST,
        Material.ENDER_CHEST, Material.DISPENSER, Material.DROPPER, Material.HOPPER,
        Material.NOTE_BLOCK, Material.JUKEBOX, Material.ENCHANTING_TABLE, Material.BEACON,
        Material.BREWING_STAND, Material.CAULDRON, Material.ANVIL, Material.COMMAND_BLOCK,
        Material.SPAWNER, Material.TNT,
        // Redstone
        Material.REDSTONE_LAMP, Material.REDSTONE_TORCH, Material.REPEATER,
        Material.COMPARATOR, Material.DAYLIGHT_DETECTOR, Material.LEVER,
        Material.OAK_PRESSURE_PLATE, Material.STONE_PRESSURE_PLATE,
        Material.HEAVY_WEIGHTED_PRESSURE_PLATE, Material.LIGHT_WEIGHTED_PRESSURE_PLATE,
        Material.OAK_BUTTON, Material.STONE_BUTTON,
        Material.RAIL, Material.POWERED_RAIL, Material.DETECTOR_RAIL, Material.ACTIVATOR_RAIL,
        Material.PISTON, Material.STICKY_PISTON,
        // Doors / fences / walls
        Material.OAK_DOOR, Material.IRON_DOOR, Material.OAK_TRAPDOOR, Material.LADDER,
        Material.OAK_FENCE, Material.OAK_FENCE_GATE,
        Material.COBBLESTONE_WALL, Material.MOSSY_COBBLESTONE_WALL,
        // Snow / ice
        Material.SNOW_BLOCK, Material.SNOW, Material.ICE,
        // Plants / organic
        Material.CACTUS, Material.PUMPKIN, Material.JACK_O_LANTERN, Material.MELON,
        Material.BROWN_MUSHROOM_BLOCK, Material.RED_MUSHROOM_BLOCK, Material.MUSHROOM_STEM,
        Material.MYCELIUM, Material.LILY_PAD, Material.COBWEB,
        Material.DANDELION, Material.POPPY, Material.BROWN_MUSHROOM, Material.RED_MUSHROOM,
        Material.TORCH, Material.CAKE, Material.FLOWER_POT,
        // Infested
        Material.INFESTED_STONE, Material.INFESTED_COBBLESTONE, Material.INFESTED_STONE_BRICKS,
        // Skulls
        Material.SKELETON_SKULL, Material.WITHER_SKELETON_SKULL,
        Material.ZOMBIE_HEAD, Material.PLAYER_HEAD, Material.CREEPER_HEAD,
        // Misc
        Material.RED_BED, Material.TRIPWIRE_HOOK,
    };

    /** Blocks to exclude from the all_1_19 dynamic palette. */
    private static final Set<Material> ALL_119_EXCLUDE = EnumSet.of(
        Material.AIR, Material.CAVE_AIR, Material.VOID_AIR,
        Material.MOVING_PISTON, Material.PISTON_HEAD,
        Material.BUBBLE_COLUMN,
        Material.END_PORTAL, Material.NETHER_PORTAL,
        Material.FROSTED_ICE,
        Material.ATTACHED_MELON_STEM, Material.ATTACHED_PUMPKIN_STEM,
        Material.LIGHT,              // debug block
        Material.BARRIER,            // debug block
        Material.STRUCTURE_VOID,     // debug
        Material.JIGSAW              // debug
    );

    static Material[] buildAll119Palette() {
        return Arrays.stream(Material.values())
            .filter(m -> m.isBlock() && !m.isAir())
            .filter(m -> !ALL_119_EXCLUDE.contains(m))
            .filter(m -> !m.name().endsWith("_WALL_SIGN"))
            .filter(m -> !m.name().endsWith("_WALL_BANNER"))
            .filter(m -> !m.name().endsWith("_WALL_SKULL"))
            .filter(m -> !m.name().endsWith("_WALL_HEAD"))
            .filter(m -> !m.name().endsWith("_WALL_FAN"))
            .filter(m -> !m.name().endsWith("_WALL_TORCH"))
            .filter(m -> !m.name().startsWith("POTTED_"))
            .toArray(Material[]::new);
    }

    /** Return the palette for a config string. */
    static Material[] forConfig(String name) {
        return switch (name == null ? "" : name.toLowerCase()) {
            case "classic_152", "classic152" -> PALETTE_CLASSIC_152;
            case "all_1_19",   "all119"      -> buildAll119Palette();
            default                           -> PALETTE_SCATTER_CLASSIC;
        };
    }

    // ── Sky shape templates ───────────────────────────────────────────────────

    private static final int[][][] SKY_SHAPES = {
        {{0, 0}},
        {{0, 0}, {1, 0}, {0, 1}, {1, 1}},
        {{0, 0}, {1, 0}, {2, 0}},
        {{0, 0}, {0, 1}, {0, 2}},
        {{0, 0}, {1, 0}, {0, 1}},
        {{0, 0}, {1, 0}, {2, 0}, {1, 1}},
        {{0, 0}, {-1, 0}, {1, 0}, {0, 1}},
        {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {2, 1}},
    };

    // ── Instance state ────────────────────────────────────────────────────────

    private final Material[] palette;

    private static final int PATH_Y     = 64;
    private static final int PATH_Y_MIN = 62;
    private static final int PATH_Y_MAX = 78;
    private static final int MAX_BRANCHES = 120;

    private volatile Map<Long, List<int[]>> chunkBlocks;
    private volatile long generatedSeed = Long.MIN_VALUE;

    public VoidScatterGenerator(Material[] palette) {
        this.palette = palette;
    }

    // ── Chunk generation ──────────────────────────────────────────────────────

    // Prevent vanilla noise density function setup — it takes 10+ seconds per world on a slow
    // CPU due to the recursive CubicSpline.mapAll() tree in 1.19.4. Custom generators must
    // explicitly opt out or Paper initialises the full vanilla overworld noise state anyway.
    @Override public boolean shouldGenerateNoise()       { return false; }
    @Override public boolean shouldGenerateSurface()     { return false; }
    @Override public boolean shouldGenerateBedrock()     { return false; }
    @Override public boolean shouldGenerateCaves()       { return false; }
    @Override public boolean shouldGenerateDecorations() { return false; }
    @Override public boolean shouldGenerateStructures()  { return false; }

    @Override
    public void generateSurface(WorldInfo worldInfo, Random random, int chunkX, int chunkZ, ChunkData chunkData) {
        ensureGenerated(worldInfo.getSeed());
        List<int[]> blocks = chunkBlocks.get(chunkKey(chunkX, chunkZ));
        if (blocks != null)
            for (int[] b : blocks)
                chunkData.setBlock(b[0], b[1], b[2], palette[b[3]]);
    }

    @Override
    public Location getFixedSpawnLocation(World world, Random random) {
        return new Location(world, 8.5, 65.0, 8.5);
    }

    // ── Lazy generation ───────────────────────────────────────────────────────

    private void ensureGenerated(long seed) {
        if (chunkBlocks != null && generatedSeed == seed) return;
        synchronized (this) {
            if (chunkBlocks != null && generatedSeed == seed) return;
            Map<Long, List<int[]>> map = new HashMap<>();
            generatePaths(seed, map);
            generateSkyBlocks(seed, map);
            chunkBlocks = Collections.unmodifiableMap(map);
            generatedSeed = seed;
        }
    }

    // ── Path generation ───────────────────────────────────────────────────────

    private void generatePaths(long seed, Map<Long, List<int[]>> out) {
        Random rng = new Random(seed);
        Deque<int[]> stack = new ArrayDeque<>();
        int branchCount = 0;

        int[][] initDirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] d : initDirs) {
            stack.push(new int[]{8, PATH_Y, 8, d[0], d[1], 80 + rng.nextInt(30), 0, 1});
            branchCount++;
        }

        while (!stack.isEmpty()) {
            int[] b = stack.pop();
            int wx = b[0], wy = b[1], wz = b[2];
            int dx = b[3], dz = b[4];
            int len = b[5];
            int stairsLeft = b[6], stairsDir = b[7];

            for (int step = 0; step < len; step++) {
                placeSegment(out, wx, wy, wz, dx, dz, rng);

                if (stairsLeft > 0) {
                    wy = Math.max(PATH_Y_MIN, Math.min(PATH_Y_MAX, wy + stairsDir));
                    stairsLeft--;
                } else if (rng.nextInt(18) == 0) {
                    stairsLeft = 3 + rng.nextInt(6);
                    stairsDir  = (wy >= PATH_Y_MAX) ? -1 : (wy <= PATH_Y_MIN) ? 1
                                 : (rng.nextBoolean() ? 1 : -1);
                }

                wx += dx;
                wz += dz;

                if (step > 14 && branchCount < MAX_BRANCHES && rng.nextInt(22) == 0) {
                    int[] nd = rng.nextBoolean() ? new int[]{-dz, dx} : new int[]{dz, -dx};
                    stack.push(new int[]{wx, wy, wz, nd[0], nd[1], 35 + rng.nextInt(50), 0, 1});
                    branchCount++;
                }
            }
        }
    }

    private void placeSegment(Map<Long, List<int[]>> out,
                               int wx, int wy, int wz, int dx, int dz, Random rng) {
        int px = -dz, pz = dx;
        for (int off = -1; off <= 1; off++)
            addBlock(out, wx + px * off, wy, wz + pz * off, rng.nextInt(palette.length));
    }

    // ── Sky splatters ─────────────────────────────────────────────────────────

    private void generateSkyBlocks(long seed, Map<Long, List<int[]>> out) {
        Random rng = new Random(seed ^ 0xCAFEBABEL);
        for (int i = 0; i < 900; i++) {
            int wx  = rng.nextInt(300) - 150;
            int wz  = rng.nextInt(300) - 150;
            int y   = 70 + rng.nextInt(22);
            int mat = rng.nextInt(palette.length);
            for (int[] off : SKY_SHAPES[rng.nextInt(SKY_SHAPES.length)])
                addBlock(out, wx + off[0], y, wz + off[1], mat);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void addBlock(Map<Long, List<int[]>> out, int wx, int y, int wz, int matIdx) {
        if (y < 1 || y > 254) return;
        int chunkX = Math.floorDiv(wx, 16);
        int chunkZ = Math.floorDiv(wz, 16);
        int lx = Math.floorMod(wx, 16);
        int lz = Math.floorMod(wz, 16);
        out.computeIfAbsent(chunkKey(chunkX, chunkZ), k -> new ArrayList<>())
           .add(new int[]{lx, y, lz, matIdx});
    }

    private static long chunkKey(int cx, int cz) {
        return ((long) cx & 0xFFFFFFFFL) | (((long) cz & 0xFFFFFFFFL) << 32);
    }
}
