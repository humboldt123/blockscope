package com.blockscope.lodestone;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.generator.ChunkGenerator;
import org.bukkit.generator.WorldInfo;

import java.util.Random;

/**
 * Void world with randomly scattered blocks from y=0..128.
 * ~1.5% of cells become a random block from SCATTER_MATERIALS.
 * Spawn chunk gets a 3×3 stone platform at y=63 so the player doesn't fall on join.
 *
 * Kept in sync with SCATTER_BLOCKS in block_vocab.py.
 */
public class VoidScatterGenerator extends ChunkGenerator {

    private static final float DENSITY = 0.015f;

    // Solid, item-holdable blocks — visually diverse subset of the known vocab.
    // Mirror of block_vocab.py :: SCATTER_BLOCKS.
    static final Material[] SCATTER_MATERIALS = {
        // Terrain
        Material.STONE, Material.COBBLESTONE, Material.DIRT, Material.GRASS_BLOCK,
        Material.SAND, Material.GRAVEL, Material.CLAY, Material.SANDSTONE,
        Material.MOSSY_COBBLESTONE, Material.STONE_BRICKS, Material.BRICKS, Material.OBSIDIAN,
        Material.COARSE_DIRT, Material.PACKED_ICE, Material.SNOW_BLOCK,
        // Ores
        Material.COAL_ORE, Material.IRON_ORE, Material.GOLD_ORE, Material.DIAMOND_ORE,
        Material.LAPIS_ORE, Material.REDSTONE_ORE, Material.EMERALD_ORE,
        // Mineral blocks
        Material.COAL_BLOCK, Material.IRON_BLOCK, Material.GOLD_BLOCK,
        Material.DIAMOND_BLOCK, Material.LAPIS_BLOCK, Material.EMERALD_BLOCK,
        // Wood
        Material.OAK_LOG, Material.OAK_PLANKS, Material.OAK_LEAVES, Material.BOOKSHELF,
        // Wool
        Material.WHITE_WOOL, Material.ORANGE_WOOL, Material.MAGENTA_WOOL, Material.LIGHT_BLUE_WOOL,
        Material.YELLOW_WOOL, Material.LIME_WOOL, Material.PINK_WOOL, Material.GRAY_WOOL,
        Material.LIGHT_GRAY_WOOL, Material.CYAN_WOOL, Material.PURPLE_WOOL, Material.BLUE_WOOL,
        Material.BROWN_WOOL, Material.GREEN_WOOL, Material.RED_WOOL, Material.BLACK_WOOL,
        // Terracotta
        Material.TERRACOTTA,
        Material.WHITE_TERRACOTTA, Material.ORANGE_TERRACOTTA, Material.MAGENTA_TERRACOTTA,
        Material.LIGHT_BLUE_TERRACOTTA, Material.YELLOW_TERRACOTTA, Material.LIME_TERRACOTTA,
        Material.PINK_TERRACOTTA, Material.GRAY_TERRACOTTA, Material.LIGHT_GRAY_TERRACOTTA,
        Material.CYAN_TERRACOTTA, Material.PURPLE_TERRACOTTA, Material.BLUE_TERRACOTTA,
        Material.BROWN_TERRACOTTA, Material.GREEN_TERRACOTTA, Material.RED_TERRACOTTA,
        Material.BLACK_TERRACOTTA,
        // Organic / structural
        Material.HAY_BLOCK, Material.PUMPKIN, Material.MELON,
        Material.BROWN_MUSHROOM_BLOCK, Material.RED_MUSHROOM_BLOCK, Material.MUSHROOM_STEM,
        // Glass
        Material.GLASS,
        Material.WHITE_STAINED_GLASS, Material.ORANGE_STAINED_GLASS, Material.MAGENTA_STAINED_GLASS,
        Material.LIGHT_BLUE_STAINED_GLASS, Material.YELLOW_STAINED_GLASS, Material.LIME_STAINED_GLASS,
        Material.PINK_STAINED_GLASS, Material.GRAY_STAINED_GLASS, Material.CYAN_STAINED_GLASS,
        Material.PURPLE_STAINED_GLASS, Material.BLUE_STAINED_GLASS, Material.BROWN_STAINED_GLASS,
        Material.GREEN_STAINED_GLASS, Material.RED_STAINED_GLASS, Material.BLACK_STAINED_GLASS,
        // Functional
        Material.CRAFTING_TABLE, Material.CHEST, Material.FURNACE,
        // Rare
        Material.NETHERRACK,
    };

    @Override
    public void generateSurface(WorldInfo worldInfo, Random random, int chunkX, int chunkZ, ChunkData chunkData) {
        // Spawn platform — 3×3 stone at y=63 so the player has somewhere to land
        if (chunkX == 0 && chunkZ == 0) {
            for (int x = 7; x <= 9; x++)
                for (int z = 7; z <= 9; z++)
                    chunkData.setBlock(x, 63, z, Material.STONE);
        }

        // Sparse block scatter y=0..127
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 128; y++) {
                    if (random.nextFloat() < DENSITY) {
                        chunkData.setBlock(x, y, z,
                            SCATTER_MATERIALS[random.nextInt(SCATTER_MATERIALS.length)]);
                    }
                }
            }
        }
    }

    @Override
    public Location getFixedSpawnLocation(World world, Random random) {
        return new Location(world, 8.5, 64, 8.5);
    }
}
