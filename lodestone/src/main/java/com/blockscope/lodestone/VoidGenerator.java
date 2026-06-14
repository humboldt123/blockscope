package com.blockscope.lodestone;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.generator.ChunkGenerator;
import org.bukkit.generator.WorldInfo;

import java.util.Random;

/**
 * Plain void generator for copied library worlds (converted build maps, hermitcraft).
 *
 * The world's existing chunks load from its region files as normal; this generator
 * only runs for chunks that are NOT on disk — i.e. anything outside the original
 * map's built area. Those become empty (void) so the exploring bot can never wander
 * off the map onto freshly-generated 1.19 terrain (which would leak modern blocks).
 *
 * All the shouldGenerate* opt-outs are required: 1.19.4 otherwise spends 10+s per
 * world building the vanilla noise density-function tree even for a custom generator
 * (see VoidScatterGenerator for the same workaround).
 */
public class VoidGenerator extends ChunkGenerator {

    @Override public boolean shouldGenerateNoise()       { return false; }
    @Override public boolean shouldGenerateSurface()     { return false; }
    @Override public boolean shouldGenerateBedrock()     { return false; }
    @Override public boolean shouldGenerateCaves()       { return false; }
    @Override public boolean shouldGenerateDecorations() { return false; }
    @Override public boolean shouldGenerateStructures()  { return false; }

    @Override
    public void generateSurface(WorldInfo worldInfo, Random random,
                                int chunkX, int chunkZ, ChunkData chunkData) {
        // Intentionally empty — out-of-bounds chunks are void.
    }

    @Override
    public Location getFixedSpawnLocation(World world, Random random) {
        // Real spawn is chosen per-session from spawn_candidates.json and applied
        // via teleport; this is only a safe fallback if that list is empty.
        return new Location(world, 0.5, 64.0, 0.5);
    }
}
