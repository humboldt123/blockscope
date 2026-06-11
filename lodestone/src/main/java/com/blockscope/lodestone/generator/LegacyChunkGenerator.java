package com.blockscope.lodestone.generator;

import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.generator.ChunkGenerator;
import org.bukkit.generator.WorldInfo;

import java.util.Random;

/**
 * Custom chunk generator implementing period-accurate terrain for two eras:
 *
 *   BETA        — Beta 1.7.3-style terrain. Wrapped=false noise (can produce
 *                 "far lands" at extreme distances). Biome height fully noise-driven.
 *
 *   RELEASE_2013 — Release 1.5.2-era terrain. Wrapped=true noise (prevents far
 *                  lands). Same noise parameters as beta; proper per-biome height
 *                  scaling is deferred — see TODO below.
 *
 * Block palette accuracy (no andesite, deepslate, etc.) is handled by datapacks,
 * not this class.
 *
 * Structures are disabled for both eras.
 * TODO: wire up old structure types (villages, strongholds, mineshafts) once
 *       the period-accurate datapack layer is stable.
 *
 * Noise parameter source: moderner-beta mod (PerlinNoiseSettings.DEFAULT/RELEASE,
 * NoiseScale.DEFAULT, NoiseLandmass.DEFAULT, ChunkProviderNoise3D). Sea level = 63.
 */
public class LegacyChunkGenerator extends ChunkGenerator {

    public enum Era { BETA, RELEASE_2013 }

    // ── Terrain constants ─────────────────────────────────────────────────────

    private static final int SEA_LEVEL    = 63;
    private static final int WORLD_HEIGHT = 128;

    // Noise grid: 2 cells per chunk horizontally (8 blocks/cell), 32 cells vertically (4 blocks/cell)
    // → 3×3×33 sample points per chunk
    private static final int CELLS_H = 2;   // 16 / 8
    private static final int CELLS_V = 32;  // 128 / 4
    private static final int RES_H   = 8;   // blocks per horizontal cell
    private static final int RES_V   = 4;   // blocks per vertical cell

    // ── NoiseScale (DEFAULT) ──────────────────────────────────────────────────

    private static final double COORD_SCALE  = 684.412;
    private static final double HEIGHT_SCALE = 684.412;
    private static final double UPPER_LIMIT  = 512.0;
    private static final double LOWER_LIMIT  = 512.0;
    private static final double DEPTH_XZ     = 200.0;
    private static final double MAIN_XZ      = 80.0;
    private static final double MAIN_Y       = 160.0;
    private static final double BASE_SIZE    = 8.5;
    private static final double STRETCH_Y    = 12.0;
    private static final double UNDERDAMP    = 4.0;
    private static final double LIMIT_BLEND  = 10.0;

    // ── NoiseLandmass (DEFAULT — both eras use DEFAULT for now) ───────────────
    // TODO: RELEASE_2013 should use NoiseLandmass.RELEASE which requires biome-
    //       height forcing (modScale/modDepth from per-biome depth/scale tables).
    //       Currently both eras use DEFAULT landmass; the only difference is
    //       wrapped=true for RELEASE_2013, which prevents far lands.

    private static final double SCALE_VAR       = 1.121;
    private static final double SCALE_OFFSET    = 0.5;
    private static final double DEPTH_NEG_INF   = 0.3;
    private static final double DEPTH_STRETCH   = 3.0;
    private static final double DEPTH_OFFSET    = -2.0;
    private static final double DEPTH_POS_DAMP  = 8.0;
    private static final double DEPTH_NEG_DAMP  = 5.6;
    private static final double DEPTH_MIN       = -1.0 / 2.8;
    private static final double DEPTH_MAX       = 1.0 / 8.0;

    // Surface noise scale — matches SurfaceProperties.DEFAULT.surfaceNoiseScale() = 0.0625f
    private static final double SURFACE_SCALE = 0.0625;

    // ── Instance ──────────────────────────────────────────────────────────────

    private final Era era;

    private volatile long initSeed = Long.MIN_VALUE;
    private LegacyPerlinOctaveNoise minLimitNoise;
    private LegacyPerlinOctaveNoise maxLimitNoise;
    private LegacyPerlinOctaveNoise mainNoise;
    private LegacyPerlinOctaveNoise scaleNoise;
    private LegacyPerlinOctaveNoise depthNoise;
    private LegacyPerlinOctaveNoise surfaceNoise;

    public LegacyChunkGenerator(Era era) {
        this.era = era;
    }

    // ── Noise initialisation ──────────────────────────────────────────────────

    private synchronized void initNoise(long seed) {
        if (initSeed == seed) return;
        boolean wrapped = (era == Era.RELEASE_2013);
        // Initialise in the same order as ChunkProviderNoise3D so that different
        // noise types receive independent RNG streams.
        Random rng = new Random(seed);
        minLimitNoise = new LegacyPerlinOctaveNoise(rng, 16, wrapped);
        maxLimitNoise = new LegacyPerlinOctaveNoise(rng, 16, wrapped);
        mainNoise     = new LegacyPerlinOctaveNoise(rng,  8, wrapped);
        // skip beach (4) and surface-perlin (4) — consumed to keep streams in sync
        new LegacyPerlinOctaveNoise(rng, 4, wrapped); // beach slot
        new LegacyPerlinOctaveNoise(rng, 4, wrapped); // surface-perlin slot
        scaleNoise    = new LegacyPerlinOctaveNoise(rng, 10, wrapped);
        depthNoise    = new LegacyPerlinOctaveNoise(rng, 16, wrapped);
        surfaceNoise  = new LegacyPerlinOctaveNoise(rng,  4, wrapped);
        initSeed = seed;
    }

    // ── Paper ChunkGenerator API ──────────────────────────────────────────────

    @Override
    public void generateNoise(WorldInfo info, Random rng, int cx, int cz, ChunkData chunk) {
        initNoise(info.getSeed());

        // Sample the 3×3×33 density grid
        double[][][] grid = new double[CELLS_H + 1][CELLS_H + 1][CELLS_V + 1];
        for (int nx = 0; nx <= CELLS_H; nx++) {
            for (int nz = 0; nz <= CELLS_H; nz++) {
                int noiseX = cx * CELLS_H + nx;
                int noiseZ = cz * CELLS_H + nz;
                double[] col = sampleColumn(noiseX, noiseZ);
                for (int ny = 0; ny <= CELLS_V; ny++) grid[nx][nz][ny] = col[ny];
            }
        }

        // Trilinearly interpolate and place blocks
        for (int bx = 0; bx < 16; bx++) {
            int cellX = bx / RES_H;
            double fx = (bx % RES_H) / (double) RES_H;
            for (int bz = 0; bz < 16; bz++) {
                int cellZ = bz / RES_H;
                double fz = (bz % RES_H) / (double) RES_H;
                for (int by = 0; by < WORLD_HEIGHT; by++) {
                    int cellY = by / RES_V;
                    double fy = (by % RES_V) / (double) RES_V;
                    double density = trilinear(
                        grid[cellX  ][cellZ  ][cellY  ], grid[cellX+1][cellZ  ][cellY  ],
                        grid[cellX  ][cellZ+1][cellY  ], grid[cellX+1][cellZ+1][cellY  ],
                        grid[cellX  ][cellZ  ][cellY+1], grid[cellX+1][cellZ  ][cellY+1],
                        grid[cellX  ][cellZ+1][cellY+1], grid[cellX+1][cellZ+1][cellY+1],
                        fx, fy, fz);
                    if (density > 0) {
                        chunk.setBlock(bx, by, bz, Material.STONE);
                    } else if (by < SEA_LEVEL) {
                        chunk.setBlock(bx, by, bz, Material.WATER);
                    }
                }
            }
        }
    }

    @Override
    public void generateSurface(WorldInfo info, Random rng, int cx, int cz, ChunkData chunk) {
        initNoise(info.getSeed());
        int startX = cx * 16, startZ = cz * 16;

        for (int bx = 0; bx < 16; bx++) {
            for (int bz = 0; bz < 16; bz++) {
                int wx = startX + bx, wz = startZ + bz;

                // Find topmost solid block
                int topY = -1;
                for (int y = WORLD_HEIGHT - 1; y >= 0; y--) {
                    if (chunk.getType(bx, y, bz) == Material.STONE) { topY = y; break; }
                }
                if (topY < 0) continue;

                // Noise-driven surface depth (1–5 blocks)
                double sv = surfaceNoise.sampleXZ(wx, wz, SURFACE_SCALE, SURFACE_SCALE);
                int depth = Math.max(1, (int)(sv / 3.0 + 3.0 + rng.nextDouble() * 0.25));

                boolean beaches = (era == Era.BETA); // EARLY_RELEASE has no beaches

                if (topY >= SEA_LEVEL + 2) {
                    chunk.setBlock(bx, topY, bz, Material.GRASS_BLOCK);
                    for (int d = 1; d < depth && topY - d >= 0; d++)
                        if (chunk.getType(bx, topY - d, bz) == Material.STONE)
                            chunk.setBlock(bx, topY - d, bz, Material.DIRT);
                } else if (beaches && topY >= SEA_LEVEL - 1) {
                    // Beta has beaches; release 1.5.2 does not (SurfaceProperties.EARLY_RELEASE)
                    for (int d = 0; d < depth && topY - d >= 0; d++)
                        if (chunk.getType(bx, topY - d, bz) == Material.STONE)
                            chunk.setBlock(bx, topY - d, bz, Material.SAND);
                } else if (topY >= SEA_LEVEL + 1) {
                    // Release: no beach transition, grass goes right to water's edge
                    chunk.setBlock(bx, topY, bz, Material.GRASS_BLOCK);
                    for (int d = 1; d < depth && topY - d >= 0; d++)
                        if (chunk.getType(bx, topY - d, bz) == Material.STONE)
                            chunk.setBlock(bx, topY - d, bz, Material.DIRT);
                }
                // Underwater stone floor unchanged
            }
        }
    }

    @Override
    public void generateBedrock(WorldInfo info, Random rng, int cx, int cz, ChunkData chunk) {
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                chunk.setBlock(x, 0, z, Material.BEDROCK);
    }

    // Disable vanilla terrain — we generate everything above ourselves
    @Override public boolean shouldGenerateNoise()       { return false; }
    @Override public boolean shouldGenerateSurface()     { return false; }
    @Override public boolean shouldGenerateBedrock()     { return false; }
    @Override public boolean shouldGenerateCaves()       { return false; }
    // Vanilla feature decoration disabled: placing no-op features via datapacks causes a
    // feature-order-cycle crash in Paper 1.19.4 (deep_dark sculk dependency graph breaks).
    // TODO: re-enable when we replace the datapack approach with era-accurate BlockPopulators
    //       that explicitly place only period-correct ores, trees and flowers.
    @Override public boolean shouldGenerateDecorations() { return false; }
    // TODO: enable old structure types (village, mineshaft, stronghold) after
    //       period-accurate datapack layer is stable and structure parameters
    //       have been sourced from moderner-beta presets.
    @Override public boolean shouldGenerateStructures()  { return false; }

    // ── Noise computation ─────────────────────────────────────────────────────

    /**
     * Fill one vertical column of density values (CELLS_V+1 = 33 values).
     * noiseX / noiseZ are noise-cell coordinates (not block coordinates).
     */
    private double[] sampleColumn(int noiseX, int noiseZ) {
        // ── Scale (terrain height multiplier) ────────────────────────────────
        double scale = (scaleNoise.sampleXZ(noiseX, noiseZ, SCALE_VAR, SCALE_VAR) + 256.0) / 512.0;
        scale = Math.max(0, Math.min(1, scale));  // clamp [0,1] before adding offset
        scale += SCALE_OFFSET;                     // always ≥ 0.5

        // ── Depth (baseline terrain height) ──────────────────────────────────
        double depth = depthNoise.sampleXZ(noiseX, noiseZ, DEPTH_XZ, DEPTH_XZ) / 8000.0;
        if (depth < 0) depth = -depth * DEPTH_NEG_INF;
        depth = depth * DEPTH_STRETCH + DEPTH_OFFSET;  // * 3 - 2
        if (depth < 0) {
            depth = Math.max(depth / DEPTH_NEG_DAMP, DEPTH_MIN);
            // Beta: negativeFlattening=true — flatten terrain over ocean areas
            // For RELEASE_2013, negativeFlattening=false per RELEASE landmass.
            // TODO: when biome height forcing is added, use era-specific negFlattening.
            if (era == Era.BETA) scale = SCALE_OFFSET; // re-apply just the offset
        } else {
            depth = Math.min(depth / DEPTH_POS_DAMP, DEPTH_MAX);
        }
        depth *= BASE_SIZE / 8.0;
        depth = BASE_SIZE + depth * 4.0;

        // ── Density column ────────────────────────────────────────────────────
        double[] col = new double[CELLS_V + 1];
        for (int ny = 0; ny <= CELLS_V; ny++) {
            double mainN = mainNoise.sample(noiseX, ny, noiseZ,
                COORD_SCALE / MAIN_XZ, HEIGHT_SCALE / MAIN_Y, COORD_SCALE / MAIN_XZ)
                / LIMIT_BLEND + 1.0;
            mainN /= 2.0;

            double density;
            if (mainN < 0) {
                density = minLimitNoise.sample(noiseX, ny, noiseZ,
                    COORD_SCALE, HEIGHT_SCALE, COORD_SCALE) / LOWER_LIMIT;
            } else if (mainN > 1) {
                density = maxLimitNoise.sample(noiseX, ny, noiseZ,
                    COORD_SCALE, HEIGHT_SCALE, COORD_SCALE) / UPPER_LIMIT;
            } else {
                double mn = minLimitNoise.sample(noiseX, ny, noiseZ,
                    COORD_SCALE, HEIGHT_SCALE, COORD_SCALE) / LOWER_LIMIT;
                double mx = maxLimitNoise.sample(noiseX, ny, noiseZ,
                    COORD_SCALE, HEIGHT_SCALE, COORD_SCALE) / UPPER_LIMIT;
                density = mn + (mx - mn) * mainN;
            }

            double offset = ((ny - depth) * STRETCH_Y) / scale;
            if (offset < 0) offset *= UNDERDAMP;
            col[ny] = density - offset;
        }
        return col;
    }

    // ── Math helpers ──────────────────────────────────────────────────────────

    private static double lerp(double a, double b, double t) { return a + (b - a) * t; }

    private static double trilinear(
        double v000, double v100, double v010, double v110,
        double v001, double v101, double v011, double v111,
        double fx, double fy, double fz
    ) {
        double x00 = lerp(v000, v100, fx), x10 = lerp(v010, v110, fx);
        double x01 = lerp(v001, v101, fx), x11 = lerp(v011, v111, fx);
        return lerp(lerp(x00, x10, fz), lerp(x01, x11, fz), fy);
    }
}
