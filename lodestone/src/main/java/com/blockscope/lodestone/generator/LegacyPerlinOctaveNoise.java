package com.blockscope.lodestone.generator;

import java.util.Random;

/**
 * Ported from moderner-beta's PerlinOctaveNoise.java.
 * Only the two sampling methods needed for beta/release terrain are kept.
 * Mth.lfloor replaced with (long)Math.floor().
 */
class LegacyPerlinOctaveNoise {

    private final LegacyPerlinNoise[] noises;
    private final int octaves;
    private final boolean wrapped;

    LegacyPerlinOctaveNoise(Random rng, int octaves, boolean wrapped) {
        this.octaves = octaves;
        this.wrapped = wrapped;
        this.noises = new LegacyPerlinNoise[octaves];
        for (int i = 0; i < octaves; i++) noises[i] = new LegacyPerlinNoise(rng);
    }

    /**
     * 2D XZ noise — used for scale and depth column sampling.
     */
    double sampleXZ(double x, double z, double scaleX, double scaleZ) {
        double total = 0.0;
        double frequency = 1.0;
        for (int i = 0; i < octaves; i++) {
            double offX = x * scaleX * frequency;
            double offZ = z * scaleZ * frequency;
            if (wrapped) {
                long ox = (long) Math.floor(offX), oz = (long) Math.floor(offZ);
                offX -= ox; offZ -= oz;
                ox %= 16777216L; oz %= 16777216L;
                offX += ox; offZ += oz;
            }
            total += noises[i].sampleXZ(offX, offZ, frequency);
            frequency /= 2.0;
        }
        return total;
    }

    /**
     * 3D noise with per-axis scale — used for main, minLimit, maxLimit density noises.
     * Matches moderner-beta's non-infdevNoiseScaling path.
     */
    double sample(double x, double y, double z, double scaleX, double scaleY, double scaleZ) {
        double total = 0.0;
        double frequency = 1.0;
        for (int i = 0; i < octaves; i++) {
            double offX = x * scaleX * frequency;
            double offZ = z * scaleZ * frequency;
            if (wrapped) {
                long ox = (long) Math.floor(offX), oz = (long) Math.floor(offZ);
                offX -= ox; offZ -= oz;
                ox %= 16777216L; oz %= 16777216L;
                offX += ox; offZ += oz;
            }
            total += noises[i].sampleXYZ(
                offX,
                y * scaleY * frequency,
                offZ,
                scaleY * frequency,
                y * scaleY * frequency
            ) / frequency;
            frequency /= 2.0;
        }
        return total;
    }
}
