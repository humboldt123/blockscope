package com.blockscope.lodestone.generator;

import java.util.Random;

/**
 * Ported from moderner-beta's PerlinNoise.java.
 * NMS (net.minecraft.util.Mth) replaced with pure Java equivalents.
 */
class LegacyPerlinNoise {

    private final int[] p;

    double offsetX, offsetY, offsetZ;

    LegacyPerlinNoise(Random rng) {
        p = new int[512];
        offsetX = rng.nextDouble() * 256.0;
        offsetY = rng.nextDouble() * 256.0;
        offsetZ = rng.nextDouble() * 256.0;

        for (int i = 0; i < 256; i++) p[i] = i;
        for (int i = 0; i < 256; i++) {
            int j = rng.nextInt(256 - i) + i;
            int k = p[i]; p[i] = p[j]; p[j] = k;
            p[i + 256] = p[i];
        }
    }

    // Standard 3D sample (with offset applied).
    double sample(double x, double y, double z) {
        x += offsetX; y += offsetY; z += offsetZ;
        int fx = mfloor(x), fy = mfloor(y), fz = mfloor(z);
        int X = fx & 0xFF, Y = fy & 0xFF, Z = fz & 0xFF;
        x -= fx; y -= fy; z -= fz;
        double u = fade(x), v = fade(y), w = fade(z);
        int A = p[X]+Y, AA = p[A]+Z, AB = p[A+1]+Z;
        int B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;
        return lerp(w,
            lerp(v, lerp(u, grad(p[AA],   x,   y,   z), grad(p[BA],   x-1, y,   z)),
                    lerp(u, grad(p[AB],   x,   y-1, z), grad(p[BB],   x-1, y-1, z))),
            lerp(v, lerp(u, grad(p[AA+1], x,   y,   z-1), grad(p[BA+1], x-1, y,   z-1)),
                    lerp(u, grad(p[AB+1], x,   y-1, z-1), grad(p[BB+1], x-1, y-1, z-1)))
        );
    }

    // 2D XZ sample used by PerlinOctaveNoise.sampleXZ.
    double sampleXZ(double x, double z, double frequency) {
        frequency = 1.0 / frequency;
        x += offsetX; z += offsetZ;
        int fx = mfloor(x), fz = mfloor(z);
        int X = fx & 0xFF, Z = fz & 0xFF;
        x -= fx; z -= fz;
        double u = fade(x), w = fade(z);
        int A = p[X]+0, AA = p[A]+Z, B = p[X+1]+0, BA = p[B]+Z;
        return lerp(w,
            lerp(u, grad(p[AA],   x,   0, z), grad(p[BA],   x-1, 0, z)),
            lerp(u, grad(p[AA+1], x,   0, z-1), grad(p[BA+1], x-1, 0, z-1))
        ) * frequency;
    }

    // 3D XYZ sample with Y-scale clamping (from vanilla PerlinNoiseSampler).
    double sampleXYZ(double x, double y, double z, double yScale, double yMax) {
        x += offsetX; y += offsetY; z += offsetZ;
        int fx = mfloor(x), fy = mfloor(y), fz = mfloor(z);
        x -= fx; z -= fz;

        double yOffset = 0.0;
        if (yScale != 0.0) {
            yOffset = (yMax >= 0.0 && yMax < y) ? yMax : y;
            yOffset = Math.floor(yOffset / yScale + 1.0000000116860974E-7) * yScale;
        }
        double localY = y - yOffset;

        int X = fx & 0xFF, Y = fy & 0xFF, Z = fz & 0xFF;
        int A = p[X]+Y, AA = p[A]+Z, AB = p[A+1]+Z;
        int B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;

        double g0 = grad(p[AA],   x,   localY,   z);
        double g1 = grad(p[BA],   x-1, localY,   z);
        double g2 = grad(p[AB],   x,   localY-1, z);
        double g3 = grad(p[BB],   x-1, localY-1, z);
        double g4 = grad(p[AA+1], x,   localY,   z-1);
        double g5 = grad(p[BA+1], x-1, localY,   z-1);
        double g6 = grad(p[AB+1], x,   localY-1, z-1);
        double g7 = grad(p[BB+1], x-1, localY-1, z-1);

        double u = fade(x), v = fade(y), w = fade(z);
        return lerp3(u, v, w, g0, g1, g2, g3, g4, g5, g6, g7);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static int mfloor(double x) { return (int) Math.floor(x); }

    private static double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    private static double lerp(double t, double a, double b) { return a + t * (b - a); }

    private static double lerp3(double u, double v, double w,
                                 double d000, double d100, double d010, double d110,
                                 double d001, double d101, double d011, double d111) {
        return lerp(w,
            lerp(v, lerp(u, d000, d100), lerp(u, d010, d110)),
            lerp(v, lerp(u, d001, d101), lerp(u, d011, d111)));
    }

    private static double grad(int hash, double x, double y, double z) {
        switch (hash & 0xF) {
            case 0x0: return  x + y;
            case 0x1: return -x + y;
            case 0x2: return  x - y;
            case 0x3: return -x - y;
            case 0x4: return  x + z;
            case 0x5: return -x + z;
            case 0x6: return  x - z;
            case 0x7: return -x - z;
            case 0x8: return  y + z;
            case 0x9: return -y + z;
            case 0xA: return  y - z;
            case 0xB: return -y - z;
            case 0xC: return  y + x;
            case 0xD: return -y + z;
            case 0xE: return  y - x;
            case 0xF: return -y - z;
            default:  return 0;
        }
    }
}
