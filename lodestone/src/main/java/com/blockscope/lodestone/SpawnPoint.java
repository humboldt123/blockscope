package com.blockscope.lodestone;

/** A single spawn candidate for a library world: integer block coords plus a label. */
public final class SpawnPoint {
    public final int x;
    public final int y;
    public final int z;
    public final String label;

    public SpawnPoint(int x, int y, int z, String label) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.label = (label == null || label.isEmpty()) ? "grid" : label;
    }

    @Override
    public String toString() {
        return label + " @ " + x + "," + y + "," + z;
    }
}
