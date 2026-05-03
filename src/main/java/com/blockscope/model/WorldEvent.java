package com.blockscope.model;

/**
 * Event-based world state tracking for efficient storage.
 *
 * ARCHITECTURE: Master Map + Diffs
 * ================================
 * Instead of saving 8x8x8 cube every tick, we use a smarter approach:
 *
 * 1. MASTER MAP: Accumulate all blocks/entities the player has SEEN (discovered via rendering)
 *    - Stored with absolute coordinates + dimension
 *    - Persisted as "block_seen" and "entity_seen" events
 *
 * 2. DIFF TRACKING: Only log CHANGES after initial discovery
 *    - Block updates: "block_changed" events
 *    - Entity moves: "entity_moved" events
 *    - Entity metadata changes: "entity_updated" events
 *
 * 3. PERIODIC SNAPSHOTS: Every ~1 minute (1200 ticks), write full nearby state
 *    - Prevents unbounded diff chains
 *    - Ensures data integrity if player is in one area for extended time
 *
 * 4. TRAINING PIPELINE RECONSTRUCTION (future implementation):
 *    - Load master map from all "seen" events
 *    - Apply diffs chronologically up to current tick
 *    - Extract 8x8x8 cube around player position at each tick
 *    - This "decompression" happens during model training, not during recording
 *
 * STORAGE BENEFITS:
 * - Reduces file size by 90%+ for typical gameplay (player stays in one area)
 * - Scales better for exploration (only save new blocks when discovered)
 * - Handles chunk unload/reload gracefully (blocks reappear in master map)
 */
public class WorldEvent {
    public int tick;
    public String event;  // "chunk_load", "camera_update", "block_changed", etc.

    // Chunk-related fields (for chunk_packet - raw NBT from server)
    public String dimension;
    public Integer chunkX, chunkZ;
    public String chunkNbt; // Raw NBT packet data as string (ReplayMod approach)

    // Camera-related fields (for camera_update - recorded each tick)
    public CameraData camera;

    // Block-related fields (for block_changed)
    public Integer x, y, z;
    public String blockId;
    public String blockStateProperties;  // Rotation, facing, powered, etc.
    public TickData.BlockEntityData blockEntity;  // Reuse TickData's BlockEntityData

    // Entity-related fields (for entity_seen, entity_moved, entity_updated)
    public String entityUuid;
    public String entityType;
    public Double entityX, entityY, entityZ;
    public Float entityPitch, entityYaw;
    public Float entityHealth, entityMaxHealth;
    public TickData.EquipmentData entityEquipment;  // Reuse TickData's EquipmentData
    public TickData.ItemStack entityItem;  // For item entities
    public Integer entityAge;              // For item entities
    public Boolean onFire, invisible;
    public String customName;

    // Snapshot fields (for periodic full dumps)
    public SnapshotData snapshot;

    public static class CameraData {
        public double x, y, z;
        public float pitch, yaw;
        public float fov;
        public String perspective;  // "first_person", "third_person_back", "third_person_front"
    }

    public static class SnapshotData {
        public BlockData[] nearbyBlocks;     // All blocks within 8x8x8
        public EntityData[] nearbyEntities;  // All entities within radius
    }

    public static class BlockData {
        public String dimension;
        public int x, y, z;
        public String blockId;
        public String blockStateProperties;
        public TickData.BlockEntityData blockEntity;  // Reuse TickData's BlockEntityData
    }

    public static class EntityData {
        public String uuid;
        public String type;
        public double x, y, z;
        public float pitch, yaw;
        public Float health, maxHealth;
        public TickData.EquipmentData equipment;  // Reuse TickData's EquipmentData
        public TickData.ItemStack item;
        public Integer age;
        public Boolean onFire, invisible;
        public String customName;
    }
}
