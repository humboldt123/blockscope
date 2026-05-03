package com.blockscope.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class TickData {
    public long tick;
    public long replayMs;   // milliseconds into the .mcpr recording (from PacketListener.getCurrentDuration)
    public PlayerState player;
    public WorldState world;
    public InventoryState inventory;  // NOTE: Only present when inventory changes (optimization)
    public GuiState gui;
    public CrosshairTarget crosshairTarget;
    public String gamemode;
    public NearbyEntitiesState nearbyEntities;  // NOTE: Only present when entities change (delta encoding)
    public NearbyBlocksState nearbyBlocks;      // NOTE: Only present when blocks change (delta encoding)

    private static final Gson GSON = new GsonBuilder().create();

    public String toJson() {
        return GSON.toJson(this);
    }

    public static class CameraState {
        public double x, y, z;   // Actual rendered camera position (collision-adjusted in 3rd-person)
        public float pitch, yaw; // Rendered camera angles (differs from player in third_person_front)
    }

    public static class PlayerState {
        public double x, y, z;
        public float pitch, yaw; // Player entity angles (NOT camera angles in F5-front)
        public CameraState camera; // Actual rendered camera — use this for ray-casting
        public float fov;
        public double mouseSensitivity;
        public float health;
        public int hunger;              // Food level (0-20)
        public float saturation;        // Food saturation
        public int armor;               // Armor value (0-20, protection stat)
        public int hotbarIndex;
        public ItemStack heldItem;

        // UI state flags
        public boolean chatOpen;
        public boolean isPaused;        // Singleplayer only; always false in multiplayer
        public boolean menuOpen;        // True if any screen is open (inventory, pause menu, etc.)
        public boolean textInputActive; // True if ANY text field is focused (chat, sign, book, anvil, creative search, command block, etc.)
        public boolean hideGui;         // F2 pressed (UI hidden)
        public boolean debugScreenVisible; // F3 pressed (debug overlay)

        // Camera settings
        public String cameraPerspective; // "first_person", "third_person_back", "third_person_front" (F5)

        // Accessibility/gameplay settings
        public boolean showSubtitles;   // Captions enabled (visual sound indicators)
        public boolean autoJumpEnabled; // Auto-jump setting (disabled by mod for 1.8.9 compatibility)

        // Active status effects
        public StatusEffect[] statusEffects; // Active potion effects (speed, strength, etc.)
    }

    public static class StatusEffect {
        public String effectId;     // e.g. "minecraft:speed"
        public int amplifier;       // Effect level (0 = I, 1 = II, etc.)
        public int duration;        // Remaining ticks
        public boolean ambient;     // From beacon (less visual particles)
        public boolean showParticles;
        public boolean showIcon;
    }

    public static class WorldState {
        public long worldTime;
        public long timeOfDay;
        public String dimension;
        public String biome;
        public String weather;
    }

    public static class InventoryState {
        // Player inventory using Minecraft's slot numbering:
        // Slots 0-8: Hotbar
        // Slots 9-35: Main inventory (3 rows of 9)
        // Slots 36-39: Armor (boots, leggings, chestplate, helmet)
        // Slot 40: Offhand
        public SlotItem[] slots; // Only non-empty slots (sparse array representation)
        public ItemStack cursorStack; // Item held by mouse cursor (null if none)
    }

    // Represents an item in a specific slot
    public static class SlotItem {
        public int slot;      // Slot index
        public ItemStack item; // The item (never null in this object)
    }

    public static class GuiState {
        public String screenType;        // null if no GUI open
        public int cursorX, cursorY;
        public Integer hoveredSlotIndex; // Slot index of hovered slot

        // Container data (chest, furnace, hopper, etc.)
        public ContainerData container; // null if not viewing a container

        // Crafting data (when player inventory is open or at crafting table)
        public CraftingData crafting;   // null if no crafting grid visible

        public boolean recipeBookOpen;  // Recipe book visibility
    }

    public static class ContainerData {
        public String type;              // "minecraft:chest", "minecraft:furnace", "minecraft:hopper", etc.
        public SlotItem[] slots;         // Container slots (only non-empty slots)
        public Integer size;             // Total container size (e.g., 27 for chest, 5 for hopper)
    }

    public static class CraftingData {
        public SlotItem[] craftingGrid;  // Crafting input slots (only non-empty)
        public ItemStack result;         // Crafting result slot (null if no result)
        public boolean largeCraftingGrid; // true = 3x3 (crafting table), false = 2x2 (inventory)
    }

    public static class CrosshairTarget {
        public String type; // "block", "entity", or "none"
        public BlockPos blockPos;
        public String blockId;
        public String entityUuid;
        public String entityType;
    }

    public static class BlockPos {
        public int x, y, z;

        public BlockPos(int x, int y, int z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    public static class ItemStack {
        public String id;
        public int count;
        public Integer damage;          // Current damage value (null if not damageable)
        public Integer maxDamage;       // Max damage value (null if not damageable)
        public Float durability;        // Durability as 0-1 (1.0 = pristine, 0.0 = broken, null if not damageable)
        public Enchantment[] enchantments; // Enchantments (null if none)

        public ItemStack(String id, int count) {
            this.id = id;
            this.count = count;
        }
    }

    public static class Enchantment {
        public String id;    // e.g. "minecraft:sharpness"
        public int level;    // Enchantment level (1 = I, 2 = II, etc.)
    }

    // Phase 4: Nearby Entity State
    public static class NearbyEntitiesState {
        public EntityData[] entities;
    }

    public static class EntityData {
        public String uuid;                    // Unique identifier for tracking across ticks
        public String type;                    // "minecraft:cow", "minecraft:zombie", etc.
        public double x, y, z;                 // Absolute position
        public float pitch, yaw;               // Rotation
        public Float health;                   // Nullable - for living entities
        public Float maxHealth;                // Nullable - for living entities
        public EquipmentData equipment;        // Nullable - for living entities
        public ItemStack item;                 // Nullable - for item entities
        public Integer age;                    // Nullable - for item entities (ticks since spawned)
        public Boolean onFire;                 // Fire status
        public Boolean invisible;              // Invisibility status
        public String customName;              // Nullable - custom name tag
    }

    public static class EquipmentData {
        public ItemStack mainHand;
        public ItemStack offhand;
        public ItemStack helmet;
        public ItemStack chestplate;
        public ItemStack leggings;
        public ItemStack boots;
    }

    // Phase 4: Nearby Block State
    public static class NearbyBlocksState {
        public BlockData[] blocks;
    }

    public static class BlockData {
        public int x, y, z;                    // Block position (absolute or relative to player)
        public String blockId;                 // "minecraft:stone", "minecraft:oak_stairs", etc.
        public String blockStateProperties;    // CRITICAL: "facing=north,half=bottom,waterlogged=false"
        public BlockEntityData blockEntity;    // Nullable - for block entities (chests, furnaces, etc.)
    }

    public static class BlockEntityData {
        public String type;                    // "minecraft:chest", "minecraft:furnace", etc.
        public ItemStack[] items;              // Nullable - for containers
        public Integer burnTime;               // Nullable - for furnaces (burn time remaining)
        public Integer cookTime;               // Nullable - for furnaces (cook progress)
        public String[] signText;              // Nullable - for signs (4 lines of text)
    }
}
