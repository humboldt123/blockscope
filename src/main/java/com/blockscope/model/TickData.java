package com.blockscope.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public class TickData {
    public long tick;
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

    public static class PlayerState {
        public double x, y, z;
        public float pitch, yaw;
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
    }

    public static class WorldState {
        public long worldTime;
        public long timeOfDay;
        public String dimension;
        public String biome;
        public String weather;
    }

    public static class InventoryState {
        public ItemStack[] mainInventory; // 36 slots
        public ItemStack[] armor;         // 4 slots
        public ItemStack offhand;
    }

    public static class GuiState {
        public String screenType;        // null if no GUI open
        public int cursorX, cursorY;
        public Integer hoveredSlotIndex; // Index of hovered slot in current screen
        public ItemStack[] containerContents; // Container items (chest, furnace, etc.) - null if not a container
        public ItemStack[] craftingGrid;      // Crafting grid items (2x2 or 3x3) - null if not crafting
        public ItemStack craftingOutput;      // Crafting result slot - null if not crafting
        public boolean recipeBookOpen;        // Recipe book visibility
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
        public Integer damage;          // Item damage/durability (null if not applicable)
        public Integer maxDamage;       // Max durability (null if not damageable)
        public String customName;       // Custom item name (null if none)
        public String[] enchantments;   // Array of enchantment strings (null if none)

        public ItemStack(String id, int count) {
            this.id = id;
            this.count = count;
        }

        public ItemStack(String id, int count, Integer damage, Integer maxDamage, String customName, String[] enchantments) {
            this.id = id;
            this.count = count;
            this.damage = damage;
            this.maxDamage = maxDamage;
            this.customName = customName;
            this.enchantments = enchantments;
        }
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
