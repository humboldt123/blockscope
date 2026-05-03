# Blockscope

**AI Training Data Collection Mod for Minecraft**

Blockscope is a Fabric mod that records rich, synchronized gameplay data for training AI models. It captures H.264 video, keyboard/mouse inputs, and comprehensive game state data with precise tick-level synchronization.

Also See: [Blockscope Visualizer](https://github.com/humboldt123/blockscope-visualizer)



## Dev Stuff

### Build from Source

```bash
# output in build/libs/blockscope-0.1.0-alpha.jar put that in your mods folder
./gradlew build

# or just run this
./gradlew runClient
```

## Installation

### Requirements
- **Minecraft**: 1.16.5 (Cross-version compatibility coming soon)
- **Fabric Loader**: 0.14.0 or later
- **Fabric API**: 0.42.0 or later
- **Java**: 8 or later (8, 11, 17, 21 all supported)

### Steps
1. Install [Fabric Loader](https://fabricmc.net/use/) for Minecraft 1.16.5
2. Download [Fabric API](https://modrinth.com/mod/fabric-api/versions) 0.42.0+1.16
3. Place `blockscope-0.1.0-alpha.jar` and `fabric-api-0.42.0+1.16.jar` in `.minecraft/mods/`
4. Launch Minecraft 1.16.5 with Fabric


## Usage

### Controls
- **Toggle Recording**: Press `R` (or your configured key)
- **Open Config GUI**: Press `B` (or your configured key)

### Recording Sessions
When you start recording, Blockscope creates a new session directory:

```
recordings/
└── session_1707829483/
    ├── metadata.json          # Session info, config, timestamps
    ├── keybindings.json       # User's keybind configuration
    ├── ticks.jsonl            # Game state data (one JSON per tick)
    ├── inputs.jsonl           # Input events (keyboard/mouse)
    └── video.mp4              # Saved MP4 (20 FPS by default to match 20tps)
```

## Data Schema

### Inventory Schema (Minecraft Slot Numbering)

Player inventory uses Minecraft's standard slot indices (0-40):

```
Slots 0-8:   Hotbar (9 slots)
Slots 9-35:  Main Inventory (27 slots, 3 rows × 9 columns)
Slots 36-39: Armor (boots, leggings, chestplate, helmet)
Slot 40:     Offhand
```

**Sparse representation:** Only non-empty slots are stored as `{slot, item}` objects.

**Cursor stack:** Item held by mouse cursor is stored separately in `inventory.cursorStack`.

#### Example Inventory JSON
```json
{
  "inventory": {
    "slots": [
      {"slot": 0, "item": {"id": "minecraft:diamond_sword", "count": 1, "durability": 0.95, "enchantments": [...]}},
      {"slot": 1, "item": {"id": "minecraft:cobblestone", "count": 64}},
      {"slot": 36, "item": {"id": "minecraft:diamond_boots", "count": 1, "durability": 0.82}},
      {"slot": 40, "item": {"id": "minecraft:shield", "count": 1}}
    ],
    "cursorStack": {"id": "minecraft:dirt", "count": 32}
  }
}
```

### Container Schema

When a player opens a container (chest, furnace, hopper, shulker, etc.), the container's contents are captured:

```json
{
  "gui": {
    "screenType": "ChestScreen",
    "container": {
      "type": "minecraft:chest",
      "size": 27,
      "slots": [
        {"slot": 0, "item": {"id": "minecraft:iron_ingot", "count": 32}},
        {"slot": 13, "item": {"id": "minecraft:diamond", "count": 5}}
      ]
    }
  }
}
```

Container slots are indexed starting from 0 (separate from player inventory slots).

### Crafting Schema

Crafting grid data is captured when using the player inventory (2×2) or crafting table (3×3):

```json
{
  "gui": {
    "screenType": "CraftingScreen",
    "crafting": {
      "largeCraftingGrid": true,
      "craftingGrid": [
        {"slot": 0, "item": {"id": "minecraft:stick", "count": 1}},
        {"slot": 3, "item": {"id": "minecraft:diamond", "count": 1}},
        {"slot": 4, "item": {"id": "minecraft:diamond", "count": 1}}
      ],
      "result": {"id": "minecraft:diamond_pickaxe", "count": 1}
    }
  }
}
```

**Crafting grid slots:**
- 2×2 grid (inventory): slots 0-3
- 3×3 grid (crafting table): slots 0-8

### Item Metadata

All items include:
- **id**: Namespaced item ID (e.g., `"minecraft:diamond_sword"`)
- **count**: Stack size (1-64)
- **durability**: Percentage (0.0-1.0, where 1.0 = pristine, 0.0 = broken) - `null` if not damageable
- **damage**: Current damage value - `null` if not damageable
- **maxDamage**: Maximum damage value - `null` if not damageable
- **enchantments**: Array of `{id, level}` objects - `null` if no enchantments

#### Example Enchanted Item
```json
{
  "id": "minecraft:diamond_sword",
  "count": 1,
  "damage": 50,
  "maxDamage": 1561,
  "durability": 0.968,
  "enchantments": [
    {"id": "minecraft:sharpness", "level": 5},
    {"id": "minecraft:unbreaking", "level": 3},
    {"id": "minecraft:mending", "level": 1}
  ]
}
```

### Status Effects

Player status effects (potions) are captured every tick:

```json
{
  "player": {
    "statusEffects": [
      {
        "effectId": "minecraft:speed",
        "amplifier": 1,
        "duration": 342,
        "ambient": false,
        "showParticles": true,
        "showIcon": true
      }
    ]
  }
}
```

**Fields:**
- **effectId**: Effect type (e.g., `"minecraft:speed"`, `"minecraft:strength"`)
- **amplifier**: Effect level (0 = I, 1 = II, 2 = III, etc.)
- **duration**: Remaining ticks
- **ambient**: `true` if from beacon (reduced particles)
- **showParticles**: Whether particles are visible
- **showIcon**: Whether HUD icon is visible

## Thanks!

Built with:
- [Fabric](https://fabricmc.net/) - Minecraft modding framework
- [FFmpeg](https://ffmpeg.org/) - H.264 video encoding
- [LWJGL](https://www.lwjgl.org/) - OpenGL bindings for frame capture
