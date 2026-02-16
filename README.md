# Blockscope

**AI Training Data Collection Mod for Minecraft**

Blockscope is a Fabric mod that records rich, synchronized gameplay data for training AI models. It captures H.264 video, keyboard/mouse inputs, and comprehensive game state data with precise tick-level synchronization.

Also see [Blockscope Visualizer](https://github.com/humboldt123/blockscope-visualizer)
---


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

## Thanks!

Built with:
- [Fabric](https://fabricmc.net/) - Minecraft modding framework
- [JCODEC](http://jcodec.org/) - Pure Java H.264 video encoding
- [LWJGL](https://www.lwjgl.org/) - OpenGL bindings for frame capture
