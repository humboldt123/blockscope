# Prism Instance Setup

Config files for the Blockscope dev Minecraft environment (MC 1.19.4, Fabric 0.14.25).

## Automatic setup

```bash
python scripts/setup_prism.py
```

Detects your platform and installs config files into the right PrismLauncher directory. Then add mods manually:

| Mod | Version | Source |
|-----|---------|--------|
| `blockscope-*.jar` | latest | `cd mods/blockscope && ./gradlew build` → `build/libs/` |
| `replaymod-1.19.4-2.6.20.jar` | 2.6.20 | [replaymod.com/download](https://www.replaymod.com/download/) |
| `baritone-fabric-1.9.5.jar` | 1.9.5 | [github.com/cabaletta/baritone/releases](https://github.com/cabaletta/baritone/releases) |
| `fabric-api-0.87.0+1.19.4.jar` | 0.87.0 | [modrinth.com/mod/fabric-api](https://modrinth.com/mod/fabric-api) |

## Manual setup

**Mac:** `~/Library/Application Support/PrismLauncher/instances/Blockscope/`

**Windows:** `%APPDATA%\PrismLauncher\instances\Blockscope\`

Copy from this directory:
- `mmc-pack.json` → instance root
- `options.txt` → `minecraft/`
- `blockscope.properties` → `minecraft/`
- `config/replaymod.json` → `minecraft/config/`

## Notes

- `blockscope.properties`: `server_url` defaults to `http://localhost:9000` for local dev. Set it to the production server URL for real data collection.
- `options.txt` includes all keybinds: `R` toggles recording, `B` opens blockscope config. ReplayMod binds are also set.
- On Mac, `./gradlew` in `mods/blockscope/` should work directly — the wrapper downloads Gradle automatically.
