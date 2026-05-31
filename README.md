# PlayerYouMean

Research, experiments, training, data collection infrastructure, etc. for training Minecraft AI agents. See `paper/` for the research design.

<br>
<blockquote>
<p><em>I see the player you mean.</em></p>
<p align="right">PLAYERNAME?</p>
<p><em>Yes. Take care. It has reached a higher level now. It can read our thoughts.</em></p>
<p align="right">— <em>Minecraft End Poem</em></p>
</blockquote>
<br>

## Components

| | Description |
|-|-------------|
| [`mods/blockscope/`](mods/blockscope/) | Fabric mod — records gameplay (video, inputs, tick state) and uploads to the server |
| [`mods/replaymod/`](mods/replaymod/) | Records raw network packets alongside blockscope for ground-truth world reconstruction |
| [`mods/baritone/`](mods/baritone/) | Pathfinding AI, driven via its API from blockscope agent modes |
| [`hopper/`](hopper/) | FastAPI upload endpoint — runs in the cloud, receives sessions from N mod users |
| [`furnace/`](furnace/) | Processing pipeline — converts raw `.mcpr` + metadata into voxel grids with visibility masks |
| [`visualizer/`](visualizer/) | OpenGL viewer — inspect collected sessions interactively |

## Data flow

```
N players running blockscope + replaymod
            │
            │  uploads: video, inputs, tick metadata, .mcpr
            ▼
      hopper  (FastAPI, Brev cloud)
            │
            │  raw recordings pulled for processing
            ▼
      furnace  (.mcpr → voxel grids + visibility masks)
            │
            ▼
      training  (on server — not in this repo)
```

## Dev setup

```bash
# Set up PrismLauncher instance (MC 1.19.4 + Fabric)
python scripts/setup_prism.py

# Build the mod
cd mods/blockscope && ./gradlew build
# Copy build/libs/blockscope-*.jar into your Prism instance mods/ folder

# Visualizer assets (run once)
cd visualizer && python extract_mc_assets.py
```

See [`instance/README.md`](instance/README.md) for full setup instructions including which mod jars to download.

## Submodules

After cloning, initialize submodules:

```bash
git submodule update --init --recursive
```
