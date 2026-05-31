# pum — furnace/pipeline

GPU-based Minecraft 1.19.4 block visibility labeler. Replaces the V1 raycaster in `furnace/pipeline/` which was fundamentally broken (inverted visibility — marked visible blocks as invisible and vice versa).

TODO: add entitiy support when we need it lol

## What it does

For each tick of a recorded Minecraft session, the pipeline computes a uint8 `(32, 32, 32)` visibility mask: `m[i,j,k] = 1` if block cell `(i,j,k)` in the 32³ window centered on the player was geometrically visible from the camera that frame.

The window covers world coordinates `[px-15, px+16) × [py-15, py+16) × [pz-15, pz+16)` where `(px, py, pz) = floor(camera position)`.

Output per tick: `labels/tick_NNNNN.npz` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `m` | uint8 (32,32,32) | Visibility mask |
| `b` | int32 (32,32,32) | Block state IDs |
| `inventory` | JSON string | Player hotbar/inventory |
| `pose` | JSON string | Camera position, yaw, pitch, FOV |

## Why it exists

V1 (`furnace/pipeline/src/python/rasterizer.py`) used a Numba DDA raycaster that produced systematically wrong masks. Rather than debug it, it uses a proper GPU pipeline: render the 32³ window from the player's exact camera, read back which pixels have non-zero alpha, decode the block index encoded as (R,G,B) color per fragment.

## Architecture

```
pipeline/
├── baker.py          # One-time: extract JAR assets → geometry cache + texture atlas
├── renderer.py       # Per-tick: GPU render passes → visibility mask / RGB image
├── pipeline.py       # Batch: JS stage → baker → renderer → NPZ files
├── viewer.py         # Interactive side-by-side: video frame vs GPU render
├── orbit.py          # Diagnostic: orbit video around one tick
└── scripts/
    └── gen_state_properties.js   # Node.js: state ID → {name, properties} map
```

### Baker

Runs once per JAR version. Steps:

1. Extracts `assets/minecraft/` from the JAR zip
2. Generates `state_properties_{version}.json` via Node.js + `minecraft-data`
3. Loads `Minecraft-Model-Reader` with the extracted assets
4. For each of the 23,725 block state IDs: loads the block mesh, extracts triangle geometry + UV coordinates
5. Builds a texture atlas PNG with biome tints baked in
6. Saves `geometry_{version}.npz` — per-state vertex arrays with atlas UVs, indexed by cull direction

### Renderer

Two GPU render passes per tick (ModernGL offscreen, no window required):

- **Pass 1 — opaque**: depth write ON, alpha cutout. Blocks with `Transparency=FullOpaque` or `Partial`. Fragment shader writes block index as `(i/255, j/255, k/255)` to the RGBA color buffer with alpha=1.
- **Pass 2 — translucent**: depth write OFF, `LEQUAL` depth test, blending ON. Only fluid blocks (water/lava). Same index encoding.

Visibility is decoded by reading both readbacks from the color texture attachment and unioning all `(R,G,B)` values where `alpha > 0`.

### Camera

Uses the same math as `furnace/pipeline/src/python/io_helpers.py`. Camera position comes directly from `ticks.jsonl` (the mod logs the actual eye position via `GameRenderer.getCamera()`). FOV is taken from the tick data and includes potion effects.

## Prerequisites

- Python ≥ 3.12 managed by `uv`
- Node.js ≥ 18 (for `parse_mcpr.js` and `gen_state_properties.js`)
- Minecraft 1.19.4 JAR at `visualizer/.cache/minecraft-1.19.4.jar`
- `furnace/Minecraft-Model-Reader/` cloned (included in repo)

## Setup

```bash
cd furnace/pipeline
uv sync          # creates .venv, installs all Python deps
```

## Dependencies

### Python (managed by uv / pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| `moderngl` | ≥5.12.0 | OpenGL offscreen rendering (standalone context) |
| `numpy` | ≥1.20 | Array ops, vertex buffers, readback decoding |
| `pillow` | ≥12.2.0 | Texture atlas loading and saving |
| `av` | ≥16.0.0 | PyAV — video frame extraction and orbit MP4 encoding |
| `pygame` | ≥2.6.1 | Viewer window and display |
| `pyglm` | ≥2.8.3 | (listed; currently unused — view matrix built with numpy) |
| `amulet-nbt` | ≥2.0 | Required by Minecraft-Model-Reader for NBT tag types |
| `minecraft-resource-pack` | local path | Minecraft-Model-Reader — block model loading |

### Vendored (pipeline/vendor/)

| Package | Origin | Purpose |
|---------|--------|---------|
| `minecraft_model_reader` | [Amulet MC](https://github.com/Amulet-Team/Minecraft-Model-Reader) (MIT) | Block model loading — 19 Java-path files, Bedrock stripped |
| `amulet_nbt` | [Amulet MC](https://github.com/Amulet-Team/amulet-nbt) (MIT) | 50-line shim; only `TAG_String` needed at runtime |

### Node.js (in `furnace/pipeline/node_modules/`)

| Package | Purpose |
|---------|---------|
| `minecraft-data` | State ID → block properties mapping (baker) |
| `prismarine-chunk` | Chunk decoding from .mcpr replay packets (JS stage) |
| `prismarine-nbt` | NBT parsing (JS stage) |
| `prismarine-world` | World state reconstruction (JS stage) |
| `yauzl` | ZIP reading of .mcpr files (JS stage) |
| `msgpack5` | MessagePack decoding (JS stage) |
| `vec3` | 3D vector math (JS stage) |

### System

- **Node.js ≥ 18** — runs `parse_mcpr.js` and `gen_state_properties.js`
- **Minecraft 1.19.4 JAR** — asset source; extracted once by baker into `cache/1.19.4/extracted_1.19.4/`

## Commands

### Baker

```bash
uv run python -m pipeline.baker [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--jar PATH` | `visualizer/.cache/minecraft-1.19.4.jar` | Minecraft JAR to extract assets from |
| `--version STR` | `1.19.4` | Version tag; sets cache subdirectory |
| `--fast-graphics` | off | Bake leaves as fully opaque (matches Minecraft "Fast" graphics recordings) |

Outputs to `cache/{version}/` (or `cache/{version}-fast/` with `--fast-graphics`). Safe to re-run — skips if cache already exists. Delete the `.npz` and `.png` files to force a re-bake.

---

### Pipeline

```bash
uv run python -m pipeline.labeler --session PATH [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session PATH` | required | Path to session directory |
| `--version STR` | `1.19.4` | Minecraft version |
| `--jar PATH` | auto-detected | JAR path (only needed if baker hasn't run yet) |
| `--max-ticks N` | all | Stop after N ticks (useful for testing) |
| `--fast-graphics` | off | Use fast-graphics cache; leaves are opaque so blocks behind them are correctly excluded from visibility |

Stages run automatically if needed: JS parse → baker → per-tick render. Already-processed ticks are skipped.

**For sessions recorded on "Fast" graphics, always pass `--fast-graphics`.**

---

### Viewer

```bash
uv run python -m pipeline.viewer --session PATH [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session PATH` | required | Path to session directory |
| `--tick-offset N` | `1` | `frame_mapping.jsonl` tick correction |
| `--version STR` | `1.19.4` | Cache version for rendering (always use fancy/default here) |

Keys: `←` / `→` step ticks, `Q` / `Esc` quit.

Left panel: video frame extracted from `video.mp4`. Right panel: GPU render of only the blocks in the tick's visibility mask.

Ticks with `frame None` in the title bar have no corresponding video frame (the game processed those ticks during a loading screen or a dropped frame). These are normal and should be filtered out at training time.

---

### Orbit video

```bash
uv run python -m pipeline.orbit --session PATH --tick N --output orbit.mp4 [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--session PATH` | required | |
| `--tick N` | required | Tick index to orbit around |
| `--output PATH` | `orbit.mp4` | Output MP4 |
| `--radius FLOAT` | `15.0` | Orbit radius in blocks |
| `--elevation FLOAT` | `5.0` | Camera height above player |
| `--frames N` | `72` | Frame count (72 = full 360° at 24fps) |
| `--fps N` | `24` | Output framerate |
| `--show-invisible` | off | Render all blocks, not just visible ones |
| `--version STR` | `1.19.4` | |

---

## Session directory layout

```
session_XXXXXXXXX/
├── *.mcpr               ← replay file (any .mcpr filename)
├── video.mp4            ← 640×360 gameplay video
├── ticks.jsonl          ← one JSON per tick: camera pose, inventory, etc.
├── frame_mapping.jsonl  ← tick → video frame number correspondence
└── labels/              ← written by the pipeline
    ├── world_states.bin ← block states per tick (written by JS stage)
    └── tick_NNNNN.npz   ← visibility labels
```

## Cache layout

```
pipeline/cache/
├── 1.19.4/                          ← fancy graphics (leaves translucent)
│   ├── extracted_1.19.4/            ← JAR assets + synthesised pack.mcmeta
│   ├── state_properties_1.19.4.json ← stateId → {name, properties}
│   ├── geometry_1.19.4.npz          ← baked vertex data for all 23,725 states
│   ├── atlas_1.19.4.png             ← texture atlas (tints baked in)
│   └── meta_1.19.4.json             ← atlas dimensions, fluid UV rects
└── 1.19.4-fast/                     ← fast graphics (leaves fully opaque)
    ├── geometry_1.19.4-fast.npz
    ├── atlas_1.19.4-fast.png
    └── meta_1.19.4-fast.json
```

## Frame sync notes

`frame_mapping.jsonl` maps server ticks to video frame numbers. Not every tick has a corresponding video frame — loading screens and occasional dropped frames leave gaps. These ticks have `frame None` in the viewer and should be skipped in training data loaders:

```python
valid_ticks = [t for t in range(tick_count) if frame_mapping.get(t) is not None]
```

The first ~30 ticks of a session typically have empty worlds (chunks not yet loaded in the replay). `m` will be all-zero for those ticks, which is correct.

---

## What V1 has that this pipeline does not

| Feature | Notes |
|---------|-------|
| Lighting computation | `lighting.py` computes sky + block light level per cell; used for Beta 1.7 shading in the V1 CPU renderer. Not needed. |
| `texture_map_1.19.4.json` | Per-state face→texture name mapping for the V1 CPU renderer. Not needed. |
