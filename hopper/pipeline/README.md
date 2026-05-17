# Blockscope Visibility Pipeline

Produces per-tick training labels (visible-block mask `m`, ground-truth blocks `b`, inventory, seen_before `s`) from Minecraft 1.19.4 gameplay recordings.

## Architecture

**Stage 1 (Node.js):** `src/js/parse_mcpr.js`  
Reads the `.mcpr` zip → `recording.tmcpr` binary stream → reconstructs the Minecraft world state incrementally by replaying `map_chunk`, `block_change`, `multi_block_change`, and `respawn` packets. Snapshots the 32³ block window around the player at each tick's `replayMs`. Output: `labels/world_states.bin`.

**Stage 2 (Python):** `src/python/pipeline.py`  
Loads `world_states.bin` and `ticks.jsonl`. Casts 640×360 rays per tick using a Numba-parallel DDA raycaster (`raycaster.py`). Runs a `seen_before` post-pass (`seen_before.py`). Writes per-tick `labels/tick_NNNNN.npz` files and `labels/manifest.json`.

## Visibility model

Binary opaque/transparent (no Beer-Lambert). Rays cast from `player.camera.*` (the actual rendered camera transform from `GameRenderer.getCamera()`). Rays walk cells via Amanatides-Woo DDA, test each cell's AABBs (from `minecraft-data` `blockCollisionShapes.json`), terminate at the first opaque AABB hit. Transparent blocks (glass, leaves, ice, etc.) are hit and marked visible but do not terminate the ray.

### Camera math

Minecraft convention: `yaw=0` faces south (+Z), `yaw` increases clockwise from above, `pitch>0` looks down. The right vector is `(cos yaw, 0, sin yaw)` — **+sin, not −sin** — confirmed by Minecraft's strafing direction.

```python
fwd   = (-sin(yaw)*cos(pitch), -sin(pitch), cos(yaw)*cos(pitch))
right = (cos(yaw), 0, sin(yaw))
up    = cross(fwd, right)
```

## F5 third-person handling

When `cameraPerspective == "third_person_back"`, the player model is projected onto the screen and pixels inside the bounding box are skipped. This is an approximation (tight AABB, not actual geometry). Rays for occluded pixels are not cast, so cells visible only through the player model are not marked.

## Block shape / opacity data

`build_state_table.js` reads `minecraft-data` 1.19.4 `blockCollisionShapes.json` and `blocks.json` and emits `data/blockstate_table_1.19.4.json` (≈2.5 MB). Opacity = `!block.transparent`. Manual overrides in `data/opacity_overrides.json`.

## Performance

No CUDA GPU available on this machine. Uses **Numba `@njit(parallel=True)`** over the pixel rows on 16 CPU cores. First call incurs ~20 s JIT compilation; subsequent calls are fast (≈50–200 ms/tick).

GPU target: install `taichi` (`pip install taichi`) and rewrite `raycaster.py` to `@ti.kernel` for 10–50× speedup.

## V1 Limitations

- **Particles** (snow, rain, smoke, redstone): ignored (2D billboards, not geometry).
- **Entities** (mobs, items, projectiles): not modelled as occluders.
- **Brightness / lighting**: ignored; implied constant 1.
- **Continuous transparency (Beer-Lambert)**: not implemented; binary only.

## Running

```bash
# One-time: build blockstate table
node src/js/build_state_table.js

# Process a session
python src/python/pipeline.py C:/path/to/session_XXXXXXXXX

# Integration test (eyeball the renders)
python -m pipeline.tests.test_session
```

## Output format

`labels/tick_NNNNN.npz` keys:
- `m` — `uint8 (32,32,32)` visibility mask
- `b` — `int32 (32,32,32)` block-state IDs (indexed into `blockstate_table_1.19.4.json`)
- `s` — `uint8 (32,32,32)` seen-before mask
- `inventory` — JSON string
- `pose` — JSON string with `x,y,z,yaw,pitch,fov,perspective`

Window: world coords `[px−15, px+16) × [py−15, py+16) × [pz−15, pz+16)` where `px = floor(camera.x)` etc.
