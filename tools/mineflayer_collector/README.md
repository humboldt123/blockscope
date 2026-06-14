# Blockscope Mineflayer Collector (productionized)

A hardened, parametrized, per-instance data-collection stack: one **Paper 1.19.4 server +
BlockscopeMirror plugin + record-only Blockscope camera client + Mineflayer controller**,
plus a fleet launcher that runs N of them in parallel (one per free GPU).

This replaces the `mineflayer_prototype/` rig. **Approach A**: a single Mineflayer controller
bot drives an exploration episode; the Paper plugin teleport-mirrors a *record-only* camera
onto the controller every tick, so recorded frames (and Furnace voxel/visibility labels) come
from the controller's exact pose. Mineflayer (not Baritone) does the pathing — it wins on dense
build maps. The camera **never** paths, so labels stay valid.

## Layout
```
controller/controller.js     tuned Mineflayer controller (distance-scaled timeouts, waypointing)
controller/package.json      mineflayer + mineflayer-pathfinder deps
mirror_plugin/               BlockscopeMirror Paper plugin (teleport-mirror + session_start)
scripts/setup_server.sh      provision ONE per-instance Paper server (unique dir/port/world)
scripts/run_instance.sh      run ONE full session end-to-end (server+camera+controller+pair+save)
scripts/run_fleet.sh         launch N instances on free GPUs, unique everything
```

## The two ReplayMod bugs this fixes (and how)

Both the **"recover recording? / Minecraft has not quit normally" modal** and the
**`.mcpr` `FileNotFoundException` on save** share ONE root cause, found in the real logs:

ReplayMod's `ReplayFilesService.initialScan()` is deferred (`runPostStartup`) until *after*
the resource-loading overlay clears — which in the headless client is the **same moment** the
camera auto-connects and starts recording. `initialScan` then (1) **moves** every
`*.mcpr.tmp` out of `<replayFolder>/recording/` into the replay root and (2) offers a recovery
prompt for any orphan `.mcpr.tmp` lacking a `.no_recover` sibling. When it catches the **live**
recording's temp dir it yanks `.mcpr.tmp/changed/` out from under the active writer →
`FileNotFoundException` (no `.mcpr` saved) and pops the recovery modal into the recorded frames.

Fix (in the `blockscope` mod — `RecordingManager` + `ReplayModIntegration` + `BlockscopeClient`):
1. **`cleanupReplayRecoveryState()` at mod init** — wipes stale `*.mcpr.tmp` / `.no_recover` /
   `.mcpr.cache` / `.mcpr.del` from `replay_recordings/` *before* ReplayMod's deferred scan
   runs, so the one-time scan sees a clean tree and is a harmless no-op (no modal).
2. **`beginReplaySession()` before every `initiateRecording()`** — points ReplayMod's
   `Setting.RECORDING_PATH` at a fresh, unique per-session dir (`replay_recordings/rec_<ts>/`).
   The live `.mcpr`/`.mcpr.tmp` now live in an isolated subtree the startup scan never touches,
   so the move/recover race can never destroy a live recording → clean `.mcpr` saves every time.

This is a **real fix in code** (no zip-the-temp-dir hack), built into
`blockscope-0.1.0-alpha.jar`. ReplayMod itself is unmodified (it ships as a separate jar).

## Pathfinding tuning (controller.js)
- **Distance-scaled per-goal timeout**: `clamp(goalBase + goalPerBlock*dist, goalMin, goalMax)`.
- **Raised planner budgets**: `pathfinder.thinkTimeout` / `tickTimeout` (lib defaults 5000/40 are
  why long cross-map routes and deep climbs time out before a path is found).
- **Waypointing** (`--waypoint-step N`): far/high goals split into short legs the planner can
  finish inside its think budget; a failed leg retries direct-to-goal then moves on.
- **Active stuck detection**: abandon-and-next instead of silently wedging.

## Launch ONE instance (proof / single run)
```bash
# on the Brev box, from this dir's scripts/
GPU=7 EPISODE=explore DURATION=240 bash scripts/run_instance.sh
# optional: WORLD=/path/to/build_map  LEVEL_TYPE=normal  EXTRA_CTRL_ARGS="--waypoint-step 24 --think-timeout 15000"
```
Produces (uploaded to Hopper under a fresh `session_<id>`): `video.mp4`, `ticks.jsonl`,
`frame_mapping.jsonl`, and a **valid `.mcpr`**. Tears down the server + container and frees the
port on exit (`KEEP=1` to leave them up).

## Launch K instances (fleet)
```bash
bash scripts/run_fleet.sh            # all free GPUs (never GPU 0)
bash scripts/run_fleet.sh 3          # up to 3 instances
GPUS="7 6 4" WORLD=/path/to/map bash scripts/run_fleet.sh
```
Each instance gets a unique GPU, port (`25600+gpu`), usernames (`cam_gpu<N>`/`ctrl_gpu<N>`),
X display (`:99+gpu`), and server/session dir under `~/blockscope_collector/<instance>/`, so
parallel instances never collide. `run_fleet.sh` auto-detects free GPUs (≤200 MiB used AND
≤5% util), excludes GPU 0, and staggers boots to avoid a thundering herd.

## Parameters (env)
| var | default | meaning |
|-----|---------|---------|
| `GPU` | (required) | free GPU index, never 0 |
| `WORLD` | (generate) | build-map dir to load |
| `EPISODE` | `explore` | controller episode |
| `DURATION` | `240` | seconds of episode activity |
| `PORT` | `25600+GPU` | server port |
| `CLIENT_DIR` | `~/blockscope_client` | blockscope mods + properties source |
| `HOPPER_URL` | `http://localhost:9000` | upload endpoint |
| `EXTRA_CTRL_ARGS` | — | passthrough tuning to controller.js |
| `KEEP` | `0` | leave server/container up after run |
