# Blockscope Mineflayer Collector (productionized)

A hardened, parametrized data-collection stack. **The VPS (`mc.vivime.info:25566`) runs
Paper + Lodestone and owns ALL server-side work** — worlds, the copy-on-join pool, and
pairing + teleport-mirror (built into the Lodestone plugin). **Brev runs ONLY clients**:
per session, one **record-only Blockscope camera client** + one **Mineflayer controller**,
both connecting to the VPS. A fleet launcher runs N sessions in parallel (one per free GPU).
No Paper, no worlds, and no per-instance port management on Brev.

Pairing is by username convention: the controller connects as `ctrl_<id>` and its camera as
`cam_<id>`. Lodestone detects the prefixes, lands the pair in the SAME world, teleport-mirrors
the camera onto the controller each tick, and sends the camera a `blockscope:session_start`
so the mod records. (`/mirrorpair <ctrl> <cam>` is an op debug override.)

This replaces the `mineflayer_prototype/` rig. **Approach A**: a single Mineflayer controller
bot drives an exploration episode; the Paper plugin teleport-mirrors a *record-only* camera
onto the controller every tick, so recorded frames (and Furnace voxel/visibility labels) come
from the controller's exact pose. Mineflayer (not Baritone) does the pathing — it wins on dense
build maps. The camera **never** paths, so labels stay valid.

## Layout
```
controller/controller.js     tuned Mineflayer controller (distance-scaled timeouts, waypointing)
controller/package.json      mineflayer + mineflayer-pathfinder deps
mirror_plugin/               (legacy) standalone BlockscopeMirror plugin — pairing now lives in Lodestone
scripts/setup_server.sh      DEPRECATED — Brev no longer runs Paper; kept for local single-box testing
scripts/run_instance.sh      run ONE session: camera container + controller bot against the VPS
scripts/run_fleet.sh         launch N sessions on free GPUs (clients only — no Paper)
```
Server-side pairing + teleport-mirror lives in `lodestone/` (`MirrorPairManager`), deployed to
the VPS, not in this dir's `mirror_plugin/` (now legacy).

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
# on the Brev box, from this dir's scripts/ (the VPS must be up with the pairing-capable Lodestone)
GPU=7 EPISODE=explore DURATION=240 bash scripts/run_instance.sh
# optional: EXTRA_CTRL_ARGS="--waypoint-step 24 --think-timeout 15000"  MC_HOST=...  MC_PORT=...
```
Produces (uploaded to Hopper under a fresh `session_<id>`): `video.mp4`, `ticks.jsonl`,
`frame_mapping.jsonl`, and a **valid `.mcpr`**. Tears down the camera container on exit
(`KEEP=1` to leave it up). No server teardown — the VPS owns and recycles the world.

## Launch K instances (fleet)
```bash
bash scripts/run_fleet.sh            # all free GPUs (never GPU 0)
bash scripts/run_fleet.sh 3          # up to 3 instances
GPUS="7 6 4" bash scripts/run_fleet.sh
```
Each instance gets a unique GPU, usernames (`cam_gpu<N>`/`ctrl_gpu<N>`), and X display
(`:99+gpu`), so parallel sessions never collide — all against the one VPS server.
`run_fleet.sh` auto-detects free GPUs (≤200 MiB used AND ≤5% util), excludes GPU 0, and
staggers boots to avoid a thundering herd.

## Parameters (env)
| var | default | meaning |
|-----|---------|---------|
| `GPU` | (required) | free GPU index, never 0 |
| `EPISODE` | `explore` | controller episode |
| `DURATION` | `240` | seconds of episode activity |
| `MC_HOST` | `mc.vivime.info` | VPS Minecraft host |
| `MC_PORT` | `25566` | VPS Minecraft port |
| `CLIENT_DIR` | `~/blockscope_client` | blockscope mods + properties source |
| `HOPPER_URL` | `http://localhost:9000` | upload endpoint |
| `EXTRA_CTRL_ARGS` | — | passthrough tuning to controller.js |
| `KEEP` | `0` | leave the camera container up after run |
