# Mineflayer-controller + mirrored-camera prototype

A self-contained, Brev-only prototype proving **Approach A**: a Mineflayer bot drives
a "controller" player on a standalone Paper 1.19.4 server, and a teleport-mirror
plugin glues our **Blockscope record-only client** ("camera") onto the controller
every tick, so the camera records a first-person session whose motion mirrors the
bot. The recorded session then goes through the **Furnace** pipeline to produce
per-tick visibility/voxel labels, validating that the labels are real and aligned.

This is the unit that, if it scales, replaces hand-driven data collection: scripted
bots provide the *behavior*, the mirror provides a *clean recordable camera*, and
Furnace provides the *labels*.

```
  Mineflayer bot (ctrl_bot)            Paper 1.19.4 (:25599, flat, peaceful, offline)
  walk-look / orbit episode  ───────►  BlockscopeMirror plugin
                                          • /mirror ctrl_bot mirror_cam
                                          • every tick: camera.teleportAsync(ctrl.getLocation())
                                          • /startcam → blockscope:session_start packet
                                                  │
                                                  ▼
  Blockscope headless client (mirror_cam, NO baritone → record-only)
  Xvfb + VirtualGL (EGL) on a free GPU, records video + ticks + .mcpr → Hopper (:9000)
                                                  │
                                                  ▼
  Furnace labeler (GPU)  → labels/tick_NNNNN.npz  (visibility + voxel grids)
```

## Components

| File | Role |
|------|------|
| `mirror_plugin/src/.../MirrorPlugin.java` | Paper plugin (api 1.19, **no ProtocolLib**). `/mirror`, `/startcam`. Per-tick teleport mirror; camera made gravity-less, flying, non-colliding, mutually hidden. |
| `mirror_plugin/resources/plugin.yml` | Plugin manifest. |
| `controller/controller.js` | Single-agent Mineflayer + pathfinder harness. `walklook` and `orbit` episodes ported from solaris-engine primitives (smooth log-normal look, `gotoWithTimeout`). |
| `controller/package.json` | Node deps (mineflayer, mineflayer-pathfinder, vec3, minecraft-data). |
| `scripts/setup_server.sh` | Download Paper 1.19.4, write flat/peaceful/offline config, build the plugin with `javac` (no maven on Brev — compiles against the API jar Paper unpacks on first run). |
| `scripts/start_server.sh` | Start Paper in a detached `screen` (`mirror_paper`). |
| `scripts/run_camera.sh` | Run the Blockscope headless image as a **baritone-free** camera on a chosen free GPU, auto-connecting to `:25599`, uploading to the local Hopper. **Refuses GPU 0** (production collector). |

## How to run (on Brev: `ssh drexel-gpu-name`)

```bash
# everything lives under ~/mirror_prototype on Brev (rsynced from this dir)
cd ~/mirror_prototype
bash scripts/setup_server.sh          # one-time: Paper + plugin build
bash scripts/start_server.sh          # Paper on :25599 in screen 'mirror_paper'

bash scripts/run_camera.sh 1          # camera on GPU 1 (NEVER 0); waits ~90s to connect+record
# wait for "Recording started: session_<id>" in the camera log

cd controller && npm install
node controller.js --host 127.0.0.1 --port 25599 --user ctrl_bot \
     --episode walklook --duration 90 &

# pair (server console via screen):
screen -S mirror_paper -p 0 -X stuff "mirror ctrl_bot mirror_cam\n"
# optional managed-session marker / duration:
screen -S mirror_paper -p 0 -X stuff "startcam 90\n"

# end the session cleanly so the camera finalizes video + ticks:
#   screen -S mirror_paper -p 0 -X stuff "kick mirror_cam done\n"
```

## Recording the .mcpr — known ReplayMod gotcha (IMPORTANT)

The headless client auto-starts ReplayMod recording on connect. On disconnect,
ReplayMod 2.6.21 in this headless container hits:

```
java.io.FileNotFoundException: .../replay_recordings/recording/<ts>.mcpr.tmp/changed/metaData.json
```

ReplayMod's `ZipReplayFile.saveTo` looks under an extra `recording/` path component,
but the temp files actually live at `replay_recordings/<ts>.mcpr.tmp/changed/`. The
temp dir is **complete and valid** (`recording.tmcpr`, `metaData.json`, `mods.json`,
`recording.tmcpr.crc32`) — only the final zip step fails. Until the client is patched,
assemble the `.mcpr` yourself before running Furnace:

```bash
TMP=/tmp/mcpr_assemble; rm -rf $TMP; mkdir -p $TMP
docker cp "<camera_container>:/app/game/replay_recordings/<ts>.mcpr.tmp/changed/." $TMP/
( cd $TMP && python3 - <<PY
import zipfile,os
with zipfile.ZipFile("<session_dir>/<id>.mcpr","w",zipfile.ZIP_DEFLATED) as z:
    for f in ["metaData.json","recording.tmcpr","recording.tmcpr.crc32","mods.json"]:
        if os.path.exists(f): z.write(f)
PY
)
```

(video.mp4, ticks.jsonl, frame_mapping.jsonl all upload normally to Hopper.)

## Validate through Furnace

```bash
cd /home/vvm33/pum/furnace/pipeline
CUDA_VISIBLE_DEVICES=1 /home/nvidia/.local/bin/uv run \
    python -m pipeline.labeler --session /data/vvm33/BLOCKSCOPE_DATA/session_<id>
# -> writes labels/tick_NNNNN.npz (visibility mask + voxel grid per tick)
```

## Isolation rules honored

- GPU 0 / container `blockscope-headless-gpu0` (production collector): never touched.
  `run_camera.sh` hard-refuses GPU 0.
- Production VPS `mc.vivime.info` / Lodestone: never touched. This server is a fresh
  Paper instance on **:25599** with its own world; the unrelated lab `lodestone`
  screen on `:25565` is left alone.
- Test sessions land in Hopper under their own `session_<id>` and are clearly
  identifiable (camera username `mirror_cam`).
