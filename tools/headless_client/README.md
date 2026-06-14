# Headless Fabric data-collection client

A Docker container that runs the Blockscope Fabric 1.19.4 client fully headless on a
GPU, auto-joins the VPS, records sessions (video + per-tick state + ReplayMod
`.mcpr`), and uploads them to Hopper — with **no human interaction**.

This is the unit that gets replicated to scale data collection to N containers.

## What it does (end to end)

1. Xvfb starts a virtual X display; VirtualGL (EGL mode) gives Minecraft a real
   **GPU** OpenGL context (`OpenGL renderer string: NVIDIA A100-...`). ReplayMod
   needs a real GL context; software GL is too slow.
2. `launch_minecraft.py` installs vanilla 1.19.4 + Fabric loader (offline), copies
   the staged mods + `blockscope.properties` into the game dir, and launches the
   client with offline auth (VPS is `online-mode=false`).
3. The blockscope mod **auto-connects** to `mc.vivime.info:25566` as soon as the
   first menu screen appears (1.19.4 has no `--quickPlayMultiplayer` flag, so the
   connect is initiated from inside the mod — see "Auto-connect" below).
4. Lodestone assigns a world + sends `blockscope:session_start`; the mod starts the
   Baritone bot + ReplayMod recording and streams video segments to Hopper live.
5. On session-end kick (or the stuck-watchdog), the mod stops recording, finalizes
   `video.mp4`, uploads the `.mcpr`, and **auto-reconnects** for the next session.

## Files

| file | purpose |
|------|---------|
| `Dockerfile` | CUDA runtime base + Java 17 (Temurin) + Xvfb + VirtualGL 3.1.1 + ffmpeg + minecraft-launcher-lib |
| `entrypoint.sh` | starts Xvfb (per-GPU display number), probes VGL/EGL, launches MC under `vglrun -d egl` |
| `launch_minecraft.py` | installs Fabric 1.19.4, stages mods/config, assembles & runs the launch command (offline auth) |
| `blockscope.properties` | container config: `server_url=http://localhost:9000`, `ffmpeg_path=ffmpeg`, `autoconnect_host/port`, 640x360@20fps |
| `run.sh` | runs ONE container pinned to a chosen free GPU |

## Build & run (on the Brev box)

```bash
# one-time: build the image
cd ~/blockscope_client/build && docker build -t blockscope-headless:latest .

# run one container on a FREE gpu (check nvidia-smi first!)
./run.sh 0          # or: BLOCKSCOPE_AUTOCONNECT_HOST=... ./run.sh 0
```

Mods + config are **bind-mounted** from `~/blockscope_client/` so re-staging a new
mod jar does not require an image rebuild. The Minecraft/Fabric install is cached in
the `blockscope-game` docker volume so only the first run downloads it.

## GPU / VirtualGL setup that works

- Base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, run with
  `--gpus device=N -e NVIDIA_DRIVER_CAPABILITIES=all`.
- `VGL_DISPLAY=egl`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, launch via
  `vglrun -d egl python3 launch_minecraft.py`.
- Verified: `vglrun -d egl glxinfo` reports `OpenGL renderer string: NVIDIA
  A100-SXM4-80GB` — GPU-accelerated rendering confirmed, no host X server needed.

## Auto-connect (the key 1.19.4 hurdle)

1.19.4 has no quick-play launch flag, so the mod connects itself. In
`BlockscopeClient.maybeAutoConnect()`:
- reads `autoconnect_host` / `autoconnect_port` from `blockscope.properties`
  (env vars `BLOCKSCOPE_AUTOCONNECT_HOST` / `BLOCKSCOPE_AUTOCONNECT_PORT` override),
- waits until a menu screen is up and the client has no world/connection,
  then (after a ~2s settle) calls `ConnectScreen.connect(...)` once.

**Important:** we do *not* gate on `instanceof TitleScreen`. With offline/invalid
auth the first menu is the "secure-chat warning" screen (`net.minecraft.class_8032`),
not `TitleScreen`, so a class check silently never fires. Gating on "any non-null
menu, no world" is robust. All subsequent rejoins use the existing
`SessionProtocol.onDisconnect` auto-reconnect path.

## Scaling to N containers

- One container per **free** GPU. The box has 8 A100s; check `nvidia-smi` and only
  use GPUs at ~4 MiB / 0% util. Never co-locate on a GPU another job uses.
- `run.sh` already derives a unique display (`:99+GPU`) and bot username
  (`bs_gpu<N>`) per GPU, so containers don't collide on the shared X socket
  namespace (required because `--network host` shares it with the host).
- `--network host` lets every container reach Hopper at `http://localhost:9000`.
- Share the `blockscope-game` volume read-only or give each its own; the install is
  identical. Each session gets a unique `session_<unix>` dir on Hopper, so parallel
  containers don't clash.
- A simple loop `for g in 0 1 6 7; do BLOCKSCOPE_USERNAME=bs_gpu$g ./run.sh $g & done`
  launches one per free GPU. For real fleet management, wrap in systemd units or a
  compose file with `deploy.resources.reservations.devices` pinned per GPU.

## Known limitation: `.mcpr` on abrupt disconnects

ReplayMod finalizes its `.mcpr` on the channel-inactive event. When a session ends
via the **stuck-watchdog** (bot didn't move 15s → immediate disconnect) the temp
recording dir is torn down before ReplayMod writes `metaData.json`, so the `.mcpr`
save throws `FileNotFoundException` and that session has no `.mcpr`. This is isolated
to a ReplayMod worker thread — the JVM survives, reconnect works, and our own
`video.mp4` + `ticks.jsonl` + `frame_mapping.jsonl` (independent of ReplayMod) are
finalized and uploaded normally. Sessions that end via the normal server-side
session-end kick produce a clean `.mcpr`.
