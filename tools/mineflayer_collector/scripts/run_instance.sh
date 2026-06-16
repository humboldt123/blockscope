#!/usr/bin/env bash
# Run ONE Blockscope collection session against the VPS server, end to end.
#
# Architecture: the VPS (mc.vivime.info:25566) runs Paper + Lodestone and owns ALL
# server-side work (worlds, copy-on-join pool, pairing + teleport-mirror). Brev runs
# ONLY two client processes per session:
#   - a record-only camera container (GPU, X display) — username cam_<INSTANCE>
#   - a Mineflayer controller bot                       — username ctrl_<INSTANCE>
# Both connect to the VPS. Lodestone detects the ctrl_/cam_ username prefixes, lands the
# pair in the same world, teleport-mirrors the camera onto the controller every tick, and
# sends the camera a session_start so the Fabric mod records.
#
# Pipeline:
#   1. start the camera container (GPU, DISPLAY) pointed at the VPS; it auto-connects as
#      cam_<INSTANCE>, gets paired by Lodestone, and starts recording on session_start.
#   2. start the Mineflayer controller bot pointed at the VPS as ctrl_<INSTANCE>.
#   3. wait for both to join (controller "spawned" log + camera log grep).
#   4. controller drives the episode for DURATION seconds, then quits.
#   5. controller quit -> Lodestone ends the pair -> camera is kicked -> the mod's
#      DISCONNECT handler finalizes video + writes the .mcpr -> watcher uploads to Hopper.
#   6. tear down the camera container. NO server, NO plugin build, NO port, NO worlds.
#
# Parameters (env or defaults). Headline knobs: GPU, EPISODE, DURATION.
#   GPU            GPU index to use                                  [required]
#   INSTANCE       instance id                                       [gpu<GPU>]
#   EPISODE        controller episode (explore|walklook)             [explore]
#   DURATION       seconds of episode activity                       [240]
#   MC_HOST        VPS Minecraft host                                [mc.vivime.info]
#   MC_PORT        VPS Minecraft port                                [25566]
#   CAMERA_USER / CONTROLLER_USER                                    [cam_<inst>/ctrl_<inst>]
#   HOPPER_URL     Hopper upload endpoint                            [http://localhost:9000]
#   CLIENT_DIR     blockscope client (mods + properties) source      [~/blockscope_client]
#   IMAGE          headless docker image                             [blockscope-headless:latest]
#   EXTRA_CTRL_ARGS  passthrough to controller.js (tuning overrides) [unset]
#   KEEP           1 = leave the camera container up after run       [0]
set -euo pipefail

GPU="${GPU:?set GPU index}"

COLLECTOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE="${INSTANCE:-gpu${GPU}}"
BASE="${COLLECTOR_BASE:-$HOME/blockscope_collector}/$INSTANCE"
EPISODE="${EPISODE:-explore}"
DURATION="${DURATION:-240}"
MC_HOST="${MC_HOST:-mc.vivime.info}"
MC_PORT="${MC_PORT:-25566}"
CAMERA_USER="${CAMERA_USER:-cam_${INSTANCE}}"
CONTROLLER_USER="${CONTROLLER_USER:-ctrl_${INSTANCE}}"
HOPPER_URL="${HOPPER_URL:-http://localhost:9000}"
CLIENT_DIR="${CLIENT_DIR:-$HOME/blockscope_client}"
IMAGE="${IMAGE:-blockscope-headless:latest}"
KEEP="${KEEP:-0}"

CAM_DIR="$BASE/camera_client"
CTRL_DIR="$BASE/controller"
CONTAINER="bs_camera_${INSTANCE}"
DISPLAY_NUM=":$((99 + GPU))"
CAM_LOG="$BASE/camera.log"
CTRL_LOG="$BASE/controller.log"

mkdir -p "$BASE"
echo "================================================================"
echo "[run:$INSTANCE] GPU=$GPU EPISODE=$EPISODE DURATION=${DURATION}s"
echo "[run:$INSTANCE] server=$MC_HOST:$MC_PORT (VPS owns worlds + pairing + mirror)"
echo "[run:$INSTANCE] camera=$CAMERA_USER controller=$CONTROLLER_USER display=$DISPLAY_NUM"
echo "[run:$INSTANCE] base=$BASE"
echo "================================================================"

cleanup() {
  echo "[run:$INSTANCE] cleanup ..."
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  # Make sure no orphan controller node process is left from this instance.
  [ -n "${CTRL_PID:-}" ] && kill "$CTRL_PID" >/dev/null 2>&1 || true
}
on_exit() { [ "$KEEP" = "1" ] || cleanup; }
trap on_exit EXIT INT TERM

# --- 1. start controller first (so Lodestone assigns it a world before camera joins) ---
# The camera uses PlayerSpawnLocationEvent to spawn directly in the controller's world
# (no cross-world teleport → no Respawn packet → no ReplayMod thread-pool teardown).
# That event reads pairWorlds, which is populated when the controller joins. So the
# controller MUST join and be assigned a world BEFORE the camera connects.
mkdir -p "$CTRL_DIR"
cp -f "$COLLECTOR_ROOT/controller/controller.js" "$CTRL_DIR/controller.js"
cp -f "$COLLECTOR_ROOT/controller/package.json" "$CTRL_DIR/package.json"
if [ ! -d "$CTRL_DIR/node_modules" ]; then
  if [ -d "$HOME/mirror_prototype/controller/node_modules" ]; then
    cp -r "$HOME/mirror_prototype/controller/node_modules" "$CTRL_DIR/node_modules"
  else
    ( cd "$CTRL_DIR" && npm install --no-audit --no-fund )
  fi
fi
echo "[run:$INSTANCE] starting controller '$CONTROLLER_USER' -> $MC_HOST:$MC_PORT (first, so world is ready for camera) ..."
( cd "$CTRL_DIR" && node controller.js \
    --host "$MC_HOST" --port "$MC_PORT" --user "$CONTROLLER_USER" \
    --episode "$EPISODE" --duration "$DURATION" ${EXTRA_CTRL_ARGS:-} ) > "$CTRL_LOG" 2>&1 &
CTRL_PID=$!
echo "[run:$INSTANCE] waiting for controller '$CONTROLLER_USER' to spawn + get world assignment ..."
CTRL_OK=0
for i in $(seq 1 90); do
  grep -qE '\[ctrl\] spawned at' "$CTRL_LOG" 2>/dev/null && { CTRL_OK=1; break; }
  kill -0 "$CTRL_PID" 2>/dev/null || break
  sleep 1
done
[ "$CTRL_OK" = "1" ] || { echo "[run:$INSTANCE] controller never spawned"; tail -20 "$CTRL_LOG"; exit 1; }
echo "[run:$INSTANCE] controller spawned — Lodestone has assigned its world. Starting camera now ..."
# Small pause for Lodestone to fully register the world in pairWorlds (one tick).
sleep 2

# --- 2. stage + start camera container -------------------------------------------
# Build a baritone-FREE mods dir -> client treats session_start as record-only.
mkdir -p "$CAM_DIR/mods"
rm -f "$CAM_DIR/mods"/*.jar
for jar in "$CLIENT_DIR/mods"/*.jar; do
  base="$(basename "$jar")"
  case "$base" in
    baritone*) ;;  # OMIT baritone => record-only camera (Approach A)
    *) cp -f "$jar" "$CAM_DIR/mods/$base" ;;
  esac
done
# Per-instance properties: connect to the VPS, upload to Hopper.
sed -e "s#^server_url=.*#server_url=${HOPPER_URL}#" \
    -e "s#^autoconnect_host=.*#autoconnect_host=${MC_HOST}#" \
    -e "s#^autoconnect_port=.*#autoconnect_port=${MC_PORT}#" \
    "$CLIENT_DIR/blockscope.properties" > "$CAM_DIR/blockscope.properties"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "[run:$INSTANCE] starting camera container (GPU $GPU, display $DISPLAY_NUM) -> $MC_HOST:$MC_PORT ..."
docker run -d --rm \
  --name "$CONTAINER" \
  --gpus "device=${GPU}" \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e BLOCKSCOPE_GPU="${GPU}" \
  -e DISPLAY_NUM="${DISPLAY_NUM}" \
  -e BLOCKSCOPE_AUTOCONNECT_HOST="${MC_HOST}" \
  -e BLOCKSCOPE_AUTOCONNECT_PORT="${MC_PORT}" \
  -e BLOCKSCOPE_USERNAME="${CAMERA_USER}" \
  -v "$CAM_DIR/mods:/assets/mods:ro" \
  -v "$CAM_DIR/blockscope.properties:/assets/blockscope.properties:ro" \
  --tmpfs /app/game/replay_recordings:exec,size=512m \
  "$IMAGE" >/dev/null
docker logs -f "$CONTAINER" > "$CAM_LOG" 2>&1 &

echo "[run:$INSTANCE] waiting for camera '$CAMERA_USER' to connect + record ..."
# A fresh container does a full Fabric+assets install and texture load on first boot,
# which can exceed 2min when the box's other GPUs are busy. 300s avoids a spurious abort
# while the client is still legitimately loading. The camera is "in" once it has either
# joined the world or begun recording (session_start from Lodestone).
CAM_OK=0
for i in $(seq 1 300); do
  if grep -qE "Connecting to ${MC_HOST}|Recording started|session_start|Joining world|Loaded" "$CAM_LOG" 2>/dev/null; then CAM_OK=1; break; fi
  sleep 1
done
[ "$CAM_OK" = "1" ] || { echo "[run:$INSTANCE] camera never connected"; tail -40 "$CAM_LOG"; exit 1; }
echo "[run:$INSTANCE] camera connected"

echo "[run:$INSTANCE] controller spawned. Lodestone is pairing + mirroring; driving episode for ${DURATION}s ..."

# --- 3. wait for the episode (controller quits when done) ------------------------
# On controller quit, Lodestone ends the pair and kicks the camera, which triggers the
# mod's clean disconnect -> finalize video -> write .mcpr -> upload to Hopper.
wait "$CTRL_PID" || true
echo "[run:$INSTANCE] controller finished + disconnected."

# --- 4. wait for the camera to save + upload ------------------------------------
# The kick from Lodestone gives the camera a CLEAN disconnect so the mod's DISCONNECT
# handler fires: stopRecording() -> finalize video -> ReplayMod writes .mcpr -> upload.
echo "[run:$INSTANCE] waiting for .mcpr save + upload (controller quit -> camera kicked -> save -> upload) ..."
MCPR_OK=0
for i in $(seq 1 180); do
  if grep -qE '\.mcpr upload complete' "$CAM_LOG" 2>/dev/null; then MCPR_OK=1; break; fi
  if grep -qE '\.mcpr watcher timed out' "$CAM_LOG" 2>/dev/null; then break; fi
  # if the container is gone and we have a terminal log, stop waiting
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$" || { sleep 2; break; }
  sleep 1
done

LATEST_SESSION="$(grep -oE 'Recording started: session_[0-9]+' "$CAM_LOG" 2>/dev/null | tail -1 | awk '{print $3}')"
echo "[run:$INSTANCE] camera session_id: ${LATEST_SESSION:-<unknown>}"
if [ "$MCPR_OK" = "1" ]; then
  echo "[run:$INSTANCE] mcpr upload status: COMPLETE"
else
  echo "[run:$INSTANCE] mcpr upload status:"; grep -E '\.mcpr upload complete|\.mcpr watcher timed out|Uploading replay' "$CAM_LOG" | tail -3 || echo "  (no mcpr outcome logged)"
fi
echo "[run:$INSTANCE] modal/recovery check (should be EMPTY):"
grep -iE 'offering recovery|has not quit normally|recover recording' "$CAM_LOG" | tail -5 || echo "  (none)"

echo "[run:$INSTANCE] DONE. session=${LATEST_SESSION:-?}  logs in $BASE"
