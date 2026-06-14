#!/usr/bin/env bash
# Run ONE self-contained Blockscope collection session, end to end.
#
# Pipeline (all isolated per-instance so K can run in parallel):
#   1. provision + start a dedicated Paper server (unique PORT, world, SERVER_DIR)
#   2. start the record-only camera container (unique GPU, X display, username, recordings)
#   3. start the Mineflayer controller bot (unique username), let both join
#   4. from the server console: op the camera, /mirror <controller> <camera>, /startcam
#   5. controller drives the episode for DURATION; camera mirrors + records
#   6. session ends -> controller quits -> camera disconnect saves video.mp4 + ticks +
#      frame_mapping + a VALID .mcpr, all uploaded to Hopper under a fresh session_id
#   7. tear down server + container, free the port (no orphans)
#
# Parameters (env or defaults). The three headline knobs are WORLD, EPISODE, GPU.
#   GPU            free GPU index (NEVER 0)                         [required]
#   INSTANCE       instance id                                       [gpu<GPU>]
#   WORLD          build-map dir to load (omit = generated world)    [unset]
#   EPISODE        controller episode (explore|walklook)             [explore]
#   DURATION       seconds of episode activity                       [240]
#   PORT           server port                                       [25600+GPU]
#   CAMERA_USER / CONTROLLER_USER                                    [cam_<inst>/ctrl_<inst>]
#   HOPPER_URL     Hopper upload endpoint                            [http://localhost:9000]
#   CLIENT_DIR     blockscope client (mods + properties) source      [~/blockscope_client]
#   IMAGE          headless docker image                             [blockscope-headless:latest]
#   EXTRA_CTRL_ARGS  passthrough to controller.js (tuning overrides) [unset]
#   KEEP           1 = leave server/container up after run           [0]
set -euo pipefail

GPU="${GPU:?set GPU (free index, never 0)}"
[ "$GPU" = "0" ] && { echo "REFUSING GPU 0 (production collector)."; exit 1; }

COLLECTOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE="${INSTANCE:-gpu${GPU}}"
BASE="${COLLECTOR_BASE:-$HOME/blockscope_collector}/$INSTANCE"
PORT="${PORT:-$((25600 + GPU))}"
EPISODE="${EPISODE:-explore}"
DURATION="${DURATION:-240}"
CAMERA_USER="${CAMERA_USER:-cam_${INSTANCE}}"
CONTROLLER_USER="${CONTROLLER_USER:-ctrl_${INSTANCE}}"
HOPPER_URL="${HOPPER_URL:-http://localhost:9000}"
CLIENT_DIR="${CLIENT_DIR:-$HOME/blockscope_client}"
IMAGE="${IMAGE:-blockscope-headless:latest}"
KEEP="${KEEP:-0}"

SERVER_DIR="$BASE/server"
CAM_DIR="$BASE/camera_client"
CTRL_DIR="$BASE/controller"
SCREEN_NAME="bs_server_${INSTANCE}"
CONTAINER="bs_camera_${INSTANCE}"
DISPLAY_NUM=":$((99 + GPU))"
SERVER_LOG="$BASE/server.log"
CAM_LOG="$BASE/camera.log"
CTRL_LOG="$BASE/controller.log"

mkdir -p "$BASE"
echo "================================================================"
echo "[run:$INSTANCE] GPU=$GPU PORT=$PORT EPISODE=$EPISODE DURATION=${DURATION}s"
echo "[run:$INSTANCE] camera=$CAMERA_USER controller=$CONTROLLER_USER display=$DISPLAY_NUM"
echo "[run:$INSTANCE] base=$BASE"
echo "================================================================"

cleanup() {
  echo "[run:$INSTANCE] cleanup ..."
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if screen -ls 2>/dev/null | grep -q "$SCREEN_NAME"; then
    screen -S "$SCREEN_NAME" -p 0 -X stuff "stop$(printf \\r)" 2>/dev/null || true
    sleep 6
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
  fi
  # free the port if anything is still bound
  fuser -k "${PORT}/tcp" 2>/dev/null || true
}
on_exit() { [ "$KEEP" = "1" ] || cleanup; }
trap on_exit EXIT INT TERM

srv() { screen -S "$SCREEN_NAME" -p 0 -X stuff "$1$(printf \\r)"; }

# --- 1. provision + start server -------------------------------------------------
INSTANCE="$INSTANCE" SERVER_DIR="$SERVER_DIR" PORT="$PORT" \
  WORLD="${WORLD:-}" LEVEL_TYPE="${LEVEL_TYPE:-normal}" PAPER_JAR="${PAPER_JAR:-}" \
  bash "$COLLECTOR_ROOT/scripts/setup_server.sh"

echo "[run:$INSTANCE] starting Paper in screen '$SCREEN_NAME' ..."
screen -dmS "$SCREEN_NAME" bash -c "cd '$SERVER_DIR' && java -Xmx3G -jar paper-1.19.4.jar nogui 2>&1 | tee '$SERVER_LOG'"

echo "[run:$INSTANCE] waiting for server 'Done' ..."
for i in $(seq 1 120); do
  grep -q 'Done (' "$SERVER_LOG" 2>/dev/null && break
  sleep 1
done
grep -q 'Done (' "$SERVER_LOG" || { echo "[run:$INSTANCE] server failed to start"; tail -20 "$SERVER_LOG"; exit 1; }
echo "[run:$INSTANCE] server up on :$PORT"

# --- 2. stage + start camera container -------------------------------------------
mkdir -p "$CAM_DIR/mods"
# Build a baritone-FREE mods dir -> client treats session_start as record-only.
for jar in "$CLIENT_DIR/mods"/*.jar; do
  base="$(basename "$jar")"
  case "$base" in
    baritone*) ;;  # OMIT baritone => record-only camera (Approach A)
    *) cp -f "$jar" "$CAM_DIR/mods/$base" ;;
  esac
done
# Per-instance properties: connect to THIS server, upload to Hopper.
sed -e "s#^server_url=.*#server_url=${HOPPER_URL}#" \
    -e "s#^autoconnect_host=.*#autoconnect_host=127.0.0.1#" \
    -e "s#^autoconnect_port=.*#autoconnect_port=${PORT}#" \
    "$CLIENT_DIR/blockscope.properties" > "$CAM_DIR/blockscope.properties"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "[run:$INSTANCE] starting camera container (GPU $GPU, display $DISPLAY_NUM) ..."
docker run -d --rm \
  --name "$CONTAINER" \
  --gpus "device=${GPU}" \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e BLOCKSCOPE_GPU="${GPU}" \
  -e DISPLAY_NUM="${DISPLAY_NUM}" \
  -e BLOCKSCOPE_AUTOCONNECT_HOST="127.0.0.1" \
  -e BLOCKSCOPE_AUTOCONNECT_PORT="${PORT}" \
  -e BLOCKSCOPE_USERNAME="${CAMERA_USER}" \
  -v "$CAM_DIR/mods:/assets/mods:ro" \
  -v "$CAM_DIR/blockscope.properties:/assets/blockscope.properties:ro" \
  "$IMAGE" >/dev/null
docker logs -f "$CONTAINER" > "$CAM_LOG" 2>&1 &

echo "[run:$INSTANCE] waiting for camera '$CAMERA_USER' to join ..."
for i in $(seq 1 120); do
  grep -q "$CAMERA_USER joined the game" "$SERVER_LOG" 2>/dev/null && break
  sleep 1
done
grep -q "$CAMERA_USER joined the game" "$SERVER_LOG" || { echo "[run:$INSTANCE] camera never joined"; tail -30 "$CAM_LOG"; exit 1; }
echo "[run:$INSTANCE] camera joined"

# --- 3. start controller ---------------------------------------------------------
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
echo "[run:$INSTANCE] starting controller '$CONTROLLER_USER' ..."
( cd "$CTRL_DIR" && node controller.js \
    --host 127.0.0.1 --port "$PORT" --user "$CONTROLLER_USER" \
    --episode "$EPISODE" --duration "$DURATION" ${EXTRA_CTRL_ARGS:-} ) > "$CTRL_LOG" 2>&1 &
CTRL_PID=$!

echo "[run:$INSTANCE] waiting for controller '$CONTROLLER_USER' to join ..."
for i in $(seq 1 60); do
  grep -q "$CONTROLLER_USER joined the game" "$SERVER_LOG" 2>/dev/null && break
  sleep 1
done
grep -q "$CONTROLLER_USER joined the game" "$SERVER_LOG" || { echo "[run:$INSTANCE] controller never joined"; tail -20 "$CTRL_LOG"; exit 1; }
echo "[run:$INSTANCE] controller joined"

# --- 4. pair + start recording from the server console ---------------------------
sleep 2
srv "op ${CAMERA_USER}"
srv "op ${CONTROLLER_USER}"
sleep 1
srv "mirror ${CONTROLLER_USER} ${CAMERA_USER}"
sleep 1
# startcam duration a bit longer than the episode so recording covers the whole run
srv "startcam $((DURATION + 30))"
echo "[run:$INSTANCE] paired + recording. Driving episode for ${DURATION}s ..."

# --- 5. wait for the episode (controller exits when done) ------------------------
wait "$CTRL_PID" || true
echo "[run:$INSTANCE] controller finished."

# --- 6. let the camera finish save + upload (.mcpr watcher needs a moment) -------
echo "[run:$INSTANCE] stopping mirror + letting camera save/upload ..."
srv "mirror stop"
sleep 20  # camera disconnect -> save video.mp4 / ticks / frame_mapping / .mcpr -> upload

LATEST_SESSION="$(grep -oE 'Recording started: session_[0-9]+' "$CAM_LOG" 2>/dev/null | tail -1 | awk '{print $3}')"
echo "[run:$INSTANCE] camera session_id: ${LATEST_SESSION:-<unknown>}"
echo "[run:$INSTANCE] mcpr upload status:"; grep -E '\.mcpr upload complete|\.mcpr watcher timed out' "$CAM_LOG" | tail -2 || true
echo "[run:$INSTANCE] modal/recovery check (should be EMPTY):"
grep -iE 'offering recovery|has not quit normally|recover recording' "$CAM_LOG" | tail -5 || echo "  (none)"

echo "[run:$INSTANCE] DONE. session=${LATEST_SESSION:-?}  logs in $BASE"
