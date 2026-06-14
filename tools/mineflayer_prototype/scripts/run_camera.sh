#!/usr/bin/env bash
# Run the Blockscope headless client as a RECORD-ONLY camera, on a chosen free GPU,
# auto-connecting to the local prototype Paper server (:25599). The camera mods dir
# deliberately OMITS the baritone jar -> the client's SessionProtocol sets
# baritonePresent=false, so session_start triggers recording only (no bot).
#
# Uploads land in the running Hopper (:9000) under a fresh session_id, then in
# $DATA_DIR (default /data/vvm33/BLOCKSCOPE_DATA). NOTE: GPU 0 runs the production
# collector container — never pass 0 here.
set -euo pipefail

GPU="${1:?usage: run_camera.sh <FREE_GPU_INDEX>}"
if [ "$GPU" = "0" ]; then echo "REFUSING GPU 0 (production collector)."; exit 1; fi

ROOT="${MIRROR_ROOT:-$HOME/mirror_prototype}"
CAM_CLIENT="$ROOT/camera_client"
IMAGE="blockscope-headless:latest"
NAME="blockscope-mirror-camera-gpu${GPU}"
PORT="${MIRROR_PORT:-25599}"
USERNAME="${CAMERA_USER:-mirror_cam}"

# Build a baritone-free mods dir for the camera from the production client mods.
mkdir -p "$CAM_CLIENT/mods"
for jar in "$HOME/blockscope_client/mods"/*.jar; do
  base="$(basename "$jar")"
  case "$base" in
    baritone*) echo "[camera] skipping $base (record-only)";;
    *) cp -f "$jar" "$CAM_CLIENT/mods/$base";;
  esac
done

# Camera properties: connect to the local prototype server, upload to local Hopper.
cp -f "$HOME/blockscope_client/blockscope.properties" "$CAM_CLIENT/blockscope.properties"

echo "[camera] mods: $(ls "$CAM_CLIENT/mods" | tr '\n' ' ')"
docker rm -f "$NAME" 2>/dev/null || true

exec docker run --rm \
  --name "$NAME" \
  --gpus "device=${GPU}" \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e BLOCKSCOPE_GPU="${GPU}" \
  -e BLOCKSCOPE_AUTOCONNECT_HOST="127.0.0.1" \
  -e BLOCKSCOPE_AUTOCONNECT_PORT="${PORT}" \
  -e BLOCKSCOPE_USERNAME="${USERNAME}" \
  -v "$CAM_CLIENT/mods:/assets/mods:ro" \
  -v "$CAM_CLIENT/blockscope.properties:/assets/blockscope.properties:ro" \
  "$IMAGE"
