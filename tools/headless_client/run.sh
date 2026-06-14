#!/usr/bin/env bash
# Run ONE headless blockscope collection container on the Brev box.
#
# Usage:  ./run.sh [GPU_INDEX]    (default GPU 0)
#
# Picks a free GPU explicitly; NEVER pass a GPU another job is using.
# Mods + config are bind-mounted from ~/blockscope_client so re-staging a new
# mod jar does NOT require an image rebuild.
set -euo pipefail

GPU="${1:-0}"
IMAGE="blockscope-headless:latest"
CLIENT_DIR="${BLOCKSCOPE_CLIENT_DIR:-$HOME/blockscope_client}"
NAME="blockscope-headless-gpu${GPU}"

AUTOCONNECT_HOST="${BLOCKSCOPE_AUTOCONNECT_HOST:-mc.vivime.info}"
AUTOCONNECT_PORT="${BLOCKSCOPE_AUTOCONNECT_PORT:-25566}"
USERNAME="${BLOCKSCOPE_USERNAME:-bs_gpu${GPU}}"

echo "[run] GPU=${GPU}  image=${IMAGE}  client=${CLIENT_DIR}"
echo "[run] auto-connect ${AUTOCONNECT_HOST}:${AUTOCONNECT_PORT} as ${USERNAME}"

docker rm -f "${NAME}" 2>/dev/null || true

# --network host  -> container reaches Hopper at http://localhost:9000
# --gpus device=N -> pin to one specific free GPU
exec docker run --rm \
    --name "${NAME}" \
    --gpus "device=${GPU}" \
    --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e BLOCKSCOPE_GPU="${GPU}" \
    -e BLOCKSCOPE_AUTOCONNECT_HOST="${AUTOCONNECT_HOST}" \
    -e BLOCKSCOPE_AUTOCONNECT_PORT="${AUTOCONNECT_PORT}" \
    -e BLOCKSCOPE_USERNAME="${USERNAME}" \
    -v "${CLIENT_DIR}/mods:/assets/mods:ro" \
    -v "${CLIENT_DIR}/blockscope.properties:/assets/blockscope.properties:ro" \
    "${IMAGE}"
