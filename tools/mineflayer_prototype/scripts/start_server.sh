#!/usr/bin/env bash
# Start the standalone Paper 1.19.4 prototype server in a detached screen.
set -euo pipefail
ROOT="${MIRROR_ROOT:-$HOME/mirror_prototype}"
SERVER_DIR="$ROOT/server"
cd "$SERVER_DIR"
screen -dmS mirror_paper bash -c "java -Xmx3G -jar paper-1.19.4.jar nogui 2>&1 | tee $SERVER_DIR/server.log"
echo "[start] Paper starting in screen 'mirror_paper'. Log: $SERVER_DIR/server.log"
