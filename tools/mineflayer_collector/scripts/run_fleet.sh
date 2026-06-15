#!/usr/bin/env bash
# Launch N Blockscope collector instances in parallel, one per FREE GPU.
#
# Each instance = 1 record-only camera container + 1 Mineflayer controller bot, BOTH
# connecting to the VPS server (mc.vivime.info:25566). The VPS owns all worlds + pairing
# + teleport-mirror. NO Paper servers on Brev, NO per-instance ports, NO world copying.
#
# Auto-detects free GPUs (nvidia-smi: ~0% util AND tiny mem use), NEVER uses GPU 0, and
# gives each instance a unique GPU / X display / usernames (cam_<inst>/ctrl_<inst>) so
# parallel runs never collide. Each instance is a full run_instance.sh pipeline.
#
# Usage:
#   ./run_fleet.sh [N]                 # launch up to N instances on free GPUs (default: all free)
#
# Env (forwarded to every instance):
#   EPISODE, DURATION, CLIENT_DIR, IMAGE, HOPPER_URL, MC_HOST, MC_PORT, EXTRA_CTRL_ARGS, KEEP
#   GPUS="7 4 6"   explicit GPU list (overrides auto-detect)
#   MEM_FREE_MAX   max MiB used to call a GPU "free"            [200]
set -euo pipefail

COLLECTOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-0}"
MEM_FREE_MAX="${MEM_FREE_MAX:-200}"

detect_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | while IFS=',' read -r idx mem util; do
      idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
      [ "$idx" = "0" ] && continue                          # NEVER GPU 0
      if [ "$mem" -le "$MEM_FREE_MAX" ] && [ "$util" -le 5 ]; then echo "$idx"; fi
    done
}

if [ -n "${GPUS:-}" ]; then
  FREE=($GPUS)
else
  FREE=($(detect_free_gpus))
fi

if [ "${#FREE[@]}" -eq 0 ]; then
  echo "[fleet] NO free GPUs (excluding GPU 0). nvidia-smi:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi

# Trim to N if requested.
if [ "$N" -gt 0 ] && [ "$N" -lt "${#FREE[@]}" ]; then
  FREE=("${FREE[@]:0:$N}")
fi

echo "[fleet] launching ${#FREE[@]} instance(s) on GPU(s): ${FREE[*]}"
PIDS=()
for gpu in "${FREE[@]}"; do
  echo "[fleet] -> instance gpu${gpu}"
  GPU="$gpu" INSTANCE="gpu${gpu}" \
    EPISODE="${EPISODE:-explore}" DURATION="${DURATION:-240}" \
    MC_HOST="${MC_HOST:-mc.vivime.info}" MC_PORT="${MC_PORT:-25566}" \
    CLIENT_DIR="${CLIENT_DIR:-$HOME/blockscope_client}" IMAGE="${IMAGE:-blockscope-headless:latest}" \
    HOPPER_URL="${HOPPER_URL:-http://localhost:9000}" EXTRA_CTRL_ARGS="${EXTRA_CTRL_ARGS:-}" \
    KEEP="${KEEP:-0}" \
    bash "$COLLECTOR_ROOT/scripts/run_instance.sh" 2>&1 | sed "s/^/[gpu${gpu}] /" &
  PIDS+=($!)
  sleep 8   # stagger so N Mojang/Fabric installs + VPS joins don't thundering-herd
done

echo "[fleet] all ${#FREE[@]} instances launched; waiting ..."
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL+1)); done
echo "[fleet] done. ${#FREE[@]} instances, $FAIL failures."
exit $((FAIL > 0 ? 1 : 0))
