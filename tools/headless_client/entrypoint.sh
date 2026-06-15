#!/usr/bin/env bash
# Headless Fabric client entrypoint: start Xvfb, then launch Minecraft under
# VirtualGL so OpenGL is GPU-accelerated via EGL (no host X server needed).
set -euo pipefail

# With --network host the X11 abstract socket namespace is shared with the host,
# so each container needs a unique display number. Derive it from the GPU index
# (overridable via DISPLAY_NUM) to keep N containers from colliding on :99.
DISPLAY_NUM="${DISPLAY_NUM:-:$(( 99 + ${BLOCKSCOPE_GPU:-0} ))}"
SCREEN_W="${SCREEN_W:-1280}"
SCREEN_H="${SCREEN_H:-720}"
SCREEN_DEPTH="${SCREEN_DEPTH:-24}"

log() { echo "[entrypoint] $*"; }

# Flag the headless collector so our vendored ReplayMod skips its startup recovery
# scan (no "recover recording?" modal, no race with the live recording).
export BLOCKSCOPE_HEADLESS=1

# Force OpenAL Soft to the null driver. The container has no ALSA/Pulse device, and the
# native OpenAL→ALSA probe can hang/wedge the render thread during sound init (intermittent
# "Failed to open OpenAL device" that sometimes stalls the client before it reaches the title
# screen). The null driver makes sound init a fast, deterministic no-op — we don't record audio.
export ALSOFT_DRIVERS=null
export SDL_AUDIODRIVER=dummy

# Belt-and-suspenders: wipe any ReplayMod recordings left in the game dir before launch.
# /app/game is normally fresh per --rm container, but if it is ever bind-mounted/persisted
# a stale *.mcpr.tmp would otherwise be a recovery candidate. Removing the whole dir
# guarantees ReplayMod boots against a clean tree.
GAME_DIR="${BLOCKSCOPE_GAME_DIR:-/app/game}"
if [[ -d "${GAME_DIR}/replay_recordings" ]]; then
    log "Wiping stale ReplayMod recordings in ${GAME_DIR}/replay_recordings ..."
    rm -rf "${GAME_DIR}/replay_recordings" || true
fi

# Suppress vanilla tutorial hints (the "Move with WASD / Jump with Space" popup that
# appears on first world join and stays on-screen until dismissed). The options.txt
# key is written before Minecraft starts so the client never shows it.
#
# Also randomize visual settings (gamma/brightness and FOV) each launch so the
# dataset covers a range of visual conditions rather than always using defaults.
#   gamma: float [0.0, 1.0]  (0.0 = moody/default, 1.0 = full bright)
#   fov:   integer [60, 110] (stored as actual degrees in options.txt)
mkdir -p "${GAME_DIR}"

# Wipe options.txt entirely each launch so stale values (e.g. an internal FOV unit
# from a previous run) never survive into the next session. Minecraft writes its own
# defaults for any missing keys, so a fresh file is always safe.
rm -f "${GAME_DIR}/options.txt"

RAND_GAMMA=$(python3 -c "import random; print(round(random.uniform(0.0, 1.0), 2))")
RAND_FOV=$(python3 -c "import random; print(random.randint(60, 110))")
log "Randomized visual settings: gamma=${RAND_GAMMA} fov=${RAND_FOV}"

# Write a clean options.txt with just the keys we care about. Minecraft fills
# all other keys with defaults on first run.
printf 'tutorialStep:none\ngamma:%s\nfov:%s\n' "${RAND_GAMMA}" "${RAND_FOV}" \
    > "${GAME_DIR}/options.txt"
log "options.txt: tutorialStep=none gamma=${RAND_GAMMA} fov=${RAND_FOV}"

cleanup() {
    log "Cleaning up ..."
    [[ -n "${XVFB_PID:-}" ]] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "Java: $(java -version 2>&1 | head -1)"
log "GPU check (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || log "nvidia-smi unavailable"

# --- Virtual framebuffer (X server with GLX so Minecraft has a display) ---
log "Starting Xvfb on ${DISPLAY_NUM} (${SCREEN_W}x${SCREEN_H}x${SCREEN_DEPTH}) ..."
Xvfb "${DISPLAY_NUM}" -screen 0 "${SCREEN_W}x${SCREEN_H}x${SCREEN_DEPTH}" +extension GLX +render -noreset &
XVFB_PID=$!
export DISPLAY="${DISPLAY_NUM}"

# Wait for Xvfb to be ready.
for i in $(seq 1 30); do
    if xdpyinfo -display "${DISPLAY_NUM}" >/dev/null 2>&1; then break; fi
    sleep 0.3
done

# --- Sanity: confirm VirtualGL can reach the GPU via EGL ---
log "VirtualGL GL renderer probe:"
if vglrun -d "${VGL_DISPLAY:-egl}" glxinfo 2>/dev/null | grep -E "OpenGL renderer|OpenGL version" ; then
    log "VirtualGL EGL context OK."
else
    log "WARNING: vglrun glxinfo probe failed; launching anyway (see Minecraft logs)."
fi

log "Auto-connect target: ${BLOCKSCOPE_AUTOCONNECT_HOST:-<unset>}:${BLOCKSCOPE_AUTOCONNECT_PORT:-<unset>}"

# --- Launch Minecraft under VirtualGL ---
log "Launching Minecraft under VirtualGL (EGL) ..."
exec vglrun -d "${VGL_DISPLAY:-egl}" python3 /app/launch_minecraft.py
