#!/usr/bin/env bash
# Provision a per-instance Paper 1.19.4 server for the Blockscope mirror collector.
#
# Parametrized & collision-free: every instance gets its own SERVER_DIR, PORT, and
# (optionally) WORLD copied in, so K instances run in parallel without clobbering each
# other's world/config/logs. Builds the BlockscopeMirror plugin with javac (no maven).
#
# Env (all overridable; INSTANCE drives the defaults):
#   INSTANCE       instance id (e.g. gpu7)                     [required]
#   SERVER_DIR     server working dir          [~/blockscope_collector/<INSTANCE>/server]
#   PORT           server port                                 [25600 + gpu offset]
#   WORLD          path to a world dir to copy in (build map)  [generate normal world]
#   LEVEL_TYPE     when no WORLD: normal|flat                  [normal]
#   PLUGIN_SRC     mirror plugin source dir     [<collector>/mirror_plugin]
#   PAPER_JAR      reuse an existing paper jar                 [download build 550]
set -euo pipefail

INSTANCE="${INSTANCE:?set INSTANCE (e.g. gpu7)}"
COLLECTOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${COLLECTOR_BASE:-$HOME/blockscope_collector}/$INSTANCE"
SERVER_DIR="${SERVER_DIR:-$BASE/server}"
PORT="${PORT:?set PORT}"
LEVEL_TYPE="${LEVEL_TYPE:-normal}"
PLUGIN_SRC="${PLUGIN_SRC:-$COLLECTOR_ROOT/mirror_plugin}"
PAPER_URL="https://api.papermc.io/v2/projects/paper/versions/1.19.4/builds/550/downloads/paper-1.19.4-550.jar"

mkdir -p "$SERVER_DIR/plugins"
cd "$SERVER_DIR"

# Paper jar: reuse a shared one if provided to avoid N downloads.
if [ -n "${PAPER_JAR:-}" ] && [ -f "$PAPER_JAR" ]; then
  cp -n "$PAPER_JAR" "$SERVER_DIR/paper-1.19.4.jar"
elif [ ! -f paper-1.19.4.jar ]; then
  echo "[setup:$INSTANCE] downloading Paper 1.19.4 ..."
  curl -fsSL -o paper-1.19.4.jar "$PAPER_URL"
fi

echo "eula=true" > eula.txt

cat > server.properties <<EOF
server-port=${PORT}
online-mode=false
level-name=world
level-type=minecraft\:${LEVEL_TYPE}
generate-structures=true
difficulty=peaceful
gamemode=survival
spawn-monsters=false
spawn-animals=false
spawn-npcs=false
allow-nether=false
max-players=10
view-distance=10
simulation-distance=10
motd=Blockscope Collector ${INSTANCE}
enable-command-block=false
white-list=false
allow-flight=true
enforce-secure-profile=false
EOF

# Optional: drop in a prebuilt build map.
if [ -n "${WORLD:-}" ]; then
  if [ -d "$WORLD" ]; then
    echo "[setup:$INSTANCE] installing world from $WORLD ..."
    rm -rf "$SERVER_DIR/world"
    cp -r "$WORLD" "$SERVER_DIR/world"
  else
    echo "[setup:$INSTANCE] WARNING: WORLD=$WORLD not found; generating fresh world"
  fi
fi

# Build the BlockscopeMirror plugin with javac (no maven on Brev).
echo "[setup:$INSTANCE] building BlockscopeMirror plugin ..."
API_JAR="$(find "$SERVER_DIR/libraries" -path '*paper-api*' -name '*.jar' 2>/dev/null | head -1 || true)"
if [ -z "$API_JAR" ]; then
  echo "[setup:$INSTANCE] paper-api not unpacked yet; booting briefly to extract it ..."
  ( cd "$SERVER_DIR" && timeout 120 java -Xmx2G -jar paper-1.19.4.jar nogui >/dev/null 2>&1 || true )
  API_JAR="$(find "$SERVER_DIR/libraries" -path '*paper-api*' -name '*.jar' 2>/dev/null | head -1)"
fi
CP="$(find "$SERVER_DIR/libraries" -name '*.jar' | tr '\n' ':')"
BUILD="$BASE/plugin_build"
rm -rf "$BUILD"; mkdir -p "$BUILD/classes"
javac -cp "$CP" -d "$BUILD/classes" $(find "$PLUGIN_SRC/src" -name '*.java')
cp "$PLUGIN_SRC/resources/plugin.yml" "$BUILD/classes/plugin.yml"
( cd "$BUILD/classes" && jar cf "$SERVER_DIR/plugins/BlockscopeMirror.jar" . )
echo "[setup:$INSTANCE] plugin -> $SERVER_DIR/plugins/BlockscopeMirror.jar"
echo "[setup:$INSTANCE] done (PORT=$PORT, SERVER_DIR=$SERVER_DIR)"
