#!/usr/bin/env bash
# Stand up a standalone Paper 1.19.4 test server on Brev for the mirror-camera
# prototype. Non-default port 25599, flat world, peaceful, online-mode=false.
# Builds the BlockscopeMirror plugin with javac against the Paper jar (no maven).
#
# Idempotent: safe to re-run. Does NOT touch the production VPS or any existing
# server (the lab "lodestone" screen on :25565 is left alone).
set -euo pipefail

ROOT="${MIRROR_ROOT:-$HOME/mirror_prototype}"
SERVER_DIR="$ROOT/server"
PLUGIN_SRC="${PLUGIN_SRC:-$ROOT/mirror_plugin}"
PORT="${MIRROR_PORT:-25599}"
PAPER_URL="https://api.papermc.io/v2/projects/paper/versions/1.19.4/builds/550/downloads/paper-1.19.4-550.jar"

mkdir -p "$SERVER_DIR/plugins"
cd "$SERVER_DIR"

if [ ! -f paper-1.19.4.jar ]; then
  echo "[setup] downloading Paper 1.19.4 ..."
  curl -fsSL -o paper-1.19.4.jar "$PAPER_URL"
fi

echo "[setup] writing server config ..."
echo "eula=true" > eula.txt

cat > server.properties <<EOF
server-port=${PORT}
online-mode=false
level-type=minecraft\:flat
generate-structures=false
difficulty=peaceful
gamemode=survival
spawn-monsters=false
spawn-animals=false
spawn-npcs=false
allow-nether=false
max-players=10
view-distance=10
simulation-distance=10
motd=Blockscope Mirror Prototype
enable-command-block=false
white-list=false
allow-flight=true
EOF

# Build the plugin jar with javac (no maven on Brev).
# Paper's launcher (paperclip) only unpacks the real paper-api jar on first server
# run, so we compile against that unpacked API jar. If it's missing, do a short
# throwaway server boot to unpack it.
echo "[setup] building BlockscopeMirror plugin ..."
API_JAR="$(find "$SERVER_DIR/libraries" -path '*paper-api*' -name '*.jar' 2>/dev/null | head -1 || true)"
if [ -z "$API_JAR" ]; then
  echo "[setup] paper-api not unpacked yet; booting server briefly to extract it ..."
  ( cd "$SERVER_DIR" && timeout 90 java -Xmx2G -jar paper-1.19.4.jar nogui >/dev/null 2>&1 || true )
  API_JAR="$(find "$SERVER_DIR/libraries" -path '*paper-api*' -name '*.jar' 2>/dev/null | head -1)"
fi
echo "[setup] compiling against API: $API_JAR"
# Adventure (kyori) and other API deps are separate jars under libraries/ — put the
# whole unpacked library set on the compile classpath.
CP="$(find "$SERVER_DIR/libraries" -name '*.jar' | tr '\n' ':')"
BUILD="$ROOT/plugin_build"
rm -rf "$BUILD"
mkdir -p "$BUILD/classes"
javac -cp "$CP" \
  -d "$BUILD/classes" \
  $(find "$PLUGIN_SRC/src" -name '*.java')
cp "$PLUGIN_SRC/resources/plugin.yml" "$BUILD/classes/plugin.yml"
( cd "$BUILD/classes" && jar cf "$SERVER_DIR/plugins/BlockscopeMirror.jar" . )
echo "[setup] plugin jar -> $SERVER_DIR/plugins/BlockscopeMirror.jar"

echo "[setup] done. Start with: scripts/start_server.sh"
