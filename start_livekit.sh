#!/usr/bin/env bash
#
# start_livekit.sh — Start the local LiveKit dev server in Docker for Lab Brain.
#
# Reads api_key / api_secret straight out of config.json so this script
# can never drift out of sync with what livekit_rooms.py signs tokens with.
# (That drift — "devsecret" in config vs. "secret" from livekit-server --dev —
# was the root cause of the earlier 503 / "Server disconnected" errors.)
#
# Usage:
#   chmod +x start_livekit.sh
#   ./start_livekit.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="config.json"
CONTAINER_NAME="lab-brain-livekit"
PORT_SIGNAL=7880
PORT_RTC_TCP=7881
PORT_RTC_UDP=7882

echo "==> Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon isn't running. Start Docker Desktop (or 'sudo systemctl start docker') and re-run this script." >&2
  exit 1
fi

echo "==> Reading LiveKit credentials from config.json..."
if [[ -f "$CONFIG_FILE" ]]; then
  # NOTE: deliberately uses a relative filename here, not an absolute path.
  # On Git Bash / MINGW64, an absolute path like /c/Users/... is MSYS
  # notation that native python.exe doesn't understand when embedded
  # inside a -c string (it's not auto-translated like a bare argv path
  # would be). Running from the already-cd'd directory with a relative
  # name sidesteps the whole translation problem.
  API_KEY=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['livekit']['api_key'])")
  API_SECRET=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['livekit']['api_secret'])")
else
  echo "config.json not found in ${SCRIPT_DIR} — falling back to devkey/secret defaults." >&2
  API_KEY="devkey"
  API_SECRET="secret"
fi
echo "    api_key=${API_KEY}  api_secret=${API_SECRET}"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "==> Removing existing '${CONTAINER_NAME}' container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "==> Starting LiveKit dev server..."
# --node-ip=127.0.0.1 is required when running LiveKit in Docker for local
# dev: without it, the server advertises its internal Docker bridge IP
# (e.g. 172.17.0.2) as the WebRTC media address, which the browser on the
# host machine can't reach — this is what causes "could not establish pc
# connection" even though signaling/auth succeed fine.
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${PORT_SIGNAL}:7880" \
  -p "${PORT_RTC_TCP}:7881" \
  -p "${PORT_RTC_UDP}:7882/udp" \
  -e LIVEKIT_KEYS="${API_KEY}: ${API_SECRET}" \
  livekit/livekit-server \
  --dev \
  --node-ip=127.0.0.1 \
  >/dev/null

echo "==> Waiting for LiveKit to come up on :${PORT_SIGNAL}..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT_SIGNAL}/" >/dev/null 2>&1; then
    echo "✅ LiveKit is up — ws://localhost:${PORT_SIGNAL}  (container: ${CONTAINER_NAME})"
    echo "   Logs:  docker logs -f ${CONTAINER_NAME}"
    echo "   Stop:  docker rm -f ${CONTAINER_NAME}"
    exit 0
  fi
  sleep 1
done

echo "⚠️  LiveKit didn't respond within 30s. Check what happened with:" >&2
echo "    docker logs ${CONTAINER_NAME}" >&2
exit 1