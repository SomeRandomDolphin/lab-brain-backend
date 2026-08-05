#!/usr/bin/env bash
#
# start_services.sh — Start all Lab Brain backend services in Docker.
#
# Services managed:
#
#   LiveKit  — WebRTC SFU for real-time audio/video routing.
#              Reads livekit.api_key / livekit.api_secret from config.json so
#              token signing in livekit_rooms.py always stays in sync.
#              (Credential drift was the root cause of earlier 503 /
#              "Server disconnected" errors.)
#
#   Supabase — Local Postgres + REST + Auth + Storage stack.
#              Uses the Supabase CLI when available; falls back to Docker
#              Compose by cloning the official supabase/supabase repo.
#              Reads supabase.url / supabase.key from config.json.
#              All Compose containers are prefixed "lab-brain-supabase".
#
#   Ollama   — Local LLM inference server (OpenAI-compatible API).
#              Pulls vision_model, dialogue_model, and embedding_model
#              straight from config.json into a shared Docker volume so
#              models survive restarts.
#              GPU passthrough is enabled automatically when the NVIDIA
#              container runtime is detected.
#
# Usage:
#   chmod +x start_services.sh
#   ./start_services.sh                        # all three services
#   ./start_services.sh --livekit-only
#   ./start_services.sh --supabase-only
#   ./start_services.sh --ollama-only
#   ./start_services.sh --no-livekit           # skip LiveKit, run the rest
#   ./start_services.sh --no-supabase
#   ./start_services.sh --no-ollama
#
set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BOLD}==>${NC} $*"; }
success() { echo -e "${GREEN}✅${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠️ ${NC} $*" >&2; }
die()     { echo -e "${RED}❌${NC} $*" >&2; exit 1; }

# ── argument parsing ──────────────────────────────────────────────────────────

RUN_LIVEKIT=true
RUN_SUPABASE=true
RUN_OLLAMA=true

for arg in "$@"; do
  case "$arg" in
    --livekit-only)  RUN_SUPABASE=false; RUN_OLLAMA=false ;;
    --supabase-only) RUN_LIVEKIT=false;  RUN_OLLAMA=false ;;
    --ollama-only)   RUN_LIVEKIT=false;  RUN_SUPABASE=false ;;
    --no-livekit)    RUN_LIVEKIT=false ;;
    --no-supabase)   RUN_SUPABASE=false ;;
    --no-ollama)     RUN_OLLAMA=false ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "Unknown argument: $arg  (try --help)" ;;
  esac
done

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="config.json"

# ── Docker guard ──────────────────────────────────────────────────────────────

info "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  die "Docker daemon isn't running. Start Docker Desktop (or 'sudo systemctl start docker') and re-run."
fi

# ── read_config <python-subscript> [default] ──────────────────────────────────
# Reads a value from config.json via python3 (no jq required).
# Example: read_config "['livekit']['api_key']" "devkey"
read_config() {
  local expr="$1"
  local default="${2:-}"
  if [[ -f "$CONFIG_FILE" ]]; then
    python3 -c "
import json
try:
    d = json.load(open('${CONFIG_FILE}'))
    v = d${expr}
    print('' if v is None else v)
except (KeyError, TypeError):
    print('${default}')
" 2>/dev/null || echo "$default"
  else
    echo "$default"
  fi
}

# ── remove_container <name> ───────────────────────────────────────────────────
remove_container() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    info "Removing existing '${name}' container..."
    docker rm -f "$name" >/dev/null
  fi
}

# ── port_in_use <port> ────────────────────────────────────────────────────────
port_in_use() {
  timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$1" 2>/dev/null
}

# ── wait_for_port_free <port> <timeout_secs> ──────────────────────────────────
# `docker rm -f` returns as soon as the container object is gone, but the
# docker-proxy process handling a published port (`-p host:container`) can
# take a beat longer to release the socket — especially for a container that
# failed mid-startup (e.g. a GPU device request failing after the port was
# already bound). Re-running `docker run -p <port>:...` immediately after
# `remove_container` can lose that race and fail with "address already in
# use" even though nothing is really still using it. This just gives it a
# short grace window instead of failing immediately.
wait_for_port_free() {
  local port="$1" timeout_s="${2:-10}" waited=0
  while port_in_use "$port"; do
    if (( waited >= timeout_s )); then
      warn "Port ${port} still appears to be in use after ${timeout_s}s. If the next" \
           "step fails with 'address already in use', check what's actually holding it:" \
           "sudo lsof -i:${port}  (or)  sudo ss -ltnp | grep ${port}"
      return 1
    fi
    sleep 0.5
    waited=$((waited + 1))
  done
  return 0
}

# ── force_remove_dir <path> ───────────────────────────────────────────────────
# Plain `rm -rf` on a git clone can fail with "Permission denied": git marks
# .git/objects/** files 0444 (read-only) on purpose, and on some setups
# (e.g. Docker Desktop's bind-mount layer) that's enough to block deletion.
# Restore write permissions first, then remove; only reach for sudo if that
# still isn't enough, and say so explicitly rather than failing silently.
force_remove_dir() {
  local dir="$1"
  [[ -e "$dir" ]] || return 0

  chmod -R u+w "$dir" 2>/dev/null || true
  if rm -rf "$dir" 2>/dev/null; then
    success "Removed ${dir}."
    return
  fi

  warn "Couldn't remove ${dir} without elevated permissions — retrying with sudo."
  if sudo rm -rf "$dir"; then
    success "Removed ${dir} (via sudo)."
  else
    warn "Failed to remove ${dir} even with sudo. Remove it manually: sudo rm -rf ${dir}"
  fi
}

# ── wait_for_http <url> <timeout_secs> <interval_secs> ───────────────────────
wait_for_http() {
  local url="$1" timeout="$2" interval="${3:-1}"
  local elapsed=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    sleep "$interval"
    elapsed=$(( elapsed + interval ))
    if (( elapsed >= timeout )); then
      return 1
    fi
  done
  return 0
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVEKIT
# ═══════════════════════════════════════════════════════════════════════════════

LIVEKIT_CONTAINER="lab-brain-livekit"
LIVEKIT_PORT_SIGNAL=7880
LIVEKIT_PORT_RTC_TCP=7881
LIVEKIT_PORT_RTC_UDP=7882

start_livekit() {
  info "Setting up LiveKit..."

  # Read credentials from config.json — same source livekit_rooms.py uses,
  # so token signing always matches the server's expected key/secret pair.
  local api_key api_secret
  api_key=$(read_config    "['livekit']['api_key']"    "devkey")
  api_secret=$(read_config "['livekit']['api_secret']" "devsecret")
  echo "    api_key=${api_key}  api_secret=${api_secret}"

  remove_container "$LIVEKIT_CONTAINER"

  info "Starting LiveKit dev server..."
  # --node-ip=127.0.0.1 is required for local Docker dev: without it the
  # server advertises its internal bridge IP (e.g. 172.17.0.2) as the WebRTC
  # media address, which the host browser can't reach — the symptom is
  # "could not establish pc connection" even though signaling succeeds.
  docker run -d \
    --name "$LIVEKIT_CONTAINER" \
    -p "${LIVEKIT_PORT_SIGNAL}:7880" \
    -p "${LIVEKIT_PORT_RTC_TCP}:7881" \
    -p "${LIVEKIT_PORT_RTC_UDP}:7882/udp" \
    -e LIVEKIT_KEYS="${api_key}: ${api_secret}" \
    livekit/livekit-server \
    --dev \
    --node-ip=127.0.0.1 \
    >/dev/null

  info "Waiting for LiveKit on :${LIVEKIT_PORT_SIGNAL}..."
  if wait_for_http "http://localhost:${LIVEKIT_PORT_SIGNAL}/" 30 1; then
    success "LiveKit is up."
    echo "  WebSocket  → ws://localhost:${LIVEKIT_PORT_SIGNAL}"
    echo "  Logs:        docker logs -f ${LIVEKIT_CONTAINER}"
    echo "  Stop:        docker rm -f ${LIVEKIT_CONTAINER}"
  else
    die "LiveKit didn't respond within 30 s. Check logs: docker logs ${LIVEKIT_CONTAINER}"
  fi
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Follows the official self-hosting guide exactly:
#   https://supabase.com/docs/guides/self-hosting/docker
#
# Steps performed:
#   1. Sparse-clone only the docker/ directory from supabase/supabase
#   2. Copy compose files into a supabase-project/ directory (the guide's layout)
#   3. Generate secrets via utils/generate-keys.sh + utils/add-new-auth-keys.sh
#      (skipped if .env already has real secrets, i.e. not a first-time setup)
#   4. Pull images
#   5. Start via `sh run.sh start`  (wraps `docker compose up -d --wait`)
#   6. Health-check via `docker compose ps`
#   7. Write supabase.url + supabase.key back into config.json
#
# All Compose-managed containers are prefixed with the project name
# "lab-brain-supabase" (set via COMPOSE_PROJECT_NAME) so they appear as
# lab-brain-supabase-db-1, lab-brain-supabase-kong-1, etc. — consistent
# with the lab-brain-livekit and lab-brain-ollama naming convention.

SUPABASE_PROJECT_DIR="${SCRIPT_DIR}/supabase-project"
SUPABASE_REPO_DIR="${SCRIPT_DIR}/.supabase-repo"
# The guide's API gateway (Kong) listens on this port.
# This is the single source of truth for the port — changing it here also
# patches supabase-project/.env (KONG_HTTP_PORT) on a fresh install via
# set_kong_port() below, so the two never drift out of sync.
SUPABASE_API_PORT=8080
# Compose project name — controls the container name prefix
SUPABASE_COMPOSE_PROJECT="lab-brain-supabase"

# Patches the Supabase docker-compose.yml so Postgres is reachable directly
# on the host instead of only through the Supavisor pooler:
#   1. Comments out the entire `supavisor:` service block.
#   2. Adds `ports: ["${POSTGRES_PORT}:${POSTGRES_PORT}"]` under `db:`.
# Matches by service-name boundaries (not fixed line numbers) so it keeps
# working if the upstream supabase/supabase compose file is reformatted on
# a future sparse-clone, and is safe to re-run (no-ops if already patched).
patch_supabase_compose() {
  local compose_file="$1"
  [[ -f "$compose_file" ]] || die "Compose file not found: ${compose_file}"

  python3 - "$compose_file" <<'PYEOF'
import re
import sys

path = sys.argv[1]
with open(path, "r") as fh:
    lines = fh.readlines()

# ── 1. Comment out the `supavisor:` service block ──────────────────────────
out = []
in_block = False
pooler_changed = False
for line in lines:
    if re.match(r"^  supavisor:\s*$", line):
        in_block = True
    elif in_block and re.match(r"^\S", line):
        # Dedented to a new top-level key (e.g. "volumes:") — block is over.
        in_block = False

    if in_block and not line.lstrip().startswith("#"):
        if line.strip():
            line = "# " + line
        pooler_changed = True
    out.append(line)
lines = out

# ── 2. Expose `db` directly on POSTGRES_PORT ────────────────────────────────
text = "".join(lines)
db_changed = False
marker = "  db:\n    container_name: supabase-db\n"
if marker in text and "- ${POSTGRES_PORT}:${POSTGRES_PORT}" not in text:
    text = text.replace(
        marker,
        marker + "    ports:\n      - ${POSTGRES_PORT}:${POSTGRES_PORT}\n",
        1,
    )
    db_changed = True

with open(path, "w") as fh:
    fh.write(text)

# ── 3. Give every healthcheck a real start_period ───────────────────────────
# Several services (rest, auth, imgproxy, functions, kong) ship with NO
# start_period at all, meaning Docker starts counting failed healthchecks
# from the instant the container starts — not once it's had a fair chance to
# boot. On a loaded VM (9+ containers competing for CPU/disk, Postgres still
# running migrations) whichever container is momentarily slow loses the race
# and gets torn down as "unhealthy", and it's a different one each run. This
# adds a 30s start_period to any healthcheck missing one, and raises any
# existing start_period under 30s — failures during start_period don't count
# against the retry budget, so a slow (but fine) boot no longer looks fatal.
hc_changed = False

def _add_start_period(m):
    global hc_changed
    hc_changed = True
    return m.group(0) + f"{m.group(1)}start_period: 30s\n"

text = re.sub(
    r"^(      )retries: \d+\n(?!      start_period)",
    _add_start_period,
    text,
    flags=re.M,
)

def _bump_start_period(m):
    global hc_changed
    indent, seconds = m.group(1), int(m.group(2))
    if seconds < 30:
        hc_changed = True
        return f"{indent}start_period: 30s\n"
    return m.group(0)

text = re.sub(
    r"^(      )start_period: (\d+)s\n",
    _bump_start_period,
    text,
    flags=re.M,
)

with open(path, "w") as fh:
    fh.write(text)

if pooler_changed:
    print("[patch_supabase_compose] commented out the supavisor service block")
if db_changed:
    print("[patch_supabase_compose] exposed db on ${POSTGRES_PORT}:${POSTGRES_PORT}")
if hc_changed:
    print("[patch_supabase_compose] gave healthchecks a 30s start_period")
if not pooler_changed and not db_changed and not hc_changed:
    print("[patch_supabase_compose] already patched — nothing to do")
PYEOF
}

# Patches supabase-project/.env so Kong's HTTP port matches SUPABASE_API_PORT
# instead of the upstream default (8000). Kong's container-side port (8000)
# is untouched — only the HOST-side KONG_HTTP_PORT env var changes, which is
# the correct way to do this (see docker-compose.yml: "${KONG_HTTP_PORT}:8000").
# Safe to re-run: no-ops if KONG_HTTP_PORT already matches.
set_kong_port() {
  local env_file="$1"
  local port="$2"
  [[ -f "$env_file" ]] || die ".env file not found: ${env_file}"

  if grep -q "^KONG_HTTP_PORT=${port}$" "$env_file"; then
    info "KONG_HTTP_PORT already set to ${port} — nothing to do."
    return
  fi

  if grep -q '^KONG_HTTP_PORT=' "$env_file"; then
    sed -i "s|^KONG_HTTP_PORT=.*|KONG_HTTP_PORT=${port}|" "$env_file"
  else
    echo "KONG_HTTP_PORT=${port}" >> "$env_file"
  fi
  success "Set KONG_HTTP_PORT=${port} in ${env_file}."
}

start_supabase() {
  info "Setting up Supabase (official self-hosting guide)..."

  local env_file="${SUPABASE_PROJECT_DIR}/.env"
  local run_sh="${SUPABASE_PROJECT_DIR}/run.sh"
  local compose_file="${SUPABASE_PROJECT_DIR}/docker-compose.yml"

  # ── Steps 1-2: sparse-clone docker/ from supabase/supabase, then copy the
  #    compose files into supabase-project/. Only done on a fresh install —
  #    if supabase-project/ already exists we skip straight past this.
  if [[ ! -d "$SUPABASE_PROJECT_DIR" ]]; then
    info "Sparse-cloning supabase/supabase docker/ into ${SUPABASE_REPO_DIR}..."
    git clone \
      --filter=blob:none \
      --no-checkout \
      --depth=1 \
      --quiet \
      https://github.com/supabase/supabase \
      "$SUPABASE_REPO_DIR" \
      || die "git clone failed — is git installed and do you have internet access?"
    git -C "$SUPABASE_REPO_DIR" sparse-checkout init --cone
    git -C "$SUPABASE_REPO_DIR" sparse-checkout set docker
    git -C "$SUPABASE_REPO_DIR" checkout --quiet
    success "Cloned supabase/supabase docker/ directory."

    info "Creating project directory: ${SUPABASE_PROJECT_DIR}"
    mkdir -p "$SUPABASE_PROJECT_DIR"
    cp -rf "${SUPABASE_REPO_DIR}/docker/." "$SUPABASE_PROJECT_DIR/"
    # Copy the example .env — real secrets generated in step 3
    cp "${SUPABASE_REPO_DIR}/docker/.env.example" "${SUPABASE_PROJECT_DIR}/.env"
    success "Compose files copied to ${SUPABASE_PROJECT_DIR}."

    # The sparse clone was only ever a means to get docker-compose.yml/.env
    # into supabase-project/ — nothing after this point reads from it, so
    # remove it rather than leaving a stray .git checkout lying around.
    force_remove_dir "$SUPABASE_REPO_DIR"
  else
    info "Project directory already exists at ${SUPABASE_PROJECT_DIR} — skipping clone/copy."
  fi

  # ── Step 2b: expose Postgres directly, bypass Supavisor ──────────────────
  patch_supabase_compose "$compose_file"

  # ── Step 2c: move Kong off its default port ──────────────────────────────
  set_kong_port "$env_file" "$SUPABASE_API_PORT"

  # ── Step 3: generate secrets (first-time only) ───────────────────────────
  local pg_pass
  pg_pass=$(grep '^POSTGRES_PASSWORD=' "$env_file" | cut -d= -f2- | tr -d '"' || true)

  if [[ "$pg_pass" == "your-super-secret-and-long-postgres-password" || -z "$pg_pass" ]]; then
    warn "Placeholder secrets detected in .env — generating real secrets now."
    warn "IMPORTANT: Never run Supabase with the default .env.example passwords."
    echo ""

    if [[ -f "${SUPABASE_PROJECT_DIR}/utils/generate-keys.sh" ]]; then
      info "Running utils/generate-keys.sh..."
      (cd "$SUPABASE_PROJECT_DIR" && sh utils/generate-keys.sh) \
        || die "generate-keys.sh failed."
    else
      warn "utils/generate-keys.sh not found — generating secrets with openssl."
      local jwt_secret
      jwt_secret=$(openssl rand -base64 48)
      local pg_password
      pg_password=$(openssl rand -hex 16)
      local secret_key_base
      secret_key_base=$(openssl rand -base64 48)
      local vault_enc_key
      vault_enc_key=$(openssl rand -hex 16)

      sed -i "s|your-super-secret-and-long-postgres-password|${pg_password}|g" "$env_file"
      sed -i "s|your-super-secret-jwt-token-with-at-least-32-characters-long|${jwt_secret}|g" "$env_file"
      sed -i "s|your-secret-key-base-at-least-32-characters|${secret_key_base}|g" "$env_file"
      sed -i "s|your-vault-enc-key-at-least-32-characters|${vault_enc_key}|g" "$env_file"
      success "Minimal secrets written to ${env_file}."
    fi

    # NOTE: utils/add-new-auth-keys.sh (below) is what's expected to bring
    # ANON_KEY/SERVICE_ROLE_KEY in sync with JWT_SECRET when the openssl
    # fallback above ran. If it's skipped or fails, ANON_KEY/SERVICE_ROLE_KEY
    # can be left as .env.example's placeholder demo tokens (iss:
    # "supabase-demo") — signed with a different, publicly-known secret —
    # while JWT_SECRET has just been randomized. Every service that checks
    # the token's signature (kong, auth, rest, storage, realtime) then
    # rejects it with 403. If that happens, re-run
    # `sh utils/add-new-auth-keys.sh` manually and restart the stack.
    if [[ -f "${SUPABASE_PROJECT_DIR}/utils/add-new-auth-keys.sh" ]]; then
      info "Running utils/add-new-auth-keys.sh (asymmetric JWT keys)..."
      (cd "$SUPABASE_PROJECT_DIR" && sh utils/add-new-auth-keys.sh) \
        || warn "add-new-auth-keys.sh failed — JWT signing will use symmetric keys only."
    else
      warn "utils/add-new-auth-keys.sh not found in ${SUPABASE_PROJECT_DIR}/utils/ — skipping."
    fi

    echo ""
    warn "─────────────────────────────────────────────────────────────────"
    warn "REVIEW YOUR SECRETS before proceeding:"
    warn "  cat ${env_file}"
    warn "  Pay special attention to: POSTGRES_PASSWORD, JWT_SECRET,"
    warn "  ANON_KEY, SERVICE_ROLE_KEY, DASHBOARD_PASSWORD"
    warn "─────────────────────────────────────────────────────────────────"
    echo ""
    read -r -p "  Secrets look good? Continue starting Supabase? [y/N] " yn
    [[ "$yn" =~ ^[Yy] ]] || die "Aborted. Edit ${env_file} and re-run."
  else
    info "Existing secrets found in .env — skipping key generation."
  fi

  # Re-read in case Step 3 just generated it — the earlier read may have
  # captured the placeholder value.
  pg_pass=$(grep '^POSTGRES_PASSWORD=' "$env_file" | cut -d= -f2- | tr -d '"' || true)

  # ── Step 4: catch stale Postgres credentials before they cause a confusing
  #    crash-loop ─────────────────────────────────────────────────────────
  # Postgres only applies POSTGRES_PASSWORD (and roles.sql's admin roles:
  # supabase_storage_admin, supabase_auth_admin, authenticator, supabase_admin)
  # the FIRST time its data directory is initialized. If .env's password ever
  # changes afterward (e.g. a re-run regenerated secrets, or the folder was
  # copied from an older attempt), every service that authenticates as one of
  # those roles fails with "password authentication failed" (28P01) — auth,
  # rest, storage, and realtime all hit this; db/kong/studio don't, so it
  # looks confusingly selective. Catch it up front instead of watching four
  # containers crash-loop.
  check_db_password_drift() {
    local db_data_dir="${SUPABASE_PROJECT_DIR}/volumes/db/data"
    local marker_file="${SUPABASE_PROJECT_DIR}/volumes/db/.postgres_password_sha256"
    local current_hash
    current_hash=$(printf '%s' "$pg_pass" | sha256sum | cut -d' ' -f1)

    if [[ -d "$db_data_dir" ]] && [[ -n "$(ls -A "$db_data_dir" 2>/dev/null)" ]]; then
      # Postgres has already been initialized at least once.
      if [[ -f "$marker_file" ]]; then
        local stored_hash
        stored_hash=$(cat "$marker_file")
        if [[ "$stored_hash" != "$current_hash" ]]; then
          die "POSTGRES_PASSWORD in ${env_file} doesn't match the password Postgres was
  initialized with — auth/rest/storage/realtime will crash-loop with
  'password authentication failed' until this is fixed.
  This is a fresh deploy, so the simplest fix is to wipe Postgres and let it
  reinitialize with the current .env:
    cd ${SUPABASE_PROJECT_DIR}
    COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} docker compose down
    sudo rm -rf volumes/db/data volumes/db/.postgres_password_sha256
    COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} sh run.sh start
  (Only do this if there's no real data in the DB yet.)"
        fi
      else
        # Data dir predates this check — record the current hash so future
        # runs can detect drift, without blocking this run.
        echo "$current_hash" > "$marker_file"
      fi
    else
      mkdir -p "$(dirname "$marker_file")"
      echo "$current_hash" > "$marker_file"
    fi
  }
  check_db_password_drift

  # ── Step 5: pull images ──────────────────────────────────────────────────
  info "Pulling Supabase Docker images (may take a while on first run)..."
  (cd "$SUPABASE_PROJECT_DIR" && COMPOSE_PROJECT_NAME="$SUPABASE_COMPOSE_PROJECT" docker compose pull) \
    || die "docker compose pull failed. Check your internet connection."

  # ── Step 6: start via run.sh ─────────────────────────────────────────────
  info "Starting Supabase stack via run.sh start..."
  if [[ -f "$run_sh" ]]; then
    (cd "$SUPABASE_PROJECT_DIR" && COMPOSE_PROJECT_NAME="$SUPABASE_COMPOSE_PROJECT" sh run.sh start) \
      || die "run.sh start failed. Check logs: cd ${SUPABASE_PROJECT_DIR} && COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} sh run.sh logs"
  else
    warn "run.sh not found — falling back to docker compose up -d --wait."
    (cd "$SUPABASE_PROJECT_DIR" && COMPOSE_PROJECT_NAME="$SUPABASE_COMPOSE_PROJECT" docker compose up -d --wait) \
      || die "docker compose up failed. Check logs: cd ${SUPABASE_PROJECT_DIR} && COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} docker compose logs"
  fi

  # ── Step 7: extract keys and update config.json ──────────────────────────
  local anon_key publishable_key api_url
  publishable_key=$(grep -E '^(SUPABASE_PUBLISHABLE_KEY|ANON_KEY)=' "$env_file" \
    | head -1 | cut -d= -f2- | tr -d '"' || true)
  anon_key="$publishable_key"
  api_url="http://localhost:${SUPABASE_API_PORT}"

  if [[ -f "$CONFIG_FILE" ]] && [[ -n "$anon_key" ]]; then
    python3 -c "
import json, sys
with open('${CONFIG_FILE}', 'r') as f:
    cfg = json.load(f)
cfg.setdefault('supabase', {})
cfg['supabase']['url'] = '${api_url}'
cfg['supabase']['key'] = '${anon_key}'
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(cfg, f, indent=2)
print('config.json updated.')
" && success "config.json updated: supabase.url=${api_url}" \
  || warn "Could not auto-update config.json — set supabase.url and supabase.key manually."
  fi

  echo ""
  local pg_port
  pg_port=$(grep '^POSTGRES_PORT=' "$env_file" | cut -d= -f2- | tr -d '"' || true)
  pg_port="${pg_port:-5432}"
  local pg_pass_display
  pg_pass_display=$(grep '^POSTGRES_PASSWORD=' "$env_file" | cut -d= -f2- | tr -d '"' || true)

  echo "  Studio (Dashboard) → http://localhost:${SUPABASE_API_PORT}"
  echo "  REST API           → http://localhost:${SUPABASE_API_PORT}/rest/v1/"
  echo "  Auth API           → http://localhost:${SUPABASE_API_PORT}/auth/v1/"
  echo "  Storage API        → http://localhost:${SUPABASE_API_PORT}/storage/v1/"
  echo "  Postgres (direct)  → postgresql+asyncpg://postgres:${pg_pass_display}@localhost:${pg_port}/postgres"
  echo "  (Set this as SUPABASE_DB_URL in your .env for Alembic migrations)"
  echo ""
  echo "  Containers:   docker ps --filter name=${SUPABASE_COMPOSE_PROJECT}"
  echo "  Credentials:  cat ${env_file}"
  echo "                cd ${SUPABASE_PROJECT_DIR} && COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} sh run.sh secrets"
  echo "  Logs:         cd ${SUPABASE_PROJECT_DIR} && COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} sh run.sh logs [service]"
  echo "  Stop:         cd ${SUPABASE_PROJECT_DIR} && COMPOSE_PROJECT_NAME=${SUPABASE_COMPOSE_PROJECT} sh run.sh stop"
  echo "  Health:       docker ps --filter name=${SUPABASE_COMPOSE_PROJECT}"
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
#  OLLAMA
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_CONTAINER="lab-brain-ollama"
OLLAMA_PORT=11434

start_ollama() {
  info "Setting up Ollama..."

  # Model names come directly from config.json — stays in sync with the backend
  local vision_model dialogue_model embedding_model
  vision_model=$(read_config    "['local_llm']['vision_model']"    "qwen3-vl:4b")
  dialogue_model=$(read_config  "['local_llm']['dialogue_model']"  "qwen3:4b")
  embedding_model=$(read_config "['local_llm']['embedding_model']" "qwen3-embedding:0.6b")
  echo "  vision_model    = ${vision_model}"
  echo "  dialogue_model  = ${dialogue_model}"
  echo "  embedding_model = ${embedding_model}"

  # GUARD: if config.json points both roles at the same model tag, every
  # dialogue/QA call queues behind vision calls on the same model-runner
  # (and vice versa) regardless of OLLAMA_MAX_LOADED_MODELS/NUM_PARALLEL —
  # there's only one distinct model to load, so nothing here can help.
  # This is the exact failure mode that produced 84s+ vision calls blocking
  # ASR/QA in earlier sessions — catch it here instead of re-discovering it
  # from a frozen frontend.
  if [[ "$vision_model" == "$dialogue_model" ]]; then
    die "config.json: local_llm.vision_model and local_llm.dialogue_model are both '${vision_model}'. \
They must be different models — a VL model for vision_model, a text-only model (e.g. qwen3:4b) for \
dialogue_model — or dialogue/QA calls will queue behind every vision call. Fix config.json and re-run."
  fi

  # GPU passthrough — auto-detected, never required.
  #
  # NOTE: `docker info | grep -qi nvidia` only lights up when the
  # nvidia-container-toolkit has registered a distinct "nvidia" OCI runtime,
  # which is how GPU support works on native Linux Docker. Docker Desktop's
  # WSL2 backend (Windows/Mac) exposes GPU passthrough transparently through
  # `--gpus all` WITHOUT registering a runtime by that name — so this check
  # reports "no GPU" on a perfectly capable Windows/WSL2 + Docker Desktop
  # machine, silently falling back to CPU. That mismatch is almost certainly
  # why Ollama was serving qwen3-vl:4b on CPU (~60s/reply) instead of GPU.
  #
  # Fix: don't pre-guess capability from `docker info` — just attempt
  # `--gpus all` directly and fall back only if the run actually fails.
  # Docker fails fast (no hang) when GPU passthrough genuinely isn't there.
  #
  # OLLAMA_KEEP_ALIVE: Ollama's default is to unload a model after 5 minutes
  # of inactivity. A meeting can easily go quieter than that between
  # summons, so without this every subsequent QA reply pays the same
  # multi-minute cold-load cost as the very first one, even with the
  # server-side warmup() call. 1h keeps it resident for any realistic
  # meeting length; adjust if RAM/VRAM is tight.
  local ollama_keep_alive="1h"

  # OLLAMA_MAX_LOADED_MODELS: this app runs THREE models against the same
  # Ollama instance — dialogue_model (qwen3:4b) for QA/summary, vision_model
  # (qwen3-vl:4b) for the periodic frame analysis in session_pipeline.py's
  # _vision_worker, and (as of the qwen3-embedding switch) embedding_model
  # (qwen3-embedding:0.6b) for LKC retrieval in lkc_retrieval.py, which now
  # gets called on essentially every QA turn. Ollama's default cap on
  # simultaneously loaded models is 1 in CPU-only setups (which this
  # already falls back to whenever GPU passthrough fails — see the
  # nvidia-smi check below). With the cap too low, any embedding call that
  # comes in while the dialogue/vision model is loaded forces Ollama to
  # evict one and reload it — a full cold-start on the next call, despite
  # warmup() and OLLAMA_KEEP_ALIVE both being set correctly. That model-swap
  # thrashing, not a slow model, is what actually produced a 147s "warm" QA
  # reply in one observed session (before embedding was even in the mix).
  # Raising this to 3 lets all three stay resident together as long as
  # there's enough RAM/VRAM for all of them at once — the embedding model is
  # small (639 MB at the 0.6b tag) so this is usually the cheapest of the
  # three slots. Confirm afterwards with `docker exec lab-brain-ollama
  # ollama ps`, which should list all THREE models at once instead of
  # swapping between them.
  local ollama_max_loaded_models="3"

  # OLLAMA_NUM_PARALLEL: number of concurrent requests a single loaded model
  # will process at once (default 1). Each extra slot allocates its own
  # KV-cache on top of the base context, so this is left at 1 unless you've
  # confirmed spare VRAM/RAM headroom on the host — with vision_model and
  # dialogue_model now split across two model slots (see guard above), 1 is
  # usually enough: the two models already run independently of each other,
  # this only matters if you expect >1 concurrent request to the SAME model
  # (e.g. two simultaneous meetings). Bump to 2 and watch `ollama ps` /
  # memory if you need that.
  local ollama_num_parallel="1"

  remove_container "$OLLAMA_CONTAINER"
  wait_for_port_free "$OLLAMA_PORT" 10

  info "Starting Ollama container (attempting GPU passthrough)..."
  gpu_err_log="$(mktemp)"
  if docker run -d \
      --name "$OLLAMA_CONTAINER" \
      --gpus all \
      -e OLLAMA_KEEP_ALIVE="${ollama_keep_alive}" \
      -e OLLAMA_MAX_LOADED_MODELS="${ollama_max_loaded_models}" \
      -e OLLAMA_NUM_PARALLEL="${ollama_num_parallel}" \
      -p "${OLLAMA_PORT}:11434" \
      -v ollama-models:/root/.ollama \
      ollama/ollama >/dev/null 2>"$gpu_err_log"; then
    success "GPU passthrough accepted — Ollama container started with --gpus all."
  else
    warn "GPU passthrough unavailable: $(tail -n1 "$gpu_err_log")"
    warn "Falling back to CPU. See notes below if you expected a GPU here."
    remove_container "$OLLAMA_CONTAINER"
    wait_for_port_free "$OLLAMA_PORT" 10
    docker run -d \
      --name "$OLLAMA_CONTAINER" \
      -e OLLAMA_KEEP_ALIVE="${ollama_keep_alive}" \
      -e OLLAMA_MAX_LOADED_MODELS="${ollama_max_loaded_models}" \
      -e OLLAMA_NUM_PARALLEL="${ollama_num_parallel}" \
      -p "${OLLAMA_PORT}:11434" \
      -v ollama-models:/root/.ollama \
      ollama/ollama >/dev/null
  fi
  rm -f "$gpu_err_log"

  info "Waiting for Ollama on :${OLLAMA_PORT}..."
  if ! wait_for_http "http://localhost:${OLLAMA_PORT}/api/tags" 30 1; then
    die "Ollama didn't respond within 30 s. Check logs: docker logs ${OLLAMA_CONTAINER}"
  fi
  success "Ollama API is up."

  # Confirm the container can actually see a GPU (passthrough can "succeed"
  # at the docker-run level yet still land on a machine with no usable
  # device, e.g. driver mismatch) — surface that clearly instead of staying
  # silent until someone notices replies are slow.
  if docker exec "$OLLAMA_CONTAINER" nvidia-smi >/dev/null 2>&1; then
    success "GPU verified inside container: $(docker exec "$OLLAMA_CONTAINER" nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  else
    warn "Container is running but nvidia-smi isn't visible inside it — Ollama is likely on CPU."
    warn "Run 'nvidia-smi' on the HOST first to confirm the driver sees a GPU at all."
  fi

  # Pull each model inside the container so files land in the shared volume
  pull_model() {
    local model="$1"
    info "Pulling model: ${model}  (may take a while on first run)..."
    docker exec "$OLLAMA_CONTAINER" ollama pull "$model" \
      && success "Model ready: ${model}" \
      || die "Failed to pull ${model}. Check connectivity inside the container."
  }

  pull_model "$vision_model"
  pull_model "$dialogue_model"
  pull_model "$embedding_model"

  # Verify all three models actually end up loaded SIMULTANEOUSLY, not just
  # pulled — a bad OLLAMA_MAX_LOADED_MODELS value, insufficient RAM/VRAM for
  # all three at once, or the vision/dialogue-model-collision bug guarded
  # against above could all still leave you with model-swap thrashing at
  # runtime. A single pull doesn't load a model into memory, so ping each
  # with a trivial call first.
  #
  # NOTE: embedding_model is an embedding-only model — it has no chat
  # template and will error out on `ollama run <model> "hi"` (that path
  # calls the generate/chat endpoint). Warm it up via /api/embed instead,
  # which is the same endpoint lkc_retrieval.py actually calls at runtime.
  info "Verifying all three models can stay resident together..."
  docker exec "$OLLAMA_CONTAINER" ollama run "$vision_model" "hi" >/dev/null 2>&1 || true
  docker exec "$OLLAMA_CONTAINER" ollama run "$dialogue_model" "hi" >/dev/null 2>&1 || true
  # NOTE: `docker exec ... curl` doesn't work here — the official
  # ollama/ollama image does not ship curl (a long-standing known gap, see
  # ollama/ollama#9781 — it's also why the image's own default healthcheck
  # is broken). That call was failing with "curl: not found" and getting
  # swallowed by `|| true`, so the embedding model was silently never
  # warmed regardless of available RAM/VRAM. Curl from the HOST instead,
  # against the already-published port — same approach wait_for_http()
  # above already uses.
  curl -sf "http://localhost:${OLLAMA_PORT}/api/embed" \
    -d "{\"model\": \"${embedding_model}\", \"input\": \"hi\"}" >/dev/null 2>&1 || true
  loaded="$(docker exec "$OLLAMA_CONTAINER" ollama ps)"
  echo "$loaded"
  if echo "$loaded" | grep -qF "$vision_model" \
    && echo "$loaded" | grep -qF "$dialogue_model" \
    && echo "$loaded" | grep -qF "$embedding_model"; then
    success "All three models resident simultaneously — no swap thrashing expected."
  else
    warn "Not all three models show as loaded. Either RAM/VRAM can't hold all of them at"
    warn "once, or OLLAMA_MAX_LOADED_MODELS (${ollama_max_loaded_models}) is being capped"
    warn "by the runtime. A dialogue/vision/embedding call arriving while another is in"
    warn "flight will trigger a cold reload each time. Check available memory, or drop to"
    warn "a smaller embedding tag (qwen3-embedding:0.6b is already the smallest)."
  fi

  echo "  API endpoint → http://localhost:${OLLAMA_PORT}/v1  (OpenAI-compatible)"
  echo "  Models:        docker exec ${OLLAMA_CONTAINER} ollama list"
  echo "  Logs:          docker logs -f ${OLLAMA_CONTAINER}"
  echo "  Stop:          docker rm -f ${OLLAMA_CONTAINER}"
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}Lab Brain — Service Launcher${NC}"
echo "────────────────────────────────────────────"
echo "  Services: LiveKit=$(${RUN_LIVEKIT} && echo on || echo off)  Supabase=$(${RUN_SUPABASE} && echo on || echo off)  Ollama=$(${RUN_OLLAMA} && echo on || echo off)"
[[ -f "$CONFIG_FILE" ]] \
  && echo "  Config:   ${SCRIPT_DIR}/${CONFIG_FILE}" \
  || warn "config.json not found in ${SCRIPT_DIR} — using defaults for all services."
echo ""

$RUN_LIVEKIT  && start_livekit
$RUN_SUPABASE && start_supabase
$RUN_OLLAMA   && start_ollama

echo ""
success "All requested services are running.  Happy hacking, Lab Brain! 🧠"