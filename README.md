# Lab Brain — Module 5 Multimodal Conversational Agent

FastAPI backend with an LKC graph, LiveKit WebRTC media layer,
SSE event streaming, Supabase persistence with Alembic-managed schema migrations,
and Supabase Auth for identity management.

---

## Project Structure

```
lab-brain-backend/
├── main.py                          # FastAPI app factory + startup hooks + uvicorn entry point
├── requirements.txt
├── alembic.ini                      # Alembic config (DB URL set at runtime from SUPABASE_DB_URL)
│
├── alembic/
│   ├── env.py                       # Async Alembic environment (asyncpg)
│   ├── script.py.mako               # Template used by `alembic revision`
│   └── versions/                    # Numbered revision scripts, 0001 … 0009
│
├── core/
│   ├── env.py                       # Loads .env into os.environ (load_env()) — runs first in main.py
│   └── logging.py                   # Logging setup
│
├── db/
│   ├── lkc_graph.py                 # LKC graph
│   ├── supabase_client.py           # Supabase client + read/write API (runtime data access only)
│   ├── migrations.py                # Alembic-based migration runner (run_migrations_async, get_migration_status)
│   ├── supabase_auth.py             # Supabase Auth layer
│   └── models.py                    # SQLAlchemy models
│
├── schemas/
│   ├── auth.py                      # RegisterRequest, AuthResponse, TosConsentRequest, …
│   ├── eval.py                      # WerRequest
│   ├── ingest.py                    # RifqiSegment
│   ├── kg_agent.py                  # KgQueryRequest / KgQueryResponse
│   ├── livekit.py                   # RoomCreateRequest / RoomCreateResponse
│   ├── migrations.py
│   └── privacy.py                   # ConsentRequest / ConsentSyncRequest
│
├── services/
│   ├── auth_service.py              # Optional custom SMTP override for reset emails
│   ├── capture.py                   # NER tagging, summon system, segment processing
│   ├── eval_metrics.py              # WER, latency, capture quality, privacy metrics
│   ├── kg_agent_client.py           # Client for the shared kg-agent literature-QA service
│   ├── lkc_retrieval.py             # Dense retrieval (qwen3-embedding via Ollama)
│   ├── privacy.py                   # Consent registry + PII redaction
│   └── vision.py                    # VLM perception layer (local LLM via OpenAI-compatible API)
│
├── pipeline/
│   ├── asr.py                       # WhisperX / faster-whisper + VAD chunker
│   ├── dialogue_service.py          # Mode FSM, speaker diarization, LLM response/summary
│   ├── livekit_rooms.py             # Token generation, room management, SSE bus
│   └── session_pipeline.py          # Audio+video → ASR → NER → Dialogue → SSE → Supabase
│
└── api/
    ├── auth.py                      # POST /auth/register|login|logout|refresh, GET/PATCH /auth/me, …
    ├── livekit.py                   # POST/GET/DELETE /livekit/room, GET /livekit/token, GET /events/{sid}
    ├── lkc.py                       # GET/DELETE /lkc, /lkc/sessions, POST /lkc/kg-query
    ├── privacy.py                   # GET/POST/DELETE /privacy/consent, POST /privacy/tos-consent
    ├── capture.py                   # /capture/ingest, /agent/summon, /ner/status, /retrieval/stats
    ├── sessions.py                  # /summary, /mode, /perception, /metrics, /config/client, /eval/wer
    ├── supabase.py                  # /supabase/sessions, /supabase/migrations
    └── websockets.py                # Legacy /ws/asr, /ws/vision, /ws/tts (backward compat)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in at minimum your Supabase and LiveKit values:

```env
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>   # used server-side only — keep secret

# Postgres connection string, used by Alembic (db/migrations.py) to run migrations.
# Find it under Project Settings → Database → Connection string ("URI" tab) in the
# Supabase dashboard, then swap the postgresql:// prefix for postgresql+asyncpg://.
SUPABASE_DB_URL=postgresql+asyncpg://postgres:<password>@db.<ref>.supabase.co:5432/postgres

LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>
```

Migrations are applied automatically on startup via Alembic — see
[Migration System](#migration-system) below.

In the Supabase dashboard, make sure **Authentication → Email Templates → Reset Password**
is configured. Supabase handles reset email delivery by default; see
[Custom SMTP override](#custom-smtp-override) below if you need to send from your own server.

### 3. Start supporting services

```bash
./start_services.sh
```

This starts LiveKit, Ollama, and Supabase and runs them on the host network, so
`host.docker.internal` (mapped via `extra_hosts` in `docker-compose.yml`) can
reach them from inside the backend container in the next step. It also
creates the `lab-brain-recordings` Docker volume that the backend and its
egress container share.

### 4. Start the backend

```bash
docker compose up --build
```

API docs: http://localhost:8000/docs

---

## Authentication

### Token flow

```
POST /auth/register  ──►  { user, token, refreshToken }
POST /auth/login     ──►  { user, token, refreshToken }
                               │           │
                    access JWT (≈1 hr)   long-lived opaque refresh token
                               │
            Authorization: Bearer <token>   (all protected endpoints)
                               │
                    POST /auth/refresh  ──►  { user, token, refreshToken }
                    (call before access token expires)
```

The `token` field in every auth response is a **Supabase JWT** (short-lived, ~1 hour).
Store both `token` and `refreshToken` client-side and call `POST /auth/refresh`
before the access token expires. The new `refreshToken` returned by `/auth/refresh`
replaces the old one (token rotation).

The SSE endpoint (`GET /events/{sid}`) is the one exception: browser
`EventSource` can't set custom headers, so `get_current_user` also accepts the
access token as a `?token=` query param for that route specifically. Every
other endpoint continues to authenticate via the `Authorization` header.

### Password reset flow

```
POST /auth/forgot-password   { email }
  └─► Supabase sends a reset email with a one-time link

User clicks link → frontend extracts ?token= from URL

POST /auth/reset-password    { token, password }
  └─► Token consumed, password updated, all sessions revoked
```

### Custom SMTP override

By default, Supabase sends reset emails through your project's email settings.
If you need to send from your own SMTP server instead, set `SMTP_HOST` and the
app will use `auth_service.send_reset_email()` instead of Supabase's delivery
(see [Environment Variables](#environment-variables) for the full `SMTP_*` set).

---

## Migration System

Schema migrations are managed by **Alembic**. Revisions live in
`alembic/versions/` as Python scripts with `upgrade()` / `downgrade()`
functions, and applied state is tracked in Postgres' own `alembic_version` table.

**Automatic on startup:** `main.py`'s `lifespan()` calls
`db.migrations.run_migrations_async()` during FastAPI startup, which runs
`alembic upgrade head`. Already-applied revisions are skipped — safe to run on
every restart. If `SUPABASE_DB_URL` isn't set, startup logs a warning and
continues rather than failing to boot.

**Manual via API** (admin only — see `require_admin` in `api/lkc.py` / `api/supabase.py`):

```bash
# Check current vs. head revision + pending list (no changes applied)
GET /supabase/migrations/status

# Apply all pending revisions
POST /supabase/migrations/run
```

**Manual via CLI** (from the project root, with `SUPABASE_DB_URL` exported):

```bash
alembic current              # show current revision
alembic history               # show full revision chain
alembic upgrade head          # apply all pending revisions
alembic downgrade -1          # roll back one revision
alembic revision -m "add x"   # scaffold a new revision file
```

**Adding a new migration:**

1. Run `alembic revision -m "describe_the_change"` — this creates a new
   numbered file in `alembic/versions/` from `script.py.mako`.
2. Fill in `upgrade()` (and `downgrade()` for rollback support) using
   `alembic.op` / `sqlalchemy` calls, e.g. `op.create_table(...)`, `op.add_column(...)`.
3. Restart the server, or call `POST /supabase/migrations/run`.

---

## API Overview

### Auth (`api/auth.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| POST | `/auth/register` | — | Create account; returns user + access + refresh tokens |
| POST | `/auth/login` | — | Sign in; returns user + access + refresh tokens |
| POST | `/auth/logout` | ✓ | Revoke current session |
| POST | `/auth/refresh` | — | Exchange refresh token for new access + refresh tokens |
| GET | `/auth/me` | ✓ | Return current user (used to rehydrate session on page load) |
| PATCH | `/auth/me` | ✓ | Update current user's name/email/avatarUrl |
| POST | `/auth/forgot-password` | — | Trigger Supabase password-reset email |
| POST | `/auth/reset-password` | — | Consume reset token, set new password, revoke all sessions |

### LiveKit / WebRTC (`api/livekit.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| POST | `/livekit/room` | ✓ | Create room, start pipeline, return session_id + JWT |
| GET | `/livekit/token?session_id=&identity=` | ✓ | Join an existing room (no anonymous guests) |
| GET | `/livekit/room/{sid}` | ✓ owner/participant | Room status + participant count |
| DELETE | `/livekit/room/{sid}` | ✓ owner | End session, snapshot metrics to Supabase |
| GET | `/events/{sid}` | ✓ owner/participant | SSE stream (transcript, agent_reply, perception, speak, …) |

### LKC Graph (`api/lkc.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| GET | `/lkc` | ✓ admin | HTML viewer of recent records |
| GET | `/lkc/stats` | ✓ | Aggregate graph statistics |
| GET | `/lkc/sessions` | ✓ | List sessions the caller can access |
| GET | `/lkc/sessions/{sid}` | ✓ owner/participant | Session records (filterable by type/time) |
| POST | `/lkc/kg-query` | ✓ | Query the shared kg-agent literature knowledge graph |
| DELETE | `/lkc` | ✓ admin | Wipe entire graph |
| DELETE | `/lkc/sessions/{sid}` | ✓ owner | Wipe one session |

### Session (`api/sessions.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| POST | `/summary/{sid}` | ✓ owner/participant | Generate LLM summary, upload report to Supabase Storage |
| GET | `/mode/{sid}` | ✓ owner/participant | Current dialogue mode + summon flag |
| GET | `/perception/{sid}` | ✓ owner/participant | Latest vision state |
| GET | `/config/client` | — | Frontend config (camera fps, tts_auto_hide_ms, lk_url) |
| GET | `/metrics` | ✓ | Session metric summaries, scoped to the caller's own sessions |
| GET | `/metrics/csv` | ✓ | CSV export, same scoping |
| POST | `/eval/wer` | ✓ owner/participant | Compute WER for a reference/hypothesis pair |

### Capture / Agent (`api/capture.py`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/capture/ingest` | Rifqi Module 2 ingest |
| GET | `/capture/confirmations/{sid}` | Poll agent confirmation queue |
| GET | `/capture/tags/{sid}` | Session tags summary |
| GET | `/capture/ner_backend` | Active NER backend status |
| GET | `/agent/summon/{sid}` | Summon status |
| POST | `/agent/summon/{sid}` | Manually summon agent |
| DELETE | `/agent/summon/{sid}` | Clear summon |
| GET | `/ner/status` | Alias for NER backend status |
| GET | `/retrieval/stats` | Embedding retriever stats |

> These endpoints currently have no auth dependency — see the `api/capture.py`
> module docstring / the note under [Project Structure](#project-structure)
> above.

### Privacy (`api/privacy.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| GET | `/privacy/status` | ✓ | Registry overview + default policy |
| POST | `/privacy/consent` | ✓ | Register/update consent (local registry only) |
| POST | `/privacy/consent/sync` | ✓ owner/participant | Dual-write: local + Supabase |
| DELETE | `/privacy/consent/{speaker}` | ✓ | Revoke consent |
| POST | `/privacy/tos-consent` | ✓ | Account-level privacy-screen decision |

### Supabase (`api/supabase.py`)

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| GET | `/supabase/status` | — | Connectivity check |
| GET | `/supabase/sessions` | ✓ | Caller's own sessions |
| GET | `/supabase/sessions/{sid}/transcripts` | ✓ owner/participant | Transcript rows |
| GET | `/supabase/sessions/{sid}/summary` | ✓ owner/participant | Persisted summary |
| GET | `/supabase/sessions/{sid}/report` | ✓ owner/participant | Report Storage URL |
| GET | `/supabase/sessions/{sid}/audio/{seg_idx}` | ✓ owner/participant | Audio segment URL |
| POST | `/supabase/migrations/run` | ✓ admin | Apply pending Alembic migrations |
| GET | `/supabase/migrations/status` | ✓ admin | Current vs. head revision + pending list |

### Legacy WebSocket (`api/websockets.py`, backward compat)

| Path | Description |
|------|-------------|
| `/ws/asr` | Raw PCM → transcript + agent_reply |
| `/ws/vision?session_id=` | JPEG bytes → perception |
| `/ws/tts?session_id=` | TTS speak events |

---

## Environment Variables

See `.env.example` for the full, authoritative list with example values. Grouped here
by area:

### App / Server

| Variable | Description |
|----------|-------------|
| `SERVER_HOST` | Host uvicorn binds to when run via `python main.py` (default `0.0.0.0`) |
| `SERVER_PORT` | Port uvicorn binds to when run via `python main.py` (default `8000`) |
| `FRONTEND_URL` | Base URL used for password-reset links |

### Supabase

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL (REST/Storage API) |
| `SUPABASE_SERVICE_KEY` | Service-role secret key (server-side only — keep secret) |
| `SUPABASE_DB_URL` | Postgres connection string used by Alembic, e.g. `postgresql+asyncpg://postgres:<pw>@<host>:5432/postgres`. Without it, startup migrations are skipped with a warning. |
| `SUPABASE_STORE_AUDIO` | Whether audio segments are persisted to Supabase Storage (default `true`) — gates `GET /supabase/sessions/{sid}/audio/{seg_idx}` |
| `SUPABASE_STORE_VISION` | Whether vision/perception snapshots are persisted to Supabase |

### LiveKit

| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | LiveKit server URL the backend connects to internally |
| `LIVEKIT_PUBLIC_URL` | Browser-facing LiveKit URL returned to clients; falls back to `LIVEKIT_URL` if unset (see `api/livekit.py` / `api/sessions.py`) — set this explicitly for any deployment where the browser reaches the server over a LAN/Tailscale/public address |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Must match whatever your LiveKit server (or `start_services.sh`'s container) was started with |
| `LIVEKIT_NODE_IP` | Node IP for LiveKit's own config |
| `LIVEKIT_EGRESS_ENABLED` | Enables the recording/egress pipeline |

### Egress recordings

| Variable | Description |
|----------|-------------|
| `RECORDINGS_APPUSER_UID` / `RECORDINGS_APPUSER_GID` | Must match the backend image's non-root `appuser` (UID/GID `1000` per the Dockerfile) so the shared `lab-brain-recordings` volume is writable by both the egress container and this app |

### Local LLM (Ollama-compatible)

| Variable | Description |
|----------|-------------|
| `LOCAL_LLM_BASE_URL` | OpenAI-compatible base URL (Ollama) used by dialogue + vision + embedding calls |
| `LOCAL_LLM_API_KEY` | Placeholder key for the OpenAI-compatible client (Ollama ignores the value) |
| `LOCAL_LLM_HF_TOKEN` | HuggingFace token, used for gated model/pipeline downloads (e.g. pyannote diarization) |
| `LOCAL_LLM_VISION_MODEL` | Model tag for `services/vision.py`'s perception calls |
| `LOCAL_LLM_DIALOGUE_MODEL` | Model tag for `pipeline/dialogue_service.py`'s response/summary generation |
| `LOCAL_LLM_EMBEDDING_MODEL` | Model tag for `services/lkc_retrieval.py`'s embedding calls |
| `LOCAL_LLM_EXTRA_MODEL` | Additional model tag reserved for other pipeline use |

### kg-agent (shared literature-QA service)

| Variable | Description |
|----------|-------------|
| `KG_AGENT_ENABLED` | Enables `POST /lkc/kg-query` and the hybrid QA fallback |
| `KG_AGENT_BASE_URL` | Base URL of the kg-agent service (on citi-condor / Tailscale) |
| `KG_AGENT_REQUEST_TIMEOUT_SECONDS` | Client timeout for kg-agent requests |
| `KG_AGENT_SOFT_DEADLINE_SECONDS` | Soft deadline before falling back to transcript-only QA |
| `KG_AGENT_FAITHFULNESS_THRESHOLD` | Minimum faithfulness score for a kg-agent answer to be marked `grounded` (default `0.7`, read directly in `api/lkc.py`) |
| `KG_AGENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | Cooldown after a kg-agent failure before retrying |

### Whisper (speech-to-text)

| Variable | Description |
|----------|-------------|
| `WHISPER_MODEL_SIZE` | e.g. `large-v3-turbo` |
| `WHISPER_DEVICE` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | e.g. `int8` |
| `WHISPER_BEAM_SIZE` | Decoding beam size |
| `WHISPER_LANGUAGE` | Forced language, or unset for auto-detect |
| `WHISPER_CPU_THREADS` / `WHISPER_NUM_WORKERS` | CPU inference tuning |

### VAD / Vision / Dialogue / Retrieval

| Variable | Default (read in `api/websockets.py` / `api/sessions.py` / `api/lkc.py`) | Description |
|----------|---------|-------------|
| `VAD_SAMPLE_RATE` | `16000` | Expected input sample rate |
| `VAD_SILENCE_THRESHOLD` | `0.03` | RMS silence cutoff |
| `VAD_SILENCE_CHUNKS` | — | Chunks of silence before a segment is closed |
| `VAD_MAX_SEGMENT_CHUNKS` | — | Max chunks before forcing a segment cut |
| `VISION_FRAME_INTERVAL` | `5` | Analyse every Nth camera frame |
| `VISION_CAMERA_FPS` | `5` | Reported to the frontend via `/config/client` |
| `VISION_CAMERA_QUALITY` | `0.6` | Reported to the frontend via `/config/client` |
| `DIALOGUE_CONTEXT_WINDOW` | `12` | Transcript lines kept for prompt context |
| `DIALOGUE_TTS_AUTO_HIDE_MS` | `8000` | Reported to the frontend via `/config/client` |
| `LKC_RETRIEVAL_TOP_K` | `4` | Chunks retrieved per QA query |

### spaCy (NER)

| Variable | Description |
|----------|-------------|
| `SPACY_MODEL` | Model name (default `en_core_web_sm`)|

### SMTP (this app's own password-reset emails — not Supabase's mail)

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | If unset, Supabase sends reset emails instead |
| `SMTP_PORT` | SMTP port |
| `SMTP_USER` / `SMTP_PASSWORD` | SMTP credentials |
| `SMTP_FROM` | From address |

Secrets should **never** be committed — `.env` is already covered by `.gitignore`.