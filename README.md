# Lab Brain — Module 5 Multimodal Conversational Agent

Refactored FastAPI backend with proper project structure, persistent SQLite LKC graph,
LiveKit WebRTC media layer, SSE event streaming, Supabase persistence with
Alembic-managed schema migrations, and Supabase Auth for identity management.

---

## Project Structure

```
lab_brain/
├── main.py                          # Entry point (uvicorn target)
├── config.json                      # Runtime config (no secrets)
├── requirements.txt
│
├── app/
│   ├── main.py                      # FastAPI app factory + startup hooks
│   │
│   ├── core/
│   │   ├── env.py                   # Loads .env into os.environ (load_env()) — runs first in app/main.py
│   │   ├── config.py                # Typed dataclass config + env-var overrides
│   │   └── logging.py               # Logging setup
│   │
│   ├── db/
│   │   ├── lkc_graph.py             # SQLite-backed LKC graph (WAL, indexed)
│   │   ├── supabase_client.py       # Supabase client + read/write API (runtime data access only)
│   │   ├── migrations.py            # Alembic-based migration runner (run_migrations, get_migration_status)
│   │   └── supabase_auth.py         # Supabase Auth layer (replaces SQLite auth.db)
│   │
│   ├── schemas/
│   │   ├── __init__.py              # Pydantic v2 request/response models
│   │   └── auth.py                  # Auth request/response models (RegisterRequest, AuthResponse, …)
│   │
│   ├── services/
│   │   ├── privacy.py               # Consent registry + PII redaction
│   │   ├── capture.py               # NER tagging, summon system, segment processing
│   │   ├── vision.py                # VLM perception layer (local LLM via OpenAI compat)
│   │   ├── lkc_retrieval.py         # Dense retrieval (mpnet → MiniLM → TF-IDF)
│   │   ├── eval_metrics.py          # WER, latency, capture quality, privacy metrics
│   │   └── auth_service.py          # Optional custom SMTP override for reset emails
│   │
│   ├── pipeline/
│   │   ├── asr.py                   # WhisperX / faster-whisper + VAD chunker
│   │   ├── dialogue_service.py      # Mode FSM, speaker diarization, LLM response/summary
│   │   ├── livekit_rooms.py         # Token generation, room management, SSE bus
│   │   └── session_pipeline.py      # Audio+video → ASR → NER → Dialogue → SSE → Supabase
│   │
│   └── api/
│       ├── deps.py                  # FastAPI dependency: get_current_user, get_optional_user
│       └── v1/
│           ├── router.py            # Aggregates all endpoint routers
│           └── endpoints/
│               ├── auth.py          # POST /auth/register|login|logout|refresh, GET /auth/me, …
│               ├── livekit.py       # POST/GET/DELETE /livekit/room, GET /events/{sid}
│               ├── privacy.py       # GET/POST/DELETE /privacy/consent
│               ├── lkc.py           # GET/DELETE /lkc, /lkc/sessions
│               ├── capture.py       # /capture/ingest, /agent/summon, /ner, /retrieval
│               ├── sessions.py      # /summary, /mode, /perception, /metrics, /config/client
│               ├── supabase.py      # /supabase/sessions, /supabase/migrations
│               └── websockets.py    # Legacy /ws/asr, /ws/vision, /ws/tts (backward compat)
│
├── alembic.ini                      # Alembic config (DB URL set at runtime from SUPABASE_DB_URL)
└── alembic/
    ├── env.py                       # Async Alembic environment (asyncpg)
    ├── script.py.mako               # Template used by `alembic revision`
    └── versions/                    # Numbered revision scripts, e.g. 0001_initial_schema.py
```

> **Note:** the old `migrations/versions/*.sql` files and the custom `exec_sql`-based
> runner have been removed. Schema changes are now plain Alembic revision scripts
> tracked in Postgres' own `alembic_version` table — see [Migration System](#migration-system).

> **Note:** `auth.db` (the old SQLite auth database) is no longer created or used.
> It can be safely deleted from any existing deployment.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Start LiveKit SFU (local dev)

```bash
docker run --rm -p 7880:7880 livekit/livekit-server --dev
```

### 3. Configure Supabase

Supabase is now required for data persistence, authentication, **and** schema migrations.

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```env
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>   # used server-side only — keep secret
SUPABASE_ANON_KEY=<anon-key>              # safe to expose to the frontend

# Postgres connection string, used by Alembic (app/db/migrations.py) to run migrations.
# Find it under Project Settings → Database → Connection string ("URI" tab) in the
# Supabase dashboard, then swap the postgresql:// prefix for postgresql+asyncpg://.
SUPABASE_DB_URL=postgresql+asyncpg://postgres:<password>@db.<ref>.supabase.co:5432/postgres
```

`.env` is loaded automatically on startup (see [Environment File](#environment-file-env)
below) — no need to `export` these manually for local dev. In production/Docker, set
real environment variables instead; they always take precedence over `.env`.

Migrations are applied automatically on startup via Alembic — see
[Migration System](#migration-system) below. There's no manual SQL-editor step anymore.

In the Supabase dashboard, make sure **Authentication → Email Templates → Reset Password**
is configured. Supabase handles reset email delivery by default; see
[Custom SMTP override](#custom-smtp-override) below if you need to send from your own server.

### 4. Start the server

```bash
python main.py
# or with auto-reload for development:
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Authentication

Auth is now fully delegated to **Supabase Auth** (GoTrue). The old SQLite `auth.db`
and `auth_store.py` have been removed.

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
app will use `auth_service.send_reset_email()` instead of Supabase's delivery:

```bash
export SMTP_HOST=smtp.example.com
export SMTP_PORT=587
export SMTP_USER=apikey
export SMTP_PASSWORD=<sendgrid-or-ses-key>
export SMTP_FROM=noreply@yourdomain.com
```

---

## Migration System

Schema migrations are managed by **Alembic** (replacing the old custom `exec_sql`
SQL-file runner). Revisions live in `alembic/versions/` as Python scripts with
`upgrade()` / `downgrade()` functions, and applied state is tracked in Postgres'
own `alembic_version` table (the old custom `schema_migrations` table is no longer used).

**Automatic on startup:** `app/main.py` calls `app.db.migrations.run_migrations()`
during FastAPI startup, which runs `alembic upgrade head`. Already-applied
revisions are skipped — safe to run on every restart. If `SUPABASE_DB_URL` isn't
set, the app logs a warning and continues rather than failing to boot.

**Manual via API:**

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

> **Status:** `alembic/versions/` currently ships **empty**. The original
> `migrations/versions/*.sql` files (`000_exec_sql_helper.sql` through
> `003_reporting_views.sql`) were not carried forward as Alembic revisions —
> they need to be hand-converted into real revision scripts before this will
> create any schema. Until then, `alembic upgrade head` is a no-op against a
> database with no tables.

---

## API Overview

### Auth

| Method | Path | Auth required | Description |
|--------|------|:---:|-------------|
| POST | `/auth/register` | — | Create account; returns user + access + refresh tokens |
| POST | `/auth/login` | — | Sign in; returns user + access + refresh tokens |
| POST | `/auth/logout` | ✓ | Revoke current session |
| POST | `/auth/refresh` | — | Exchange refresh token for new access + refresh tokens |
| GET | `/auth/me` | ✓ | Return current user (used to rehydrate session on page load) |
| POST | `/auth/forgot-password` | — | Trigger Supabase password-reset email |
| POST | `/auth/reset-password` | — | Consume reset token, set new password, revoke all sessions |

### LiveKit / WebRTC

| Method | Path | Description |
|--------|------|-------------|
| POST | `/livekit/room` | Create room, start pipeline, return session_id + JWT |
| GET | `/livekit/token?session_id=&identity=` | Re-issue JWT for guest join |
| GET | `/livekit/room/{sid}` | Room status + participant count |
| DELETE | `/livekit/room/{sid}` | End session, snapshot metrics to Supabase |
| GET | `/events/{sid}` | SSE stream (transcript, agent_reply, perception, speak, …) |

### LKC Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/lkc` | HTML viewer of recent records |
| GET | `/lkc/stats` | Graph statistics |
| GET | `/lkc/sessions` | List all sessions |
| GET | `/lkc/sessions/{sid}` | Session records (filterable by type/time) |
| DELETE | `/lkc` | Wipe entire graph |
| DELETE | `/lkc/sessions/{sid}` | Wipe one session |

### Session

| Method | Path | Description |
|--------|------|-------------|
| POST | `/summary/{sid}` | Generate LLM summary, upload report to Supabase Storage |
| GET | `/mode/{sid}` | Current dialogue mode + summon flag |
| GET | `/perception/{sid}` | Latest vision state |
| GET | `/metrics` | All session metric summaries |
| GET | `/metrics/csv` | CSV export |
| POST | `/eval/wer` | Compute WER for a reference/hypothesis pair |
| GET | `/config/client` | Frontend config (camera fps, tts_auto_hide_ms, lk_url) |

### Capture / Agent

| Method | Path | Description |
|--------|------|-------------|
| POST | `/capture/ingest` | Rifqi Module 2 ingest |
| GET | `/capture/tags/{sid}` | Session tags summary |
| GET | `/agent/summon/{sid}` | Summon status |
| POST | `/agent/summon/{sid}` | Manually summon agent |
| DELETE | `/agent/summon/{sid}` | Clear summon |

### Privacy

| Method | Path | Description |
|--------|------|-------------|
| GET | `/privacy/status` | Registry overview |
| POST | `/privacy/consent` | Register/update consent (local only) |
| POST | `/privacy/consent/sync` | Dual-write: local + Supabase |
| DELETE | `/privacy/consent/{speaker}` | Revoke consent |

### Supabase

| Method | Path | Description |
|--------|------|-------------|
| GET | `/supabase/status` | Connectivity check |
| GET | `/supabase/sessions` | Recent sessions from Supabase |
| GET | `/supabase/sessions/{sid}/transcripts` | Transcript rows |
| GET | `/supabase/sessions/{sid}/summary` | Persisted summary |
| GET | `/supabase/sessions/{sid}/report` | Report Storage URL |
| POST | `/supabase/migrations/run` | Apply pending Alembic migrations (`alembic upgrade head`) |
| GET | `/supabase/migrations/status` | Current vs. head revision + pending list (no changes applied) |

### Legacy WebSocket (backward compat)

| Path | Description |
|------|-------------|
| `/ws/asr?session_id=` | Raw PCM → transcript + agent_reply |
| `/ws/vision?session_id=` | JPEG bytes → perception |
| `/ws/tts?session_id=` | TTS speak events |

---

## Environment File (`.env`)

`.env` files are now loaded automatically — no extra setup required.

```bash
cp .env.example .env
# fill in your values, then just start the app normally:
python main.py
```

`app/core/env.py` (`load_env()`) parses `.env` from the project root and applies its
`KEY=VALUE` pairs to `os.environ` as the very first thing `app/main.py` does — before
`app.core.config`, `app.db.supabase_client`, or `app.db.migrations` get a chance to read
any environment variables. It's a small, dependency-free parser (no `python-dotenv`
required) that supports:

- Blank lines and full-line `#` comments
- Trailing inline comments (`KEY=value # like this`)
- Single- or double-quoted values (`KEY="some value"`)
- An optional leading `export ` per line (shell-sourceable `.env` files work too)

**Precedence:** real environment variables always win. If a variable is already set
(exported in your shell, injected by Docker, Render, Railway, etc.), the value in `.env`
is ignored for that key — `.env` only fills in whatever isn't already set. This means
the same `.env` file is safe to keep around in every environment: it's a local-dev
convenience, not a way to override production config.

A missing `.env` file is not an error — it just means "nothing to load," which is the
normal case in production where real env vars are set directly.

> **Note:** `load_env()` is called at the top of `app/main.py`, before `app.core.config`
> is imported. That's early enough as long as the top-level `main.py` (the uvicorn
> entry point) does `from app.main import app` without importing anything else that
> reads `os.environ` at import time first. If you add such an import to the top-level
> `main.py`, move the `load_env()` call there instead, as the very first line.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | **Yes** | Supabase project URL (REST/Storage API) |
| `SUPABASE_SERVICE_KEY` | **Yes** | Service-role secret key (server-side only) |
| `SUPABASE_ANON_KEY` | Recommended | Anon/public key (used for user-facing auth flows) |
| `SUPABASE_DB_URL` | **Yes**, for migrations | Postgres connection string used by Alembic, e.g. `postgresql+asyncpg://postgres:<pw>@db.<ref>.supabase.co:5432/postgres`. Without it, startup migrations are skipped with a warning. |
| `FRONTEND_URL` | No | Base URL for password-reset links (default: `http://localhost:5173`) |
| `HF_TOKEN` | No | HuggingFace token for pyannote diarization |
| `SMTP_HOST` | No | Custom SMTP host; if unset, Supabase sends reset emails |
| `SMTP_PORT` | No | SMTP port (default: `587`) |
| `SMTP_USER` | No | SMTP username |
| `SMTP_PASSWORD` | No | SMTP password |
| `SMTP_FROM` | No | From address (default: `noreply@labbrain.local`) |

Secrets should **never** be placed in `config.json` — use environment variables only.