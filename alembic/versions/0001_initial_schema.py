"""001 — exec_sql helper + pgcrypto + initial schema (all tables).

Revision ID: 0001
Revises:
Create Date: 2026-06-25
"""

from alembic import op

# ---------------------------------------------------------------------------
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ── exec_sql RPC helper ──────────────────────────────────────────────────
    # Previously applied manually in the Supabase SQL editor.
    # Now managed by Alembic — safe to run via the migration runner.
    op.execute("""
        CREATE OR REPLACE FUNCTION exec_sql(sql TEXT)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
          EXECUTE sql;
        END;
        $$
    """)
    op.execute("REVOKE ALL ON FUNCTION exec_sql(TEXT) FROM PUBLIC")

    # ── Extensions ───────────────────────────────────────────────────────────
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # ── schema_migrations ────────────────────────────────────────────────────
    # Kept for reference; Alembic uses alembic_version instead.
    op.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename    TEXT        PRIMARY KEY,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── sessions ─────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id        TEXT        PRIMARY KEY,
            host_identity     TEXT        NOT NULL DEFAULT 'browser-user',
            started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at          TIMESTAMPTZ,
            livekit_room_sid  TEXT,
            metadata          JSONB       NOT NULL DEFAULT '{}',
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── transcripts ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id               BIGSERIAL        PRIMARY KEY,
            session_id       TEXT             NOT NULL
                                 REFERENCES sessions(session_id) ON DELETE CASCADE,
            segment_index    INT              NOT NULL DEFAULT 0,
            speaker          TEXT             NOT NULL,
            text             TEXT             NOT NULL,
            language         TEXT,
            mode             TEXT,
            timestamp_iso    TEXT,
            timestamp_unix   DOUBLE PRECISION NOT NULL,
            tags             JSONB            NOT NULL DEFAULT '{}',
            word_timestamps  JSONB            NOT NULL DEFAULT '[]',
            asr_latency_ms   INT              NOT NULL DEFAULT 0,
            e2e_latency_ms   INT              NOT NULL DEFAULT 0,
            created_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_transcripts_session
            ON transcripts (session_id, timestamp_unix)
    """)

    # ── agent_replies ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_replies (
            id              BIGSERIAL        PRIMARY KEY,
            session_id      TEXT             NOT NULL
                                REFERENCES sessions(session_id) ON DELETE CASCADE,
            text            TEXT             NOT NULL,
            mode            TEXT,
            timestamp_iso   TEXT,
            timestamp_unix  DOUBLE PRECISION NOT NULL,
            grounded        BOOLEAN          NOT NULL DEFAULT FALSE,
            lkc_context     TEXT,
            created_at      TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_agent_replies_session
            ON agent_replies (session_id, timestamp_unix)
    """)

    # ── vision_frames ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS vision_frames (
            id                BIGSERIAL        PRIMARY KEY,
            session_id        TEXT             NOT NULL
                                  REFERENCES sessions(session_id) ON DELETE CASCADE,
            timestamp_iso     TEXT,
            timestamp_unix    DOUBLE PRECISION NOT NULL,
            scene_summary     TEXT,
            present_speakers  JSONB            NOT NULL DEFAULT '[]',
            engagement_cues   JSONB            NOT NULL DEFAULT '{}',
            environment_state JSONB            NOT NULL DEFAULT '{}',
            latency_ms        INT              NOT NULL DEFAULT 0,
            created_at        TIMESTAMPTZ      NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_vision_frames_session
            ON vision_frames (session_id, timestamp_unix)
    """)

    # ── session_summaries ─────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id   TEXT        PRIMARY KEY
                             REFERENCES sessions(session_id) ON DELETE CASCADE,
            summary_md   TEXT        NOT NULL,
            tags         JSONB       NOT NULL DEFAULT '{}',
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── eval_metrics ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_metrics (
            session_id   TEXT        PRIMARY KEY
                             REFERENCES sessions(session_id) ON DELETE CASCADE,
            snapshot     JSONB       NOT NULL DEFAULT '{}',
            snapshot_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── consent_registry ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS consent_registry (
            speaker_label  TEXT        PRIMARY KEY,
            consented      BOOLEAN     NOT NULL DEFAULT FALSE,
            real_name      TEXT,
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.execute("DROP TABLE IF EXISTS consent_registry")
    op.execute("DROP TABLE IF EXISTS eval_metrics")
    op.execute("DROP TABLE IF EXISTS session_summaries")
    op.execute("DROP TABLE IF EXISTS vision_frames")
    op.execute("DROP TABLE IF EXISTS agent_replies")
    op.execute("DROP TABLE IF EXISTS transcripts")
    op.execute("DROP TABLE IF EXISTS sessions")
    op.execute("DROP TABLE IF EXISTS schema_migrations")
    op.execute('DROP EXTENSION IF EXISTS "pgcrypto"')
    op.execute("DROP FUNCTION IF EXISTS exec_sql(TEXT)")