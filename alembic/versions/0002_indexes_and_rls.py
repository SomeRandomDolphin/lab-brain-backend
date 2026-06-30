"""002 — extra indexes, updated_at triggers, RLS policies.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-25
"""

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------

_RLS_TABLES = (
    "sessions",
    "transcripts",
    "agent_replies",
    "vision_frames",
    "session_summaries",
    "eval_metrics",
    "consent_registry",
)


def upgrade() -> None:
    # ── Additional composite indexes ─────────────────────────────────────────
    op.create_index(
        "idx_transcripts_speaker",
        "transcripts",
        ["session_id", "speaker"],
    )
    op.create_index(
        "idx_transcripts_text_fts",
        "transcripts",
        [sa.text("to_tsvector('english', text)")],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_transcripts_tags",
        "transcripts",
        ["tags"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_vision_frames_latest",
        "vision_frames",
        ["session_id", sa.text("timestamp_unix DESC")],
    )
    op.create_index(
        "idx_sessions_host",
        "sessions",
        ["host_identity", sa.text("started_at DESC")],
    )

    # ── updated_at trigger function ───────────────────────────────────────────
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$
    """))

    # ── Attach triggers (idempotent via DO block) ─────────────────────────────
    op.execute(sa.text("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger WHERE tgname = 'sessions_set_updated_at'
          ) THEN
            CREATE TRIGGER sessions_set_updated_at
              BEFORE UPDATE ON sessions
              FOR EACH ROW EXECUTE FUNCTION set_updated_at();
          END IF;
        END $$
    """))
    op.execute(sa.text("""
        DO $$ BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_trigger WHERE tgname = 'session_summaries_set_updated_at'
          ) THEN
            CREATE TRIGGER session_summaries_set_updated_at
              BEFORE UPDATE ON session_summaries
              FOR EACH ROW EXECUTE FUNCTION set_updated_at();
          END IF;
        END $$
    """))

    # ── Row Level Security ────────────────────────────────────────────────────
    for table in _RLS_TABLES:
        op.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))

    # Deny anon key access to every table; service role bypasses RLS.
    op.execute(sa.text("""
        DO $$ BEGIN
          DROP POLICY IF EXISTS sessions_deny_anon          ON sessions;
          DROP POLICY IF EXISTS transcripts_deny_anon       ON transcripts;
          DROP POLICY IF EXISTS agent_replies_deny_anon     ON agent_replies;
          DROP POLICY IF EXISTS vision_frames_deny_anon     ON vision_frames;
          DROP POLICY IF EXISTS session_summaries_deny_anon ON session_summaries;
          DROP POLICY IF EXISTS eval_metrics_deny_anon      ON eval_metrics;
          DROP POLICY IF EXISTS consent_registry_deny_anon  ON consent_registry;

          CREATE POLICY sessions_deny_anon
              ON sessions FOR ALL TO anon USING (false);
          CREATE POLICY transcripts_deny_anon
              ON transcripts FOR ALL TO anon USING (false);
          CREATE POLICY agent_replies_deny_anon
              ON agent_replies FOR ALL TO anon USING (false);
          CREATE POLICY vision_frames_deny_anon
              ON vision_frames FOR ALL TO anon USING (false);
          CREATE POLICY session_summaries_deny_anon
              ON session_summaries FOR ALL TO anon USING (false);
          CREATE POLICY eval_metrics_deny_anon
              ON eval_metrics FOR ALL TO anon USING (false);
          CREATE POLICY consent_registry_deny_anon
              ON consent_registry FOR ALL TO anon USING (false);
        END $$
    """))


def downgrade() -> None:
    # Disable RLS
    for table in _RLS_TABLES:
        op.execute(sa.text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))

    # Drop triggers
    op.execute(sa.text("DROP TRIGGER IF EXISTS sessions_set_updated_at ON sessions"))
    op.execute(sa.text("DROP TRIGGER IF EXISTS session_summaries_set_updated_at ON session_summaries"))
    op.execute(sa.text("DROP FUNCTION IF EXISTS set_updated_at()"))

    # Drop extra indexes
    op.drop_index("idx_sessions_host",          table_name="sessions")
    op.drop_index("idx_vision_frames_latest",   table_name="vision_frames")
    op.drop_index("idx_transcripts_tags",       table_name="transcripts")
    op.drop_index("idx_transcripts_text_fts",   table_name="transcripts")
    op.drop_index("idx_transcripts_speaker",    table_name="transcripts")