"""001 — exec_sql helper + pgcrypto + initial schema (all tables).

Revision ID: 0001
Revises:
Create Date: 2026-06-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

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
    op.execute(sa.text("""
        CREATE OR REPLACE FUNCTION exec_sql(sql TEXT)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
          EXECUTE sql;
        END;
        $$
    """))
    op.execute(sa.text("REVOKE ALL ON FUNCTION exec_sql(TEXT) FROM PUBLIC"))

    # ── Extensions ───────────────────────────────────────────────────────────
    op.execute(sa.text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))

    # ── schema_migrations ────────────────────────────────────────────────────
    # Kept for reference; Alembic uses alembic_version instead.
    op.create_table(
        "schema_migrations",
        sa.Column("filename",   sa.Text(),                    nullable=False, primary_key=True),
        sa.Column("applied_at", sa.DateTime(timezone=True),   nullable=False, server_default=sa.text("NOW()")),
    )

    # ── sessions ─────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("session_id",       sa.Text(),                   nullable=False, primary_key=True),
        sa.Column("host_identity",    sa.Text(),                   nullable=False, server_default="browser-user"),
        sa.Column("started_at",       sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("ended_at",         sa.DateTime(timezone=True),  nullable=True),
        sa.Column("livekit_room_sid", sa.Text(),                   nullable=True),
        sa.Column("metadata",         JSONB(),                     nullable=False, server_default=sa.text("'{}'")),
        sa.Column("updated_at",       sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
    )

    # ── transcripts ──────────────────────────────────────────────────────────
    op.create_table(
        "transcripts",
        sa.Column("id",              sa.BigInteger(),             nullable=False, primary_key=True, autoincrement=True),
        sa.Column("session_id",      sa.Text(),                   nullable=False),
        sa.Column("segment_index",   sa.Integer(),                nullable=False, server_default="0"),
        sa.Column("speaker",         sa.Text(),                   nullable=False),
        sa.Column("text",            sa.Text(),                   nullable=False),
        sa.Column("language",        sa.Text(),                   nullable=True),
        sa.Column("mode",            sa.Text(),                   nullable=True),
        sa.Column("timestamp_iso",   sa.Text(),                   nullable=True),
        sa.Column("timestamp_unix",  sa.Double(),                 nullable=False),
        sa.Column("tags",            JSONB(),                     nullable=False, server_default=sa.text("'{}'")),
        sa.Column("word_timestamps", JSONB(),                     nullable=False, server_default=sa.text("'[]'")),
        sa.Column("asr_latency_ms",  sa.Integer(),                nullable=False, server_default="0"),
        sa.Column("e2e_latency_ms",  sa.Integer(),                nullable=False, server_default="0"),
        sa.Column("created_at",      sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_transcripts_session", "transcripts", ["session_id", "timestamp_unix"])

    # ── agent_replies ─────────────────────────────────────────────────────────
    op.create_table(
        "agent_replies",
        sa.Column("id",             sa.BigInteger(),            nullable=False, primary_key=True, autoincrement=True),
        sa.Column("session_id",     sa.Text(),                  nullable=False),
        sa.Column("text",           sa.Text(),                  nullable=False),
        sa.Column("mode",           sa.Text(),                  nullable=True),
        sa.Column("timestamp_iso",  sa.Text(),                  nullable=True),
        sa.Column("timestamp_unix", sa.Double(),                nullable=False),
        sa.Column("grounded",       sa.Boolean(),               nullable=False, server_default="false"),
        sa.Column("lkc_context",    sa.Text(),                  nullable=True),
        sa.Column("created_at",     sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_agent_replies_session", "agent_replies", ["session_id", "timestamp_unix"])

    # ── vision_frames ─────────────────────────────────────────────────────────
    op.create_table(
        "vision_frames",
        sa.Column("id",                sa.BigInteger(),            nullable=False, primary_key=True, autoincrement=True),
        sa.Column("session_id",        sa.Text(),                  nullable=False),
        sa.Column("timestamp_iso",     sa.Text(),                  nullable=True),
        sa.Column("timestamp_unix",    sa.Double(),                nullable=False),
        sa.Column("scene_summary",     sa.Text(),                  nullable=True),
        sa.Column("present_speakers",  JSONB(),                    nullable=False, server_default=sa.text("'[]'")),
        sa.Column("engagement_cues",   JSONB(),                    nullable=False, server_default=sa.text("'{}'")),
        sa.Column("environment_state", JSONB(),                    nullable=False, server_default=sa.text("'{}'")),
        sa.Column("latency_ms",        sa.Integer(),               nullable=False, server_default="0"),
        sa.Column("created_at",        sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_vision_frames_session", "vision_frames", ["session_id", "timestamp_unix"])

    # ── session_summaries ─────────────────────────────────────────────────────
    op.create_table(
        "session_summaries",
        sa.Column("session_id",  sa.Text(),                   nullable=False, primary_key=True),
        sa.Column("summary_md",  sa.Text(),                   nullable=False),
        sa.Column("tags",        JSONB(),                     nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at",  sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at",  sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )

    # ── eval_metrics ──────────────────────────────────────────────────────────
    op.create_table(
        "eval_metrics",
        sa.Column("session_id",  sa.Text(),                   nullable=False, primary_key=True),
        sa.Column("snapshot",    JSONB(),                     nullable=False, server_default=sa.text("'{}'")),
        sa.Column("snapshot_at", sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at",  sa.DateTime(timezone=True),  nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )

    # ── consent_registry ──────────────────────────────────────────────────────
    op.create_table(
        "consent_registry",
        sa.Column("speaker_label", sa.Text(),                  nullable=False, primary_key=True),
        sa.Column("consented",     sa.Boolean(),               nullable=False, server_default="false"),
        sa.Column("real_name",     sa.Text(),                  nullable=True),
        sa.Column("updated_at",    sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.drop_table("consent_registry")
    op.drop_table("eval_metrics")
    op.drop_table("session_summaries")
    op.drop_index("idx_vision_frames_session", table_name="vision_frames")
    op.drop_table("vision_frames")
    op.drop_index("idx_agent_replies_session", table_name="agent_replies")
    op.drop_table("agent_replies")
    op.drop_index("idx_transcripts_session", table_name="transcripts")
    op.drop_table("transcripts")
    op.drop_table("sessions")
    op.drop_table("schema_migrations")
    op.execute(sa.text('DROP EXTENSION IF EXISTS "pgcrypto"'))
    op.execute(sa.text("DROP FUNCTION IF EXISTS exec_sql(TEXT)"))