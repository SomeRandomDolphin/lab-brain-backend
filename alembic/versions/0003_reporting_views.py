"""003 — reporting views (session_overview, action_items_all, decisions_all, latency_stats).

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-25
"""

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ── session_overview ──────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE OR REPLACE VIEW session_overview AS
        SELECT
            s.session_id,
            s.host_identity,
            s.started_at,
            s.ended_at,
            EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at))::INT
                AS duration_s,
            COUNT(DISTINCT t.id)   AS transcript_count,
            COUNT(DISTINCT ar.id)  AS agent_reply_count,
            COUNT(DISTINCT vf.id)  AS vision_frame_count,
            ss.summary_md IS NOT NULL AS has_summary
        FROM sessions s
        LEFT JOIN transcripts       t  ON t.session_id  = s.session_id
        LEFT JOIN agent_replies     ar ON ar.session_id = s.session_id
        LEFT JOIN vision_frames     vf ON vf.session_id = s.session_id
        LEFT JOIN session_summaries ss ON ss.session_id = s.session_id
        GROUP BY
            s.session_id, s.host_identity, s.started_at, s.ended_at, ss.summary_md
    """))

    # ── action_items_all ──────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE OR REPLACE VIEW action_items_all AS
        SELECT
            session_id,
            timestamp_iso,
            speaker,
            jsonb_array_elements_text(tags->'action_items') AS action_item
        FROM transcripts
        WHERE tags ? 'action_items'
          AND jsonb_array_length(tags->'action_items') > 0
    """))

    # ── decisions_all ─────────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE OR REPLACE VIEW decisions_all AS
        SELECT
            session_id,
            timestamp_iso,
            speaker,
            jsonb_array_elements_text(tags->'decisions') AS decision
        FROM transcripts
        WHERE tags ? 'decisions'
          AND jsonb_array_length(tags->'decisions') > 0
    """))

    # ── latency_stats ─────────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE OR REPLACE VIEW latency_stats AS
        SELECT
            session_id,
            COUNT(*)                                                           AS segment_count,
            ROUND(AVG(asr_latency_ms))                                        AS avg_asr_ms,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY asr_latency_ms)     AS p50_asr_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY asr_latency_ms)     AS p95_asr_ms,
            ROUND(AVG(e2e_latency_ms))                                        AS avg_e2e_ms,
            PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY e2e_latency_ms)     AS p50_e2e_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e2e_latency_ms)     AS p95_e2e_ms
        FROM transcripts
        GROUP BY session_id
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP VIEW IF EXISTS latency_stats"))
    op.execute(sa.text("DROP VIEW IF EXISTS decisions_all"))
    op.execute(sa.text("DROP VIEW IF EXISTS action_items_all"))
    op.execute(sa.text("DROP VIEW IF EXISTS session_overview"))