"""004 — lkc_records table (Supabase-backed LKC event graph).

Replaces the old SQLite lkc_graph.db. Stores the full LKC event log —
transcripts, vision frames, agent replies, and session summaries — as a
unified append-only table in Postgres.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# ---------------------------------------------------------------------------
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ── lkc_records ───────────────────────────────────────────────────────────
    op.create_table(
        "lkc_records",
        sa.Column("id",             sa.BigInteger(),            nullable=False, primary_key=True, autoincrement=True),
        sa.Column("session_id",     sa.Text(),                  nullable=False),
        sa.Column("record_type",    sa.Text(),                  nullable=False),
        sa.Column("timestamp_unix", sa.Double(),                nullable=False),
        sa.Column("timestamp_iso",  sa.Text(),                  nullable=False),
        sa.Column("speaker",        sa.Text(),                  nullable=True),
        sa.Column("text",           sa.Text(),                  nullable=True),
        sa.Column("mode",           sa.Text(),                  nullable=True),
        sa.Column("language",       sa.Text(),                  nullable=True),
        sa.Column("payload",        JSONB(),                    nullable=False),
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    op.create_index("idx_lkc_session",    "lkc_records", ["session_id"])
    op.create_index("idx_lkc_type",       "lkc_records", ["record_type"])
    op.create_index("idx_lkc_ts",         "lkc_records", ["timestamp_unix"])
    op.create_index("idx_lkc_session_ts", "lkc_records", ["session_id", "timestamp_unix"])

    # GIN index on payload for fast JSONB key lookups
    op.create_index(
        "idx_lkc_payload",
        "lkc_records",
        ["payload"],
        postgresql_using="gin",
    )

    # ── Row Level Security ────────────────────────────────────────────────────
    op.execute(sa.text("ALTER TABLE lkc_records ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("""
        DO $$ BEGIN
          DROP POLICY IF EXISTS lkc_records_deny_anon ON lkc_records;
          CREATE POLICY lkc_records_deny_anon
              ON lkc_records FOR ALL TO anon USING (false);
        END $$
    """))


def downgrade() -> None:
    op.execute(sa.text("ALTER TABLE lkc_records DISABLE ROW LEVEL SECURITY"))
    op.drop_index("idx_lkc_payload",    table_name="lkc_records")
    op.drop_index("idx_lkc_session_ts", table_name="lkc_records")
    op.drop_index("idx_lkc_ts",         table_name="lkc_records")
    op.drop_index("idx_lkc_type",       table_name="lkc_records")
    op.drop_index("idx_lkc_session",    table_name="lkc_records")
    op.drop_table("lkc_records")