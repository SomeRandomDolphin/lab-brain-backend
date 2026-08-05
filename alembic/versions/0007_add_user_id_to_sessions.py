"""007 — add user_id to sessions (multi-tenancy foundation).

Backfill decision (confirmed): option (c) — this is still pre-production/
thesis-phase data, so existing sessions are truncated rather than backfilled
with a synthetic owner. TRUNCATE ... CASCADE also clears every dependent
table (transcripts, agent_replies, vision_frames, session_summaries,
eval_metrics, lkc_records) since they all FK to sessions.session_id.

Because of the truncate, user_id can be added as NOT NULL directly — there
are no existing rows left to violate the constraint, so no separate
"add nullable, backfill, then enforce NOT NULL" follow-up migration is
needed the way it would be for a live production table.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-03
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# ---------------------------------------------------------------------------
revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # Wipe dev data (decided: option c) before adding a NOT NULL FK column.
    op.execute(sa.text("TRUNCATE TABLE sessions CASCADE"))

    op.add_column(
        "sessions",
        sa.Column("user_id", UUID(as_uuid=False), nullable=False),
    )
    # auth.users lives in the same Postgres instance (Supabase Auth schema),
    # so a real cross-schema FK is valid here, same as session_participants
    # in migration 0008.
    op.create_foreign_key(
        "fk_sessions_user_id",
        "sessions",
        "users",
        ["user_id"],
        ["id"],
        referent_schema="auth",
        ondelete="CASCADE",
    )
    op.create_index("idx_sessions_user_id", "sessions", ["user_id"])


def downgrade() -> None:
    op.drop_index("idx_sessions_user_id", table_name="sessions")
    op.drop_constraint("fk_sessions_user_id", "sessions", type_="foreignkey")
    op.drop_column("sessions", "user_id")