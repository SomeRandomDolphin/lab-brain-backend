"""008 — session_participants table.

Confirmed decision: yes, add this table. `sessions.user_id` (0007) tracks
who *created* a session; this table tracks who has *joined* it via
GET /livekit/token, since login is now required to join too and a joiner
other than the host needs a real access record (owner OR participant),
not just owner-only.

A row is inserted every time GET /livekit/token succeeds for a session
(see app/api/deps.py: require_session_access).

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-03
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# ---------------------------------------------------------------------------
revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    op.create_table(
        "session_participants",
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("user_id", UUID(as_uuid=False), nullable=False),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.session_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["auth.users.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("session_id", "user_id"),
    )
    op.create_index(
        "idx_session_participants_user", "session_participants", ["user_id"]
    )

    op.execute(sa.text("ALTER TABLE session_participants ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("""
        DO $$ BEGIN
          DROP POLICY IF EXISTS session_participants_deny_anon ON session_participants;
          CREATE POLICY session_participants_deny_anon
              ON session_participants FOR ALL TO anon USING (false);
        END $$
    """))


def downgrade() -> None:
    op.execute(sa.text("ALTER TABLE session_participants DISABLE ROW LEVEL SECURITY"))
    op.drop_index("idx_session_participants_user", table_name="session_participants")
    op.drop_table("session_participants")