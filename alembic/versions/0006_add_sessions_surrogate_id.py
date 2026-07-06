"""006 — add surrogate id to sessions (missed by 0005).

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-02
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
    )
    op.create_unique_constraint("uq_sessions_id", "sessions", ["id"])


def downgrade() -> None:
    op.drop_constraint("uq_sessions_id", "sessions", type_="unique")
    op.drop_column("sessions", "id")