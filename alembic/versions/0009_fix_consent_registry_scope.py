"""009 — scope consent_registry to (session_id, speaker_label).

Real bug, independent of the tenancy work: consent_registry was keyed
globally by speaker_label alone ("Person A", "Person B") -- these labels
are diarization-assigned and collide across unrelated sessions, so one
session's consent upsert could silently overwrite another session's
consent record for the same generic label.

consent_registry has no FK to sessions today and predates the sessions
truncate in 0007, so existing rows have no reliable session to backfill
against -- cleared out here rather than guessed, consistent with the
option-(c) approach already taken for `sessions`.

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-03
"""

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None
# ---------------------------------------------------------------------------


def upgrade() -> None:
    op.execute(sa.text("TRUNCATE TABLE consent_registry"))

    op.add_column(
        "consent_registry", sa.Column("session_id", sa.Text(), nullable=False)
    )
    op.create_foreign_key(
        "fk_consent_registry_session_id",
        "consent_registry",
        "sessions",
        ["session_id"],
        ["session_id"],
        ondelete="CASCADE",
    )

    # Replace the single-column PK (speaker_label) with a composite one.
    # The `id` surrogate added in 0005 is untouched -- it stays a unique
    # identity column, independent of whatever the PK/ON CONFLICT target is.
    op.drop_constraint("consent_registry_pkey", "consent_registry", type_="primary")
    op.create_primary_key(
        "consent_registry_pkey", "consent_registry", ["session_id", "speaker_label"]
    )


def downgrade() -> None:
    op.drop_constraint("consent_registry_pkey", "consent_registry", type_="primary")
    op.create_primary_key(
        "consent_registry_pkey", "consent_registry", ["speaker_label"]
    )
    op.drop_constraint(
        "fk_consent_registry_session_id", "consent_registry", type_="foreignkey"
    )
    op.drop_column("consent_registry", "session_id")