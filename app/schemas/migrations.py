"""
app/schemas/migrations.py — Alembic migration-run response model.
"""

from __future__ import annotations
from pydantic import BaseModel


class MigrationRunResponse(BaseModel):
    applied:  list[str]
    skipped:  list[str]
    errors:   list[str]
    dry_run:  bool