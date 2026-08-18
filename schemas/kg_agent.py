"""
app/schemas/kg_agent.py — kg-agent literature Q&A request & response models.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class KgQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class KgDocumentUsed(BaseModel):
    name:   str
    chunks: int


class KgQueryResponse(BaseModel):
    """
    Native kg-agent fields, adapted (not the old LKC response shape — see
    the migration notes in app/api/v1/endpoints/lkc.py). `passed` from the
    upstream service is intentionally NOT exposed: it's a constant-false
    data artifact on this deployment, not a quality signal. Use `grounded`
    instead, which this endpoint derives from faithfulness + document
    coverage the same way the live QA hybrid path does.
    """
    answer:                     str
    grounded:                   bool
    faithfulness:                float
    overall_confidence:          float
    temporal_validity_status:    str
    documents_used:              list[KgDocumentUsed]
    disclaimer:                  Optional[str] = None
    strategy:                    Optional[str] = None