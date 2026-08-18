"""
app/api/v1/endpoints/capture.py — Capture, agent summon, NER, retrieval endpoints.

POST   /capture/ingest                — Rifqi Module 2 ingest
GET    /capture/confirmations/{sid}   — poll agent confirmation queue
GET    /capture/tags/{sid}            — session tags summary
GET    /capture/ner_backend           — active NER backend status

GET    /agent/summon/{sid}            — summon status
POST   /agent/summon/{sid}            — manually summon agent
DELETE /agent/summon/{sid}            — clear summon flag

GET    /ner/status                    — alias for NER backend
GET    /retrieval/stats               — embedding retriever stats
"""

import time
import logging

from fastapi import APIRouter

from app.db import lkc_graph
from app.services import capture as _capture
from app.services import lkc_retrieval
from app.schemas.ingest import RifqiSegment

log = logging.getLogger(__name__)

capture_router   = APIRouter(prefix="/capture",  tags=["capture"])
agent_router     = APIRouter(prefix="/agent",    tags=["agent"])
ner_router       = APIRouter(prefix="/ner",      tags=["ner"])
retrieval_router = APIRouter(prefix="/retrieval", tags=["retrieval"])


# ── Rifqi Module 2 ingest ─────────────────────────────────────────────────────

@capture_router.post("/ingest")
async def ingest_from_rifqi(seg: RifqiSegment):
    ts_unix = time.time()
    if seg.timestamp:
        try:
            from dateutil import parser as dtparser
            ts_unix = dtparser.parse(seg.timestamp).timestamp()
        except Exception:
            pass

    record = await _capture.process_segment(
        seg.session_id, seg.speaker, seg.text,
        ts_unix, seg.mode, seg.language,
        confirm_agent=False,
    )
    if seg.source:
        record["source"] = seg.source
    if seg.extra:
        record["extra"] = seg.extra

    log.info(
        f"[capture] Rifqi ingest: session={seg.session_id} speaker={seg.speaker} "
        f"has_tags={_capture.has_tags(record['tags'])}"
    )
    return {"ok": True, "record": record}


@capture_router.get("/confirmations/{session_id}")
async def poll_confirmations(session_id: str):
    return {
        "session_id":    session_id,
        "confirmations": _capture.get_pending_confirmations(session_id),
    }


@capture_router.get("/tags/{session_id}")
async def get_session_tags(session_id: str):
    records = await lkc_graph.read_lkc(session_id=session_id, record_type="transcript")

    action_items: list[str] = []
    decisions:    list[str] = []
    entities:     set[str]  = set()
    deadlines:    list[str] = []

    for r in records:
        tags = r.get("tags", {})
        action_items.extend(tags.get("action_items", []))
        decisions.extend(tags.get("decisions",    []))
        deadlines.extend(tags.get("deadlines",    []))
        entities.update(tags.get("entities",      []))

    return {
        "session_id":   session_id,
        "action_items": action_items,
        "decisions":    decisions,
        "deadlines":    deadlines,
        "entities":     sorted(entities),
    }


@capture_router.get("/ner_backend")
async def ner_backend_status():
    _capture._load_spacy()
    return {
        "backend":         "spacy_en_core_web_sm" if _capture.SPACY_AVAILABLE else "regex_fallback",
        "spacy_available": _capture.SPACY_AVAILABLE,
    }


# ── Agent summon ──────────────────────────────────────────────────────────────

@agent_router.get("/summon/{session_id}")
async def get_summon_status(session_id: str):
    return {"session_id": session_id, "summoned": _capture.is_summoned(session_id)}


@agent_router.post("/summon/{session_id}")
async def manual_summon(session_id: str):
    _capture.force_summon(session_id)
    log.info(f"[agent] Manual summon: {session_id}")
    return {"session_id": session_id, "summoned": True}


@agent_router.delete("/summon/{session_id}")
async def clear_summon(session_id: str):
    _capture.clear_summon(session_id)
    return {"session_id": session_id, "summoned": False}


# ── NER status ────────────────────────────────────────────────────────────────

@ner_router.get("/status")
async def ner_status():
    _capture._load_spacy()
    return {
        "backend":         "spacy_en_core_web_sm" if _capture.SPACY_AVAILABLE else "regex_fallback",
        "spacy_available": _capture.SPACY_AVAILABLE,
    }


# ── Retrieval stats ───────────────────────────────────────────────────────────

@retrieval_router.get("/stats")
async def retrieval_stats():
    return lkc_retrieval.get_retriever().stats()