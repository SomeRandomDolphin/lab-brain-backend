"""
app/api/v1/endpoints/sessions.py — Session-level endpoints.

POST   /summary/{sid}         — generate + persist LLM summary (owner or participant)
GET    /mode/{sid}            — current dialogue mode + summon flag (owner or participant)
GET    /perception/{sid}      — latest vision perception state (owner or participant)
GET    /config/client         — frontend config (camera fps, tts hide ms, lk_url)
GET    /metrics               — session metric summaries, scoped to the caller's own sessions
GET    /metrics/csv           — CSV export of raw metric samples, same scoping
POST   /eval/wer              — compute WER for a reference/hypothesis pair (owner or participant)
"""

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.deps import get_current_user, require_session_access
from app.core.config import cfg
from app.db import lkc_graph, supabase_client
from app.services import vision, eval_metrics
from app.services.capture import is_summoned
from app.schemas import WerRequest

# Deferred imports to avoid circular at module load
def _get_dialogue_module():
    from app.pipeline import dialogue_service
    return dialogue_service

router = APIRouter(tags=["sessions"])


@router.post("/summary/{session_id}")
async def post_summary(session_id: str = Depends(require_session_access)):
    import logging
    _log = logging.getLogger(__name__)
    try:
        dialogue = _get_dialogue_module()
        dlg_state = dialogue.get_dialogue(session_id)
        records   = await lkc_graph.read_lkc(session_id=session_id, record_type="transcript")

        tags: dict = {"action_items": [], "decisions": [], "deadlines": [], "entities": []}
        for r in records:
            t = r.get("tags", {})
            tags["action_items"].extend(t.get("action_items", []))
            tags["decisions"].extend(t.get("decisions",    []))
            tags["deadlines"].extend(t.get("deadlines",    []))
            tags["entities"].extend(t.get("entities",      []))
        tags["entities"] = sorted(set(tags["entities"]))

        try:
            summary_md = await dialogue.generate_summary(dlg_state, tags)
        except Exception as exc:
            # LLM unavailable (e.g. Ollama not running) — return a graceful
            # stub so the frontend session teardown can still complete cleanly.
            _log.warning(f"[summary:{session_id}] generate_summary failed (LLM unavailable?): {exc}")
            summary_md = (
                "## Session Summary\n\n"
                "_Summary generation failed — the local LLM may not be running._\n\n"
                f"**Action items captured:** {len(tags.get('action_items', []))}\n"
                f"**Decisions captured:** {len(tags.get('decisions', []))}\n"
                f"**Entities mentioned:** {', '.join(tags.get('entities', [])) or 'none'}\n"
            )

        await lkc_graph.write_to_lkc({
            "type":          "session_summary",
            "session_id":    session_id,
            "timestamp_iso": datetime.utcnow().isoformat() + "Z",
            "summary":       summary_md,
            "tags":          tags,
        })

        await supabase_client.upsert_session_summary(session_id, summary_md, tags)
        transcript_rows = await supabase_client.get_transcripts(session_id) or [
            {"timestamp_iso": r.get("timestamp_iso", ""),
             "speaker": r.get("speaker", ""),
             "text": r.get("text", "")}
            for r in records
        ]
        report_url = await supabase_client.export_report(session_id, summary_md, tags, transcript_rows)

        return {
            "session_id": session_id,
            "summary":    summary_md,
            "report_url": report_url,
        }
    except Exception as exc:
        _log.error(f"[summary:{session_id}] post_summary failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=200,  # Return 200 so frontend teardown is not blocked
            content={
                "session_id": session_id,
                "summary":    "_Summary unavailable — an error occurred during generation._",
                "report_url": None,
                "error":      str(exc),
            }
        )


@router.get("/mode/{session_id}")
async def get_mode(session_id: str = Depends(require_session_access)):
    dialogue  = _get_dialogue_module()
    dlg_state = dialogue.get_dialogue(session_id)
    return {
        "session_id": session_id,
        "mode":       dlg_state.mode.value,
        "summoned":   is_summoned(session_id),
    }


@router.get("/perception/{session_id}")
async def get_perception(session_id: str = Depends(require_session_access)):
    state = vision.get_state(session_id)
    return {
        "session_id":        session_id,
        "present_speakers":  state.present_speakers,
        "engagement_cues":   state.engagement_cues,
        "scene_summary":     state.scene_summary,
        "environment_state": state.environment_state,
        "last_updated":      state.last_updated,
        "frame_count":       state.frame_count,
    }


@router.get("/config/client")
async def client_config():
    return {
        "camera_fps":       cfg.vision.camera_fps,
        "camera_quality":   cfg.vision.camera_quality,
        "tts_auto_hide_ms": cfg.dialogue.tts_auto_hide_ms,
        "lk_url":           cfg.livekit.public_url,
    }


@router.get("/metrics")
async def get_all_metrics(current_user: dict = Depends(get_current_user)):
    """
    eval_metrics keeps its per-session snapshots in-memory, keyed only by
    session_id — it has no notion of ownership. Scoping happens here: fetch
    the caller's own (owned + participant) session ids from the DB and
    filter the in-memory summaries down to just those.
    """
    accessible = {
        s["session_id"]
        for s in await supabase_client.get_sessions(current_user["id"], limit=10_000)
    }
    summaries = await eval_metrics.all_summaries()
    scoped = [s for s in summaries if s.get("session_id") in accessible]
    return JSONResponse(scoped)


@router.get("/metrics/csv", response_class=PlainTextResponse)
async def get_metrics_csv(current_user: dict = Depends(get_current_user)):
    accessible = {
        s["session_id"]
        for s in await supabase_client.get_sessions(current_user["id"], limit=10_000)
    }
    raw_csv = await eval_metrics.all_csv()

    reader = csv.DictReader(io.StringIO(raw_csv))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["session_id", "metric", "value"])
    writer.writeheader()
    for row in reader:
        if row.get("session_id") in accessible:
            writer.writerow(row)

    return PlainTextResponse(buf.getvalue(), media_type="text/csv")


@router.post("/eval/wer")
async def evaluate_wer(req: WerRequest, current_user: dict = Depends(get_current_user)):
    # session_id arrives in the request body here, not as a path param, so it
    # can't go through Depends(require_session_access) the usual way — same
    # ownership/participant check, applied manually.
    owner, participants = await supabase_client.get_session_access(req.session_id)
    if owner is None or (current_user["id"] != owner and current_user["id"] not in participants):
        raise HTTPException(status_code=404, detail="Session not found.")

    # Bonus fix: get_metrics() is a synchronous function in eval_metrics.py
    # (not `async def`) — the previous `await eval_metrics.get_metrics(...)`
    # would have raised a TypeError at runtime on every call to this route.
    metrics = eval_metrics.get_metrics(req.session_id)
    wer     = metrics.record_wer(req.reference, req.hypothesis)
    return {"session_id": req.session_id, "wer": round(wer, 4)}