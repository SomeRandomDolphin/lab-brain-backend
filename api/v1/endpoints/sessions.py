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

from api.deps import get_current_user, require_session_access
from db import lkc_graph, supabase_client
from services import vision, eval_metrics
from services.capture import is_summoned
from schemas.eval import WerRequest

# Deferred imports to avoid circular at module load
def _get_dialogue_module():
    from pipeline import dialogue_service
    return dialogue_service

router = APIRouter(tags=["sessions"])

import os
DIALOGUE_CONTEXT_WINDOW   = int(os.environ.get("DIALOGUE_CONTEXT_WINDOW", "12"))
DIALOGUE_TTS_AUTO_HIDE_MS = int(os.environ.get("DIALOGUE_TTS_AUTO_HIDE_MS", "8000"))
VISION_CAMERA_FPS         = int(os.environ.get("VISION_CAMERA_FPS", "5"))
VISION_CAMERA_QUALITY     = float(os.environ.get("VISION_CAMERA_QUALITY", "0.6"))
LIVEKIT_PUBLIC_URL = os.environ.get("LIVEKIT_PUBLIC_URL") or os.environ.get(
    "LIVEKIT_URL", "ws://host.docker.internal:7880"
)


@router.post("/summary/{session_id}")
async def post_summary(session_id: str = Depends(require_session_access)):
    import logging
    _log = logging.getLogger(__name__)
    try:
        dialogue = _get_dialogue_module()
        records  = await lkc_graph.read_lkc(session_id=session_id, record_type="transcript")

        # Chronological order isn't guaranteed by read_lkc — sort defensively
        # so the summary prompt reads as an actual conversation rather than
        # whatever order the records happen to come back in. Records without
        # a timestamp (shouldn't normally happen) sort first rather than
        # raising on a missing key.
        records = sorted(records, key=lambda r: r.get("timestamp_unix", 0))

        tags: dict = {"action_items": [], "decisions": [], "deadlines": [], "entities": []}
        for r in records:
            t = r.get("tags", {})
            tags["action_items"].extend(t.get("action_items", []))
            tags["decisions"].extend(t.get("decisions",    []))
            tags["deadlines"].extend(t.get("deadlines",    []))
            tags["entities"].extend(t.get("entities",      []))
        tags["entities"] = sorted(set(tags["entities"]))

        # Built from the same durable `records` as `tags` above, NOT from
        # dialogue.get_dialogue(session_id)'s in-memory transcript_context.
        # DELETE /livekit/room/{session_id} calls clear_dialogue(session_id)
        # as part of normal teardown — a perfectly reasonable thing for it
        # to do — but that meant a completely ordinary frontend flow (end
        # session, then fetch the recap) raced against it: get_dialogue()
        # would silently hand back a fresh, empty DialogueState instead of
        # the one the meeting had actually populated, so this endpoint saw
        # "0 words" and returned the "not enough was captured" stub even
        # when a real conversation had just happened and was already
        # sitting right here in `records`. Reading from `records` instead
        # means summary generation no longer depends on in-memory state
        # that another endpoint may have already torn down, and no longer
        # cares what order the frontend calls DELETE vs POST /summary in.
        #
        # Capped to the same window size the old in-memory context used, so
        # a very long session doesn't blow up the prompt token budget.
        transcript_lines = [
            f"{r.get('speaker', 'Unknown')}: {r.get('text', '')}"
            for r in records if r.get("text")
        ]
        transcript_text = "\n".join(transcript_lines[-DIALOGUE_CONTEXT_WINDOW:])

        try:
            summary_md = await dialogue.generate_summary(session_id, transcript_text, tags)
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
        "camera_fps":       VISION_CAMERA_FPS,
        "camera_quality":   VISION_CAMERA_QUALITY,
        "tts_auto_hide_ms": DIALOGUE_TTS_AUTO_HIDE_MS,
        "lk_url":           LIVEKIT_PUBLIC_URL,
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