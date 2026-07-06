"""
app/api/v1/endpoints/sessions.py — Session-level endpoints.

POST   /summary/{sid}         — generate + persist LLM summary
GET    /mode/{sid}            — current dialogue mode + summon flag
GET    /perception/{sid}      — latest vision perception state
GET    /config/client         — frontend config (camera fps, tts hide ms, lk_url)
GET    /metrics               — all session metric summaries
GET    /metrics/csv           — CSV export of raw metric samples
POST   /eval/wer              — compute WER for a reference/hypothesis pair
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

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
async def post_summary(session_id: str):
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
async def get_mode(session_id: str):
    dialogue  = _get_dialogue_module()
    dlg_state = dialogue.get_dialogue(session_id)
    return {
        "session_id": session_id,
        "mode":       dlg_state.mode.value,
        "summoned":   is_summoned(session_id),
    }


@router.get("/perception/{session_id}")
async def get_perception(session_id: str):
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
        "lk_url":           cfg.livekit.url,
    }


@router.get("/metrics")
async def get_all_metrics():
    return JSONResponse(await eval_metrics.all_summaries())


@router.get("/metrics/csv", response_class=PlainTextResponse)
async def get_metrics_csv():
    return PlainTextResponse(await eval_metrics.all_csv(), media_type="text/csv")


@router.post("/eval/wer")
async def evaluate_wer(req: WerRequest):
    metrics = await eval_metrics.get_metrics(req.session_id)
    wer     = metrics.record_wer(req.reference, req.hypothesis)
    return {"session_id": req.session_id, "wer": round(wer, 4)}