"""
app/api/v1/endpoints/websockets.py — Legacy WebSocket endpoints.

Kept for backward compatibility with the Month 5 index.html client.
New frontend uses POST /livekit/room + GET /events/{sid} (SSE) instead.

/ws/asr    — send raw float32 PCM, receive transcript + agent_reply JSON
/ws/vision — send JPEG bytes, receive perception JSON
/ws/tts    — receive TTS speak events
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import os
from db import lkc_graph
from services import vision, eval_metrics
from services import capture as _capture
from services import privacy as _privacy
from services import lkc_retrieval

log = logging.getLogger(__name__)

router = APIRouter(tags=["websockets_legacy"])

SAMPLE_RATE           = int(os.environ.get("VAD_SAMPLE_RATE", "16000"))
SILENCE_THRESHOLD     = float(os.environ.get("VAD_SILENCE_THRESHOLD", "0.03"))
VISION_FRAME_INTERVAL = int(os.environ.get("VISION_FRAME_INTERVAL", "5"))
LKC_RETRIEVAL_TOP_K   = int(os.environ.get("LKC_RETRIEVAL_TOP_K", "4"))

# Per-session TTS queues (fed by both WS and LiveKit pipelines)
_tts_queues: dict[str, asyncio.Queue] = {}


def get_tts_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _tts_queues:
        _tts_queues[session_id] = asyncio.Queue()
    return _tts_queues[session_id]


def drop_tts_queue(session_id: str) -> None:
    _tts_queues.pop(session_id, None)


def _make_agent_record(session_id, ts, reply_text, mode):
    from datetime import datetime
    return {
        "type":           "agent_reply",
        "session_id":     session_id,
        "timestamp_iso":  datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "text":           reply_text,
        "mode":           mode,
    }


@router.websocket("/ws/asr")
async def asr_endpoint(ws: WebSocket):
    """
    Raw PCM WebSocket — retained for backward compat with Month 5 index.html.
    New clients use LiveKit + SSE.
    """
    from pipeline.asr import VadChunker, transcribe, WHISPERX_AVAILABLE
    from pipeline.dialogue_service import (
        get_dialogue, assign_speaker, assign_speaker_words,
        update_mode, push_context, generate_response, ConvMode,
    )

    await ws.accept()
    session_id    = str(uuid.uuid4())[:8]
    chunker       = VadChunker()
    dlg           = get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever()
    metrics       = eval_metrics.get_metrics(session_id)
    segment_index = 0
    _wx_align_cache: dict = {}
    _known_speakers: set[str] = set()

    log.info(f"ASR (WS compat) session {session_id} connected")
    await ws.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            request_received_at = time.time()
            data = await ws.receive_bytes()
            pcm  = np.frombuffer(data, dtype=np.float32)
            if pcm.size == 0:
                continue

            segment_audio = chunker.push(pcm)
            if segment_audio is None:
                await ws.send_json({
                    "type":     "listening",
                    "mode":     dlg.mode.value,
                    "summoned": _capture.is_summoned(session_id),
                })
                continue

            if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
                continue

            seg_start = time.time()
            segment_index += 1
            loop = asyncio.get_event_loop()

            full_text, detected_lang, raw_word_ts = await transcribe(
                segment_audio, loop, _wx_align_cache
            )

            if not full_text:
                continue

            asr_latency = round((time.time() - seg_start) * 1000)
            metrics.record_asr(asr_latency)

            perc_state = vision.get_state(session_id)
            speaker    = assign_speaker(dlg, audio_segment=segment_audio)
            if perc_state.present_speakers:
                speaker = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]

            word_timestamps = (
                assign_speaker_words(dlg, raw_word_ts, segment_audio)
                if raw_word_ts else raw_word_ts
            )
            redacted_text = await _privacy.redact_async(full_text) if _privacy.check_consent(speaker) else full_text
            summoned      = _capture.check_summon(session_id, full_text)

            current_known = set(perc_state.present_speakers)
            new_speakers  = list(current_known - _known_speakers)
            _known_speakers.update(current_known)

            pending_confirms     = _capture.get_pending_confirmations(session_id)
            pending_confirm_text = pending_confirms[0] if pending_confirms else None

            prev_mode = dlg.mode
            new_mode, entry_utterance = update_mode(
                dlg, full_text, perc_state.present_speakers, new_speakers,
                pending_confirmation=pending_confirm_text,
                summoned=summoned,
            )
            if new_mode != prev_mode:
                metrics.record_mode_switch(new_mode.value)

            record = await _capture.process_segment(   # was missing `await` — returned a coroutine
                session_id, speaker, redacted_text, seg_start,
                new_mode.value, detected_lang,
                confirm_agent=True,
                word_timestamps=word_timestamps,
            )
            metrics.record_tags(record.get("tags", {}))
            push_context(dlg, speaker, full_text)

            e2e = round((time.time() - request_received_at) * 1000)
            metrics.record_e2e(e2e)

            await ws.send_json({
                "type": "transcript", "segment": segment_index,
                "session_id": session_id, "speaker": speaker,
                "text": full_text, "language": detected_lang,
                "latency_ms": asr_latency, "e2e_ms": e2e,
                "mode": new_mode.value, "timestamp": record["timestamp_iso"],
                "engagement": perc_state.engagement_cues.get(speaker, "unknown"),
                "tags": record.get("tags", {}), "environment": perc_state.environment_state,
                "word_timestamps": word_timestamps, "summoned": summoned,
            })

            if entry_utterance:
                await lkc_graph.write_to_lkc(
                    _make_agent_record(session_id, time.time(), entry_utterance, new_mode.value)
                )
                get_tts_queue(session_id).put_nowait(entry_utterance)
                await ws.send_json({"type": "agent_reply", "text": entry_utterance, "mode": new_mode.value})

            if new_mode == ConvMode.QA:
                asyncio.create_task(
                    _handle_qa_ws(session_id, ws, dlg, full_text, retriever, metrics, new_mode)
                )

    except WebSocketDisconnect:
        log.info(f"ASR (WS) session {session_id} disconnected")
    except Exception as e:
        log.error(f"ASR (WS) session {session_id} error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        vision.clear_state(session_id)
        from pipeline.dialogue_service import clear_dialogue
        clear_dialogue(session_id)
        _capture.clear_summon(session_id)
        drop_tts_queue(session_id)


async def _handle_qa_ws(session_id, ws, dlg, full_text, retriever, metrics, mode):
    from pipeline.dialogue_service import generate_response, ConvMode
    lkc_context = await retriever.query(full_text, top_k=LKC_RETRIEVAL_TOP_K, session_id=session_id)
    reply = await generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        await lkc_graph.write_to_lkc(_make_agent_record(session_id, ts, reply, mode.value))
        get_tts_queue(session_id).put_nowait(reply)
        try:
            await ws.send_json({"type": "agent_reply", "text": reply,
                                "mode": mode.value, "grounded": bool(lkc_context.strip())})
        except Exception:
            pass
    _capture.clear_summon(session_id)


@router.websocket("/ws/vision")
async def vision_endpoint(ws: WebSocket):
    await ws.accept()
    session_id    = ws.query_params.get("session_id", str(uuid.uuid4())[:8])
    frame_counter = 0
    log.info(f"Vision (WS compat) session {session_id} connected")
    try:
        while True:
            jpeg_bytes = await ws.receive_bytes()
            frame_counter += 1
            if frame_counter % VISION_FRAME_INTERVAL != 0:
                continue
            t0         = time.time()
            state      = await vision.analyse_frame(session_id, jpeg_bytes)
            latency_ms = round((time.time() - t0) * 1000)
            m = eval_metrics.get_metrics(session_id)
            m.record_vision(
                latency_ms,
                ok=not bool(state.error_count and state.frame_count == state.error_count),
                stub=not vision.GEMINI_AVAILABLE,
            )
            env_valid = bool(
                state.environment_state.get("layout") not in (None, "unknown")
                or state.environment_state.get("objects")
            )
            m.record_environment(env_valid)
            stub_marker = "[Vision stub"
            if state.scene_summary and not state.scene_summary.startswith(stub_marker):
                from datetime import datetime
                await lkc_graph.write_to_lkc({
                    "type":             "vision",
                    "session_id":       session_id,
                    "timestamp_iso":    datetime.utcfromtimestamp(time.time()).isoformat() + "Z",
                    "timestamp_unix":   round(time.time(), 3),
                    "scene_summary":    state.scene_summary,
                    "present_speakers": state.present_speakers,
                    "engagement_cues":  state.engagement_cues,
                    "environment_state": state.environment_state,
                })
            await ws.send_json({
                "type": "perception", "session_id": session_id,
                "present_speakers": state.present_speakers,
                "engagement_cues": state.engagement_cues,
                "scene_summary": state.scene_summary,
                "environment_state": state.environment_state,
                "latency_ms": latency_ms,
            })
    except WebSocketDisconnect:
        log.info(f"Vision (WS) session {session_id} disconnected")
    except Exception as e:
        log.error(f"Vision (WS) session {session_id} error: {e}", exc_info=True)


@router.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id")
    if not session_id:
        await ws.close(code=1008)
        return
    queue = get_tts_queue(session_id)
    log.info(f"TTS relay (WS compat) {session_id} connected")
    try:
        while True:
            text = await queue.get()
            await ws.send_json({"type": "speak", "text": text})
    except WebSocketDisconnect:
        log.info(f"TTS relay (WS) {session_id} disconnected")
    except Exception as e:
        log.error(f"TTS relay (WS) {session_id} error: {e}", exc_info=True)