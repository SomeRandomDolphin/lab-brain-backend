"""
app/pipeline/session_pipeline.py — LiveKit audio+video pipeline.

Consumes audio/video frames from the LiveKit subscriber queues and routes them
through the full Lab Brain pipeline: VAD → ASR → NER → Dialogue → SSE.
Also writes to Supabase (transcripts, vision frames, agent replies).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from app.core.config import cfg
from app.db import lkc_graph, supabase_client
from app.services import vision, eval_metrics, capture as _capture, privacy as _privacy
from app.services import lkc_retrieval
from app.pipeline.asr import VadChunker, transcribe, resample_livekit_frame
from app.pipeline.dialogue_service import (
    get_dialogue, assign_speaker, assign_speaker_words,
    update_mode, push_context, generate_response, clear_dialogue, ConvMode,
)
from app.pipeline.livekit_rooms import broadcast

log = logging.getLogger(__name__)

SAMPLE_RATE           = cfg.vad.sample_rate
SILENCE_THRESHOLD     = cfg.vad.silence_threshold
VISION_FRAME_INTERVAL = cfg.vision.frame_interval


# ── LKC record builders ───────────────────────────────────────────────────────

def _vision_record(session_id, ts, scene_summary, speakers, cues, env_state):
    r = {
        "type":             "vision",
        "session_id":       session_id,
        "timestamp_iso":    datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix":   round(ts, 3),
        "scene_summary":    scene_summary,
        "present_speakers": speakers,
        "engagement_cues":  cues,
    }
    if env_state:
        r["environment_state"] = env_state
    return r


def _agent_record(session_id, ts, reply_text, mode):
    return {
        "type":           "agent_reply",
        "session_id":     session_id,
        "timestamp_iso":  datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "text":           reply_text,
        "mode":           mode,
    }


# ── QA reply (SSE path) ───────────────────────────────────────────────────────

async def _handle_qa_sse(session_id, dlg, full_text, retriever, mode, tts_queue):
    lkc_context = retriever.query(
        full_text, top_k=cfg.lkc.retrieval_top_k, session_id=session_id
    )
    reply = await generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        lkc_graph.write_to_lkc(_agent_record(session_id, ts, reply, mode.value))
        tts_queue.put_nowait(reply)
        broadcast(session_id, {"type": "agent_reply", "text": reply,
                               "mode": mode.value, "grounded": bool(lkc_context.strip())})
        broadcast(session_id, {"type": "speak", "text": reply})
        supabase_client.insert_agent_reply(
            session_id=session_id, text=reply, mode=mode.value,
            timestamp_unix=ts, grounded=bool(lkc_context.strip()), lkc_context=lkc_context,
        )
    _capture.clear_summon(session_id)


# ── Main pipeline coroutine ───────────────────────────────────────────────────

async def livekit_pipeline(
    session_id: str,
    audio_q: asyncio.Queue,
    video_q: asyncio.Queue,
) -> None:
    """
    Entry point called by livekit_rooms.start_subscriber().
    Runs until the subscriber task is cancelled (DELETE /livekit/room/{sid}).
    """
    chunker       = VadChunker()
    dlg           = get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever()
    metrics       = eval_metrics.get_metrics(session_id)
    tts_queue: asyncio.Queue = asyncio.Queue()
    segment_index = 0
    frame_counter = 0
    _known_speakers: set[str] = set()
    _wx_align_cache: dict     = {}

    log.info(f"[pipeline:{session_id}] started")
    broadcast(session_id, {"type": "session", "session_id": session_id})

    # ── Audio processor ───────────────────────────────────────────────────────

    async def _process_audio(raw_frame) -> None:
        nonlocal segment_index

        pcm_f32 = resample_livekit_frame(raw_frame)
        request_received_at = time.time()
        segment_audio = chunker.push(pcm_f32)

        if segment_audio is None:
            broadcast(session_id, {
                "type":     "listening",
                "mode":     dlg.mode.value,
                "summoned": _capture.is_summoned(session_id),
            })
            return

        if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
            return

        seg_start = time.time()
        segment_index += 1
        loop = asyncio.get_event_loop()

        full_text, detected_lang, raw_word_ts = await transcribe(
            segment_audio, loop, _wx_align_cache
        )
        if not full_text:
            return

        asr_latency = round((time.time() - seg_start) * 1000)
        metrics.record_asr(asr_latency)

        # Speaker
        perc_state = vision.get_state(session_id)
        speaker    = assign_speaker(dlg, audio_segment=segment_audio)
        if perc_state.present_speakers:
            speaker = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]

        # Word-level speaker alignment
        word_timestamps = (
            assign_speaker_words(dlg, raw_word_ts, segment_audio)
            if raw_word_ts else raw_word_ts
        )

        # Privacy
        redacted_text = (
            _privacy.redact(full_text) if _privacy.check_consent(speaker) else full_text
        )
        summoned = _capture.check_summon(session_id, full_text)

        # New speaker detection
        current_known = set(perc_state.present_speakers)
        new_speakers  = list(current_known - _known_speakers)
        _known_speakers.update(current_known)

        pending_confirms     = _capture.get_pending_confirmations(session_id)
        pending_confirm_text = pending_confirms[0] if pending_confirms else None

        # Mode FSM
        prev_mode = dlg.mode
        new_mode, entry_utterance = update_mode(
            dlg, full_text, perc_state.present_speakers, new_speakers,
            pending_confirmation=pending_confirm_text,
            summoned=summoned,
        )
        if new_mode != prev_mode:
            metrics.record_mode_switch(new_mode.value)
            broadcast(session_id, {"type": "mode_change", "mode": new_mode.value})

        # Write to LKC graph
        record = _capture.process_segment(
            session_id, speaker, redacted_text,
            seg_start, new_mode.value, detected_lang,
            confirm_agent=True,
            word_timestamps=word_timestamps,
        )
        metrics.record_tags(record.get("tags", {}))
        push_context(dlg, speaker, full_text)

        e2e = round((time.time() - request_received_at) * 1000)
        metrics.record_e2e(e2e)

        # Persist to Supabase
        supabase_client.insert_transcript(
            session_id=session_id, speaker=speaker, text=redacted_text,
            language=detected_lang, mode=new_mode.value,
            timestamp_unix=seg_start, timestamp_iso=record["timestamp_iso"],
            tags=record.get("tags", {}), word_timestamps=word_timestamps,
            asr_latency_ms=asr_latency, e2e_latency_ms=e2e,
            segment_index=segment_index,
        )
        if cfg.supabase.store_audio:
            supabase_client.upload_audio_segment(session_id, segment_index, segment_audio)

        log.info(
            f"[{session_id}] seg#{segment_index} "
            f"asr={asr_latency}ms e2e={e2e}ms [{detected_lang}] "
            f"mode={new_mode.value} summoned={summoned}: {full_text[:60]}"
        )

        # Broadcast transcript via SSE
        broadcast(session_id, {
            "type":            "transcript",
            "segment":         segment_index,
            "session_id":      session_id,
            "speaker":         speaker,
            "text":            full_text,
            "language":        detected_lang,
            "latency_ms":      asr_latency,
            "e2e_ms":          e2e,
            "mode":            new_mode.value,
            "timestamp":       record["timestamp_iso"],
            "engagement":      perc_state.engagement_cues.get(speaker, "unknown"),
            "tags":            record.get("tags", {}),
            "environment":     perc_state.environment_state,
            "word_timestamps": word_timestamps,
            "summoned":        summoned,
        })

        # Entry utterance (greeting / confirmation)
        if entry_utterance:
            ts = time.time()
            lkc_graph.write_to_lkc(_agent_record(session_id, ts, entry_utterance, new_mode.value))
            tts_queue.put_nowait(entry_utterance)
            broadcast(session_id, {"type": "agent_reply", "text": entry_utterance, "mode": new_mode.value})
            broadcast(session_id, {"type": "speak", "text": entry_utterance})

        # QA reply
        if new_mode == ConvMode.QA:
            asyncio.create_task(
                _handle_qa_sse(session_id, dlg, full_text, retriever, new_mode, tts_queue)
            )

    # ── Video processor ───────────────────────────────────────────────────────

    async def _process_video(jpeg_bytes: bytes) -> None:
        nonlocal frame_counter
        frame_counter += 1
        if frame_counter % VISION_FRAME_INTERVAL != 0:
            return

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
            ts = time.time()
            lkc_graph.write_to_lkc(
                _vision_record(
                    session_id, ts, state.scene_summary,
                    state.present_speakers, state.engagement_cues, state.environment_state,
                )
            )
            if cfg.supabase.store_vision:
                supabase_client.insert_vision_frame(
                    session_id=session_id, timestamp_unix=ts,
                    scene_summary=state.scene_summary,
                    present_speakers=state.present_speakers,
                    engagement_cues=state.engagement_cues,
                    environment_state=state.environment_state,
                    latency_ms=latency_ms,
                )

        broadcast(session_id, {
            "type":              "perception",
            "session_id":        session_id,
            "present_speakers":  state.present_speakers,
            "engagement_cues":   state.engagement_cues,
            "scene_summary":     state.scene_summary,
            "environment_state": state.environment_state,
            "latency_ms":        latency_ms,
        })

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            done, pending = await asyncio.wait(
                [
                    asyncio.ensure_future(audio_q.get()),
                    asyncio.ensure_future(video_q.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                item = future.result()
                if isinstance(item, bytes):
                    await _process_video(item)
                else:
                    await _process_audio(item)
            for future in pending:
                future.cancel()

    except asyncio.CancelledError:
        log.info(f"[pipeline:{session_id}] cancelled")
        raise
