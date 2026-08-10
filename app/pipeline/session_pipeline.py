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
from typing import TYPE_CHECKING, Optional

import numpy as np

from app.core.config import cfg
from app.db import lkc_graph, supabase_client
from app.services import vision, eval_metrics, capture as _capture, privacy as _privacy
from app.services import lkc_retrieval
from app.pipeline.asr import VadChunker, transcribe, resample_livekit_frame
from app.pipeline.dialogue_service import (
    get_dialogue, assign_speaker, assign_speaker_words,
    update_mode, push_context, generate_response, clear_dialogue, ConvMode,
    QA_FOLLOW_UP_WINDOW_SECONDS,
)
from app.pipeline.livekit_rooms import broadcast, get_known_identity

log = logging.getLogger(__name__)

SAMPLE_RATE       = cfg.vad.sample_rate
SILENCE_THRESHOLD = cfg.vad.silence_threshold
# VISION_FRAME_INTERVAL decimation now happens in livekit_rooms.py's
# _drain_video (before the JPEG encode), which reads cfg.vision.frame_interval
# directly — see _process_video below.

# Sessions with a QA reply currently generating. generate_response() is a
# slow local-LLM call (many seconds), and clear_summon() used to only run
# once that call finished — so any speech captured while the first reply
# was still generating was itself treated as a fresh summon and spawned a
# SECOND concurrent _handle_qa_sse task for the same session. Those then
# queue up behind each other against Ollama (which serves one request at a
# time), roughly doubling (or worse) the wait for an answer to the question
# actually asked. This set is a simple per-session guard against that.
_qa_in_flight: set[str] = set()


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
    lkc_context = await retriever.query(
        full_text, top_k=cfg.lkc.retrieval_top_k, session_id=session_id
    )
    reply = await generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        await lkc_graph.write_to_lkc(_agent_record(session_id, ts, reply, mode.value))
        tts_queue.put_nowait(reply)
        broadcast(session_id, {"type": "agent_reply", "text": reply,
                               "mode": mode.value, "grounded": bool(lkc_context.strip())})
        broadcast(session_id, {"type": "speak", "text": reply})
        await supabase_client.insert_agent_reply(
            session_id=session_id, text=reply, mode=mode.value,
            timestamp_unix=ts, grounded=bool(lkc_context.strip()), lkc_context=lkc_context,
        )
    # NOTE: clear_summon() used to live here, i.e. it only ran once the LLM
    # reply finished. It's now consumed at task-creation time in
    # _handle_segment instead — see _qa_in_flight comment above for why.

    # ── Exit QA mode now that the reply has been delivered ───────────────
    # update_mode()'s FSM only transitions OUT of QA via rule 3: a LATER
    # segment carrying >=3 words of non-summoned speech. Nothing in that
    # FSM ever runs again just because this task finished — so if nobody
    # happens to speak a full sentence after the question, dlg.mode sits on
    # QA indefinitely and the frontend's mode indicator looks permanently
    # stuck, even though the summon button itself already reset. Return to
    # MEETING_CAPTURE/AMBIENT here explicitly instead of waiting on a
    # hypothetical future utterance. Guarded on dlg.mode == QA in case a
    # concurrent segment already moved it on while this call was in flight.
    if dlg.mode == ConvMode.QA:
        perc_state = vision.get_state(session_id)
        dlg.mode = ConvMode.MEETING_CAPTURE if perc_state.present_speakers else ConvMode.AMBIENT
        dlg.mode_entered_at = time.time()
        # Arm a short follow-up window so a question right after this reply
        # doesn't need the wake word / summon button again — see rule 2b in
        # dialogue_service.update_mode(). Only worth arming if a reply was
        # actually delivered; on failure/empty reply there's nothing to
        # follow up on, so leave it unarmed.
        if reply:
            dlg.qa_follow_up_until = time.time() + QA_FOLLOW_UP_WINDOW_SECONDS
        broadcast(session_id, {"type": "mode_change", "mode": dlg.mode.value})


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
    # One VadChunker per LiveKit participant identity, not one shared chunker.
    # A single shared chunker was fine for the original one-shared-mic design
    # (one track, possibly several people picked up by pyannote diarization),
    # but with several REMOTE participants each on their own track, a shared
    # chunker received raw PCM frames from independent tracks interleaved by
    # whatever order asyncio happened to schedule them — not time-aligned,
    # producing garbled audio rather than two real streams. Keying by
    # identity keeps each participant's own audio properly contiguous.
    _chunkers: dict[str, VadChunker] = {}

    def _chunker_for(identity: str) -> VadChunker:
        if identity not in _chunkers:
            _chunkers[identity] = VadChunker()
        return _chunkers[identity]

    # Every distinct identity that has actually sent audio in this session.
    # Used at segment-handling time to tell the two supported topologies
    # apart: exactly one identity means the classic shared-mic/shared-camera
    # case (several people, one track) where pyannote diarization is still
    # the right tool; more than one identity means separate remote
    # participants, each already disambiguated at the track level — running
    # diarization there would be redundant and diarization's own "Person A"/
    # "Person B" labels would be strictly worse than the real identities we
    # already have.
    _seen_audio_identities: set[str] = set()

    # dlg (dialogue/mode/QA state) stays a single shared instance across all
    # participants by design — per your call, wake-word/QA is session-wide,
    # not per-participant, so whoever says the wake word triggers one shared
    # reply rather than each participant getting independent QA state.
    dlg           = get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever()
    metrics       = eval_metrics.get_metrics(session_id)
    tts_queue: asyncio.Queue = asyncio.Queue()
    segment_index = 0
    _known_speakers: set[str] = set()
    _wx_align_cache: dict     = {}

    # _process_audio runs on every raw LiveKit audio frame (~50/sec at a
    # typical 20ms Opus frame size). Both "still buffering" and "segment
    # dropped as silence" below used to broadcast a "listening" SSE event on
    # EVERY one of those frames, so the frontend received ~50 "listening"
    # events/sec continuously for the whole session — a firehose of no-op
    # updates that, combined with a full-store Zustand subscription on the
    # frontend, produced a near-continuous re-render loop (see useSSE.ts /
    # page.tsx fix). "listening" only needs to signal a STATE CHANGE
    # (not-listening -> listening), so track the last broadcast state and
    # only emit when it actually flips.
    _last_listening_state: Optional[bool] = None

    def _broadcast_listening() -> None:
        nonlocal _last_listening_state
        if _last_listening_state is True:
            return
        _last_listening_state = True
        broadcast(session_id, {
            "type":     "listening",
            "mode":     dlg.mode.value,
            "summoned": _capture.is_summoned(session_id),
        })

    log.info(f"[pipeline:{session_id}] started")
    broadcast(session_id, {"type": "session", "session_id": session_id})

    segment_q: asyncio.Queue = asyncio.Queue()

    # Hard cap on how many completed-but-unprocessed segments may sit in
    # segment_q at once. _handle_segment does transcribe + word-level
    # diarization realignment + Supabase persistence + LKC writes, all
    # awaited in sequence by a single consumer (_segment_consumer below).
    # On CPU-only inference, total time-per-segment can end up exceeding the
    # real-time duration of the segment's own audio — and because the queue
    # was previously unbounded with no backlog handling, every later segment
    # inherited the FULL accumulated backlog on top of its own processing.
    # That's what made "asr_latency" (measured from enqueue time) look like
    # it was climbing forever in one session (6s -> 78s) even though actual
    # per-segment model inference time was roughly constant — it was queue
    # wait time compounding, not the model getting slower. Capping the
    # backlog and dropping stale segments once it's exceeded keeps the agent
    # close to real-time instead of dutifully transcribing further and
    # further into the past.
    MAX_SEGMENT_BACKLOG = 10000

    # ── Audio processor: fast stage (runs in the hot per-frame loop) ──────────
    # This must never block on ASR/diarization — its only job is to keep
    # draining audio_q in real time. Completed VAD segments are handed off to
    # segment_q for the (slow) heavy stage below, rather than processed here.

    async def _process_audio(identity: str, raw_frame) -> None:
        nonlocal segment_index, _last_listening_state

        _seen_audio_identities.add(identity)
        pcm_f32 = resample_livekit_frame(raw_frame)
        request_received_at = time.time()
        segment_audio = _chunker_for(identity).push(pcm_f32)

        if segment_audio is None:
            _broadcast_listening()
            return

        # NOTE: this used to gate on SILENCE_THRESHOLD * 2 (0.06 with the
        # default config). VadChunker already gates individual chunks against
        # SILENCE_THRESHOLD (0.03) before a segment is ever flushed here, so
        # doubling it again on the segment-level RMS was stricter than the
        # chunk-level gate that already ran — real speech segments in the
        # 0.04-0.05 range (normal speaking volume / quieter mic) were passing
        # VadChunker but then getting silently thrown away here. Use the
        # configured threshold directly; tune cfg.vad.silence_threshold if
        # you need this looser/stricter rather than re-multiplying here.
        seg_rms = float(np.sqrt(np.mean(segment_audio ** 2)))
        if seg_rms < SILENCE_THRESHOLD:
            # log.info(
            #     f"[pipeline:{session_id}] segment dropped as silence "
            #     f"(rms={seg_rms:.5f} < {SILENCE_THRESHOLD:.5f}, "
            #     f"{len(segment_audio)} samples / {len(segment_audio) / SAMPLE_RATE:.2f}s)"
            # )
            _broadcast_listening()
            return

        # A real (non-silent) segment is being handed off for transcription —
        # we're leaving the "listening" state, so the NEXT buffering frame
        # after this should announce "listening" again rather than staying
        # silently suppressed by the guard above.
        _last_listening_state = False

        segment_index += 1
        # Handing off here (instead of `await`ing the heavy chain inline) is
        # the fix: this coroutine returns immediately and _audio_consumer goes
        # straight back to `audio_q.get()`, so real-time audio ingestion is
        # never gated on how long ASR/diarization/persistence take.
        segment_q.put_nowait((identity, segment_index, segment_audio, time.time(), request_received_at))

    # ── Audio processor: heavy stage (runs in its own worker, FIFO order) ────
    # Processed one segment at a time, in submission order, by a dedicated
    # worker — so segments never get reordered relative to each other, but a
    # slow segment no longer stalls audio ingestion the way it did when this
    # ran inline inside _process_audio.

    async def _handle_segment(identity: str, seg_idx: int, segment_audio, seg_start: float, dequeued_at: float, request_received_at: float) -> None:
        loop = asyncio.get_event_loop()

        # Time spent sitting in segment_q behind earlier segments, before
        # this one even started processing. This used to get silently
        # folded into "asr_latency" below (which was measured from seg_start,
        # i.e. enqueue time) — separating it out is what actually explains
        # the growing numbers: it's backlog, not the model slowing down.
        queue_wait_ms = round((dequeued_at - seg_start) * 1000)

        transcribe_started_at = time.time()
        full_text, detected_lang, raw_word_ts = await transcribe(
            segment_audio, loop, _wx_align_cache
        )
        if not full_text:
            return

        # Now measured purely around the transcribe() call — this is actual
        # model inference time, not inference-time-plus-queue-wait.
        asr_latency = round((time.time() - transcribe_started_at) * 1000)
        metrics.record_asr(asr_latency)

        perc_state = vision.get_state(session_id)

        # Multiple distinct participant identities have sent audio in this
        # session => genuinely separate remote participants, each already
        # disambiguated at the track level. Running pyannote diarization
        # here would be redundant (we already know who this segment's audio
        # came from) and its generic "Person A"/"Person B" labels would be
        # strictly worse than the real name/identity we already have. This
        # does NOT touch the single-identity case below, which is the
        # original shared-mic/shared-camera design (several people picked up
        # by one track) and still needs diarization to tell them apart.
        if len(_seen_audio_identities) > 1:
            speaker = get_known_identity(session_id, identity) or identity
            # One LiveKit identity == one distinct remote participant here,
            # so it's safe (and collision-free vs a display name) to gate
            # consent on the identity itself.
            consent_key = identity
            word_timestamps = (
                [{**w, "speaker": speaker} for w in raw_word_ts] if raw_word_ts else raw_word_ts
            )
        else:
            # Word-level speaker alignment. This is the single (executor-offloaded)
            # diarization pass for this segment — the segment-level speaker below
            # is derived from it rather than re-running diarization a second time.
            word_timestamps = (
                await assign_speaker_words(dlg, raw_word_ts, segment_audio)
                if raw_word_ts else raw_word_ts
            )

            # Speaker (segment-level) — resolved as the majority vote across the
            # word-level labels, which come from the same audio diarization pass.
            # NOTE: this must NOT be overwritten with perc_state.present_speakers.
            # present_speakers is vision-derived and indexes purely off
            # segment_index, so it's uncorrelated with who is actually talking in
            # this segment — using it here previously pinned every segment to
            # whichever face vision happened to detect (e.g. a lone "Person
            # (anon)"), which is what caused the audio/vision speaker mismatches.
            if word_timestamps:
                word_speakers = [w.get("speaker") for w in word_timestamps if w.get("speaker")]
                speaker = (
                    max(set(word_speakers), key=word_speakers.count)
                    if word_speakers else await assign_speaker(dlg, audio_segment=segment_audio)
                )
            else:
                speaker = await assign_speaker(dlg, audio_segment=segment_audio)

            # Shared mic/camera: a single LiveKit identity can carry several
            # physically-distinct, differently-consenting people (diarization
            # is exactly what's telling them apart). Gating on `identity`
            # here would silently apply the account holder's consent to
            # everyone else picked up on that mic, so this MUST stay keyed
            # on the per-voice diarization label, not the account identity.
            consent_key = speaker

        # Privacy
        redacted_text = (
            await _privacy.redact_async(full_text) if _privacy.check_consent(consent_key) else full_text
        )
        # summoned = wake-word spoken in THIS segment, OR the frontend's
        # manual "summon agent" button was pressed (force_summon) and hasn't
        # been consumed yet. Previously only the spoken wake-word was checked
        # here, so pressing the manual summon button set a flag that nothing
        # ever read — the next segment would only enter QA mode if the user
        # also happened to say a wake-word in the same utterance.
        summoned = _capture.is_summoned(session_id) or _capture.check_summon(session_id, full_text)

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

        # Write to LKC graph (failure is caught inside process_segment and logged;
        # the record dict is always returned so the broadcast below is never blocked)
        record = await _capture.process_segment(
            session_id, speaker, redacted_text,
            seg_start, new_mode.value, detected_lang,
            confirm_agent=True,
            word_timestamps=word_timestamps,
        )
        metrics.record_tags(record.get("tags", {}))
        push_context(dlg, speaker, full_text)

        e2e = round((time.time() - request_received_at) * 1000)
        metrics.record_e2e(e2e)

        log.info(
            f"[{session_id}] seg#{seg_idx} "
            f"queue_wait={queue_wait_ms}ms asr={asr_latency}ms e2e={e2e}ms [{detected_lang}] "
            f"mode={new_mode.value} summoned={summoned}: {full_text[:60]}"
        )

        # ── SSE broadcast — happens FIRST, before any persistence ────────────
        # Previously the broadcast was the last line, so any Supabase/LKC
        # exception caused _audio_consumer to catch-and-log, skip the rest of
        # _process_audio, and leave the frontend with no transcript at all.
        # Now the frontend always receives the event regardless of DB health.
        broadcast(session_id, {
            "type":            "transcript",
            "segment":         seg_idx,
            "session_id":      session_id,
            "speaker":         speaker,
            "text":            full_text,
            "language":        detected_lang,
            "latency_ms":      asr_latency,
            "queue_wait_ms":   queue_wait_ms,
            "e2e_ms":          e2e,
            "mode":            new_mode.value,
            "timestamp":       record.get("timestamp_iso", ""),
            "engagement":      perc_state.engagement_cues.get(speaker, "unknown"),
            "tags":            record.get("tags", {}),
            "environment":     perc_state.environment_state,
            "word_timestamps": word_timestamps,
            "summoned":        summoned,
        })

        # ── Supabase persistence — isolated so failures never block SSE ───────
        # Fire-and-forget: this is network I/O (a DB insert + optional audio
        # upload) with no downstream step in THIS function depending on its
        # result, and it was previously awaited inline — meaning every ms
        # this took was added directly onto _segment_consumer's per-segment
        # time, which is exactly the budget that determines whether the
        # pipeline keeps up with real-time audio (see MAX_SEGMENT_BACKLOG
        # comment above). Moving it to a background task lets the consumer
        # go straight back to segment_q.get() for the next segment while
        # this persists. Errors are still caught and logged inside the task
        # so a failure here can never surface as an unhandled exception.
        async def _persist_to_supabase() -> None:
            try:
                await supabase_client.insert_transcript(
                    session_id=session_id, speaker=speaker, text=redacted_text,
                    language=detected_lang, mode=new_mode.value,
                    timestamp_unix=seg_start,
                    timestamp_iso=record.get("timestamp_iso", ""),
                    tags=record.get("tags", {}), word_timestamps=word_timestamps,
                    asr_latency_ms=asr_latency, e2e_latency_ms=e2e,
                    segment_index=seg_idx,
                )
                if cfg.supabase.store_audio:
                    await supabase_client.upload_audio_segment(
                        session_id, seg_idx, segment_audio
                    )
            except Exception as exc:
                log.error(
                    f"[pipeline:{session_id}] Supabase persist failed for "
                    f"seg#{seg_idx}: {exc}",
                    exc_info=True,
                )

        asyncio.create_task(_persist_to_supabase())

        # Entry utterance (greeting / confirmation)
        if entry_utterance:
            ts = time.time()
            await lkc_graph.write_to_lkc(_agent_record(session_id, ts, entry_utterance, new_mode.value))
            tts_queue.put_nowait(entry_utterance)
            broadcast(session_id, {"type": "agent_reply", "text": entry_utterance, "mode": new_mode.value})
            broadcast(session_id, {"type": "speak", "text": entry_utterance})

        # QA reply
        if new_mode == ConvMode.QA:
            # Consume the summon flag now, before the (slow) LLM call, not
            # after it returns. Previously clear_summon() ran at the end of
            # _handle_qa_sse, so any speech captured while a reply was still
            # generating was still seen as "summoned" and could trigger a
            # second, unrelated QA task for the same session.
            _capture.clear_summon(session_id)

            if session_id in _qa_in_flight:
                log.info(
                    f"[{session_id}] QA reply already in flight — not spawning "
                    f"a second one for: {full_text[:60]!r}"
                )
            else:
                _qa_in_flight.add(session_id)
                qa_task = asyncio.create_task(
                    _handle_qa_sse(session_id, dlg, full_text, retriever, new_mode, tts_queue)
                )
                qa_task.add_done_callback(
                    lambda _t, sid=session_id: _qa_in_flight.discard(sid)
                )

    # ── Video processor: fast stage (runs in the hot per-frame loop) ─────────
    # Frames arriving here are already decimated to every VISION_FRAME_INTERVAL-
    # th frame (see livekit_rooms.py's _drain_video). This stage must never
    # `await` vision inference directly, or it stalls draining video_q the
    # same way the old inline ASR/diarization stalled audio_q.

    # (identity, jpeg_bytes) of the most recent video frame across ALL
    # participants — Ollama serves one request at a time regardless, so
    # analysing every participant's feed concurrently isn't an option; this
    # keeps the existing latest-frame-wins behavior but now remembers whose
    # frame it is, so the right participant's known name gets substituted
    # rather than always looking up a single session-wide identity.
    _vision_latest: Optional[tuple[str, bytes]] = None
    _vision_event = asyncio.Event()

    async def _process_video(identity: str, jpeg_bytes: bytes) -> None:
        nonlocal _vision_latest
        # VISION_FRAME_INTERVAL decimation now happens upstream, in
        # livekit_rooms.py's _drain_video, before the RGBA convert + JPEG
        # encode — every jpeg_bytes that reaches this function has already
        # passed that filter, so filtering again here would decimate twice
        # (frame_interval=5 would become an effective 1-in-25, not 1-in-5).

        # Latest-frame-wins: if the vision worker is still busy with a
        # previous frame, overwrite the pending slot rather than queuing —
        # analysing a stale frame once the worker catches up is worse than
        # just analysing whatever is current when it's ready.
        _vision_latest = (identity, jpeg_bytes)
        _vision_event.set()

    # ── Video processor: heavy stage (its own worker, latest-frame-wins) ─────

    async def _run_vision_analysis(frame: tuple[str, bytes]) -> None:
        identity, jpeg_bytes = frame
        t0         = time.time()
        state      = await vision.analyse_frame(
            session_id, jpeg_bytes, known_identity=get_known_identity(session_id, identity)
        )
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
            await lkc_graph.write_to_lkc(
                _vision_record(
                    session_id, ts, state.scene_summary,
                    state.present_speakers, state.engagement_cues, state.environment_state,
                )
            )
            if cfg.supabase.store_vision:
                await supabase_client.insert_vision_frame(
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
    # FIXED: asyncio.wait on ensure_future(queue.get()) is fatally broken —
    # cancelling a pending get() drops the item silently, and a cancelled
    # future from the previous iteration can re-raise CancelledError via
    # future.result(), which propagates out and calls room.disconnect(),
    # making the agent disappear. Two persistent consumer tasks fix both.

    async def _audio_consumer() -> None:
        while True:
            identity, frame = await audio_q.get()
            try:
                await _process_audio(identity, frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f"[pipeline:{session_id}] _process_audio raised unexpectedly")

    async def _segment_consumer() -> None:
        while True:
            identity, seg_idx, segment_audio, seg_start, request_received_at = await segment_q.get()
            dequeued_at = time.time()

            # If more items piled up behind this one while we were busy,
            # drop the stale ones instead of dutifully working through the
            # whole backlog in order. FIFO get_nowait() below removes from
            # the front (the oldest remaining), so this keeps the freshest
            # MAX_SEGMENT_BACKLOG segments and discards the staler ones —
            # the fix for the compounding-delay pattern described above.
            backlog = segment_q.qsize()
            if backlog > MAX_SEGMENT_BACKLOG:
                dropped = 0
                while segment_q.qsize() > MAX_SEGMENT_BACKLOG:
                    try:
                        segment_q.get_nowait()
                        dropped += 1
                    except asyncio.QueueEmpty:
                        break
                log.warning(
                    f"[pipeline:{session_id}] segment backlog was {backlog}, dropped "
                    f"{dropped} stale segment(s) to catch back up to real-time"
                )

            try:
                await _handle_segment(identity, seg_idx, segment_audio, seg_start, dequeued_at, request_received_at)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f"[pipeline:{session_id}] _handle_segment raised unexpectedly (seg#{seg_idx})")

    async def _video_consumer() -> None:
        while True:
            identity, jpeg = await video_q.get()
            try:
                await _process_video(identity, jpeg)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f"[pipeline:{session_id}] _process_video raised unexpectedly")

    # Minimum time between successive vision dispatches (start-to-start).
    # The previous version derived this from cfg.vision.camera_fps (5),
    # giving a 0.2s floor — meaningless, since every real call already
    # takes 4-8s on its own, so `remaining` was always negative and
    # asyncio.sleep() never actually fired. This is a real, deliberate cap:
    # check the vision LLM at most once every 10s, explicitly, regardless
    # of how fast any individual call happens to come back. That leaves
    # much more idle time on Ollama's single request slot for QA calls to
    # land in without queueing behind a passive perception check.
    _VISION_MIN_INTERVAL = 10.0

    async def _vision_worker() -> None:
        last_dispatch_at = 0.0
        while True:
            await _vision_event.wait()
            _vision_event.clear()
            frame = _vision_latest
            if frame is None:
                continue

            # Let a user-facing QA reply have Ollama's single request slot
            # uncontested. Skipping this frame (rather than blocking here)
            # keeps the worker responsive to _vision_event — the next frame
            # that arrives once QA finishes will be picked up normally.
            if session_id in _qa_in_flight:
                continue

            since_last = time.time() - last_dispatch_at
            if since_last < _VISION_MIN_INTERVAL:
                # Not our turn yet. Sleep out the remainder, but re-check
                # _qa_in_flight afterwards rather than dispatching blindly —
                # a QA call may have started during this wait.
                await asyncio.sleep(_VISION_MIN_INTERVAL - since_last)
                if session_id in _qa_in_flight:
                    continue
                # A newer frame may have arrived while we waited; use it —
                # same latest-frame-wins principle as everywhere else here.
                frame = _vision_latest

            last_dispatch_at = time.time()
            try:
                await _run_vision_analysis(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f"[pipeline:{session_id}] _run_vision_analysis raised unexpectedly")

    audio_task   = asyncio.create_task(_audio_consumer(), name=f"audio-{session_id}")
    segment_task = asyncio.create_task(_segment_consumer(), name=f"segment-{session_id}")
    video_task   = asyncio.create_task(_video_consumer(), name=f"video-{session_id}")
    vision_task  = asyncio.create_task(_vision_worker(), name=f"vision-{session_id}")

    try:
        results = await asyncio.gather(
            audio_task, segment_task, video_task, vision_task, return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                log.error(
                    f"[pipeline:{session_id}] consumer task exited with error: {result!r}",
                    exc_info=result,
                )
    except asyncio.CancelledError:
        log.info(f"[pipeline:{session_id}] cancelled")
        audio_task.cancel()
        segment_task.cancel()
        video_task.cancel()
        vision_task.cancel()
        await asyncio.gather(audio_task, segment_task, video_task, vision_task, return_exceptions=True)
        raise