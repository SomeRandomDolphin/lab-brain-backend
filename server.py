"""
Module 5 — Month 6 Server

Month 6 additions over Month 5
-------------------------------
  NEW  livekit_rooms.py   — token endpoint, room management, audio/video subscriber,
                             SSE broadcaster replacing the three WebSocket endpoints
  UPD  CORS               — adds http://localhost:5173 (Vite dev server)
  UPD  config.json/py     — new [livekit] section

New REST endpoints
------------------
  POST   /livekit/room                  — create room, return session_id + token
  GET    /livekit/token                 — re-issue token (reconnect)
  GET    /livekit/room/{sid}            — room status
  DELETE /livekit/room/{sid}            — end session, stop subscriber
  GET    /events/{sid}                  — SSE stream (replaces /ws/asr /ws/tts /ws/vision)

All Month 5 endpoints (/lkc/*, /agent/summon/*, /metrics, /capture/*, /summary/*,
/privacy/*, /ner/status, /retrieval/stats, /config/client, /mode/*, /perception/*,
WebSocket /ws/asr, /ws/vision, /ws/tts) are retained UNCHANGED for backward compat.

Audio/video pipeline (Month 6)
-------------------------------
Browser publishes audio + video tracks into a LiveKit room.
Backend subscribes as "lab-brain-server" participant via livekit_rooms.py.
Audio frames → resample 48kHz stereo → 16kHz mono → existing VadChunker → WhisperX.
Video frames → JPEG bytes → existing vision.analyse_frame() (throttled).
Results broadcast via SSE instead of WebSocket sends.

Run:
  # 1. Start LiveKit SFU (dev mode — no config needed)
  docker run --rm -p 7880:7880 livekit/livekit-server --dev
  # 2. Start the backend
  python server.py
  # 3. Open the new React frontend at http://localhost:5173
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse, FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── WhisperX / faster-whisper ────────────────────────────────────────────────
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    from faster_whisper import WhisperModel

# ── Module imports ─────────────────────────────────────────────────────────────
import lkc_graph
import vision
import dialogue
import lkc_retrieval
import eval_metrics
import capture
import privacy
import livekit_rooms                         # Month 6
from config import cfg
from dialogue import ConvMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE           = cfg.vad.sample_rate
SILENCE_THRESHOLD     = cfg.vad.silence_threshold
VAD_SILENCE_CHUNKS    = cfg.vad.silence_chunks
MAX_SEGMENT_CHUNKS    = cfg.vad.max_segment_chunks
VISION_FRAME_INTERVAL = cfg.vision.frame_interval
LKC_LOG               = Path(cfg.lkc.log_file)

lkc_graph.configure(db_path=LKC_LOG.with_suffix(".db"), jsonl_path=LKC_LOG)
capture.set_lkc_path(LKC_LOG)

# ── ASR model ──────────────────────────────────────────────────────────────────
if WHISPERX_AVAILABLE:
    log.info(f"Loading WhisperX '{cfg.whisper.model_size}' on {cfg.whisper.device}…")
    _wx_model = whisperx.load_model(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        language=cfg.whisper.language,
    )
    _wx_align_model: dict = {}
    log.info("WhisperX ready.")
else:
    log.warning("whisperx not installed — falling back to faster-whisper")
    _fw_model = WhisperModel(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        cpu_threads=cfg.whisper.cpu_threads,
        num_workers=cfg.whisper.num_workers,
    )
    log.info("faster-whisper ready.")


# ── LKC helpers ────────────────────────────────────────────────────────────────
def write_to_lkc(record: dict) -> None:
    lkc_graph.write_to_lkc(record)

def lkc_vision_record(session_id, ts, scene_summary, present_speakers, engagement_cues, environment_state=None):
    rec = {
        "type":             "vision",
        "session_id":       session_id,
        "timestamp_iso":    datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix":   round(ts, 3),
        "scene_summary":    scene_summary,
        "present_speakers": present_speakers,
        "engagement_cues":  engagement_cues,
    }
    if environment_state:
        rec["environment_state"] = environment_state
    return rec

def lkc_agent_record(session_id, ts, reply_text, mode):
    return {
        "type":           "agent_reply",
        "session_id":     session_id,
        "timestamp_iso":  datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "text":           reply_text,
        "mode":           mode,
    }


# ── VAD chunker ────────────────────────────────────────────────────────────────
class VadChunker:
    def __init__(self):
        self.buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.chunk_count  = 0

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def push(self, pcm: np.ndarray) -> np.ndarray | None:
        self.buffer.append(pcm)
        self.chunk_count += 1
        is_silent = self.rms(pcm) < SILENCE_THRESHOLD
        self.silent_count = self.silent_count + 1 if is_silent else 0
        should_flush = (
            self.silent_count >= VAD_SILENCE_CHUNKS or
            self.chunk_count  >= MAX_SEGMENT_CHUNKS
        )
        if should_flush and len(self.buffer) > VAD_SILENCE_CHUNKS:
            segment = np.concatenate(self.buffer)
            self.buffer       = []
            self.silent_count = 0
            self.chunk_count  = 0
            return segment
        return None


# ── TTS relay queues (Month 5 — kept for /ws/tts backward compat) ─────────────
_tts_queues: dict[str, asyncio.Queue] = {}

def _get_tts_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _tts_queues:
        _tts_queues[session_id] = asyncio.Queue()
    return _tts_queues[session_id]


# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lab Brain — Module 5 Month 6",
    description=(
        "Multimodal Conversational Agent. "
        "Month 6: LiveKit WebRTC media layer + SSE event stream."
    ),
    version="6.0.0",
)

# CORS — "*" covers all dev origins. In production, replace with explicit
# frontend domain(s) so credentials aren't exposed to arbitrary sites.
# Vite dev server and LiveKit-served frontends on non-localhost origins must
# be listed explicitly when allow_credentials=True in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(capture.router, prefix="/capture")
app.include_router(privacy.router, prefix="/privacy")


# ═══════════════════════════════════════════════════════════════════════════════
# Month 6: LiveKit endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class RoomCreateRequest(BaseModel):
    display_name: str = "browser-user"


class RoomCreateResponse(BaseModel):
    session_id: str
    token:      str
    lk_url:     str


@app.post("/livekit/room", response_model=RoomCreateResponse)
async def livekit_create_room(req: RoomCreateRequest = None):
    """
    Create a new LiveKit room and return the session_id, a signed JWT token
    for the browser participant, and the LiveKit server URL.

    Accepts an optional JSON body:
        { "display_name": "Rio" }
    The display_name becomes the participant's identity in the video grid.
    Defaults to "browser-user" if omitted.
    """
    if req is None:
        req = RoomCreateRequest()

    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "LiveKit SDK not installed. Run: pip install livekit livekit-api"},
        )

    session_id = str(uuid.uuid4())[:8]

    try:
        await livekit_rooms.create_room(session_id)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    token = livekit_rooms.create_token(session_id, identity=req.display_name)

    # Start the backend subscriber task that drives the existing pipeline
    livekit_rooms.start_subscriber(session_id, _livekit_pipeline)

    log.info(f"[server] LiveKit room created: {session_id} host={req.display_name}")
    return RoomCreateResponse(
        session_id=session_id,
        token=token,
        lk_url=cfg.livekit.url,
    )


@app.get("/livekit/token")
async def livekit_get_token(session_id: str, identity: str = "browser-user"):
    """
    Issue a JWT for a guest joining an existing room.

    Returns 404 if the room doesn't exist — guests must verify the room is
    live before joining.  Never starts a pipeline or creates a room.
    """
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    # Guard: room must already exist (host must have called POST /livekit/room first)
    room_info = await livekit_rooms.get_room(session_id)
    if room_info is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Room '{session_id}' does not exist. Ask the host for the correct session ID."},
        )
    try:
        token = livekit_rooms.create_token(session_id, identity=identity)
        return {"session_id": session_id, "token": token, "lk_url": cfg.livekit.url}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/livekit/room/{session_id}")
async def livekit_room_status(session_id: str):
    """Return participant count and recording status for a room."""
    if not livekit_rooms.LIVEKIT_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "LiveKit SDK not installed"})
    info = await livekit_rooms.get_room(session_id)
    if info is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return info


@app.delete("/livekit/room/{session_id}")
async def livekit_delete_room(session_id: str):
    """
    End a session: stop the backend subscriber task and delete the LiveKit room.
    The frontend calls this on the Stop button after POST /summary/{sid}.
    """
    await livekit_rooms.stop_subscriber(session_id)
    deleted = await livekit_rooms.delete_room(session_id)

    # Cleanup session state (mirrors what /ws/asr disconnect did in Month 5)
    vision.clear_state(session_id)
    dialogue.clear_dialogue(session_id)
    capture.clear_summon(session_id)
    _tts_queues.pop(session_id, None)

    return {"session_id": session_id, "deleted": deleted}


# ── SSE event stream ───────────────────────────────────────────────────────────

@app.get("/events/{session_id}")
async def sse_events(session_id: str):
    """
    Server-Sent Events stream for a session.

    Replaces the three WebSocket endpoints:
      /ws/asr   → 'transcript' and 'agent_reply' events
      /ws/vision → 'perception' events
      /ws/tts   → 'speak' events (TTS text)
      (new)     → 'mode_change', 'listening', 'error' events

    Event format (all events share the outer 'data:' SSE frame):
        data: {"type": "transcript", "text": "...", ...}

    The frontend parses msg.type to route events to the right component.
    """
    return StreamingResponse(
        livekit_rooms.sse_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Month 6: LiveKit pipeline (audio + video → existing internals → SSE)
# ═══════════════════════════════════════════════════════════════════════════════

async def _livekit_pipeline(
    session_id: str,
    audio_q: asyncio.Queue,
    video_q: asyncio.Queue,
) -> None:
    """
    Coroutine started by livekit_rooms.start_subscriber().
    Consumes audio/video frames from the LiveKit subscriber and pipes them
    through the unchanged Month 5 pipeline, broadcasting results via SSE.
    """
    chunker       = VadChunker()
    dlg           = dialogue.get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever()
    metrics       = eval_metrics.get_metrics(session_id)
    segment_index = 0
    frame_counter = 0
    _known_speakers: set[str] = set()
    _wx_align_cache: dict = {}

    log.info(f"[pipeline:{session_id}] LiveKit pipeline started")

    # Announce session to SSE clients
    livekit_rooms.broadcast(session_id, {
        "type":       "session",
        "session_id": session_id,
    })

    async def _process_audio_frame(raw_frame) -> None:
        """
        Resample a LiveKit AudioFrame (48kHz int16 stereo) to 16kHz mono
        float32, then push through VadChunker → WhisperX.
        """
        nonlocal segment_index

        import av  # pip install av (pulled by whisperx)
        # Convert int16 → float32
        pcm_int16 = np.frombuffer(bytes(raw_frame.data), dtype=np.int16)
        # Mix to mono if stereo
        if raw_frame.num_channels > 1:
            pcm_int16 = pcm_int16.reshape(-1, raw_frame.num_channels).mean(axis=1)
        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0

        # Resample from LiveKit's sample rate (usually 48000) to 16000
        src_rate = raw_frame.sample_rate
        if src_rate != SAMPLE_RATE:
            import soxr  # pip install soxr
            pcm_f32 = soxr.resample(pcm_f32, src_rate, SAMPLE_RATE)

        request_received_at = time.time()
        segment_audio = chunker.push(pcm_f32)

        if segment_audio is None:
            livekit_rooms.broadcast(session_id, {
                "type":     "listening",
                "mode":     dlg.mode.value,
                "summoned": capture.is_summoned(session_id),
            })
            return

        if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
            return

        seg_start = time.time()
        segment_index += 1

        # ── ASR ───────────────────────────────────────────────────────────────
        loop = asyncio.get_event_loop()

        if WHISPERX_AVAILABLE:
            wx_result = await loop.run_in_executor(
                None,
                lambda: _wx_model.transcribe(
                    segment_audio, batch_size=8, language=cfg.whisper.language
                )
            )
            detected_lang = wx_result.get("language", cfg.whisper.language or "en")

            if detected_lang not in _wx_align_cache:
                align_model, align_meta = await loop.run_in_executor(
                    None,
                    lambda: whisperx.load_align_model(
                        language_code=detected_lang, device=cfg.whisper.device
                    )
                )
                _wx_align_cache[detected_lang] = (align_model, align_meta)
            align_model, align_meta = _wx_align_cache[detected_lang]

            aligned = await loop.run_in_executor(
                None,
                lambda: whisperx.align(
                    wx_result["segments"], align_model, align_meta,
                    segment_audio, cfg.whisper.device, return_char_alignments=False
                )
            )
            wx_segments = aligned.get("segments", wx_result.get("segments", []))
            full_text   = " ".join(s["text"].strip() for s in wx_segments).strip()

            raw_word_ts: list[dict] = []
            for seg in wx_segments:
                for w in seg.get("words", []):
                    raw_word_ts.append({
                        "word":  w.get("word", ""),
                        "start": round(w.get("start", 0.0), 3),
                        "end":   round(w.get("end",   0.0), 3),
                        "score": round(w.get("score", 1.0), 3),
                    })
        else:
            segments_iter, info = await loop.run_in_executor(
                None,
                lambda: _fw_model.transcribe(
                    segment_audio, language=cfg.whisper.language,
                    vad_filter=False, beam_size=cfg.whisper.beam_size
                )
            )
            full_text     = " ".join(seg.text for seg in segments_iter).strip()
            detected_lang = info.language
            raw_word_ts   = []

        if not full_text:
            return

        asr_latency = round((time.time() - seg_start) * 1000)
        metrics.record_asr(asr_latency)

        # ── Speaker diarization ────────────────────────────────────────────────
        perc_state = vision.get_state(session_id)
        speaker    = dialogue.assign_speaker(dlg, audio_segment=segment_audio)
        if perc_state.present_speakers:
            speaker = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]

        # ── Word-level speaker alignment ───────────────────────────────────────
        if raw_word_ts:
            word_timestamps = dialogue.assign_speaker_words(dlg, raw_word_ts, segment_audio)
        else:
            word_timestamps = raw_word_ts

        # ── Privacy ────────────────────────────────────────────────────────────
        redacted_text = privacy.redact(full_text) if privacy.check_consent(speaker) else full_text

        # ── Wake-word detection ────────────────────────────────────────────────
        summoned = capture.check_summon(session_id, full_text)

        # ── New speaker detection ──────────────────────────────────────────────
        current_known = set(perc_state.present_speakers)
        new_speakers  = list(current_known - _known_speakers)
        _known_speakers.update(current_known)

        pending_confirms     = capture.get_pending_confirmations(session_id)
        pending_confirm_text = pending_confirms[0] if pending_confirms else None

        # ── Mode FSM ──────────────────────────────────────────────────────────
        prev_mode = dlg.mode
        new_mode, entry_utterance = dialogue.update_mode(
            dlg, full_text, perc_state.present_speakers, new_speakers,
            pending_confirmation=pending_confirm_text,
            summoned=summoned,
        )
        if new_mode != prev_mode:
            metrics.record_mode_switch(new_mode.value)
            livekit_rooms.broadcast(session_id, {
                "type": "mode_change",
                "mode": new_mode.value,
            })

        # ── Write to LKC graph ─────────────────────────────────────────────────
        record = capture.process_segment(
            session_id, speaker, redacted_text,
            seg_start, new_mode.value, detected_lang,
            confirm_agent=True,
            word_timestamps=word_timestamps,
        )
        metrics.record_tags(record.get("tags", {}))
        dlg.push_context(speaker, full_text)

        e2e = round((time.time() - request_received_at) * 1000)
        metrics.record_e2e(e2e)

        log.info(
            f"[{session_id}] seg#{segment_index} "
            f"asr={asr_latency}ms e2e={e2e}ms [{detected_lang}] "
            f"mode={new_mode.value} summoned={summoned}: {full_text[:60]}"
        )

        # ── Broadcast transcript via SSE ───────────────────────────────────────
        livekit_rooms.broadcast(session_id, {
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

        # ── Entry utterance (greeting / confirmation) ──────────────────────────
        if entry_utterance:
            write_to_lkc(lkc_agent_record(session_id, time.time(), entry_utterance, new_mode.value))
            _get_tts_queue(session_id).put_nowait(entry_utterance)
            livekit_rooms.broadcast(session_id, {
                "type": "agent_reply",
                "text": entry_utterance,
                "mode": new_mode.value,
            })
            livekit_rooms.broadcast(session_id, {
                "type": "speak",
                "text": entry_utterance,
            })

        # ── QA mode ──────────────────────────────────────────────────────────
        if new_mode == ConvMode.QA:
            asyncio.create_task(
                _handle_qa_sse(session_id, dlg, full_text, retriever, metrics, new_mode)
            )

    async def _process_video_frame(jpeg_bytes: bytes) -> None:
        """Pipe a JPEG frame through the existing vision pipeline, broadcast via SSE."""
        nonlocal frame_counter
        frame_counter += 1

        if frame_counter % VISION_FRAME_INTERVAL != 0:
            return

        t0    = time.time()
        state = await vision.analyse_frame(session_id, jpeg_bytes)
        latency_ms = round((time.time() - t0) * 1000)

        m = eval_metrics.get_metrics(session_id)
        m.record_vision(
            latency_ms,
            ok=not bool(state.error_count and state.frame_count == state.error_count),
            stub=not vision.GEMINI_AVAILABLE,
        )
        env_valid = bool(
            state.environment_state.get("layout") not in (None, "unknown") or
            state.environment_state.get("objects")
        )
        m.record_environment(env_valid)

        stub_marker = "[Vision stub"
        if state.scene_summary and not state.scene_summary.startswith(stub_marker):
            write_to_lkc(lkc_vision_record(
                session_id, time.time(),
                state.scene_summary,
                state.present_speakers,
                state.engagement_cues,
                state.environment_state,
            ))

        livekit_rooms.broadcast(session_id, {
            "type":              "perception",
            "session_id":        session_id,
            "present_speakers":  state.present_speakers,
            "engagement_cues":   state.engagement_cues,
            "scene_summary":     state.scene_summary,
            "environment_state": state.environment_state,
            "latency_ms":        latency_ms,
        })

    # ── Main pipeline loop ─────────────────────────────────────────────────────
    try:
        while True:
            # Drain whichever queue has data first; audio has priority
            done, _ = await asyncio.wait(
                [
                    asyncio.ensure_future(audio_q.get()),
                    asyncio.ensure_future(video_q.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for future in done:
                item = future.result()
                if isinstance(item, bytes):
                    # JPEG video frame
                    await _process_video_frame(item)
                else:
                    # LiveKit AudioFrame
                    await _process_audio_frame(item)

            # Cancel pending futures to avoid leaking tasks
            for future in _ :
                future.cancel()

    except asyncio.CancelledError:
        log.info(f"[pipeline:{session_id}] pipeline cancelled")
        raise


async def _handle_qa_sse(session_id, dlg, full_text, retriever, metrics, mode):
    """QA reply coroutine for the LiveKit pipeline — broadcasts via SSE."""
    lkc_context = retriever.query(full_text, top_k=cfg.lkc.retrieval_top_k, session_id=session_id)
    reply = await dialogue.generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        write_to_lkc(lkc_agent_record(session_id, ts, reply, mode.value))
        _get_tts_queue(session_id).put_nowait(reply)
        livekit_rooms.broadcast(session_id, {
            "type":     "agent_reply",
            "text":     reply,
            "mode":     mode.value,
            "grounded": bool(lkc_context.strip()),
        })
        # Separate 'speak' event so the frontend TTS handler is unambiguous
        livekit_rooms.broadcast(session_id, {
            "type": "speak",
            "text": reply,
        })
    capture.clear_summon(session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# All Month 5 REST endpoints — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.get("/lkc", response_class=HTMLResponse)
async def lkc_viewer():
    records = lkc_graph.read_lkc(limit=500)
    return HTMLResponse(
        f"<pre style='font-family:monospace;font-size:13px'>"
        f"{json.dumps(records, indent=2, ensure_ascii=False)}</pre>"
    )

@app.delete("/lkc")
async def clear_lkc():
    count = lkc_graph.clear_all()
    return {"cleared": True, "records_deleted": count}

@app.get("/lkc/stats")
async def lkc_stats():
    return lkc_graph.graph_stats()

@app.get("/lkc/sessions")
async def list_sessions():
    return {"sessions": lkc_graph.read_sessions()}

@app.get("/lkc/sessions/{session_id}")
async def get_session_records(
    session_id: str,
    record_type: str | None = Query(default=None),
    since_unix:  float | None = Query(default=None),
    limit:       int          = Query(default=200, le=2000),
):
    records = lkc_graph.read_lkc(
        session_id=session_id, record_type=record_type,
        since_unix=since_unix, limit=limit,
    )
    return {"session_id": session_id, "count": len(records), "records": records}

@app.delete("/lkc/sessions/{session_id}")
async def delete_session(session_id: str):
    count = lkc_graph.clear_session(session_id)
    return {"session_id": session_id, "records_deleted": count}

@app.get("/agent/summon/{session_id}")
async def get_summon_status(session_id: str):
    return {"session_id": session_id, "summoned": capture.is_summoned(session_id)}

@app.post("/agent/summon/{session_id}")
async def manual_summon(session_id: str):
    capture._summon_state[session_id] = True
    log.info(f"[server] Manual summon: {session_id}")
    return {"session_id": session_id, "summoned": True}

@app.delete("/agent/summon/{session_id}")
async def clear_summon(session_id: str):
    capture.clear_summon(session_id)
    return {"session_id": session_id, "summoned": False}

@app.get("/config/client")
async def client_config():
    return {
        "camera_fps":       cfg.vision.camera_fps,
        "camera_quality":   cfg.vision.camera_quality,
        "tts_auto_hide_ms": cfg.dialogue.tts_auto_hide_ms,
        "lk_url":           cfg.livekit.url,    # Month 6: expose SFU URL to client
    }

@app.get("/mode/{session_id}")
async def get_mode(session_id: str):
    state = dialogue.get_dialogue(session_id)
    return {"session_id": session_id, "mode": state.mode.value,
            "summoned": capture.is_summoned(session_id)}

@app.get("/perception/{session_id}")
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

@app.get("/metrics")
async def get_metrics():
    return JSONResponse(eval_metrics.all_summaries())

@app.get("/metrics/csv", response_class=PlainTextResponse)
async def get_metrics_csv():
    return PlainTextResponse(
        eval_metrics.all_csv(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=m6_metrics.csv"},
    )

class WERRequest(BaseModel):
    session_id: str
    reference:  str
    hypothesis: str

@app.post("/eval/reference")
async def post_wer_reference(req: WERRequest):
    m   = eval_metrics.get_metrics(req.session_id)
    wer = m.record_wer(req.reference, req.hypothesis)
    return {"session_id": req.session_id, "wer": round(wer, 4)}

@app.get("/ner/status")
async def ner_status():
    capture._load_spacy()
    return {
        "backend":         "spacy_en_core_web_sm" if capture.SPACY_AVAILABLE else "regex_fallback",
        "spacy_available": capture.SPACY_AVAILABLE,
    }

@app.get("/retrieval/stats")
async def retrieval_stats():
    retriever = lkc_retrieval.get_retriever()
    return retriever.stats()

@app.post("/summary/{session_id}")
async def post_summary(session_id: str):
    dlg_state = dialogue.get_dialogue(session_id)
    records   = lkc_graph.read_lkc(session_id=session_id, record_type="transcript")
    tags: dict = {"action_items": [], "decisions": [], "deadlines": [], "entities": []}
    for r in records:
        t = r.get("tags", {})
        tags["action_items"].extend(t.get("action_items", []))
        tags["decisions"].extend(t.get("decisions", []))
        tags["deadlines"].extend(t.get("deadlines", []))
        tags["entities"].extend(t.get("entities", []))
    tags["entities"] = sorted(set(tags["entities"]))

    summary_md = await dialogue.generate_summary(dlg_state, tags)
    write_to_lkc({
        "type":          "session_summary",
        "session_id":    session_id,
        "timestamp_iso": datetime.utcnow().isoformat() + "Z",
        "summary":       summary_md,
        "tags":          tags,
    })
    return {"session_id": session_id, "summary": summary_md}


# ═══════════════════════════════════════════════════════════════════════════════
# Month 5 WebSocket endpoints — KEPT for backward compat (index.html / Rifqi)
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/asr")
async def asr_endpoint(ws: WebSocket):
    """
    Retained from Month 5 for backward compatibility with the existing
    index.html client and any scripts that send raw PCM directly.
    New React frontend uses POST /livekit/room + SSE /events/{sid} instead.
    """
    await ws.accept()
    session_id    = str(uuid.uuid4())[:8]
    chunker       = VadChunker()
    dlg           = dialogue.get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever()
    metrics       = eval_metrics.get_metrics(session_id)
    segment_index = 0
    _wx_align_cache_ws: dict = {}
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
                    "summoned": capture.is_summoned(session_id),
                })
                continue

            if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
                continue

            seg_start = time.time()
            segment_index += 1
            loop = asyncio.get_event_loop()

            if WHISPERX_AVAILABLE:
                wx_result = await loop.run_in_executor(
                    None, lambda: _wx_model.transcribe(
                        segment_audio, batch_size=8, language=cfg.whisper.language
                    )
                )
                detected_lang = wx_result.get("language", cfg.whisper.language or "en")
                if detected_lang not in _wx_align_cache_ws:
                    align_model, align_meta = await loop.run_in_executor(
                        None, lambda: whisperx.load_align_model(
                            language_code=detected_lang, device=cfg.whisper.device
                        )
                    )
                    _wx_align_cache_ws[detected_lang] = (align_model, align_meta)
                align_model, align_meta = _wx_align_cache_ws[detected_lang]
                aligned = await loop.run_in_executor(
                    None, lambda: whisperx.align(
                        wx_result["segments"], align_model, align_meta,
                        segment_audio, cfg.whisper.device, return_char_alignments=False
                    )
                )
                wx_segments = aligned.get("segments", wx_result.get("segments", []))
                full_text   = " ".join(s["text"].strip() for s in wx_segments).strip()
                raw_word_ts = []
                for seg in wx_segments:
                    for w in seg.get("words", []):
                        raw_word_ts.append({
                            "word": w.get("word",""), "start": round(w.get("start",0.),3),
                            "end": round(w.get("end",0.),3), "score": round(w.get("score",1.),3),
                        })
            else:
                segments_iter, info = await loop.run_in_executor(
                    None, lambda: _fw_model.transcribe(
                        segment_audio, language=cfg.whisper.language,
                        vad_filter=False, beam_size=cfg.whisper.beam_size
                    )
                )
                full_text     = " ".join(seg.text for seg in segments_iter).strip()
                detected_lang = info.language
                raw_word_ts   = []

            if not full_text:
                continue

            asr_latency = round((time.time() - seg_start) * 1000)
            metrics.record_asr(asr_latency)

            perc_state = vision.get_state(session_id)
            speaker    = dialogue.assign_speaker(dlg, audio_segment=segment_audio)
            if perc_state.present_speakers:
                speaker = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]

            word_timestamps = (
                dialogue.assign_speaker_words(dlg, raw_word_ts, segment_audio)
                if raw_word_ts else raw_word_ts
            )
            redacted_text = privacy.redact(full_text) if privacy.check_consent(speaker) else full_text
            summoned      = capture.check_summon(session_id, full_text)

            current_known = set(perc_state.present_speakers)
            new_speakers  = list(current_known - _known_speakers)
            _known_speakers.update(current_known)

            pending_confirms     = capture.get_pending_confirmations(session_id)
            pending_confirm_text = pending_confirms[0] if pending_confirms else None

            prev_mode = dlg.mode
            new_mode, entry_utterance = dialogue.update_mode(
                dlg, full_text, perc_state.present_speakers, new_speakers,
                pending_confirmation=pending_confirm_text, summoned=summoned,
            )
            if new_mode != prev_mode:
                metrics.record_mode_switch(new_mode.value)

            record = capture.process_segment(
                session_id, speaker, redacted_text, seg_start,
                new_mode.value, detected_lang, confirm_agent=True,
                word_timestamps=word_timestamps,
            )
            metrics.record_tags(record.get("tags", {}))
            dlg.push_context(speaker, full_text)

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
                write_to_lkc(lkc_agent_record(session_id, time.time(), entry_utterance, new_mode.value))
                _get_tts_queue(session_id).put_nowait(entry_utterance)
                await ws.send_json({"type": "agent_reply", "text": entry_utterance, "mode": new_mode.value})

            if new_mode == ConvMode.QA:
                asyncio.create_task(
                    _handle_qa(session_id, ws, dlg, full_text, retriever, metrics, new_mode)
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
        dialogue.clear_dialogue(session_id)
        capture.clear_summon(session_id)
        _tts_queues.pop(session_id, None)


async def _handle_qa(session_id, ws, dlg, full_text, retriever, metrics, mode):
    """QA reply for the WebSocket compat path."""
    lkc_context = retriever.query(full_text, top_k=cfg.lkc.retrieval_top_k, session_id=session_id)
    reply = await dialogue.generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        write_to_lkc(lkc_agent_record(session_id, ts, reply, mode.value))
        _get_tts_queue(session_id).put_nowait(reply)
        try:
            await ws.send_json({"type": "agent_reply", "text": reply,
                                "mode": mode.value, "grounded": bool(lkc_context.strip())})
        except Exception:
            pass
    capture.clear_summon(session_id)


@app.websocket("/ws/vision")
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
            m.record_vision(latency_ms,
                ok=not bool(state.error_count and state.frame_count == state.error_count),
                stub=not vision.GEMINI_AVAILABLE)
            env_valid = bool(
                state.environment_state.get("layout") not in (None, "unknown") or
                state.environment_state.get("objects")
            )
            m.record_environment(env_valid)
            stub_marker = "[Vision stub"
            if state.scene_summary and not state.scene_summary.startswith(stub_marker):
                write_to_lkc(lkc_vision_record(
                    session_id, time.time(), state.scene_summary,
                    state.present_speakers, state.engagement_cues, state.environment_state,
                ))
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


@app.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id")
    if not session_id:
        await ws.close(code=1008)
        return
    queue = _get_tts_queue(session_id)
    log.info(f"TTS relay (WS compat) {session_id} connected")
    try:
        while True:
            text = await queue.get()
            await ws.send_json({"type": "speak", "text": text})
    except WebSocketDisconnect:
        log.info(f"TTS relay (WS) {session_id} disconnected")
    except Exception as e:
        log.error(f"TTS relay (WS) {session_id} error: {e}", exc_info=True)


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host=cfg.server.host, port=cfg.server.port, reload=False)
