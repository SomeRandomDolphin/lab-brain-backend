"""
Module 5 — Month 2 Server
Extends Month 1 (streaming ASR + LKC ingestion) with:

  NEW /ws/vision  — camera frame WebSocket → VLM perception (vision.py)
  NEW /ws/tts     — agent reply WebSocket  → sends spoken text to browser
                    (browser uses Web Speech API to vocalise)
  NEW /mode       — GET current conversation mode
  NEW /perception — GET latest VLM perception state
  NEW /metrics    — GET evaluation summary JSON
  NEW /metrics/csv— GET evaluation data as CSV (for thesis)
  NEW /eval/reference — POST {session_id, reference, hypothesis} for WER

Architecture (Month 2):
  ┌─────────────────────────────────────────────────────────────┐
  │  Browser                                                    │
  │  ├─ mic PCM → /ws/asr → VAD → Whisper → transcript         │
  │  │                              ↓                          │
  │  │                         dialogue.py                     │
  │  │                         ├─ mode FSM                     │
  │  │                         ├─ LKC retrieval (TF-IDF)       │
  │  │                         └─ Gemini reply → /ws/tts       │
  │  │                                                         │
  │  └─ camera JPEG → /ws/vision → vision.py (Gemini Flash)    │
  │                                → PerceptionState           │
  │                                → LKC entry (scene events)  │
  └─────────────────────────────────────────────────────────────┘

Run:
  python server.py          # reads config.json automatically
  open http://localhost:8000

Windows note: add at the very top if you hit event-loop errors:
  import asyncio, sys
  if sys.platform == "win32":
      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

# Month 2 modules
import vision
import dialogue
import lkc_retrieval
import eval_metrics
from config import cfg
from dialogue import ConvMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config (all values from config.json via cfg) ──────────────────────────────
SAMPLE_RATE           = cfg.vad.sample_rate
SILENCE_THRESHOLD     = cfg.vad.silence_threshold
VAD_SILENCE_CHUNKS    = cfg.vad.silence_chunks
MAX_SEGMENT_CHUNKS    = cfg.vad.max_segment_chunks
VISION_FRAME_INTERVAL = cfg.vision.frame_interval
LKC_LOG               = Path(cfg.lkc.log_file)

# ── Load Whisper once ─────────────────────────────────────────────────────────
log.info(f"Loading faster-whisper '{cfg.whisper.model_size}'…")
whisper_model = WhisperModel(
    cfg.whisper.model_size,
    device=cfg.whisper.device,
    compute_type=cfg.whisper.compute_type,
    cpu_threads=cfg.whisper.cpu_threads,
    num_workers=cfg.whisper.num_workers,
)
log.info("Whisper ready.")

# ── LKC helpers ───────────────────────────────────────────────────────────────
def write_to_lkc(record: dict) -> None:
    with LKC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def lkc_transcript_record(session_id, speaker, text, ts, mode, lang):
    return {
        "type": "transcript",
        "session_id": session_id,
        "timestamp_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "speaker": speaker,
        "text": text.strip(),
        "mode": mode,
        "language": lang,
    }

def lkc_vision_record(session_id, ts, scene_summary, present_speakers, engagement_cues):
    return {
        "type": "vision",
        "session_id": session_id,
        "timestamp_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "scene_summary": scene_summary,
        "present_speakers": present_speakers,
        "engagement_cues": engagement_cues,
    }

def lkc_agent_record(session_id, ts, reply_text, mode):
    return {
        "type": "agent_reply",
        "session_id": session_id,
        "timestamp_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "text": reply_text,
        "mode": mode,
    }

# ── VAD chunker (unchanged from Month 1) ─────────────────────────────────────
class VadChunker:
    def __init__(self):
        self.buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.chunk_count = 0

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def push(self, pcm: np.ndarray) -> np.ndarray | None:
        self.buffer.append(pcm)
        self.chunk_count += 1
        is_silent = self.rms(pcm) < SILENCE_THRESHOLD
        self.silent_count = self.silent_count + 1 if is_silent else 0
        should_flush = (
            self.silent_count >= VAD_SILENCE_CHUNKS or
            self.chunk_count >= MAX_SEGMENT_CHUNKS
        )
        if should_flush and len(self.buffer) > VAD_SILENCE_CHUNKS:
            segment = np.concatenate(self.buffer)
            self.buffer = []
            self.silent_count = 0
            self.chunk_count = 0
            return segment
        return None

# ── Per-session TTS reply queues ──────────────────────────────────────────────
# Maps session_id → asyncio.Queue of reply strings
_tts_queues: dict[str, asyncio.Queue] = {}

def _get_tts_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _tts_queues:
        _tts_queues[session_id] = asyncio.Queue()
    return _tts_queues[session_id]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Module 5 Month 2 — Multimodal Conversational Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Static pages ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

# ── LKC viewer ────────────────────────────────────────────────────────────────
@app.get("/lkc", response_class=HTMLResponse)
async def lkc_viewer():
    if not LKC_LOG.exists():
        return HTMLResponse("<pre>No entries yet.</pre>")
    lines = LKC_LOG.read_text().strip().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    return HTMLResponse(
        f"<pre style='font-family:monospace;font-size:13px'>"
        f"{json.dumps(records, indent=2, ensure_ascii=False)}</pre>"
    )

@app.delete("/lkc")
async def clear_lkc():
    LKC_LOG.write_text("")
    return {"cleared": True}

# ── Status endpoints (Month 2 additions) ─────────────────────────────────────
@app.get("/config/client")
async def client_config():
    """Expose browser-relevant config values so index.html doesn't hardcode them."""
    return {
        "camera_fps":      cfg.vision.camera_fps,
        "camera_quality":  cfg.vision.camera_quality,
        "tts_auto_hide_ms": cfg.dialogue.tts_auto_hide_ms,
    }

@app.get("/mode/{session_id}")
async def get_mode(session_id: str):
    state = dialogue.get_dialogue(session_id)
    return {"session_id": session_id, "mode": state.mode.value}

@app.get("/perception/{session_id}")
async def get_perception(session_id: str):
    state = vision.get_state(session_id)
    return {
        "session_id": session_id,
        "present_speakers": state.present_speakers,
        "engagement_cues": state.engagement_cues,
        "scene_summary": state.scene_summary,
        "last_updated": state.last_updated,
        "frame_count": state.frame_count,
    }

@app.get("/metrics")
async def get_metrics():
    return JSONResponse(eval_metrics.all_summaries())

@app.get("/metrics/csv", response_class=PlainTextResponse)
async def get_metrics_csv():
    return PlainTextResponse(
        eval_metrics.all_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=m5_metrics.csv"},
    )

class WERRequest(BaseModel):
    session_id: str
    reference: str
    hypothesis: str

@app.post("/eval/reference")
async def post_wer_reference(req: WERRequest):
    m = eval_metrics.get_metrics(req.session_id)
    wer = m.record_wer(req.reference, req.hypothesis)
    return {"session_id": req.session_id, "wer": round(wer, 4)}

# ── WebSocket: ASR (Month 1 core + Month 2 dialogue layer) ────────────────────
@app.websocket("/ws/asr")
async def asr_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    chunker    = VadChunker()
    dlg        = dialogue.get_dialogue(session_id)
    retriever  = lkc_retrieval.get_retriever(LKC_LOG)
    metrics    = eval_metrics.get_metrics(session_id)
    segment_index = 0

    # Known speakers from vision (updated asynchronously)
    _known_speakers: set[str] = set()

    log.info(f"ASR session {session_id} connected")
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
                await ws.send_json({"type": "listening", "mode": dlg.mode.value})
                continue

            if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
                continue

            seg_start = time.time()
            segment_index += 1

            # ── ASR ──────────────────────────────────────────────────────────
            loop = asyncio.get_event_loop()
            segments_iter, info = await loop.run_in_executor(
                None,
                lambda: whisper_model.transcribe(
                    segment_audio,
                    language=cfg.whisper.language,
                    vad_filter=False,   # VadChunker already strips silence upstream
                    beam_size=cfg.whisper.beam_size,
                )
            )
            text_parts = [seg.text for seg in segments_iter]
            full_text  = " ".join(text_parts).strip()
            if not full_text:
                continue

            asr_latency = round((time.time() - seg_start) * 1000)
            metrics.record_asr(asr_latency)

            detected_lang = info.language

            # ── Speaker label: use VLM perception if available ────────────────
            perc_state = vision.get_state(session_id)
            if perc_state.present_speakers:
                # Assign segments round-robin among detected speakers
                # (real diarisation with pyannote arrives in Month 3)
                speaker = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]
            else:
                speaker = "SPEAKER_00"

            # ── Dialogue mode FSM ─────────────────────────────────────────────
            current_known = set(perc_state.present_speakers)
            new_speakers  = list(current_known - _known_speakers)
            _known_speakers = current_known

            prev_mode = dlg.mode
            new_mode, entry_utterance = dialogue.update_mode(
                dlg, full_text, perc_state.present_speakers, new_speakers
            )
            if new_mode != prev_mode:
                metrics.record_mode_switch(new_mode.value)

            # ── LKC write ─────────────────────────────────────────────────────
            record = lkc_transcript_record(
                session_id, speaker, full_text, seg_start, new_mode.value, detected_lang
            )
            write_to_lkc(record)

            # Update dialogue context
            dlg.push_context(speaker, full_text)

            # ── E2E latency ───────────────────────────────────────────────────
            e2e = round((time.time() - request_received_at) * 1000)
            metrics.record_e2e(e2e)

            log.info(
                f"[{session_id}] seg#{segment_index} "
                f"asr={asr_latency}ms e2e={e2e}ms [{detected_lang}] "
                f"mode={new_mode.value}: {full_text[:60]}"
            )

            # ── Send transcript to UI ─────────────────────────────────────────
            await ws.send_json({
                "type":        "transcript",
                "segment":     segment_index,
                "session_id":  session_id,
                "speaker":     speaker,
                "text":        full_text,
                "language":    detected_lang,
                "latency_ms":  asr_latency,
                "e2e_ms":      e2e,
                "mode":        new_mode.value,
                "timestamp":   record["timestamp_iso"],
                "engagement":  perc_state.engagement_cues.get(speaker, "unknown"),
            })

            # ── Handle entry utterance (mode greeting) ────────────────────────
            if entry_utterance:
                lkc_rec = lkc_agent_record(session_id, time.time(), entry_utterance, new_mode.value)
                write_to_lkc(lkc_rec)
                _get_tts_queue(session_id).put_nowait(entry_utterance)
                await ws.send_json({
                    "type":  "agent_reply",
                    "text":  entry_utterance,
                    "mode":  new_mode.value,
                })

            # ── QA response (async, non-blocking) ────────────────────────────
            if new_mode == ConvMode.QA:
                asyncio.create_task(
                    _handle_qa(session_id, ws, dlg, full_text, retriever, metrics, new_mode)
                )

    except WebSocketDisconnect:
        log.info(f"ASR session {session_id} disconnected")
    except Exception as e:
        log.error(f"ASR session {session_id} error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        vision.clear_state(session_id)
        dialogue.clear_dialogue(session_id)
        _tts_queues.pop(session_id, None)


async def _handle_qa(session_id, ws, dlg, full_text, retriever, metrics, mode):
    """Generate and send a QA response without blocking the main ASR loop."""
    lkc_context = retriever.query(full_text, top_k=cfg.lkc.retrieval_top_k)
    reply = await dialogue.generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        write_to_lkc(lkc_agent_record(session_id, ts, reply, mode.value))
        _get_tts_queue(session_id).put_nowait(reply)
        try:
            await ws.send_json({
                "type":  "agent_reply",
                "text":  reply,
                "mode":  mode.value,
                "grounded": bool(lkc_context.strip()),
            })
        except Exception:
            pass


# ── WebSocket: Vision ─────────────────────────────────────────────────────────
@app.websocket("/ws/vision")
async def vision_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id", str(uuid.uuid4())[:8])
    frame_counter = 0
    log.info(f"Vision session {session_id} connected")

    try:
        while True:
            jpeg_bytes = await ws.receive_bytes()
            frame_counter += 1

            # Rate-limit: only analyse every Nth frame
            if frame_counter % VISION_FRAME_INTERVAL != 0:
                continue

            t0 = time.time()
            state = await vision.analyse_frame(session_id, jpeg_bytes)
            latency_ms = round((time.time() - t0) * 1000)

            m = eval_metrics.get_metrics(session_id)
            m.record_vision(
                latency_ms,
                ok=not bool(state.error_count and state.frame_count == state.error_count),
                stub=not vision.GEMINI_AVAILABLE,
            )

            # Write vision event to LKC (only when something meaningful is detected)
            if state.scene_summary and state.scene_summary != "[Vision stub — set api_key in config.json to enable]":
                rec = lkc_vision_record(
                    session_id, time.time(),
                    state.scene_summary,
                    state.present_speakers,
                    state.engagement_cues,
                )
                write_to_lkc(rec)

            await ws.send_json({
                "type":             "perception",
                "session_id":       session_id,
                "present_speakers": state.present_speakers,
                "engagement_cues":  state.engagement_cues,
                "scene_summary":    state.scene_summary,
                "latency_ms":       latency_ms,
            })

    except WebSocketDisconnect:
        log.info(f"Vision session {session_id} disconnected")
    except Exception as e:
        log.error(f"Vision session {session_id} error: {e}", exc_info=True)


# ── WebSocket: TTS relay ──────────────────────────────────────────────────────
@app.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket):
    """
    Long-lived connection that drains the TTS queue for a session.
    Browser listens here and speaks each reply via Web Speech API.
    """
    await ws.accept()
    session_id = ws.query_params.get("session_id")
    if not session_id:
        await ws.close(code=1008)
        return

    queue = _get_tts_queue(session_id)
    log.info(f"TTS relay {session_id} connected")
    try:
        while True:
            text = await queue.get()
            await ws.send_json({"type": "speak", "text": text})
    except WebSocketDisconnect:
        log.info(f"TTS relay {session_id} disconnected")
    except Exception as e:
        log.error(f"TTS relay {session_id} error: {e}", exc_info=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host=cfg.server.host, port=cfg.server.port, reload=False)