"""
Module 5 — Month 4 Server
Extends Month 3 (Rifqi ingest + autonomous capture + CONFIRMATION mode) with:

  UPD WhisperX replaces faster-whisper for word-level timestamps and
      diarization alignment — `segments_iter` now carries per-word
      start/end/speaker fields which are written to the LKC record.
  UPD assign_speaker() in dialogue.py now calls pyannote.audio 3.3 when
      available and falls back to round-robin when not.
  UPD lkc_retrieval.py uses sentence-transformers dense embeddings
      instead of TF-IDF; TF-IDF is kept as a fallback.

Month 3 features retained unchanged:
  /capture/*       — Rifqi (Module 2) ingest + autonomous tag endpoints
  /privacy/*       — consent registration and PII gating
  /summary/{sid}   — POST end-of-session LLM summary
  /perception      — includes environment_state (Physical AI)
  /ws/asr          — CONFIRMATION mode, capture.py tagging, privacy redact,
                     diarization (now real pyannote)
  /ws/vision       — environment_state forwarded to UI and LKC

Architecture (Month 4):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Browser                                                            │
  │  ├─ mic PCM → /ws/asr → VAD → WhisperX → transcript + word timestamps│
  │  │                              ↓                                  │
  │  │                    capture.py (auto-tag)                         │
  │  │                    privacy.py (redact PII + consent gate)        │
  │  │                    dialogue.py (mode FSM + pyannote diarization) │
  │  │                    ├─ LKC retrieval (sentence-transformers)      │
  │  │                    └─ LLM reply → /ws/tts                        │
  │  │                                                                  │
  │  ├─ camera JPEG → /ws/vision → vision.py (Physical AI + gating)    │
  │  │                              → PerceptionState + environment     │
  │  │                              → LKC entry (scene + environment)   │
  │  │                                                                  │
  │  └─ Rifqi Module 2 → POST /capture/ingest → LKC (shared store)     │
  └─────────────────────────────────────────────────────────────────────┘

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
from pydantic import BaseModel

# ── Month 4: WhisperX replaces faster-whisper ────────────────────────────────
# WhisperX adds word-level timestamps and optional diarization alignment on
# top of faster-whisper's CTranslate2 backend.  We fall back to
# faster-whisper if whisperx is not installed so the server still starts.
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    from faster_whisper import WhisperModel   # noqa: F401 — used in fallback branch

# Month 2 modules
import vision
import dialogue
import lkc_retrieval
import eval_metrics
from config import cfg
from dialogue import ConvMode

# Month 3 modules
import capture
import privacy
capture.set_lkc_path(Path(cfg.lkc.log_file))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE           = cfg.vad.sample_rate
SILENCE_THRESHOLD     = cfg.vad.silence_threshold
VAD_SILENCE_CHUNKS    = cfg.vad.silence_chunks
MAX_SEGMENT_CHUNKS    = cfg.vad.max_segment_chunks
VISION_FRAME_INTERVAL = cfg.vision.frame_interval
LKC_LOG               = Path(cfg.lkc.log_file)

# ── Load ASR model once ───────────────────────────────────────────────────────
# WhisperX path: loads the CTranslate2 model via whisperx.load_model(), then
# separately loads the alignment model so we get per-word start/end times.
# faster-whisper path: unchanged from Month 3 (fallback).
if WHISPERX_AVAILABLE:
    log.info(f"Loading WhisperX '{cfg.whisper.model_size}' on {cfg.whisper.device}…")
    _wx_model = whisperx.load_model(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        language=cfg.whisper.language,
    )
    # Alignment model is language-specific; loaded lazily in the ASR path
    # because whisperx needs to know the detected language first.
    _wx_align_model: dict = {}   # cache: lang → (model, metadata)
    log.info("WhisperX ready.")
else:
    log.warning("whisperx not installed — falling back to faster-whisper (no word timestamps)")
    _fw_model = WhisperModel(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        cpu_threads=cfg.whisper.cpu_threads,
        num_workers=cfg.whisper.num_workers,
    )
    log.info("faster-whisper ready.")

# ── LKC helpers ───────────────────────────────────────────────────────────────
def write_to_lkc(record: dict) -> None:
    with LKC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        rec["environment_state"] = environment_state   # Month 3: Physical AI
    return rec

def lkc_agent_record(session_id, ts, reply_text, mode):
    return {
        "type":          "agent_reply",
        "session_id":    session_id,
        "timestamp_iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
        "timestamp_unix": round(ts, 3),
        "text":          reply_text,
        "mode":          mode,
    }

# ── VAD chunker ───────────────────────────────────────────────────────────────
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
_tts_queues: dict[str, asyncio.Queue] = {}

def _get_tts_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _tts_queues:
        _tts_queues[session_id] = asyncio.Queue()
    return _tts_queues[session_id]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Module 5 Month 3 — Multimodal Conversational Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Month 3 routers
app.include_router(capture.router, prefix="/capture")
app.include_router(privacy.router, prefix="/privacy")

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

# ── Status / config endpoints ─────────────────────────────────────────────────
@app.get("/config/client")
async def client_config():
    return {
        "camera_fps":       cfg.vision.camera_fps,
        "camera_quality":   cfg.vision.camera_quality,
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
        "session_id":        session_id,
        "present_speakers":  state.present_speakers,
        "engagement_cues":   state.engagement_cues,
        "scene_summary":     state.scene_summary,
        "environment_state": state.environment_state,   # Month 3
        "last_updated":      state.last_updated,
        "frame_count":       state.frame_count,
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
    reference:  str
    hypothesis: str

@app.post("/eval/reference")
async def post_wer_reference(req: WERRequest):
    m = eval_metrics.get_metrics(req.session_id)
    wer = m.record_wer(req.reference, req.hypothesis)
    return {"session_id": req.session_id, "wer": round(wer, 4)}

# ── Month 3: End-of-session summary ──────────────────────────────────────────
@app.post("/summary/{session_id}")
async def post_summary(session_id: str):
    """
    Generate an LLM end-of-session summary combining transcript context,
    captured tags (action items, decisions), and key entities.
    """
    dlg_state = dialogue.get_dialogue(session_id)
    # Fetch tags from LKC
    tags = {
        "action_items": [],
        "decisions":    [],
        "deadlines":    [],
        "entities":     [],
    }
    if LKC_LOG.exists():
        for line in LKC_LOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("session_id") != session_id:
                continue
            t = r.get("tags", {})
            tags["action_items"].extend(t.get("action_items", []))
            tags["decisions"].extend(t.get("decisions", []))
            tags["deadlines"].extend(t.get("deadlines", []))
            tags["entities"].extend(t.get("entities", []))

    # Deduplicate entities
    tags["entities"] = sorted(set(tags["entities"]))

    summary_md = await dialogue.generate_summary(dlg_state, tags)
    # Write summary to LKC
    write_to_lkc({
        "type":          "session_summary",
        "session_id":    session_id,
        "timestamp_iso": datetime.utcnow().isoformat() + "Z",
        "summary":       summary_md,
        "tags":          tags,
    })
    return {"session_id": session_id, "summary": summary_md}

# ── WebSocket: ASR ────────────────────────────────────────────────────────────
@app.websocket("/ws/asr")
async def asr_endpoint(ws: WebSocket):
    await ws.accept()
    session_id    = str(uuid.uuid4())[:8]
    chunker       = VadChunker()
    dlg           = dialogue.get_dialogue(session_id)
    retriever     = lkc_retrieval.get_retriever(LKC_LOG)
    metrics       = eval_metrics.get_metrics(session_id)
    segment_index = 0
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

            if WHISPERX_AVAILABLE:
                # WhisperX transcribe returns a dict with "segments" (list) and
                # "language".  Each segment has "text", "start", "end", and
                # after alignment a "words" list with per-word timestamps.
                wx_result = await loop.run_in_executor(
                    None,
                    lambda: _wx_model.transcribe(
                        segment_audio,
                        batch_size=8,
                        language=cfg.whisper.language,
                    )
                )
                detected_lang = wx_result.get("language", cfg.whisper.language or "en")

                # Load (or reuse cached) alignment model for this language
                if detected_lang not in _wx_align_model:
                    align_model, align_meta = await loop.run_in_executor(
                        None,
                        lambda: whisperx.load_align_model(
                            language_code=detected_lang,
                            device=cfg.whisper.device,
                        )
                    )
                    _wx_align_model[detected_lang] = (align_model, align_meta)
                align_model, align_meta = _wx_align_model[detected_lang]

                aligned = await loop.run_in_executor(
                    None,
                    lambda: whisperx.align(
                        wx_result["segments"],
                        align_model,
                        align_meta,
                        segment_audio,
                        cfg.whisper.device,
                        return_char_alignments=False,
                    )
                )

                wx_segments = aligned.get("segments", wx_result.get("segments", []))
                full_text   = " ".join(s["text"].strip() for s in wx_segments).strip()
                # Word-level timestamps for LKC enrichment
                word_timestamps: list[dict] = []
                for seg in wx_segments:
                    for w in seg.get("words", []):
                        word_timestamps.append({
                            "word":  w.get("word", ""),
                            "start": round(w.get("start", 0.0), 3),
                            "end":   round(w.get("end", 0.0), 3),
                            "score": round(w.get("score", 1.0), 3),
                        })
            else:
                # faster-whisper fallback — no word timestamps
                segments_iter, info = await loop.run_in_executor(
                    None,
                    lambda: _fw_model.transcribe(
                        segment_audio,
                        language=cfg.whisper.language,
                        vad_filter=False,
                        beam_size=cfg.whisper.beam_size,
                    )
                )
                full_text     = " ".join(seg.text for seg in segments_iter).strip()
                detected_lang = info.language
                word_timestamps = []

            if not full_text:
                continue

            asr_latency = round((time.time() - seg_start) * 1000)
            metrics.record_asr(asr_latency)

            # ── Month 4: Speaker diarization (pyannote 3.3 via dialogue.assign_speaker) ──
            perc_state = vision.get_state(session_id)
            # Pass segment_audio so pyannote can diarize; falls back to
            # round-robin automatically if pyannote is unavailable.
            speaker = dialogue.assign_speaker(dlg, audio_segment=segment_audio)
            # Override with VLM label if available and consented
            if perc_state.present_speakers:
                raw_sp = perc_state.present_speakers[segment_index % len(perc_state.present_speakers)]
                speaker = raw_sp  # already privacy-gated in vision.py

            # ── Month 3: Privacy — redact PII before LKC write ───────────────
            redacted_text = privacy.redact(full_text) if privacy.check_consent(speaker) else full_text

            # ── Month 3: Autonomous capture + tagging ─────────────────────────
            # Only write to LKC via capture.process_segment (replaces bare lkc write)
            current_known = set(perc_state.present_speakers)
            new_speakers  = list(current_known - _known_speakers)
            _known_speakers = current_known

            # Pull any pending confirmations queued by earlier segments
            pending_confirms = capture.get_pending_confirmations(session_id)
            pending_confirm_text = pending_confirms[0] if pending_confirms else None

            # ── Dialogue mode FSM ─────────────────────────────────────────────
            prev_mode = dlg.mode
            new_mode, entry_utterance = dialogue.update_mode(
                dlg, full_text, perc_state.present_speakers, new_speakers,
                pending_confirmation=pending_confirm_text,
            )
            if new_mode != prev_mode:
                metrics.record_mode_switch(new_mode.value)

            # Write enriched LKC record (capture.py handles the write + tagging)
            record = capture.process_segment(
                session_id, speaker, redacted_text,
                seg_start, new_mode.value, detected_lang,
                confirm_agent=True,
            )
            # Month 4: attach word-level timestamps to the LKC record
            if word_timestamps:
                record["word_timestamps"] = word_timestamps

            # Month 3: record capture tag counts in metrics
            metrics.record_tags(record.get("tags", {}))

            dlg.push_context(speaker, full_text)

            e2e = round((time.time() - request_received_at) * 1000)
            metrics.record_e2e(e2e)

            log.info(
                f"[{session_id}] seg#{segment_index} "
                f"asr={asr_latency}ms e2e={e2e}ms [{detected_lang}] "
                f"mode={new_mode.value} tags={bool(capture.has_tags(record['tags']))}"
                f" words={len(word_timestamps)}: {full_text[:60]}"
            )

            # ── Send transcript to UI ─────────────────────────────────────────
            await ws.send_json({
                "type":             "transcript",
                "segment":          segment_index,
                "session_id":       session_id,
                "speaker":          speaker,
                "text":             full_text,
                "language":         detected_lang,
                "latency_ms":       asr_latency,
                "e2e_ms":           e2e,
                "mode":             new_mode.value,
                "timestamp":        record["timestamp_iso"],
                "engagement":       perc_state.engagement_cues.get(speaker, "unknown"),
                # Month 3 additions
                "tags":             record.get("tags", {}),
                "environment":      perc_state.environment_state,
                # Month 4: word-level timestamps from WhisperX
                "word_timestamps":  word_timestamps,
            })

            # ── Handle entry utterance (mode greeting / confirmation) ─────────
            if entry_utterance:
                write_to_lkc(lkc_agent_record(session_id, time.time(), entry_utterance, new_mode.value))
                _get_tts_queue(session_id).put_nowait(entry_utterance)
                await ws.send_json({
                    "type": "agent_reply",
                    "text": entry_utterance,
                    "mode": new_mode.value,
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
    lkc_context = retriever.query(full_text, top_k=cfg.lkc.retrieval_top_k)
    reply = await dialogue.generate_response(dlg, full_text, lkc_context)
    if reply:
        ts = time.time()
        write_to_lkc(lkc_agent_record(session_id, ts, reply, mode.value))
        _get_tts_queue(session_id).put_nowait(reply)
        try:
            await ws.send_json({
                "type":     "agent_reply",
                "text":     reply,
                "mode":     mode.value,
                "grounded": bool(lkc_context.strip()),
            })
        except Exception:
            pass


# ── WebSocket: Vision ─────────────────────────────────────────────────────────
@app.websocket("/ws/vision")
async def vision_endpoint(ws: WebSocket):
    await ws.accept()
    session_id    = ws.query_params.get("session_id", str(uuid.uuid4())[:8])
    frame_counter = 0
    log.info(f"Vision session {session_id} connected")

    try:
        while True:
            jpeg_bytes = await ws.receive_bytes()
            frame_counter += 1

            if frame_counter % VISION_FRAME_INTERVAL != 0:
                continue

            t0    = time.time()
            state = await vision.analyse_frame(session_id, jpeg_bytes)
            latency_ms = round((time.time() - t0) * 1000)

            m = eval_metrics.get_metrics(session_id)
            m.record_vision(
                latency_ms,
                ok=not bool(state.error_count and state.frame_count == state.error_count),
                stub=not vision.GEMINI_AVAILABLE,
            )
            # Month 3: track environment coverage
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
                    state.environment_state,   # Month 3
                ))

            await ws.send_json({
                "type":              "perception",
                "session_id":        session_id,
                "present_speakers":  state.present_speakers,
                "engagement_cues":   state.engagement_cues,
                "scene_summary":     state.scene_summary,
                "environment_state": state.environment_state,  # Month 3
                "latency_ms":        latency_ms,
            })

    except WebSocketDisconnect:
        log.info(f"Vision session {session_id} disconnected")
    except Exception as e:
        log.error(f"Vision session {session_id} error: {e}", exc_info=True)


# ── WebSocket: TTS relay ──────────────────────────────────────────────────────
@app.websocket("/ws/tts")
async def tts_endpoint(ws: WebSocket):
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