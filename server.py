"""
Streaming ASR Server
Month 1 deliverable: streaming transcription with speaker session tagging
into the LKC (simulated here as an append-only JSONL log).

Architecture:
  Browser mic → WebSocket (raw PCM) → VAD chunker → faster-whisper → transcript
  Each segment is tagged with: session_id, timestamp, speaker_placeholder, text
  and written to lkc_stream.jsonl (simulating LKC ingestion).

Run:
  python server.py
  open http://localhost:8000
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz — faster-whisper expects 16kHz mono float32
CHUNK_MS = 500               # ms per audio chunk from browser
SILENCE_THRESHOLD = 0.01     # RMS below this = silence
VAD_SILENCE_CHUNKS = 4       # consecutive silent chunks before flushing segment
MAX_SEGMENT_CHUNKS = 30      # flush segment after this many chunks (~15 s)
MODEL_SIZE = "small"          # tiny | base | small | medium — tiny for PoC speed
LKC_LOG = Path("lkc_stream.jsonl")

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
log.info(f"Loading faster-whisper model '{MODEL_SIZE}' (this may take a moment)...")
whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
log.info("Model ready.")

# ---------------------------------------------------------------------------
# LKC sink — append each segment as a JSON line
# ---------------------------------------------------------------------------
def write_to_lkc(session_id: str, speaker: str, text: str, start_ts: float):
    record = {
        "session_id": session_id,
        "timestamp_iso": datetime.utcfromtimestamp(start_ts).isoformat() + "Z",
        "timestamp_unix": round(start_ts, 3),
        "speaker": speaker,
        "text": text.strip(),
    }
    with LKC_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record

# ---------------------------------------------------------------------------
# VAD chunker — accumulates PCM, flushes on silence or max length
# ---------------------------------------------------------------------------
class VadChunker:
    def __init__(self):
        self.buffer: list[np.ndarray] = []
        self.silent_count = 0
        self.chunk_count = 0

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def push(self, pcm: np.ndarray) -> np.ndarray | None:
        """Push one chunk. Returns accumulated audio to transcribe, or None."""
        self.buffer.append(pcm)
        self.chunk_count += 1
        is_silent = self.rms(pcm) < SILENCE_THRESHOLD

        if is_silent:
            self.silent_count += 1
        else:
            self.silent_count = 0

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

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Streaming ASR PoC")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.get("/lkc", response_class=HTMLResponse)
async def lkc_viewer():
    """Return current LKC log as pretty JSON array."""
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

@app.websocket("/ws/asr")
async def asr_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    speaker = "SPEAKER_00"   # placeholder — pyannote diarization added in Month 2
    chunker = VadChunker()
    segment_index = 0

    log.info(f"Session {session_id} connected")
    await ws.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_bytes()

            # Browser sends raw 32-bit float PCM at 16 kHz
            pcm = np.frombuffer(data, dtype=np.float32)
            if pcm.size == 0:
                continue

            segment_audio = chunker.push(pcm)
            if segment_audio is None:
                # Send a "still listening" heartbeat so the UI knows we're alive
                await ws.send_json({"type": "listening"})
                continue

            # Skip near-silent segments (avoid transcribing breath noise)
            if float(np.sqrt(np.mean(segment_audio ** 2))) < SILENCE_THRESHOLD * 2:
                continue

            seg_start = time.time()
            segment_index += 1

            # Run faster-whisper (synchronous — offload to thread pool)
            loop = asyncio.get_event_loop()
            segments_iter, info = await loop.run_in_executor(
                None,
                lambda: whisper.transcribe(
                    segment_audio,
                    language=None,          # auto-detect
                    vad_filter=True,        # built-in silero VAD
                    beam_size=5,
                )
            )

            text_parts = []
            for seg in segments_iter:
                text_parts.append(seg.text)

            full_text = " ".join(text_parts).strip()
            if not full_text:
                continue

            detected_lang = info.language
            latency_ms = round((time.time() - seg_start) * 1000)

            # Write to LKC
            record = write_to_lkc(session_id, speaker, full_text, seg_start)
            log.info(f"[{session_id}] seg#{segment_index} ({latency_ms}ms) [{detected_lang}]: {full_text}")

            # Send transcript segment to client
            await ws.send_json({
                "type": "transcript",
                "segment": segment_index,
                "session_id": session_id,
                "speaker": speaker,
                "text": full_text,
                "language": detected_lang,
                "latency_ms": latency_ms,
                "timestamp": record["timestamp_iso"],
            })

    except WebSocketDisconnect:
        log.info(f"Session {session_id} disconnected")
    except Exception as e:
        log.error(f"Session {session_id} error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
