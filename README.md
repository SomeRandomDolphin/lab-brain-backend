# Streaming ASR PoC (Month 1)

Real-time speech-to-text client with LKC ingestion.
Part of **Lab Brain / TEEP 2026**.

## What this does

- Opens a WebSocket between your browser and the server
- Streams raw 16kHz PCM audio from your microphone in 500ms chunks
- Applies a simple energy-based VAD to detect speech segments
- Runs **faster-whisper** (turbo model, int8) on each segment
- Tags every segment with `session_id`, `speaker` placeholder, `timestamp`, `language`
- Writes each segment to `lkc_stream.jsonl` — simulating LKC ingestion
- Displays a live transcript in the browser with speaker tags and latency info

## Architecture

```
Browser mic
  └─ ScriptProcessor (500ms PCM chunks)
       └─ WebSocket (raw float32 PCM)
            └─ FastAPI server
                 ├─ VadChunker (energy VAD)
                 └─ faster-whisper (int8 CPU inference)
                      └─ lkc_stream.jsonl  ← simulated LKC write
```

## Setup

No system-level audio library needed. Mic capture runs entirely in the browser
via the Web Audio API — the server only receives PCM over WebSocket.

```bash
# 1. Install Python packages (works on Windows, macOS, Linux — no extra setup)
pip install -r requirements.txt

# 2. Run server (downloads ~75MB whisper model on first run)
python server.py

# 3. Open browser at http://localhost:8000
```

> **Windows note:** If you hit a `RuntimeError` about the asyncio event loop,
> add these two lines at the very top of `server.py`, before the other imports:
> ```python
> import asyncio, sys
> if sys.platform == "win32":
>     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
> ```

### Optional: server-side mic capture (not needed for this PoC)

If you later want to record audio directly on the server (e.g. a headless CLI tool),
use `sounddevice` — it bundles its own PortAudio binary on Windows and macOS, so
no separate system install is required:

```bash
pip install sounddevice
```

Avoid `pyaudio` on Windows — it requires a manual PortAudio build or a fragile
unofficial wheel from an external source.

## LKC output format

Each transcribed segment is appended to `lkc_stream.jsonl`:

```json
{
  "session_id": "a3f2b1c0",
  "timestamp_iso": "2026-06-03T08:14:22.831Z",
  "timestamp_unix": 1748938462.831,
  "speaker": "SPEAKER_00",
  "text": "The knowledge graph needs to link back to the paper entities."
}
```

You can view the live LKC log at `http://localhost:8000/lkc`.

## Month 2 extensions (next steps)

1. **Speaker diarization** — swap `SPEAKER_00` placeholder with real speaker IDs
   from `pyannote.audio`. Run diarization on each VAD segment.

2. **Word-level timestamps** — use `whisperx` instead of `faster-whisper` directly
   for sub-word alignment (needed for linking entities to exact moments).

3. **Entity extraction** — after transcription, run an LLM over each segment to
   extract named entities, decisions, and action items, then link them to LKC graph nodes.

4. **VLM channel** — add a second WebSocket or embed in the same frame for
   camera frames → Gemini Live-style face/presence detection.

## Model size vs speed tradeoff

| Model  | Size   | WER    | Approx. CPU latency |
|--------|--------|--------|---------------------|
| tiny   | ~75MB  | ~10%   | ~300ms              |
| base   | ~145MB | ~7%    | ~600ms              |
| small  | ~466MB | ~4%    | ~1.5s               |
| medium | ~1.5GB | ~3%    | ~4s                 |

For Month 1 PoC, `tiny` is fine. Upgrade to `small` when accuracy matters.
Change `MODEL_SIZE` in `server.py`.