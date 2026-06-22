# Lab Brain — Module 5: Real-Time Multimodal Conversational Agent

Part of **TEEP 2026 / Lab Brain** — a humanistic intelligence platform for digitalising knowledge creation using the SECI model.

> **Current milestone: Month 4** — WhisperX word-level timestamps · pyannote speaker diarization · sentence-transformers dense LKC retrieval

---

## What this does

- Streams 16 kHz PCM audio from the browser mic over WebSocket
- Runs **WhisperX** (word-level timestamps + alignment, int8 CPU/GPU) with energy-based VAD; falls back to **faster-whisper** if WhisperX is not installed
- Sends camera frames to a local **LLaVA** vision model (via Ollama) for presence detection, engagement cues, and **environment/object awareness** (Physical AI)
- Runs **pyannote/speaker-diarization-3.1** to assign speaker labels from audio; falls back to round-robin if pyannote is unavailable
- Manages four conversation modes — **Greeting · Meeting Capture · Q&A · Ambient** — plus a **Confirmation** mode for autonomous tag verification
- Autonomously detects and tags **action items, decisions, deadlines, and entities** from the live transcript
- Retrieves grounded answers from in-session LKC history using **sentence-transformers** dense embeddings; falls back to TF-IDF when sentence-transformers is not installed
- Enforces **privacy & consent gating** — PII redaction, per-speaker opt-in, face anonymisation for non-consenting speakers
- Accepts structured segments from **Rifqi's Module 2** pipeline via REST ingest
- Generates an **end-of-session summary** (decisions, action items, open questions) using the local LLM
- Writes everything to `lkc_stream.jsonl` — the shared LKC store for all Lab Brain modules

---

## Architecture

```
Browser
├── Mic PCM (float32, 16kHz, 500ms chunks)
│    └── /ws/asr ──► VadChunker ──► WhisperX (word timestamps + alignment)
│                         │               └── faster-whisper fallback
│                    capture.py  ← autonomous tagger (action items, decisions)
│                    privacy.py  ← PII redaction + consent gate
│                    dialogue.py ← mode FSM (Greeting/Capture/QA/Ambient/Confirmation)
│                         │    └── pyannote diarization (round-robin fallback)
│                         ├── lkc_retrieval.py (sentence-transformers over lkc_stream.jsonl)
│                         │                    └── TF-IDF fallback
│                         └── local LLM reply (llama3.2:3b via Ollama)
│                              └── /ws/tts ──► SpeechSynthesis (browser)
│
├── Camera JPEG (1 FPS)
│    └── /ws/vision ──► vision.py ──► llava:7b via Ollama
│                            ├── present_speakers (privacy-gated)
│                            ├── engagement_cues
│                            ├── scene_summary
│                            └── environment_state  ← Physical AI
│
└── Rifqi Module 2
     └── POST /capture/ingest ──► capture.py ──► lkc_stream.jsonl
                                       │
                                  autonomous tagging
                                  (same pipeline as live ASR)

lkc_stream.jsonl  ←  shared store (transcript · vision · agent_reply · session_summary)
```

---

## Setup

No system-level audio library needed. Mic and camera run in the browser via Web Audio / MediaDevices APIs.

### 1. Start Ollama with the required models

```bash
ollama pull llava:7b        # vision model (Physical AI + presence detection)
ollama pull llama3.2:3b    # dialogue + summary model
ollama serve                # starts at http://localhost:11434
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

> **pyannote note:** `pyannote.audio` requires accepting the model licence on Hugging Face before the pipeline will download. Run `huggingface-cli login` (or set the `HF_TOKEN` environment variable) after accepting the terms at https://huggingface.co/pyannote/speaker-diarization-3.1. If you skip this, the server falls back to round-robin speaker labelling automatically.

> **GPU note:** WhisperX, pyannote, and sentence-transformers all run on CPU. A CUDA-capable GPU gives a significant speed improvement, but is not required.

### 3. Configure (optional)

Copy and edit `config.json` to override any default. All fields are optional — the server starts without it using the defaults shown below:

```json
{
  "local_llm": {
    "base_url":       "http://localhost:11434/v1",
    "vision_model":   "llava:7b",
    "dialogue_model": "llama3.2:3b"
  },
  "whisper": {
    "model_size":  "small",
    "device":      "cpu",
    "compute_type":"int8",
    "language":    null,
    "cpu_threads": 8
  },
  "vad": {
    "silence_threshold": 0.01,
    "silence_chunks":    4
  },
  "lkc": {
    "log_file":       "lkc_stream.jsonl",
    "retrieval_top_k": 4
  }
}
```

### 4. Run

```bash
python server.py
# open http://localhost:8000
```

> **Windows note:** If you hit a `RuntimeError` about the asyncio event loop,
> add at the very top of `server.py`, before other imports:
> ```python
> import asyncio, sys
> if sys.platform == "win32":
>     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
> ```

---

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI app — WebSocket endpoints for ASR (WhisperX), vision, TTS relay; REST endpoints for metrics, summary, LKC |
| `dialogue.py` | Conversation mode FSM, pyannote speaker diarization, end-of-session summary generator |
| `vision.py` | VLM perception layer — Physical AI environment sensing, privacy-gated speaker labels |
| `capture.py` | Autonomous tagger + Rifqi Module 2 ingest endpoint |
| `privacy.py` | Consent registry, PII redaction, face anonymisation gate |
| `lkc_retrieval.py` | Dense sentence-transformers retrieval (TF-IDF fallback) over `lkc_stream.jsonl` for grounded Q&A |
| `eval_metrics.py` | In-memory metrics (ASR/vision latency, WER, capture quality, confirmation rate, privacy) |
| `config.py` | Typed config loader — reads `config.json`, falls back to safe defaults |
| `static/index.html` | Single-file browser client |

---

## LKC output format

All records are appended to `lkc_stream.jsonl`. Four record types:

**Transcript segment** (Month 4 — enriched with tags, environment, and word-level timestamps)
```json
{
  "type":           "transcript",
  "session_id":     "a3f2b1c0",
  "timestamp_iso":  "2026-06-03T08:14:22Z",
  "timestamp_unix": 1748938462.831,
  "speaker":        "Person A",
  "text":           "We decided to use sentence-transformers for Month 4.",
  "mode":           "meeting_capture",
  "language":       "en",
  "tags": {
    "action_items": [],
    "decisions":    ["We decided to use sentence-transformers for Month 4"],
    "deadlines":    [],
    "entities":     ["Month 4"]
  },
  "word_timestamps": [
    {"word": "We",      "start": 0.0,  "end": 0.18, "score": 0.99},
    {"word": "decided", "start": 0.2,  "end": 0.56, "score": 0.98}
  ]
}
```

**Vision frame**
```json
{
  "type":             "vision",
  "session_id":       "a3f2b1c0",
  "timestamp_iso":    "2026-06-03T08:14:25Z",
  "scene_summary":    "Two people at a desk with a whiteboard behind them.",
  "present_speakers": ["Person A", "Person B"],
  "engagement_cues":  {"Person A": "focused", "Person B": "distracted"},
  "environment_state": {
    "objects":  ["whiteboard", "laptop", "coffee cup"],
    "layout":   "huddle",
    "lighting": "bright",
    "ambient":  "quiet"
  }
}
```

**Agent reply**
```json
{
  "type":       "agent_reply",
  "session_id": "a3f2b1c0",
  "text":       "Based on the earlier discussion, the team agreed on sentence-transformers.",
  "mode":       "qa"
}
```

**Session summary** (generated by `POST /summary/{session_id}`)
```json
{
  "type":       "session_summary",
  "session_id": "a3f2b1c0",
  "summary":    "## Summary\n- ...\n## Decisions\n- ...\n## Action Items\n- ...",
  "tags": { "action_items": [...], "decisions": [...] }
}
```

View the live LKC log at `http://localhost:8000/lkc`. Clear with `DELETE /lkc`.

---

## REST API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser client |
| `GET` | `/lkc` | LKC viewer (JSON) |
| `DELETE` | `/lkc` | Clear LKC log |
| `GET` | `/mode/{session_id}` | Current conversation mode |
| `GET` | `/perception/{session_id}` | Latest vision state (speakers, engagement, environment) |
| `GET` | `/metrics` | All session metrics (JSON) |
| `GET` | `/metrics/csv` | Download metrics as CSV |
| `POST` | `/eval/reference` | Submit reference transcript for WER calculation |
| `POST` | `/summary/{session_id}` | Generate end-of-session LLM summary |
| `GET` | `/config/client` | Client-facing config (FPS, TTS hide delay) |
| `POST` | `/capture/ingest` | **Rifqi Module 2** — ingest a structured segment |
| `GET` | `/capture/tags/{session_id}` | All captured action items, decisions, deadlines, entities |
| `GET` | `/capture/confirmations/{session_id}` | Poll pending agent confirmation messages |
| `GET` | `/privacy/status` | Consent registry + default policy |
| `POST` | `/privacy/consent` | Grant or deny consent for a speaker |
| `DELETE` | `/privacy/consent/{speaker}` | Remove a speaker from the registry |

WebSocket endpoints: `/ws/asr` · `/ws/vision?session_id=...` · `/ws/tts?session_id=...`

---

## Rifqi Module 2 integration

Send a POST to `/capture/ingest` from Module 2's pipeline:

```python
import httpx

httpx.post("http://localhost:8000/capture/ingest", json={
    "session_id": "shared-session-id",
    "speaker":    "Zharif",          # real name already resolved by Module 2
    "text":       "We agreed to submit the draft by end of sprint.",
    "timestamp":  "2026-06-03T08:14:22Z",
    "source":     "module2",
})
```

The segment is run through the same autonomous tagger and written to `lkc_stream.jsonl` under the shared session ID, making it immediately available to Module 5's LKC retrieval layer.

---

## Privacy & Consent

By default the system operates **opt-in** (`DEFAULT_CONSENT = False` in `privacy.py`) — no speaker is recorded or identified until they consent. Change to `True` for opt-out semantics.

Register consent via the UI sidebar or the API:

```bash
curl -X POST http://localhost:8000/privacy/consent \
  -H "Content-Type: application/json" \
  -d '{"speaker": "Person A", "consented": true, "real_name": "Zharif"}'
```

Non-consenting speakers are anonymised to `"Person (anon)"` in vision output. PII (emails, Indonesian phone numbers, NIK) is scrubbed from transcripts before LKC writes.

Consent records persist in `consent.json` across server restarts.

---

## Evaluation metrics

`GET /metrics` returns per-session stats across seven axes:

| Axis | Fields |
|---|---|
| ASR latency | `p50_ms`, `p95_ms`, segment count |
| Vision latency | `p50_ms`, `p95_ms`, ok/error/stub frame counts, reliability |
| WER | Mean WER, sample count (requires reference via `/eval/reference`) |
| End-to-end | `p50_ms`, `p95_ms` (mic chunk received → transcript sent) |
| Capture quality | `action_items`, `decisions`, `deadlines` counts per session |
| Environment coverage | Valid / invalid frames, coverage ratio |
| Confirmation resolution | Sent / accepted / denied, accept rate |
| Privacy | PII tokens redacted |

Download as CSV: `GET /metrics/csv`

---

## Conversation modes

| Mode | Trigger | Agent behaviour |
|---|---|---|
| **Ambient** | No speakers present | Silent — low-activity background |
| **Greeting** | New face detected | Speaks a welcome; immediately transitions to Meeting Capture |
| **Meeting Capture** | Active speech (non-question) | Silent capture — tags and writes to LKC |
| **Q&A** | Question detected in transcript | Retrieves from LKC, then calls local LLM for a ≤2-sentence reply |
| **Confirmation** | Autonomous tagger detects an action item or decision | Speaks *"I captured [tag]. Is that correct?"* — resolves on yes/no |

---

## Speaker diarization

`assign_speaker()` in `dialogue.py` runs `pyannote/speaker-diarization-3.1` on each audio segment to identify the dominant speaker. The raw pyannote labels (e.g. `SPEAKER_01`) are mapped to stable per-session human labels (`Person A`, `Person B`, …) that remain consistent across segments within the same session.

**Fallback behaviour:** if pyannote is not installed, the HF token is missing, or the model is unavailable, the server automatically falls back to round-robin label assignment — no manual configuration required. The fallback is logged at `WARNING` level on startup.

To enable real diarization:
```bash
pip install pyannote.audio>=3.3.2 torch>=2.1.0
huggingface-cli login   # or export HF_TOKEN=hf_...
# Accept the model licence at https://huggingface.co/pyannote/speaker-diarization-3.1
```

---

## WhisperX ASR

WhisperX wraps the same CTranslate2 backend as faster-whisper but adds a two-pass pipeline: transcription followed by a phoneme-level forced alignment step that produces per-word `start`, `end`, and `score` fields. These are written into each LKC transcript record under `word_timestamps` and forwarded to the browser UI.

**Fallback behaviour:** if `whisperx` is not installed, the server loads `faster_whisper.WhisperModel` instead (same model weights, no word timestamps). A `WARNING` is logged on startup.

**WhisperX model tradeoffs:**

| Model | Size | WER (approx.) | CPU latency |
|---|---|---|---|
| `tiny` | ~75 MB | ~10% | ~300 ms |
| `base` | ~145 MB | ~7% | ~600 ms |
| `small` | ~466 MB | ~4% | ~1.5 s |
| `medium` | ~1.5 GB | ~3% | ~4 s |

Default is `small`. Change `model_size` in `config.json`. The alignment model for the detected language is downloaded once on first use and cached for the rest of the session.

---

## LKC retrieval

`lkc_retrieval.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`, ~22 MB) to encode all LKC documents as dense vectors. At query time, cosine similarity is computed via a normalised matrix dot product — no vector database required. The index is rebuilt lazily whenever `lkc_stream.jsonl` is updated.

**Fallback behaviour:** if `sentence-transformers` is not installed, the retriever automatically falls back to TF-IDF (sklearn). Both paths expose the same `query()` interface; call `retriever.stats()` to confirm which backend is active.

The relevance floor is `0.15` for dense retrieval and `0.05` for TF-IDF, calibrated so that only meaningfully related segments are injected into the LLM prompt.

---

## Month 5 upgrade path

- **Persistent LKC graph** — survive server restarts; index the full session history, not just the current file
- **WhisperX diarization alignment** — combine pyannote speaker segments with WhisperX word timestamps for word-level speaker attribution
- **Larger embedding model** — swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` for higher retrieval accuracy on longer sessions
- **spaCy NER** — replace regex entity extraction in `capture.py` with a fine-tuned English NER model for more precise entity detection