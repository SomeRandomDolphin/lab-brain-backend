# Lab Brain — Module 5: Real-Time Multimodal Conversational Agent

Part of **TEEP 2026 / Lab Brain** — a humanistic intelligence platform for digitalising knowledge creation using the SECI model.

> **Current milestone: Month 5** — Persistent SQLite LKC graph · all-mpnet-base-v2 dense retrieval · spaCy NER · WhisperX + pyannote word-level speaker alignment · Summon-gated QA (agent only speaks when addressed)

---

## What's new in Month 5

### 1. Persistent LKC Graph (`lkc_graph.py`)
The flat `lkc_stream.jsonl` append log is now backed by a **SQLite WAL database** (`lkc_graph.db`). All records are written to both the graph (primary) and the JSONL file (backward-compat mirror for Rifqi / Wildan pipelines).

Benefits:
- Survives server restarts — no records lost between sessions
- Indexed queries by `session_id`, `record_type`, and `timestamp_unix` — no full-file scans
- Cross-session history available for retrieval (Q&A can reach back into last week's meeting)
- New REST endpoints for browsing, filtering, and deleting session data

### 2. Upgraded Embedding Model (`lkc_retrieval.py`)
`all-mpnet-base-v2` (~420 MB) replaces `all-MiniLM-L6-v2` (~22 MB) as the primary dense retrieval model. Cosine similarity relevance floor raised from 0.15 → 0.20 to reflect mpnet's generally higher scores.

Fallback chain: mpnet → MiniLM → TF-IDF (sklearn) → disabled.

### 3. spaCy NER (`capture.py`)
Regex-based entity extraction replaced with `en_core_web_sm` (or `en_core_web_trf` for transformer-level accuracy). Entity types captured: `PERSON`, `ORG`, `PRODUCT`, `GPE`, `DATE`, `EVENT`, `WORK_OF_ART`.

Regex patterns are kept as a fallback when spaCy is not installed.

### 4. Word-Level Speaker Attribution (`dialogue.py`)
`assign_speaker_words()` combines pyannote diarization turn boundaries with WhisperX per-word timestamps to annotate each word with the speaker who said it. The result is stored in `lkc_stream.jsonl` under `word_timestamps[].speaker`.

### 5. Summon-Gated QA (agent goes quiet by default)
The agent no longer interrupts every question in the room. It only enters QA mode when **explicitly addressed** with a wake-word:

| Phrase | Example |
|--------|---------|
| `lab brain` | *"Lab Brain, what did we decide about embeddings?"* |
| `hey brain` | *"Hey Brain, summarise the last ten minutes."* |
| `@lab` | *"@lab what's the action item count?"* |
| `brain,` / `brain?` | *"Brain, can you recap?"* |

Configure phrases in `config.json` under `summon.phrases`. Set `require_summon: false` to restore Month 4 question-triggered behaviour.

The frontend can also summon or dismiss the agent via REST:
```
POST   /agent/summon/{session_id}   — summon (e.g. clicking a mic button)
DELETE /agent/summon/{session_id}   — dismiss
GET    /agent/summon/{session_id}   — check current state
```

---

## Architecture

```
Browser
├── Mic PCM (float32, 16kHz, 500ms chunks)
│    └── /ws/asr ──► VadChunker ──► WhisperX (word timestamps + alignment)
│                         │               └── faster-whisper fallback
│                         │
│                    capture.py  ← spaCy NER + wake-word detection
│                    privacy.py  ← PII redaction + consent gate
│                    dialogue.py ← mode FSM (summon-gated QA)
│                         │    └── pyannote diarization + word-level alignment
│                         ├── lkc_retrieval.py (mpnet over SQLite graph)
│                         │                    └── MiniLM / TF-IDF fallback
│                         └── local LLM reply (llama3.2:3b via Ollama)
│                              └── /ws/tts ──► SpeechSynthesis (browser)
│
├── Camera JPEG (1 FPS)
│    └── /ws/vision ──► vision.py ──► llava:7b via Ollama
│
└── Rifqi Module 2
     └── POST /capture/ingest ──► capture.py ──► lkc_graph.db + lkc_stream.jsonl

lkc_graph.db         ←  persistent SQLite store (primary, survives restarts)
lkc_stream.jsonl     ←  JSONL mirror (backward compat for other modules)
```

---

## Setup

### 1. Start Ollama

```bash
ollama pull llava:7b
ollama pull llama3.2:3b
ollama serve
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
# Optional — higher accuracy, transformer-based (slower on CPU):
# python -m spacy download en_core_web_trf
```

### 4. Configure

Edit `config.json`. All fields are optional — the server starts with safe defaults.

```json
{
  "local_llm": {
    "base_url":       "http://localhost:11434/v1",
    "vision_model":   "llava:7b",
    "dialogue_model": "llama3.2:3b"
  },
  "lkc": {
    "log_file":        "lkc_stream.jsonl",
    "db_file":         "lkc_graph.db",
    "retrieval_top_k": 4
  },
  "summon": {
    "phrases":        ["lab brain", "hey brain", "@lab"],
    "require_summon": true
  },
  "spacy": {
    "model": "en_core_web_sm"
  }
}
```

### 5. Run

```bash
python server.py
# open http://localhost:8000
```

---

## Files

| File | Role |
|---|---|
| `server.py` | FastAPI app — WebSocket ASR/vision/TTS, REST API (Month 5 graph + summon endpoints) |
| `dialogue.py` | Mode FSM, summon-gated QA, pyannote diarization, word-level speaker alignment |
| `capture.py` | spaCy NER tagger, wake-word detection, Rifqi Module 2 ingest |
| `lkc_graph.py` | **NEW** — SQLite-backed persistent LKC graph |
| `lkc_retrieval.py` | Dense retrieval (mpnet → MiniLM → TF-IDF) over the full graph |
| `vision.py` | VLM perception — Physical AI, privacy-gated speaker labels |
| `privacy.py` | Consent registry, PII redaction, face anonymisation |
| `eval_metrics.py` | In-memory metrics (ASR/vision latency, WER, capture, privacy) |
| `config.py` | Typed config loader with Month 5 `SummonConfig` and `SpacyConfig` |
| `static/index.html` | Single-file browser client |

---

## REST API Reference

### LKC (Month 5 additions)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/lkc/stats` | Graph-wide statistics (total records, sessions, timestamps) |
| `GET` | `/lkc/sessions` | List all sessions with record counts |
| `GET` | `/lkc/sessions/{sid}` | Records for a session. Query params: `record_type`, `since_unix`, `limit` |
| `DELETE` | `/lkc/sessions/{sid}` | Delete all records for a session |
| `GET` | `/lkc` | Legacy HTML viewer (last 500 records) |
| `DELETE` | `/lkc` | Wipe entire graph |

### Agent Summon (Month 5)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/agent/summon/{sid}` | Is the agent currently summoned? |
| `POST` | `/agent/summon/{sid}` | Manually summon the agent (e.g. UI button press) |
| `DELETE` | `/agent/summon/{sid}` | Dismiss the agent |

### Diagnostics (Month 5)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ner/status` | NER backend: `spacy_en_core_web_sm` or `regex_fallback` |
| `GET` | `/retrieval/stats` | Embedding model, index size, backend in use |

### All Month 3/4 endpoints (unchanged)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mode/{sid}` | Current mode + summoned flag |
| `GET` | `/perception/{sid}` | Latest vision state |
| `GET` | `/metrics` | All session metrics (JSON) |
| `GET` | `/metrics/csv` | Metrics CSV download |
| `POST` | `/eval/reference` | Submit WER reference transcript |
| `POST` | `/summary/{sid}` | Generate end-of-session LLM summary |
| `GET` | `/config/client` | Camera FPS, TTS hide delay |
| `POST` | `/capture/ingest` | Rifqi Module 2 segment ingest |
| `GET` | `/capture/tags/{sid}` | Action items, decisions, entities, deadlines |
| `GET` | `/capture/confirmations/{sid}` | Pending confirmation messages |
| `GET` | `/capture/ner_backend` | NER backend status |
| `GET` | `/privacy/status` | Consent registry |
| `POST` | `/privacy/consent` | Grant/deny consent for a speaker |
| `DELETE` | `/privacy/consent/{speaker}` | Remove speaker from registry |

WebSocket endpoints: `/ws/asr` · `/ws/vision?session_id=...` · `/ws/tts?session_id=...`

---

## Summon System — Frontend Integration

The `transcript` WebSocket message now includes a `summoned` field:

```json
{
  "type":     "transcript",
  "text":     "Lab Brain, what did we decide about embeddings?",
  "summoned": true,
  "mode":     "qa",
  ...
}
```

Suggested UI patterns:
- Show a pulsing indicator when `summoned: true` (agent is "thinking")
- Add a **"Ask Lab Brain"** button that calls `POST /agent/summon/{session_id}` — useful when the speaker doesn't want to say the wake-word aloud
- Add a **"Dismiss"** button that calls `DELETE /agent/summon/{session_id}` to cancel a pending response

---

## LKC Output Format (Month 5 additions)

**Transcript segment** now includes per-word speaker attribution:

```json
{
  "type":       "transcript",
  "session_id": "a3f2b1c0",
  "speaker":    "Person A",
  "text":       "We decided to use mpnet for Month 5.",
  "tags": {
    "action_items": [],
    "decisions":    ["We decided to use mpnet for Month 5"],
    "entities":     ["Month 5"],
    "deadlines":    []
  },
  "word_timestamps": [
    {"word": "We",      "start": 0.0,  "end": 0.18, "score": 0.99, "speaker": "Person A"},
    {"word": "decided", "start": 0.2,  "end": 0.56, "score": 0.98, "speaker": "Person A"}
  ]
}
```

---

## Conversation Modes

| Mode | Trigger | Agent behaviour |
|------|---------|-----------------|
| **Ambient** | No speakers present | Silent |
| **Greeting** | New face detected | Speaks welcome; returns to Meeting Capture |
| **Meeting Capture** | Active speech (not summoned) | **Silent** — logs everything to LKC |
| **QA** | Active speech **+ summon wake-word** | Retrieves from LKC + LLM reply (≤2 sentences) |
| **Confirmation** | Tagger detects action item / decision | Speaks confirmation prompt; resolves on yes/no |

> **Month 5 change:** QA mode previously triggered on any question detected in the room. It now requires an explicit summon phrase. Set `require_summon: false` in `config.json` to restore the old behaviour.

---

## Month 6 upgrade path

- **Cross-session entity graph** — build a property graph linking People, Projects, Decisions, and Action Items across all sessions for structured Q&A
- **Streaming TTS** — replace browser SpeechSynthesis with a local TTS model (Kokoro, Coqui) for consistent voice and offline operation
- **WhisperX large-v3** — upgrade ASR model for lower WER on accented Indonesian-English code-switching
- **Diarization + NER fusion** — use spaCy dependency parsing to resolve pronouns ("he", "she", "they") back to named speakers from the diarization timeline
