"""
app/pipeline/asr.py — ASR pipeline: VAD chunker + WhisperX / faster-whisper.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.core.config import cfg

log = logging.getLogger(__name__)

SAMPLE_RATE        = cfg.vad.sample_rate
SILENCE_THRESHOLD  = cfg.vad.silence_threshold
VAD_SILENCE_CHUNKS = cfg.vad.silence_chunks
MAX_SEGMENT_CHUNKS = cfg.vad.max_segment_chunks

# ── ASR backend ───────────────────────────────────────────────────────────────
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    log.info(f"Loading WhisperX '{cfg.whisper.model_size}' on {cfg.whisper.device}…")
    _wx_model = whisperx.load_model(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        language=cfg.whisper.language,
    )
    _wx_align_cache: dict = {}
    log.info("WhisperX ready.")
except ImportError:
    WHISPERX_AVAILABLE = False
    from faster_whisper import WhisperModel
    log.warning("whisperx not installed — falling back to faster-whisper")
    _fw_model = WhisperModel(
        cfg.whisper.model_size,
        device=cfg.whisper.device,
        compute_type=cfg.whisper.compute_type,
        cpu_threads=cfg.whisper.cpu_threads,
        num_workers=cfg.whisper.num_workers,
    )
    log.info("faster-whisper ready.")


# ── VAD chunker ───────────────────────────────────────────────────────────────

class VadChunker:
    def __init__(self) -> None:
        self.buffer:       list[np.ndarray] = []
        self.silent_count: int = 0
        self.chunk_count:  int = 0

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    def push(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        self.buffer.append(pcm)
        self.chunk_count += 1
        is_silent = self.rms(pcm) < SILENCE_THRESHOLD
        self.silent_count = self.silent_count + 1 if is_silent else 0
        should_flush = (
            self.silent_count >= VAD_SILENCE_CHUNKS
            or self.chunk_count >= MAX_SEGMENT_CHUNKS
        )
        if should_flush and len(self.buffer) > VAD_SILENCE_CHUNKS:
            segment           = np.concatenate(self.buffer)
            self.buffer       = []
            self.silent_count = 0
            self.chunk_count  = 0
            return segment
        return None


# ── Transcription ─────────────────────────────────────────────────────────────

async def transcribe(
    segment_audio: np.ndarray,
    loop,
    align_cache: Optional[dict] = None,
) -> tuple[str, str, list[dict]]:
    """
    Transcribe a float32 PCM segment.

    Returns (full_text, detected_lang, word_timestamps).
    word_timestamps is a list of {word, start, end, score} dicts (WhisperX only).
    """
    if WHISPERX_AVAILABLE:
        wx_result = await loop.run_in_executor(
            None,
            lambda: _wx_model.transcribe(
                segment_audio, batch_size=8, language=cfg.whisper.language
            )
        )
        detected_lang = wx_result.get("language", cfg.whisper.language or "en")
        cache = align_cache if align_cache is not None else _wx_align_cache

        if detected_lang not in cache:
            align_model, align_meta = await loop.run_in_executor(
                None,
                lambda: whisperx.load_align_model(
                    language_code=detected_lang, device=cfg.whisper.device
                )
            )
            cache[detected_lang] = (align_model, align_meta)

        align_model, align_meta = cache[detected_lang]
        aligned = await loop.run_in_executor(
            None,
            lambda: whisperx.align(
                wx_result["segments"], align_model, align_meta,
                segment_audio, cfg.whisper.device, return_char_alignments=False
            )
        )
        wx_segments = aligned.get("segments", wx_result.get("segments", []))
        full_text   = " ".join(s["text"].strip() for s in wx_segments).strip()
        raw_word_ts: list[dict] = [
            {
                "word":  w.get("word", ""),
                "start": round(w.get("start", 0.0), 3),
                "end":   round(w.get("end",   0.0), 3),
                "score": round(w.get("score", 1.0), 3),
            }
            for seg in wx_segments
            for w in seg.get("words", [])
        ]
        return full_text, detected_lang, raw_word_ts

    else:
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
        return full_text, detected_lang, []


def resample_livekit_frame(raw_frame) -> np.ndarray:
    """
    Convert a LiveKit AudioFrame (48kHz int16 stereo) to 16kHz mono float32.
    """
    pcm_int16 = np.frombuffer(bytes(raw_frame.data), dtype=np.int16)
    if raw_frame.num_channels > 1:
        pcm_int16 = pcm_int16.reshape(-1, raw_frame.num_channels).mean(axis=1)
    pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
    src_rate = raw_frame.sample_rate
    if src_rate != SAMPLE_RATE:
        import soxr
        pcm_f32 = soxr.resample(pcm_f32, src_rate, SAMPLE_RATE)
    return pcm_f32
