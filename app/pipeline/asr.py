"""
app/pipeline/asr.py — ASR pipeline: VAD chunker + WhisperX / faster-whisper.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.core.config import cfg
import torch
torch.set_num_threads(cfg.whisper.cpu_threads)

# PyTorch 2.6 changed torch.load's default from weights_only=False to
# weights_only=True. The pyannote VAD checkpoint that whisperx.load_model()
# pulls in below pickles several omegaconf classes (ListConfig,
# ContainerMetadata, and potentially others) that aren't in torch's default
# safe-globals allowlist — allowlisting them one at a time as each new
# UnpicklingError surfaces is a whack-a-mole. We want to force
# weights_only=False for this one call. Safe here since the checkpoint is
# pyannote's own official release, not untrusted input.
#
# NOTE: functools.partial(torch.load, weights_only=False) does NOT work here.
# lightning_fabric's internal loader (lightning_fabric/utilities/cloud_io.py
# _load()) always calls torch.load(..., weights_only=weights_only) with an
# *explicit* keyword — its own default just happens to be None. A partial's
# preset kwargs only fill in arguments the caller omits; they're silently
# discarded the moment the caller supplies that keyword itself, even as
# None. So the explicit weights_only=None from lightning_fabric overrides
# our partial's weights_only=False, None falls through to torch's
# safe-globals resolution path, and we get the exact UnpicklingError this
# comment is trying to prevent.
#
# A real wrapper function doesn't have this problem — it can unconditionally
# overwrite the kwarg before forwarding, regardless of what the caller passed.
import functools
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

log = logging.getLogger(__name__)

SAMPLE_RATE        = cfg.vad.sample_rate
SILENCE_THRESHOLD  = cfg.vad.silence_threshold
VAD_SILENCE_CHUNKS = cfg.vad.silence_chunks
MAX_SEGMENT_CHUNKS = cfg.vad.max_segment_chunks
MIN_FLUSH_CHUNKS = 5  # ~50ms; just prevents flushing a near-empty buffer

# Minimum audio length before we attempt transcription.
# WhisperX hallucinates common phrases ("Thank you.", "Thanks for watching.")
# on segments shorter than ~1 s; discard anything below this floor.
MIN_SEGMENT_SAMPLES = SAMPLE_RATE  # 1 second at 16 kHz = 16 000 samples

# Minimum mean word-alignment score to accept a WhisperX result.
# Alignment scores < 0.40 indicate the model found no real speech and invented
# text — the canonical hallucination signature.
MIN_WORD_SCORE = 0.40

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
        asr_options={"beam_size": cfg.whisper.beam_size},
    )
    _wx_align_cache: dict = {}
    log.info("WhisperX ready.")
    if cfg.whisper.language:
        log.info(f"Pre-loading alignment model for '{cfg.whisper.language}'…")
        _align_model, _align_meta = whisperx.load_align_model(
            language_code=cfg.whisper.language, device=cfg.whisper.device
        )
        _wx_align_cache[cfg.whisper.language] = (_align_model, _align_meta)
        log.info("Alignment model ready.")
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
        # Minimum buffer length is just a sanity floor to avoid flushing an
        # almost-empty buffer right after a reset — NOT tied to the silence
        # or max-segment thresholds themselves.
        if should_flush and len(self.buffer) >= MIN_FLUSH_CHUNKS:
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
    Returns ("", detected_lang, []) when the segment is too short or the
    result is detected as a hallucination.
    """
    # ── Guard: segment too short ──────────────────────────────────────────────
    # Whisper hallucinates on anything under ~1 s.  Reject early so we never
    # even run inference on noise blips that slipped past the VAD.
    if len(segment_audio) < MIN_SEGMENT_SAMPLES:
        log.info(
            f"[asr] segment too short ({len(segment_audio)} samples, "
            f"{len(segment_audio) / SAMPLE_RATE:.2f}s < "
            f"{MIN_SEGMENT_SAMPLES / SAMPLE_RATE:.2f}s min); skipping transcription."
        )
        return "", cfg.whisper.language or "en", []

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

        # ── Guard: hallucination filter ───────────────────────────────────────
        # WhisperX alignment scores reflect how well each word anchors to the
        # audio.  A mean score below MIN_WORD_SCORE means the model invented
        # text (e.g. "Thank you.", "Thanks for watching.") on near-silence.
        if raw_word_ts:
            avg_score = sum(w["score"] for w in raw_word_ts) / len(raw_word_ts)
            if avg_score < MIN_WORD_SCORE:
                log.warning(
                    f"[asr] hallucination detected — avg word score {avg_score:.3f} "
                    f"< {MIN_WORD_SCORE}; discarding: {full_text!r}"
                )
                return "", detected_lang, []

        return full_text, detected_lang, raw_word_ts

    else:
        # faster-whisper returns a lazy generator — the actual inference is driven
        # by iterating it.  We must consume the generator *inside* the executor
        # so all CPU work stays off the event loop thread.
        def _fw_transcribe() -> tuple[str, str]:
            segs, info = _fw_model.transcribe(
                segment_audio,
                language=cfg.whisper.language,
                vad_filter=False,
                beam_size=cfg.whisper.beam_size,
            )
            return " ".join(seg.text for seg in segs).strip(), info.language

        full_text, detected_lang = await loop.run_in_executor(None, _fw_transcribe)
        return full_text, detected_lang, []


def resample_livekit_frame(raw_frame) -> np.ndarray:
    """
    Convert a LiveKit AudioFrame (48kHz int16 stereo) to 16kHz mono float32.
    """
    pcm_int16 = np.frombuffer(bytes(raw_frame.data), dtype=np.int16)
    # log.info(
    #     f"[asr:debug] channels={raw_frame.num_channels} "
    #     f"rate={raw_frame.sample_rate} raw_len={len(pcm_int16)} "
    #     f"raw_max_abs={np.abs(pcm_int16).max() if len(pcm_int16) else 0} "
    #     f"raw_rms={np.sqrt(np.mean(pcm_int16.astype(np.float64)**2)):.2f}"
    # )
    if raw_frame.num_channels > 1:
        pcm_int16 = pcm_int16.reshape(-1, raw_frame.num_channels).mean(axis=1)
    pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
    src_rate = raw_frame.sample_rate
    if src_rate != SAMPLE_RATE:
        import soxr
        pcm_f32 = soxr.resample(pcm_f32, src_rate, SAMPLE_RATE)
    return pcm_f32