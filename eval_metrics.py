"""
eval_metrics.py — Evaluation Metrics (Module 5, Month 3)

Tracks the evaluation axes specified in the research plan:
  1. Transcription accuracy  — WER against user-supplied reference (optional)
  2. Recognition reliability — vision hit/miss/error rates across frames
  3. Latency                 — ASR p50/p95, vision p50/p95, end-to-end

Month 3 additions:
  4. Capture quality         — action item / decision tag counts per session
  5. Environment coverage    — how often a valid environment_state is received
  6. Confirmation resolution — ratio of agent confirmations accepted vs denied
  7. Privacy compliance      — count of redacted PII tokens per session

All metrics are accumulated in-memory and exposed via /metrics JSON endpoint.
A CSV export endpoint (/metrics/csv) is also provided for thesis reporting.

WER implementation uses a pure-Python edit-distance approach so there is no
jiwer/nltk dependency for the PoC.
"""

from __future__ import annotations

import csv
import io
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median, quantiles
from typing import Optional


# ── WER helpers ───────────────────────────────────────────────────────────────
def _edit_distance(r: list, h: list) -> int:
    """Levenshtein distance between two token lists."""
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if r[i-1] == h[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate in [0, 1]. Returns NaN if reference is empty."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return float("nan")
    return _edit_distance(ref, hyp) / len(ref)


# ── Per-session metrics accumulator ──────────────────────────────────────────
@dataclass
class SessionMetrics:
    session_id: str
    started_at: float = field(default_factory=time.time)

    # ASR latency (ms)
    asr_latencies: list[float] = field(default_factory=list)

    # Vision latency (ms) and reliability
    vision_latencies: list[float] = field(default_factory=list)
    vision_frames_ok: int = 0
    vision_frames_err: int = 0
    vision_frames_stub: int = 0  # gemini.api_key not set in config.json

    # WER samples (reference must be provided by user via /eval/reference endpoint)
    wer_samples: list[float] = field(default_factory=list)

    # End-to-end latency: time from audio chunk received → transcript sent (ms)
    e2e_latencies: list[float] = field(default_factory=list)

    # Mode dwell times {mode_name: total_seconds}
    mode_dwell: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _mode_last_switch: float = field(default_factory=time.time, repr=False)
    _current_mode: str = "ambient"

    # Segment count
    segment_count: int = 0

    # Month 3: Autonomous capture quality
    tag_action_items_count: int = 0
    tag_decisions_count:    int = 0
    tag_deadlines_count:    int = 0

    # Month 3: Environment coverage (frames that returned a valid env state)
    env_frames_valid:   int = 0
    env_frames_invalid: int = 0

    # Month 3: Confirmation loop resolution
    confirmations_sent:     int = 0
    confirmations_accepted: int = 0
    confirmations_denied:   int = 0

    # Month 3: Privacy — PII tokens redacted
    pii_tokens_redacted: int = 0

    def record_asr(self, latency_ms: float) -> None:
        self.asr_latencies.append(latency_ms)
        self.segment_count += 1

    def record_vision(self, latency_ms: float, ok: bool, stub: bool = False) -> None:
        self.vision_latencies.append(latency_ms)
        if stub:
            self.vision_frames_stub += 1
        elif ok:
            self.vision_frames_ok += 1
        else:
            self.vision_frames_err += 1

    def record_wer(self, reference: str, hypothesis: str) -> float:
        w = compute_wer(reference, hypothesis)
        if not math.isnan(w):
            self.wer_samples.append(w)
        return w

    def record_e2e(self, latency_ms: float) -> None:
        self.e2e_latencies.append(latency_ms)

    def record_mode_switch(self, new_mode: str) -> None:
        now = time.time()
        self.mode_dwell[self._current_mode] += now - self._mode_last_switch
        self._mode_last_switch = now
        self._current_mode = new_mode

    # Month 3 record helpers ──────────────────────────────────────────────────
    def record_tags(self, tags: dict) -> None:
        """Called by server.py after capture.process_segment writes a record."""
        self.tag_action_items_count += len(tags.get("action_items", []))
        self.tag_decisions_count    += len(tags.get("decisions",    []))
        self.tag_deadlines_count    += len(tags.get("deadlines",    []))

    def record_environment(self, valid: bool) -> None:
        if valid:
            self.env_frames_valid   += 1
        else:
            self.env_frames_invalid += 1

    def record_confirmation(self, sent: bool = False, accepted: bool = False, denied: bool = False) -> None:
        if sent:     self.confirmations_sent     += 1
        if accepted: self.confirmations_accepted += 1
        if denied:   self.confirmations_denied   += 1

    def record_pii_redaction(self, token_count: int) -> None:
        self.pii_tokens_redacted += token_count

    # ── Derived stats ──────────────────────────────────────────────────────────
    @staticmethod
    def _percentile(data: list[float], p: float) -> Optional[float]:
        if not data:
            return None
        sorted_d = sorted(data)
        idx = (len(sorted_d) - 1) * p
        lo = int(idx)
        hi = lo + 1
        if hi >= len(sorted_d):
            return sorted_d[lo]
        return sorted_d[lo] + (sorted_d[hi] - sorted_d[lo]) * (idx - lo)

    def summary(self) -> dict:
        # Flush current mode dwell
        now = time.time()
        dwell = dict(self.mode_dwell)
        dwell[self._current_mode] = dwell.get(self._current_mode, 0) + (now - self._mode_last_switch)

        total_frames = self.vision_frames_ok + self.vision_frames_err + self.vision_frames_stub
        vision_reliability = (
            round(self.vision_frames_ok / max(total_frames, 1), 3)
        )

        return {
            "session_id": self.session_id,
            "duration_s": round(now - self.started_at, 1),
            "segments": self.segment_count,
            "asr": {
                "p50_ms": self._percentile(self.asr_latencies, 0.50),
                "p95_ms": self._percentile(self.asr_latencies, 0.95),
                "count": len(self.asr_latencies),
            },
            "vision": {
                "p50_ms": self._percentile(self.vision_latencies, 0.50),
                "p95_ms": self._percentile(self.vision_latencies, 0.95),
                "total_frames": total_frames,
                "ok": self.vision_frames_ok,
                "errors": self.vision_frames_err,
                "stub": self.vision_frames_stub,
                "reliability": vision_reliability,
            },
            "wer": {
                "mean": round(sum(self.wer_samples) / max(len(self.wer_samples), 1), 3)
                        if self.wer_samples else None,
                "samples": len(self.wer_samples),
            },
            "e2e": {
                "p50_ms": self._percentile(self.e2e_latencies, 0.50),
                "p95_ms": self._percentile(self.e2e_latencies, 0.95),
            },
            "mode_dwell_s": {k: round(v, 1) for k, v in dwell.items()},
            # Month 3
            "capture": {
                "action_items": self.tag_action_items_count,
                "decisions":    self.tag_decisions_count,
                "deadlines":    self.tag_deadlines_count,
            },
            "environment": {
                "valid_frames":   self.env_frames_valid,
                "invalid_frames": self.env_frames_invalid,
                "coverage":       round(
                    self.env_frames_valid / max(self.env_frames_valid + self.env_frames_invalid, 1), 3
                ),
            },
            "confirmations": {
                "sent":     self.confirmations_sent,
                "accepted": self.confirmations_accepted,
                "denied":   self.confirmations_denied,
                "accept_rate": round(
                    self.confirmations_accepted / max(self.confirmations_sent, 1), 3
                ) if self.confirmations_sent else None,
            },
            "privacy": {
                "pii_tokens_redacted": self.pii_tokens_redacted,
            },
        }

    def to_csv_rows(self) -> list[dict]:
        """Flat row per ASR segment for CSV export."""
        return [
            {"session_id": self.session_id, "metric": "asr_latency_ms", "value": v}
            for v in self.asr_latencies
        ] + [
            {"session_id": self.session_id, "metric": "vision_latency_ms", "value": v}
            for v in self.vision_latencies
        ] + [
            {"session_id": self.session_id, "metric": "wer", "value": v}
            for v in self.wer_samples
        ] + [
            {"session_id": self.session_id, "metric": "e2e_latency_ms", "value": v}
            for v in self.e2e_latencies
        ]


# ── Module-level registry ─────────────────────────────────────────────────────
_sessions: dict[str, SessionMetrics] = {}

def get_metrics(session_id: str) -> SessionMetrics:
    if session_id not in _sessions:
        _sessions[session_id] = SessionMetrics(session_id=session_id)
    return _sessions[session_id]

def all_summaries() -> list[dict]:
    return [m.summary() for m in _sessions.values()]

def all_csv() -> str:
    rows = []
    for m in _sessions.values():
        rows.extend(m.to_csv_rows())
    if not rows:
        return "session_id,metric,value\n"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["session_id", "metric", "value"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()