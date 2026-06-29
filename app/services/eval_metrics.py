"""
app/services/eval_metrics.py — Evaluation Metrics Service.

Tracks: WER, ASR/vision/e2e latency, capture quality, environment coverage,
confirmation resolution, PII redaction counts.
"""

from __future__ import annotations

import csv
import io
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


def _edit_distance(r: list, h: list) -> int:
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if r[i-1] == h[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return float("nan")
    return _edit_distance(ref, hyp) / len(ref)


@dataclass
class SessionMetrics:
    session_id: str
    started_at: float = field(default_factory=time.time)

    asr_latencies:    list[float] = field(default_factory=list)
    vision_latencies: list[float] = field(default_factory=list)
    vision_frames_ok:   int = 0
    vision_frames_err:  int = 0
    vision_frames_stub: int = 0
    wer_samples:      list[float] = field(default_factory=list)
    e2e_latencies:    list[float] = field(default_factory=list)
    mode_dwell:       dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _mode_last_switch: float = field(default_factory=time.time, repr=False)
    _current_mode:     str   = "ambient"
    segment_count:     int   = 0

    tag_action_items_count: int = 0
    tag_decisions_count:    int = 0
    tag_deadlines_count:    int = 0
    env_frames_valid:   int = 0
    env_frames_invalid: int = 0
    confirmations_sent:     int = 0
    confirmations_accepted: int = 0
    confirmations_denied:   int = 0
    pii_tokens_redacted:    int = 0

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

    def record_tags(self, tags: dict) -> None:
        self.tag_action_items_count += len(tags.get("action_items", []))
        self.tag_decisions_count    += len(tags.get("decisions",    []))
        self.tag_deadlines_count    += len(tags.get("deadlines",    []))

    def record_environment(self, valid: bool) -> None:
        if valid:
            self.env_frames_valid   += 1
        else:
            self.env_frames_invalid += 1

    def record_confirmation(self, sent=False, accepted=False, denied=False) -> None:
        if sent:     self.confirmations_sent     += 1
        if accepted: self.confirmations_accepted += 1
        if denied:   self.confirmations_denied   += 1

    def record_pii_redaction(self, token_count: int) -> None:
        self.pii_tokens_redacted += token_count

    @staticmethod
    def _pct(data: list[float], p: float) -> Optional[float]:
        if not data:
            return None
        s   = sorted(data)
        idx = (len(s) - 1) * p
        lo  = int(idx)
        hi  = lo + 1
        if hi >= len(s):
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    def summary(self) -> dict:
        now   = time.time()
        dwell = dict(self.mode_dwell)
        dwell[self._current_mode] = dwell.get(self._current_mode, 0) + (now - self._mode_last_switch)
        total = self.vision_frames_ok + self.vision_frames_err + self.vision_frames_stub
        return {
            "session_id": self.session_id,
            "duration_s": round(now - self.started_at, 1),
            "segments":   self.segment_count,
            "asr": {
                "p50_ms": self._pct(self.asr_latencies, 0.50),
                "p95_ms": self._pct(self.asr_latencies, 0.95),
                "count":  len(self.asr_latencies),
            },
            "vision": {
                "p50_ms":      self._pct(self.vision_latencies, 0.50),
                "p95_ms":      self._pct(self.vision_latencies, 0.95),
                "total_frames": total,
                "ok":          self.vision_frames_ok,
                "errors":      self.vision_frames_err,
                "stub":        self.vision_frames_stub,
                "reliability": round(self.vision_frames_ok / max(total, 1), 3),
            },
            "wer": {
                "mean":    round(sum(self.wer_samples) / max(len(self.wer_samples), 1), 3)
                           if self.wer_samples else None,
                "samples": len(self.wer_samples),
            },
            "e2e": {
                "p50_ms": self._pct(self.e2e_latencies, 0.50),
                "p95_ms": self._pct(self.e2e_latencies, 0.95),
            },
            "mode_dwell_s": {k: round(v, 1) for k, v in dwell.items()},
            "capture": {
                "action_items": self.tag_action_items_count,
                "decisions":    self.tag_decisions_count,
                "deadlines":    self.tag_deadlines_count,
            },
            "environment": {
                "valid_frames":   self.env_frames_valid,
                "invalid_frames": self.env_frames_invalid,
                "coverage": round(
                    self.env_frames_valid / max(self.env_frames_valid + self.env_frames_invalid, 1), 3
                ),
            },
            "confirmations": {
                "sent":       self.confirmations_sent,
                "accepted":   self.confirmations_accepted,
                "denied":     self.confirmations_denied,
                "accept_rate": round(
                    self.confirmations_accepted / max(self.confirmations_sent, 1), 3
                ) if self.confirmations_sent else None,
            },
            "privacy": {"pii_tokens_redacted": self.pii_tokens_redacted},
        }

    def to_csv_rows(self) -> list[dict]:
        return (
            [{"session_id": self.session_id, "metric": "asr_latency_ms", "value": v}
             for v in self.asr_latencies]
            + [{"session_id": self.session_id, "metric": "vision_latency_ms", "value": v}
               for v in self.vision_latencies]
            + [{"session_id": self.session_id, "metric": "wer", "value": v}
               for v in self.wer_samples]
            + [{"session_id": self.session_id, "metric": "e2e_latency_ms", "value": v}
               for v in self.e2e_latencies]
        )


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
    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["session_id", "metric", "value"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
