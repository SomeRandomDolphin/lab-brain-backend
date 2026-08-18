"""
app/core/logging.py — Centralised logging setup.

Fixes:
- basicConfig() is a no-op if any handler already exists on the root logger
  (e.g. from config.py or uvicorn importing before setup_logging() runs).
  We now force-replace all root handlers unconditionally.
- Uvicorn installs its own handlers on "uvicorn" and "uvicorn.access" loggers
  which can swallow or duplicate messages — we align them to our format too.
- DEBUG level is exposed via the LIVEKIT_LOG_LEVEL env var so you can get
  verbose output without code changes.
"""

import logging
import os
import sys


def setup_logging(level: int | None = None) -> None:
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    # Force-replace all handlers on the root logger regardless of what
    # basicConfig / uvicorn / any other import already attached.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Align uvicorn's own loggers to the same handler + format so their
    # output isn't swallowed or duplicated.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.setLevel(level)
        lg.propagate = False  # prevent double-printing via root

    # Make sure our key namespaces are always at DEBUG so the livekit
    # diagnostic logs we just added are never filtered out by a
    # namespace-level override.
    for name in ("pipeline.livekit_rooms", "pipeline.session_pipeline",
                 "livekit-test", "livekit"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    logging.getLogger(__name__).info(
        f"Logging initialised: level={logging.getLevelName(level)} "
        f"(override with LOG_LEVEL env var)"
    )

    # ── TEMP DIAGNOSTIC ──
    lk = logging.getLogger("pipeline.livekit_rooms")
    print(
        f"[diag] pipeline.livekit_rooms → disabled={lk.disabled} "
        f"level={logging.getLevelName(lk.level)} propagate={lk.propagate} "
        f"handlers={lk.handlers}",
        flush=True,
    )