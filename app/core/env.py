"""
app/core/env.py — Minimal .env file loader.

Parses a .env file and applies its KEY=VALUE pairs to os.environ *before*
the rest of the app reads its configuration (app/core/config.py,
app/db/supabase_client.py, app/db/migrations.py, etc. all just call
os.environ.get(...) — they don't know or care whether a value came from
the real environment or from .env).

Why a custom loader instead of python-dotenv? This keeps the project
dependency-free for something this small. The supported format covers
everything in .env.example: simple KEY=VALUE lines, blank lines, full-line
or trailing '#' comments, and optionally single- or double-quoted values.
If you need more (variable interpolation, multiline values, export syntax),
swap this for `pip install python-dotenv` + `from dotenv import load_dotenv`.

Precedence
----------
Real environment variables always win. If a key is already set in
os.environ (e.g. exported in your shell, or injected by Docker/Render/
Railway/etc.), the value in .env is ignored for that key — .env only fills
in whatever isn't already set. Pass override=True to change that.

Usage
-----
Call load_env() as early as possible — before importing anything that
reads os.environ at import time:

    from app.core.env import load_env
    load_env()

    from app.main import app   # safe — config/db modules import after this
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_env(
    path: Optional[Union[str, Path]] = None,
    override: bool = False,
) -> dict[str, str]:
    """
    Parse a .env file and apply its values to os.environ.

    Parameters
    ----------
    path : path to the .env file. Defaults to <project_root>/.env.
           Missing file is not an error — just means "nothing to load",
           which is the normal case in production where real env vars
           are set directly.
    override : if True, values from the file replace existing os.environ
               values. Default False: real environment variables win.

    Returns
    -------
    Dict of the variables actually written to os.environ by this call
    (keys skipped because they already existed and override=False are
    not included).
    """
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH

    if not env_path.exists():
        log.debug(f"[env] no .env file at {env_path} — skipping (using real env vars only).")
        return {}

    applied: dict[str, str] = {}
    already_set: list[str] = []
    malformed = 0

    # utf-8-sig strips a leading UTF-8 BOM if present (e.g. file saved by
    # Notepad/VS Code as "UTF-8 with BOM"); it's a no-op otherwise, so this
    # is safe either way and avoids a BOM silently glueing itself onto the
    # first key (`\ufeffSUPABASE_URL`), which would fail key matching with
    # no visible error.
    lines = env_path.read_text(encoding="utf-8-sig").splitlines()

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        # Tolerate a leading `export ` (shell-sourceable .env files).
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        if "=" not in line:
            malformed += 1
            log.warning(f"[env] {env_path.name}:{lineno} — skipping malformed line: {raw_line!r}")
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = _clean_value(value.strip())

        if not key:
            continue

        # A key that's *present* in os.environ but blank (e.g. a stray
        # `set VAR=` on Windows, or an empty exported value) is functionally
        # unset — treat it as fillable rather than letting it silently block
        # the .env value. Only a genuinely non-empty existing value wins.
        if not override and os.environ.get(key):
            already_set.append(key)
            continue

        os.environ[key] = value
        applied[key] = value

    # Previously this only ever logged a count, which made a "0 loaded"
    # result indistinguishable from "file is all comments/placeholders"
    # vs "everything was already shadowed by real env vars" vs "every line
    # was malformed" — you had to go hexdump the file to find out which.
    # Surfacing the breakdown here means the next time this happens, the
    # log line itself tells you why.
    log.info(
        f"[env] {env_path.name}: {len(lines)} line(s) read, {len(applied)} applied, "
        f"{len(already_set)} already-set-in-environment, {malformed} malformed"
    )
    if already_set:
        log.info(f"[env] kept pre-existing environment values for: {', '.join(already_set)}")
    return applied


def _clean_value(value: str) -> str:
    """Strip one layer of matching quotes, or a trailing ' # inline comment'."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return value