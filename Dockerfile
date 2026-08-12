# syntax=docker/dockerfile:1

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: builder — has compilers, git, and the pip cache. None of this
# ships in the final image; only the resulting venv gets copied forward.
# ═══════════════════════════════════════════════════════════════════════════
# Pinned to -bookworm explicitly (not the rolling `python:3.11-slim` tag):
# that tag recently moved from Debian bookworm (glibc 2.36) to trixie
# (glibc 2.41). Newer glibc's dynamic loader refuses to mprotect an
# executable stack the way the prebuilt ctranslate2 wheel's bundled
# libctranslate2-*.so expects, and both whisperx and faster-whisper import
# ctranslate2 — so on trixie the app fails at import time with
# "ImportError: ... cannot enable executable stack as shared object
# requires: Invalid argument" before uvicorn ever starts. See
# https://github.com/OpenNMT/CTranslate2/issues/1849 — the fix (#1852) is
# closed upstream but hasn't actually shipped in a published wheel as of
# ctranslate2 4.8.1, so pinning the base image (not a Python dependency)
# is the reliable fix. Revisit this pin once a ctranslate2 release fixes
# the underlying .so.
FROM python:3.11-slim-bookworm AS builder

# build-essential : compiles native extensions for any package without a
#                   prebuilt wheel for this platform/Python version
# git             : some pip packages install straight from git refs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set to 1 to instead let torch/torchaudio/torchvision resolve to their
# default (CUDA-enabled) wheels — only needed if something in THIS
# container does GPU inference. As deployed, whisper.device defaults to
# "cpu" in config.json and LLM inference happens on the separately-managed
# Ollama container, so CPU-only is the right default.
ARG USE_CUDA_TORCH=0

ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

# --mount=type=cache persists pip's download/wheel cache BETWEEN builds on
# your machine (never baked into an image layer) — requires BuildKit,
# which `docker build`/`docker compose build` already use by default on
# any reasonably current Docker install.
#
# IMPORTANT: this resolves EVERYTHING in requirements.txt in a single pip
# call, with the CPU wheel index available for that whole resolution —
# not as a separate pre-install step. A separate `pip install torch
# torchaudio` step before `pip install -r requirements.txt` looks like it
# should work but doesn't: whisperx pins torchaudio~=2.8.0 and
# torchvision~=0.23.0, and the second command (no index flag) re-resolves
# against default PyPI to satisfy those, silently uninstalling the CPU
# build and reinstalling the CUDA one — pulling back in nvidia-cublas,
# nvidia-cudnn, nvidia-cufft, nvidia-nccl, triton, etc. (~4-5GB) in the
# process. Passing --extra-index-url on the SAME command that installs
# whisperx/pyannote.audio means pip sees the CPU wheels as an option
# while resolving their torch/torchaudio/torchvision pins too.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ "$USE_CUDA_TORCH" = "0" ]; then \
        sed -E 's/^(torch|torchaudio|torchvision)==([0-9][0-9A-Za-z.]*)$/\1==\2+cpu/' requirements.txt > requirements.resolved.txt && \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.resolved.txt; \
    else \
        pip install -r requirements.txt; \
    fi
    # requirements.resolved.txt only appends +cpu to torch/torchaudio/torchvision's
    # own pinned lines (requirements.txt otherwise unchanged) — still ONE pip
    # install call, so the whole graph (including transformers' backtracking)
    # resolves against a hard-pinned, ABI-matched +cpu build for all three,
    # instead of leaving pip free to land on a mismatched combination. See
    # requirements.txt's comment on the torch/torchaudio/torchvision pins for
    # the failure mode (`undefined symbol: torch_library_impl`) this avoids.

# spaCy NER model used by app/services/capture.py — README calls this out
# explicitly as a required post-install step.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: runtime — slim base + only what's needed to RUN the app.
# No compilers, no git, no pip cache.
# ═══════════════════════════════════════════════════════════════════════════
# Pinned to -bookworm — see the matching comment on the builder stage's
# FROM line above; both stages must use the same glibc, and the runtime
# stage is the one that actually imports ctranslate2 at process startup.
FROM python:3.11-slim-bookworm AS runtime

# ffmpeg          : required by whisperx / av for audio+video decode
# libsndfile1     : required by soundfile (whisperx/pyannote dependency)
# curl            : used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
COPY --from=builder $VENV_PATH $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app
COPY . .

# Run as non-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app $VENV_PATH
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/docs > /dev/null || exit 1

# Equivalent to `python main.py` per the README, but explicit about host/port
# and without dev-mode --reload. Swap in `--workers N` if you need more than
# one worker process (note: in-memory/session state in the pipeline modules
# isn't necessarily safe to run behind >1 worker without checking first).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]