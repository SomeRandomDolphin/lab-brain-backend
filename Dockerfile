# syntax=docker/dockerfile:1

# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: builder — has compilers, git, and the pip cache. None of this
# ships in the final image; only the resulting venv gets copied forward.
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

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
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# spaCy NER model used by app/services/capture.py — README calls this out
# explicitly as a required post-install step.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: runtime — slim base + only what's needed to RUN the app.
# No compilers, no git, no pip cache.
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

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
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8080/docs > /dev/null || exit 1

# Equivalent to `python main.py` per the README, but explicit about host/port
# and without dev-mode --reload. Swap in `--workers N` if you need more than
# one worker process (note: in-memory/session state in the pipeline modules
# isn't necessarily safe to run behind >1 worker without checking first).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]