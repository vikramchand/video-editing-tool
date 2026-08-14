# Video Understanding API
#
# This image contains the API, FFmpeg and Whisper. It does NOT contain Ollama.
#
# That split is deliberate. Docker Desktop on macOS runs Linux containers in a
# VM with no access to the Mac's GPU, so an Ollama instance inside this image
# would be stuck on CPU and several times slower. Ollama therefore runs on the
# host, where it reaches Apple's Metal backend, and the container talks to it
# over http://host.docker.internal:11434.
#
#   Mac
#    |- Ollama (host, Metal GPU) <--- HTTP ---.
#    '- Docker                                |
#        '- Video Understanding API ----------'

FROM python:3.12-slim-bookworm

# FFmpeg is a hard requirement of the pipeline; everything else is build glue.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IN_DOCKER=1

WORKDIR /app

# Copy only what the build backend needs first, so dependency installation is
# cached independently of source edits.
COPY pyproject.toml README.md ./
COPY video_understanding ./video_understanding

# Whisper is installed in-image: speech-to-text runs on CPU anyway, so it gains
# nothing from the host GPU and everything from being self-contained.
#
# PySceneDetect is deliberately NOT installed here - it pulls in a full OpenCV
# build (and its system libraries) for a modest accuracy gain over FFmpeg's own
# `scene` filter, which the pipeline falls back to automatically. To enable it,
# uncomment the second install and set SCENE_DETECTOR=auto.
RUN pip install --no-cache-dir ".[whisper]"
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/* \
#     && pip install --no-cache-dir ".[scenes]"

# Pre-download the Whisper weights so the first request is not also the first
# model download. Override with --build-arg WHISPER_MODEL=base to shrink.
ARG WHISPER_MODEL=small
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')" \
    || echo 'warning: whisper model was not pre-cached; it will download on first use'

# Run as a non-root user; /app/data is a volume mount point.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/output \
    && chown -R appuser:appuser /app
USER appuser

ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output \
    HOST=0.0.0.0 \
    PORT=8000 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    SCENE_DETECTOR=ffmpeg

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "video_understanding.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
