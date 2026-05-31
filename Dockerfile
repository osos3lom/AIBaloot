# syntax=docker/dockerfile:1.7
# Multi-stage build for hakim-vision.
# Stage 1 resolves and installs Python deps with uv into a venv.
# Stage 2 is a slim runtime image with only what we need at run time.

ARG PYTHON_VERSION=3.12-slim-bookworm

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dep manifests first so the deps layer caches independently of source.
COPY pyproject.toml ./
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python -e .

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python -e .

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# OpenCV runtime needs libGL + glib; nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 hakim \
    && useradd  --system --uid 1000 --gid hakim --create-home hakim

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src

WORKDIR /app
USER hakim

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["hakim-vision", "--help"]

LABEL org.opencontainers.image.title="hakim-vision" \
      org.opencontainers.image.description="Synthetic playing-card dataset generator (Hakim Baloot AI platform)." \
      org.opencontainers.image.source="https://github.com/osos3lom/AIBaloot" \
      org.opencontainers.image.licenses="MIT"
