# syntax=docker/dockerfile:1.6
# BTC Sentiment-Driven LSTM Trading Pipeline — Streamlit dashboard image.
#
# Multi-stage build: builder installs deps into a venv, runtime copies
# only the venv + app code. Non-root user, slim base.
#
# Build:
#   docker build -t btc-llm-sentiment:latest .
# Run:
#   docker run --rm -p 8501:8501 btc-llm-sentiment:latest
#
# NOTE: This image uses tensorflow-cpu (~500 MB). For GPU inference,
# build with --build-arg TF_PACKAGE=tensorflow==2.17.0 and a GPU base image.

ARG PYTHON_VERSION=3.11

# ────────────────────────────────────────────────────────────
# Stage 1: builder
# ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────
# Stage 2: runtime
# ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app"

# Runtime deps only (no build-essential); curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 btc \
    && useradd  --uid 1001 --gid btc --create-home --shell /bin/bash btc

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy app code (notebooks/interim/ contains the committed model artifacts)
COPY --chown=btc:btc . /app/

USER btc
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
