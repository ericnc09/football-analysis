# syntax=docker/dockerfile:1.6
#
# Multi-stage, CPU-only image for the Football GNN xG dashboard.
#
# Stage 1 (builder) installs torch + torch-geometric + app deps into an
# isolated prefix so we can copy the fully-resolved site-packages into a
# clean final layer. This keeps the runtime image small and deterministic.
#
# Pin rationale:
#   torch==2.7.0            — known to exist on the official CPU wheel index
#   torch-geometric==2.7.0  — matching PyG wheel URL at data.pyg.org
#   torch-scatter/sparse    — built against the same torch 2.7.0 + cpu
#
# Update both the TORCH_VERSION ARG and the PyG -f URL together.

ARG TORCH_VERSION=2.7.0
ARG PYG_VERSION=2.7.0

# ───── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG TORCH_VERSION
ARG PYG_VERSION

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1 is required by torch/scikit-learn at runtime; curl for healthcheck.
# build-essential is only needed if PyG falls back to source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. PyTorch CPU wheel (stable index)
RUN pip install --prefix=/install "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu

# 2. PyTorch Geometric + companion wheels
RUN pip install --prefix=/install "torch-geometric==${PYG_VERSION}" \
    torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"

# 3. App dependencies
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ───── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built Python environment from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source. .dockerignore excludes .venv/, data/, .git/ etc.
COPY . .

# At startup, app.py downloads model weights from the HF Hub model repo if
# absent. Set the HF_REPO_ID env var (or Space secret) on the host.
ENV HF_REPO_ID="" \
    HF_TOKEN="" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
