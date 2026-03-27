FROM python:3.11-slim

WORKDIR /app

# System deps for mplsoccer / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# ── 1. PyTorch (CPU-only, pinned to match dev env) ──────────────────────────
RUN pip install --no-cache-dir \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. PyTorch Geometric (CPU wheels) ───────────────────────────────────────
RUN pip install --no-cache-dir \
    torch-geometric==2.7.0 \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

# ── 3. App dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 4. Copy application ──────────────────────────────────────────────────────
COPY . .

# Graphs and model weights live in data/processed/ (gitignored).
# At startup, app.py downloads them from HuggingFace Hub if absent.
# Set HF_TOKEN as a secret env var in your deployment platform.
ENV HF_REPO=""

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
