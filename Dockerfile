FROM python:3.10-slim

# ── Labels ──────────────────────────────────────────────────────────
LABEL maintainer="SRM University AP — NLP Project"
LABEL description="IQAS: Intelligent Question Answering System"
LABEL version="1.0"

WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────
# curl  → needed for HEALTHCHECK
# build-essential → needed by some Python C extensions (faiss, spacy)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── spaCy language model ─────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Application code ─────────────────────────────────────────────────
COPY . .

# ── Runtime directories ──────────────────────────────────────────────
RUN mkdir -p \
        data/uploads \
        data/processed \
        data/sample_docs \
        models/faiss_index \
        models/embeddings_cache \
        logs

# ── Streamlit config (disable file-watcher to suppress torchvision noise) ──
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

# ── Port ─────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
