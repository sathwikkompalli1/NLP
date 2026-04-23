"""
IQAS Global Configuration
========================
Central configuration for all paths, model names, and hyperparameters.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ──────────────────────────── Base Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
MODELS_DIR = PROJECT_ROOT / "models"
FAISS_INDEX_DIR = MODELS_DIR / "faiss_index"
EMBEDDINGS_CACHE_DIR = MODELS_DIR / "embeddings_cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# ──────────────────────────── Auto-create directories ────────────────────────────
for _dir in [
    UPLOAD_DIR, PROCESSED_DIR, SAMPLE_DOCS_DIR,
    FAISS_INDEX_DIR, EMBEDDINGS_CACHE_DIR, LOGS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Model Configuration ────────────────────────────
MODEL_NAME = os.getenv("IQAS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
QA_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RERANKER_MODEL = os.getenv("IQAS_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SPACY_MODEL = "en_core_web_sm"
EMBEDDING_DIM = 384  # Dimension for MiniLM models

# ──────────────────────────── Chunking Configuration ────────────────────────────
CHUNK_SIZE = int(os.getenv("IQAS_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("IQAS_CHUNK_OVERLAP", "50"))
SENTENCE_CHUNK_MAX_TOKENS = 400
PARAGRAPH_CHUNK_MAX_TOKENS = 500

# ──────────────────────────── Retrieval Configuration ────────────────────────────
TOP_K_RETRIEVE = int(os.getenv("IQAS_TOP_K_RETRIEVE", "20"))
TOP_K_RERANK = int(os.getenv("IQAS_TOP_K_RERANK", "5"))
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # Reciprocal Rank Fusion constant

# ──────────────────────────── FAISS Configuration ────────────────────────────
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
FAISS_IVF_NLIST = 100  # Number of Voronoi cells for IVF index
FAISS_LARGE_CORPUS_THRESHOLD = 10000  # Switch to IVF above this

# ──────────────────────────── Embedding Configuration ────────────────────────────
EMBED_BATCH_SIZE = 32
EMBEDDING_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings.npy"

# ──────────────────────────── Logging ────────────────────────────
LOG_LEVEL = os.getenv("IQAS_LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "iqas.log"

# ──────────────────────────── Streamlit ────────────────────────────
APP_TITLE = "🧠 IntelliRetrieve AI — Intelligent Document Retrieval & QA"
APP_ICON = "🧠"
APP_LAYOUT = "wide"

# ──────────────────────────── UI Colors ────────────────────────────
COLORS = {
    "primary": "#1E3A5F",
    "accent": "#00B4D8",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "error": "#E74C3C",
    "bg_dark": "#0F1117",
    "bg_light": "#FFFFFF",
    "text_primary": "#E8EAED",
    "text_secondary": "#9AA0A6",
}

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MED = 0.5
