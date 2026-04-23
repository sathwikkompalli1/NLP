# 🧠 INTELLIGENT QUESTION ANSWERING SYSTEM — MASTER BUILD PROMPT
### NLP Final Year Project | SRM University AP | Claude Opus Execution Prompt

---

## ⚙️ HOW TO USE THIS PROMPT

> Copy everything inside the **`OPUS PROMPT`** section below and paste it into Claude Opus as a single message. Opus will build the full project end-to-end.

---

---

# 🔷 OPUS PROMPT — START

---

You are a senior NLP engineer and full-stack developer. Build a **complete, production-ready Intelligent Question Answering System (IQAS)** from scratch. This system answers user questions from a document corpus (PDFs, textbooks, lecture notes) using classical NLP + semantic search.

Follow every instruction below **exactly**. Do not skip any section. Build the full system in order.

---

## 📐 PROJECT IDENTITY

| Field | Value |
|---|---|
| **Project Name** | IQAS — Intelligent Question Answering System |
| **Domain** | Natural Language Processing (NLP) |
| **Purpose** | Answer user questions from uploaded documents using tokenization, POS tagging, semantic similarity, and FAISS-based information retrieval |
| **Stack** | Python 3.10+, spaCy, Sentence-Transformers, FAISS, Streamlit |
| **Deployment** | Local (Streamlit) + optional Docker |

---

## 📁 FULL PROJECT STRUCTURE

Create the following directory and file structure **exactly**:

```
iqas/
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── ui/
│   │   ├── components.py        # Reusable UI widgets
│   │   └── styles.css           # Custom CSS for Streamlit
│   └── pages/
│       ├── upload.py            # Document upload page
│       ├── qa.py                # Q&A interface page
│       └── analytics.py        # Query analytics page
│
├── core/
│   ├── __init__.py
│   ├── document_loader.py       # PDF/TXT/DOCX ingestion
│   ├── preprocessor.py          # Tokenization, POS, NER, cleaning
│   ├── indexer.py               # FAISS index builder
│   ├── retriever.py             # Semantic search + BM25 hybrid
│   ├── answer_extractor.py      # Passage ranking + answer synthesis
│   └── pipeline.py              # End-to-end QA pipeline orchestrator
│
├── nlp/
│   ├── __init__.py
│   ├── tokenizer.py             # Custom spaCy tokenization
│   ├── pos_tagger.py            # POS tagging + noun phrase extraction
│   ├── ner.py                   # Named entity recognition
│   ├── embedder.py              # Sentence-Transformers embedding
│   └── similarity.py            # Cosine + semantic similarity utils
│
├── models/
│   ├── faiss_index/             # Saved FAISS index files
│   │   └── .gitkeep
│   └── embeddings_cache/        # Cached document embeddings
│       └── .gitkeep
│
├── data/
│   ├── uploads/                 # User-uploaded documents
│   │   └── .gitkeep
│   ├── processed/               # Chunked + processed text
│   │   └── .gitkeep
│   └── sample_docs/             # Sample textbooks/notes for demo
│       └── sample_nlp_notes.txt
│
├── utils/
│   ├── __init__.py
│   ├── chunker.py               # Smart text chunking strategies
│   ├── cleaner.py               # Text normalization utilities
│   ├── logger.py                # Logging setup
│   └── config.py                # Global config / constants
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_retriever.py
│   ├── test_pipeline.py
│   └── fixtures/
│       └── sample_qa.json
│
├── notebooks/
│   └── exploration.ipynb        # Development + analysis notebook
│
├── requirements.txt
├── setup.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧩 SYSTEM DESIGN — ARCHITECTURE

### High-Level Data Flow

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│  ┌──────────┐   ┌───────────┐   ┌────────────────────┐  │
│  │  Upload  │   │  Q&A UI   │   │ Analytics Dashboard │  │
│  │  Page    │   │  Page     │   │  Page               │  │
│  └──────────┘   └───────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────┘
     │                    │
     ▼                    ▼
┌──────────┐      ┌───────────────┐
│ Document │      │  QA Pipeline  │
│ Ingestion│      │ Orchestrator  │
└──────────┘      └───────────────┘
     │                    │
     ▼                    ├──────────────────────┐
┌──────────────┐          ▼                      ▼
│ Preprocessor │   ┌────────────┐        ┌──────────────┐
│ (spaCy)      │   │  Question  │        │   Retriever  │
│ Tokenize     │   │  Analyzer  │        │  (FAISS +    │
│ POS Tag      │   │  (POS/NER) │        │   BM25)      │
│ NER          │   └────────────┘        └──────────────┘
└──────────────┘          │                      │
     │                    ▼                      ▼
     ▼             ┌────────────┐        ┌──────────────┐
┌──────────────┐   │  Keyword   │        │  Top-K       │
│  Chunker     │   │ Extraction │        │  Passages    │
│ (smart split)│   └────────────┘        └──────────────┘
└──────────────┘          │                      │
     │                    └──────────┬───────────┘
     ▼                               ▼
┌──────────────┐            ┌──────────────────┐
│  Embedder    │            │ Answer Extractor  │
│ (Sentence-   │            │ (Passage Ranking  │
│  Transformers│            │  + Synthesis)     │
└──────────────┘            └──────────────────┘
     │                               │
     ▼                               ▼
┌──────────────┐            ┌──────────────────┐
│  FAISS Index │            │  Final Answer    │
│  (stored on  │            │  with Source     │
│   disk)      │            │  Citations       │
└──────────────┘            └──────────────────┘
```

---

## 🔬 NLP PIPELINE — DETAILED DESIGN

### Stage 1: Document Ingestion (`core/document_loader.py`)
- Accept PDF, TXT, DOCX file uploads via Streamlit
- Extract raw text using PyMuPDF (fitz) for PDFs, python-docx for DOCX
- Preserve page numbers and document metadata (title, source, page)
- Output: list of `Document` objects with `{text, source, page, doc_id}`

### Stage 2: Preprocessing (`core/preprocessor.py` + `nlp/`)
Run all of the following using **spaCy** (`en_core_web_sm`):

1. **Tokenization** (`nlp/tokenizer.py`)
   - Word tokenization using spaCy's tokenizer
   - Sentence boundary detection
   - Handle abbreviations, hyphenated words, URLs

2. **POS Tagging** (`nlp/pos_tagger.py`)
   - Assign Universal POS tags (NOUN, VERB, ADJ, etc.)
   - Extract Noun Phrases (NP chunks) as candidate answers
   - Filter stopwords using POS-aware filtering

3. **Named Entity Recognition** (`nlp/ner.py`)
   - Extract named entities (PERSON, ORG, DATE, CONCEPT)
   - Use entities to boost relevance scoring of passages
   - Tag entities in retrieved passages for highlighting

4. **Text Cleaning** (`utils/cleaner.py`)
   - Lowercase, remove special chars, normalize whitespace
   - Remove headers/footers from PDF extraction
   - Handle LaTeX equations (strip or preserve)

### Stage 3: Chunking (`utils/chunker.py`)
Implement **three chunking strategies** (configurable):
- **Fixed-size**: 512 tokens with 50-token overlap
- **Sentence-aware**: Group sentences until ~400 tokens, no mid-sentence splits
- **Paragraph-aware**: Split on double newlines, then size-limit

Each chunk stores: `{chunk_id, text, doc_id, source, page, start_char, end_char}`

### Stage 4: Embedding (`nlp/embedder.py`)
- Use `sentence-transformers/all-MiniLM-L6-v2` as default model
- Also support `multi-qa-MiniLM-L6-cos-v1` (QA-optimized)
- Batch encode all chunks (batch_size=32)
- Normalize vectors (L2) for cosine similarity via dot product
- Cache embeddings as `.npy` files to avoid recomputation

### Stage 5: Indexing (`core/indexer.py`)
Build a **hybrid FAISS index**:
- Primary: `faiss.IndexFlatIP` (Inner Product for cosine similarity after L2 norm)
- For large corpora (>10k chunks): `faiss.IndexIVFFlat` with nlist=100
- Store chunk metadata separately as JSON alongside index
- Save index to `models/faiss_index/`
- Support incremental updates (add new documents without full rebuild)

### Stage 6: Retrieval (`core/retriever.py`)
Implement **hybrid retrieval**:

1. **Dense Retrieval (FAISS)**: Embed the query, search top-K=20 nearest chunks
2. **Sparse Retrieval (BM25)**: Use `rank_bm25` library, score chunks by TF-IDF-like BM25
3. **Fusion**: Reciprocal Rank Fusion (RRF) to merge dense and sparse results
4. **Re-ranking**: Re-score fused top-10 with cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`

Return top-5 passages with scores and metadata.

### Stage 7: Question Analysis (`nlp/pos_tagger.py`, `nlp/ner.py`)
When a question arrives:
- Detect question type: WHO / WHAT / WHEN / WHERE / WHY / HOW / DEFINE
- Extract key entities and noun phrases from the question
- Use question type to prioritize entity types in retrieved passages
  - WHO → prioritize PERSON, ORG entities
  - WHEN → prioritize DATE, TIME entities
  - WHAT/DEFINE → prioritize noun phrases, definitions

### Stage 8: Answer Extraction (`core/answer_extractor.py`)
1. Select the **best passage** from top-5 retrieved chunks
2. Within the passage, find the **most relevant sentence(s)** using sentence-level cosine similarity to the question
3. Extract answer span:
   - For factoid Qs: extract the relevant 1-3 sentences
   - For definitional Qs: extract the full definition sentence + following sentence
4. Add **source citation**: document name + page number
5. **Confidence score**: weighted average of retrieval score + re-ranker score

---

## 💻 COMPLETE CODE — ALL FILES

Write **complete, working, production-quality Python code** for every file listed. Do not write placeholders or `# TODO` comments. Every function must be fully implemented.

### `utils/config.py`
```python
# Write full config with:
# - MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# - RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  
# - CHUNK_SIZE = 512
# - CHUNK_OVERLAP = 50
# - TOP_K_RETRIEVE = 20
# - TOP_K_RERANK = 5
# - FAISS_INDEX_PATH
# - UPLOAD_DIR
# - PROCESSED_DIR
# - LOG_LEVEL
# - All paths as pathlib.Path objects
```

### `utils/logger.py`
```python
# Structured logger with:
# - File handler (logs/iqas.log)
# - Console handler
# - Timestamps, module names
# - Log levels: DEBUG for dev, INFO for production
```

### `utils/cleaner.py`
```python
# Implement:
# clean_text(text: str) -> str
# - Remove PDF artifacts (form feeds, excessive whitespace)
# - Normalize unicode (unicodedata.normalize NFKD)
# - Remove non-printable chars
# - Fix hyphenated line breaks ("computa-\ntion" -> "computation")
# - Preserve sentence boundaries
```

### `utils/chunker.py`
```python
# Implement all three chunking strategies:
# class TextChunker:
#   chunk_fixed(text, chunk_size, overlap) -> List[Chunk]
#   chunk_by_sentence(text, max_tokens) -> List[Chunk]
#   chunk_by_paragraph(text, max_tokens) -> List[Chunk]
# 
# Use spaCy for sentence detection.
# Each chunk is a dataclass with all metadata fields.
```

### `core/document_loader.py`
```python
# class DocumentLoader:
#   load_pdf(path) -> List[Document]      # use PyMuPDF (fitz)
#   load_txt(path) -> List[Document]
#   load_docx(path) -> List[Document]    # use python-docx
#   load_any(path) -> List[Document]     # auto-detect by extension
#   batch_load(paths) -> List[Document]
#
# Document dataclass: id, text, source, filename, page_num, total_pages
```

### `nlp/tokenizer.py`
```python
# class NLPTokenizer:
#   __init__: load spaCy en_core_web_sm
#   tokenize(text) -> List[Token]
#   sent_tokenize(text) -> List[str]
#   word_tokenize(text, remove_stopwords=False) -> List[str]
#   get_lemmas(text) -> List[str]
```

### `nlp/pos_tagger.py`
```python
# class POSTagger:
#   tag(text) -> List[Tuple[str, str]]        # (word, POS tag)
#   get_noun_phrases(text) -> List[str]        # NP chunks
#   get_keywords(text, top_n=10) -> List[str]  # NOUN + PROPN + key VERB lemmas
#   detect_question_type(question) -> str      # WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE
#   extract_question_focus(question) -> List[str]  # key terms
```

### `nlp/ner.py`
```python
# class NERExtractor:
#   extract(text) -> List[Entity]    # Entity: text, label, start, end
#   get_entities_by_type(text, label) -> List[str]
#   highlight_entities(text) -> str  # returns HTML with highlighted entities
#   get_answer_entities(text, q_type) -> List[str]  # focused on q_type
```

### `nlp/similarity.py`
```python
# Implement:
# cosine_similarity(v1, v2) -> float
# batch_cosine_similarity(query_vec, matrix) -> np.ndarray
# sentence_similarity(sent1, sent2, model) -> float
# find_most_similar_sentence(query, sentences, model) -> Tuple[str, float]
```

### `nlp/embedder.py`
```python
# class TextEmbedder:
#   __init__(model_name): load SentenceTransformer, set device (cuda/cpu)
#   embed(text: str) -> np.ndarray
#   embed_batch(texts: List[str], batch_size=32) -> np.ndarray
#   embed_and_cache(texts, cache_path) -> np.ndarray  # load from cache if exists
#   normalize(vectors) -> np.ndarray
```

### `core/indexer.py`
```python
# class FAISSIndexer:
#   __init__(dim=384)
#   build(embeddings, chunks_metadata)  # builds IndexFlatIP
#   build_ivf(embeddings, chunks_metadata, nlist=100)  # for large corpus
#   save(path)
#   load(path)
#   add(new_embeddings, new_metadata)   # incremental update
#   get_chunk_by_id(chunk_id) -> dict
```

### `core/retriever.py`
```python
# class HybridRetriever:
#   __init__(indexer, embedder, chunks)
#   dense_search(query, top_k=20) -> List[RetrievedChunk]
#   bm25_search(query, top_k=20) -> List[RetrievedChunk]
#   reciprocal_rank_fusion(dense_results, bm25_results) -> List[RetrievedChunk]
#   rerank(query, candidates) -> List[RetrievedChunk]  # cross-encoder
#   retrieve(query, top_k=5) -> List[RetrievedChunk]   # full hybrid pipeline
```

### `core/answer_extractor.py`
```python
# class AnswerExtractor:
#   __init__(embedder)
#   extract_answer(question, passages) -> Answer
#   find_best_sentences(question, passage_text, n=2) -> List[str]
#   compute_confidence(retrieval_score, rerank_score) -> float
#   format_answer(sentences, source, page, confidence) -> Answer
#
# Answer dataclass: text, source, page, confidence, supporting_passage, entities
```

### `core/pipeline.py`
```python
# class QAPipeline:
#   __init__: initialize all components lazily
#   ingest_documents(paths) -> IndexStats
#   ask(question: str) -> Answer
#   load_index(path)
#   get_stats() -> dict   # num docs, num chunks, index size
```

### `app/main.py`
```python
# Streamlit multi-page app entrypoint:
# - st.set_page_config (wide layout, custom title, brain emoji icon)
# - Sidebar navigation with icons
# - Initialize QAPipeline in st.session_state (load once)
# - Route to: Upload, Q&A, Analytics pages
# - Custom CSS injection from styles.css
# - Show system status (index loaded, doc count) in sidebar
```

### `app/pages/upload.py`
```python
# Streamlit page:
# - st.file_uploader (accept PDF, TXT, DOCX, multiple files)
# - Progress bar during ingestion
# - Show extracted text preview per document
# - Show NLP stats: token count, sentence count, top entities
# - Chunking strategy selector (radio: fixed / sentence / paragraph)
# - "Build Index" button → trigger pipeline.ingest_documents()
# - Show index build time, chunk count
# - Success/error states with proper messaging
```

### `app/pages/qa.py`
```python
# Streamlit page:
# - Text input for question
# - "Ask" button
# - Display answer prominently with confidence badge (color-coded)
# - Show source citation (document name + page)
# - Expandable "Supporting Passage" section
# - Expandable "NLP Breakdown" section showing:
#   - Question type detected
#   - POS tags of question (table)
#   - Key entities in answer (highlighted HTML)
#   - Similarity scores of top-5 retrieved passages
# - Chat history: show last 10 Q&A pairs
# - "Clear History" button
```

### `app/pages/analytics.py`
```python
# Streamlit page:
# - Queries per session (bar chart using st.bar_chart)
# - Question type distribution (pie chart via plotly)
# - Top entities across all answers
# - Average confidence scores
# - Document coverage heatmap (which docs were retrieved most)
# - Export analytics as CSV
```

---

## 📦 REQUIREMENTS FILE

Write `requirements.txt` with **exact version pins**:
```
streamlit==1.32.0
spacy==3.7.4
sentence-transformers==2.6.1
faiss-cpu==1.7.4
rank-bm25==0.2.2
pymupdf==1.23.26
python-docx==1.1.0
transformers==4.38.2
torch==2.2.1
numpy==1.26.4
pandas==2.2.1
plotly==5.19.0
python-dotenv==1.0.1
tqdm==4.66.2
loguru==0.7.2
pydantic==2.6.3
```

Also write the **setup command** in README:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🎨 UI/UX DESIGN SPECIFICATION

### Color Scheme
- Primary: `#1E3A5F` (deep navy)
- Accent: `#00B4D8` (cyan-blue)
- Success: `#2ECC71`
- Warning: `#F39C12`
- Background: `#0F1117` (dark) / `#FFFFFF` (light)
- Confidence High (>0.8): green badge
- Confidence Med (0.5–0.8): yellow badge
- Confidence Low (<0.5): red badge

### Custom CSS (`app/ui/styles.css`)
Write full CSS for:
- Custom answer card with left blue border
- Confidence badge (pill shape, color-coded)
- Entity highlight spans (inline colored backgrounds per type)
- Source citation chip style
- Upload drag-and-drop zone styling
- Sidebar branding area

---

## 🧪 TESTS

Write pytest tests for:

### `tests/test_preprocessor.py`
- `test_tokenize_basic()`: verify token count on known string
- `test_pos_tags_sentence()`: verify NOUN/VERB tags detected
- `test_ner_extracts_person()`: verify PERSON entity from "Dr. Smith teaches NLP"
- `test_chunker_fixed_size()`: verify chunk count and overlap correctness
- `test_chunker_no_mid_sentence_split()`: verify sentence-aware mode

### `tests/test_retriever.py`
- `test_bm25_returns_relevant()`: verify BM25 ranks relevant chunk higher
- `test_faiss_cosine_similarity()`: verify cosine similarity ordering
- `test_rrf_fusion()`: verify reciprocal rank fusion correctness
- `test_retrieve_returns_top5()`: verify final retrieve() count

### `tests/test_pipeline.py`
- `test_ingest_and_ask()`: end-to-end test with sample_nlp_notes.txt
- `test_answer_has_source()`: verify source citation present
- `test_confidence_range()`: verify confidence in [0, 1]

---

## 🐳 DOCKER CONFIGURATION

### `Dockerfile`
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### `docker-compose.yml`
```yaml
version: "3.9"
services:
  iqas:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

---

## 📊 SAMPLE DATA

Create `data/sample_docs/sample_nlp_notes.txt` with ~500 words of realistic NLP lecture notes covering:
- Definition of Natural Language Processing
- Tokenization explanation
- POS tagging with examples
- Named Entity Recognition
- Word embeddings and word2vec
- Transformers and BERT overview

This file is used for demo purposes and tests.

---

## 📖 README.md

Write a complete README with:
1. Project overview
2. Architecture diagram (ASCII)
3. Setup instructions (venv + pip + spacy download)
4. How to run (`streamlit run app/main.py`)
5. How to run tests (`pytest tests/`)
6. Docker instructions
7. Feature list
8. NLP techniques table (Technique | Library | Purpose | Stage)
9. Limitations and future work

---

## ✅ EXECUTION ORDER

Build all files in this exact order to avoid import errors:
1. `utils/config.py`
2. `utils/logger.py`
3. `utils/cleaner.py`
4. `utils/chunker.py`
5. `nlp/__init__.py` + `nlp/tokenizer.py`
6. `nlp/pos_tagger.py`
7. `nlp/ner.py`
8. `nlp/similarity.py`
9. `nlp/embedder.py`
10. `core/__init__.py` + `core/document_loader.py`
11. `core/indexer.py`
12. `core/retriever.py`
13. `core/answer_extractor.py`
14. `core/pipeline.py`
15. `app/ui/styles.css` + `app/ui/components.py`
16. `app/pages/upload.py`
17. `app/pages/qa.py`
18. `app/pages/analytics.py`
19. `app/main.py`
20. All `tests/` files
21. `data/sample_docs/sample_nlp_notes.txt`
22. `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`

---

## 🚀 FINAL DELIVERABLES CHECK

Before finishing, verify:
- [ ] All imports resolve correctly across modules
- [ ] No circular imports
- [ ] spaCy model loaded once and shared (not reloaded per call)
- [ ] SentenceTransformer loaded once in session_state
- [ ] FAISS index persists across Streamlit reruns via session_state
- [ ] All dataclasses have proper type hints
- [ ] Chunker handles edge cases: empty text, single sentence, very long paragraphs
- [ ] Answer includes citation even when confidence is low
- [ ] CSS file is injected correctly into Streamlit

---

# 🔷 OPUS PROMPT — END

---

---

## 📋 QUICK REFERENCE: NLP TECHNIQUES SUMMARY

| Technique | Library | Stage | Purpose |
|---|---|---|---|
| Tokenization | spaCy | Preprocessing | Split text into words/sentences |
| POS Tagging | spaCy | Preprocessing | Identify nouns, verbs for keywords |
| NER | spaCy | Preprocessing + Answer | Extract named entities |
| Text Cleaning | Python/unicodedata | Preprocessing | Normalize raw text |
| Sentence Splitting | spaCy | Chunking | Preserve sentence boundaries |
| Sentence Embeddings | Sentence-Transformers | Embedding | Dense vector representations |
| Cosine Similarity | numpy/FAISS | Retrieval | Measure semantic similarity |
| BM25 | rank-bm25 | Retrieval | Sparse keyword matching |
| RRF Fusion | Custom | Retrieval | Combine dense + sparse scores |
| Cross-Encoder Reranking | HuggingFace | Reranking | Fine-grained passage scoring |
| Noun Phrase Extraction | spaCy | Answer | Candidate answer spans |
| Confidence Scoring | Custom | Answer | Quality estimation |

---

## 🛠️ TOOLS & VERSIONS REFERENCE

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| spaCy | 3.7.4 | Tokenization, POS, NER |
| Sentence-Transformers | 2.6.1 | Semantic embeddings |
| FAISS-CPU | 1.7.4 | Vector index + search |
| rank-bm25 | 0.2.2 | Sparse retrieval |
| PyMuPDF | 1.23.26 | PDF text extraction |
| python-docx | 1.1.0 | DOCX parsing |
| HuggingFace Transformers | 4.38.2 | Cross-encoder reranking |
| Streamlit | 1.32.0 | Web frontend |
| Plotly | 5.19.0 | Analytics charts |

---

*Generated for NLP Project — SRM University AP | B.Tech CSE (AI & ML)*
