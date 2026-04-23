# 🧠 IntelliRetrieve AI — Intelligent Question Answering System

A production-ready Intelligent Question Answering System that answers user questions from uploaded documents (PDFs, textbooks, lecture notes) using classical NLP + semantic search — featuring a unified workspace, knowledge graph exploration, and hybrid retrieval.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                      │
│  ┌─────────────┐  ┌───────────┐  ┌───────────────────┐  │
│  │  Workspace   │  │ Analytics │  │  Knowledge Graph  │  │
│  │ (Upload+Q&A) │  │ Dashboard │  │     Explorer      │  │
│  └─────────────┘  └───────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────┘
     │                │                      │
     ▼                ▼                      ▼
┌──────────┐   ┌───────────────┐    ┌──────────────┐
│ Document │   │  QA Pipeline  │    │  Knowledge   │
│ Ingestion│   │ Orchestrator  │    │ Graph Builder│
└──────────┘   └───────────────┘    └──────────────┘
     │                │                      │
     ▼                ├──────────────┐       ▼
┌──────────────┐      ▼              ▼  ┌───────────────┐
│ Preprocessor │  ┌────────────┐ ┌──────────┐│ Dep. Parsing  │
│ (spaCy)      │  │  Question  │ │Retriever ││ Triple Extract│
│ Tokenize     │  │  Analyzer  │ │(FAISS +  ││ Co-occurrence │
│ POS Tag      │  │  (POS/NER) │ │  BM25)   │└───────────────┘
│ NER          │  └────────────┘ └──────────┘
└──────────────┘        │              │
     │                  ▼              ▼
     ▼           ┌────────────┐ ┌──────────────┐
┌──────────────┐ │  Keyword   │ │  Top-K       │
│  Chunker     │ │ Extraction │ │  Passages    │
│ (smart split)│ └────────────┘ └──────────────┘
└──────────────┘        │              │
     │                  └──────┬───────┘
     ▼                         ▼
┌──────────────┐     ┌──────────────────┐
│  Embedder    │     │ Answer Extractor  │
│ (Sentence-   │     │ (Passage Ranking  │
│  Transformers│     │  + Synthesis)     │
└──────────────┘     └──────────────────┘
     │                         │
     ▼                         ▼
┌──────────────┐     ┌──────────────────┐
│  FAISS Index │     │  Final Answer    │
│  (stored on  │     │  with Source     │
│   disk)      │     │  Citations       │
└──────────────┘     └──────────────────┘
```

---

## ✨ Features

- **🧠 Unified Workspace**: Upload documents and ask questions — all in a single integrated interface
- **📤 Document Upload**: Support for PDF, TXT, and DOCX files with inline attachment
- **🔍 Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) + Reciprocal Rank Fusion
- **🎯 Cross-Encoder Re-ranking**: Fine-grained passage scoring with `ms-marco-MiniLM`
- **📝 Sentence-Aware Chunking**: Intelligent text splitting that preserves sentence boundaries
- **💡 Question Analysis**: Auto-detect WHO/WHAT/WHEN/WHERE/WHY/HOW/DEFINE question types
- **📖 Supporting Passages**: View the exact text passages that answer your questions
- **🧠 NLP Breakdown**: POS tags, keywords, entity extraction, and retrieval score details
- **🏷️ Entity Highlights**: Named entities color-coded in answers
- **📊 Analytics Dashboard**: Question type distribution, confidence trends, document coverage
- **🌐 Knowledge Graph Explorer**: Automatic entity-relationship extraction, interactive network visualization, triple tables, co-occurrence heatmaps
- **💾 Persistent Index**: FAISS index saved to disk for instant loading
- **🐳 Docker Ready**: Containerized deployment with Docker Compose

---

## 🛠️ NLP Techniques

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
| Dependency Parsing | spaCy | Knowledge Graph | Subject → Verb → Object triple extraction |
| Co-occurrence Analysis | spaCy | Knowledge Graph | Entity pairs in same sentence |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sathwikkompalli1/NLP.git
cd NLP

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Copy environment file
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac

# 6. Run the application
streamlit run app/main.py
```

The app will open at `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload Documents**: Attach PDF/TXT/DOCX files in the Workspace
2. **Build Index**: Click "Build Search Index" to index your documents
3. **Ask Questions**: Type your question and click Ask to get answers with source citations
4. **Explore Results**: Expand Supporting Passage, NLP Breakdown, and Entity Highlights for deep analysis
5. **View Analytics**: Check the Analytics page for query insights and confidence trends
6. **Explore Knowledge Graph**: Build interactive entity-relationship graphs from your text

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_preprocessor.py -v
pytest tests/test_retriever.py -v
pytest tests/test_pipeline.py -v
```

---

## 🐳 Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t intelliretrieve .
docker run -p 8501:8501 -v ./data:/app/data -v ./models:/app/models intelliretrieve
```

---

## 📁 Project Structure

```
NLP/
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── ui/
│   │   ├── components.py        # Reusable UI widgets
│   │   └── styles.css           # Custom CSS (premium dark theme)
│   └── views/
│       ├── workspace.py         # Unified Upload + Q&A workspace
│       ├── analytics.py         # Query analytics dashboard
│       └── knowledge.py         # Knowledge Graph Explorer page
├── core/
│   ├── document_loader.py       # PDF/TXT/DOCX ingestion
│   ├── indexer.py               # FAISS index builder
│   ├── retriever.py             # Semantic search + BM25 hybrid
│   ├── answer_extractor.py      # Passage ranking + answer synthesis
│   └── pipeline.py              # End-to-end QA pipeline
├── nlp/
│   ├── tokenizer.py             # spaCy tokenization
│   ├── pos_tagger.py            # POS tagging + noun phrases
│   ├── ner.py                   # Named entity recognition
│   ├── embedder.py              # Sentence-Transformers embedding
│   ├── similarity.py            # Cosine similarity utils
│   └── knowledge_graph.py       # Entity-relationship graph builder
├── utils/
│   ├── chunker.py               # Smart text chunking
│   ├── cleaner.py               # Text normalization
│   ├── logger.py                # Logging setup
│   └── config.py                # Global configuration
├── data/
│   ├── uploads/                 # User-uploaded documents
│   ├── processed/               # Processed text
│   └── sample_docs/             # Sample demo data
├── models/
│   ├── faiss_index/             # Saved FAISS indexes
│   └── embeddings_cache/        # Cached embeddings
├── tests/
│   ├── test_preprocessor.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🌐 Knowledge Graph Explorer

The Knowledge Graph page automatically discovers relationships in text:

- **Triple Extraction**: Subject → Relation → Object triples via dependency parsing
- **Interactive Network Graph**: Plotly-powered entity network with hover details
- **Entity Frequency Chart**: Horizontal bar chart of most mentioned entities
- **Co-occurrence Heatmap**: Which entities appear together in the same sentences
- **Source Sentences**: View the original text behind each extracted relationship

Powered by spaCy's dependency parser and NER — no external API required.

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Answers are extractive (not generative) — limited to text found in documents
- Single-language support (English only via `en_core_web_sm`)
- CPU-only inference (no GPU acceleration)
- No authentication or multi-user support

### Future Enhancements
- [ ] Add generative answers using LLM (GPT, Gemini)
- [ ] Multi-language support with multilingual models
- [ ] GPU acceleration for faster embedding
- [ ] User authentication and document management
- [ ] API endpoint for programmatic access
- [ ] Fine-tuning on domain-specific data
- [ ] Conversational follow-up questions

---

## 🔧 Tools & Versions

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| spaCy | 3.8+ | Tokenization, POS, NER, Dependency Parsing |
| Sentence-Transformers | 5.x | Semantic embeddings |
| FAISS-CPU | 1.13+ | Vector index + search |
| rank-bm25 | 0.2.2 | Sparse retrieval |
| PyMuPDF | 1.27+ | PDF text extraction |
| python-docx | 1.2+ | DOCX parsing |
| HuggingFace Transformers | 5.x | Cross-encoder reranking |
| Streamlit | 1.56+ | Web frontend |
| Plotly | 5.19+ | Analytics & graph visualization |

---

*IntelliRetrieve AI v1.0*
