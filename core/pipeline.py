"""
IQAS QA Pipeline Orchestrator
===============================
End-to-end pipeline: document ingestion → indexing → question answering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.config import (
    FAISS_INDEX_DIR,
    EMBEDDING_CACHE_PATH,
    TOP_K_RERANK,
)
from utils.logger import get_logger

log = get_logger("pipeline")


# ──────────────────────────── Stats Model ────────────────────────────


@dataclass
class IndexStats:
    """Statistics about the ingested document corpus."""
    num_documents: int
    num_pages: int
    num_chunks: int
    total_tokens: int
    index_time_seconds: float
    chunking_strategy: str


# ──────────────────────────── QA Pipeline ────────────────────────────


class QAPipeline:
    """
    End-to-end Question Answering Pipeline.

    Orchestrates:
        - Document loading and preprocessing
        - Text chunking and embedding
        - FAISS index building
        - Hybrid retrieval (Dense + BM25 + RRF + re-ranking)
        - Answer extraction with confidence and citations
    """

    def __init__(self):
        """Initialize pipeline components lazily."""
        self._document_loader = None
        self._chunker = None
        self._embedder = None
        self._indexer = None
        self._retriever = None
        self._answer_extractor = None
        self._cleaner = None
        self._pos_tagger = None

        # State
        self._chunks_data: List[Dict] = []
        self._is_indexed = False
        self._stats: Optional[IndexStats] = None

    # ──────────────────── Lazy Component Access ────────────────────

    @property
    def document_loader(self):
        if self._document_loader is None:
            from core.document_loader import DocumentLoader
            self._document_loader = DocumentLoader()
        return self._document_loader

    @property
    def chunker(self):
        if self._chunker is None:
            from utils.chunker import TextChunker
            self._chunker = TextChunker()
        return self._chunker

    @property
    def embedder(self):
        if self._embedder is None:
            from nlp.embedder import TextEmbedder
            self._embedder = TextEmbedder()
        return self._embedder

    @property
    def indexer(self):
        if self._indexer is None:
            from core.indexer import FAISSIndexer
            self._indexer = FAISSIndexer()
        return self._indexer

    @property
    def cleaner(self):
        if self._cleaner is None:
            from utils.cleaner import clean_text
            self._cleaner = clean_text
        return self._cleaner

    @property
    def pos_tagger(self):
        if self._pos_tagger is None:
            from nlp.pos_tagger import POSTagger
            self._pos_tagger = POSTagger()
        return self._pos_tagger

    @property
    def answer_extractor(self):
        if self._answer_extractor is None:
            from core.answer_extractor import AnswerExtractor
            self._answer_extractor = AnswerExtractor(self.embedder)
        return self._answer_extractor

    def _ensure_retriever(self):
        """Build the retriever from current index and chunks."""
        if self._retriever is None and self._is_indexed:
            from core.retriever import HybridRetriever
            self._retriever = HybridRetriever(
                indexer=self.indexer,
                embedder=self.embedder,
                chunks=self._chunks_data,
            )

    # ──────────────────── Document Ingestion ────────────────────

    def ingest_documents(
        self,
        paths: List[Union[str, Path]],
        strategy: str = "sentence",
        progress_callback=None,
    ) -> IndexStats:
        """
        Ingest documents: load → clean → chunk → embed → index.

        Args:
            paths: List of document file paths.
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph').
            progress_callback: Optional callback(stage, progress) for UI updates.

        Returns:
            IndexStats with ingestion statistics.
        """
        start_time = time.time()

        # Stage 1: Load documents
        if progress_callback:
            progress_callback("Loading documents...", 0.1)

        documents = self.document_loader.batch_load(paths)
        if not documents:
            raise ValueError("No documents could be loaded")

        log.info(f"Loaded {len(documents)} document sections")

        # Stage 2: Clean text
        if progress_callback:
            progress_callback("Cleaning text...", 0.2)

        for doc in documents:
            doc.text = self.cleaner(doc.text)

        # Stage 3: Chunk documents
        if progress_callback:
            progress_callback("Chunking documents...", 0.3)

        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(
                text=doc.text,
                doc_id=doc.id,
                source=doc.filename,
                page=doc.page_num,
                strategy=strategy,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks generated from documents")

        log.info(f"Generated {len(all_chunks)} chunks using '{strategy}' strategy")

        # Convert chunks to metadata dicts
        self._chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "page": chunk.page,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
            }
            for chunk in all_chunks
        ]

        # Stage 4: Embed chunks
        if progress_callback:
            progress_callback("Computing embeddings...", 0.5)

        chunk_texts = [c["text"] for c in self._chunks_data]
        embeddings = self.embedder.embed_and_cache(
            chunk_texts, cache_path=EMBEDDING_CACHE_PATH
        )

        log.info(f"Computed embeddings: shape {embeddings.shape}")

        # Stage 5: Build FAISS index
        if progress_callback:
            progress_callback("Building search index...", 0.8)

        self.indexer.build(embeddings, self._chunks_data)
        self.indexer.save()
        self._is_indexed = True

        # Stage 6: Initialize retriever
        self._retriever = None  # Reset to force rebuild
        self._ensure_retriever()

        elapsed = time.time() - start_time

        self._stats = IndexStats(
            num_documents=len(set(d.filename for d in documents)),
            num_pages=len(documents),
            num_chunks=len(all_chunks),
            total_tokens=sum(c.token_count for c in all_chunks),
            index_time_seconds=round(elapsed, 2),
            chunking_strategy=strategy,
        )

        if progress_callback:
            progress_callback("Done!", 1.0)

        log.info(
            f"Ingestion complete: {self._stats.num_documents} docs, "
            f"{self._stats.num_chunks} chunks in {self._stats.index_time_seconds}s"
        )

        return self._stats

    # ──────────────────── Question Answering ────────────────────

    def ask(self, question: str, top_k: int = TOP_K_RERANK):
        """
        Answer a question using the indexed corpus.

        Args:
            question: User question string.
            top_k: Number of passages to retrieve.

        Returns:
            Answer object with text, source, confidence, and entities.
        """
        if not self._is_indexed:
            from core.answer_extractor import Answer
            return Answer(
                text="No documents have been indexed yet. Please upload and index documents first.",
                source="System",
                page=None,
                confidence=0.0,
                supporting_passage="",
            )

        self._ensure_retriever()

        # Retrieve passages
        passages = self._retriever.retrieve(question, top_k=top_k)

        # Extract answer
        answer = self.answer_extractor.extract_answer(question, passages)

        log.info(
            f"Answer: confidence={answer.confidence:.3f}, "
            f"source={answer.source}, type={answer.question_type}"
        )

        return answer

    # ──────────────────── Index Management ────────────────────

    def load_index(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load a previously saved index from disk.

        Args:
            path: Path to index directory. Uses default if None.

        Returns:
            True if loaded successfully.
        """
        loaded = self.indexer.load(path)
        if loaded:
            self._chunks_data = self.indexer.metadata
            self._is_indexed = True
            self._retriever = None  # Reset to force rebuild
            self._ensure_retriever()
            log.info(f"Loaded index with {self.indexer.size} chunks")
        return loaded

    def get_stats(self) -> Dict:
        """
        Get current system statistics.

        Returns:
            Dict with document count, chunk count, index size, etc.
        """
        return {
            "is_indexed": self._is_indexed,
            "num_chunks": len(self._chunks_data),
            "index_size": self.indexer.size if self._is_indexed else 0,
            "num_documents": len(set(c.get("source", "") for c in self._chunks_data)) if self._chunks_data else 0,
            "embedding_model": self.embedder.model_name if self._embedder else "Not loaded",
            "stats": {
                "num_documents": self._stats.num_documents if self._stats else 0,
                "num_pages": self._stats.num_pages if self._stats else 0,
                "num_chunks": self._stats.num_chunks if self._stats else 0,
                "total_tokens": self._stats.total_tokens if self._stats else 0,
                "index_time_seconds": self._stats.index_time_seconds if self._stats else 0,
                "chunking_strategy": self._stats.chunking_strategy if self._stats else "N/A",
            },
        }

    @property
    def is_ready(self) -> bool:
        """Check if the pipeline is ready to answer questions."""
        return self._is_indexed and self.indexer.size > 0
