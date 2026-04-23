"""
IQAS Tests — Retriever
========================
Tests for BM25, FAISS, RRF fusion, and full hybrid retrieval.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np


class TestFAISSIndexer:
    """Tests for FAISSIndexer."""

    def setup_method(self):
        from core.indexer import FAISSIndexer
        self.indexer = FAISSIndexer(dim=384)

    def test_build_and_search(self):
        """Verify basic FAISS index build and search."""
        # Create random normalized embeddings
        embeddings = np.random.randn(10, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        metadata = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(10)]

        self.indexer.build(embeddings, metadata)
        assert self.indexer.size == 10

        # Search
        query = embeddings[0]  # Should find itself
        results = self.indexer.search(query, top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0.9  # Should be very similar to itself

    def test_faiss_cosine_similarity(self):
        """Verify cosine similarity ordering in FAISS."""
        dim = 384
        # Create a target and two vectors — one similar, one different
        target = np.random.randn(dim).astype(np.float32)
        similar = target + np.random.randn(dim).astype(np.float32) * 0.1  # Small perturbation
        different = np.random.randn(dim).astype(np.float32)

        # Normalize
        target = target / np.linalg.norm(target)
        similar = similar / np.linalg.norm(similar)
        different = different / np.linalg.norm(different)

        embeddings = np.array([similar, different])
        metadata = [
            {"chunk_id": "similar", "text": "similar chunk"},
            {"chunk_id": "different", "text": "different chunk"},
        ]

        self.indexer.build(embeddings, metadata)
        results = self.indexer.search(target.reshape(1, -1), top_k=2)

        assert results[0]["chunk_id"] == "similar"

    def test_incremental_add(self):
        """Verify incremental vector addition."""
        embeddings1 = np.random.randn(5, 384).astype(np.float32)
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        meta1 = [{"chunk_id": f"c{i}"} for i in range(5)]

        self.indexer.build(embeddings1, meta1)
        assert self.indexer.size == 5

        embeddings2 = np.random.randn(3, 384).astype(np.float32)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        meta2 = [{"chunk_id": f"c{i}"} for i in range(5, 8)]

        self.indexer.add(embeddings2, meta2)
        assert self.indexer.size == 8


class TestBM25:
    """Tests for BM25 retrieval."""

    def test_bm25_returns_relevant(self):
        """Verify BM25 ranks relevant chunk higher."""
        from core.indexer import FAISSIndexer
        from nlp.embedder import TextEmbedder
        from core.retriever import HybridRetriever

        embedder = TextEmbedder()

        chunks = [
            {"chunk_id": "c0", "text": "The weather is sunny and warm today.", "source": "doc1", "page": 1, "doc_id": "d1"},
            {"chunk_id": "c1", "text": "Tokenization splits text into individual words and subwords.", "source": "doc2", "page": 1, "doc_id": "d2"},
            {"chunk_id": "c2", "text": "Dogs and cats are popular household pets.", "source": "doc3", "page": 1, "doc_id": "d3"},
        ]

        # Build embeddings and index
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        indexer = FAISSIndexer(dim=embedder.dim)
        indexer.build(embeddings, chunks)

        retriever = HybridRetriever(indexer, embedder, chunks)

        # BM25 search for tokenization
        results = retriever.bm25_search("tokenization splits text words", top_k=3)
        assert len(results) > 0
        # The tokenization chunk should appear in results
        result_texts = " ".join(r.text.lower() for r in results)
        assert "tokenization" in result_texts


class TestRRF:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_fusion(self):
        """Verify reciprocal rank fusion correctness."""
        from core.retriever import HybridRetriever, RetrievedChunk
        from core.indexer import FAISSIndexer
        from nlp.embedder import TextEmbedder

        # Create mock results
        dense = [
            RetrievedChunk(chunk_id="c1", text="chunk 1", score=0.9, dense_score=0.9),
            RetrievedChunk(chunk_id="c2", text="chunk 2", score=0.7, dense_score=0.7),
            RetrievedChunk(chunk_id="c3", text="chunk 3", score=0.5, dense_score=0.5),
        ]
        bm25 = [
            RetrievedChunk(chunk_id="c2", text="chunk 2", score=5.0, sparse_score=5.0),
            RetrievedChunk(chunk_id="c4", text="chunk 4", score=3.0, sparse_score=3.0),
            RetrievedChunk(chunk_id="c1", text="chunk 1", score=2.0, sparse_score=2.0),
        ]

        # Create minimal retriever for RRF method
        embedder = TextEmbedder()
        indexer = FAISSIndexer()
        chunks = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(5)]

        # Build a valid index
        emb = embedder.embed_batch([c["text"] for c in chunks])
        indexer.build(emb, chunks)

        retriever = HybridRetriever(indexer, embedder, chunks)

        fused = retriever.reciprocal_rank_fusion(dense, bm25, k=60)

        # c2 appears in both lists — should have highest fused score
        assert fused[0].chunk_id == "c2"

    def test_retrieve_returns_top5(self):
        """Verify final retrieve() count."""
        from core.indexer import FAISSIndexer
        from nlp.embedder import TextEmbedder
        from core.retriever import HybridRetriever

        embedder = TextEmbedder()

        chunks = [
            {"chunk_id": f"c{i}", "text": f"This is test chunk number {i} about NLP.", "source": "test", "page": 1, "doc_id": "d1"}
            for i in range(20)
        ]

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        indexer = FAISSIndexer(dim=embedder.dim)
        indexer.build(embeddings, chunks)

        retriever = HybridRetriever(indexer, embedder, chunks)
        results = retriever.retrieve("What is NLP?", top_k=5)

        assert len(results) <= 5
        assert len(results) > 0
