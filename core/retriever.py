"""
IQAS Hybrid Retriever
======================
Dense (FAISS) + Sparse (BM25) retrieval with Reciprocal Rank Fusion and cross-encoder re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from core.indexer import FAISSIndexer
from nlp.embedder import TextEmbedder
from utils.config import TOP_K_RETRIEVE, TOP_K_RERANK, RRF_K, RERANKER_MODEL
from utils.logger import get_logger

log = get_logger("retriever")


# ──────────────────────────── Data Model ────────────────────────────


@dataclass
class RetrievedChunk:
    """A retrieved chunk with scores and metadata."""
    chunk_id: str
    text: str
    score: float
    source: str = ""
    page: Optional[int] = None
    doc_id: str = ""
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "page": self.page,
            "doc_id": self.doc_id,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
        }


# ──────────────────────────── Hybrid Retriever ────────────────────────────


class HybridRetriever:
    """
    Hybrid retrieval combining Dense (FAISS) and Sparse (BM25) search
    with Reciprocal Rank Fusion and cross-encoder re-ranking.
    """

    def __init__(
        self,
        indexer: FAISSIndexer,
        embedder: TextEmbedder,
        chunks: List[Dict],
    ):
        """
        Initialize the hybrid retriever.

        Args:
            indexer: FAISS indexer with pre-built index.
            embedder: Text embedder for query encoding.
            chunks: List of chunk metadata dicts (must have 'text' field).
        """
        self.indexer = indexer
        self.embedder = embedder
        self.chunks = chunks

        # Build BM25 index from chunk texts
        tokenized_corpus = [
            self._tokenize_for_bm25(chunk.get("text", ""))
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # Lazy-load cross-encoder
        self._reranker = None

        log.info(f"Initialized HybridRetriever with {len(chunks)} chunks")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple whitespace tokenization for BM25."""
        return text.lower().split()

    def _get_reranker(self):
        """Lazy-load the cross-encoder re-ranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
                log.info(f"Loaded cross-encoder: {RERANKER_MODEL}")
            except Exception as e:
                log.warning(f"Failed to load cross-encoder: {e}")
                self._reranker = None
        return self._reranker

    def dense_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[RetrievedChunk]:
        """
        Dense retrieval using FAISS vector similarity search.

        Args:
            query: Query string.
            top_k: Number of results.

        Returns:
            List of RetrievedChunk objects ranked by dense similarity.
        """
        query_vec = self.embedder.embed(query)
        results = self.indexer.search(query_vec, top_k=top_k)

        retrieved = []
        for r in results:
            retrieved.append(RetrievedChunk(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                source=r.get("source", ""),
                page=r.get("page"),
                doc_id=r.get("doc_id", ""),
                dense_score=r.get("score", 0.0),
            ))

        return retrieved

    def bm25_search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[RetrievedChunk]:
        """
        Sparse retrieval using BM25 keyword matching.

        Args:
            query: Query string.
            top_k: Number of results.

        Returns:
            List of RetrievedChunk objects ranked by BM25 score.
        """
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = self._tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            retrieved.append(RetrievedChunk(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                score=float(scores[idx]),
                source=chunk.get("source", ""),
                page=chunk.get("page"),
                doc_id=chunk.get("doc_id", ""),
                sparse_score=float(scores[idx]),
            ))

        return retrieved

    def reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
        k: int = RRF_K,
    ) -> List[RetrievedChunk]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each result list.

        Args:
            dense_results: Results from dense retrieval.
            bm25_results: Results from BM25 retrieval.
            k: RRF constant (default: 60).

        Returns:
            Fused and re-ranked list of RetrievedChunk objects.
        """
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Score from dense results
        for rank, chunk in enumerate(dense_results):
            key = chunk.chunk_id or chunk.text[:50]
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = chunk

        # Score from BM25 results
        for rank, chunk in enumerate(bm25_results):
            key = chunk.chunk_id or chunk.text[:50]
            rrf_score = 1.0 / (k + rank + 1)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = chunk
            else:
                # Merge scores
                chunk_map[key].sparse_score = chunk.sparse_score

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        fused = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            chunk.score = fused_scores[key]
            fused.append(chunk)

        return fused

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = TOP_K_RERANK) -> List[RetrievedChunk]:
        """
        Re-rank candidates using a cross-encoder model.

        Args:
            query: Query string.
            candidates: Candidate chunks from retrieval.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of RetrievedChunk objects.
        """
        if not candidates:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            log.warning("Cross-encoder not available — returning candidates as-is")
            return candidates[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk in candidates]

        try:
            scores = reranker.predict(pairs)

            # Update scores
            for chunk, score in zip(candidates, scores):
                chunk.rerank_score = float(score)

            # Sort by rerank score
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)

            # Update final score to be the rerank score
            for chunk in candidates:
                chunk.score = chunk.rerank_score

        except Exception as e:
            log.error(f"Re-ranking failed: {e}")

        return candidates[:top_k]

    def retrieve(self, query: str, top_k: int = TOP_K_RERANK) -> List[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
            1. Dense search (FAISS)
            2. Sparse search (BM25)
            3. Reciprocal Rank Fusion
            4. Cross-encoder re-ranking

        Args:
            query: User question.
            top_k: Number of final results.

        Returns:
            Top-K re-ranked passages.
        """
        log.info(f"Retrieving for query: '{query[:80]}...'")

        # Step 1 & 2: Parallel dense and sparse retrieval
        dense_results = self.dense_search(query, top_k=TOP_K_RETRIEVE)
        bm25_results = self.bm25_search(query, top_k=TOP_K_RETRIEVE)

        log.debug(f"Dense: {len(dense_results)} results, BM25: {len(bm25_results)} results")

        # Step 3: Fusion
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        fused_top = fused[:min(10, len(fused))]  # Take top-10 for re-ranking

        log.debug(f"Fused: {len(fused_top)} candidates for re-ranking")

        # Step 4: Re-rank
        final = self.rerank(query, fused_top, top_k=top_k)

        log.info(f"Retrieved {len(final)} final passages")
        return final
