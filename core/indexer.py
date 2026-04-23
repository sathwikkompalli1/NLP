"""
IQAS FAISS Indexer
===================
Build, save, load, and update FAISS vector indexes for dense retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import faiss

from utils.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    FAISS_IVF_NLIST,
    FAISS_LARGE_CORPUS_THRESHOLD,
)
from utils.logger import get_logger

log = get_logger("indexer")


class FAISSIndexer:
    """
    FAISS-based vector index for dense retrieval.

    Features:
        - IndexFlatIP (Inner Product after L2 normalization = cosine similarity)
        - IndexIVFFlat for large corpora (>10k chunks)
        - Incremental updates (add documents without full rebuild)
        - Persistent save/load with metadata
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        """
        Initialize the indexer.

        Args:
            dim: Embedding dimension (384 for MiniLM).
        """
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []  # chunk metadata parallel to index vectors
        self._is_ivf = False

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    def build(self, embeddings: np.ndarray, chunks_metadata: List[Dict]) -> None:
        """
        Build a flat inner product index (for cosine similarity with L2-normalized vectors).

        Args:
            embeddings: L2-normalized embeddings of shape (n, dim).
            chunks_metadata: List of metadata dicts, one per embedding.
        """
        if embeddings.shape[0] != len(chunks_metadata):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) != metadata count ({len(chunks_metadata)})"
            )

        n = embeddings.shape[0]

        if n > FAISS_LARGE_CORPUS_THRESHOLD:
            log.info(f"Large corpus ({n} chunks) — building IVF index")
            self.build_ivf(embeddings, chunks_metadata)
            return

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = list(chunks_metadata)
        self._is_ivf = False

        log.info(f"Built FAISS FlatIP index: {self.size} vectors, dim={self.dim}")

    def build_ivf(
        self,
        embeddings: np.ndarray,
        chunks_metadata: List[Dict],
        nlist: int = FAISS_IVF_NLIST,
    ) -> None:
        """
        Build an IVF index for large corpora.

        Args:
            embeddings: L2-normalized embeddings of shape (n, dim).
            chunks_metadata: List of metadata dicts.
            nlist: Number of Voronoi cells.
        """
        n = embeddings.shape[0]
        actual_nlist = min(nlist, n)  # Can't have more cells than vectors

        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index
        self.index.train(embeddings.astype(np.float32))
        self.index.add(embeddings.astype(np.float32))
        self.metadata = list(chunks_metadata)
        self._is_ivf = True

        # Set nprobe for search quality
        self.index.nprobe = min(10, actual_nlist)

        log.info(f"Built FAISS IVF index: {self.size} vectors, nlist={actual_nlist}")

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict]:
        """
        Search the index for top-K nearest vectors.

        Args:
            query_vector: Query embedding of shape (dim,) or (1, dim).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'score', 'index', and all metadata fields.
        """
        if self.index is None or self.size == 0:
            log.warning("Index is empty — cannot search")
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        top_k = min(top_k, self.size)
        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            result["index"] = int(idx)
            results.append(result)

        return results

    def add(self, new_embeddings: np.ndarray, new_metadata: List[Dict]) -> None:
        """
        Incrementally add new vectors to the index.

        Args:
            new_embeddings: New L2-normalized embeddings.
            new_metadata: Corresponding metadata dicts.
        """
        if self.index is None:
            self.build(new_embeddings, new_metadata)
            return

        self.index.add(new_embeddings.astype(np.float32))
        self.metadata.extend(new_metadata)
        log.info(f"Added {len(new_metadata)} vectors to index (total: {self.size})")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            path: Directory path to save to (default: configured path).
        """
        if self.index is None:
            log.warning("No index to save")
            return

        index_path = Path(path) / "index.faiss" if path else FAISS_INDEX_PATH
        meta_path = Path(path) / "metadata.json" if path else FAISS_METADATA_PATH

        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dim": self.dim,
                "is_ivf": self._is_ivf,
                "count": self.size,
                "chunks": self.metadata,
            }, f, ensure_ascii=False, indent=2)

        log.info(f"Saved FAISS index ({self.size} vectors) to {index_path}")

    def load(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load a FAISS index and metadata from disk.

        Args:
            path: Directory path to load from (default: configured path).

        Returns:
            True if loaded successfully, False otherwise.
        """
        index_path = Path(path) / "index.faiss" if path else FAISS_INDEX_PATH
        meta_path = Path(path) / "metadata.json" if path else FAISS_METADATA_PATH

        if not index_path.exists() or not meta_path.exists():
            log.info("No saved index found")
            return False

        try:
            self.index = faiss.read_index(str(index_path))

            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.dim = data.get("dim", EMBEDDING_DIM)
            self._is_ivf = data.get("is_ivf", False)
            self.metadata = data.get("chunks", [])

            log.info(f"Loaded FAISS index: {self.size} vectors from {index_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load index: {e}")
            return False

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve chunk metadata by chunk_id.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            Metadata dict if found, None otherwise.
        """
        for meta in self.metadata:
            if meta.get("chunk_id") == chunk_id:
                return meta
        return None

    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = None
        self.metadata = []
        self._is_ivf = False
        log.info("Cleared FAISS index")
