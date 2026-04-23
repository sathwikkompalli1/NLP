"""
IQAS Text Embedder
===================
Sentence-Transformers embedding with batch encoding, L2 normalization, and caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from utils.config import MODEL_NAME, EMBED_BATCH_SIZE, EMBEDDING_DIM
from utils.logger import get_logger

log = get_logger("embedder")


class TextEmbedder:
    """
    Embeds text into dense vectors using Sentence-Transformers.

    Features:
        - Single and batch embedding
        - L2 normalization for cosine similarity via dot product
        - .npy caching to avoid recomputation
        - CPU-only execution
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace Sentence-Transformers model name.
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        log.info(f"Loaded embedding model: {model_name} (dim={self.dim})")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text.

        Returns:
            L2-normalized embedding vector of shape (dim,).
        """
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)

        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = EMBED_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Batch encode multiple texts.

        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding.
            show_progress: Show tqdm progress bar.

        Returns:
            L2-normalized embedding matrix of shape (n, dim).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Replace empty strings with a placeholder
        cleaned = [t if t and t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_and_cache(
        self,
        texts: List[str],
        cache_path: Union[str, Path],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Embed texts with disk caching — loads from cache if available.

        Args:
            texts: List of text strings.
            cache_path: Path to save/load .npy cache.
            batch_size: Batch size for encoding.

        Returns:
            Embedding matrix of shape (n, dim).
        """
        cache_path = Path(cache_path)

        if cache_path.exists():
            try:
                cached = np.load(str(cache_path))
                if cached.shape[0] == len(texts):
                    log.info(f"Loaded cached embeddings from {cache_path}")
                    return cached
                else:
                    log.warning(
                        f"Cache size mismatch ({cached.shape[0]} vs {len(texts)}) — recomputing"
                    )
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")

        log.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embed_batch(texts, batch_size=batch_size, show_progress=True)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        log.info(f"Saved embeddings cache to {cache_path}")

        return embeddings

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize vectors for cosine similarity via dot product.

        Args:
            vectors: Array of shape (n, d) or (d,).

        Returns:
            Normalized vectors.
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else vectors

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return vectors / norms
