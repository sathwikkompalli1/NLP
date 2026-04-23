"""
IQAS Similarity Utilities
==========================
Cosine similarity functions for vectors, sentences, and batches.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("similarity")


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row in a matrix.

    Args:
        query_vec: Query vector of shape (d,).
        matrix: Matrix of shape (n, d) where each row is a vector.

    Returns:
        Array of shape (n,) with cosine similarity scores.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])

    # Normalize query
    query_normalized = query_vec / query_norm

    # Normalize matrix rows
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)  # Avoid division by zero
    matrix_normalized = matrix / row_norms

    # Dot product gives cosine similarity
    similarities = matrix_normalized @ query_normalized

    return similarities


def sentence_similarity(
    sent1: str,
    sent2: str,
    embedder,
) -> float:
    """
    Compute semantic similarity between two sentences using embeddings.

    Args:
        sent1: First sentence.
        sent2: Second sentence.
        embedder: TextEmbedder instance with .embed() method.

    Returns:
        Cosine similarity score.
    """
    v1 = embedder.embed(sent1)
    v2 = embedder.embed(sent2)
    return cosine_similarity(v1, v2)


def find_most_similar_sentence(
    query: str,
    sentences: List[str],
    embedder,
) -> Tuple[str, float, int]:
    """
    Find the most semantically similar sentence to a query.

    Args:
        query: Query string.
        sentences: List of candidate sentences.
        embedder: TextEmbedder instance.

    Returns:
        Tuple of (best_sentence, similarity_score, index).
    """
    if not sentences:
        return ("", 0.0, -1)

    query_vec = embedder.embed(query)
    sent_vecs = embedder.embed_batch(sentences)

    similarities = batch_cosine_similarity(query_vec, sent_vecs)
    best_idx = int(np.argmax(similarities))

    return (sentences[best_idx], float(similarities[best_idx]), best_idx)


def top_k_similar(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find top-K most similar vectors.

    Args:
        query_vec: Query vector.
        matrix: Matrix of vectors to compare against.
        k: Number of results.

    Returns:
        List of (index, similarity_score) tuples, sorted descending.
    """
    similarities = batch_cosine_similarity(query_vec, matrix)
    k = min(k, len(similarities))

    # Get top-K indices
    top_indices = np.argpartition(similarities, -k)[-k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
