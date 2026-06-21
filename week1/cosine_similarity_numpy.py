"""NumPy-accelerated cosine similarity helpers and top-k retrieval.

This module demonstrates creating fake 384-d embeddings for documents
and a fast `top_k_similar_vectors_np` implementation using NumPy.

Run the file directly to see a small example with fake embeddings.
"""
from typing import List, Tuple
import numpy as np


def fake_embeddings(documents: List[str], dim: int = 384) -> np.ndarray:
    """Generate fake embeddings for `documents` with shape (N, dim)."""
    return np.random.rand(len(documents), dim)


def top_k_similar_vectors_np(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """Return top-k most similar vectors to `query_embedding` using cosine similarity.

    Args:
        query_embedding: 1-D array of shape (D,).
        embeddings: 2-D array of shape (N, D).
        k: number of top results to return.

    Returns:
        List of `(index, similarity)` sorted by descending similarity.

    Notes:
        - Similarity is cosine similarity in [-1.0, 1.0].
        - Zero vectors are treated as having similarity -1.0.
        - Uses `np.argpartition` for O(N) average-time top-k selection.
    """
    if k <= 0:
        return []

    q = np.asarray(query_embedding, dtype=float)
    X = np.asarray(embeddings, dtype=float)

    if q.ndim != 1:
        raise ValueError("query_embedding must be a 1-D array")
    if X.ndim != 2 or X.shape[1] != q.shape[0]:
        raise ValueError("embeddings must be shape (N, D) matching query length")

    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        raise ValueError("query embedding is the zero vector")
    q_unit = q / q_norm

    norms = np.linalg.norm(X, axis=1)
    sims = np.full(X.shape[0], -1.0, dtype=float)
    valid = norms > 0
    if np.any(valid):
        sims[valid] = X[valid].dot(q_unit) / norms[valid]

    top_n = min(k, X.shape[0])
    # get top_n indices (unsorted), then sort them by similarity descending
    idx_unsorted = np.argpartition(-sims, top_n - 1)[:top_n]
    idx_sorted = idx_unsorted[np.argsort(-sims[idx_unsorted])]

    return [(int(i), float(sims[i])) for i in idx_sorted]


if __name__ == "__main__":
    documents = [
        "I love machine learning",
        "I love cricket",
        "Backend engineering with Go",
    ]

    embeddings = fake_embeddings(documents, dim=384)
    query_text = "machine learning"
    query_embedding = fake_embeddings([query_text], dim=384)[0]  # fake single-query embedding

    top = top_k_similar_vectors_np(query_embedding, embeddings, k=2)
    print("Top results:")
    for idx, score in top:
        print(idx, score, "->", documents[idx])
