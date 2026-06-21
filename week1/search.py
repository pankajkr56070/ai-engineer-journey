"""Search utilities: NumPy-powered top-k cosine similarity retrieval."""
from typing import List, Tuple
import numpy as np


def top_k_similar_vectors_np(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    """Return top-k most similar vectors to `query_embedding` using cosine similarity.

    Returns a list of `(index, similarity)` sorted by descending similarity.
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
    idx_unsorted = np.argpartition(-sims, top_n - 1)[:top_n]
    idx_sorted = idx_unsorted[np.argsort(-sims[idx_unsorted])]

    return [(int(i), float(sims[i])) for i in idx_sorted]


def nearest_documents(query_embedding: np.ndarray, embeddings: np.ndarray, documents: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """Return the top-k `(document, score)` pairs for a query embedding."""
    top = top_k_similar_vectors_np(query_embedding, embeddings, k=k)
    return [(documents[i], score) for i, score in top]


if __name__ == "__main__":
    # Quick demo when running this file directly
    from embeddings import embed_texts, embed_text

    docs = ["I love machine learning", "I love cricket", "Backend engineering with Go"]
    embs = embed_texts(docs)
    q = embed_text("I enjoy learning about models")
    results = nearest_documents(q, embs, docs, k=2)
    for doc, score in results:
        print(f"{score:.4f}	{doc}")
