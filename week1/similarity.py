"""Utilities for computing cosine similarity between vectors.

This module provides a lightweight, dependency-free implementation of
`cosine_similarity` that accepts sequences or numpy arrays.
"""
from collections.abc import Iterable as _Iterable
from typing import Iterable, Union
import math

import numpy as np


def dot_product(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute dot product between two numbers or two equal-length iterables."""

    def _is_iterable(x):
        return isinstance(x, _Iterable) and not isinstance(x, (str, bytes))

    if _is_iterable(a) and _is_iterable(b):
        a_list = list(a)
        b_list = list(b)
        if len(a_list) != len(b_list):
            raise ValueError("Vectors must be the same length")
        return sum(float(x) * float(y) for x, y in zip(a_list, b_list))
    return float(a) * float(b)


def magnitude(a: Union[Iterable[float], float]) -> float:
    """Return magnitude of a number or iterable."""
    if isinstance(a, _Iterable) and not isinstance(a, (str, bytes)):
        s = 0.0
        for x in a:
            s += float(x) * float(x)
        return math.sqrt(s)
    return math.sqrt(float(a) * float(a))


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError("Vectors must be the same length")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a_list, b_list):
        x = float(x)
        y = float(y)
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vector")

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def top_k_similar_vectors(
    query: Iterable[float], vectors: Iterable[Iterable[float]], k: int = 5
) -> list[tuple[int, float]]:
    """Return the top-k most similar vectors to `query` from `vectors`."""
    if k <= 0:
        return []

    sims: list[tuple[int, float]] = []
    for idx, vec in enumerate(vectors):
        try:
            score = cosine_similarity(query, vec)
        except ValueError:
            score = -1.0
        sims.append((idx, score))

    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[: min(k, len(sims))]


def cosine_similarity_numpy(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute cosine similarity using NumPy."""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if xa.ndim != 1 or xb.ndim != 1:
        raise ValueError("Inputs must be 1-D array-like vectors")
    if xa.shape[0] != xb.shape[0]:
        raise ValueError("Vectors must be the same length")

    na = np.linalg.norm(xa)
    nb = np.linalg.norm(xb)
    if na == 0.0 or nb == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vector")

    return float(np.dot(xa, xb) / (na * nb))


def batch_cosine_similarity(query: Iterable[float], embeddings: Iterable[Iterable[float]]) -> np.ndarray:
    """Compute cosine similarity between a query vector and a batch of vectors."""
    q = np.asarray(query, dtype=float)
    e = np.asarray(embeddings, dtype=float)

    if q.ndim != 1:
        raise ValueError("Query must be a 1-D array-like vector")
    if e.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array-like matrix")
    if q.shape[0] != e.shape[1]:
        raise ValueError("Query dimension must match embedding dimension")

    q_norm = np.linalg.norm(q)
    e_norms = np.linalg.norm(e, axis=1)

    if q_norm == 0.0 or np.any(e_norms == 0.0):
        raise ValueError("Cosine similarity is undefined for zero vector")

    return (e @ q) / (e_norms * q_norm)


batch_cosine_similarity_numpy = batch_cosine_similarity



if __name__ == "__main__":
    # Quick manual checks
    examples = [
        ([1, 0, 0], [0, 1, 0]),  # orthogonal -> 0
        ([1, 1], [1, 1]),        # identical -> 1
        ([1, 2, 3], [4, 5, 6]),  # positive correlation
    ]

    for x, y in examples:
        sim = cosine_similarity(x, y)
        print(f"cosine_similarity({x}, {y}) = {sim}")
    
    # Test batch cosine similarity
    query = [1, 0, 0]
    embeddings = [[0, 1, 0], [1, 1, 0], [1, 2, 3]]
    batch_sims = batch_cosine_similarity_numpy(query, embeddings)
    print(f"Batch cosine similarities: {batch_sims}")
