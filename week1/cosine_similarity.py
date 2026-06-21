"""Utilities for computing cosine similarity between vectors.

This module provides a lightweight, dependency-free implementation of
`cosine_similarity` that accepts sequences or numpy arrays.
"""
from typing import Iterable
from collections.abc import Iterable as _Iterable
import math



def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product between two numbers or two equal-length iterables.

    Examples:
        dot_product([1,2,3], [4,5,6]) -> 32.0
        dot_product(2, 3) -> 6.0
    """
    def _is_iter(x):
        return isinstance(x, _Iterable) and not isinstance(x, (str, bytes))

    if _is_iter(a) and _is_iter(b):
        a_list = list(a)
        b_list = list(b)
        if len(a_list) != len(b_list):
            raise ValueError("Vectors must be the same length")
        return sum(float(x) * float(y) for x, y in zip(a_list, b_list))
    # fallback to scalar multiplication
    return float(a) * float(b)

def magnitude(a):
    """Return magnitude of a number or iterable.

    If `a` is an iterable, returns sqrt(sum(x*x)). If `a` is a scalar, returns abs(a).
    """
    if isinstance(a, _Iterable) and not isinstance(a, (str, bytes)):
        s = 0.0
        for x in a:
            s += float(x) * float(x)
        return math.sqrt(s)
    return math.sqrt(float(a) * float(a))


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: An iterable of numeric values (list, tuple, or numpy array).
        b: An iterable of numeric values (same length as `a`).

    Returns:
        Cosine similarity as a float in [-1.0, 1.0].

    Raises:
        ValueError: If the vectors have different lengths or either is a zero vector.

    Examples:
        >>> cosine_similarity([1, 0, 0], [0, 1, 0])
        0.0
        >>> cosine_similarity([1, 1], [1, 1])
        1.0
    """
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


def top_k_similar_vectors(query: Iterable[float], vectors: Iterable[Iterable[float]], k: int = 5):
    """Return the top-k most similar vectors to `query` from `vectors` using cosine similarity.

    Args:
        query: Iterable of numbers representing the query vector.
        vectors: An iterable of vectors (each an iterable of numbers).
        k: Number of top similar items to return. If k > len(vectors), returns all.

    Returns:
        A list of tuples `(index, similarity)` sorted by descending similarity.

    Notes:
        - Uses the `cosine_similarity` implementation in this module.
        - Ties are broken by lower index (stable sort).
    """
    if k <= 0:
        return []

    sims = []
    for idx, vec in enumerate(vectors):
        try:
            score = cosine_similarity(query, vec)
        except ValueError:
            # If lengths differ or zero vector, treat similarity as -1 (lowest)
            score = -1.0
        sims.append((idx, score))

    # Partial sort would be more efficient for large lists; keep simple and clear.
    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[:min(k, len(sims))]


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
