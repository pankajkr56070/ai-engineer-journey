"""Embeddings helpers for week1.

This module provides simple fake 384-d embeddings for demonstration.
Replace the fake generator with a real embedding model in production.
"""
from typing import List
import numpy as np

# embedding dimension used in examples
DIM = 384


def embed_text(text: str) -> np.ndarray:
    """Return a fake embedding for a single text (1-D array of length DIM)."""
    return np.random.rand(DIM)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings for a list of texts as an array of shape (N, DIM)."""
    return np.stack([embed_text(t) for t in texts], axis=0)


if __name__ == "__main__":
    docs = ["hello world", "I love machine learning"]
    embs = embed_texts(docs)
    print("embeddings shape:", embs.shape)
