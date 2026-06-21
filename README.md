# Week 1 — Embeddings & Simple Search

This small project demonstrates creating fake 384-dimensional embeddings for text
and performing top-k nearest-neighbor search using cosine similarity.

Structure:

- `embeddings.py` — fake embedding generators (`embed_text`, `embed_texts`).
- `search.py` — NumPy-accelerated `top_k_similar_vectors_np` and `nearest_documents` helpers.
- `main.py` — demo script that builds embeddings and prints the top matches for a query.

Quick start:

1. Install NumPy if you don't have it:

```bash
pip3 install numpy
```

2. Run the demo:

```bash
python3 week1/main.py
```

Notes:

- The project uses `np.random.rand(384)` to generate fake embeddings — replace this with
  real model embeddings for meaningful results.
- `search.py` uses `np.argpartition` for efficient top-k selection.
