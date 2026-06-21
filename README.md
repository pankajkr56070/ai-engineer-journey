# Tests: ![Tests](https://github.com/pankajkr56070/ai-engineer-journey/actions/workflows/tests.yml/badge.svg)

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

Run tests
---------

Run the unit test suite locally with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Benchmark
---------

Compare the pure-Python and NumPy implementations with the included benchmark:

```bash
python3 week1/benchmark_cosine.py
```

Environment / Requirements
--------------------------

- Python 3.12 is recommended (matches CI).
- Install NumPy:

```bash
pip3 install numpy
```

Notes
-----

- Tests are run in CI via `python -m unittest discover -s tests -p 'test_*.py' -v`.
- Replace the fake embeddings in `week1/embeddings.py` with real model outputs for production use.

Notes:

- The project uses `np.random.rand(384)` to generate fake embeddings — replace this with
  real model embeddings for meaningful results.
- `search.py` uses `np.argpartition` for efficient top-k selection.
