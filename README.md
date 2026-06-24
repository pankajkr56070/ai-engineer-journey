# Tests: ![Tests](https://github.com/pankajkr56070/ai-engineer-journey/actions/workflows/tests.yml/badge.svg)

# Week 1 — Embeddings & Simple Search

This small project demonstrates creating fake 384-dimensional embeddings for text
and performing top-k nearest-neighbor search using cosine similarity.

## Overview

- Embedding dimension: 384
- Similarity metric: cosine similarity
- Formula: $\text{cosine}(q, e) = \frac{q \cdot e}{\|q\|\|e\|}$
- Why NumPy is faster: vectorized operations avoid slow Python loops for large batches

## Structure

- [week1/embeddings.py](week1/embeddings.py) — fake embedding generators (`embed_text`, `embed_texts`)
- [week1/search.py](week1/search.py) — nearest-neighbor helpers
- [week1/main.py](week1/main.py) — demo script that builds embeddings and prints the top matches for a query

## Quick start

1. Install NumPy if you don't have it:

```bash
pip3 install numpy
```

2. Run the demo:

```bash
python3 week1/main.py
```

## Run tests

Run the unit test suite locally with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Benchmark

Compare the pure-Python and NumPy implementations with the included benchmark:

```bash
python3 week1/benchmark.py
```

The benchmark also asserts that the NumPy and pure-Python results are equivalent.

Example benchmark output:

```text
Computed 10000 similarities using NumPy in 0.0064 seconds
Computed 10000 similarities using pure Python in 0.8571 seconds
```

## Environment / Requirements

- Python 3.12 is recommended (matches CI)
- Install NumPy:

```bash
pip3 install numpy
```

## Future work

- HNSW for approximate nearest-neighbor search
- FAISS for scalable vector search
- Real embeddings from a pretrained model instead of random vectors

## Notes

- Tests are run in CI via `python3 -m unittest discover -s tests -p 'test_*.py' -v`.
- Replace the fake embeddings in [week1/embeddings.py](week1/embeddings.py) with real model outputs for production use.
- [week1/search.py](week1/search.py) uses `np.argpartition` for efficient top-k selection.
