"""Benchmark pure-Python vs NumPy cosine similarity implementations.

Run this script from the project root:
    python3 week1/benchmark_cosine.py

It will print timing results using `timeit`.
"""
import timeit
import numpy as np
from week1.cosine_similarity import cosine_similarity, cosine_similarity_numpy


def make_vectors(n=1000, dim=384):
    rng = np.random.default_rng(42)
    return [list(rng.random(dim)) for _ in range(n)], rng.random(dim)


def bench(n=1000, dim=384, repeat=5):
    vectors, query = make_vectors(n=n, dim=dim)

    # prepare callable wrappers
    def py_call():
        # run one pass over vectors
        for v in vectors:
            cosine_similarity(query, v)

    def np_call():
        for v in vectors:
            cosine_similarity_numpy(query, v)

    py_times = timeit.repeat(py_call, repeat=repeat, number=1)
    np_times = timeit.repeat(np_call, repeat=repeat, number=1)

    print(f"Benchmark: n={n}, dim={dim}, repeat={repeat}")
    print(f"Pure Python times: {py_times}")
    print(f"NumPy times:       {np_times}")
    print(f"Median speedup: {float(np.median(py_times) / np.median(np_times)):.2f}x")


if __name__ == "__main__":
    try:
        bench(n=1000, dim=384, repeat=3)
    except ModuleNotFoundError as e:
        print("Missing dependency:", e)
        print("Install numpy: pip3 install numpy")
