
import time

import numpy as np

from similarity import batch_cosine_similarity_numpy, cosine_similarity

query = np.random.rand(384)

embeddings = np.random.rand(
    10000,
    384
)


if __name__ == "__main__":

    """Similarity Benchmarking using NumPy for batch cosine similarity computation."""
    start_time = time.time()
    similarities = batch_cosine_similarity_numpy(query, embeddings)
    end_time = time.time()
    print(f"Computed {len(similarities)} similarities using NumPy in {end_time - start_time:.4f} seconds")


    """Similarity Benchmarking using pure Python for batch cosine similarity computation."""
    start_time = time.time()
    scores = []
    for emb in embeddings:
        score = cosine_similarity(
            query,
            emb
        )
        scores.append(score)
    end_time = time.time()
    print(f"Computed {len(scores)} similarities using pure Python in {end_time - start_time:.4f} seconds")
