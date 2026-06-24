
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week1.similarity import batch_cosine_similarity, cosine_similarity


def main() -> None:
    query = np.random.rand(384)
    embeddings = np.random.rand(10000, 384)

    start_time = time.perf_counter()
    similarities = batch_cosine_similarity(query, embeddings)
    end_time = time.perf_counter()
    print(
        f"Computed {len(similarities)} similarities using NumPy in {end_time - start_time:.4f} seconds"
    )

    start_time = time.perf_counter()
    scores = [cosine_similarity(query, emb) for emb in embeddings]
    end_time = time.perf_counter()
    print(
        f"Computed {len(scores)} similarities using pure Python in {end_time - start_time:.4f} seconds"
    )

    assert np.allclose(similarities, np.array(scores)), "NumPy and pure Python results differ"


if __name__ == "__main__":
    main()
