import os
import sys
import unittest

# Ensure project root is on sys.path so tests can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week1.cosine_similarity import cosine_similarity, top_k_similar_vectors


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_and_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [0, 1, 0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1])
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0, 0], [1, 2, 3])


class TestTopKSimilarVectors(unittest.TestCase):
    def test_top_k_basic(self):
        query = [1, 0, 0]
        vectors = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
        res = top_k_similar_vectors(query, vectors, k=2)
        # expect index 0 first, then index 2
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][0], 0)
        self.assertEqual(res[1][0], 2)
        self.assertAlmostEqual(res[0][1], 1.0)

    def test_top_k_with_invalid_vector(self):
        query = [1, 0]
        vectors = [[1, 0], [1, 0, 0], [0, 1]]
        res = top_k_similar_vectors(query, vectors, k=3)
        # the invalid-length vector (index 1) should be treated as lowest similarity
        indices = [i for i, _ in res]
        self.assertIn(0, indices)
        self.assertIn(2, indices)
        # index 1 should be last
        self.assertEqual(indices[-1], 1)


if __name__ == "__main__":
    unittest.main()
