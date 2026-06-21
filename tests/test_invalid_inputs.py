import os
import sys
import unittest

# Ensure project root is on sys.path so tests can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week1.cosine_similarity import dot_product, cosine_similarity


class TestInvalidInputs(unittest.TestCase):
    def test_dot_product_empty(self):
        self.assertEqual(dot_product([], []), 0.0)

    def test_dot_product_length_mismatch(self):
        with self.assertRaises(ValueError):
            dot_product([1], [1, 2])

    def test_cosine_similarity_zero_vector(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
