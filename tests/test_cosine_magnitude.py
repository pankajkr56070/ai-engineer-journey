import os
import sys
import unittest

# ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week1.cosine_similarity import cosine_similarity, magnitude


class TestCosineMagnitude(unittest.TestCase):
    def test_cosine_similarity_identical(self):
        self.assertEqual(cosine_similarity([1, 1], [1, 1]), 1.0)

    def test_cosine_similarity_orthogonal(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=9)

    def test_magnitude(self):
        self.assertEqual(magnitude([3, 4]), 5.0)
        self.assertEqual(magnitude(3), 3.0)

    def test_negative_values(self):
        # vectors pointing in opposite directions
        self.assertAlmostEqual(cosine_similarity([1, -1], [-1, 1]), -1.0, places=9)


if __name__ == "__main__":
    unittest.main()
