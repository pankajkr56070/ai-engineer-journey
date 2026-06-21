import unittest

from week1.cosine_similarity import dot_product


class TestDotProduct(unittest.TestCase):
    def test_dot_product(self):
        assert dot_product([1,2,3],[4,5,6]) == float(32)


if __name__ == "__main__":
    unittest.main()