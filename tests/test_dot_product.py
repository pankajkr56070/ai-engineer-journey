import unittest

from week1.similarity import dot_product


class TestDotProduct(unittest.TestCase):
    def test_dot_product(self):
        self.assertEqual(
            dot_product([1, 2, 3], [4, 5, 6]),
            32.0
        )


if __name__ == "__main__":
    unittest.main()