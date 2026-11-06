"""Regression tests for the LCS teaching example (stdlib only)."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "algotithms/2025/01-06/longest_common_substring.py"
spec = importlib.util.spec_from_file_location("lcs_example", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def is_subsequence(candidate, text):
    iterator = iter(text)
    return all(any(char == item for item in iterator) for char in candidate)


class LcsTests(unittest.TestCase):
    def test_known_cases(self):
        for a, b, expected in (("abcde", "ace", 3), ("abc", "abc", 3),
                               ("abcdf", "zabc", 3), ("ABCBDAB", "BDCABA", 4)):
            with self.subTest(a=a, b=b):
                self.assertEqual(module.lcs(a, b), expected)
                result = module.lcs_return_substring(a, b)
                self.assertEqual(len(result), expected)
                self.assertTrue(is_subsequence(result, a))
                self.assertTrue(is_subsequence(result, b))

    def test_empty_and_disjoint(self):
        for a, b in (("", ""), ("abc", ""), ("", "abc"), ("abc", "xyz")):
            self.assertEqual(module.lcs(a, b), 0)
            self.assertEqual(module.lcs_return_substring(a, b), "")

    def test_repeated_characters(self):
        for a, b in (("aaaa", "aa"), ("aabbaa", "aba"), ("banana", "ananas")):
            result = module.lcs_return_substring(a, b)
            self.assertEqual(len(result), module.lcs(a, b))
            self.assertTrue(is_subsequence(result, a))
            self.assertTrue(is_subsequence(result, b))


if __name__ == "__main__":
    unittest.main()
