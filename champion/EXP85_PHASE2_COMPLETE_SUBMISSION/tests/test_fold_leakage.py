from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from folds import build_folds, fold_id_vector


class FoldLeakageTests(unittest.TestCase):
    def test_folds_are_reproducible_disjoint_and_cover_rows(self):
        labels = np.asarray([0, 1] * 100)
        first = build_folds(labels, 5, 42)
        second = build_folds(labels, 5, 42)
        self.assertEqual(len(first), 5)
        np.testing.assert_array_equal(fold_id_vector(labels, first), fold_id_vector(labels, second))
        for fit, valid in first:
            self.assertEqual(np.intersect1d(fit, valid).size, 0)
            self.assertEqual(len(fit) + len(valid), len(labels))
        self.assertTrue(np.all(fold_id_vector(labels, first) > 0))

    def test_different_seed_changes_fold_membership(self):
        labels = np.asarray([0, 1] * 100)
        first = build_folds(labels, 5, 42)
        second = build_folds(labels, 5, 43)
        self.assertFalse(all(np.array_equal(a[1], b[1]) for a, b in zip(first, second)))


if __name__ == "__main__":
    unittest.main()
