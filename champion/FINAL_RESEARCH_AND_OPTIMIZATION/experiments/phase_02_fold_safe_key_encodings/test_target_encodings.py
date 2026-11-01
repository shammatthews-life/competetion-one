import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_experiment import (  # noqa: E402
    encode_keys,
    nested_training_encodings,
)


class FoldSafeTargetEncodingTests(unittest.TestCase):
    def test_apply_labels_cannot_change_encodings(self):
        keys = np.array(["a", "a", "b", "b", "a", "unknown"])
        labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
        fit = np.array([0, 1, 2, 3])
        apply = np.array([4, 5])
        original = encode_keys(keys, labels, fit, apply, "test", 10.0)
        altered = labels.copy()
        altered[apply] = 1 - altered[apply]
        changed = encode_keys(keys, altered, fit, apply, "test", 10.0)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])

    def test_fit_and_apply_rows_must_be_disjoint(self):
        keys = np.array(["a", "b", "a"])
        labels = np.array([0, 1, 0], dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            encode_keys(
                keys,
                labels,
                np.array([0, 1]),
                np.array([1, 2]),
                "test",
                10.0,
            )

    def test_nested_training_encodings_cover_every_outer_training_row(self):
        rng = np.random.default_rng(42)
        labels = np.tile(np.array([0, 1], dtype=np.int8), 50)
        keys = rng.integers(0, 12, size=len(labels)).astype(str)
        result = nested_training_encodings(
            {"FEE_Q3": keys, "FEE_Q4": keys, "FEE_Q5": keys, "PRODUCT_K": keys},
            labels,
            np.arange(len(labels)),
        )
        self.assertEqual(len(result), len(labels))
        self.assertEqual(result.shape[1], 8)
        self.assertTrue(np.isfinite(result.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
