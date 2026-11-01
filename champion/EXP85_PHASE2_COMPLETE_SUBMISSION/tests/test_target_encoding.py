from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from phase2_target_encoding import (
    KEY_SPECS,
    apply_training_encodings,
    encode_keys,
    nested_training_encodings,
)


def key_map(n: int) -> dict[str, np.ndarray]:
    base = np.asarray([f"key_{i % 7}" for i in range(n)], dtype=str)
    return {name: base.copy() for name, _, _ in KEY_SPECS}


class TargetEncodingTests(unittest.TestCase):
    def test_smoothing_formula_and_unseen_category_fallback(self):
        keys = np.asarray(["a", "a", "b", "b", "a", "new"])
        labels = np.asarray([0, 1, 0, 0, 1, 1], dtype=np.int8)
        encoded, counts = encode_keys(keys, labels, np.arange(4), np.asarray([4, 5]), "x", 2.0)
        prior = 0.25
        self.assertAlmostEqual(float(encoded[0]), (1 + 2 * prior) / 4)
        self.assertAlmostEqual(float(encoded[1]), prior)
        self.assertEqual(counts.tolist(), [2.0, 0.0])

    def test_apply_labels_cannot_change_training_mappings(self):
        keys = key_map(12)
        labels = np.asarray([0, 1] * 6, dtype=np.int8)
        fit = np.arange(8)
        apply = np.arange(8, 12)
        before = apply_training_encodings(keys, labels, fit, apply)
        changed = labels.copy()
        changed[apply] = 1 - changed[apply]
        after = apply_training_encodings(keys, changed, fit, apply)
        pd.testing.assert_frame_equal(before, after)

    def test_self_target_is_excluded_from_encoded_row(self):
        keys = np.asarray(["same"] * 8 + ["other"] * 4)
        labels = np.asarray([0, 1] * 6, dtype=np.int8)
        fit = np.asarray([0, 1, 2, 3, 4, 5])
        apply = np.asarray([6])
        first, _ = encode_keys(keys, labels, fit, apply, "test", 10.0)
        changed = labels.copy()
        changed[6] = 1 - changed[6]
        second, _ = encode_keys(keys, changed, fit, apply, "test", 10.0)
        np.testing.assert_array_equal(first, second)

    def test_outer_validation_labels_cannot_change_nested_training_encodings(self):
        labels = np.asarray([0, 1] * 50, dtype=np.int8)
        keys = key_map(len(labels))
        outer_fit = np.arange(80)
        before = nested_training_encodings(keys, labels, outer_fit)
        changed = labels.copy()
        changed[80:] = 1 - changed[80:]
        after = nested_training_encodings(keys, changed, outer_fit)
        pd.testing.assert_frame_equal(before, after)

    def test_fit_apply_overlap_is_rejected(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            encode_keys(np.asarray(["a"] * 4), labels, np.asarray([0, 1]), np.asarray([1]), "x", 1.0)


if __name__ == "__main__":
    unittest.main()
