from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_exp07 import (  # noqa: E402
    build_signature_keys,
    cross_fit_target_rate,
    fold_local_signature_features,
    select_blend_weight,
)


class SignatureFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "SK_ID_CURR": [10, 11, 12, 13, 14, 15, 16, 17],
                "DAYS_BIRTH": [-12000, -12000, -13000, -13000, -14000, -14000, -15000, -15000],
                "DAYS_REGISTRATION": [-1000, -1000, -2000, -2000, -3000, -3000, -4000, -4000],
                "DAYS_ID_PUBLISH": [-500, -500, -600, -600, -700, -700, -800, -800],
                "CODE_GENDER": ["F", "F", "M", "M", "F", "F", "M", "M"],
                "NAME_EDUCATION_TYPE": ["A", "A", "B", "B", "A", "A", "B", "B"],
            }
        )

    def test_signature_uses_covariates_but_never_id_or_target(self) -> None:
        keys = build_signature_keys(self.train)
        self.assertEqual(keys[0], keys[1])
        changed_id = self.train.copy()
        changed_id["SK_ID_CURR"] += 1_000_000
        self.assertEqual(keys, build_signature_keys(changed_id))
        with self.assertRaises(ValueError):
            build_signature_keys(self.train.assign(TARGET=0))

    def test_cross_fit_target_rate_excludes_apply_row_label(self) -> None:
        keys = [
            ("a",), ("b",), ("a",), ("b",),
            ("c",), ("c",), ("d",), ("d",),
        ]
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        partitions = [
            (np.asarray([2, 3, 4, 5, 6, 7]), np.asarray([0, 1])),
            (np.asarray([0, 1, 4, 5, 6, 7]), np.asarray([2, 3])),
            (np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7])),
        ]
        encoded = cross_fit_target_rate(
            keys,
            labels,
            seed=1,
            n_splits=3,
            partitions=partitions,
        )
        changed = labels.copy()
        changed[0] = 0
        encoded_changed = cross_fit_target_rate(
            keys,
            changed,
            seed=1,
            n_splits=3,
            partitions=partitions,
        )
        self.assertEqual(float(encoded[0]), float(encoded_changed[0]))
        self.assertNotEqual(float(encoded[2]), float(encoded_changed[2]))

    def test_apply_features_ignore_apply_labels_and_counts_are_fit_local(self) -> None:
        train_keys = build_signature_keys(self.train)
        test_keys = [train_keys[0], ("unseen", None, "F", "C")]
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        fit = np.asarray([0, 1, 2, 3, 4, 5])
        apply = np.asarray([6, 7])
        first = fold_local_signature_features(
            train_keys,
            test_keys,
            labels,
            fit,
            apply,
            seed=19,
            crossfit_splits=3,
        )
        changed_labels = labels.copy()
        changed_labels[6:] = 1 - changed_labels[6:]
        second = fold_local_signature_features(
            train_keys,
            test_keys,
            changed_labels,
            fit,
            apply,
            seed=19,
            crossfit_splits=3,
        )
        for name in first[0]:
            self.assertTrue(np.allclose(first[0][name], second[0][name], equal_nan=True))
            self.assertTrue(np.allclose(first[1][name], second[1][name], equal_nan=True))
        self.assertAlmostEqual(
            float(first[0]["EXP07_SIG_COUNT_LOG"][0]), np.log1p(1)
        )
        self.assertAlmostEqual(float(first[1]["EXP07_SIG_COUNT_LOG"][0]), np.log1p(2))
        self.assertEqual(float(first[1]["EXP07_SIG_COUNT_LOG"][1]), 0.0)
        self.assertTrue(first[2]["fit_row_target_excluded_by_crossfit"])
        self.assertFalse(first[2]["apply_labels_used"])

    def test_fold_local_signature_features_reject_overlapping_partitions(self) -> None:
        keys = build_signature_keys(self.train)
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            fold_local_signature_features(
                keys,
                keys[:1],
                labels,
                np.asarray([0, 1, 2, 3, 4, 5]),
                np.asarray([5, 6]),
                seed=1,
                crossfit_splits=3,
            )

    def test_nested_blend_weight_is_selected_on_inner_vectors(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        baseline = np.asarray([0.4, 0.3, 0.2, 0.1])
        candidate = np.asarray([0.1, 0.2, 0.7, 0.8])
        selected = select_blend_weight(labels, baseline, candidate)
        self.assertAlmostEqual(selected["candidate_weight"], 0.3)
        self.assertEqual(selected["inner_auc"], 1.0)
        self.assertEqual(len(selected["grid"]), 21)
        self.assertIn("inner validation only", selected["selection_labels_source"])


if __name__ == "__main__":
    unittest.main()
