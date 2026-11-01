import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_final_gate import build_decoded_features, phase02  # noqa: E402
from run_frequency_ablation import no_frequency_features  # noqa: E402


class FinalGateLeakageTests(unittest.TestCase):
    def test_fold_local_frequency_reference_excludes_validation_rows(self):
        reference = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
                "AMT_CREDIT": [10000.0, 20000.0],
                "AMT_ANNUITY": [1000.0, 2000.0],
                "AMT_GOODS_PRICE": [9000.0, 18000.0],
            }
        )
        validation = pd.DataFrame(
            {
                "SK_ID_CURR": [3, 4],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
                "AMT_CREDIT": [10000.0, 20000.0],
                "AMT_ANNUITY": [2000.0, 4000.0],
                "AMT_GOODS_PRICE": [9000.0, 18000.0],
            }
        )
        features, audit = build_decoded_features(reference, validation)
        self.assertTrue(audit["reference_contains_validation_or_test_rows"] is False)
        self.assertEqual(audit["count_reference_rows"], len(reference))
        self.assertTrue((features["FEE_RAW_RATIO_COUNT"] == 0).all())
        self.assertTrue((features["FEE_ADJUSTED_RATIO_COUNT"] == 0).all())
        self.assertFalse(features["FEE_DECODED"].any())

    def test_unsupervised_fee_features_do_not_depend_on_any_labels(self):
        reference = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "TARGET": [0, 1, 0],
                "NAME_CONTRACT_TYPE": ["Cash loans"] * 3,
                "AMT_CREDIT": [10000.0, 20000.0, 30000.0],
                "AMT_ANNUITY": [1129.5, 2129.5, 3129.5],
                "AMT_GOODS_PRICE": [9500.0, 19000.0, 28500.0],
            }
        )
        apply = reference.copy()
        original, _ = build_decoded_features(reference, apply)
        relabeled_reference = reference.copy()
        relabeled_reference["TARGET"] = 1 - relabeled_reference["TARGET"]
        relabeled_apply = apply.copy()
        relabeled_apply["TARGET"] = 1 - relabeled_apply["TARGET"]
        changed, _ = build_decoded_features(relabeled_reference, relabeled_apply)
        pd.testing.assert_frame_equal(original, changed)
        self.assertNotIn("TARGET", original.columns)
        self.assertNotIn("SK_ID_CURR", original.columns)

    def test_frequency_ablation_disables_decoder_but_preserves_product_features(self):
        raw = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
                "AMT_CREDIT": [10000.0, 20000.0],
                "AMT_ANNUITY": [1129.5, 2129.5],
                "AMT_GOODS_PRICE": [
                    10000.0 / (1.0 + 0.0132 * 5.0),
                    20000.0 / (1.0 + 0.0132 * 5.0),
                ],
            }
        )
        features = no_frequency_features(raw)
        expected_raw_ratio = raw["AMT_ANNUITY"] / raw["AMT_CREDIT"]
        np.testing.assert_allclose(
            features["ANNUITY_CREDIT_RATIO_DECODED"],
            expected_raw_ratio,
        )
        self.assertFalse(features["FEE_DECODED"].any())
        self.assertTrue((features["FEE_RAW_RATIO_COUNT"] == 0).all())
        self.assertTrue((features["FEE_ADJUSTED_RATIO_COUNT"] == 0).all())
        self.assertTrue((features["PRODUCT_KEY_VALID"] == 1).all())

    def test_each_training_encoding_is_inner_oof(self):
        labels = np.tile(np.array([0, 1], dtype=np.int8), 20)
        keys = np.array([f"k{i % 5}" for i in range(len(labels))])
        keys_by_name = {
            name: np.roll(keys, offset)
            for offset, (name, _, _) in enumerate(phase02.KEY_SPECS)
        }
        outer_fit = np.arange(len(labels), dtype=np.int64)
        actual = phase02.nested_training_encodings(
            keys_by_name, labels, outer_fit
        )
        expected = np.full(actual.shape, np.nan, dtype=np.float32)
        inner = StratifiedKFold(
            n_splits=phase02.INNER_SPLITS,
            shuffle=True,
            random_state=phase02.INNER_SEED,
        )
        for fit_local, apply_local in inner.split(
            np.zeros(len(outer_fit)), labels[outer_fit]
        ):
            fit_indices = outer_fit[fit_local]
            apply_indices = outer_fit[apply_local]
            self.assertFalse(np.intersect1d(fit_indices, apply_indices).size)
            for key_index, (name, _, smoothing) in enumerate(phase02.KEY_SPECS):
                encoded, counts = phase02.encode_keys(
                    keys_by_name[name],
                    labels,
                    fit_indices,
                    apply_indices,
                    name,
                    smoothing,
                )
                expected[apply_local, 2 * key_index] = encoded
                expected[apply_local, 2 * key_index + 1] = np.log1p(counts)
        np.testing.assert_array_equal(actual.to_numpy(), expected)

    def test_outer_validation_labels_cannot_change_validation_encodings(self):
        labels = np.tile(np.array([0, 1], dtype=np.int8), 20)
        keys = np.array([f"k{i % 7}" for i in range(len(labels))])
        fit = np.arange(30, dtype=np.int64)
        apply = np.arange(30, 40, dtype=np.int64)
        before, _ = phase02.encode_keys(
            keys, labels, fit, apply, "validation", smoothing=20.0
        )
        changed_labels = labels.copy()
        changed_labels[apply] = 1 - changed_labels[apply]
        after, _ = phase02.encode_keys(
            keys, changed_labels, fit, apply, "validation", smoothing=20.0
        )
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
