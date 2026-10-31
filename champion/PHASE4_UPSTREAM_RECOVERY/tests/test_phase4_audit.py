import unittest

import numpy as np

from source.catboost_fold5_repeats import (
    comparison_metrics,
    id_sha256,
    restore_permuted_predictions,
)


class CatBoostAuditHelperTests(unittest.TestCase):
    def test_comparison_metrics_report_exact_match(self):
        values = np.array([0.1, 0.4, 0.2, 0.8], dtype=np.float64)
        metrics = comparison_metrics(values, values.copy())
        self.assertTrue(metrics["exact_value_match"])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["maximum_rank_displacement"], 0.0)

    def test_restore_permuted_predictions_recovers_original_order(self):
        original = np.array([2.0, 7.0, 5.0, 1.0])
        permutation = np.array([2, 0, 3, 1])
        shuffled = original[permutation]
        np.testing.assert_array_equal(
            restore_permuted_predictions(shuffled, permutation), original
        )

    def test_restore_rejects_non_permutation(self):
        with self.assertRaises(ValueError):
            restore_permuted_predictions(
                np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1])
            )

    def test_id_hash_is_order_sensitive(self):
        self.assertNotEqual(
            id_sha256(np.array([101, 202, 303])),
            id_sha256(np.array([303, 202, 101])),
        )


if __name__ == "__main__":
    unittest.main()
