from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_exp09 import (  # noqa: E402
    canonical_id_hash,
    check_same_folds,
    prediction_diagnostics,
    read_fold_indices,
    verify_oof_row_alignment,
    verify_test_row_alignment,
)


class Exp09Tests(unittest.TestCase):
    def test_ordered_id_hash_changes_when_order_changes(self) -> None:
        self.assertNotEqual(canonical_id_hash([10, 20, 30]), canonical_id_hash([30, 20, 10]))
        self.assertEqual(canonical_id_hash([10, 20]), canonical_id_hash([10, 20]))

    def test_fold_indices_require_complete_disjoint_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for fold_number, indices in enumerate(([0, 1], [2, 3]), start=1):
                np.savez(directory / f"fold_{fold_number}_indices.npz", valid_indices=indices)
            folds, digest = read_fold_indices(directory, 4, fold_count=2)
            self.assertEqual(len(folds), 2)
            self.assertEqual(len(digest), 64)
            self.assertTrue(check_same_folds(folds, [np.array([0, 1]), np.array([2, 3])]))

    def test_fold_membership_comparison_is_exact(self) -> None:
        self.assertTrue(
            check_same_folds(
                [np.array([0, 1]), np.array([2, 3])],
                [np.array([0, 1]), np.array([2, 3])],
            )
        )
        self.assertFalse(
            check_same_folds(
                [np.array([0, 1]), np.array([2, 3])],
                [np.array([0, 2]), np.array([1, 3])],
            )
        )

    def test_prediction_diagnostics_report_risk_segments_and_overlap(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        baseline = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55])
        candidate = baseline[::-1]
        result = prediction_diagnostics(labels, baseline, candidate)
        self.assertIn("pearson_vs_phase2", result)
        self.assertIn("delta_auc_by_baseline_risk_decile", result)
        self.assertIn("binary_error_jaccard_at_prevalence", result)
        self.assertGreaterEqual(result["top_prevalence_overlap"], 0.0)
        self.assertLessEqual(result["top_prevalence_overlap"], 1.0)
        self.assertGreaterEqual(result["binary_error_jaccard_at_prevalence"], 0.0)
        self.assertLessEqual(result["binary_error_jaccard_at_prevalence"], 1.0)

    def test_oof_predictions_must_match_saved_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            labels = np.array([0, 1, 0, 1, 0], dtype=np.int8)
            values = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
            for fold_number, index in enumerate(range(5), start=1):
                np.savez(
                    directory / f"fold_{fold_number}_indices.npz",
                    valid_indices=np.array([index]),
                )
                np.savez(
                    directory / f"fold_{fold_number}_predictions.npz",
                    oof=values[[index]],
                )

            aligned, _ = verify_oof_row_alignment(
                directory, "exp04", directory / "oof.npy", values, labels, np.arange(5)
            )
            self.assertTrue(aligned)
            reordered = values.copy()
            reordered[[0, 1]] = reordered[[1, 0]]
            aligned, _ = verify_oof_row_alignment(
                directory, "exp04", directory / "oof.npy", reordered, labels, np.arange(5)
            )
            self.assertFalse(aligned)

    def test_test_predictions_require_ordered_id_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            test_ids = np.array([20, 10, 30])
            predictions = np.array([0.2, 0.1, 0.3])
            pd.DataFrame(
                {"SK_ID_CURR": test_ids, "TARGET": predictions}
            ).to_csv(directory / "submission.csv", index=False)
            aligned, _ = verify_test_row_alignment(
                directory, "unknown", predictions, test_ids
            )
            self.assertTrue(aligned)
            aligned, _ = verify_test_row_alignment(
                directory, "unknown", predictions, test_ids[::-1]
            )
            self.assertFalse(aligned)


if __name__ == "__main__":
    unittest.main()
