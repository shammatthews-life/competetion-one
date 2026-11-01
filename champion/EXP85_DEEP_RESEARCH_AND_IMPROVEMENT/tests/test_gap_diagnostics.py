from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from gap_diagnostics import align_submission, grouped_error_metrics, make_prediction_slices, prediction_metrics


class GapDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_features = pd.DataFrame(
            {
                "SK_ID_CURR": [10, 20, 30, 40],
                "EXT_SOURCE_1": [0.1, np.nan, 0.2, np.nan],
                "EXT_SOURCE_2": [0.2, 0.3, np.nan, np.nan],
                "EXT_SOURCE_3": [0.3, 0.4, 0.5, np.nan],
                "AMT_INCOME_TOTAL": [100.0, 200.0, 300.0, 400.0],
                "DAYS_BIRTH": [-8000, -12000, -16000, -20000],
                "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving", "Cash"],
                "CODE_GENDER": ["F", "M", "F", "M"],
            }
        )
        self.reference = pd.DataFrame(
            {"SK_ID_CURR": [10, 20, 30, 40], "TARGET": [0.1, 0.2, 0.3, 0.4]}
        )

    def test_alignment_uses_ids_not_candidate_row_order(self) -> None:
        candidate = pd.DataFrame(
            {"SK_ID_CURR": [30, 10, 40, 20], "TARGET": [0.6, 0.2, 0.8, 0.4]}
        )
        reference, aligned_candidate, details = align_submission(
            self.reference, candidate, self.test_features["SK_ID_CURR"]
        )
        self.assertFalse(details["candidate_row_order_matches_test"])
        self.assertEqual(aligned_candidate.tolist(), [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(prediction_metrics(reference, aligned_candidate)["spearman"], 1.0)

    def test_duplicate_or_missing_ids_fail_closed(self) -> None:
        duplicate = pd.DataFrame(
            {"SK_ID_CURR": [10, 10, 30, 40], "TARGET": [0.1, 0.2, 0.3, 0.4]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            align_submission(self.reference, duplicate, self.test_features["SK_ID_CURR"])
        missing = self.reference.iloc[:3]
        with self.assertRaisesRegex(ValueError, "ID set"):
            align_submission(missing, self.reference, self.test_features["SK_ID_CURR"])

    def test_slice_metrics_are_descriptive_and_partition_every_row(self) -> None:
        reference = self.reference["TARGET"].to_numpy()
        candidate = reference + np.array([0.0, 0.01, -0.02, 0.03])
        slices = make_prediction_slices(reference, self.test_features)
        self.assertEqual(slices["ext_source_missingness"].tolist(), [
            "0_missing", "1_missing", "1_missing", "2plus_missing"
        ])
        rows = grouped_error_metrics("candidate", reference, candidate, slices)
        ext_rows = [row for row in rows if row["slice"] == "ext_source_missingness"]
        self.assertEqual(sum(row["rows"] for row in ext_rows), 4)
        self.assertTrue(all(row["mae"] >= 0.0 for row in rows))

    def test_prediction_metrics_reject_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            prediction_metrics(np.array([0.1, 0.2]), np.array([0.2, np.inf]))


if __name__ == "__main__":
    unittest.main()
