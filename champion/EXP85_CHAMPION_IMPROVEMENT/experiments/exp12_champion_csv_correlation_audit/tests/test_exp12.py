from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.run_exp12 import (
    _duplicate_rows,
    align_by_id,
    calculate_metrics,
    normalize_id,
)


class Exp12AlignmentTests(unittest.TestCase):
    def test_alignment_uses_ids_not_candidate_row_order(self) -> None:
        champion, candidate, ids = align_by_id(
            ["101", "102", "103"],
            [0.1, 0.2, 0.3],
            ["103", "101", "102"],
            [0.35, 0.15, 0.25],
        )
        self.assertEqual(ids, ["101", "102", "103"])
        np.testing.assert_allclose(champion, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(candidate, [0.15, 0.25, 0.35])

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            align_by_id(["1", "1"], [0.1, 0.2], ["1", "2"], [0.1, 0.2])

    def test_missing_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            align_by_id(["1", ""], [0.1, 0.2], ["1", "2"], [0.1, 0.2])

    def test_numeric_id_strings_are_canonicalized(self) -> None:
        self.assertEqual(normalize_id("0012.0"), "12")

    def test_exact_file_and_aligned_stream_duplicates_are_reported(self) -> None:
        inventory = [
            {"path": "a.csv", "sha256": "same-file"},
            {"path": "b.csv", "sha256": "same-file"},
        ]
        streams = [
            {
                "stream_sha256": "same-stream",
                "source_file_count": 2,
                "representative": "a.csv",
                "source_paths": ["a.csv", "b.csv"],
            }
        ]
        rows = _duplicate_rows(inventory, streams)
        self.assertEqual(
            {row["duplicate_type"] for row in rows},
            {
                "BYTE_IDENTICAL_CSV",
                "IDENTICAL_ID_ALIGNED_PREDICTION_STREAM",
            },
        )


class Exp12MetricTests(unittest.TestCase):
    def test_known_correlation_differences_and_risk_overlap(self) -> None:
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        candidate = reference + 0.01
        metrics = calculate_metrics(reference, candidate, [str(i) for i in range(10)])

        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["kendall_tau"], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_difference"], 0.01)
        self.assertAlmostEqual(metrics["rmse"], 0.01)
        self.assertAlmostEqual(metrics["mean_signed_difference"], 0.01)
        self.assertEqual(metrics["top_10_overlap_count"], 1)
        self.assertEqual(metrics["bottom_10_overlap_count"], 1)

    def test_identical_predictions_have_zero_error(self) -> None:
        values = [0.1, 0.3, 0.2]
        metrics = calculate_metrics(values, values, ["a", "b", "c"])
        self.assertEqual(metrics["mean_absolute_difference"], 0.0)
        self.assertEqual(metrics["maximum_absolute_difference"], 0.0)
        self.assertEqual(metrics["below_0_001_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
