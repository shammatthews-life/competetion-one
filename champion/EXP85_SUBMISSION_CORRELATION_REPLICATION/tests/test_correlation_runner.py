from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from correlation_runner import ComparisonError, compare_predictions, load_predictions, run


class CorrelationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference_path = self.root / "reference.csv"
        self.candidate_path = self.root / "candidate.csv"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_pair(self, reference: list[tuple[int, float]], candidate: list[tuple[int, float]]) -> None:
        pd.DataFrame(reference, columns=["SK_ID_CURR", "TARGET"]).to_csv(
            self.reference_path, index=False
        )
        pd.DataFrame(candidate, columns=["SK_ID_CURR", "TARGET"]).to_csv(
            self.candidate_path, index=False
        )

    def load_pair(self):
        return (
            load_predictions(self.reference_path, "SK_ID_CURR", "TARGET"),
            load_predictions(self.candidate_path, "SK_ID_CURR", "TARGET"),
        )

    def test_id_alignment_is_independent_of_candidate_row_order(self) -> None:
        self.write_pair(
            [(10, 0.1), (20, 0.2), (30, 0.4)],
            [(30, 0.8), (10, 0.2), (20, 0.4)],
        )
        reference, candidate = self.load_pair()
        alignment, metrics = compare_predictions(reference, candidate, "id_aligned")
        self.assertFalse(alignment["row_order_matches"])
        self.assertEqual(alignment["aligned_row_count"], 3)
        self.assertEqual(metrics["pearson"], 1.0)
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["rank_displacement"]["maximum"], 0.0)

    def test_legacy_notebook_mode_requires_same_verified_id_order(self) -> None:
        self.write_pair([(10, 0.1), (20, 0.2)], [(20, 0.4), (10, 0.2)])
        reference, candidate = self.load_pair()
        with self.assertRaisesRegex(ComparisonError, "identical verified ID row order"):
            compare_predictions(reference, candidate, "notebook_stage9")

    def test_legacy_notebook_mode_reproduces_ordinal_rank_calculation(self) -> None:
        self.write_pair([(10, 0.1), (20, 0.2), (30, 0.4)], [(10, 0.1), (20, 0.4), (30, 0.2)])
        reference, candidate = self.load_pair()
        alignment, metrics = compare_predictions(reference, candidate, "notebook_stage9")
        self.assertTrue(alignment["row_order_matches"])
        self.assertEqual(metrics["rank_method"], "ordinal")
        self.assertEqual(metrics["rank_displacement"]["mean"], 2.0 / 3.0)
        self.assertEqual(metrics["rank_displacement"]["maximum"], 1.0)

    def test_duplicate_ids_fail_closed(self) -> None:
        self.write_pair([(10, 0.1), (10, 0.2)], [(10, 0.1), (10, 0.2)])
        reference, candidate = self.load_pair()
        with self.assertRaisesRegex(ComparisonError, "Duplicate IDs"):
            compare_predictions(reference, candidate, "id_aligned")

    def test_missing_and_extra_ids_are_reported(self) -> None:
        self.write_pair([(10, 0.1), (20, 0.2)], [(10, 0.1), (30, 0.3)])
        reference, candidate = self.load_pair()
        with self.assertRaises(ComparisonError) as raised:
            compare_predictions(reference, candidate, "id_aligned")
        self.assertEqual(raised.exception.alignment["missing_from_candidate_ids"], 1)
        self.assertEqual(raised.exception.alignment["extra_in_candidate_ids"], 1)

    def test_nonfinite_and_out_of_range_values_are_rejected(self) -> None:
        self.write_pair([(10, 0.1), (20, 0.2)], [(10, 0.1), (20, 1.1)])
        with self.assertRaisesRegex(ComparisonError, "outside \\[0, 1\\]"):
            load_predictions(self.candidate_path, "SK_ID_CURR", "TARGET")

    def test_rank_displacement_summary_has_requested_quantiles(self) -> None:
        self.write_pair(
            [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)],
            [(1, 0.1), (2, 0.3), (3, 0.2), (4, 0.4)],
        )
        reference, candidate = self.load_pair()
        _, metrics = compare_predictions(reference, candidate, "id_aligned")
        ranks = metrics["rank_displacement"]
        self.assertEqual(set(ranks), {
            "mean", "median", "p95", "maximum", "exact_rank_agreement_count"
        })

    def test_run_enforces_expected_rows_and_records_probability_bounds(self) -> None:
        self.write_pair(
            [(10, 0.1), (20, 0.4), (30, 0.7)],
            [(10, 0.2), (20, 0.3), (30, 0.8)],
        )
        report = run(
            self.reference_path,
            self.candidate_path,
            self.root / "valid_run",
            "SK_ID_CURR",
            "TARGET",
            "id_aligned",
            None,
            None,
            "Synthetic test.",
            expected_rows=3,
        )
        self.assertEqual(report["status"], "VALID")
        self.assertEqual(report["prediction_validation"]["reference"]["minimum"], 0.1)
        self.assertEqual(report["prediction_validation"]["candidate"]["maximum"], 0.8)

    def test_run_rejects_unexpected_row_count(self) -> None:
        self.write_pair([(10, 0.1), (20, 0.4)], [(10, 0.2), (20, 0.3)])
        report = run(
            self.reference_path,
            self.candidate_path,
            self.root / "invalid_run",
            "SK_ID_CURR",
            "TARGET",
            "id_aligned",
            None,
            None,
            "Synthetic test.",
            expected_rows=3,
        )
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("Expected exactly 3 rows", report["error"])
        self.assertEqual(report["alignment"]["reference_rows"], 2)


if __name__ == "__main__":
    unittest.main()
