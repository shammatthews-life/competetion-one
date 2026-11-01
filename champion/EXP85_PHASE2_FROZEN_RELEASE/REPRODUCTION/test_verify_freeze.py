from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from verify_freeze import validate_candidate_order


def write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class CandidateOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate.csv"
        self.test = self.root / "test.csv"
        self.sample = self.root / "sample.csv"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_valid_inputs(self) -> None:
        write_csv(self.candidate, ["SK_ID_CURR", "TARGET"], [[12, 0.2], [18, 0.8]])
        write_csv(self.test, ["SK_ID_CURR", "FEATURE"], [[12, 1], [18, 2]])
        write_csv(self.sample, ["SK_ID_CURR", "TARGET"], [[12, 0], [18, 0]])

    def test_exact_id_alignment_and_probability_validation(self) -> None:
        self.write_valid_inputs()
        result = validate_candidate_order(
            self.candidate, self.test, self.sample, expected_rows=2
        )
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["ids_match_test_order"])
        self.assertTrue(result["ids_match_sample_order"])
        self.assertTrue(result["finite_predictions"])
        self.assertTrue(result["probability_range_valid"])

    def test_candidate_is_aligned_by_id_order_not_row_count(self) -> None:
        self.write_valid_inputs()
        write_csv(self.candidate, ["SK_ID_CURR", "TARGET"], [[18, 0.8], [12, 0.2]])
        result = validate_candidate_order(
            self.candidate, self.test, self.sample, expected_rows=2
        )
        self.assertFalse(result["ids_match_test_order"])
        self.assertTrue(any("order" in issue for issue in result["issues"]))

    def test_duplicate_ids_are_rejected(self) -> None:
        self.write_valid_inputs()
        write_csv(self.candidate, ["SK_ID_CURR", "TARGET"], [[12, 0.2], [12, 0.8]])
        result = validate_candidate_order(
            self.candidate, self.test, self.sample, expected_rows=2
        )
        self.assertFalse(result["unique_ids"])
        self.assertTrue(any("duplicated" in issue for issue in result["issues"]))

    def test_out_of_range_probabilities_are_rejected(self) -> None:
        self.write_valid_inputs()
        write_csv(self.candidate, ["SK_ID_CURR", "TARGET"], [[12, 1.2], [18, 0.8]])
        result = validate_candidate_order(
            self.candidate, self.test, self.sample, expected_rows=2
        )
        self.assertFalse(result["probability_range_valid"])
        self.assertTrue(any("[0, 1]" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
