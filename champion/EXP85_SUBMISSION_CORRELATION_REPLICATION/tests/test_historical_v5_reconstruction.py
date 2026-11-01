from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from reconstruct_historical_v5 import (
    SUITE_KEYS,
    SUITE_WEIGHTS,
    compose_candidate,
    id_aligned_metrics,
    probit,
    rank_uniform,
    suite_probit,
    validate_probability_vector,
)


class HistoricalV5ReconstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.members = {
            name: np.linspace(0.05 + index * 0.01, 0.95 - index * 0.01, 5)
            for index, name in enumerate(SUITE_KEYS)
        }

    def test_explicit_historical_suite_mapping_and_weights(self) -> None:
        suite = suite_probit(self.members)
        expected = np.zeros(5, dtype=np.float64)
        total = sum(SUITE_WEIGHTS.values())
        for name, weight in SUITE_WEIGHTS.items():
            expected += (weight / total) * norm.ppf(
                np.clip(rank_uniform(self.members[name]), 1e-6, 1 - 1e-6)
            )
        np.testing.assert_array_equal(suite, expected)

    def test_missing_suite_member_is_rejected(self) -> None:
        incomplete = dict(self.members)
        incomplete.pop("xgb10")
        with self.assertRaisesRegex(ValueError, "member mapping mismatch"):
            suite_probit(incomplete)

    def test_rank_normalization_preserves_float32_arithmetic(self) -> None:
        values = np.asarray(
            [0.0289683882, 0.2839717865, 0.1040754393, 0.9206993580],
            dtype=np.float32,
        )
        expected = (rankdata(values, method="average") - 0.5) / len(values)
        actual = rank_uniform(values)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected)
        expected_probit = norm.ppf(np.clip(expected, 1e-6, 1.0 - 1e-6))
        np.testing.assert_array_equal(probit(actual), expected_probit)

    def test_member_shape_nonfinite_and_probability_bounds_are_checked(self) -> None:
        invalid = dict(self.members)
        invalid["xgb10"] = np.asarray([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "row counts"):
            suite_probit(invalid)
        invalid["xgb10"] = np.asarray([0.1, 0.2, 0.3, 0.4, np.nan])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            suite_probit(invalid)
        invalid["xgb10"] = np.asarray([0.1, 0.2, 0.3, 0.4, 1.1])
        with self.assertRaisesRegex(ValueError, "outside"):
            suite_probit(invalid)

    def test_final_composition_uses_exact_85_15_formula_and_ordinal_ranks(self) -> None:
        anchor = np.asarray([0.12, 0.25, 0.48, 0.73, 0.91])
        suite = suite_probit(self.members)
        expected_z = 0.85 * norm.ppf(np.clip(anchor, 1e-6, 1 - 1e-6)) + 0.15 * suite
        expected = (rankdata(expected_z, method="ordinal") - 0.5) / len(anchor)
        np.testing.assert_array_equal(compose_candidate(anchor, suite), expected)

    def test_probability_validation_rejects_bad_dimensions_and_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_probability_vector(np.ones((5, 1)), "member", 5)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_probability_vector(
                np.asarray([0.1, 1.1]), "member", 2
            )

    def test_metrics_align_by_ids_when_rows_are_reordered(self) -> None:
        reference = pd.DataFrame(
            {"SK_ID_CURR": [10, 20, 30], "TARGET": [0.1, 0.2, 0.4]}
        )
        candidate = pd.DataFrame(
            {"SK_ID_CURR": [30, 10, 20], "TARGET": [0.8, 0.2, 0.4]}
        )
        metrics = id_aligned_metrics(reference, candidate)
        self.assertEqual(metrics["aligned_rows"], 3)
        self.assertEqual(metrics["alignment"], "one-to-one SK_ID_CURR join in raw test ID order")
        self.assertEqual(metrics["exact_prediction_matches"], 0)

    def test_duplicate_or_different_ids_fail_closed(self) -> None:
        reference = pd.DataFrame(
            {"SK_ID_CURR": [10, 20], "TARGET": [0.1, 0.2]}
        )
        duplicate = pd.DataFrame(
            {"SK_ID_CURR": [10, 10], "TARGET": [0.1, 0.2]}
        )
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            id_aligned_metrics(reference, duplicate)


if __name__ == "__main__":
    unittest.main()
