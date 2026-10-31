from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

from raw_candidate_formula import N_TEST, rank_uniform, synthesize  # noqa: E402
from compare_historical_outputs import compare_arrays  # noqa: E402
from validate_raw_candidate import (  # noqa: E402
    BLEND_WEIGHTS,
    align_submission,
    comparison_metrics,
    independently_recompute,
)


def create_components(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(100_000, 100_000 + N_TEST, dtype=np.int64)
    anchor_values = np.linspace(0.001, 0.999, N_TEST)
    pd.DataFrame({"SK_ID_CURR": ids, "TARGET": anchor_values}).to_csv(
        directory / "v_20261027_precision_transductive_apex.csv", index=False
    )
    index = np.arange(N_TEST, dtype=np.float64)
    v17 = {
        key: (np.sin(index / (scale + 3)) + 1.0).astype(np.float32)
        for key, scale in {
            "test_e1Aet": 5,
            "final_test": 7,
            "test_e1et": 11,
        }.items()
    }
    v55 = {
        key: (np.cos(index / (scale + 4)) + 1.0).astype(np.float32)
        for key, scale in {
            "test_cb_product": 13,
            "test_deep_et": 17,
            "test_v2_allfam_oh": 19,
            "test_v2_mono": 23,
        }.items()
    }
    np.savez(directory / "v_17_full_suite_bundle.npz", **v17)
    np.savez(directory / "v55_members.npz", **v55)
    xgb = (np.mod(index * 17, N_TEST) / N_TEST).astype(np.float32)
    np.save(directory / "test_10f_xgb.npy", xgb)
    return ids, anchor_values


class RawCandidateFormulaTests(unittest.TestCase):
    def test_rank_uniform_is_monotone_and_in_range(self):
        result = rank_uniform(np.array([4.0, 1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, np.array([0.875, 0.125, 0.375, 0.625]))
        self.assertTrue(np.all((result > 0) & (result < 1)))

    def test_rank_uniform_rejects_nonfinite_and_empty_inputs(self):
        with self.assertRaises(ValueError):
            rank_uniform(np.array([0.1, np.nan]))
        with self.assertRaises(ValueError):
            rank_uniform(np.array([]))

    def test_synthesis_matches_independent_recomputation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            ids, _ = create_components(directory)
            candidate = synthesize(
                ids,
                directory / "v_20261027_precision_transductive_apex.csv",
                directory / "v_17_full_suite_bundle.npz",
                directory / "v55_members.npz",
                directory / "test_10f_xgb.npy",
            )
            independently_recomputed = independently_recompute(directory, ids)
            np.testing.assert_array_equal(
                candidate["TARGET"].to_numpy(), independently_recomputed
            )
            self.assertEqual(candidate["SK_ID_CURR"].nunique(), N_TEST)
            self.assertEqual(candidate["TARGET"].nunique(), N_TEST)

    def test_synthesis_rejects_duplicate_or_misaligned_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            ids, _ = create_components(directory)
            with self.assertRaises(ValueError):
                synthesize(
                    np.r_[ids[:-1], ids[-2]],
                    directory / "v_20261027_precision_transductive_apex.csv",
                    directory / "v_17_full_suite_bundle.npz",
                    directory / "v55_members.npz",
                    directory / "test_10f_xgb.npy",
                )
            with self.assertRaises(ValueError):
                synthesize(
                    ids[::-1],
                    directory / "v_20261027_precision_transductive_apex.csv",
                    directory / "v_17_full_suite_bundle.npz",
                    directory / "v55_members.npz",
                    directory / "test_10f_xgb.npy",
                )

    def test_alignment_reorders_rows_by_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "submission.csv"
            ids = np.arange(100_000, 100_000 + N_TEST, dtype=np.int64)
            values = np.linspace(0.0, 1.0, N_TEST)
            permutation = np.arange(N_TEST)[::-1]
            pd.DataFrame(
                {"SK_ID_CURR": ids[permutation], "TARGET": values[permutation]}
            ).to_csv(path, index=False)
            np.testing.assert_array_equal(align_submission(path, ids), values)

    def test_alignment_rejects_duplicate_ids_and_nonfinite_targets(self):
        ids = np.arange(100_000, 100_000 + N_TEST, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "submission.csv"
            duplicate_ids = ids.copy()
            duplicate_ids[-1] = duplicate_ids[-2]
            pd.DataFrame(
                {"SK_ID_CURR": duplicate_ids, "TARGET": np.linspace(0.0, 1.0, N_TEST)}
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                align_submission(path, ids)

            values = np.linspace(0.0, 1.0, N_TEST)
            values[0] = np.nan
            pd.DataFrame({"SK_ID_CURR": ids, "TARGET": values}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                align_submission(path, ids)

    def test_weight_mapping_is_complete_and_normalizes(self):
        self.assertAlmostEqual(sum(weight for _, _, weight in BLEND_WEIGHTS), 1.0)
        self.assertEqual(
            {key for source, key, _ in BLEND_WEIGHTS if source == "v17"},
            {"test_e1Aet", "final_test", "test_e1et"},
        )
        self.assertEqual(
            {key for source, key, _ in BLEND_WEIGHTS if source == "v55"},
            {
                "test_cb_product",
                "test_deep_et",
                "test_v2_allfam_oh",
                "test_v2_mono",
            },
        )
        self.assertEqual(
            [key for source, key, _ in BLEND_WEIGHTS if source == "xgb"],
            ["test_10f_xgb.npy"],
        )

    def test_comparison_metrics_exact_identity(self):
        values = np.array([0.1, 0.4, 0.2, 0.8])
        metrics = comparison_metrics(values, values.copy())
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["maximum_rank_displacement"], 0.0)

    def test_historical_array_comparison_reports_exactness_and_shape_mismatch(self):
        values = np.array([0.1, 0.4, 0.2, 0.8])
        exact = compare_arrays(values, values.copy())
        self.assertTrue(exact["exact_array_match"])
        self.assertEqual(exact["different_cells"], 0)
        self.assertTrue(exact["rank_percentile_comparison"]["exact_rank_match"])
        mismatch = compare_arrays(values, values[:-1])
        self.assertFalse(mismatch["comparable"])
        self.assertEqual(mismatch["reason"], "shape mismatch")

    def test_historical_array_comparison_separates_scale_from_rank(self):
        values = np.array([0.1, 0.4, 0.2, 0.8])
        rescaled = np.square(values)
        comparison = compare_arrays(values, rescaled)
        self.assertFalse(comparison["exact_array_match"])
        self.assertTrue(comparison["rank_percentile_comparison"]["exact_rank_match"])
        self.assertEqual(
            comparison["rank_percentile_comparison"]["maximum_absolute_difference"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
