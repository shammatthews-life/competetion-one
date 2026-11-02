from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_exp10 import (  # noqa: E402
    OUTER_SEED,
    PAIR_COMPONENTS,
    baseline_discrepancy,
    canonical_array_hash,
    fold_indices,
    passes_gate,
    reconstruct_exp08_lgbcat,
    verify_fold_oof_stream,
)


class Exp10Tests(unittest.TestCase):
    def test_selected_pair_is_exact_and_fixed(self) -> None:
        self.assertEqual(PAIR_COMPONENTS, ("exp07_signature", "exp08_lgbcat"))
        self.assertEqual(OUTER_SEED, 20261102)

    def test_advancement_gate_keeps_both_thresholds(self) -> None:
        self.assertTrue(passes_gate(0.001, 4))
        self.assertFalse(passes_gate(0.000999, 5))
        self.assertFalse(passes_gate(0.0011, 3))

    def test_baseline_discrepancy_is_reported_not_normalized(self) -> None:
        authoritative = np.array([0.1, 0.4, 0.8], dtype=np.float64)
        diagnostic = np.array([0.1, 0.5, 0.7], dtype=np.float64)
        result = baseline_discrepancy(authoritative, diagnostic)
        self.assertFalse(result["identical"])
        self.assertAlmostEqual(result["max_absolute_difference"], 0.1)
        self.assertAlmostEqual(result["mean_absolute_difference"], 0.06666666666666667)

    def test_array_hash_includes_shape_and_dtype(self) -> None:
        values = np.array([1, 2], dtype=np.int64)
        self.assertNotEqual(
            canonical_array_hash(values),
            canonical_array_hash(values.reshape(1, 2)),
        )
        self.assertNotEqual(
            canonical_array_hash(values),
            canonical_array_hash(values.astype(np.int32)),
        )

    def test_exp08_component_reconstructed_without_xgboost_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            expected_oof = np.linspace(0.1, 0.9, 10)
            for fold in range(1, 6):
                idx = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
                np.savez(
                    run_dir / f"fold_{fold}_indices.npz",
                    fit_indices=np.array([i for i in range(10) if i not in idx]),
                    valid_indices=idx,
                )
                np.savez(
                    run_dir / f"fold_{fold}_predictions.npz",
                    historical_lgb_cat_valid=expected_oof[idx],
                    historical_test=np.array([0.25, 0.75]),
                )
            oof, test = reconstruct_exp08_lgbcat(run_dir, 10, 2)
            np.testing.assert_allclose(oof, expected_oof)
            np.testing.assert_allclose(test, [0.25, 0.75])

    def test_exp08_component_removes_xgboost_from_saved_three_model_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            component_test = np.array([0.2, 0.7])
            xgb_test = np.array([0.9, 0.1])
            candidate_mean = (component_test + xgb_test) / 2.0
            expected_oof = np.linspace(0.1, 0.9, 10)
            for fold in range(1, 6):
                idx = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
                np.savez(
                    run_dir / f"fold_{fold}_indices.npz",
                    fit_indices=np.array([i for i in range(10) if i not in idx]),
                    valid_indices=idx,
                )
                np.savez(
                    run_dir / f"fold_{fold}_predictions.npz",
                    historical_lgb_cat_valid=expected_oof[idx],
                    historical_test=candidate_mean,
                    historical_xgb_test=xgb_test,
                )
            _, test = reconstruct_exp08_lgbcat(run_dir, 10, 2)
            np.testing.assert_allclose(test, component_test)

    def test_oof_stream_must_match_each_fold_at_saved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            values = np.linspace(0.1, 0.9, 10)
            for fold in range(1, 6):
                idx = np.array([2 * (fold - 1), 2 * (fold - 1) + 1])
                np.savez(
                    run_dir / f"fold_{fold}_indices.npz",
                    fit_indices=np.array([i for i in range(10) if i not in idx]),
                    valid_indices=idx,
                )
                np.savez(
                    run_dir / f"fold_{fold}_predictions.npz",
                    historical_lgb_cat_valid=values[idx],
                )
            verify_fold_oof_stream(
                run_dir, "EXP08 LGB/Cat", "historical_lgb_cat_valid", values
            )
            wrong = values.copy()
            wrong[0] = 0.5
            with self.assertRaises(ValueError):
                verify_fold_oof_stream(
                    run_dir, "EXP08 LGB/Cat", "historical_lgb_cat_valid", wrong
                )

    def test_fold_indices_read_exact_training_and_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(
                directory / "fold_1_indices.npz",
                fit_indices=np.array([0, 2]),
                valid_indices=np.array([1, 3]),
            )
            fit, valid = fold_indices(directory, 1)
            np.testing.assert_array_equal(fit, [0, 2])
            np.testing.assert_array_equal(valid, [1, 3])


if __name__ == "__main__":
    unittest.main()
