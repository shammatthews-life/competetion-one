from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp04_catboost_lossguide.run_experiment import (
    ID_COLUMN,
    MODEL_CONFIG,
    TARGET_COLUMN,
    compare_oof,
    inner_early_stopping_split,
    model_parameters,
    validate_probabilities,
    validate_submission,
)


class LossguideExperimentTests(unittest.TestCase):
    def test_gpu_configuration_is_explicit_and_bounded(self) -> None:
        parameters = model_parameters(20261102)
        self.assertEqual(parameters["task_type"], "GPU")
        self.assertEqual(parameters["grow_policy"], "Lossguide")
        self.assertEqual(parameters["devices"], "0")
        self.assertLessEqual(parameters["gpu_ram_part"], 0.8)
        self.assertEqual(parameters["random_seed"], 20261102)

    def test_probability_validation_rejects_bad_values(self) -> None:
        validate_probabilities(np.asarray([0.1, 0.9]), 2, "valid")
        with self.assertRaises(ValueError):
            validate_probabilities(np.asarray([0.1, np.nan]), 2, "nan")
        with self.assertRaises(ValueError):
            validate_probabilities(np.asarray([0.1, 1.1]), 2, "range")

    def test_inner_stopping_split_stays_inside_outer_fit(self) -> None:
        labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
        outer_fit = np.arange(20, 180, dtype=np.int64)
        inner_fit, inner_valid = inner_early_stopping_split(
            labels, outer_fit, seed=13
        )
        self.assertFalse(np.intersect1d(inner_fit, inner_valid).size)
        self.assertTrue(np.isin(inner_fit, outer_fit).all())
        self.assertTrue(np.isin(inner_valid, outer_fit).all())

    def test_oof_comparison_uses_paired_folds_and_fixed_gate(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        reference = np.asarray([0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.1])
        candidate = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.95])
        folds = [
            (np.asarray([2, 3, 6, 7]), np.asarray([0, 1, 4, 5])),
            (np.asarray([0, 1, 4, 5]), np.asarray([2, 3, 6, 7])),
        ]
        result = compare_oof(labels, candidate, reference, folds)
        self.assertEqual(result["positive_folds"], 2)
        self.assertGreater(result["pooled_auc_delta"], 0)
        self.assertEqual(result["advancement_gate"]["result"], "REJECT_SCREEN")

    def test_submission_requires_exact_ids_order_and_predictions(self) -> None:
        sample = pd.DataFrame(
            {ID_COLUMN: [10, 11], TARGET_COLUMN: [np.nan, np.nan]}
        )
        candidate = pd.DataFrame(
            {ID_COLUMN: [10, 11], TARGET_COLUMN: [0.2, 0.8]}
        )
        validate_submission(candidate, sample, np.asarray([0.2, 0.8]))
        reordered = candidate.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate_submission(reordered, sample, np.asarray([0.8, 0.2]))


if __name__ == "__main__":
    unittest.main()
