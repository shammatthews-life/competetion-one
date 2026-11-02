from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_experiment import (  # noqa: E402
    FTTransformer,
    FoldPreprocessor,
    build_inner_split,
    seed_everything,
    validate_feature_audit,
    validate_fold_partition,
    validate_probabilities,
)


class FTTransformerTests(unittest.TestCase):
    def test_fold_partition_requires_exactly_once_validation_coverage(self) -> None:
        folds = [
            (np.asarray([2, 3]), np.asarray([0, 1])),
            (np.asarray([0, 1]), np.asarray([2, 3])),
        ]
        validate_fold_partition(folds, 4)
        with self.assertRaises(ValueError):
            validate_fold_partition([folds[0], (folds[1][0], np.asarray([1, 3]))], 4)

    def test_inner_split_is_disjoint_and_inside_outer_fit(self) -> None:
        labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
        outer_fit = np.arange(20, 180, dtype=np.int64)
        fit, valid = build_inner_split(labels, outer_fit, 71)
        self.assertEqual(np.intersect1d(fit, valid).size, 0)
        self.assertTrue(np.isin(fit, outer_fit).all())
        self.assertTrue(np.isin(valid, outer_fit).all())

    def test_preprocessing_is_fit_local_and_handles_unknown_categories(self) -> None:
        frame = pd.DataFrame(
            {
                "numeric": [2.0, 6.0, np.nan, 1000.0],
                "category": ["a", "b", "a", "new"],
            }
        )
        processor = FoldPreprocessor(("category",)).fit(
            frame, np.asarray([0, 1, 2])
        )
        numeric, categorical = processor.transform(frame)
        self.assertEqual(float(np.mean(numeric[:3, 0])), 0.0)
        self.assertEqual(categorical[3, 0], 0)
        self.assertTrue(np.isfinite(numeric).all())

    def test_target_encoding_audit_rejects_apply_row_target_leakage(self) -> None:
        good = {
            "target_encoding": {
                "apply_row_targets_used_by_mapping": False,
                "validation_encoded_from_outer_fit_only": True,
                "test_encoded_from_outer_fit_only": True,
            },
            "frequency": {},
        }
        validate_feature_audit(good, "test")
        bad = {
            **good,
            "target_encoding": {
                **good["target_encoding"],
                "apply_row_targets_used_by_mapping": True,
            },
        }
        with self.assertRaises(ValueError):
            validate_feature_audit(bad, "test")

    def test_ft_transformer_uses_attention_and_returns_finite_logits(self) -> None:
        model = FTTransformer(
            numeric_count=6,
            category_sizes=[4, 5],
            token_dimension=16,
            attention_heads=4,
            transformer_layers=2,
            feedforward_dimension=32,
            dropout=0.1,
        )
        self.assertIsInstance(model.transformer, torch.nn.TransformerEncoder)
        logits = model(
            torch.randn(12, 6),
            torch.zeros((12, 2), dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (12,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_seed_reproduces_initial_transformer_weights(self) -> None:
        seed_everything(8821)
        first = FTTransformer(3, [4], 16, 4, 1, 32, 0.1).state_dict()
        seed_everything(8821)
        second = FTTransformer(3, [4], 16, 4, 1, 32, 0.1).state_dict()
        self.assertTrue(
            all(torch.equal(first[key], second[key]) for key in first)
        )

    def test_probability_shape_range_and_finiteness(self) -> None:
        validate_probabilities(np.asarray([0.1, 0.8]), 2, "test")
        with self.assertRaises(ValueError):
            validate_probabilities(np.asarray([0.1, np.nan]), 2, "test")


if __name__ == "__main__":
    unittest.main()
