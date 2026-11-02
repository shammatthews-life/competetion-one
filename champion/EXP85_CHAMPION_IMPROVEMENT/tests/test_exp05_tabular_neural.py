from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp05_tabular_neural.run_experiment import (  # noqa: E402
    FoldPreprocessor,
    ResidualTabularNet,
    build_inner_early_stopping_split,
    seed_everything,
)


class TabularNeuralTests(unittest.TestCase):
    def test_inner_split_is_disjoint_and_stays_inside_outer_fit(self) -> None:
        labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
        outer_fit = np.arange(20, 180, dtype=np.int64)
        fit, valid = build_inner_early_stopping_split(labels, outer_fit, 73)
        self.assertFalse(np.intersect1d(fit, valid).size)
        self.assertTrue(np.isin(fit, outer_fit).all())
        self.assertTrue(np.isin(valid, outer_fit).all())

    def test_preprocessor_uses_fit_statistics_and_unknown_category_bucket(self) -> None:
        frame = pd.DataFrame(
            {
                "value": [1.0, 3.0, np.nan, 1000.0],
                "category": ["a", "b", "a", "new"],
            }
        )
        preprocessor = FoldPreprocessor(("category",)).fit(
            frame, np.asarray([0, 1, 2])
        )
        numeric, categorical = preprocessor.transform(frame)
        self.assertEqual(preprocessor.numeric_columns, ["value"])
        self.assertEqual(categorical[3, 0], 0)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertLessEqual(float(np.max(np.abs(numeric))), 12.0)

    def test_model_has_residual_hidden_layers_and_embeddings(self) -> None:
        model = ResidualTabularNet(
            numeric_count=3,
            category_sizes=[4, 5],
            embedding_dimensions=[4, 4],
            hidden_width=16,
            residual_blocks=2,
            dropout=0.1,
        )
        logits = model(
            torch.randn(8, 3),
            torch.zeros((8, 2), dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (8,))

    def test_seed_reproduces_initial_model_parameters(self) -> None:
        seed_everything(20261101)
        first = ResidualTabularNet(3, [4], [4], 16, 1, 0.1).state_dict()
        seed_everything(20261101)
        second = ResidualTabularNet(3, [4], [4], 16, 1, 0.1).state_dict()
        self.assertTrue(
            all(torch.equal(first[name], second[name]) for name in first)
        )


if __name__ == "__main__":
    unittest.main()
