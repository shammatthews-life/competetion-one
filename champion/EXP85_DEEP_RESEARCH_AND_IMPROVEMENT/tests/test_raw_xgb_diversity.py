from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

from run_raw_xgb_diversity import build_fold_matrices, rank_percentile


class RawXGBoostFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "AMT_CREDIT": [100.0, np.nan, 300.0, 400.0],
                "CODE_GENDER": ["F", "M", "F", "M"],
                "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving", "Cash"],
                "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working", "Other"],
                "NAME_EDUCATION_TYPE": ["Higher", "Secondary", "Higher", "Lower"],
                "NAME_FAMILY_STATUS": ["Single", "Married", "Single", "Married"],
                "NAME_HOUSING_TYPE": ["House", "Rent", "House", "Rent"],
                "OCCUPATION_TYPE": ["Laborers", "Managers", "Laborers", "Drivers"],
                "ORGANIZATION_TYPE": ["Business", "School", "Business", "Trade"],
            }
        )
        self.test = self.train.iloc[:2].copy().reset_index(drop=True)
        self.test.loc[0, "ORGANIZATION_TYPE"] = "Unseen"

    def test_fold_onehot_uses_fit_categories_and_allows_unseen_categories(self):
        fit, valid, test, info = build_fold_matrices(
            self.train,
            self.test,
            np.array([0, 1]),
            np.array([2, 3]),
        )
        self.assertEqual(fit.shape[1], valid.shape[1])
        self.assertEqual(fit.shape[1], test.shape[1])
        self.assertEqual(info["numeric_feature_count"], 1)
        self.assertTrue(np.isnan(fit.toarray()[1, 0]))
        self.assertEqual(test.shape[0], 2)

    def test_target_and_id_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exclude target and applicant ID"):
            build_fold_matrices(
                self.train.assign(TARGET=[0, 1, 0, 1]),
                self.test,
                np.array([0, 1]),
                np.array([2, 3]),
            )
        with self.assertRaisesRegex(ValueError, "exclude target and applicant ID"):
            build_fold_matrices(
                self.train.assign(SK_ID_CURR=[10, 11, 12, 13]),
                self.test,
                np.array([0, 1]),
                np.array([2, 3]),
            )

    def test_rank_percentiles_are_finite_and_order_preserving(self):
        ranks = rank_percentile(np.array([0.2, 0.1, 0.3]))
        self.assertTrue(np.isfinite(ranks).all())
        self.assertLess(ranks[1], ranks[0])
        self.assertLess(ranks[0], ranks[2])


if __name__ == "__main__":
    unittest.main()
