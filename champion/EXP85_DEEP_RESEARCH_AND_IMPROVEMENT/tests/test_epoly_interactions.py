from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

from run_epoly_interactions import POLY_COLUMNS, add_epoly_features, build_epoly_features


class EPolyInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = pd.DataFrame(
            {
                "SK_ID_CURR": [10, 11],
                "EXT_SOURCE_1": [0.2, np.nan],
                "EXT_SOURCE_2": [0.4, 0.5],
                "EXT_SOURCE_3": [0.6, 0.7],
                "AMT_ANNUITY": [12.0, 15.0],
                "AMT_CREDIT": [120.0, 0.0],
                "AMT_INCOME_TOTAL": [100.0, 200.0],
                "AMT_GOODS_PRICE": [90.0, 150.0],
                "DAYS_BIRTH": [-3652.5, -7305.0],
            }
        )

    def test_interactions_are_rowwise_finite_or_missing(self):
        features = build_epoly_features(self.raw)
        self.assertEqual(list(features.columns), list(POLY_COLUMNS))
        self.assertAlmostEqual(features.loc[0, "P_EXTMEAN_PAY"], 0.04)
        self.assertAlmostEqual(features.loc[0, "P_EXTMEAN_CTI"], 0.48)
        self.assertAlmostEqual(features.loc[0, "P_EXTMEAN_PAY_AGE"], 0.4)
        self.assertAlmostEqual(features.loc[0, "P_EXTMEAN_ANN_INC"], 0.048)
        self.assertTrue(np.isnan(features.loc[1, "P_EXTMEAN_PAY"]))
        self.assertTrue(np.isfinite(features.to_numpy()).any())
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_target_is_rejected_and_id_is_not_added_as_a_feature(self):
        raw = self.raw.assign(TARGET=[0, 1])
        with self.assertRaisesRegex(ValueError, "must not receive TARGET"):
            build_epoly_features(raw)
        base = pd.DataFrame({"BASE": [1.0, 2.0]})
        combined = add_epoly_features(self.raw, base)
        self.assertNotIn("SK_ID_CURR", combined.columns)
        self.assertEqual(list(combined.columns), ["BASE", *POLY_COLUMNS])


if __name__ == "__main__":
    unittest.main()
