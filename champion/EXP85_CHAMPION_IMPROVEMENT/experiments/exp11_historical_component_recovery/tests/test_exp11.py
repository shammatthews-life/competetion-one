from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.run_exp11 import build_xgb_features, fit_partition_one_hot


class Exp11FeatureTests(unittest.TestCase):
    def test_unknown_categories_do_not_expand_fit_vocabulary(self) -> None:
        from source.run_exp11 import CATEGORICAL_COLUMNS

        fit = pd.DataFrame(
            {column: ["known", "known", np.nan] for column in CATEGORICAL_COLUMNS}
        )
        apply = pd.DataFrame(
            {column: ["unseen", np.nan] for column in CATEGORICAL_COLUMNS}
        )
        fit_dummy, apply_dummy = fit_partition_one_hot(fit, apply)

        self.assertEqual(tuple(fit_dummy.columns), tuple(apply_dummy.columns))
        self.assertEqual(fit_dummy.shape[1], len(CATEGORICAL_COLUMNS) * 3)
        for column in CATEGORICAL_COLUMNS:
            mask = [f"_{column}_" in name for name in fit_dummy.columns]
            self.assertEqual(int(apply_dummy.loc[0, mask].sum()), 1)
            self.assertEqual(int(apply_dummy.loc[1, mask].sum()), 1)

    def test_raw_feature_recipe_has_stable_fit_apply_schema(self) -> None:

        rows = []
        for index in range(16):
            row = {
                "SK_ID_CURR": 1000 + index,
                "TARGET": index % 2,
                "NAME_CONTRACT_TYPE": "Cash loans" if index % 2 else "Revolving loans",
                "CODE_GENDER": "M" if index % 2 else "F",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Single",
                "NAME_HOUSING_TYPE": "House",
                "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "DAYS_EMPLOYED": -1000.0 - index,
                "AMT_INCOME_TOTAL": 100000.0 + index,
                "AMT_CREDIT": 500000.0 + index,
                "AMT_ANNUITY": 25000.0 + index,
                "AMT_GOODS_PRICE": 450000.0 + index,
                "DAYS_BIRTH": -12000.0 - index,
                "DAYS_REGISTRATION": -3000.0 - index,
                "DAYS_ID_PUBLISH": -2000.0 - index,
                "EXT_SOURCE_1": 0.1 + 0.01 * index,
                "EXT_SOURCE_2": 0.2 + 0.01 * index,
                "EXT_SOURCE_3": 0.3 + 0.01 * index,
                "CNT_FAM_MEMBERS": 2.0,
                "DAYS_LAST_PHONE_CHANGE": -100.0 - index,
                "FLAG_OWN_CAR": "Y" if index % 2 else "N",
                "FLAG_OWN_REALTY": "Y",
                "OWN_CAR_AGE": float(index),
            }
            for name in (
                "AMT_REQ_CREDIT_BUREAU_HOUR",
                "AMT_REQ_CREDIT_BUREAU_DAY",
                "AMT_REQ_CREDIT_BUREAU_WEEK",
                "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT",
                "AMT_REQ_CREDIT_BUREAU_YEAR",
            ):
                row[name] = float(index)
            rows.append(row)
        frame = pd.DataFrame(rows)
        fit, apply = build_xgb_features(frame.iloc[:12], frame.iloc[12:])

        self.assertEqual(fit.shape[0], 12)
        self.assertEqual(apply.shape[0], 4)
        self.assertGreater(fit.shape[1], 60)
        self.assertTrue(
            np.isfinite(
                fit.select_dtypes(include=[np.number]).to_numpy()
            ).any()
        )
        self.assertEqual(tuple(fit.columns), tuple(apply.columns))


if __name__ == "__main__":
    unittest.main()
