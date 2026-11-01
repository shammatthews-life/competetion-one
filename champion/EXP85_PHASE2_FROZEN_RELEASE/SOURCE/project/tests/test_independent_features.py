import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_DIR))

from run_independent_baseline import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    build_rowwise_features,
    make_lightgbm_frames,
    model_metrics_summary,
)


def sample_raw() -> pd.DataFrame:
    rows = [
        {
            "SK_ID_CURR": 101,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F",
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 120000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House",
            "REGION_POPULATION_RELATIVE": 0.02,
            "DAYS_BIRTH": -14000.0,
            "DAYS_EMPLOYED": -1200.0,
            "DAYS_REGISTRATION": -3000.0,
            "DAYS_ID_PUBLISH": -5000.0,
            "OWN_CAR_AGE": 5.0,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "FLAG_WORK_PHONE": 0,
            "OCCUPATION_TYPE": "Laborers",
            "CNT_FAM_MEMBERS": 2.0,
            "REGION_RATING_CLIENT": 2,
            "ORGANIZATION_TYPE": "Business Entity Type 3",
            "EXT_SOURCE_1": 0.3,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": np.nan,
            "DAYS_LAST_PHONE_CHANGE": -1000.0,
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
            "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
            "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 2.0,
        },
        {
            "SK_ID_CURR": 202,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "CNT_CHILDREN": 1,
            "AMT_INCOME_TOTAL": 80000.0,
            "AMT_CREDIT": 200000.0,
            "AMT_ANNUITY": 10000.0,
            "AMT_GOODS_PRICE": np.nan,
            "NAME_INCOME_TYPE": "Commercial associate",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_HOUSING_TYPE": "House",
            "REGION_POPULATION_RELATIVE": 0.01,
            "DAYS_BIRTH": -18000.0,
            "DAYS_EMPLOYED": 365243.0,
            "DAYS_REGISTRATION": -2500.0,
            "DAYS_ID_PUBLISH": -6000.0,
            "OWN_CAR_AGE": np.nan,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 0,
            "FLAG_WORK_PHONE": 1,
            "OCCUPATION_TYPE": np.nan,
            "CNT_FAM_MEMBERS": 3.0,
            "REGION_RATING_CLIENT": 3,
            "ORGANIZATION_TYPE": "XNA",
            "EXT_SOURCE_1": np.nan,
            "EXT_SOURCE_2": 0.2,
            "EXT_SOURCE_3": 0.6,
            "DAYS_LAST_PHONE_CHANGE": -2000.0,
            "AMT_REQ_CREDIT_BUREAU_HOUR": np.nan,
            "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
            "AMT_REQ_CREDIT_BUREAU_QRT": 1.0,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        },
    ]
    return pd.DataFrame(rows)


class IndependentFeatureTests(unittest.TestCase):
    def test_features_are_rowwise_and_id_invariant(self):
        raw = sample_raw()
        original = build_rowwise_features(raw)
        permuted = raw.iloc[::-1]
        rebuilt = build_rowwise_features(permuted).iloc[::-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, rebuilt)
        self.assertNotIn(ID_COLUMN, original.columns)
        self.assertNotIn(TARGET_COLUMN, original.columns)

    def test_categorical_missing_values_are_explicit(self):
        features = build_rowwise_features(sample_raw())
        self.assertEqual(
            set(features[list(CATEGORICAL_COLUMNS)].iloc[1].tolist()).intersection(
                {"MISSING"}
            ),
            {"MISSING"},
        )

    def test_lightgbm_train_test_schemas_are_aligned(self):
        raw = sample_raw()
        train = build_rowwise_features(raw)
        test = build_rowwise_features(raw.iloc[:1].copy())
        X_train, X_test = make_lightgbm_frames(train, test)
        self.assertEqual(list(X_train.columns), list(X_test.columns))
        values = X_test.to_numpy(dtype=np.float64)
        self.assertFalse(np.isinf(values).any())
        self.assertTrue(np.isnan(values).any())

    def test_infinite_raw_numeric_values_are_rejected(self):
        raw = sample_raw()
        raw.loc[0, "AMT_INCOME_TOTAL"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            build_rowwise_features(raw)

    def test_metrics_summary_excludes_saved_prediction_arrays(self):
        summary = model_metrics_summary(
            {
                "lightgbm": {
                    "auc": 0.75,
                    "oof": np.array([0.1, 0.9]),
                    "test": np.array([0.2]),
                },
                "blend_auc": 0.76,
            }
        )
        self.assertEqual(summary["lightgbm"], {"auc": 0.75})
        self.assertEqual(summary["blend_auc"], 0.76)
        json.dumps(summary, allow_nan=False)

    def test_all_missing_ext_scores_have_stable_missing_features(self):
        raw = sample_raw()
        raw.loc[:, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = np.nan
        features = build_rowwise_features(raw)
        self.assertTrue(
            features[["EXT_MEAN", "EXT_MIN", "EXT_MAX", "EXT_PRODUCT"]]
            .iloc[0]
            .isna()
            .all()
        )
        self.assertEqual(features.loc[0, "EXT_STD"], 0.0)

    def test_target_column_is_rejected(self):
        raw = sample_raw()
        raw[TARGET_COLUMN] = [0, 1]
        with self.assertRaises(ValueError):
            build_rowwise_features(raw)


if __name__ == "__main__":
    unittest.main()
