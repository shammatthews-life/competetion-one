from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from phase1_features import build_decoded_loan_features
from preprocessing import CATEGORICAL_COLUMNS, build_rowwise_features, make_lightgbm_frames


def raw_rows(target: bool = False) -> pd.DataFrame:
    rows = []
    for index, (credit, annuity, ext) in enumerate(
        ((10000.0, 329.5, 0.3), (20000.0, 529.5, 0.5), (30000.0, 729.5, np.nan))
    ):
        row = {
            "SK_ID_CURR": 101 + index,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F" if index % 2 == 0 else "M",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House",
            "OCCUPATION_TYPE": "Laborers",
            "ORGANIZATION_TYPE": "Business",
            "CNT_CHILDREN": index,
            "AMT_INCOME_TOTAL": 90000.0 + index * 1000,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": credit / 1.132,
            "CNT_FAM_MEMBERS": 2.0,
            "DAYS_BIRTH": -14000.0 - index,
            "DAYS_EMPLOYED": -1200.0,
            "DAYS_REGISTRATION": -3000.0,
            "DAYS_ID_PUBLISH": -5000.0,
            "DAYS_LAST_PHONE_CHANGE": -1000.0,
            "EXT_SOURCE_1": ext,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.6,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 2.0,
        }
        if target:
            row["TARGET"] = index % 2
        rows.append(row)
    return pd.DataFrame(rows)


class Phase1FeatureTests(unittest.TestCase):
    def test_decoded_fee_features_and_product_tier_are_recovered(self):
        raw = raw_rows(target=True)
        fee_train, fee_test, info = build_decoded_loan_features(raw, raw.iloc[:1])
        self.assertEqual(info["fee_amount"], 129.5)
        self.assertTrue(fee_train["FEE_DECODED"].iloc[:2].eq(1).all())
        self.assertTrue(fee_train["PRODUCT_KEY_VALID"].eq(1).all())
        self.assertTrue(fee_train["PRODUCT_KEY_K"].eq(10).all())
        self.assertEqual(len(fee_test), 1)

    def test_features_are_label_independent_and_preserve_rows(self):
        raw = raw_rows(target=True)
        train = build_rowwise_features(raw.drop(columns="TARGET"))
        changed_target = raw.copy()
        changed_target["TARGET"] = 1 - changed_target["TARGET"]
        fee_a, test_a, _ = build_decoded_loan_features(raw, raw.iloc[:1])
        fee_b, test_b, _ = build_decoded_loan_features(changed_target, raw.iloc[:1])
        pd.testing.assert_frame_equal(train, build_rowwise_features(changed_target.drop(columns="TARGET")))
        pd.testing.assert_frame_equal(fee_a, fee_b)
        pd.testing.assert_frame_equal(test_a, test_b)
        self.assertEqual(len(train), len(raw))
        self.assertTrue(set(CATEGORICAL_COLUMNS).issubset(train.columns))

    def test_lightgbm_train_test_matrices_have_identical_schema(self):
        raw = raw_rows()
        train = build_rowwise_features(raw)
        test = build_rowwise_features(raw.iloc[:2])
        X_train, X_test = make_lightgbm_frames(train, test)
        self.assertEqual(list(X_train.columns), list(X_test.columns))
        self.assertFalse(np.isinf(X_train.to_numpy(dtype=np.float64)).any())


if __name__ == "__main__":
    unittest.main()
