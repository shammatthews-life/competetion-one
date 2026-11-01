import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_experiment import build_decoded_loan_features  # noqa: E402


def loan_rows(targets):
    credit = np.array([10000.0, 20000.0, 30000.0, 40000.0])
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(len(credit)),
            "TARGET": targets,
            "NAME_CONTRACT_TYPE": ["Cash loans"] * len(credit),
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit * 0.1 + 129.5,
            "AMT_GOODS_PRICE": credit / (1.0 + 0.0132 * 5.0),
        }
    )


class DecodedLoanFeatureTests(unittest.TestCase):
    def test_fee_and_product_keys_detect_constructed_patterns(self):
        raw = loan_rows([0, 1, 0, 1])
        train, test, diagnostics = build_decoded_loan_features(
            raw, raw.drop(columns=["TARGET"])
        )
        self.assertTrue((train["FEE_DECODED"] == 1).all())
        self.assertTrue((train["PRODUCT_KEY_VALID"] == 1).all())
        np.testing.assert_allclose(train["PRODUCT_KEY_K"], 5.0, atol=1e-10)
        self.assertEqual(len(train), len(raw))
        self.assertEqual(len(test), len(raw))
        self.assertEqual(diagnostics["fee_decoded_rows"], 2 * len(raw))

    def test_labels_do_not_affect_decoded_features(self):
        first = loan_rows([0, 0, 1, 1])
        second = loan_rows([1, 1, 0, 0])
        first_train, first_test, _ = build_decoded_loan_features(
            first, first.drop(columns=["TARGET"])
        )
        second_train, second_test, _ = build_decoded_loan_features(
            second, second.drop(columns=["TARGET"])
        )
        pd.testing.assert_frame_equal(first_train, second_train)
        pd.testing.assert_frame_equal(first_test, second_test)

    def test_missing_and_zero_denominators_remain_finite_or_missing(self):
        raw = loan_rows([0, 1, 0, 1])
        raw.loc[0, "AMT_CREDIT"] = 0.0
        raw.loc[1, "AMT_GOODS_PRICE"] = np.nan
        train, test, _ = build_decoded_loan_features(
            raw, raw.drop(columns=["TARGET"])
        )
        self.assertFalse(
            np.isinf(train.to_numpy(dtype=np.float64, copy=False)).any()
        )
        self.assertFalse(
            np.isinf(test.to_numpy(dtype=np.float64, copy=False)).any()
        )
        self.assertTrue(
            np.isnan(train.loc[0, "ANNUITY_CREDIT_RATIO_DECODED"])
        )


if __name__ == "__main__":
    unittest.main()
