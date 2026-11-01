from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from phase2_group_context import build_fold_group_context


def make_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "NAME_INCOME_TYPE": ["A", "A", "A", "A", "B", "B"],
            "NAME_EDUCATION_TYPE": ["E", "E", "E", "E", "F", "F"],
            "OCCUPATION_TYPE": ["O", "O", "O", "O", "P", "P"],
            "ORGANIZATION_TYPE": ["ORG", "ORG", "ORG", "ORG", "ORG2", "ORG2"],
            "AMT_INCOME_TOTAL": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "AMT_CREDIT": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "AMT_ANNUITY": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "AMT_GOODS_PRICE": [90.0, 180.0, 270.0, 360.0, 450.0, 540.0],
            "EXT_SOURCE_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "EXT_SOURCE_2": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66],
            "EXT_SOURCE_3": [0.12, 0.24, 0.36, 0.48, 0.60, 0.72],
            "DAYS_BIRTH": [-3652.5, -7305.0, -10957.5, -14610.0, -18262.5, -21915.0],
            "DAYS_EMPLOYED": [-365.25, -730.5, -1095.75, -1461.0, 365243.0, 365243.0],
        }
    )
    test = train.iloc[:2].copy().reset_index(drop=True)
    return train, test


class GroupContextTests(unittest.TestCase):
    def test_training_peer_mean_excludes_own_measurement(self) -> None:
        train, test = make_raw()
        fit_indices = np.array([0, 1, 2, 3])
        valid_indices = np.array([4, 5])
        fit_features, _ = build_fold_group_context(
            train, test, fit_indices, valid_indices
        )
        mean_column = "GROUP_MEAN__NAME_INCOME_TYPE__AMT_INCOME_TOTAL"
        self.assertAlmostEqual(float(fit_features.loc[0, mean_column]), 30.0)
        self.assertAlmostEqual(float(fit_features.loc[1, mean_column]), 26.666666, places=5)

    def test_validation_values_do_not_change_fold_peer_statistics(self) -> None:
        train, test = make_raw()
        fit_indices = np.array([0, 1, 2, 3])
        valid_indices = np.array([4, 5])
        first, _ = build_fold_group_context(train, test, fit_indices, valid_indices)
        changed = train.copy()
        changed.loc[4, "AMT_INCOME_TOTAL"] = 900000.0
        second, _ = build_fold_group_context(changed, test, fit_indices, valid_indices)
        mean_column = "GROUP_MEAN__NAME_INCOME_TYPE__AMT_INCOME_TOTAL"
        np.testing.assert_array_equal(
            first.loc[fit_indices, mean_column].to_numpy(),
            second.loc[fit_indices, mean_column].to_numpy(),
        )

    def test_unseen_groups_fall_back_to_training_global_mean(self) -> None:
        train, test = make_raw()
        test.loc[:, "NAME_INCOME_TYPE"] = "UNSEEN"
        _, test_features = build_fold_group_context(
            train,
            test,
            np.array([0, 1, 2, 3]),
            np.array([4, 5]),
        )
        mean_column = "GROUP_MEAN__NAME_INCOME_TYPE__AMT_INCOME_TOTAL"
        self.assertTrue(np.allclose(test_features[mean_column], 25.0))
        self.assertTrue((test_features["LOG_GROUP_COUNT__NAME_INCOME_TYPE"] == 0).all())

    def test_target_or_id_in_inputs_is_rejected(self) -> None:
        train, test = make_raw()
        train["TARGET"] = [0, 1, 0, 1, 0, 1]
        with self.assertRaisesRegex(ValueError, "target or ID"):
            build_fold_group_context(
                train,
                test,
                np.array([0, 1, 2, 3]),
                np.array([4, 5]),
            )

    def test_fold_indices_must_partition_train_rows(self) -> None:
        train, test = make_raw()
        with self.assertRaisesRegex(ValueError, "partition"):
            build_fold_group_context(
                train,
                test,
                np.array([0, 1, 2]),
                np.array([4, 5]),
            )


if __name__ == "__main__":
    unittest.main()
