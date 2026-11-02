from __future__ import annotations

import unittest
import os
import runpy
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from historical_features import (
    EVALUATED_GROUPS,
    EXCLUDED_GROUPS,
    FeatureAuditError,
    build_feature_frames,
    feature_groups,
    formula_for,
)
from run_exp08 import add_historical_blocks, historical_blocks


def fixture(rows: int = 12) -> pd.DataFrame:
    sequence = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": 60000.0 + sequence * 9000.0,
            "AMT_CREDIT": 50000.0 + sequence * 12000.0,
            "AMT_ANNUITY": 3000.0 + sequence * 500.0,
            "AMT_GOODS_PRICE": 45000.0 + sequence * 10000.0,
            "DAYS_BIRTH": -12000.0 - sequence * 365.0,
            "DAYS_EMPLOYED": -1000.0 - sequence * 40.0,
            "DAYS_REGISTRATION": -500.0 - sequence * 20.0,
            "DAYS_ID_PUBLISH": -300.0 - sequence * 15.0,
            "DAYS_LAST_PHONE_CHANGE": -100.0 - sequence * 5.0,
            "CNT_FAM_MEMBERS": 1.0 + (sequence % 4),
            "CNT_CHILDREN": sequence % 3,
            "EXT_SOURCE_1": np.linspace(0.1, 0.9, rows),
            "EXT_SOURCE_2": np.linspace(0.9, 0.1, rows),
            "EXT_SOURCE_3": np.linspace(0.2, 0.8, rows),
            "NAME_FAMILY_STATUS": np.where(sequence % 2, "Married", "Single"),
            "CODE_GENDER": np.where(sequence % 2, "F", "M"),
            "NAME_EDUCATION_TYPE": np.where(
                sequence % 2, "Higher education", "Secondary / secondary special"
            ),
            "AMT_REQ_CREDIT_BUREAU_HOUR": sequence % 2,
            "AMT_REQ_CREDIT_BUREAU_MON": sequence % 4,
            "AMT_REQ_CREDIT_BUREAU_QRT": sequence % 3,
            "AMT_REQ_CREDIT_BUREAU_YEAR": sequence % 5,
            "FLAG_EMP_PHONE": sequence % 2,
            "FLAG_WORK_PHONE": (sequence + 1) % 2,
            "FLAG_MOBIL": np.ones(rows),
            "REGION_RATING_CLIENT": 1.0 + (sequence % 3),
            "REGION_POPULATION_RELATIVE": 0.01 + sequence / 1000.0,
        }
    )
    frame.loc[0, "AMT_GOODS_PRICE"] = np.nan
    frame.loc[1, "EXT_SOURCE_1"] = np.nan
    frame.loc[2, "EXT_SOURCE_2"] = np.nan
    frame.loc[3, "EXT_SOURCE_3"] = np.nan
    frame.loc[4, "DAYS_EMPLOYED"] = 365243.0
    return frame


class HistoricalFeatureTests(unittest.TestCase):
    def test_feature_count_order_and_group_sizes(self) -> None:
        raw = fixture()
        fit, apply = build_feature_frames(raw.iloc[:8], raw.iloc[8:])
        self.assertEqual(list(fit.columns), list(apply.columns))
        groups = feature_groups(tuple(fit.columns))
        self.assertEqual(tuple(groups), EVALUATED_GROUPS)
        self.assertEqual(len(fit.columns), 660)
        self.assertEqual(len(groups["H_INTERACTIONS"]), 570)
        self.assertEqual(
            [len(groups[group]) for group in EVALUATED_GROUPS],
            [34, 11, 11, 8, 10, 16, 570],
        )
        self.assertEqual(len(EXCLUDED_GROUPS), 2)

    def test_feature_formulas_and_ext_ranks_are_fit_local(self) -> None:
        raw = fixture()
        fit = raw.iloc[:8].copy()
        apply = raw.iloc[8:].copy()
        fit_a, apply_a = build_feature_frames(fit, apply, ("B_RISK",))
        changed_apply = apply.copy()
        changed_apply[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = 1000.0
        fit_b, apply_b = build_feature_frames(fit, changed_apply, ("B_RISK",))
        np.testing.assert_array_equal(
            fit_a.to_numpy(dtype=np.float32),
            fit_b.to_numpy(dtype=np.float32),
        )
        self.assertFalse(
            np.array_equal(
                apply_a["CLAYTON_COPULA_2_5"].to_numpy(),
                apply_b["CLAYTON_COPULA_2_5"].to_numpy(),
            )
        )
        self.assertTrue(np.isfinite(fit_a["MERTON_DD_25"]).all())
        self.assertEqual(
            formula_for("P_RATIO_EXT_MEAN_div_LTV"),
            "core_left / (abs(core_right) + 1) (float32)",
        )

    def test_historical_infinity_is_preserved_and_not_sanitized(self) -> None:
        raw = fixture()
        raw.loc[0, "DAYS_ID_PUBLISH"] = -1.0
        fit, _ = build_feature_frames(raw, raw.iloc[:0].copy(), ("C_TEMPORAL",))
        self.assertTrue(np.isinf(fit.loc[0, "HOUSING_STABILITY"]))

    def test_id_target_and_raw_infinity_are_rejected(self) -> None:
        raw = fixture()
        with self.assertRaises(FeatureAuditError):
            build_feature_frames(raw.assign(TARGET=0), raw)
        with self.assertRaises(FeatureAuditError):
            build_feature_frames(raw.assign(SK_ID_CURR=100), raw.assign(SK_ID_CURR=200))
        contaminated = raw.copy()
        contaminated.loc[0, "AMT_CREDIT"] = np.inf
        with self.assertRaises(FeatureAuditError):
            build_feature_frames(contaminated, raw)

    def test_excluded_signature_and_target_memory_are_not_exposed_as_features(self) -> None:
        raw = fixture()
        fit, apply = build_feature_frames(raw.iloc[:8], raw.iloc[8:])
        for excluded in (
            "N_APPS_ALL",
            "APP_ORDER",
            "DAYS_SINCE_PREV_APP",
            "DAYS_TO_NEXT_APP",
            "UIDY_N",
            "UIDY_SUM",
            "UIDY_NEAR_Y",
            "UIDY_NEAR_D",
            "UIDY_ANYBAD",
        ):
            self.assertNotIn(excluded, fit.columns)
            self.assertNotIn(excluded, apply.columns)

    def test_bounded_parity_against_legacy_executable(self) -> None:
        source = (
            Path(__file__).resolve().parents[4]
            / "05_FEATURE_ENGINEERING"
            / "FEATURE_BUILDERS"
            / "build_hasan_600_features.py"
        )
        if not source.is_file():
            self.skipTest("Historical executable is not present in this checkout")

        raw = fixture(15)
        train = raw.iloc[:10].copy()
        train.insert(0, "SK_ID_CURR", np.arange(100, 110))
        train["TARGET"] = np.tile([0, 1], 5)
        test = raw.iloc[10:].copy()
        test.insert(0, "SK_ID_CURR", np.arange(200, 205))
        legacy_frames = iter((train.copy(), test.copy()))
        with tempfile.TemporaryDirectory() as working_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(working_dir)
                with mock.patch.object(
                    pd,
                    "read_csv",
                    side_effect=lambda *args, **kwargs: next(legacy_frames),
                ), mock.patch.object(
                    pd.DataFrame,
                    "to_feather",
                    lambda *args, **kwargs: None,
                    create=True,
                ), mock.patch("warnings.filterwarnings"):
                    legacy_namespace = runpy.run_path(
                        str(source), run_name="exp08_legacy_parity"
                    )
            finally:
                os.chdir(previous_cwd)

        pooled = pd.concat(
            (train.drop(columns=["SK_ID_CURR", "TARGET"]), test.drop(columns=["SK_ID_CURR"])),
            ignore_index=True,
        )
        reconstructed, _ = build_feature_frames(pooled, pooled.iloc[:0].copy())
        legacy = legacy_namespace["df_features"]
        excluded = {
            "N_APPS_ALL",
            "APP_ORDER",
            "DAYS_SINCE_PREV_APP",
            "DAYS_TO_NEXT_APP",
            "UIDY_N",
            "UIDY_SUM",
            "UIDY_NEAR_Y",
            "UIDY_NEAR_D",
            "UIDY_ANYBAD",
        }
        expected_columns = set(legacy.columns) - excluded
        self.assertEqual(len(legacy.columns), 669)
        self.assertEqual(set(reconstructed.columns), expected_columns)
        for column in reconstructed.columns:
            np.testing.assert_array_equal(
                reconstructed[column].to_numpy(),
                legacy[column].to_numpy(),
                err_msg=f"Legacy parity mismatch for {column}",
            )

    def test_selected_features_are_aligned_to_global_fold_rows(self) -> None:
        train = fixture(12)
        test = fixture(4)
        fit_indices = np.arange(8)
        valid_indices = np.arange(8, 12)
        blocks = historical_blocks(
            train,
            test,
            fit_indices,
            valid_indices,
            ("A_FINANCIAL",),
        )
        selected = blocks["A_FINANCIAL"]["fit"].columns[:2].tolist()
        baseline_cat = pd.DataFrame({"BASE_CAT": np.arange(12, dtype=np.float32)})
        baseline_cat_test = pd.DataFrame({"BASE_CAT": np.arange(4, dtype=np.float32)})
        baseline_lgb = pd.DataFrame({"BASE_LGB": np.arange(12, dtype=np.float32)})
        baseline_lgb_test = pd.DataFrame({"BASE_LGB": np.arange(4, dtype=np.float32)})
        cat, cat_test, lgb, lgb_test, aliases = add_historical_blocks(
            baseline_cat,
            baseline_cat_test,
            baseline_lgb,
            baseline_lgb_test,
            blocks,
            selected,
            fit_indices,
            valid_indices,
            len(test),
        )
        self.assertEqual(list(aliases.values()), selected)
        self.assertEqual(cat.shape, (12, 3))
        self.assertEqual(cat_test.shape, (4, 3))
        self.assertEqual(lgb.columns.tolist(), lgb_test.columns.tolist())
        self.assertTrue(cat.loc[:7, f"EXP08_{selected[0]}"].notna().all())
        self.assertTrue(cat.loc[8:, f"EXP08_{selected[0]}"].notna().all())
        self.assertNotIn("TARGET", cat.columns)
        self.assertNotIn("SK_ID_CURR", cat.columns)


if __name__ == "__main__":
    unittest.main()
