#!/usr/bin/env python3
"""Nested, fold-safe evaluation of reconstructable historical Hasan features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from historical_features import (
    EVALUATED_GROUPS,
    EXCLUDED_GROUPS,
    FeatureAuditError,
    build_feature_frames,
    feature_groups,
    formula_for,
    source_columns_for,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
CHAMPION_DIR = REPOSITORY_ROOT / "champion"
IMPROVEMENT_DIR = CHAMPION_DIR / "EXP85_CHAMPION_IMPROVEMENT"
FROZEN_DIR = CHAMPION_DIR / "EXP85_PHASE2_FROZEN_RELEASE"
EXP06_DIR = IMPROVEMENT_DIR / "experiments" / "exp06_ft_transformer"
EXP07_DIR = IMPROVEMENT_DIR / "experiments" / "exp07_gbdt_feature_ensemble"
EXP06_RUN_DIR = EXP06_DIR / "runs" / "exp06_ft_transformer_seed20261101_v5"
EXP06_RUNNER = EXP06_DIR / "run_experiment.py"
EXP07_RUNNER = EXP07_DIR / "run_exp07.py"
PHASE2_SOURCE = (
    FROZEN_DIR / "SOURCE" / "project" / "source" / "run_independent_baseline.py"
)
PHASE4_SOURCE = (
    FROZEN_DIR
    / "SOURCE"
    / "project"
    / "experiments"
    / "phase_04_final_validation_gate"
    / "run_final_gate.py"
)
HISTORICAL_SOURCE = (
    CHAMPION_DIR
    / "05_FEATURE_ENGINEERING"
    / "FEATURE_BUILDERS"
    / "build_hasan_600_features.py"
)
HISTORICAL_SPEC = (
    CHAMPION_DIR
    / "05_FEATURE_ENGINEERING"
    / "FEATURES_669"
    / "FEATURES_669_MATHEMATICAL_SPECIFICATION.md"
)
HISTORICAL_PARITY_AUDIT = (
    CHAMPION_DIR
    / "05_FEATURE_ENGINEERING"
    / "FEATURE_VALIDATION"
    / "FEATURE_MATRIX_PARITY_AUDIT.md"
)
HISTORICAL_LINEAGE = (
    CHAMPION_DIR / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT" / "FEATURE_LINEAGE.md"
)
HISTORICAL_ARCHITECTURE = (
    CHAMPION_DIR
    / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT"
    / "HISTORICAL_ARCHITECTURE.md"
)
EXTERNAL_HISTORICAL_SOURCE = Path(
    r"D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\build_hasan_600_features.py"
)
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp08_historical_seed20261101_v1"
OUTER_SEED = 20261101
INNER_SEED_BASE = 20261101
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
HOUSING_INFINITY_FEATURE = "HOUSING_STABILITY"
MAX_SELECTED_GROUPS = 3
MAX_INTERACTION_FEATURES = 60
BLEND_WEIGHTS = tuple(np.linspace(0.0, 1.0, 21).tolist())
ADVANCEMENT_GATE = {
    "minimum_pooled_auc_delta": 0.001,
    "minimum_positive_folds": 4,
    "minimum_mean_delta_standard_errors": 2.0,
    "folds": 5,
}
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
XGB_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 5,
    "eta": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "lambda": 4.0,
    "alpha": 0.0,
    "max_bin": 256,
    "nthread": 8,
    "verbosity": 0,
}
EXCLUDED_FEATURE_FORMULAS = {
    "N_APPS_ALL": "count of rows sharing pooled train/test entity key",
    "APP_ORDER": "groupby(entity key).cumcount() after pooled timeline sort",
    "DAYS_SINCE_PREV_APP": "within-entity registration-day diff; missing filled -999",
    "DAYS_TO_NEXT_APP": "within-entity reverse registration-day diff; missing filled -999",
    "UIDY_N": "count of matching entity labels in global seed-42 OOF fold reference",
    "UIDY_SUM": "sum of matching entity labels in global seed-42 OOF fold reference",
    "UIDY_NEAR_Y": "nearest matching entity target in global seed-42 OOF fold reference",
    "UIDY_NEAR_D": "absolute day gap to nearest matching entity target",
    "UIDY_ANYBAD": "indicator that matching entity labels include a positive target",
}
EXCLUDED_FEATURE_SOURCES = {
    "N_APPS_ALL": (
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
    ),
    "APP_ORDER": (
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
    ),
    "DAYS_SINCE_PREV_APP": ("DAYS_REGISTRATION",),
    "DAYS_TO_NEXT_APP": ("DAYS_REGISTRATION",),
    "UIDY_N": ("DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "TARGET"),
    "UIDY_SUM": ("DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "TARGET"),
    "UIDY_NEAR_Y": ("DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "TARGET"),
    "UIDY_NEAR_D": ("DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"),
    "UIDY_ANYBAD": (
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "TARGET",
    ),
}
EXPECTED_GROUP_SIZES = {
    "A_FINANCIAL": 34,
    "B_RISK": 11,
    "C_TEMPORAL": 11,
    "D_DEMOGRAPHIC": 8,
    "E_INQUIRY": 10,
    "F_EXTERNAL": 16,
    "H_INTERACTIONS": 570,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def load_module(name: str, path: Path) -> Any:
    require(path.is_file(), f"Required Python source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_record(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(actual["sha256"] == record["sha256"], f"{label} SHA-256 mismatch")
    require(actual["size_bytes"] == record["size_bytes"], f"{label} size mismatch")
    return path


def _raw_without_identity(frame: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    columns = [ID_COLUMN]
    if has_target:
        columns.append(TARGET_COLUMN)
    return frame.drop(columns=columns).reset_index(drop=True)


def build_phase2_frames(
    runtime: Any,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    raw_train_features: pd.DataFrame,
    raw_test_features: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[pd.DataFrame, ...]:
    frames = runtime.make_variant_fold(
        "phase2_fold_local",
        train_raw,
        test_raw,
        raw_train_features,
        raw_test_features,
        fit_indices,
        valid_indices,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    runtime.validate_feature_audit(frames[4], "EXP08 Phase 2 feature generation")
    return frames


def historical_blocks(
    train_without_identity: pd.DataFrame,
    test_without_identity: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    groups: tuple[str, ...] = EVALUATED_GROUPS,
) -> dict[str, dict[str, pd.DataFrame]]:
    fit_raw = train_without_identity.iloc[fit_indices].reset_index(drop=True)
    valid_raw = train_without_identity.iloc[valid_indices].reset_index(drop=True)
    apply_raw = pd.concat(
        (valid_raw, test_without_identity.reset_index(drop=True)),
        ignore_index=True,
    )
    blocks: dict[str, dict[str, pd.DataFrame]] = {}
    for group in groups:
        fit_features, apply_features = build_feature_frames(
            fit_raw, apply_raw, (group,)
        )
        distribution: dict[str, dict[str, int]] = {}
        inf_columns = [
            name
            for name in fit_features.columns
            if np.isinf(fit_features[name].to_numpy(dtype=np.float64)).any()
            or np.isinf(apply_features[name].to_numpy(dtype=np.float64)).any()
        ]
        forbidden_inf = [
            name for name in inf_columns if name != HOUSING_INFINITY_FEATURE
        ]
        if forbidden_inf:
            raise FeatureAuditError(
                f"{group} has unexpected infinite historical features: "
                f"{forbidden_inf[:10]}"
            )
        for name in fit_features.columns:
            values = np.concatenate(
                (
                    fit_features[name].to_numpy(dtype=np.float64),
                    apply_features[name].to_numpy(dtype=np.float64),
                )
            )
            distribution[name] = {
                "missing": int(np.isnan(values).sum()),
                "infinite": int(np.isinf(values).sum()),
            }
        if HOUSING_INFINITY_FEATURE in fit_features.columns:
            fit_features = fit_features.drop(columns=[HOUSING_INFINITY_FEATURE])
            apply_features = apply_features.drop(
                columns=[HOUSING_INFINITY_FEATURE]
            )
        blocks[group] = {
            "fit": fit_features,
            "apply": apply_features,
            "distribution": distribution,
        }
    return blocks


def _numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(left.dtype):
        return False
    left_values = left.to_numpy(dtype=np.float64, na_value=np.nan)
    right_values = right.to_numpy(dtype=np.float64, na_value=np.nan)
    return bool(np.array_equal(left_values, right_values, equal_nan=True))


def append_family_to_lgb(
    base_fit: pd.DataFrame,
    base_apply: pd.DataFrame,
    hist_fit: pd.DataFrame,
    hist_apply: pd.DataFrame,
    drop_exact_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
    fit_result = base_fit.reset_index(drop=True).copy()
    apply_result = base_apply.reset_index(drop=True).copy()
    aliases: dict[str, str] = {}
    exact_duplicates: list[str] = []
    for original in hist_fit.columns:
        if drop_exact_duplicates and (
            original in fit_result.columns
            and original in apply_result.columns
            and _numeric_equal(fit_result[original], hist_fit[original])
            and _numeric_equal(apply_result[original], hist_apply[original])
        ):
            exact_duplicates.append(original)
            continue
        alias = f"EXP08_{original}"
        require(
            alias not in fit_result.columns and alias not in apply_result.columns,
            f"Historical feature alias collides with existing feature: {alias}",
        )
        fit_result[alias] = hist_fit[original].to_numpy(dtype=np.float32)
        apply_result[alias] = hist_apply[original].to_numpy(dtype=np.float32)
        aliases[alias] = original
    require(
        fit_result.columns.tolist() == apply_result.columns.tolist(),
        "Historical family added different fit/apply feature schemas",
    )
    return fit_result, apply_result, aliases, exact_duplicates


def add_historical_blocks(
    base_cat: pd.DataFrame,
    base_cat_test: pd.DataFrame,
    base_lgb: pd.DataFrame,
    base_lgb_test: pd.DataFrame,
    blocks: dict[str, dict[str, pd.DataFrame]],
    selected_original_names: list[str],
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not selected_original_names:
        return (
            base_cat.copy(),
            base_cat_test.copy(),
            base_lgb.copy(),
            base_lgb_test.copy(),
            {},
        )
    selected_groups = [
        group
        for group in EVALUATED_GROUPS
        if group in blocks
        and any(
            name in blocks[group]["fit"].columns
            for name in selected_original_names
        )
    ]
    fit_parts = []
    apply_parts = []
    for group in selected_groups:
        names = [
            name
            for name in selected_original_names
            if name in blocks[group]["fit"].columns
        ]
        if names:
            fit_parts.append(blocks[group]["fit"].loc[:, names])
            apply_parts.append(blocks[group]["apply"].loc[:, names])
    hist_fit = pd.concat(fit_parts, axis=1)
    hist_apply = pd.concat(apply_parts, axis=1)
    require(
        hist_fit.columns.tolist() == selected_original_names
        and hist_apply.columns.tolist() == selected_original_names,
        "Selected historical feature order differs across fit/apply blocks",
    )
    expanded = np.full(
        (len(base_cat), len(selected_original_names)),
        np.nan,
        dtype=np.float32,
    )
    expanded[fit_indices] = hist_fit.to_numpy(dtype=np.float32)
    expanded[valid_indices] = hist_apply.iloc[: len(valid_indices)].to_numpy(
        dtype=np.float32
    )
    test_values = hist_apply.iloc[len(valid_indices) :].to_numpy(dtype=np.float32)
    require(len(test_values) == test_count, "Historical test block row count differs")
    historical_train = pd.DataFrame(
        expanded,
        columns=selected_original_names,
    )
    historical_test = pd.DataFrame(
        test_values,
        columns=selected_original_names,
    )

    cat_train = base_cat.copy()
    cat_test = base_cat_test.copy()
    lgb_train = base_lgb.copy()
    lgb_test = base_lgb_test.copy()
    aliases: dict[str, str] = {}
    for original in selected_original_names:
        alias = f"EXP08_{original}"
        require(
            alias not in cat_train.columns
            and alias not in cat_test.columns
            and alias not in lgb_train.columns
            and alias not in lgb_test.columns,
            f"Selected historical alias collides with baseline: {alias}",
        )
        cat_train[alias] = historical_train[original]
        cat_test[alias] = historical_test[original]
        lgb_train[alias] = historical_train[original]
        lgb_test[alias] = historical_test[original]
        aliases[alias] = original
    require(
        cat_train.columns.tolist() == cat_test.columns.tolist()
        and lgb_train.columns.tolist() == lgb_test.columns.tolist(),
        "Combined candidate train/test feature schemas differ",
    )
    require(
        not {ID_COLUMN, TARGET_COLUMN}.intersection(cat_train.columns)
        and not {ID_COLUMN, TARGET_COLUMN}.intersection(lgb_train.columns),
        "ID or target reached candidate features",
    )
    return cat_train, cat_test, lgb_train, lgb_test, aliases


def fit_family_inner(
    runtime: Any,
    base_fit: pd.DataFrame,
    base_valid: pd.DataFrame,
    hist_fit: pd.DataFrame,
    hist_apply: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold_number: int,
) -> dict[str, Any]:
    candidate_fit, candidate_valid, aliases, duplicates = append_family_to_lgb(
        base_fit.iloc[fit_indices],
        base_valid,
        hist_fit,
        hist_apply.iloc[: len(valid_indices)],
    )
    settings, rounds, patience = runtime.make_inner_lgbm_settings(fold_number)
    training = lgb.Dataset(
        candidate_fit,
        label=labels[fit_indices],
        free_raw_data=True,
    )
    validation = lgb.Dataset(
        candidate_valid,
        label=labels[valid_indices],
        reference=training,
        free_raw_data=True,
    )
    model = lgb.train(
        settings,
        training,
        num_boost_round=min(rounds, 600),
        valid_sets=[validation],
        callbacks=[lgb.early_stopping(min(patience, 50), verbose=False)],
    )
    best_iteration = max(1, int(model.best_iteration))
    prediction = model.predict(
        candidate_valid,
        num_iteration=best_iteration,
    ).astype(np.float32)
    importance = {
        name: float(value)
        for name, value in zip(
            candidate_fit.columns,
            model.feature_importance(importance_type="gain"),
        )
    }
    del model, training, validation
    require(np.isfinite(prediction).all(), "Inner family prediction is non-finite")
    return {
        "prediction": prediction,
        "auc": float(roc_auc_score(labels[valid_indices], prediction)),
        "best_iteration": best_iteration,
        "importance": importance,
        "aliases": aliases,
        "exact_duplicates": duplicates,
    }


def fit_family_outer(
    runtime: Any,
    base_lgb: pd.DataFrame,
    base_lgb_test: pd.DataFrame,
    hist_fit: pd.DataFrame,
    hist_apply: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold_number: int,
    iterations: int,
    selected_original_names: list[str],
) -> dict[str, Any]:
    hist_fit = hist_fit.loc[:, selected_original_names]
    hist_apply = hist_apply.loc[:, selected_original_names]
    candidate_fit, candidate_apply, aliases, duplicates = append_family_to_lgb(
        base_lgb.iloc[fit_indices],
        pd.concat(
            (
                base_lgb.iloc[valid_indices].reset_index(drop=True),
                base_lgb_test.reset_index(drop=True),
            ),
            ignore_index=True,
        ),
        hist_fit,
        hist_apply,
        drop_exact_duplicates=False,
    )
    settings, _, _ = runtime.make_inner_lgbm_settings(fold_number)
    training = lgb.Dataset(
        candidate_fit,
        label=labels[fit_indices],
        free_raw_data=True,
    )
    model = lgb.train(settings, training, num_boost_round=iterations)
    valid_prediction = model.predict(
        candidate_apply.iloc[: len(valid_indices)]
    ).astype(np.float32)
    test_prediction = model.predict(
        candidate_apply.iloc[len(valid_indices) :]
    ).astype(np.float32)
    importance = {
        name: float(value)
        for name, value in zip(
            candidate_fit.columns,
            model.feature_importance(importance_type="gain"),
        )
    }
    del model, training
    require(
        np.isfinite(valid_prediction).all()
        and np.isfinite(test_prediction).all(),
        "Outer family predictions are non-finite",
    )
    return {
        "valid_prediction": valid_prediction,
        "test_prediction": test_prediction,
        "importance": importance,
        "aliases": aliases,
        "exact_duplicates": duplicates,
    }


def selected_features(
    group: str,
    inner_result: dict[str, Any],
    max_interaction_features: int = MAX_INTERACTION_FEATURES,
) -> list[str]:
    aliases = inner_result["aliases"]
    ranked = sorted(
        (
            (original, float(inner_result["importance"].get(alias, 0.0)))
            for alias, original in aliases.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    positive = [(name, gain) for name, gain in ranked if gain > 0.0]
    if group == "H_INTERACTIONS":
        return [name for name, _ in positive[:max_interaction_features]]
    return [name for name, _ in positive]


def xgboost_gpu_preflight() -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as error:
        return {
            "available": False,
            "reason": f"xgboost import failed: {error}",
            "cpu_fallback_used": False,
        }
    try:
        if int(xgb.__version__.split(".", maxsplit=1)[0]) >= 2:
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 2,
                "verbosity": 0,
            }
        else:
            params = {
                "objective": "binary:logistic",
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "max_depth": 2,
                "verbosity": 0,
            }
        data = xgb.DMatrix(
            np.asarray(
                [[0.0, 0.1], [1.0, 0.9], [0.2, 0.4], [0.8, 0.6]],
                dtype=np.float32,
            ),
            label=np.asarray([0, 1, 0, 1], dtype=np.float32),
        )
        model = xgb.train(params, data, num_boost_round=2)
        prediction = model.predict(data)
        require(np.isfinite(prediction).all(), "XGBoost CUDA preflight was non-finite")
        config = model.save_config()
        require(
            '"device":"cuda"' in config or '"device":"cuda:0"' in config,
            "XGBoost did not confirm CUDA in saved model configuration",
        )
        return {
            "available": True,
            "version": xgb.__version__,
            "device": "CUDA",
            "parameters": params,
            "cpu_fallback_used": False,
        }
    except Exception as error:
        return {
            "available": False,
            "version": xgb.__version__,
            "reason": f"{type(error).__name__}: {error}",
            "cpu_fallback_used": False,
        }


def catboost_gpu_preflight() -> dict[str, Any]:
    model = CatBoostClassifier(
        iterations=2,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32),
        np.asarray([0, 1, 0, 1], dtype=np.int8),
    )
    actual_task = str(model.get_all_params().get("task_type", "")).upper()
    require(actual_task == "GPU", f"CatBoost preflight left GPU: {actual_task}")
    return {
        "available": True,
        "task_type": actual_task,
        "device": "CUDA:0",
        "iterations": 2,
    }


def fit_xgb_inner(
    xgb: Any,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    params = {**XGB_CONFIG, "seed": seed}
    training = xgb.DMatrix(
        train_frame.iloc[fit_indices],
        label=labels[fit_indices],
        feature_names=list(train_frame.columns),
    )
    validation = xgb.DMatrix(
        valid_frame,
        label=labels[valid_indices],
        feature_names=list(valid_frame.columns),
    )
    model = xgb.train(
        params,
        training,
        num_boost_round=1200,
        evals=[(validation, "inner_valid")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    rounds = max(1, int(model.best_iteration) + 1)
    prediction = model.predict(
        validation,
        iteration_range=(0, rounds),
    ).astype(np.float32)
    device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    require(str(device).startswith("cuda"), f"XGBoost inner model left CUDA: {device}")
    del model, training, validation
    require(np.isfinite(prediction).all(), "XGBoost inner prediction is non-finite")
    return {"prediction": prediction, "best_iteration": rounds}


def fit_xgb_outer(
    xgb: Any,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    iterations: int,
    seed: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    params = {**XGB_CONFIG, "seed": seed}
    training = xgb.DMatrix(
        train_frame.iloc[fit_indices],
        label=labels[fit_indices],
        feature_names=list(train_frame.columns),
    )
    valid = xgb.DMatrix(
        train_frame.iloc[valid_indices],
        feature_names=list(train_frame.columns),
    )
    test = xgb.DMatrix(test_frame, feature_names=list(test_frame.columns))
    model = xgb.train(params, training, num_boost_round=iterations)
    device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    require(str(device).startswith("cuda"), f"XGBoost outer model left CUDA: {device}")
    valid_prediction = model.predict(
        valid, iteration_range=(0, iterations)
    ).astype(np.float32)
    test_prediction = model.predict(
        test, iteration_range=(0, iterations)
    ).astype(np.float32)
    model.save_model(output_path)
    del model, training, valid, test
    require(
        np.isfinite(valid_prediction).all()
        and np.isfinite(test_prediction).all(),
        "XGBoost outer prediction is non-finite",
    )
    return {
        "valid_prediction": valid_prediction,
        "test_prediction": test_prediction,
    }


def _auc_folds(
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    return [
        float(roc_auc_score(labels[valid], predictions[valid]))
        for _, valid in folds
    ]


def _save_arrays(run_dir: Path, arrays: dict[str, np.ndarray]) -> list[str]:
    paths = []
    for name, array in arrays.items():
        path = run_dir / name
        np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
        paths.append(path.name)
    return paths


def _render_results(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    phase2 = metrics["strict_phase2_matched"]
    candidate = metrics["historical_candidate"]
    blend = metrics["nested_blend"]
    lines = [
        "# EXP08 — Historical Feature Bridge Results",
        "",
        f"- Run: `{summary['experiment_id']}`",
        f"- Status: **{summary['status']} — {summary['advancement_gate']['decision']}**",
        f"- Outer seed: `{summary['outer_seed']}`",
        "- OOF AUC and public leaderboard AUC are different evaluation measures.",
        "",
        "## Matched OOF comparison",
        "",
        "| Model | Pooled OOF AUC | Delta vs Phase 2 | Five outer-fold AUCs |",
        "|---|---:|---:|---|",
        (
            f"| Strict Phase 2 | {phase2['pooled_oof_auc']:.8f} | — | "
            f"{', '.join(f'{value:.6f}' for value in phase2['fold_auc'])} |"
        ),
        (
            f"| Historical candidate | {candidate['pooled_oof_auc']:.8f} | "
            f"{candidate['pooled_delta_vs_phase2']:+.8f} | "
            f"{', '.join(f'{value:.6f}' for value in candidate['fold_auc'])} |"
        ),
        (
            f"| Nested Phase 2 blend | {blend['pooled_oof_auc']:.8f} | "
            f"{blend['pooled_delta_vs_phase2']:+.8f} | "
            f"{', '.join(f'{value:.6f}' for value in blend['fold_auc'])} |"
        ),
        "",
        "### Paired nested-blend deltas by fold",
        "",
        "| Fold | Phase 2 AUC | Blend AUC | Paired delta |",
        "|---:|---:|---:|---:|",
    ]
    for fold_number, (baseline, blended, delta) in enumerate(
        zip(phase2["fold_auc"], blend["fold_auc"], blend["fold_delta_vs_phase2"]),
        1,
    ):
        lines.append(
            f"| {fold_number} | {baseline:.6f} | {blended:.6f} | {delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Historical family evaluation",
            "",
            "| Family | Pooled OOF AUC | Fold AUCs | Positive folds vs Phase 2 |",
            "|---|---:|---|---:|",
        ]
    )
    for family, result in metrics["historical_families"].items():
        lines.append(
            f"| {family} | {result['pooled_oof_auc']:.8f} | "
            f"{', '.join(f'{value:.6f}' for value in result['fold_auc'])} | "
            f"{result['positive_folds_vs_phase2']} / 5 |"
        )
    lines.extend(
        [
            "",
            "## Controls and evidence",
            "",
            f"- Reconstructed {summary['feature_research']['reconstructed_formula_count']} eligible "
            f"features from {summary['feature_research']['legacy_formula_count']} historical formulas; "
            f"{summary['feature_research']['excluded_formula_count']} lineage-unsafe formulas excluded.",
            "- Feature family selection, feature importance cutoffs, early stopping, and blend weights "
            "were selected on inner validation only.",
            f"- CatBoost GPU preflight: `{summary['models']['catboost_gpu_preflight']}`.",
            f"- CUDA XGBoost preflight: `{summary['models']['xgboost_gpu_preflight']}`.",
            f"- GPU peak utilization: {summary['gpu']['telemetry']['maximum_utilization_percent']}%; "
            f"peak memory: {summary['gpu']['telemetry']['maximum_memory_used_mib']} MiB; "
            f"samples: {summary['gpu']['telemetry']['sample_count']}.",
            f"- Advancement gate: pooled delta {blend['pooled_delta_vs_phase2']:+.8f}; "
            f"{blend['positive_folds']} positive folds; mean paired delta "
            f"{blend['mean_fold_delta']:+.8f} (SE {blend['fold_delta_standard_error']:.8f}); "
            f"decision **{summary['advancement_gate']['decision']}**.",
            "",
            "## Previous experiment diagnostics",
            "",
            "| Reference | Pooled OOF AUC | EXP08 blend delta |",
            "|---|---:|---:|",
        ]
    )
    for name, result in metrics["previous_experiments"].items():
        if "pooled_oof_auc" in result:
            lines.append(
                f"| {name} | {result['pooled_oof_auc']:.8f} | "
                f"{result['candidate_pooled_auc_delta']:+.8f} |"
            )
        else:
            lines.append(f"| {name} | not recomputed | not comparable |")
    submission = summary["submission"]
    lines.extend(
        [
            "",
            "## Submission and reproducibility",
            "",
            f"- Submission rows/schema: {submission['rows']} / "
            f"`{', '.join(submission['columns'])}`.",
            f"- Submission SHA-256: `{submission['sha256']}`.",
            f"- Ordered test IDs validated: `{submission['ordered_ids_match_sample']}`.",
            f"- Runtime: {summary['runtime_seconds']:.1f}s total; "
            f"{summary['training_seconds']:.1f}s across fold training.",
            "- Input, source, reference, run-output, and feature-inventory hashes are recorded "
            "in `sha256_manifest.json`.",
            "- The independent verifier is `verify_exp08.py`.",
            "",
            "No official public leaderboard evaluation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _feature_schema(
    train_without_identity: pd.DataFrame,
    test_without_identity: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    full_raw = pd.concat(
        (
            train_without_identity.head(16).reset_index(drop=True),
            test_without_identity.head(8).reset_index(drop=True),
        ),
        ignore_index=True,
    )
    fit_features, _ = build_feature_frames(full_raw, full_raw.iloc[:0].copy())
    group_map = feature_groups(list(fit_features.columns))
    require(
        set(group_map) == set(EVALUATED_GROUPS),
        "Historical feature group schema differs",
    )
    sizes = {group: len(names) for group, names in group_map.items()}
    require(sizes == EXPECTED_GROUP_SIZES, f"Historical feature group sizes differ: {sizes}")
    return list(fit_features.columns), sizes


def _write_feature_inventory(
    path: Path,
    feature_names: list[str],
    raw_columns: set[str],
    baseline_cat_columns: list[str],
    baseline_lgb_columns: list[str],
    feature_distribution: dict[str, dict[str, int]],
    exact_value_duplicates: set[str],
) -> dict[str, Any]:
    group_map = feature_groups(feature_names)
    group_by_name = {
        name: group for group, names in group_map.items() for name in names
    }
    records: list[dict[str, Any]] = []
    for name in feature_names:
        sources = source_columns_for(name, raw_columns)
        records.append(
            {
                "feature_name": name,
                "family": group_by_name[name],
                "formula": formula_for(name),
                "source_columns": ";".join(sources),
                "source_columns_available": all(
                    source in raw_columns or source in feature_names
                    for source in sources
                ),
                "availability": "reconstructable from available raw inputs",
                "leakage_risk": (
                    "low; unsupervised, fold-local"
                    if not name.startswith("CLAYTON_COPULA_")
                    else "low after fit-only empirical ranks; no labels used"
                ),
                "phase2_cat_exact_name_overlap": name in baseline_cat_columns,
                "phase2_lgb_exact_name_overlap": name in baseline_lgb_columns,
                "phase2_lgb_exact_value_duplicate": name in exact_value_duplicates,
                "missing_values": feature_distribution[name]["missing"],
                "infinite_values": feature_distribution[name]["infinite"],
                "evaluation_status": (
                    "excluded_from_models_historical_infinity_risk"
                    if name == HOUSING_INFINITY_FEATURE
                    else "eligible_for_nested_evaluation"
                ),
                "rationale": (
                    "Original executable feature retained for auditable family testing."
                ),
            }
        )
    excluded_groups = {
        name: feature_group
        for feature_group, names in (
            ("G_ENTITY_TIMELINE", (
                "N_APPS_ALL",
                "APP_ORDER",
                "DAYS_SINCE_PREV_APP",
                "DAYS_TO_NEXT_APP",
            )),
            ("UIDY_SUPERVISED_MEMORY", (
                "UIDY_N",
                "UIDY_SUM",
                "UIDY_NEAR_Y",
                "UIDY_NEAR_D",
                "UIDY_ANYBAD",
            )),
        )
        for name in names
    }
    for name, family in excluded_groups.items():
        formula = EXCLUDED_FEATURE_FORMULAS[name]
        sources = EXCLUDED_FEATURE_SOURCES[name]
        records.append(
            {
                "feature_name": name,
                "family": family,
                "formula": formula,
                "source_columns": ";".join(sources),
                "source_columns_available": all(
                    source in raw_columns for source in sources
                ),
                "availability": (
                    "raw columns available; historical pooled or non-nested label "
                    "construction is not accepted"
                ),
                "leakage_risk": EXCLUDED_GROUPS[family],
                "phase2_cat_exact_name_overlap": name in baseline_cat_columns,
                "phase2_lgb_exact_name_overlap": name in baseline_lgb_columns,
                "phase2_lgb_exact_value_duplicate": False,
                "missing_values": "",
                "infinite_values": "",
                "evaluation_status": "excluded_by_predeclared_lineage_gate",
                "rationale": "Avoids pooled-cohort signature reuse or non-nested label memory.",
            }
        )
    inventory = pd.DataFrame.from_records(records)
    require(len(inventory) == 669, f"Feature inventory has {len(inventory)} rows")
    inventory.to_csv(path, index=False, lineterminator="\n")
    return {
        "inventory_path": str(path.resolve()),
        "rows": int(len(inventory)),
        "reconstructed_features": int(len(feature_names)),
        "excluded_features": int(len(excluded_groups)),
        "phase2_exact_name_overlap_count": int(
            inventory["phase2_cat_exact_name_overlap"].sum()
            + inventory["phase2_lgb_exact_name_overlap"].sum()
        ),
    }


def _load_reference_artifacts(
    helper: Any,
    runtime: Any,
    data_dir: Path,
    input_records: dict[str, Any],
    outer_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Path, dict[str, Any]]:
    exp06_manifest_path = EXP06_RUN_DIR / "sha256_manifest.json"
    exp06_manifest = json.loads(exp06_manifest_path.read_text(encoding="utf-8"))
    verify_record(
        exp06_manifest["outputs"]["experiment_summary.json"],
        "EXP06 experiment summary",
    )
    exp06_summary = json.loads(
        (EXP06_RUN_DIR / "experiment_summary.json").read_text(encoding="utf-8")
    )
    require(
        exp06_summary["status"] == "COMPLETED"
        and exp06_summary["model"]["gpu_training_confirmed"] is True,
        "Strict EXP06/Phase 2 reference is incomplete",
    )
    for name, record in input_records.items():
        require(
            exp06_manifest["inputs"][name]["sha256"] == record["sha256"],
            f"EXP06 raw {name} hash differs from current input",
        )
    for source_name, path in (
        ("phase2_raw_feature_builder", PHASE2_SOURCE),
        ("phase2_fold_local_pipeline", PHASE4_SOURCE),
    ):
        expected = exp06_manifest["sources"][source_name]
        require(
            sha256_file(path) == expected["sha256"],
            f"Frozen Phase 2 source changed: {source_name}",
        )
    for name in ("exp04_oof", "exp05_oof"):
        verify_record(exp06_manifest["inputs"][name], name)
    if outer_seed == OUTER_SEED:
        reference_dir = (
            FROZEN_DIR / "EVIDENCE" / "second_seed" / "main"
        )
        reference_oof_record = exp06_manifest["outputs"][
            "oof_phase2_strict_matched.npy"
        ]
        reference_test_record = exp06_manifest["outputs"][
            "test_phase2_strict_matched.npy"
        ]
        oof_path = verify_record(reference_oof_record, "Strict Phase 2 OOF")
        test_path = verify_record(reference_test_record, "Strict Phase 2 test")
        references = {
            "phase2_oof": np.load(oof_path, allow_pickle=False).astype(np.float32),
            "phase2_test": np.load(test_path, allow_pickle=False).astype(np.float32),
            "exp04_oof": np.load(
                verify_record(exp06_manifest["inputs"]["exp04_oof"], "EXP04 OOF"),
                allow_pickle=False,
            ).astype(np.float32),
            "exp05_oof": np.load(
                verify_record(exp06_manifest["inputs"]["exp05_oof"], "EXP05 OOF"),
                allow_pickle=False,
            ).astype(np.float32),
            "exp06_oof": np.load(
                verify_record(
                    exp06_manifest["outputs"]["oof_ft_transformer.npy"],
                    "EXP06 OOF",
                ),
                allow_pickle=False,
            ).astype(np.float32),
        }
    else:
        reference_dir = (
            FROZEN_DIR / "EVIDENCE" / "original_runs" / "phase_02"
        )
        references = {}
    return exp06_manifest, references, reference_dir, exp06_summary


def run_experiment(
    data_dir: Path,
    run_dir: Path,
    outer_seed: int = OUTER_SEED,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    require(
        outer_seed in (OUTER_SEED, 42),
        "Only seed 20261101 and frozen seed-42 partitions are supported",
    )
    require(
        not run_dir.exists() or not any(run_dir.iterdir()),
        f"Run directory must be new or empty: {run_dir}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.perf_counter()
    try:
        raw_paths = {
            "train": data_dir / "train.csv",
            "test": data_dir / "test.csv",
            "sample": data_dir / "sample_submission.csv",
        }
        input_records: dict[str, Any] = {}
        for name, path in raw_paths.items():
            require(path.is_file(), f"Input {name} CSV is missing: {path}")
            record = file_record(path)
            require(
                record["sha256"] == EXPECTED_INPUT_SHA256[name],
                f"Input {name} SHA-256 mismatch",
            )
            input_records[name] = record

        gpu_before_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        gpu_lines = gpu_before_result.stdout.strip().splitlines()
        require(len(gpu_lines) == 1, "EXP08 expects one visible CUDA device")
        gpu_name, gpu_driver, gpu_total, gpu_free, gpu_util = [
            field.strip() for field in gpu_lines[0].split(",")
        ]
        require("RTX 4050" in gpu_name, f"Unexpected GPU: {gpu_name}")
        require(int(gpu_free) >= 1000, f"Insufficient free GPU memory: {gpu_free}")
        require(int(gpu_util) < 50, f"GPU is already busy: {gpu_util}%")
        gpu_preflight = {
            "name": gpu_name,
            "driver_version": gpu_driver,
            "total_memory_mib": int(gpu_total),
            "free_memory_mib_before_run": int(gpu_free),
            "utilization_percent_before_run": int(gpu_util),
        }

        helper = load_module("exp08_exp07_helpers", EXP07_RUNNER)
        runtime = helper.load_exp06_runtime()
        catboost_preflight = catboost_gpu_preflight()
        xgb_preflight = xgboost_gpu_preflight()
        xgb = None
        if xgb_preflight["available"]:
            import xgboost as xgb

        source_paths = {
            "exp08_runner": Path(__file__).resolve(),
            "historical_feature_reconstruction": EXPERIMENT_DIR
            / "historical_features.py",
            "exp08_independent_verifier": EXPERIMENT_DIR / "verify_exp08.py",
            "exp08_feature_tests": EXPERIMENT_DIR
            / "tests"
            / "test_exp08_historical_features.py",
            "exp08_readme": EXPERIMENT_DIR / "README.md",
            "exp08_parity_report": EXPERIMENT_DIR / "FEATURE_PARITY.md",
            "historical_executable": HISTORICAL_SOURCE,
            "historical_mathematical_specification": HISTORICAL_SPEC,
            "historical_matrix_parity_audit": HISTORICAL_PARITY_AUDIT,
            "historical_feature_lineage": HISTORICAL_LINEAGE,
            "historical_architecture": HISTORICAL_ARCHITECTURE,
            "exp07_protocol_helpers": EXP07_RUNNER,
            "exp06_protocol_runner": EXP06_RUNNER,
            "phase2_raw_feature_builder": PHASE2_SOURCE,
            "phase2_fold_local_pipeline": PHASE4_SOURCE,
        }
        for name, path in source_paths.items():
            require(path.is_file(), f"Source {name} is missing: {path}")
        if EXTERNAL_HISTORICAL_SOURCE.is_file():
            require(
                sha256_file(EXTERNAL_HISTORICAL_SOURCE)
                == sha256_file(HISTORICAL_SOURCE),
                "External historical executable differs from repository copy",
            )
        source_records = {
            name: file_record(path) for name, path in source_paths.items()
        }

        exp06_manifest, references, reference_dir, exp06_summary = (
            _load_reference_artifacts(
                helper, runtime, data_dir, input_records, outer_seed
            )
        )
        phase2_manifest_path = reference_dir / "sha256_manifest.json"
        if outer_seed == OUTER_SEED:
            phase2_manifest_reference = verify_record(
                exp06_manifest["inputs"]["phase2_manifest"],
                "Frozen Phase 2 manifest",
            )
            require(
                phase2_manifest_reference.resolve()
                == phase2_manifest_path.resolve(),
                "EXP06 points to a different frozen Phase 2 manifest",
            )
        phase2_manifest = json.loads(
            phase2_manifest_path.read_text(encoding="utf-8")
        )
        reference_records = {
            "exp06_manifest": file_record(EXP06_RUN_DIR / "sha256_manifest.json"),
            "exp06_summary": file_record(EXP06_RUN_DIR / "experiment_summary.json"),
            "phase2_manifest": file_record(phase2_manifest_path),
        }
        for key in ("exp04_oof", "exp05_oof", "oof_ft_transformer.npy"):
            record = (
                exp06_manifest["inputs"][key]
                if key in exp06_manifest["inputs"]
                else exp06_manifest["outputs"][key]
            )
            path = verify_record(record, key)
            reference_records[key] = file_record(path)
        if outer_seed == OUTER_SEED:
            for key in ("oof_phase2_strict_matched.npy", "test_phase2_strict_matched.npy"):
                path = verify_record(exp06_manifest["outputs"][key], key)
                reference_records[key] = file_record(path)
        train_raw = pd.read_csv(raw_paths["train"])
        test_raw = pd.read_csv(raw_paths["test"])
        sample = pd.read_csv(raw_paths["sample"])
        require(
            sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
            and np.array_equal(
                sample[ID_COLUMN].to_numpy(),
                test_raw[ID_COLUMN].to_numpy(),
            ),
            "Test and sample-submission ID schema/order differs",
        )
        require(
            train_raw[ID_COLUMN].is_unique
            and test_raw[ID_COLUMN].is_unique
            and set(train_raw[ID_COLUMN]).isdisjoint(set(test_raw[ID_COLUMN])),
            "Train/test IDs are duplicated or overlap",
        )
        labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
        require(
            len(np.unique(labels)) == 2 and np.isin(labels, [0, 1]).all(),
            "TARGET is not a binary two-class vector",
        )
        if outer_seed == OUTER_SEED:
            require(
                len(references["phase2_oof"]) == len(labels)
                and len(references["phase2_test"]) == len(test_raw),
                "Matched Phase 2 predictions do not match raw row counts",
            )

        folds = list(
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=outer_seed,
            ).split(np.zeros(len(labels)), labels)
        )
        runtime.validate_fold_partition(folds, len(labels))
        runtime.INNER_SEED_BASE = outer_seed
        reference_fold_evidence = (
            runtime.verify_reference_folds(folds)
            if outer_seed == OUTER_SEED
            else helper.verify_seed_fold_reference(folds, reference_dir, phase2_manifest)
        )
        raw_train_features = runtime.build_rowwise_features(
            train_raw.drop(columns=[TARGET_COLUMN])
        )
        raw_test_features = runtime.build_rowwise_features(test_raw)
        require(
            raw_train_features.columns.tolist() == raw_test_features.columns.tolist(),
            "Phase 2 raw feature schemas differ",
        )
        train_without_identity = _raw_without_identity(train_raw, has_target=True)
        test_without_identity = _raw_without_identity(test_raw, has_target=False)
        feature_names, group_sizes = _feature_schema(
            train_without_identity, test_without_identity
        )
        require(len(feature_names) == 660, "Reconstructed feature count differs")
        feature_distribution: dict[str, dict[str, int]] = {}

        reference_oof = references.get("phase2_oof")
        reference_test = references.get("phase2_test")
        baseline_oof = (
            reference_oof.copy()
            if reference_oof is not None
            else np.full(len(labels), np.nan, dtype=np.float32)
        )
        candidate_oof = np.full(len(labels), np.nan, dtype=np.float32)
        blend_oof = np.full(len(labels), np.nan, dtype=np.float32)
        baseline_test_folds: list[np.ndarray] = []
        candidate_test_folds: list[np.ndarray] = []
        fold_records: list[dict[str, Any]] = []
        family_oof = {
            group: np.full(len(labels), np.nan, dtype=np.float32)
            for group in EVALUATED_GROUPS
        }
        family_test_folds = {group: [] for group in EVALUATED_GROUPS}
        importance_records: list[dict[str, Any]] = []
        exact_overlap_by_feature: dict[str, dict[str, bool]] = {}
        family_models_config = {
            "individual_family_screen": (
                "LightGBM CPU; Phase 2 inner settings capped at 600 rounds and "
                "50-round early stopping; outer rounds fixed by inner validation"
            ),
            "combined_candidate": (
                "Phase 2 LightGBM CPU + CatBoost GPU; CUDA XGBoost added only "
                "when an explicit CUDA preflight succeeds"
            ),
        }

        log_message(
            log_path,
            f"Starting EXP08 on {gpu_name}; outer seed={outer_seed}; "
            f"XGBoost CUDA supported={xgb_preflight['available']}",
        )
        fold_started_all = time.perf_counter()
        with runtime.GpuTelemetryMonitor(interval_seconds=1.0) as gpu_monitor:
            for fold_number, (outer_fit, outer_valid) in enumerate(folds, 1):
                fold_started = time.perf_counter()
                inner_fit, inner_valid = runtime.build_inner_split(
                    labels,
                    outer_fit,
                    outer_seed + fold_number,
                )
                inner_frames = build_phase2_frames(
                    runtime,
                    train_raw,
                    test_raw,
                    raw_train_features,
                    raw_test_features,
                    inner_fit,
                    inner_valid,
                )
                inner_cat, inner_cat_test, inner_lgb, inner_lgb_test, inner_audit = (
                    inner_frames
                )
                outer_frames = build_phase2_frames(
                    runtime,
                    train_raw,
                    test_raw,
                    raw_train_features,
                    raw_test_features,
                    outer_fit,
                    outer_valid,
                )
                outer_cat, outer_cat_test, outer_lgb, outer_lgb_test, outer_audit = (
                    outer_frames
                )
                require(
                    inner_audit["target_encoding"][
                        "validation_encoded_from_outer_fit_only"
                    ]
                    and outer_audit["target_encoding"][
                        "validation_encoded_from_outer_fit_only"
                    ],
                    "Phase 2 fold-local target-encoding audit failed",
                )
                base_inner = helper.fit_inner_models(
                    runtime,
                    inner_cat,
                    inner_lgb,
                    labels,
                    inner_fit,
                    inner_valid,
                    fold_number,
                    f"fold {fold_number} Phase 2",
                )
                base_inner_auc = float(
                    roc_auc_score(labels[inner_valid], base_inner["lgb_prediction"])
                )
                inner_blocks = historical_blocks(
                    train_without_identity,
                    test_without_identity,
                    inner_fit,
                    inner_valid,
                )
                outer_blocks = historical_blocks(
                    train_without_identity,
                    test_without_identity,
                    outer_fit,
                    outer_valid,
                )
                if not feature_distribution:
                    feature_distribution = {
                        name: record
                        for group in EVALUATED_GROUPS
                        for name, record in outer_blocks[group][
                            "distribution"
                        ].items()
                    }
                    require(
                        set(feature_distribution) == set(feature_names),
                        "Feature distribution audit missed historical features",
                    )

                family_inner: dict[str, dict[str, Any]] = {}
                family_outer: dict[str, dict[str, Any]] = {}
                for group in EVALUATED_GROUPS:
                    inner_block = inner_blocks[group]
                    inner_valid_hist = inner_block["apply"].iloc[
                        : len(inner_valid)
                    ].reset_index(drop=True)
                    inner_fit_hist = inner_block["fit"].reset_index(drop=True)
                    inner_result = fit_family_inner(
                        runtime,
                        inner_lgb,
                        inner_lgb.iloc[inner_valid],
                        inner_fit_hist,
                        inner_valid_hist,
                        labels,
                        inner_fit,
                        inner_valid,
                        fold_number,
                    )
                    family_inner[group] = inner_result

                    outer_block = outer_blocks[group]
                    outer_result = fit_family_outer(
                        runtime,
                        outer_lgb,
                        outer_lgb_test,
                        outer_block["fit"],
                        outer_block["apply"],
                        labels,
                        outer_fit,
                        outer_valid,
                        fold_number,
                        inner_result["best_iteration"],
                        list(inner_result["aliases"].values()),
                    )
                    family_outer[group] = outer_result
                    family_oof[group][outer_valid] = outer_result[
                        "valid_prediction"
                    ]
                    family_test_folds[group].append(
                        outer_result["test_prediction"]
                    )
                    exact_overlap_by_feature.update(
                        {
                            name: {
                                "exact_duplicate_in_phase2_lgb": True,
                                "exact_duplicate_in_phase2_catboost": False,
                            }
                            for name in inner_result["exact_duplicates"]
                        }
                    )
                    inner_delta = inner_result["auc"] - base_inner_auc
                    selected = selected_features(group, inner_result)
                    family_inner[group]["selected_for_combined"] = selected
                    importance_records.extend(
                        {
                            "fold": fold_number,
                            "family": group,
                            "feature": original,
                            "inner_gain": float(
                                inner_result["importance"].get(alias, 0.0)
                            ),
                            "inner_auc_delta_vs_phase2_lgb": float(inner_delta),
                            "selected_for_combined": original in selected,
                        }
                        for alias, original in inner_result["aliases"].items()
                    )
                    log_message(
                        log_path,
                        f"Fold {fold_number} {group}: inner AUC={inner_result['auc']:.6f}, "
                        f"delta={inner_delta:+.6f}, added={len(inner_result['aliases'])}, "
                        f"combined candidates={len(selected)}",
                    )

                positive_groups = [
                    group
                    for group in EVALUATED_GROUPS
                    if family_inner[group]["auc"] > base_inner_auc
                    and family_inner[group]["selected_for_combined"]
                ]
                positive_groups.sort(
                    key=lambda group: (
                        family_inner[group]["auc"] - base_inner_auc,
                        group,
                    ),
                    reverse=True,
                )
                selected_groups = positive_groups[:MAX_SELECTED_GROUPS]
                selected_names = [
                    name
                    for group in EVALUATED_GROUPS
                    if group in selected_groups
                    for name in family_inner[group]["selected_for_combined"]
                ]
                selected_names = list(dict.fromkeys(selected_names))
                combined_inner = add_historical_blocks(
                    inner_cat,
                    inner_cat_test,
                    inner_lgb,
                    inner_lgb_test,
                    inner_blocks,
                    selected_names,
                    inner_fit,
                    inner_valid,
                    len(test_raw),
                )
                candidate_inner = helper.fit_inner_models(
                    runtime,
                    combined_inner[0],
                    combined_inner[2],
                    labels,
                    inner_fit,
                    inner_valid,
                    fold_number,
                    f"fold {fold_number} historical combined",
                )
                candidate_inner_components = [
                    candidate_inner["blend_prediction"].astype(np.float64)
                ]
                xgb_inner = None
                xgb_rounds = None
                if xgb is not None:
                    xgb_inner_result = fit_xgb_inner(
                        xgb,
                        combined_inner[2],
                        combined_inner[2].iloc[inner_valid],
                        labels,
                        inner_fit,
                        inner_valid,
                        outer_seed + 30_000 + fold_number,
                    )
                    xgb_inner = xgb_inner_result["prediction"]
                    xgb_rounds = xgb_inner_result["best_iteration"]
                    candidate_inner_components.append(xgb_inner.astype(np.float64))
                candidate_inner_prediction = np.mean(
                    candidate_inner_components, axis=0
                ).astype(np.float32)
                blend_selection = helper.select_blend_weight(
                    labels[inner_valid],
                    base_inner["blend_prediction"],
                    candidate_inner_prediction,
                )
                if reference_oof is None:
                    baseline_outer = helper.fit_outer_variant(
                        runtime,
                        outer_cat,
                        outer_cat_test,
                        outer_lgb,
                        outer_lgb_test,
                        labels,
                        outer_fit,
                        outer_valid,
                        fold_number,
                        base_inner["lgb_best_iteration"],
                        base_inner["cat_best_iteration"],
                        "phase2_reference",
                        run_dir,
                    )
                    baseline_valid = (
                        baseline_outer["lgb_valid"].astype(np.float64)
                        + baseline_outer["cat_valid"].astype(np.float64)
                    ).astype(np.float32) / 2.0
                    baseline_test_fold = (
                        baseline_outer["lgb_test"].astype(np.float64)
                        + baseline_outer["cat_test"].astype(np.float64)
                    ).astype(np.float32) / 2.0
                    baseline_oof[outer_valid] = baseline_valid
                    baseline_test_folds.append(baseline_test_fold)
                else:
                    baseline_valid = reference_oof[outer_valid]

                outer_combined = add_historical_blocks(
                    outer_cat,
                    outer_cat_test,
                    outer_lgb,
                    outer_lgb_test,
                    outer_blocks,
                    selected_names,
                    outer_fit,
                    outer_valid,
                    len(test_raw),
                )
                candidate_outer = helper.fit_outer_variant(
                    runtime,
                    outer_combined[0],
                    outer_combined[1],
                    outer_combined[2],
                    outer_combined[3],
                    labels,
                    outer_fit,
                    outer_valid,
                    fold_number,
                    candidate_inner["lgb_best_iteration"],
                    candidate_inner["cat_best_iteration"],
                    "historical_combined",
                    run_dir,
                )
                candidate_outer_components = [
                    (
                        candidate_outer["lgb_valid"].astype(np.float64)
                        + candidate_outer["cat_valid"].astype(np.float64)
                    )
                    / 2.0
                ]
                candidate_test_components = [
                    (
                        candidate_outer["lgb_test"].astype(np.float64)
                        + candidate_outer["cat_test"].astype(np.float64)
                    )
                    / 2.0
                ]
                xgb_outer = None
                if xgb is not None:
                    require(xgb_rounds is not None, "CUDA XGBoost best iteration is missing")
                    xgb_outer = fit_xgb_outer(
                        xgb,
                        outer_combined[2],
                        outer_combined[3],
                        labels,
                        outer_fit,
                        outer_valid,
                        xgb_rounds,
                        outer_seed + 40_000 + fold_number,
                        run_dir / f"fold_{fold_number}_historical_xgb.json",
                    )
                    candidate_outer_components.append(
                        xgb_outer["valid_prediction"].astype(np.float64)
                    )
                    candidate_test_components.append(
                        xgb_outer["test_prediction"].astype(np.float64)
                    )
                candidate_valid = np.mean(
                    candidate_outer_components, axis=0
                ).astype(np.float32)
                candidate_test = np.mean(
                    candidate_test_components, axis=0
                ).astype(np.float32)
                candidate_oof[outer_valid] = candidate_valid
                weight = blend_selection["candidate_weight"]
                blend_oof[outer_valid] = (
                    (1.0 - weight) * baseline_valid.astype(np.float64)
                    + weight * candidate_valid.astype(np.float64)
                ).astype(np.float32)
                candidate_test_folds.append(candidate_test)
                fold_prediction_values = {
                    "phase2_valid": baseline_valid,
                    "historical_lgb_cat_valid": candidate_outer_components[0],
                    "historical_valid": candidate_valid,
                    "nested_blend_valid": blend_oof[outer_valid],
                    "historical_test": candidate_test,
                }
                if xgb_outer is not None:
                    fold_prediction_values["historical_xgb_valid"] = xgb_outer[
                        "valid_prediction"
                    ]
                    fold_prediction_values["historical_xgb_test"] = xgb_outer[
                        "test_prediction"
                    ]
                if reference_oof is None:
                    fold_prediction_values["phase2_test"] = baseline_test_fold
                np.savez_compressed(
                    run_dir / f"fold_{fold_number}_predictions.npz",
                    **fold_prediction_values,
                )
                family_results = {}
                for group in EVALUATED_GROUPS:
                    pred = family_outer[group]["valid_prediction"]
                    family_results[group] = {
                        "outer_auc": float(roc_auc_score(labels[outer_valid], pred)),
                        "inner_auc": float(family_inner[group]["auc"]),
                        "inner_delta_vs_phase2_lgb": float(
                            family_inner[group]["auc"] - base_inner_auc
                        ),
                        "best_iteration": int(family_inner[group]["best_iteration"]),
                        "selected_for_combined": group in selected_groups,
                        "selected_features": family_inner[group][
                            "selected_for_combined"
                        ],
                        "exact_duplicate_features": family_inner[group][
                            "exact_duplicates"
                        ],
                    }
                fold_records.append(
                    {
                        "fold": fold_number,
                        "outer_fit_rows": int(len(outer_fit)),
                        "outer_validation_rows": int(len(outer_valid)),
                        "inner_fit_rows": int(len(inner_fit)),
                        "inner_validation_rows": int(len(inner_valid)),
                        "phase2_feature_count": int(len(outer_lgb.columns)),
                        "inner_phase2_auc": float(
                            roc_auc_score(
                                labels[inner_valid],
                                base_inner["blend_prediction"],
                            )
                        ),
                        "selected_groups": selected_groups,
                        "selected_feature_count": len(selected_names),
                        "inner_blend_selection": blend_selection,
                        "candidate_lgb_best_iteration": int(
                            candidate_inner["lgb_best_iteration"]
                        ),
                        "candidate_catboost_best_iteration": int(
                            candidate_inner["cat_best_iteration"]
                        ),
                        "candidate_xgboost_best_iteration": xgb_rounds,
                        "candidate_xgboost_included": xgb is not None,
                        "phase2_outer_auc": float(
                            roc_auc_score(labels[outer_valid], baseline_valid)
                        ),
                        "historical_candidate_outer_auc": float(
                            roc_auc_score(labels[outer_valid], candidate_valid)
                        ),
                        "historical_blend_outer_auc": float(
                            roc_auc_score(labels[outer_valid], blend_oof[outer_valid])
                        ),
                        "folded_families": family_results,
                        "phase2_inner_audit": inner_audit,
                        "phase2_outer_audit": outer_audit,
                        "elapsed_seconds": time.perf_counter() - fold_started,
                    }
                )
                log_message(
                    log_path,
                    f"Fold {fold_number} complete: groups={selected_groups}, "
                    f"selected_features={len(selected_names)}, "
                    f"candidate_outer_auc="
                    f"{fold_records[-1]['historical_candidate_outer_auc']:.6f}, "
                    f"blend_weight={weight:.2f}",
                )
                del inner_blocks, outer_blocks, inner_frames, outer_frames
                del combined_inner, outer_combined

        require(np.isfinite(baseline_oof).all(), "Phase 2 baseline OOF is incomplete")
        require(np.isfinite(candidate_oof).all(), "EXP08 candidate OOF is incomplete")
        require(np.isfinite(blend_oof).all(), "EXP08 nested blend OOF is incomplete")
        baseline_auc = float(roc_auc_score(labels, baseline_oof))
        candidate_auc = float(roc_auc_score(labels, candidate_oof))
        blend_auc = float(roc_auc_score(labels, blend_oof))
        baseline_fold_auc = _auc_folds(labels, baseline_oof, folds)
        candidate_fold_auc = _auc_folds(labels, candidate_oof, folds)
        blend_fold_auc = _auc_folds(labels, blend_oof, folds)
        fold_deltas = [
            candidate_score - baseline_score
            for candidate_score, baseline_score in zip(
                blend_fold_auc, baseline_fold_auc
            )
        ]
        mean_delta = float(np.mean(fold_deltas))
        delta_se = float(np.std(fold_deltas, ddof=1) / math.sqrt(len(fold_deltas)))
        positive_folds = int(sum(delta > 0.0 for delta in fold_deltas))
        gate_passed = (
            blend_auc - baseline_auc
            >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
            and positive_folds >= ADVANCEMENT_GATE["minimum_positive_folds"]
            and mean_delta
            >= ADVANCEMENT_GATE["minimum_mean_delta_standard_errors"] * delta_se
        )
        decision = "ELIGIBLE_FOR_INDEPENDENT_REPLICATION" if gate_passed else "REJECTED"

        if reference_test is None:
            baseline_test = np.mean(baseline_test_folds, axis=0).astype(np.float32)
        else:
            baseline_test = reference_test
        candidate_test = np.mean(candidate_test_folds, axis=0).astype(np.float32)
        selected_weights = np.asarray(
            [row["inner_blend_selection"]["candidate_weight"] for row in fold_records],
            dtype=np.float64,
        )
        mean_weight = float(selected_weights.mean())
        blended_test = (
            (1.0 - mean_weight) * baseline_test.astype(np.float64)
            + mean_weight * candidate_test.astype(np.float64)
        ).astype(np.float32)
        require(
            len(blended_test) == len(sample)
            and np.isfinite(blended_test).all()
            and np.all((blended_test >= 0.0) & (blended_test <= 1.0)),
            "EXP08 test probabilities are invalid",
        )
        submission = pd.DataFrame(
            {
                ID_COLUMN: test_raw[ID_COLUMN].to_numpy(),
                TARGET_COLUMN: blended_test,
            }
        )
        require(
            submission.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
            and np.array_equal(
                submission[ID_COLUMN].to_numpy(),
                sample[ID_COLUMN].to_numpy(),
            )
            and submission[ID_COLUMN].is_unique,
            "EXP08 submission schema or ordered IDs differ",
        )
        submission_path = run_dir / "submission_exp08_nested_blend.csv"
        submission.to_csv(submission_path, index=False, lineterminator="\n")

        family_summary: dict[str, Any] = {}
        for group in EVALUATED_GROUPS:
            family_predictions = family_oof[group]
            require(
                np.isfinite(family_predictions).all(),
                f"Family OOF predictions are incomplete: {group}",
            )
            family_auc = float(roc_auc_score(labels, family_predictions))
            family_fold_auc = _auc_folds(labels, family_predictions, folds)
            family_summary[group] = {
                "pooled_oof_auc": family_auc,
                "fold_auc": family_fold_auc,
                "fold_delta_vs_phase2": [
                    value - baseline
                    for value, baseline in zip(family_fold_auc, baseline_fold_auc)
                ],
                "positive_folds_vs_phase2": int(
                    sum(
                        value > baseline
                        for value, baseline in zip(
                            family_fold_auc, baseline_fold_auc
                        )
                    )
                ),
                "outer_models": "LightGBM CPU; iteration counts selected on inner validation",
            }
            _save_arrays(
                run_dir,
                {
                    f"oof_{group.lower()}.npy": family_predictions,
                    f"test_{group.lower()}.npy": np.mean(
                        family_test_folds[group], axis=0
                    ),
                },
            )

        if outer_seed == OUTER_SEED:
            prior_comparisons = {}
            for key, record_name in (
                ("EXP04_CatBoost_Lossguide", "exp04_oof"),
                ("EXP05_Residual_MLP", "exp05_oof"),
                ("EXP06_FT_Transformer", "exp06_oof"),
            ):
                prediction = references[record_name]
                prior_comparisons[key] = {
                    "pooled_oof_auc": float(roc_auc_score(labels, prediction)),
                    "fold_auc": _auc_folds(labels, prediction, folds),
                    "candidate_pooled_auc_delta": float(
                        blend_auc - roc_auc_score(labels, prediction)
                    ),
                    "candidate_fold_delta": [
                        candidate_score - previous_score
                        for candidate_score, previous_score in zip(
                            blend_fold_auc,
                            _auc_folds(labels, prediction, folds),
                        )
                    ],
                    "comparison": "matched outer folds; diagnostic only",
                }
        else:
            prior_comparisons = {
                "EXP04_EXP05_EXP06": {
                    "comparison": "not recomputed on seed-42 folds; see matched seed-20261101 diagnostics",
                }
            }

        _save_arrays(
            run_dir,
            {
                "oof_phase2_matched.npy": baseline_oof,
                "oof_historical_candidate.npy": candidate_oof,
                "oof_nested_blend.npy": blend_oof,
                "test_phase2_matched.npy": baseline_test,
                "test_historical_candidate.npy": candidate_test,
                "test_nested_blend.npy": blended_test,
            },
        )
        fold_index_files = []
        for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
            path = run_dir / f"fold_{fold_number}_indices.npz"
            np.savez_compressed(
                path,
                fit_indices=fit_indices,
                valid_indices=valid_indices,
            )
            fold_index_files.append(path.name)

        feature_inventory = _write_feature_inventory(
            EXPERIMENT_DIR / "FEATURE_INVENTORY.csv",
            feature_names,
            set(train_without_identity.columns),
            list(outer_cat.columns),
            list(outer_lgb.columns),
            feature_distribution,
            set(exact_overlap_by_feature),
        )
        feature_inventory["sha256"] = sha256_file(
            EXPERIMENT_DIR / "FEATURE_INVENTORY.csv"
        )
        pd.DataFrame(
            gpu_monitor.samples,
            columns=(
                "timestamp",
                "index",
                "name",
                "memory_used_mib",
                "utilization_percent",
            ),
        ).to_csv(
            run_dir / "gpu_telemetry.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(importance_records).to_csv(
            run_dir / "inner_feature_importance.csv",
            index=False,
            lineterminator="\n",
        )
        family_correlations = {
            left: {
                right: float(
                    np.corrcoef(family_oof[left], family_oof[right])[0, 1]
                )
                for right in EVALUATED_GROUPS
            }
            for left in EVALUATED_GROUPS
        }
        family_complementarity = {
            group: {
                "pearson_vs_phase2": float(
                    np.corrcoef(family_oof[group], baseline_oof)[0, 1]
                ),
                "spearman_vs_phase2": float(
                    runtime.spearmanr(family_oof[group], baseline_oof).statistic
                ),
                "residual_correlation_vs_phase2": float(
                    np.corrcoef(
                        family_oof[group] - labels,
                        baseline_oof - labels,
                    )[0, 1]
                ),
                "pooled_auc_delta_vs_phase2": float(
                    family_summary[group]["pooled_oof_auc"] - baseline_auc
                ),
            }
            for group in EVALUATED_GROUPS
        }
        write_json(
            run_dir / "prediction_correlations.json",
            {
                "family_pearson_correlation_matrix": family_correlations,
                "family_complementarity_vs_phase2": family_complementarity,
                "candidate_vs_phase2_pearson": float(
                    np.corrcoef(candidate_oof, baseline_oof)[0, 1]
                ),
                "candidate_error_correlation_vs_phase2": float(
                    np.corrcoef(
                        candidate_oof - labels,
                        baseline_oof - labels,
                    )[0, 1]
                ),
            },
        )
        fold_metrics = pd.DataFrame(
            {
                "fold": np.arange(1, 6),
                "phase2_auc": baseline_fold_auc,
                "historical_candidate_auc": candidate_fold_auc,
                "nested_blend_auc": blend_fold_auc,
                "nested_blend_delta_vs_phase2": fold_deltas,
            }
        )
        fold_metrics.to_csv(
            run_dir / "fold_metrics.csv",
            index=False,
            lineterminator="\n",
        )
        summary = {
            "experiment_id": run_dir.name,
            "status": "COMPLETED",
            "outer_seed": outer_seed,
            "feature_research": {
                "historical_executable": str(HISTORICAL_SOURCE.resolve()),
                "legacy_formula_count": 669,
                "reconstructed_formula_count": len(feature_names),
                "excluded_formula_count": 9,
                "excluded_features": EXCLUDED_FEATURE_FORMULAS,
                "family_sizes": group_sizes,
                "formula_parity": (
                    "660/660 reconstructed features matched the historical executable "
                    "on a bounded synthetic fixture; the parity test runs from memory "
                    "and writes no historical cache files"
                ),
                "full_raw_missing_and_infinity_counts": feature_distribution,
                "inventory": feature_inventory,
                "historical_specification_conflict": (
                    "The prose specification describes bivariate Clayton and a "
                    "region-volatility Merton model; the executable implements a "
                    "trivariate EXT_SOURCE Clayton transform and fixed-parameter "
                    "disposable-income Merton-like transforms. Executable formulas "
                    "were followed."
                ),
                "full_historical_matrix_hash_parity": (
                    "Not independently reproduced: historical cached matrices were unavailable."
                ),
                "exact_phase2_name_overlaps": exact_overlap_by_feature,
            },
            "fold_protocol": {
                "outer_folds": 5,
                "outer_seed": outer_seed,
                "fold_reference": reference_fold_evidence,
                "inner_split": (
                    "EXP06 StratifiedShuffleSplit on outer-fit rows; seed=outer seed + fold"
                ),
                "outer_validation_used_for_selection": False,
                "inner_only_early_stopping": True,
                "inner_only_group_and_feature_selection": True,
                "inner_only_blend_weight_selection": True,
                "phase2_target_encoding_audited": True,
                "historical_clayton_ranks": (
                    "Empirical average ranks fitted on model-fit rows only and applied "
                    "to inner/outer validation and test rows."
                ),
                "infinite_feature_policy": (
                    f"Preserve historical values for audit, record counts, and exclude "
                    f"{HOUSING_INFINITY_FEATURE} before model input; fail on any other "
                    "infinite generated feature. No infinity-to-NaN conversion."
                ),
                "selected_groups_per_fold": {
                    str(row["fold"]): row["selected_groups"] for row in fold_records
                },
            },
            "models": {
                "family_screen_and_outer_family_models": family_models_config[
                    "individual_family_screen"
                ],
                "combined_candidate_models": family_models_config[
                    "combined_candidate"
                ],
                "catboost_device": gpu_name,
                "catboost_gpu_preflight": catboost_preflight,
                "lightgbm_device": "CPU; installed LightGBM build is CPU-only",
                "xgboost_gpu_preflight": xgb_preflight,
                "xgboost_cuda_models_included": xgb is not None,
                "xgboost_configuration": XGB_CONFIG if xgb is not None else None,
                "combined_blend": (
                    "fixed equal-weight average of candidate LightGBM/CatBoost and "
                    "CUDA XGBoost when supported; candidate/reference mixing weight "
                    "selected only on inner validation"
                ),
                "phase2_cat_lgb_blend": "fixed 50:50",
            },
            "metrics": {
                "strict_phase2_matched": {
                    "pooled_oof_auc": baseline_auc,
                    "fold_auc": baseline_fold_auc,
                },
                "historical_candidate": {
                    "pooled_oof_auc": candidate_auc,
                    "fold_auc": candidate_fold_auc,
                    "pooled_delta_vs_phase2": candidate_auc - baseline_auc,
                    "fold_delta_vs_phase2": [
                        value - baseline
                        for value, baseline in zip(
                            candidate_fold_auc, baseline_fold_auc
                        )
                    ],
                },
                "nested_blend": {
                    "pooled_oof_auc": blend_auc,
                    "fold_auc": blend_fold_auc,
                    "pooled_delta_vs_phase2": blend_auc - baseline_auc,
                    "fold_delta_vs_phase2": fold_deltas,
                    "positive_folds": positive_folds,
                    "mean_fold_delta": mean_delta,
                    "fold_delta_standard_error": delta_se,
                    "mean_candidate_weight": mean_weight,
                    "fold_candidate_weights": selected_weights.tolist(),
                },
                "historical_families": family_summary,
                "previous_experiments": prior_comparisons,
            },
            "advancement_gate": {
                **ADVANCEMENT_GATE,
                "pooled_auc_delta": blend_auc - baseline_auc,
                "positive_folds": positive_folds,
                "mean_delta": mean_delta,
                "fold_delta_standard_error": delta_se,
                "passed": gate_passed,
                "decision": decision,
            },
            "prediction_diagnostics": {
                "candidate_vs_phase2_pearson": float(
                    np.corrcoef(candidate_oof, baseline_oof)[0, 1]
                ),
                "candidate_vs_phase2_spearman": float(
                    runtime.spearmanr(candidate_oof, baseline_oof).statistic
                ),
                "candidate_error_correlation_vs_phase2": float(
                    np.corrcoef(candidate_oof - labels, baseline_oof - labels)[0, 1]
                ),
                "candidate_test_vs_phase2_pearson": float(
                    np.corrcoef(candidate_test, baseline_test)[0, 1]
                ),
                "family_pairwise_pearson": family_correlations,
                "family_complementarity_vs_phase2": family_complementarity,
            },
            "gpu": {
                "preflight": gpu_preflight,
                "telemetry": {
                    "sample_count": len(gpu_monitor.samples),
                    "maximum_utilization_percent": max(
                        (
                            item["utilization_percent"]
                            for item in gpu_monitor.samples
                        ),
                        default=0,
                    ),
                    "maximum_memory_used_mib": max(
                        (
                            item["memory_used_mib"]
                            for item in gpu_monitor.samples
                        ),
                        default=0,
                    ),
                    "monitor_errors": gpu_monitor.errors,
                },
            },
            "submission": {
                "path": str(submission_path.resolve()),
                "sha256": sha256_file(submission_path),
                "rows": len(submission),
                "columns": submission.columns.tolist(),
                "unique_ids": int(submission[ID_COLUMN].nunique()),
                "ordered_ids_match_sample": True,
            },
            "fold_index_files": fold_index_files,
            "input_records": input_records,
            "reference_records": reference_records,
            "source_records": source_records,
            "runtime_seconds": time.perf_counter() - started,
            "training_seconds": time.perf_counter() - fold_started_all,
            "folds": fold_records,
        }
        write_json(run_dir / "experiment_summary.json", summary)
        results_markdown = _render_results(summary)
        (run_dir / "RESULTS.md").write_text(
            results_markdown,
            encoding="utf-8",
            newline="\n",
        )
        (EXPERIMENT_DIR / "RESULTS.md").write_text(
            results_markdown,
            encoding="utf-8",
            newline="\n",
        )
        write_json(
            run_dir / "feature_config.json",
            {
                "evaluated_groups": list(EVALUATED_GROUPS),
                "group_sizes": group_sizes,
                "excluded_groups": EXCLUDED_GROUPS,
                "selected_feature_caps": {
                    "maximum_groups_per_fold": MAX_SELECTED_GROUPS,
                    "maximum_interactions_per_fold": MAX_INTERACTION_FEATURES,
                },
                "blend_weight_grid": list(BLEND_WEIGHTS),
                "model_configurations": family_models_config,
                "xgboost_cuda_preflight": xgb_preflight,
            },
        )
        log_message(
            log_path,
            f"Completed: decision={decision}, nested_blend_auc={blend_auc:.8f}, "
            f"Phase2_auc={baseline_auc:.8f}, delta={blend_auc - baseline_auc:+.8f}",
        )
        output_records = {
            str(path.relative_to(run_dir)): file_record(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        }
        output_records["FEATURE_INVENTORY.csv"] = file_record(
            EXPERIMENT_DIR / "FEATURE_INVENTORY.csv"
        )
        output_records["RESULTS.md"] = file_record(EXPERIMENT_DIR / "RESULTS.md")
        manifest = {
            "experiment_id": run_dir.name,
            "inputs": input_records,
            "references": reference_records,
            "sources": source_records,
            "outputs": output_records,
        }
        write_json(run_dir / "sha256_manifest.json", manifest)
        return summary
    except Exception as error:
        failure = {
            "experiment_id": run_dir.name,
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "FAILED.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--outer-seed", type=int, default=OUTER_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir, args.run_dir, args.outer_seed)


if __name__ == "__main__":
    main()
