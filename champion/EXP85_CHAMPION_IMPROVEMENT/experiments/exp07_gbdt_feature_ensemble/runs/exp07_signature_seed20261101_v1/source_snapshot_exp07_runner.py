#!/usr/bin/env python3
"""Fold-safe GBDT feature and nested-blending screen for EXP07."""

from __future__ import annotations

import argparse
import collections
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


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
CHAMPION_DIR = REPOSITORY_ROOT / "champion"
IMPROVEMENT_DIR = CHAMPION_DIR / "EXP85_CHAMPION_IMPROVEMENT"
FROZEN_DIR = CHAMPION_DIR / "EXP85_PHASE2_FROZEN_RELEASE"
EXP06_DIR = IMPROVEMENT_DIR / "experiments" / "exp06_ft_transformer"
EXP06_RUN_DIR = (
    EXP06_DIR / "runs" / "exp06_ft_transformer_seed20261101_v5"
)
EXP04_RUN_DIR = (
    IMPROVEMENT_DIR
    / "experiments"
    / "exp04_catboost_lossguide"
    / "runs"
    / "exp04_lossguide_strict_seed20261101"
)
EXP05_RUN_DIR = (
    IMPROVEMENT_DIR
    / "experiments"
    / "exp05_tabular_neural"
    / "runs"
    / "exp05_cuda_seed20261101_v1"
)
REFERENCE_DIR = FROZEN_DIR / "EVIDENCE" / "second_seed" / "main"
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
EXP06_RUNNER = EXP06_DIR / "run_experiment.py"
HISTORICAL_FEATURE_SOURCE = Path(
    r"D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\build_hasan_600_features.py"
)
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp07_signature_seed20261101_v1"

EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
OUTER_SEED = 20261101
INNER_SEED_BASE = 20261101
SIGNATURE_SMOOTHING = 20.0
SIGNATURE_CROSSFIT_SPLITS = 5
BLEND_WEIGHTS = tuple(np.linspace(0.0, 1.0, 21).tolist())
ADVANCEMENT_GATE = {
    "minimum_pooled_auc_delta": 0.001,
    "minimum_positive_folds": 4,
    "minimum_mean_delta_standard_errors": 2.0,
    "folds": 5,
}
FEATURE_NAMES = ("EXP07_SIG_COUNT_LOG", "EXP07_SIG_TARGET_RATE")
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
SIGNATURE_COLUMNS = (
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
)


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


def numeric_day_difference(left: np.ndarray, right: np.ndarray) -> list[int | None]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    result: list[int | None] = []
    for value in difference:
        if not np.isfinite(value):
            result.append(None)
            continue
        result.append(int(value))
    return result


def build_signature_keys(raw: pd.DataFrame) -> list[tuple[Any, ...]]:
    """Build a non-ID application signature from raw columns only."""
    require(TARGET_COLUMN not in raw.columns, "Target reached signature builder")
    missing = set(SIGNATURE_COLUMNS) - set(raw.columns)
    require(not missing, f"Signature columns are missing: {sorted(missing)}")
    first_gap = numeric_day_difference(
        raw["DAYS_BIRTH"].to_numpy(),
        raw["DAYS_REGISTRATION"].to_numpy(),
    )
    second_gap = numeric_day_difference(
        raw["DAYS_BIRTH"].to_numpy(),
        raw["DAYS_ID_PUBLISH"].to_numpy(),
    )
    gender = raw["CODE_GENDER"].fillna("__MISSING__").astype(str).tolist()
    education = (
        raw["NAME_EDUCATION_TYPE"].fillna("__MISSING__").astype(str).tolist()
    )
    return list(zip(first_gap, second_gap, gender, education))


def validate_partition(
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    row_count: int,
) -> None:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    require(
        not np.intersect1d(fit_indices, apply_indices).size,
        "Signature fit/apply rows overlap",
    )
    require(
        len(np.unique(fit_indices)) == len(fit_indices)
        and len(np.unique(apply_indices)) == len(apply_indices),
        "Signature partition contains duplicate rows",
    )
    require(
        np.all((fit_indices >= 0) & (fit_indices < row_count))
        and np.all((apply_indices >= 0) & (apply_indices < row_count)),
        "Signature partition contains out-of-range rows",
    )


def encode_target_rate(
    reference_keys: list[tuple[Any, ...]],
    reference_labels: np.ndarray,
    apply_keys: list[tuple[Any, ...]],
    smoothing: float,
) -> np.ndarray:
    reference_labels = np.asarray(reference_labels, dtype=np.float64)
    require(len(reference_keys) == len(reference_labels), "Key/label length mismatch")
    require(
        len(reference_labels) > 0
        and np.isfinite(reference_labels).all()
        and set(np.unique(reference_labels)).issubset({0.0, 1.0}),
        "Target-rate reference labels must be nonempty binary values",
    )
    require(smoothing > 0.0, "Smoothing must be positive")
    prior = float(reference_labels.mean())
    stats: dict[tuple[Any, ...], list[float]] = {}
    for key, label in zip(reference_keys, reference_labels):
        state = stats.setdefault(key, [0.0, 0.0])
        state[0] += 1.0
        state[1] += float(label)
    result = np.empty(len(apply_keys), dtype=np.float64)
    for index, key in enumerate(apply_keys):
        count_sum = stats.get(key)
        if count_sum is None:
            result[index] = prior
        else:
            count, positive_sum = count_sum
            result[index] = (
                positive_sum + smoothing * prior
            ) / (count + smoothing)
    return result.astype(np.float32)


def cross_fit_target_rate(
    fit_keys: list[tuple[Any, ...]],
    fit_labels: np.ndarray,
    seed: int,
    smoothing: float = SIGNATURE_SMOOTHING,
    n_splits: int = SIGNATURE_CROSSFIT_SPLITS,
    partitions: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    fit_labels = np.asarray(fit_labels, dtype=np.int8)
    require(len(fit_keys) == len(fit_labels), "Key/label length mismatch")
    require(
        len(fit_labels) >= n_splits
        and min(np.bincount(fit_labels, minlength=2)) >= n_splits,
        "Not enough examples per class for target-rate cross-fitting",
    )
    if partitions is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        partitions = list(
            splitter.split(np.zeros(len(fit_labels)), fit_labels)
        )
    seen = np.zeros(len(fit_labels), dtype=np.int8)
    encoded = np.full(len(fit_labels), np.nan, dtype=np.float32)
    for train_local, apply_local in partitions:
        train_local = np.asarray(train_local, dtype=np.int64)
        apply_local = np.asarray(apply_local, dtype=np.int64)
        validate_partition(train_local, apply_local, len(fit_labels))
        require(
            not seen[apply_local].any(),
            "Cross-fit apply rows occur in more than one partition",
        )
        seen[apply_local] = 1
        encoded[apply_local] = encode_target_rate(
            [fit_keys[index] for index in train_local],
            fit_labels[train_local],
            [fit_keys[index] for index in apply_local],
            smoothing,
        )
    require(np.all(seen == 1), "Cross-fit partitions do not cover every fit row")
    require(np.isfinite(encoded).all(), "Cross-fitted target rates are non-finite")
    require(
        np.all((encoded >= 0.0) & (encoded <= 1.0)),
        "Cross-fitted target rates are outside [0, 1]",
    )
    return encoded


def fold_local_signature_features(
    train_keys: list[tuple[Any, ...]],
    test_keys: list[tuple[Any, ...]],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    seed: int,
    smoothing: float = SIGNATURE_SMOOTHING,
    crossfit_splits: int = SIGNATURE_CROSSFIT_SPLITS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Create cross-fit training rates and fit-only apply rates/counts."""
    labels = np.asarray(labels, dtype=np.int8)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    require(len(train_keys) == len(labels), "Training key/label length mismatch")
    validate_partition(fit_indices, apply_indices, len(train_keys))
    require(len(fit_indices) > 0, "Signature fit partition is empty")
    fit_keys = [train_keys[index] for index in fit_indices]
    fit_labels = labels[fit_indices]
    fit_target_rate = cross_fit_target_rate(
        fit_keys,
        fit_labels,
        seed=seed,
        smoothing=smoothing,
        n_splits=crossfit_splits,
    )
    fit_counts: collections.Counter[tuple[Any, ...]] = collections.Counter(
        fit_keys
    )
    train_features = {
        name: np.full(len(train_keys), np.nan, dtype=np.float32)
        for name in FEATURE_NAMES
    }
    test_features = {
        name: np.empty(len(test_keys), dtype=np.float32)
        for name in FEATURE_NAMES
    }
    train_features["EXP07_SIG_TARGET_RATE"][fit_indices] = fit_target_rate
    train_features["EXP07_SIG_COUNT_LOG"][fit_indices] = np.log1p(
        [
            max(0, fit_counts[key] - 1)
            for key in fit_keys
        ]
    ).astype(np.float32)

    apply_keys = [train_keys[index] for index in apply_indices]
    apply_rates = encode_target_rate(
        fit_keys,
        fit_labels,
        apply_keys,
        smoothing,
    )
    train_features["EXP07_SIG_TARGET_RATE"][apply_indices] = apply_rates
    train_features["EXP07_SIG_COUNT_LOG"][apply_indices] = np.log1p(
        [fit_counts[key] for key in apply_keys]
    ).astype(np.float32)
    test_features["EXP07_SIG_TARGET_RATE"] = encode_target_rate(
        fit_keys, fit_labels, test_keys, smoothing
    )
    test_features["EXP07_SIG_COUNT_LOG"] = np.log1p(
        [fit_counts[key] for key in test_keys]
    ).astype(np.float32)

    diagnostics = {
        "fit_rows": int(len(fit_indices)),
        "apply_rows": int(len(apply_indices)),
        "fit_unique_signatures": int(len(fit_counts)),
        "fit_rows_with_repeated_signature": int(
            sum(count for count in fit_counts.values() if count > 1)
        ),
        "fit_repeated_signature_groups": int(
            sum(count > 1 for count in fit_counts.values())
        ),
        "max_fit_signature_count": int(max(fit_counts.values())),
        "apply_rows_matching_fit_signature": int(
            sum(key in fit_counts for key in apply_keys)
        ),
        "test_rows_matching_fit_signature": int(
            sum(key in fit_counts for key in test_keys)
        ),
        "target_rate_crossfit_splits": crossfit_splits,
        "target_rate_smoothing": smoothing,
        "fit_row_target_excluded_by_crossfit": True,
        "apply_labels_used": False,
    }
    for values in (*train_features.values(), *test_features.values()):
        selected = values[np.isfinite(values)]
        require(
            np.isfinite(selected).all(),
            "Signature features contain non-finite assigned values",
        )
    return train_features, test_features, diagnostics


def select_blend_weight(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    weights: tuple[float, ...] = BLEND_WEIGHTS,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.float64)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.float64)
    require(
        labels.shape == baseline_predictions.shape == candidate_predictions.shape,
        "Inner blend vectors have different shapes",
    )
    require(
        np.isfinite(baseline_predictions).all()
        and np.isfinite(candidate_predictions).all(),
        "Inner blend predictions are non-finite",
    )
    candidates = []
    for weight in weights:
        blended = (1.0 - weight) * baseline_predictions + weight * candidate_predictions
        candidates.append(
            {
                "candidate_weight": float(weight),
                "auc": float(roc_auc_score(labels, blended)),
            }
        )
    best = max(
        candidates,
        key=lambda item: (item["auc"], -item["candidate_weight"]),
    )
    return {
        "candidate_weight": best["candidate_weight"],
        "inner_auc": best["auc"],
        "baseline_inner_auc": float(
            roc_auc_score(labels, baseline_predictions)
        ),
        "candidate_inner_auc": float(
            roc_auc_score(labels, candidate_predictions)
        ),
        "grid": candidates,
        "selection_rows": int(len(labels)),
        "selection_labels_source": "this outer fold's inner validation only",
    }


def append_signature_features(
    cat_train: pd.DataFrame,
    cat_test: pd.DataFrame,
    lgb_train: pd.DataFrame,
    lgb_test: pd.DataFrame,
    train_values: dict[str, np.ndarray],
    test_values: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [cat_train.copy(), cat_test.copy(), lgb_train.copy(), lgb_test.copy()]
    for frame in frames:
        duplicates = set(FEATURE_NAMES) & set(frame.columns)
        require(not duplicates, f"EXP07 features already exist: {sorted(duplicates)}")
    for name in FEATURE_NAMES:
        require(
            len(train_values[name]) == len(cat_train) == len(lgb_train)
            and len(test_values[name]) == len(cat_test) == len(lgb_test),
            f"EXP07 feature length mismatch: {name}",
        )
        frames[0][name] = train_values[name]
        frames[1][name] = test_values[name]
        frames[2][name] = train_values[name]
        frames[3][name] = test_values[name]
    require(
        frames[0].columns.tolist() == frames[1].columns.tolist()
        and frames[2].columns.tolist() == frames[3].columns.tolist(),
        "EXP07 train/test feature schemas differ",
    )
    return frames[0], frames[1], frames[2], frames[3]


def verify_hash_record(path_record: dict[str, Any], label: str) -> Path:
    path = Path(path_record["path"])
    require(path.is_file(), f"{label} is missing: {path}")
    actual = file_record(path)
    require(
        actual["sha256"] == path_record["sha256"],
        f"{label} SHA-256 mismatch: {path}",
    )
    require(
        actual["size_bytes"] == path_record["size_bytes"],
        f"{label} size mismatch: {path}",
    )
    return path


def load_exp06_runtime():
    require(EXP06_RUNNER.is_file(), f"EXP06 runner is missing: {EXP06_RUNNER}")
    spec = importlib.util.spec_from_file_location(
        "exp07_exp06_protocol_runtime", EXP06_RUNNER
    )
    require(spec is not None and spec.loader is not None, "Cannot load EXP06 runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    runtime.validate_feature_audit(frames[4], "EXP07 Phase 2 feature build")
    return frames


def fit_inner_models(
    runtime: Any,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold_number: int,
    label: str,
) -> dict[str, Any]:
    lgb_settings, max_rounds, lgb_patience = runtime.make_inner_lgbm_settings(
        fold_number
    )
    lgb_train = lgb.Dataset(
        lgb_frame.iloc[fit_indices],
        label=labels[fit_indices],
        free_raw_data=True,
    )
    lgb_valid = lgb.Dataset(
        lgb_frame.iloc[valid_indices],
        label=labels[valid_indices],
        reference=lgb_train,
        free_raw_data=True,
    )
    lgb_model = lgb.train(
        lgb_settings,
        lgb_train,
        num_boost_round=max_rounds,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(lgb_patience, verbose=False)],
    )
    lgb_best_iteration = max(1, int(lgb_model.best_iteration))
    lgb_prediction = lgb_model.predict(
        lgb_frame.iloc[valid_indices],
        num_iteration=lgb_best_iteration,
    ).astype(np.float32)
    del lgb_model, lgb_train, lgb_valid

    cat_settings = runtime.make_catboost_settings(
        fold_number,
        runtime.PHASE2_MODEL_CONFIG["catboost"]["iterations"],
    )
    cat_patience = runtime.PHASE2_MODEL_CONFIG["catboost"]["early_stopping_rounds"]
    cat_model = CatBoostClassifier(**cat_settings)
    cat_model.fit(
        cat_frame.iloc[fit_indices],
        labels[fit_indices],
        cat_features=list(runtime.CATEGORICAL_COLUMNS),
        eval_set=(cat_frame.iloc[valid_indices], labels[valid_indices]),
        early_stopping_rounds=cat_patience,
        verbose=False,
    )
    cat_params = cat_model.get_all_params()
    actual_task = str(cat_params.get("task_type", "")).upper()
    require(actual_task == "GPU", f"{label} CatBoost left GPU: {actual_task}")
    cat_best_iteration = max(1, int(cat_model.best_iteration_) + 1)
    cat_prediction = cat_model.predict_proba(
        cat_frame.iloc[valid_indices]
    )[:, 1].astype(np.float32)
    del cat_model
    require(
        np.isfinite(lgb_prediction).all()
        and np.isfinite(cat_prediction).all(),
        f"{label} inner predictions are non-finite",
    )
    return {
        "lgb_prediction": lgb_prediction,
        "cat_prediction": cat_prediction,
        "blend_prediction": (
            (lgb_prediction.astype(np.float64) + cat_prediction.astype(np.float64))
            / 2.0
        ).astype(np.float32),
        "lgb_best_iteration": lgb_best_iteration,
        "cat_best_iteration": cat_best_iteration,
        "catboost_task_type": actual_task,
    }


def fit_outer_candidate(
    runtime: Any,
    cat_frame: pd.DataFrame,
    cat_test: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    lgb_test: pd.DataFrame,
    labels: np.ndarray,
    outer_fit_indices: np.ndarray,
    outer_valid_indices: np.ndarray,
    fold_number: int,
    lgb_best_iteration: int,
    cat_best_iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    lgb_settings, _, _ = runtime.make_inner_lgbm_settings(fold_number)
    lgb_train = lgb.Dataset(
        lgb_frame.iloc[outer_fit_indices],
        label=labels[outer_fit_indices],
        free_raw_data=True,
    )
    lgb_model = lgb.train(
        lgb_settings,
        lgb_train,
        num_boost_round=lgb_best_iteration,
    )
    lgb_valid = lgb_model.predict(
        lgb_frame.iloc[outer_valid_indices],
        num_iteration=lgb_best_iteration,
    ).astype(np.float32)
    lgb_test_prediction = lgb_model.predict(
        lgb_test, num_iteration=lgb_best_iteration
    ).astype(np.float32)
    lgb_model.save_model(str(run_dir / f"fold_{fold_number}_signature_lgb.txt"))
    lgb_importance = {
        name: float(value)
        for name, value in zip(
            lgb_frame.columns,
            lgb_model.feature_importance(importance_type="gain"),
        )
    }
    del lgb_model, lgb_train

    cat_settings = runtime.make_catboost_settings(
        fold_number,
        cat_best_iteration,
    )
    cat_model = CatBoostClassifier(**cat_settings)
    cat_model.fit(
        cat_frame.iloc[outer_fit_indices],
        labels[outer_fit_indices],
        cat_features=list(runtime.CATEGORICAL_COLUMNS),
        verbose=False,
    )
    cat_params = cat_model.get_all_params()
    actual_task = str(cat_params.get("task_type", "")).upper()
    require(actual_task == "GPU", f"Outer CatBoost left GPU: {actual_task}")
    cat_valid = cat_model.predict_proba(
        cat_frame.iloc[outer_valid_indices]
    )[:, 1].astype(np.float32)
    cat_test_prediction = cat_model.predict_proba(cat_test)[:, 1].astype(np.float32)
    cat_model.save_model(
        str(run_dir / f"fold_{fold_number}_signature_cat.cbm")
    )
    cat_importance = {
        name: float(value)
        for name, value in zip(
            cat_model.feature_names_,
            cat_model.get_feature_importance(),
        )
    }
    del cat_model
    return {
        "lgb_valid": lgb_valid,
        "lgb_test": lgb_test_prediction,
        "cat_valid": cat_valid,
        "cat_test": cat_test_prediction,
        "lgb_importance": lgb_importance,
        "cat_importance": cat_importance,
        "catboost_task_type": actual_task,
    }


def auc_by_fold(
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    return [
        float(roc_auc_score(labels[valid], predictions[valid]))
        for _, valid in folds
    ]


def build_submission(
    ids: np.ndarray,
    predictions: np.ndarray,
    sample: pd.DataFrame,
    name: str,
    output_path: Path,
) -> pd.DataFrame:
    predictions = np.asarray(predictions, dtype=np.float64)
    require(
        predictions.shape == (len(ids),)
        and np.isfinite(predictions).all()
        and np.all((predictions >= 0.0) & (predictions <= 1.0)),
        f"{name} probabilities are invalid",
    )
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: predictions})
    require(
        submission.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and submission[ID_COLUMN].is_unique
        and np.array_equal(
            submission[ID_COLUMN].to_numpy(),
            sample[ID_COLUMN].to_numpy(),
        ),
        f"{name} submission schema/ID alignment failed",
    )
    submission.to_csv(output_path, index=False)
    return submission


def run_experiment(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir, run_dir = data_dir.resolve(), run_dir.resolve()
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
        input_records = {}
        for name, path in raw_paths.items():
            require(path.is_file(), f"Required raw {name} CSV is missing: {path}")
            record = file_record(path)
            require(
                record["sha256"] == EXPECTED_INPUT_SHA256[name],
                f"Raw {name} SHA-256 mismatch",
            )
            input_records[name] = record

        gpu_before = gpu_preflight()
        runtime = load_exp06_runtime()
        source_paths = {
            "exp07_runner": Path(__file__).resolve(),
            "exp06_protocol_runner": EXP06_RUNNER,
            "phase2_rowwise_feature_builder": PHASE2_SOURCE,
            "phase4_fold_local_pipeline": PHASE4_SOURCE,
        }
        for name, path in source_paths.items():
            require(path.is_file(), f"Required source missing ({name}): {path}")
        source_records = {
            name: file_record(path) for name, path in source_paths.items()
        }
        phase2_manifest_path = REFERENCE_DIR / "sha256_manifest.json"
        phase2_manifest = json.loads(
            phase2_manifest_path.read_text(encoding="utf-8")
        )
        exp06_manifest_path = EXP06_RUN_DIR / "sha256_manifest.json"
        exp06_manifest = json.loads(exp06_manifest_path.read_text(encoding="utf-8"))
        exp06_summary = json.loads(
            (EXP06_RUN_DIR / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            exp06_summary["status"] == "COMPLETED"
            and exp06_summary["model"]["gpu_training_confirmed"] is True,
            "Strict EXP06 matched reference is incomplete",
        )
        for name, record in input_records.items():
            require(
                exp06_manifest["inputs"][name]["sha256"] == record["sha256"],
                f"EXP06 raw {name} reference hash differs",
            )
            require(
                phase2_manifest["inputs"][name]["sha256"] == record["sha256"],
                f"Phase 2 raw {name} reference hash differs",
            )
        for source_name, path in (
            ("phase2_raw_feature_builder", PHASE2_SOURCE),
            ("phase2_fold_local_pipeline", PHASE4_SOURCE),
        ):
            expected_source = exp06_manifest["sources"][source_name]
            require(
                sha256_file(path) == expected_source["sha256"],
                f"Frozen Phase 2 source changed since EXP06: {source_name}",
            )
        phase2_oof_path = EXP06_RUN_DIR / "oof_phase2_strict_matched.npy"
        phase2_oof_record = exp06_manifest["outputs"][
            "oof_phase2_strict_matched.npy"
        ]
        require(
            sha256_file(phase2_oof_path) == phase2_oof_record["sha256"],
            "Strict Phase 2 OOF reference hash mismatch",
        )
        phase2_oof = np.load(phase2_oof_path, allow_pickle=False).astype(
            np.float32
        )
        phase2_test_path = EXP06_RUN_DIR / "test_phase2_strict_matched.npy"
        phase2_test_record = exp06_manifest["outputs"][
            "test_phase2_strict_matched.npy"
        ]
        require(
            sha256_file(phase2_test_path) == phase2_test_record["sha256"],
            "Strict Phase 2 test reference hash mismatch",
        )
        phase2_test = np.load(phase2_test_path, allow_pickle=False).astype(
            np.float32
        )

        train_raw = pd.read_csv(raw_paths["train"])
        test_raw = pd.read_csv(raw_paths["test"])
        sample = pd.read_csv(raw_paths["sample"])
        require(
            sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
            and np.array_equal(
                sample[ID_COLUMN].to_numpy(),
                test_raw[ID_COLUMN].to_numpy(),
            ),
            "Raw test/sample submission IDs or order differ",
        )
        require(
            train_raw[ID_COLUMN].is_unique
            and test_raw[ID_COLUMN].is_unique
            and set(train_raw[ID_COLUMN]).isdisjoint(set(test_raw[ID_COLUMN])),
            "Raw train/test applicant IDs are duplicated or overlap",
        )
        labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
        require(
            len(labels) == len(phase2_oof)
            and len(test_raw) == len(phase2_test),
            "Reference predictions do not match raw row counts",
        )
        require(
            np.isfinite(phase2_oof).all()
            and np.isfinite(phase2_test).all()
            and np.all((phase2_oof >= 0.0) & (phase2_oof <= 1.0))
            and np.all((phase2_test >= 0.0) & (phase2_test <= 1.0)),
            "Strict Phase 2 reference probabilities are invalid",
        )

        folds = list(
            StratifiedKFold(
                n_splits=5, shuffle=True, random_state=OUTER_SEED
            ).split(np.zeros(len(labels)), labels)
        )
        runtime.validate_fold_partition(folds, len(labels))
        reference_folds = runtime.verify_reference_folds(folds)
        raw_train_features = runtime.build_rowwise_features(
            train_raw.drop(columns=[TARGET_COLUMN])
        )
        raw_test_features = runtime.build_rowwise_features(test_raw)
        require(
            raw_train_features.columns.tolist()
            == raw_test_features.columns.tolist(),
            "Phase 2 raw feature schemas differ",
        )
        train_keys = build_signature_keys(
            train_raw.drop(columns=[TARGET_COLUMN])
        )
        test_keys = build_signature_keys(test_raw)
        historical_feature_reference = (
            file_record(HISTORICAL_FEATURE_SOURCE)
            if HISTORICAL_FEATURE_SOURCE.is_file()
            else {
                "path": str(HISTORICAL_FEATURE_SOURCE),
                "available": False,
                "classification": "read-only design reference, not runtime input",
            }
        )

        exp04_oof, exp04_details = runtime.verified_reference(
            EXP04_RUN_DIR,
            "oof_catboost_lossguide.npy",
            "COMPLETED",
            labels,
            input_records,
        )
        exp05_oof, exp05_details = runtime.verified_reference(
            EXP05_RUN_DIR,
            "oof_tabular_neural.npy",
            "COMPLETED",
            labels,
            input_records,
        )
        exp06_oof_path = EXP06_RUN_DIR / "oof_ft_transformer.npy"
        exp06_oof_record = exp06_manifest["outputs"]["oof_ft_transformer.npy"]
        require(
            sha256_file(exp06_oof_path) == exp06_oof_record["sha256"],
            "EXP06 OOF candidate hash mismatch",
        )
        exp06_oof = np.load(exp06_oof_path, allow_pickle=False)
        runtime.validate_probabilities(exp06_oof, len(labels), "EXP06 OOF")

        candidate_oof = np.full(len(labels), np.nan, dtype=np.float32)
        blend_oof = np.full(len(labels), np.nan, dtype=np.float32)
        candidate_test_folds: list[np.ndarray] = []
        fold_records: list[dict[str, Any]] = []
        importance_records: list[dict[str, Any]] = []
        fold_started_all = time.perf_counter()
        log_message(
            log_path,
            "Starting five-fold application-signature feature experiment "
            f"on {gpu_before['name']}",
        )
        with runtime.GpuTelemetryMonitor(interval_seconds=1.0) as gpu_monitor:
            for fold_number, (outer_fit, outer_valid) in enumerate(folds, 1):
                fold_started = time.perf_counter()
                inner_fit, inner_valid = runtime.build_inner_split(
                    labels,
                    outer_fit,
                    OUTER_SEED + fold_number,
                )
                (
                    inner_cat,
                    inner_cat_test,
                    inner_lgb,
                    inner_lgb_test,
                    inner_audit,
                ) = build_phase2_frames(
                    runtime,
                    train_raw,
                    test_raw,
                    raw_train_features,
                    raw_test_features,
                    inner_fit,
                    inner_valid,
                )
                outer_cat, outer_cat_test, outer_lgb, outer_lgb_test, outer_audit = (
                    build_phase2_frames(
                        runtime,
                        train_raw,
                        test_raw,
                        raw_train_features,
                        raw_test_features,
                        outer_fit,
                        outer_valid,
                    )
                )
                inner_signature_train, inner_signature_test, inner_sig_audit = (
                    fold_local_signature_features(
                        train_keys,
                        test_keys,
                        labels,
                        inner_fit,
                        inner_valid,
                        seed=INNER_SEED_BASE + 10_000 + fold_number,
                    )
                )
                outer_signature_train, outer_signature_test, outer_sig_audit = (
                    fold_local_signature_features(
                        train_keys,
                        test_keys,
                        labels,
                        outer_fit,
                        outer_valid,
                        seed=OUTER_SEED + 20_000 + fold_number,
                    )
                )
                (
                    inner_candidate_cat,
                    inner_candidate_cat_test,
                    inner_candidate_lgb,
                    inner_candidate_lgb_test,
                ) = append_signature_features(
                    inner_cat,
                    inner_cat_test,
                    inner_lgb,
                    inner_lgb_test,
                    inner_signature_train,
                    inner_signature_test,
                )
                (
                    outer_candidate_cat,
                    outer_candidate_cat_test,
                    outer_candidate_lgb,
                    outer_candidate_lgb_test,
                ) = append_signature_features(
                    outer_cat,
                    outer_cat_test,
                    outer_lgb,
                    outer_lgb_test,
                    outer_signature_train,
                    outer_signature_test,
                )
                require(
                    inner_audit["target_encoding"][
                        "validation_encoded_from_outer_fit_only"
                    ]
                    and outer_audit["target_encoding"][
                        "validation_encoded_from_outer_fit_only"
                    ],
                    "Phase 2 target encoding audit failed",
                )

                baseline_inner = fit_inner_models(
                    runtime,
                    inner_cat,
                    inner_lgb,
                    labels,
                    inner_fit,
                    inner_valid,
                    fold_number,
                    f"fold {fold_number} baseline",
                )
                candidate_inner = fit_inner_models(
                    runtime,
                    inner_candidate_cat,
                    inner_candidate_lgb,
                    labels,
                    inner_fit,
                    inner_valid,
                    fold_number,
                    f"fold {fold_number} signature candidate",
                )
                blend_selection = select_blend_weight(
                    labels[inner_valid],
                    baseline_inner["blend_prediction"],
                    candidate_inner["blend_prediction"],
                )
                candidate_outer = fit_outer_candidate(
                    runtime,
                    outer_candidate_cat,
                    outer_candidate_cat_test,
                    outer_candidate_lgb,
                    outer_candidate_lgb_test,
                    labels,
                    outer_fit,
                    outer_valid,
                    fold_number,
                    candidate_inner["lgb_best_iteration"],
                    candidate_inner["cat_best_iteration"],
                    run_dir,
                )
                candidate_valid = (
                    candidate_outer["lgb_valid"].astype(np.float64)
                    + candidate_outer["cat_valid"].astype(np.float64)
                ).astype(np.float32) / 2.0
                candidate_test = (
                    candidate_outer["lgb_test"].astype(np.float64)
                    + candidate_outer["cat_test"].astype(np.float64)
                ).astype(np.float32) / 2.0
                candidate_oof[outer_valid] = candidate_valid
                weight = blend_selection["candidate_weight"]
                blend_oof[outer_valid] = (
                    (1.0 - weight) * phase2_oof[outer_valid]
                    + weight * candidate_valid
                ).astype(np.float32)
                candidate_test_folds.append(candidate_test)
                baseline_auc = float(
                    roc_auc_score(labels[outer_valid], phase2_oof[outer_valid])
                )
                candidate_auc = float(
                    roc_auc_score(labels[outer_valid], candidate_valid)
                )
                blend_auc = float(
                    roc_auc_score(labels[outer_valid], blend_oof[outer_valid])
                )
                fold_record = {
                    "fold": fold_number,
                    "outer_fit_rows": int(len(outer_fit)),
                    "outer_validation_rows": int(len(outer_valid)),
                    "inner_fit_rows": int(len(inner_fit)),
                    "inner_validation_rows": int(len(inner_valid)),
                    "inner_baseline_lgb_best_iteration": baseline_inner[
                        "lgb_best_iteration"
                    ],
                    "inner_baseline_cat_best_iteration": baseline_inner[
                        "cat_best_iteration"
                    ],
                    "inner_candidate_lgb_best_iteration": candidate_inner[
                        "lgb_best_iteration"
                    ],
                    "inner_candidate_cat_best_iteration": candidate_inner[
                        "cat_best_iteration"
                    ],
                    "inner_baseline_auc": blend_selection["baseline_inner_auc"],
                    "inner_candidate_auc": blend_selection["candidate_inner_auc"],
                    "inner_blend_weight": weight,
                    "inner_blend_auc": blend_selection["inner_auc"],
                    "inner_blend_grid": blend_selection["grid"],
                    "baseline_outer_auc": baseline_auc,
                    "candidate_outer_auc": candidate_auc,
                    "nested_blend_outer_auc": blend_auc,
                    "candidate_fold_delta": candidate_auc - baseline_auc,
                    "blend_fold_delta": blend_auc - baseline_auc,
                    "signature_inner_audit": inner_sig_audit,
                    "signature_outer_audit": outer_sig_audit,
                    "phase2_inner_feature_audit": inner_audit,
                    "phase2_outer_feature_audit": outer_audit,
                    "candidate_gpu_device": gpu_before["name"],
                    "catboost_task_type": candidate_outer[
                        "catboost_task_type"
                    ],
                    "elapsed_seconds": time.perf_counter() - fold_started,
                }
                fold_records.append(fold_record)
                for model_name, importance in (
                    ("lightgbm", candidate_outer["lgb_importance"]),
                    ("catboost", candidate_outer["cat_importance"]),
                ):
                    importance_records.extend(
                        {
                            "fold": fold_number,
                            "model": model_name,
                            "feature": feature,
                            "importance": float(value),
                        }
                        for feature, value in importance.items()
                    )
                np.savez_compressed(
                    run_dir / f"fold_{fold_number}_indices.npz",
                    fit_indices=outer_fit.astype(np.int32),
                    valid_indices=outer_valid.astype(np.int32),
                    inner_fit_indices=inner_fit.astype(np.int32),
                    inner_valid_indices=inner_valid.astype(np.int32),
                )
                np.savez_compressed(
                    run_dir / f"fold_{fold_number}_predictions.npz",
                    valid_indices=outer_valid.astype(np.int32),
                    phase2_reference=phase2_oof[outer_valid],
                    candidate_signature=candidate_valid,
                    nested_blend=blend_oof[outer_valid],
                    inner_baseline=baseline_inner["blend_prediction"],
                    inner_candidate=candidate_inner["blend_prediction"],
                    candidate_test=candidate_test,
                )
                log_message(
                    log_path,
                    f"Fold {fold_number}/5: candidate={candidate_auc:.8f} "
                    f"Phase2={baseline_auc:.8f} blend={blend_auc:.8f} "
                    f"weight={weight:.2f} CatBoost=GPU "
                    f"elapsed={fold_record['elapsed_seconds']:.1f}s",
                )

        require(
            gpu_monitor.samples,
            "No GPU telemetry samples were recorded during model training",
        )
        require(
            all("RTX 4050" in sample["name"] for sample in gpu_monitor.samples),
            "GPU telemetry detected a different device",
        )
        require(
            np.isfinite(candidate_oof).all()
            and np.isfinite(blend_oof).all(),
            "OOF arrays do not cover all rows",
        )
        candidate_test_mean = np.mean(
            np.stack(candidate_test_folds, axis=0).astype(np.float64),
            axis=0,
        ).astype(np.float32)
        mean_weight = float(
            np.mean([row["inner_blend_weight"] for row in fold_records])
        )
        blended_test = (
            (1.0 - mean_weight) * phase2_test.astype(np.float64)
            + mean_weight * candidate_test_mean.astype(np.float64)
        ).astype(np.float32)
        candidate_oof_auc = float(roc_auc_score(labels, candidate_oof))
        blend_oof_auc = float(roc_auc_score(labels, blend_oof))
        phase2_auc = float(roc_auc_score(labels, phase2_oof))
        candidate_fold_auc = auc_by_fold(labels, candidate_oof, folds)
        blend_fold_auc = auc_by_fold(labels, blend_oof, folds)
        baseline_fold_auc = auc_by_fold(labels, phase2_oof, folds)
        candidate_fold_deltas = [
            candidate - baseline
            for candidate, baseline in zip(candidate_fold_auc, baseline_fold_auc)
        ]
        blend_fold_deltas = [
            candidate - baseline
            for candidate, baseline in zip(blend_fold_auc, baseline_fold_auc)
        ]
        positive_folds = int(sum(delta > 0.0 for delta in blend_fold_deltas))
        mean_delta = float(np.mean(blend_fold_deltas))
        delta_se = float(
            np.std(blend_fold_deltas, ddof=1) / math.sqrt(len(blend_fold_deltas))
        )
        gate_passed = (
            blend_oof_auc - phase2_auc
            >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
            and positive_folds
            >= ADVANCEMENT_GATE["minimum_positive_folds"]
            and mean_delta
            > ADVANCEMENT_GATE["minimum_mean_delta_standard_errors"] * delta_se
        )
        decision = (
            "PROMISING_SCREEN_REQUIRES_INDEPENDENT_SEED"
            if gate_passed
            else "REJECTED"
        )
        submission_model = build_submission(
            test_raw[ID_COLUMN].to_numpy(),
            candidate_test_mean,
            sample,
            "signature model",
            run_dir / "submission_signature_model.csv",
        )
        submission_blend = build_submission(
            test_raw[ID_COLUMN].to_numpy(),
            blended_test,
            sample,
            "nested blend",
            run_dir / "submission_nested_blend.csv",
        )
        np.save(run_dir / "oof_signature_model.npy", candidate_oof)
        np.save(run_dir / "oof_nested_blend.npy", blend_oof)
        np.save(run_dir / "oof_phase2_strict_matched.npy", phase2_oof)
        np.save(run_dir / "test_signature_model.npy", candidate_test_mean)
        np.save(run_dir / "test_nested_blend.npy", blended_test)
        np.save(run_dir / "test_phase2_strict_matched.npy", phase2_test)
        pd.DataFrame(importance_records).to_csv(
            run_dir / "feature_importance.csv", index=False
        )
        write_json(run_dir / "fold_metrics.json", fold_records)
        write_json(run_dir / "gpu_telemetry.json", gpu_monitor.samples)
        write_json(
            run_dir / "feature_config.json",
            {
                "feature_names": list(FEATURE_NAMES),
                "signature_columns": list(SIGNATURE_COLUMNS),
                "key_formula": [
                    "int(DAYS_BIRTH - DAYS_REGISTRATION)",
                    "int(DAYS_BIRTH - DAYS_ID_PUBLISH)",
                    "CODE_GENDER with missing sentinel",
                    "NAME_EDUCATION_TYPE with missing sentinel",
                ],
                "target_rate_smoothing": SIGNATURE_SMOOTHING,
                "target_rate_training_encoding": (
                    "5-fold stratified cross-fit within each model-fit partition"
                ),
                "validation_test_target_rate": (
                    "fit-partition labels only; no apply-row labels"
                ),
                "frequency_count": (
                    "fit-partition covariates only; subtract self for fit rows"
                ),
                "blend_weight_grid": list(BLEND_WEIGHTS),
                "blend_weight_selection": (
                    "inner-validation ROC-AUC only; lower candidate weight wins ties"
                ),
            },
        )
        summary = {
            "experiment_id": run_dir.name,
            "status": "COMPLETED_SCREEN",
            "decision": decision,
            "official_competition_score": "NOT_EVALUATED",
            "candidate_public_score": "NOT_EVALUATED",
            "outer_seed": OUTER_SEED,
            "inputs": input_records,
            "sources": source_records,
            "references": {
                "strict_phase2_oof": file_record(phase2_oof_path),
                "strict_phase2_test": file_record(phase2_test_path),
                "strict_phase2_manifest": file_record(phase2_manifest_path),
                "exp04_oof": exp04_details,
                "exp05_oof": exp05_details,
                "exp06_summary": file_record(
                    EXP06_RUN_DIR / "experiment_summary.json"
                ),
                "exp06_manifest": file_record(exp06_manifest_path),
                "exp04_oof": exp04_details,
                "exp05_oof": exp05_details,
                "exp06_oof": {
                    "path": str(exp06_oof_path.resolve()),
                    "sha256": exp06_oof_record["sha256"],
                    "classification": "diagnostic only; not a feature or blend input",
                },
                "historical_feature_builder_reference": historical_feature_reference,
                "folds_verified": reference_folds,
            },
            "feature_inventory": {
                "file": str((EXPERIMENT_DIR / "FEATURE_INVENTORY.md").resolve()),
                "sha256": sha256_file(EXPERIMENT_DIR / "FEATURE_INVENTORY.md"),
                "new_features": list(FEATURE_NAMES),
                "feature_count_added": len(FEATURE_NAMES),
                "phase2_base_feature_count": int(len(outer_cat.columns)),
            },
            "fold_protocol": {
                "outer_folds": 5,
                "outer_validation_isolated_from_model_and_blend_selection": True,
                "inner_validation_fraction": runtime.INNER_VALIDATION_FRACTION,
                "inner_seed_base": INNER_SEED_BASE,
                "signature_target_encoding_splits": SIGNATURE_CROSSFIT_SPLITS,
                "outer_training_signature_encoding": "cross-fitted OOF within outer-fit",
                "inner_training_signature_encoding": "cross-fitted OOF within inner-fit",
                "validation_test_signature_encoding": "fit partition labels only",
            },
            "model": {
                "configuration": runtime.PHASE2_MODEL_CONFIG,
                "lightgbm_device": "CPU (frozen installed build)",
                "catboost_device": gpu_before["name"],
                "catboost_gpu_task_type_verified": True,
                "blend_components": "fixed 50:50 LightGBM/CatBoost",
                "nested_ensemble_weight_grid": list(BLEND_WEIGHTS),
                "mean_selected_candidate_weight": mean_weight,
                "fold_selected_candidate_weights": [
                    row["inner_blend_weight"] for row in fold_records
                ],
                "gpu_preflight": gpu_before,
                "gpu_telemetry": {
                    "sample_count": len(gpu_monitor.samples),
                    "max_utilization_percent": max(
                        row["utilization_percent"]
                        for row in gpu_monitor.samples
                    ),
                    "max_memory_used_mib": max(
                        row["memory_used_mib"] for row in gpu_monitor.samples
                    ),
                    "monitor_errors": gpu_monitor.errors,
                },
            },
            "feature_repetition_diagnostics": {
                "training_rows_with_repeated_signature": int(
                    sum(
                        row["signature_outer_audit"][
                            "fit_rows_with_repeated_signature"
                        ]
                        for row in fold_records
                    )
                    / len(fold_records)
                ),
                "test_rows_matching_fit_signature_mean": float(
                    np.mean(
                        [
                            row["signature_outer_audit"][
                                "test_rows_matching_fit_signature"
                            ]
                            for row in fold_records
                        ]
                    )
                ),
            },
            "metrics": {
                "candidate_signature": {
                    "pooled_oof_auc": candidate_oof_auc,
                    "macro_fold_auc": float(np.mean(candidate_fold_auc)),
                    "fold_auc": candidate_fold_auc,
                    "fold_delta_vs_phase2": candidate_fold_deltas,
                    "pooled_delta_vs_phase2": candidate_oof_auc - phase2_auc,
                    "positive_folds": int(
                        sum(delta > 0.0 for delta in candidate_fold_deltas)
                    ),
                },
                "nested_blend": {
                    "pooled_oof_auc": blend_oof_auc,
                    "macro_fold_auc": float(np.mean(blend_fold_auc)),
                    "fold_auc": blend_fold_auc,
                    "fold_delta_vs_phase2": blend_fold_deltas,
                    "pooled_delta_vs_phase2": blend_oof_auc - phase2_auc,
                    "positive_folds": positive_folds,
                    "mean_fold_delta": mean_delta,
                    "fold_delta_standard_error": delta_se,
                },
                "strict_phase2_matched": {
                    "pooled_oof_auc": phase2_auc,
                    "fold_auc": baseline_fold_auc,
                },
                "exp04_catboost_lossguide": {
                    "pooled_oof_auc": float(roc_auc_score(labels, exp04_oof)),
                    "fold_auc": auc_by_fold(labels, exp04_oof, folds),
                    "candidate_pooled_delta": blend_oof_auc
                    - float(roc_auc_score(labels, exp04_oof)),
                },
                "exp05_residual_mlp": {
                    "pooled_oof_auc": float(roc_auc_score(labels, exp05_oof)),
                    "fold_auc": auc_by_fold(labels, exp05_oof, folds),
                    "candidate_pooled_delta": blend_oof_auc
                    - float(roc_auc_score(labels, exp05_oof)),
                },
                "exp06_ft_transformer": {
                    "pooled_oof_auc": float(roc_auc_score(labels, exp06_oof)),
                    "fold_auc": auc_by_fold(labels, exp06_oof, folds),
                    "candidate_pooled_delta": blend_oof_auc
                    - float(roc_auc_score(labels, exp06_oof)),
                    "comparison_classification": (
                        "secondary architecture comparison; diagnostic only"
                    ),
                },
            },
            "advancement_gate": {
                **ADVANCEMENT_GATE,
                "pooled_auc_delta": blend_oof_auc - phase2_auc,
                "positive_folds": positive_folds,
                "mean_delta": mean_delta,
                "fold_delta_standard_error": delta_se,
                "passed": gate_passed,
                "decision": decision,
            },
            "prediction_diagnostics": {
                "candidate_oof_pearson_vs_phase2": float(
                    np.corrcoef(candidate_oof, phase2_oof)[0, 1]
                ),
                "candidate_oof_spearman_vs_phase2": float(
                    runtime.spearmanr(candidate_oof, phase2_oof).statistic
                ),
                "candidate_oof_error_correlation_vs_phase2": float(
                    np.corrcoef(candidate_oof - labels, phase2_oof - labels)[0, 1]
                ),
                "candidate_test_pearson_vs_phase2": float(
                    np.corrcoef(candidate_test_mean, phase2_test)[0, 1]
                ),
                "nested_blend_test_distribution": summarize(blended_test),
                "candidate_test_distribution": summarize(candidate_test_mean),
            },
            "submission": {
                "candidate_path": str(
                    (run_dir / "submission_nested_blend.csv").resolve()
                ),
                "candidate_sha256": sha256_file(
                    run_dir / "submission_nested_blend.csv"
                ),
                "candidate_rows": int(len(submission_blend)),
                "candidate_schema": list(submission_blend.columns),
                "candidate_unique_ids": int(
                    submission_blend[ID_COLUMN].nunique()
                ),
                "signature_model_sha256": sha256_file(
                    run_dir / "submission_signature_model.csv"
                ),
            },
            "runtime_seconds": time.perf_counter() - started,
            "training_seconds": time.perf_counter() - fold_started_all,
        }
        log_message(
            log_path,
            f"Completed: decision={decision}, candidate_auc={candidate_oof_auc:.8f}, "
            f"nested_blend_auc={blend_oof_auc:.8f}, Phase2_auc={phase2_auc:.8f}",
        )
        write_json(run_dir / "experiment_summary.json", summary)
        output_records = {
            str(path.relative_to(run_dir)): file_record(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        }
        manifest = {
            "experiment_id": run_dir.name,
            "inputs": {
                **input_records,
                "phase2_strict_oof": file_record(phase2_oof_path),
                "phase2_strict_test": file_record(phase2_test_path),
                "phase2_manifest": file_record(phase2_manifest_path),
                "exp06_manifest": file_record(exp06_manifest_path),
                "historical_feature_builder_reference": historical_feature_reference,
            },
            "sources": {
                **source_records,
                "feature_inventory": file_record(
                    EXPERIMENT_DIR / "FEATURE_INVENTORY.md"
                ),
            },
            "outputs": output_records,
        }
        write_json(run_dir / "sha256_manifest.json", manifest)
        return summary
    except Exception as error:
        failure = {
            "experiment_id": run_dir.name,
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "FAILED.json", failure)
        raise


def gpu_preflight() -> dict[str, Any]:
    result = subprocess.run(
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
    lines = result.stdout.strip().splitlines()
    require(len(lines) == 1, "EXP07 expects one visible CUDA device")
    name, driver, total, free, utilization = [
        part.strip() for part in lines[0].split(",")
    ]
    require("RTX 4050" in name, f"Unexpected GPU device: {name}")
    require(int(free) >= 1000, f"Insufficient free GPU VRAM: {free} MiB")
    require(
        int(utilization) < 50,
        f"GPU is already busy ({utilization}%); do not overlap training jobs",
    )
    return {
        "name": name,
        "driver_version": driver,
        "total_memory_mib": int(total),
        "free_memory_mib_before_run": int(free),
        "utilization_percent_before_run": int(utilization),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()
