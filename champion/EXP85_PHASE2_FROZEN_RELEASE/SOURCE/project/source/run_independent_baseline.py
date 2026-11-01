#!/usr/bin/env python3
"""Train a raw-only, inductive CatBoost/LightGBM baseline with resumable OOF folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import sklearn
from catboost import CatBoostClassifier
from scipy.special import ndtr
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
HISTORICAL_REFERENCE = Path(
    r"D:\gci\gemini\v_20261027_optimal_macro_apex_v85.csv"
)
CATEGORICAL_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
N_SPLITS = 5
FOLD_SEED = 42
CPU_THREADS = min(8, os.cpu_count() or 1)

MODEL_CONFIG = {
    "lightgbm": {
        "device": "CPU",
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.10,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "num_threads": CPU_THREADS,
        "verbosity": -1,
        "num_boost_round": 3000,
        "early_stopping_rounds": 150,
    },
    "catboost": {
        "device": "GPU:0",
        "iterations": 2500,
        "depth": 7,
        "learning_rate": 0.045,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "border_count": 128,
        "task_type": "GPU",
        "devices": "0",
        "early_stopping_rounds": 150,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def build_rowwise_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive features independently per row; no labels or other rows are read."""
    missing_columns = set(CATEGORICAL_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing categorical columns: {sorted(missing_columns)}")
    if ID_COLUMN not in raw or TARGET_COLUMN in raw:
        if TARGET_COLUMN in raw:
            raise ValueError("Feature builder must not receive the target column")
        raise ValueError(f"Missing required ID column: {ID_COLUMN}")

    numeric = raw.drop(columns=[ID_COLUMN]).select_dtypes(include=[np.number]).copy()
    if np.isinf(numeric.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Raw numeric features contain infinite values")
    categorical = raw.loc[:, CATEGORICAL_COLUMNS].reset_index(drop=True).copy()
    categorical = categorical.fillna("MISSING").astype(str)
    features = numeric.reset_index(drop=True)

    for column in numeric.columns:
        features[f"NA_{column}"] = numeric[column].isna().to_numpy(
            dtype=np.float32
        )
    features["RAW_MISSING_COUNT"] = raw.drop(columns=[ID_COLUMN]).isna().sum(axis=1).to_numpy(
        dtype=np.float32
    )

    get = lambda name: raw[name].to_numpy(dtype=np.float64)
    income = get("AMT_INCOME_TOTAL")
    credit = get("AMT_CREDIT")
    annuity = get("AMT_ANNUITY")
    goods = get("AMT_GOODS_PRICE")
    family = raw["CNT_FAM_MEMBERS"].fillna(1.0).to_numpy(dtype=np.float64)
    birth = get("DAYS_BIRTH")
    emp_raw = get("DAYS_EMPLOYED")
    emp_clean = np.where(emp_raw == 365243.0, np.nan, emp_raw)
    registration = get("DAYS_REGISTRATION")
    id_published = get("DAYS_ID_PUBLISH")
    phone = get("DAYS_LAST_PHONE_CHANGE")
    goods_filled = np.where(np.isnan(goods), credit, goods)

    derived: dict[str, np.ndarray] = {
        "DAYS_EMPLOYED_CLEAN": emp_clean,
        "DAYS_EMPLOYED_SENTINEL": (emp_raw == 365243.0).astype(np.float32),
        "AGE_YEARS": -birth / 365.25,
        "EMPLOYED_YEARS": -emp_clean / 365.25,
        "EMPLOYED_TO_AGE": safe_ratio(emp_clean, birth + 1.0),
        "REGISTRATION_TO_AGE": safe_ratio(registration, birth + 1.0),
        "ID_PUBLISH_TO_AGE": safe_ratio(id_published, birth + 1.0),
        "PHONE_CHANGE_YEARS": -phone / 365.25,
        "CREDIT_TO_INCOME": safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "CREDIT_TO_GOODS": safe_ratio(credit, goods_filled),
        "ANNUITY_TO_CREDIT": safe_ratio(annuity, credit),
        "CREDIT_TERM": safe_ratio(credit, annuity),
        "INCOME_PER_FAMILY": safe_ratio(income, family),
        "CREDIT_PER_FAMILY": safe_ratio(credit, family),
        "ANNUITY_PER_FAMILY": safe_ratio(annuity, family),
        "DISPOSABLE_INCOME": income - 12.0 * annuity,
        "DISPOSABLE_INCOME_TO_INCOME": safe_ratio(
            income - 12.0 * annuity, income
        ),
        "DEBT_SERVICE_COVERAGE": safe_ratio(income / 12.0, annuity),
        "CREDIT_TERM_ROUND_1": safe_ratio(credit, annuity + 1.0).round(1),
        "CREDIT_TO_GOODS_ROUND_2": safe_ratio(credit, goods_filled + 1.0).round(2),
    }

    ext = np.column_stack(
        [get("EXT_SOURCE_1"), get("EXT_SOURCE_2"), get("EXT_SOURCE_3")]
    )
    ext_count = np.isfinite(ext).sum(axis=1)
    ext_sum = np.where(np.isfinite(ext), ext, 0.0).sum(axis=1)
    ext_mean = np.divide(
        ext_sum,
        ext_count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_min = np.where(np.isfinite(ext), ext, np.inf).min(axis=1)
    ext_max = np.where(np.isfinite(ext), ext, -np.inf).max(axis=1)
    ext_min[ext_count == 0] = np.nan
    ext_max[ext_count == 0] = np.nan
    ext_variance = np.divide(
        np.where(np.isfinite(ext), (ext - ext_mean[:, None]) ** 2, 0.0).sum(
            axis=1
        ),
        ext_count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_std = np.sqrt(ext_variance)
    ext_product = np.where(np.isfinite(ext), ext, 1.0).prod(axis=1)
    ext_product[ext_count == 0] = np.nan
    derived.update(
        {
            "EXT_MEAN": ext_mean,
            "EXT_MIN": ext_min,
            "EXT_MAX": ext_max,
            "EXT_STD": np.where(ext_count == 0, 0.0, ext_std),
            "EXT_RANGE": ext_max - ext_min,
            "EXT_COUNT": ext_count.astype(np.float32),
            "EXT_PRODUCT": ext_product,
        }
    )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        derived[f"EXT_DIFF_{left}{right}"] = ext[:, left] - ext[:, right]
        derived[f"EXT_PRODUCT_{left}{right}"] = ext[:, left] * ext[:, right]
        derived[f"EXT_RATIO_{left}{right}"] = safe_ratio(
            ext[:, left], ext[:, right]
        )

    term_years = np.clip(
        safe_ratio(credit, annuity + 1.0) / 12.0, 0.1, 10.0
    )
    disposable_ratio = np.maximum(
        0.01, safe_ratio(income / 12.0, annuity + 1.0)
    )
    for mu, sigma, suffix in ((0.02, 0.20, "20"), (0.03, 0.25, "25")):
        distance = (
            np.log(disposable_ratio)
            + (mu - 0.5 * sigma**2) * term_years
        ) / (sigma * np.sqrt(term_years))
        derived[f"MERTON_DISTANCE_{suffix}"] = np.clip(
            distance, -10.0, 10.0
        )
        derived[f"MERTON_PD_{suffix}"] = ndtr(-distance)

    bureau_columns = [
        column
        for column in raw.columns
        if column.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    bureau = raw[bureau_columns]
    derived["BUREAU_INQUIRY_SUM"] = bureau.sum(axis=1, min_count=1).to_numpy()
    derived["INCOME_LOG1P"] = np.log1p(np.maximum(income, 0.0))
    derived["CREDIT_LOG1P"] = np.log1p(np.maximum(credit, 0.0))
    derived["ANNUITY_LOG1P"] = np.log1p(np.maximum(annuity, 0.0))

    derived_frame = pd.DataFrame(derived)
    if np.isinf(derived_frame.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Feature engineering produced infinite values")
    features = pd.concat([features, derived_frame], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise TypeError(f"Non-numeric derived feature: {column}")
    return pd.concat([features, categorical], axis=1).reset_index(drop=True)


def make_lightgbm_frames(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        column
        for column in train_features.columns
        if column not in CATEGORICAL_COLUMNS
    ]
    train_parts = [train_features[numeric_columns].astype(np.float32)]
    test_parts = [test_features[numeric_columns].astype(np.float32)]
    for column in CATEGORICAL_COLUMNS:
        levels = sorted(train_features[column].astype(str).unique().tolist())
        train_cat = pd.Categorical(
            train_features[column].astype(str), categories=levels
        )
        test_cat = pd.Categorical(
            test_features[column].astype(str), categories=levels
        )
        category_columns = [
            f"CAT_{column}_{index}" for index in range(len(levels))
        ]
        train_dummy = pd.get_dummies(train_cat, dtype=np.float32)
        test_dummy = pd.get_dummies(test_cat, dtype=np.float32)
        train_dummy.columns = category_columns
        test_dummy.columns = category_columns
        train_parts.append(train_dummy)
        test_parts.append(test_dummy)
    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1).reindex(
        columns=X_train.columns, fill_value=0.0
    )
    if not X_train.columns.is_unique or list(X_train.columns) != list(X_test.columns):
        raise ValueError("LightGBM train/test feature schemas differ")
    if any(not column.replace("_", "").isalnum() for column in X_train.columns):
        raise ValueError("LightGBM feature names must be alphanumeric with underscores")
    if np.isinf(X_train.to_numpy(dtype=np.float64, copy=False)).any() or np.isinf(
        X_test.to_numpy(dtype=np.float64, copy=False)
    ).any():
        raise ValueError("LightGBM features contain infinite values")
    return X_train, X_test


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {"available": True, "query": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def compute_fold_fingerprint(base: dict[str, Any], model_name: str, fold: int) -> str:
    return canonical_hash(
        {
            "base": base,
            "model": MODEL_CONFIG[model_name],
            "model_name": model_name,
            "fold": fold,
        }
    )


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def train_lightgbm_fold(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, int, float, pd.DataFrame]:
    settings = MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in settings.items()
        if key
        not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = FOLD_SEED + fold
    train_set = lgb.Dataset(
        X_train.iloc[train_indices],
        label=y[train_indices],
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        X_train.iloc[valid_indices],
        label=y[valid_indices],
        reference=train_set,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=settings["num_boost_round"],
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(
                settings["early_stopping_rounds"], verbose=False
            )
        ],
    )
    valid_prediction = model.predict(
        X_train.iloc[valid_indices], num_iteration=model.best_iteration
    )
    test_prediction = model.predict(X_test, num_iteration=model.best_iteration)
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
            "fold": fold,
        }
    )
    fold_auc = roc_auc_score(y[valid_indices], valid_prediction)
    return (
        valid_prediction.astype(np.float32),
        test_prediction.astype(np.float32),
        int(model.best_iteration),
        float(fold_auc),
        importance,
    )


def train_catboost_fold(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, int, float, pd.DataFrame]:
    settings = MODEL_CONFIG["catboost"]
    model = CatBoostClassifier(
        iterations=settings["iterations"],
        depth=settings["depth"],
        learning_rate=settings["learning_rate"],
        loss_function=settings["loss_function"],
        eval_metric=settings["eval_metric"],
        l2_leaf_reg=settings["l2_leaf_reg"],
        random_strength=settings["random_strength"],
        bootstrap_type=settings["bootstrap_type"],
        subsample=settings["subsample"],
        border_count=settings["border_count"],
        task_type=settings["task_type"],
        devices=settings["devices"],
        random_seed=FOLD_SEED + fold,
        allow_writing_files=False,
        verbose=False,
        thread_count=CPU_THREADS,
    )
    model.fit(
        X_train.iloc[train_indices],
        y[train_indices],
        cat_features=list(CATEGORICAL_COLUMNS),
        eval_set=(X_train.iloc[valid_indices], y[valid_indices]),
        early_stopping_rounds=settings["early_stopping_rounds"],
        verbose=False,
    )
    valid_prediction = model.predict_proba(X_train.iloc[valid_indices])[:, 1]
    test_prediction = model.predict_proba(X_test)[:, 1]
    fold_auc = roc_auc_score(y[valid_indices], valid_prediction)
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
            "fold": fold,
        }
    )
    return (
        valid_prediction.astype(np.float32),
        test_prediction.astype(np.float32),
        int(model.best_iteration_),
        float(fold_auc),
        importance,
    )


def model_metrics_summary(results: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {name: value for name, value in item.items() if name not in ("oof", "test")}
            if isinstance(item, dict)
            else item
        )
        for key, item in results.items()
    }


def load_fold_checkpoint(
    path: Path, fingerprint: str, validation_indices: np.ndarray
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as checkpoint:
        if str(checkpoint["fingerprint"].item()) != fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
        if not np.array_equal(checkpoint["valid_indices"], validation_indices):
            raise ValueError(f"Checkpoint fold assignment mismatch: {path}")
        return {
            "valid_indices": checkpoint["valid_indices"],
            "oof_prediction": checkpoint["oof_prediction"],
            "test_prediction": checkpoint["test_prediction"],
            "best_iteration": int(checkpoint["best_iteration"]),
            "fold_auc": float(checkpoint["fold_auc"]),
            "feature_names": checkpoint["feature_names"].astype(str),
            "feature_importance": checkpoint["feature_importance"],
        }


def fit_model(
    run_dir: Path,
    model_name: str,
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    base_fingerprint_data: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    model_dir = run_dir / "checkpoints" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    test_fold_predictions = []
    fold_results = []
    importances = []

    for fold_number, (train_indices, valid_indices) in enumerate(folds, 1):
        checkpoint_path = model_dir / f"fold_{fold_number}.npz"
        fingerprint = compute_fold_fingerprint(
            base_fingerprint_data, model_name, fold_number
        )
        start = time.perf_counter()
        if resume and checkpoint_path.is_file():
            result = load_fold_checkpoint(
                checkpoint_path, fingerprint, valid_indices
            )
            status = "RESUMED"
        else:
            print(
                f"[{model_name}] fold {fold_number}/{len(folds)} "
                f"device={MODEL_CONFIG[model_name]['device']} "
                f"train={len(train_indices)} valid={len(valid_indices)}",
                flush=True,
            )
            if model_name == "lightgbm":
                valid_prediction, test_prediction, best_iteration, fold_auc, imp = (
                    train_lightgbm_fold(
                        X_train,
                        y,
                        X_test,
                        train_indices,
                        valid_indices,
                        fold_number,
                    )
                )
                feature_names = imp["feature"].astype(str).to_numpy()
                feature_importance = imp["gain"].to_numpy(dtype=np.float64)
            else:
                (
                    valid_prediction,
                    test_prediction,
                    best_iteration,
                    fold_auc,
                    imp,
                ) = train_catboost_fold(
                    X_train,
                    y,
                    X_test,
                    train_indices,
                    valid_indices,
                    fold_number,
                )
                feature_names = imp["feature"].astype(str).to_numpy()
                feature_importance = imp["importance"].to_numpy(dtype=np.float64)
            result = {
                "valid_indices": valid_indices,
                "oof_prediction": valid_prediction,
                "test_prediction": test_prediction,
                "best_iteration": best_iteration,
                "fold_auc": fold_auc,
                "feature_names": feature_names,
                "feature_importance": feature_importance,
            }
            atomic_save_npz(
                checkpoint_path,
                fingerprint=np.asarray(fingerprint),
                valid_indices=valid_indices.astype(np.int32),
                oof_prediction=valid_prediction,
                test_prediction=test_prediction,
                best_iteration=np.asarray(best_iteration, dtype=np.int32),
                fold_auc=np.asarray(fold_auc, dtype=np.float64),
                feature_names=feature_names.astype(str),
                feature_importance=feature_importance,
            )
            status = "TRAINED"
            importances.append(imp)

        oof[result["valid_indices"]] = result["oof_prediction"]
        test_fold_predictions.append(result["test_prediction"])
        fold_results.append(
            {
                "fold": fold_number,
                "status": status,
                "best_iteration": result["best_iteration"],
                "auc": result["fold_auc"],
                "elapsed_seconds": time.perf_counter() - start,
            }
        )

    if np.isnan(oof).any():
        raise ValueError(f"{model_name} OOF predictions have uncovered rows")
    test_prediction = np.mean(
        np.stack(test_fold_predictions, axis=0), axis=0, dtype=np.float64
    ).astype(np.float32)

    importance_folds = {
        int(frame["fold"].iloc[0])
        for frame in importances
        if not frame.empty
    }
    for fold_number in range(1, len(folds) + 1):
        checkpoint_path = model_dir / f"fold_{fold_number}.npz"
        if fold_number not in importance_folds:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                feature_names = checkpoint["feature_names"].astype(str)
                values = checkpoint["feature_importance"].astype(np.float64)
            importances.append(
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": values,
                        "fold": fold_number,
                    }
                )
            )
    importance_frame = pd.concat(importances, ignore_index=True)
    if "gain" not in importance_frame.columns:
        importance_frame = importance_frame.rename(
            columns={"importance": "gain"}
        )
    importance_frame.to_csv(
        run_dir / f"{model_name}_feature_importance.csv", index=False
    )
    np.save(run_dir / f"oof_{model_name}.npy", oof)
    np.save(run_dir / f"test_{model_name}.npy", test_prediction)
    return {
        "oof": oof,
        "test": test_prediction,
        "folds": fold_results,
        "auc": float(roc_auc_score(y, oof)),
        "mean_fold_auc": float(np.mean([row["auc"] for row in fold_results])),
        "std_fold_auc": float(np.std([row["auc"] for row in fold_results])),
        "best_iterations": [row["best_iteration"] for row in fold_results],
        "feature_importance_path": str(
            run_dir / f"{model_name}_feature_importance.csv"
        ),
    }


def aligned_reference_metrics(
    candidate: pd.DataFrame, reference_path: Path
) -> dict[str, Any] | None:
    if not reference_path.is_file():
        return None
    reference = pd.read_csv(reference_path)
    required = {ID_COLUMN, TARGET_COLUMN}
    if set(reference.columns) != required:
        raise ValueError(
            f"Historical reference schema mismatch: {reference.columns.tolist()}"
        )
    joined = candidate.merge(
        reference.rename(columns={TARGET_COLUMN: "historical_TARGET"}),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["historical_TARGET"].isna().any():
        raise ValueError("Historical reference is missing candidate IDs")
    current = joined[TARGET_COLUMN].to_numpy(dtype=np.float64)
    historical = joined["historical_TARGET"].to_numpy(dtype=np.float64)
    rank_current = rankdata(current, method="ordinal")
    rank_historical = rankdata(historical, method="ordinal")
    rank_displacement = np.abs(rank_current - rank_historical)
    return {
        "classification": "historical reference comparison only; not a model input",
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "pearson": float(np.corrcoef(current, historical)[0, 1]),
        "spearman": float(spearmanr(current, historical).statistic),
        "mae": float(np.mean(np.abs(current - historical))),
        "rmse": float(np.sqrt(np.mean((current - historical) ** 2))),
        "mean_rank_displacement": float(np.mean(rank_displacement)),
        "maximum_rank_displacement": int(np.max(rank_displacement)),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required raw competition input missing: {path}")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Run directory is not empty; use a new timestamped directory or --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    start_utc = datetime.now(timezone.utc).isoformat()
    train_hash = sha256_file(train_path)
    test_hash = sha256_file(test_path)
    sample_hash = sha256_file(sample_path)
    source_hash = sha256_file(Path(__file__).resolve())
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.SubprocessError as error:
        git_commit = f"unavailable: {type(error).__name__}: {error}"

    import lightgbm
    import catboost

    fingerprint_data = {
        "raw_train_sha256": train_hash,
        "raw_test_sha256": test_hash,
        "sample_submission_sha256": sample_hash,
        "source_sha256": source_hash,
        "git_commit": git_commit,
        "folds": {"n_splits": N_SPLITS, "shuffle": True, "seed": FOLD_SEED},
        "models": MODEL_CONFIG,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
        "experiment": "raw-only inductive CatBoost-GPU and LightGBM-CPU 5-fold baseline",
    }
    fingerprint = canonical_hash(fingerprint_data)
    config_path = run_dir / "config.json"
    if args.resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("fingerprint") != fingerprint:
            raise ValueError("Cannot resume: run configuration/input fingerprint changed")
    else:
        config_path.write_text(
            json.dumps(
                {
                    **fingerprint_data,
                    "fingerprint": fingerprint,
                    "command": sys.argv,
                    "start_utc": start_utc,
                    "raw_data_used_for_model_training": [
                        str(train_path),
                        str(test_path),
                    ],
                    "historical_prediction_inputs": [],
                    "transductive_feature_inputs": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not gpu_snapshot()["available"]:
        raise RuntimeError(
            "nvidia-smi did not report an NVIDIA GPU; GPU CatBoost run is required"
        )
    gpu_before = gpu_snapshot()
    print(f"GPU before training: {gpu_before['query']}", flush=True)

    data_start = time.perf_counter()
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if list(train_raw.columns).count(TARGET_COLUMN) != 1:
        raise ValueError("Training data must contain exactly one TARGET column")
    if TARGET_COLUMN in test_raw:
        raise ValueError("TARGET is unexpectedly present in raw test input")
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")
    if not np.array_equal(sample[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()):
        raise ValueError("Raw test IDs/order do not match sample submission")
    if train_raw[ID_COLUMN].duplicated().any() or test_raw[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate raw train/test IDs")
    if set(train_raw[ID_COLUMN]).intersection(set(test_raw[ID_COLUMN])):
        raise ValueError("Train and test IDs overlap")

    y = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Target values must be binary 0/1")
    X_train_raw = train_raw.drop(columns=[TARGET_COLUMN])
    X_train = build_rowwise_features(X_train_raw)
    X_test = build_rowwise_features(test_raw)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature schema mismatch between train and test")
    if ID_COLUMN in X_train or TARGET_COLUMN in X_train:
        raise ValueError("ID/target leakage detected in features")
    X_lgb_train, X_lgb_test = make_lightgbm_frames(X_train, X_test)
    print(
        f"Raw inputs loaded; rowwise features={X_train.shape[1]} "
        f"LightGBM columns={X_lgb_train.shape[1]} "
        f"feature_seconds={time.perf_counter() - data_start:.2f}",
        flush=True,
    )

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED
        ).split(np.zeros(len(y)), y)
    )
    for train_indices, valid_indices in folds:
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Train/validation fold indices overlap")
        if len(np.unique(y[valid_indices])) != 2:
            raise ValueError("Validation fold is missing a target class")

    model_results = {}
    model_results["lightgbm"] = fit_model(
        run_dir,
        "lightgbm",
        X_lgb_train,
        y,
        X_lgb_test,
        folds,
        fingerprint_data,
        args.resume,
    )
    model_results["catboost"] = fit_model(
        run_dir,
        "catboost",
        X_train,
        y,
        X_test,
        folds,
        fingerprint_data,
        args.resume,
    )

    oof_lgb = model_results["lightgbm"]["oof"]
    oof_cat = model_results["catboost"]["oof"]
    test_lgb = model_results["lightgbm"]["test"]
    test_cat = model_results["catboost"]["test"]
    oof_blend = (oof_lgb.astype(np.float64) + oof_cat.astype(np.float64)) / 2.0
    test_blend = (test_lgb.astype(np.float64) + test_cat.astype(np.float64)) / 2.0
    for name, predictions in (("lightgbm", oof_lgb), ("catboost", oof_cat)):
        model_results[name]["auc"] = float(roc_auc_score(y, predictions))
    model_results["oof_pearson_lgbm_catboost"] = float(
        np.corrcoef(oof_lgb, oof_cat)[0, 1]
    )
    model_results["oof_spearman_lgbm_catboost"] = float(
        spearmanr(oof_lgb, oof_cat).statistic
    )
    model_results["fixed_50_50_probability_blend"] = {
        "auc": float(roc_auc_score(y, oof_blend)),
        "oof_sha256": None,
        "test_sha256": None,
        "selection_note": "predeclared equal probability blend; no weight tuning",
    }

    candidate = pd.DataFrame(
        {
            ID_COLUMN: test_raw[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: test_blend,
        }
    )
    if candidate.shape[0] != len(sample):
        raise ValueError("Candidate row count differs from sample submission")
    if candidate[ID_COLUMN].duplicated().any():
        raise ValueError("Candidate contains duplicate IDs")
    if not np.array_equal(candidate[ID_COLUMN], sample[ID_COLUMN]):
        raise ValueError("Candidate IDs/order differ from sample submission")
    if not np.isfinite(candidate[TARGET_COLUMN]).all():
        raise ValueError("Candidate predictions contain NaN/Inf")
    if (candidate[TARGET_COLUMN] < 0.0).any() or (
        candidate[TARGET_COLUMN] > 1.0
    ).any():
        raise ValueError("Candidate predictions fall outside [0, 1]")
    submission_path = run_dir / "submission_independent_baseline.csv"
    candidate.to_csv(submission_path, index=False, float_format="%.10f")

    oof_blend_path = run_dir / "oof_fixed_50_50_probability_blend.npy"
    test_blend_path = run_dir / "test_fixed_50_50_probability_blend.npy"
    np.save(oof_blend_path, oof_blend)
    np.save(test_blend_path, test_blend)
    model_results["fixed_50_50_probability_blend"]["oof_sha256"] = sha256_file(
        oof_blend_path
    )
    model_results["fixed_50_50_probability_blend"]["test_sha256"] = sha256_file(
        test_blend_path
    )
    fold_metrics = []
    for index in range(N_SPLITS):
        fold_metrics.append(
            {
                "fold": index + 1,
                "lightgbm_auc": model_results["lightgbm"]["folds"][index]["auc"],
                "catboost_auc": model_results["catboost"]["folds"][index]["auc"],
                "fixed_blend_auc": float(
                    roc_auc_score(
                        y[folds[index][1]], oof_blend[folds[index][1]]
                    )
                ),
            }
        )
    model_results["fold_metrics"] = fold_metrics
    model_results["prediction_correlation"] = {
        "pearson": model_results["oof_pearson_lgbm_catboost"],
        "spearman": model_results["oof_spearman_lgbm_catboost"],
    }
    model_results["historical_reference_comparison"] = aligned_reference_metrics(
        candidate, HISTORICAL_REFERENCE
    )
    model_results["official_competition_score"] = "NOT_EVALUATED"
    model_results["training_lineage"] = "INDEPENDENT_RAW_ONLY_NO_HISTORICAL_PREDICTIONS"
    model_results["test_prediction_summary"] = {
        "mean": float(np.mean(test_blend)),
        "std": float(np.std(test_blend)),
        "min": float(np.min(test_blend)),
        "max": float(np.max(test_blend)),
        "positive_rate_at_0_5": float(np.mean(test_blend >= 0.5)),
    }
    gpu_after = gpu_snapshot()
    model_results["gpu_before"] = gpu_before
    model_results["gpu_after"] = gpu_after
    metrics_summary = model_metrics_summary(model_results)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    prediction_paths = [
        run_dir / "oof_lightgbm.npy",
        run_dir / "test_lightgbm.npy",
        run_dir / "oof_catboost.npy",
        run_dir / "test_catboost.npy",
        run_dir / "oof_fixed_50_50_probability_blend.npy",
        run_dir / "test_fixed_50_50_probability_blend.npy",
        submission_path,
        metrics_path,
    ]
    outputs = {
        str(path.relative_to(run_dir)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in prediction_paths
    }
    manifest = {
        "experiment_fingerprint": fingerprint,
        "raw_inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": "raw competition input",
            }
            for path in (train_path, test_path, sample_path)
        },
        "source": {
            str(Path(__file__).resolve()): {
                "sha256": source_hash,
                "role": "experiment implementation",
            }
        },
        "historical_prediction_inputs": [],
        "generated_outputs": outputs,
    }
    manifest_path = run_dir / "dependency_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "COMPLETED_VALIDATED_CV_CANDIDATE",
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None,
        "exit_code": 0,
        "command": sys.argv,
        "experiment_fingerprint": fingerprint,
        "git_commit": git_commit,
        "training_lineage": "INDEPENDENT_RAW_ONLY_NO_HISTORICAL_PREDICTIONS",
        "official_competition_score": "NOT_EVALUATED",
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "metrics_path": str(metrics_path),
        "dependency_manifest_path": str(manifest_path),
        "candidate_path": str(submission_path),
        "candidate_sha256": sha256_file(submission_path),
        "metrics": metrics_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    args.run_dir = args.run_dir.resolve()
    try:
        result = execute(args)
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.run_dir / "execution.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "exit_code": result["exit_code"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "candidate_sha256": result["candidate_sha256"],
                    "metrics_path": result["metrics_path"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAILED",
            "start_utc": start_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "exit_code": 1,
            "command": sys.argv,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (args.run_dir / "execution.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        print(failure["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
