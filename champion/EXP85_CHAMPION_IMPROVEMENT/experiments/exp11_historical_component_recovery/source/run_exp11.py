#!/usr/bin/env python3
"""Strict, fold-safe raw-data recovery of EXP85's historical XGB10 stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
IMPROVEMENT_DIR = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = EXPERIMENT_DIR / "source"
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_BASELINE = (
    IMPROVEMENT_DIR
    / "experiments"
    / "exp10_pair_replication"
    / "runs"
    / "exp10_pair_seed20261102_final_verified"
    / "shared_baseline_contract.npz"
)
HISTORICAL_SOURCE = (
    SOURCE_DIR / "historical_xgb10_source.py"
)
EXPECTED_HISTORICAL_SOURCE_SHA256 = (
    "7fbeaa89fb7b4473aefa9dbdb2c83b6ee337389b1ffc783b442a1640f9c1c900"
)
HISTORICAL_CACHE = Path(r"D:\gci\champion\cache")
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp11_xgb10_seed20261102"

EXPECTED_INPUT_SHA256 = {
    "train.csv": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test.csv": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
}
EXPECTED_BASELINE_SHA256 = (
    "781e27bcd96422083c8fac554f0bdd2ff78e4b7dd5e9be57217192a39358a1ab"
)
FOLD_SEED = 20261102
MODEL_SEED = 20261103
FOLD_COUNT = 5
MAX_ESTIMATORS = 450
EARLY_STOPPING_ROUNDS = 30
MINIMUM_POOLED_DELTA = 0.001
MINIMUM_POSITIVE_FOLDS = 4
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def fit_partition_one_hot(
    fit: pd.DataFrame, apply: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unseen = "__EXP11_UNSEEN_CATEGORY__"
    combined: dict[str, pd.Categorical] = {}
    for column in CATEGORICAL_COLUMNS:
        fit_values = fit[column].astype("category")
        categories = list(fit_values.cat.categories)
        require(unseen not in categories, f"Reserved category token is present: {column}")
        apply_values = apply[column].astype(object).copy()
        unknown = apply_values.notna() & ~apply_values.isin(categories)
        apply_values.loc[unknown] = unseen
        combined[column] = pd.Categorical(
            pd.concat(
                [
                    fit_values.astype(object).reset_index(drop=True),
                    apply_values.reset_index(drop=True),
                ],
                ignore_index=True,
            ),
            categories=categories + [unseen],
        )
    categorical_frame = pd.DataFrame(combined)
    dummy = pd.get_dummies(
        categorical_frame,
        dummy_na=True,
        dtype=np.float32,
    )
    dummy.columns = [
        f"OH_{index}_{column}".replace(":", "_")
        .replace("|", "_")
        .replace(" ", "_")
        .replace("-", "_")
        for index, column in enumerate(dummy.columns)
    ]
    return (
        dummy.iloc[: len(fit)].reset_index(drop=True),
        dummy.iloc[len(fit) :].reset_index(drop=True),
    )


def build_xgb_features(
    fit_rows: pd.DataFrame, apply_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical XGB10 features with fold-local categorical vocabularies."""
    fit = fit_rows.drop(columns=[TARGET_COLUMN], errors="ignore").reset_index(drop=True)
    apply = apply_rows.drop(columns=[TARGET_COLUMN], errors="ignore").reset_index(drop=True)
    require(
        tuple(fit.columns) == tuple(apply.columns),
        "Training and apply columns differ",
    )
    n_fit = len(fit)
    all_rows = pd.concat([fit, apply], ignore_index=True)

    emp_raw = all_rows["DAYS_EMPLOYED"].to_numpy(dtype=np.float64)
    days_emp = np.where(emp_raw == 365243.0, np.nan, emp_raw)
    income = all_rows["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64)
    credit = all_rows["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = all_rows["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = all_rows["AMT_GOODS_PRICE"].fillna(all_rows["AMT_CREDIT"]).to_numpy(
        dtype=np.float64
    )
    birth = all_rows["DAYS_BIRTH"].to_numpy(dtype=np.float64)
    registration = all_rows["DAYS_REGISTRATION"].to_numpy(dtype=np.float64)
    id_publish = all_rows["DAYS_ID_PUBLISH"].to_numpy(dtype=np.float64)
    phone = all_rows["DAYS_LAST_PHONE_CHANGE"].to_numpy(dtype=np.float64)
    ext1 = all_rows["EXT_SOURCE_1"].to_numpy(dtype=np.float64)
    ext2 = all_rows["EXT_SOURCE_2"].to_numpy(dtype=np.float64)
    ext3 = all_rows["EXT_SOURCE_3"].to_numpy(dtype=np.float64)

    ext_stack = np.column_stack([ext1, ext2, ext3])
    features: dict[str, np.ndarray] = {}
    features["DAYS_EMPLOYED_CLEAN"] = days_emp.astype(np.float32)
    features["IS_RETIRED"] = (emp_raw == 365243.0).astype(np.float32)
    features["MISS_CNT"] = all_rows.isna().sum(axis=1).to_numpy(dtype=np.float32)
    features["AGE_YEARS"] = (-birth / 365.25).astype(np.float32)
    features["EMPLOYED_YEARS"] = (-days_emp / 365.25).astype(np.float32)
    features["EMP_TO_AGE"] = (days_emp / (birth + 1.0)).astype(np.float32)
    features["REG_TO_AGE"] = (registration / (birth + 1.0)).astype(np.float32)
    features["ID_TO_AGE"] = (id_publish / (birth + 1.0)).astype(np.float32)
    features["PHONE_TO_AGE"] = (phone / (birth + 1.0)).astype(np.float32)

    term = credit / (annuity + 1.0)
    features["CREDIT_TERM"] = term.astype(np.float32)
    features["EAR"] = np.clip(
        24.0 * (annuity * term - credit) / (credit * (term + 1.0) + 1.0),
        -1.0,
        5.0,
    ).astype(np.float32)
    surplus = income - 12.0 * annuity
    features["SURPLUS"] = surplus.astype(np.float32)
    features["SURPLUS_TO_INC"] = (surplus / (income + 1.0)).astype(np.float32)
    features["TRUE_COST"] = (annuity * term - credit).astype(np.float32)
    features["TRUE_COST_RATIO"] = (
        (annuity * term - credit) / (credit + 1.0)
    ).astype(np.float32)
    features["LTV"] = np.clip(credit / (goods + 1.0), 0.0, 5.0).astype(np.float32)
    features["DTI"] = np.clip(credit / (income + 1.0), 0.0, 20.0).astype(np.float32)
    features["DEBT_BURDEN"] = np.clip(
        annuity / (income / 12.0 + 1.0), 0.0, 5.0
    ).astype(np.float32)
    features["DSCR"] = np.clip(
        ((income / 12.0) * 0.65) / (annuity + 1.0), 0.0, 20.0
    ).astype(np.float32)
    features["BUFFER_MONTHS"] = np.clip(
        surplus / (12.0 * annuity + 1.0), -10.0, 50.0
    ).astype(np.float32)
    features["INC_PER_FAM"] = (
        income / (all_rows["CNT_FAM_MEMBERS"].fillna(1.0) + 1e-3)
    ).to_numpy(dtype=np.float32)

    features["EXT_MEAN"] = np.nanmean(ext_stack, axis=1).astype(np.float32)
    features["EXT_MIN"] = np.nanmin(ext_stack, axis=1).astype(np.float32)
    features["EXT_MAX"] = np.nanmax(ext_stack, axis=1).astype(np.float32)
    features["EXT_STD"] = np.nan_to_num(
        np.nanstd(ext_stack, axis=1), nan=0.0
    ).astype(np.float32)
    features["EXT_W"] = (
        np.nan_to_num(0.2 * ext1)
        + np.nan_to_num(0.4 * ext2)
        + np.nan_to_num(0.4 * ext3)
    ).astype(np.float32)
    features["EXT_DIFF_12"] = (ext1 - ext2).astype(np.float32)
    features["EXT_DIFF_23"] = (ext2 - ext3).astype(np.float32)
    features["EXT_PROD"] = (
        np.nan_to_num(ext1, nan=0.5)
        * np.nan_to_num(ext2, nan=0.5)
        * np.nan_to_num(ext3, nan=0.5)
    ).astype(np.float32)
    features["LOG_INCOME"] = np.log1p(income).astype(np.float32)
    features["LOG_CREDIT"] = np.log1p(credit).astype(np.float32)
    features["LOG_ANNUITY"] = np.log1p(annuity).astype(np.float32)
    term_years = np.clip(term / 12.0, 0.1, 10.0)
    disposable_ratio = np.maximum(0.01, (income / 12.0) / (annuity + 1.0))
    for mean_return, volatility, tag in (
        (0.02, 0.20, "20"),
        (0.03, 0.25, "25"),
    ):
        distance = (
            np.log(disposable_ratio)
            + (mean_return - 0.5 * volatility**2) * term_years
        ) / (volatility * np.sqrt(term_years))
        features[f"MERTON_DD_{tag}"] = np.clip(distance, -10.0, 10.0).astype(
            np.float32
        )

    feature_frame = pd.DataFrame(features)
    fit_dummy, apply_dummy = fit_partition_one_hot(fit, apply)
    dummy = pd.concat([fit_dummy, apply_dummy], ignore_index=True)
    numeric_columns = all_rows.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column not in (ID_COLUMN, TARGET_COLUMN) and column not in feature_frame:
            feature_frame[column] = all_rows[column].to_numpy(dtype=np.float64)
    feature_frame = pd.concat([feature_frame, dummy], axis=1)
    feature_frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    clean_columns = [
        column.replace(":", "_")
        .replace("|", "_")
        .replace(" ", "_")
        .replace("-", "_")
        for column in feature_frame.columns
    ]
    feature_frame.columns = clean_columns
    fit_features = feature_frame.iloc[:n_fit].reset_index(drop=True)
    apply_features = feature_frame.iloc[n_fit:].reset_index(drop=True)
    require(
        tuple(fit_features.columns) == tuple(apply_features.columns),
        "Feature column order differs between fit and apply",
    )
    return fit_features, apply_features


def model_parameters(seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": 0.035,
        "max_depth": 6,
        "subsample": 0.80,
        "colsample_bytree": 0.55,
        "reg_alpha": 3.0,
        "reg_lambda": 8.0,
        "min_child_weight": 30,
        "tree_method": "hist",
        "device": "cuda:0",
        "eval_metric": "auc",
        "random_state": int(seed),
        "n_jobs": 4,
    }


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    first = result.stdout.strip().splitlines()[0].split(",")
    require(len(first) == 5, "Unexpected nvidia-smi response")
    return {
        "name": first[0].strip(),
        "driver_version": first[1].strip(),
        "total_memory_mib": int(first[2].strip()),
        "memory_used_mib": int(first[3].strip()),
        "utilization_percent": int(first[4].strip()),
    }


class GpuMonitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.samples.append(gpu_snapshot())
            except Exception as error:  # noqa: BLE001 - record telemetry failure
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop_event.wait(1.0)

    def __enter__(self) -> "GpuMonitor":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)


def telemetry_summary(
    before: dict[str, Any], monitor: GpuMonitor, elapsed_seconds: float
) -> dict[str, Any]:
    require(not monitor.errors, f"GPU telemetry failed: {monitor.errors}")
    require(monitor.samples, "GPU telemetry collected no samples")
    return {
        "device": before["name"],
        "driver_version": before["driver_version"],
        "total_memory_mib": before["total_memory_mib"],
        "memory_used_mib_before": before["memory_used_mib"],
        "maximum_memory_used_mib": max(
            sample["memory_used_mib"] for sample in monitor.samples
        ),
        "maximum_utilization_percent": max(
            sample["utilization_percent"] for sample in monitor.samples
        ),
        "sample_count": len(monitor.samples),
        "training_runtime_seconds": elapsed_seconds,
        "monitor_errors": monitor.errors,
    }


def source_file_record(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "repository_path": path.resolve()
        .relative_to(repository_root.resolve())
        .as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_contract(
    baseline_path: Path, train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, np.ndarray]:
    require(baseline_path.is_file(), f"Shared baseline is missing: {baseline_path}")
    require(
        sha256_file(baseline_path) == EXPECTED_BASELINE_SHA256,
        "Shared baseline contract hash differs from the verified EXP10 artifact",
    )
    with np.load(baseline_path, allow_pickle=False) as archive:
        required = {
            "train_ids",
            "test_ids",
            "labels",
            "fold_membership",
            "phase2_oof",
            "phase2_test",
        }
        require(required.issubset(archive.files), "Baseline contract is incomplete")
        contract = {name: archive[name].copy() for name in sorted(required)}
    require(
        np.array_equal(contract["train_ids"], train[ID_COLUMN].to_numpy()),
        "Ordered training IDs do not match the shared baseline",
    )
    require(
        np.array_equal(contract["test_ids"], test[ID_COLUMN].to_numpy()),
        "Ordered test IDs do not match the shared baseline",
    )
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(np.array_equal(contract["labels"], labels), "Baseline labels differ")
    require(
        np.array_equal(np.unique(contract["fold_membership"]), np.arange(1, 6)),
        "Baseline must contain exactly five 1-indexed folds",
    )
    require(
        np.all(np.isfinite(contract["phase2_oof"]))
        and np.all((contract["phase2_oof"] >= 0) & (contract["phase2_oof"] <= 1)),
        "Shared Phase 2 OOF predictions are invalid",
    )
    return contract


def historical_cache_evidence(labels: np.ndarray) -> dict[str, Any]:
    oof_path = HISTORICAL_CACHE / "oof_10f_xgb.npy"
    test_path = HISTORICAL_CACHE / "test_10f_xgb.npy"
    if not oof_path.is_file() or not test_path.is_file():
        return {
            "classification": "UNAVAILABLE",
            "details": "Historical OOF/test cache files were not present at the inspected cache path.",
        }
    oof = np.load(oof_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    require(oof.shape == labels.shape, "Historical XGB10 OOF cache shape mismatch")
    require(test.shape == (61500,), "Historical XGB10 test cache shape mismatch")
    return {
        "classification": "HISTORICAL_CACHE_REPLAY_EVIDENCE",
        "oof_path": str(oof_path),
        "oof_sha256": sha256_file(oof_path),
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path),
        "pooled_oof_auc_recomputed": float(roc_auc_score(labels, oof)),
        "fold_membership_available": False,
        "independent_training": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    baseline_path = Path(args.baseline_contract).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    require(train_path.is_file() and test_path.is_file(), "Raw train/test CSV is missing")
    for path in (HISTORICAL_SOURCE, Path(__file__).resolve()):
        require(path.is_file(), f"Required source file is missing: {path}")
    require(
        sha256_file(HISTORICAL_SOURCE) == EXPECTED_HISTORICAL_SOURCE_SHA256,
        "Pinned historical XGB10 source snapshot changed",
    )
    require(
        xgb.__version__ == "3.4.1",
        f"Unexpected XGBoost version {xgb.__version__}; pinned run expects 3.4.1",
    )
    for path, label in ((train_path, "train.csv"), (test_path, "test.csv")):
        require(
            sha256_file(path) == EXPECTED_INPUT_SHA256[label],
            f"Raw input hash mismatch: {path}",
        )

    log(log_path, "Loading raw competition train/test data")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    require(
        train[ID_COLUMN].is_unique and test[ID_COLUMN].is_unique,
        "Training or test IDs are not unique",
    )
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    contract = load_contract(baseline_path, train, test)
    require(
        FOLD_SEED == 20261102,
        "The matched baseline fold seed is fixed and must not be changed",
    )
    gpu_before = gpu_snapshot()
    require(
        "RTX 4050" in gpu_before["name"],
        f"Expected RTX 4050, found {gpu_before['name']}",
    )

    probe = xgb.XGBClassifier(**model_parameters(MODEL_SEED, 2))
    probe.fit(
        np.asarray([[0.0], [1.0], [0.5], [1.5]], dtype=np.float32),
        np.asarray([0, 1, 0, 1], dtype=np.int8),
        verbose=False,
    )
    probe_config = json.loads(probe.get_booster().save_config())
    probe_device = probe_config["learner"]["generic_param"]["device"]
    require(
        probe_device.startswith("cuda"),
        f"XGBoost CUDA preflight fell back from GPU: {probe_device}",
    )
    del probe

    fold_membership = contract["fold_membership"].astype(np.int8)
    oof = np.full(len(train), np.nan, dtype=np.float32)
    test_by_fold = np.zeros((FOLD_COUNT, len(test)), dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    started = time.monotonic()
    with GpuMonitor() as monitor:
        for fold in range(1, FOLD_COUNT + 1):
            fold_started = time.monotonic()
            outer_train = np.flatnonzero(fold_membership != fold)
            outer_valid = np.flatnonzero(fold_membership == fold)
            require(
                not np.intersect1d(outer_train, outer_valid).size,
                f"Outer train/validation overlap in fold {fold}",
            )
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.10,
                random_state=MODEL_SEED + 1000 * fold,
            )
            inner_fit_local, inner_stop_local = next(
                inner.split(np.zeros(len(outer_train)), labels[outer_train])
            )
            inner_fit = outer_train[inner_fit_local]
            inner_stop = outer_train[inner_stop_local]
            require(
                not np.intersect1d(outer_valid, np.concatenate([inner_fit, inner_stop])).size,
                f"Outer validation labels entered inner fitting in fold {fold}",
            )

            log(
                log_path,
                f"Fold {fold}/{FOLD_COUNT}: building inner-fit and inner-stop features",
            )
            x_inner_fit, x_inner_stop = build_xgb_features(
                train.iloc[inner_fit],
                train.iloc[inner_stop],
            )
            selector = xgb.XGBClassifier(
                **model_parameters(MODEL_SEED + fold, MAX_ESTIMATORS),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            selector.fit(
                x_inner_fit,
                labels[inner_fit],
                eval_set=[(x_inner_stop, labels[inner_stop])],
                verbose=False,
            )
            selector_config = json.loads(selector.get_booster().save_config())
            selector_device = selector_config["learner"]["generic_param"]["device"]
            require(
                selector_device.startswith("cuda"),
                f"Fold {fold} inner model did not use CUDA: {selector_device}",
            )
            best_iteration = int(selector.best_iteration)
            best_rounds = max(1, best_iteration + 1)
            del selector, selector_config

            log(
                log_path,
                f"Fold {fold}/{FOLD_COUNT}: refitting {best_rounds} rounds on all outer-train rows",
            )
            x_outer_train, x_outer_apply = build_xgb_features(
                train.iloc[outer_train],
                pd.concat(
                    [train.iloc[outer_valid], test],
                    ignore_index=True,
                ),
            )
            x_outer_valid = x_outer_apply.iloc[: len(outer_valid)].reset_index(drop=True)
            x_test = x_outer_apply.iloc[len(outer_valid) :].reset_index(drop=True)
            final_model = xgb.XGBClassifier(
                **model_parameters(MODEL_SEED + fold, best_rounds)
            )
            final_model.fit(
                x_outer_train,
                labels[outer_train],
                verbose=False,
            )
            final_config = json.loads(final_model.get_booster().save_config())
            final_device = final_config["learner"]["generic_param"]["device"]
            require(
                final_device.startswith("cuda"),
                f"Fold {fold} final model did not use CUDA: {final_device}",
            )
            valid_prediction = final_model.predict_proba(x_outer_valid)[:, 1]
            test_prediction = final_model.predict_proba(x_test)[:, 1]
            require(
                np.isfinite(valid_prediction).all()
                and np.isfinite(test_prediction).all()
                and np.all((valid_prediction >= 0) & (valid_prediction <= 1))
                and np.all((test_prediction >= 0) & (test_prediction <= 1)),
                f"Fold {fold} produced invalid probabilities",
            )
            oof[outer_valid] = valid_prediction.astype(np.float32)
            test_by_fold[fold - 1] = test_prediction.astype(np.float32)
            model_path = run_dir / f"fold_{fold}_xgb.ubj"
            final_model.save_model(model_path)
            index_path = run_dir / f"fold_{fold}_indices.npz"
            np.savez_compressed(
                index_path,
                outer_train=outer_train.astype(np.int32),
                outer_valid=outer_valid.astype(np.int32),
                inner_fit=inner_fit.astype(np.int32),
                inner_stop=inner_stop.astype(np.int32),
            )
            fold_auc = float(roc_auc_score(labels[outer_valid], valid_prediction))
            baseline_auc = float(
                roc_auc_score(
                    labels[outer_valid],
                    contract["phase2_oof"][outer_valid],
                )
            )
            fold_record = {
                "fold": fold,
                "outer_train_rows": len(outer_train),
                "outer_validation_rows": len(outer_valid),
                "inner_fit_rows": len(inner_fit),
                "inner_stop_rows": len(inner_stop),
                "early_stopping_best_iteration": best_iteration,
                "final_rounds": best_rounds,
                "feature_count": int(x_outer_train.shape[1]),
                "candidate_auc": fold_auc,
                "phase2_auc": baseline_auc,
                "delta_vs_phase2": fold_auc - baseline_auc,
                "positive_fold": bool(fold_auc > baseline_auc),
                "training_runtime_seconds": time.monotonic() - fold_started,
                "model_file": model_path.name,
                "indices_file": index_path.name,
                "xgboost_device": final_device,
            }
            fold_records.append(fold_record)
            write_json(run_dir / f"fold_{fold}_metrics.json", fold_record)
            log(
                log_path,
                f"Fold {fold}: candidate AUC={fold_auc:.8f}, "
                f"Phase2 AUC={baseline_auc:.8f}, rounds={best_rounds}",
            )
            del final_model, final_config, x_outer_train, x_outer_apply
            del x_outer_valid, x_test
    runtime = time.monotonic() - started

    require(np.isfinite(oof).all(), "Some OOF rows were not predicted")
    test_prediction = np.mean(test_by_fold, axis=0, dtype=np.float64).astype(np.float32)
    require(
        test_prediction.shape == (len(test),)
        and np.isfinite(test_prediction).all()
        and np.all((test_prediction >= 0) & (test_prediction <= 1)),
        "Aggregated test predictions are invalid",
    )
    phase2_auc = float(roc_auc_score(labels, contract["phase2_oof"]))
    candidate_auc = float(roc_auc_score(labels, oof))
    delta = candidate_auc - phase2_auc
    positive_folds = sum(record["positive_fold"] for record in fold_records)
    gate_passed = (
        delta >= MINIMUM_POOLED_DELTA
        and positive_folds >= MINIMUM_POSITIVE_FOLDS
    )
    residual_phase2 = labels.astype(np.float64) - contract["phase2_oof"]
    residual_candidate = labels.astype(np.float64) - oof
    corr = np.corrcoef(oof, contract["phase2_oof"])[0, 1]
    residual_corr = np.corrcoef(residual_phase2, residual_candidate)[0, 1]
    absolute_error_corr = np.corrcoef(
        np.abs(residual_phase2), np.abs(residual_candidate)
    )[0, 1]
    ext_missing = train[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].isna().sum(axis=1).to_numpy()
    subgroup_metrics = []
    for missing_count in range(4):
        subset = ext_missing == missing_count
        if not subset.any():
            continue
        subgroup_metrics.append(
            {
                "ext_source_missing_count": missing_count,
                "rows": int(subset.sum()),
                "positive_labels": int(labels[subset].sum()),
                "phase2_auc": float(
                    roc_auc_score(labels[subset], contract["phase2_oof"][subset])
                ),
                "candidate_auc": float(roc_auc_score(labels[subset], oof[subset])),
            }
        )

    np.save(run_dir / "oof_xgb10_strict.npy", oof)
    np.save(run_dir / "test_xgb10_strict.npy", test_prediction)
    np.save(run_dir / "test_xgb10_by_fold.npy", test_by_fold)
    np.save(run_dir / "fold_membership.npy", fold_membership)
    np.savez_compressed(
        run_dir / "ordered_ids_labels.npz",
        train_ids=contract["train_ids"],
        test_ids=contract["test_ids"],
        labels=labels,
    )
    np.savez_compressed(
        run_dir / "shared_baseline_contract.npz",
        train_ids=contract["train_ids"],
        test_ids=contract["test_ids"],
        labels=contract["labels"],
        fold_membership=contract["fold_membership"],
        phase2_oof=contract["phase2_oof"],
        phase2_test=contract["phase2_test"],
    )
    gpu = telemetry_summary(gpu_before, monitor, runtime)
    with (run_dir / "gpu_telemetry.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "name",
                "driver_version",
                "total_memory_mib",
                "memory_used_mib",
                "utilization_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(monitor.samples)

    hist = historical_cache_evidence(labels)
    lineage = {
        "selected_component": "EXP85 8-engine suite member xgb10",
        "historical_component_lineage_table": [
            {
                "component": "Anchor 74 precision transductive anchor",
                "historical_role": "85% of final EXP85 probit-space synthesis",
                "source_and_entrypoint": [
                    "champion/04_MODEL_LINEAGE/ANCHOR74/generate_anchor74_from_scratch.py",
                    "D:/gci/gemini/build_precision_transductive_apex_champion.py",
                ],
                "inputs_and_transformations": (
                    "Cached 0.80084 transductive submission and five-model "
                    "hasan_expanded_zoo test streams; regime discrepancy gate, "
                    "test-pair smoothing, bounded rank shift"
                ),
                "historical_test_artifact": (
                    "D:/gci/gemini/v_20261027_precision_transductive_apex.csv"
                ),
                "raw_data_training_reproducible": False,
                "independent_leakage_safe_oof_available": False,
                "classification": "exact historical test artifact/cache synthesis; not independent training",
                "blocking_dependencies": [
                    "No OOF counterpart for the 0.80084 transductive anchor",
                    "Named TabM/DAE prediction archives are missing",
                ],
            },
            {
                "component": "V17 full-suite bundle",
                "historical_role": "V17 members collectively contribute to the 15% suite; e1Aet reported at 29.32% of that suite",
                "source_and_entrypoint": [
                    "champion/04_MODEL_LINEAGE/V17/generate_v17_from_scratch.py",
                    "D:/gci/gemini/train_full_v16_suite.py",
                ],
                "inputs_and_transformations": (
                    "Eight cached V16 view matrices, residual features, fold "
                    "encodings, and LightGBM/CatBoost member streams"
                ),
                "historical_prediction_artifact": (
                    "D:/gci/gemini/cache/v_17_full_suite_bundle.npz"
                ),
                "raw_data_training_reproducible": "partial; source and raw builders exist, but the historical entrypoint consumes cached views/residuals/encodings",
                "independent_leakage_safe_oof_available": False,
                "classification": "historical cache replay; raw rebuild requires further provenance validation",
                "blocking_dependencies": [
                    "Fold provenance is not embedded in the prediction bundle",
                    "Cached engineered views and residual/encoding artifacts are upstream dependencies",
                ],
            },
            {
                "component": "V55 suite members",
                "historical_role": "V55 members collectively contribute to the 15% suite",
                "source_and_entrypoint": [
                    "champion/04_MODEL_LINEAGE/V55/generate_v55_from_scratch.py",
                    "D:/gci/gemini/train_clean_honors_champion.py",
                ],
                "inputs_and_transformations": (
                    "Cached V16 feature views plus residual, hierarchical, WOE, "
                    "KNN, reconstructed-external-source, domain, and contract-risk blocks"
                ),
                "historical_prediction_artifact": "D:/gci/gemini/cache/v55_members.npz",
                "raw_data_training_reproducible": "partial; training source exists but consumes derived feature caches",
                "independent_leakage_safe_oof_available": False,
                "classification": "historical cache replay; not selected for fresh validation",
                "blocking_dependencies": [
                    "Fold-local provenance for the cached feature blocks is incomplete",
                    "Prediction bundle contains no fold assignments",
                ],
            },
            {
                "component": "XGB10 suite member",
                "historical_role": "25.96% of the historical 8-engine suite",
                "source_and_entrypoint": [
                    "exp11_historical_component_recovery/source/historical_xgb10_source.py",
                    "exp11_historical_component_recovery/source/run_exp11.py",
                ],
                "inputs_and_transformations": (
                    "Raw train/test CSV; financial ratios, demographics, external "
                    "source aggregates, Merton features, one-hot demographics, raw numeric columns"
                ),
                "historical_prediction_artifact": [
                    "D:/gci/champion/cache/oof_10f_xgb.npy",
                    "D:/gci/champion/cache/test_10f_xgb.npy",
                ],
                "training_configuration": (
                    "Hist-XGBoost, 450 trees maximum, learning_rate 0.035, "
                    "max_depth 6, subsample 0.80, colsample_bytree 0.55, "
                    "reg_alpha 3, reg_lambda 8, min_child_weight 30; historical "
                    "seed 42 and 10 folds"
                ),
                "raw_data_training_reproducible": True,
                "independent_leakage_safe_oof_available": "EXP11 refit; historical cache has no fold index",
                "classification": "fresh raw-data training in EXP11; historical cache separately identified",
                "blocking_dependencies": [],
            },
            {
                "component": "TabM/DAE and group-A neural streams",
                "historical_role": "Named historical neural prediction streams",
                "source_and_entrypoint": "Exact source/training entrypoint not recovered",
                "inputs_and_transformations": "Unavailable for the named streams",
                "historical_prediction_artifact": [
                    "group_a_oof_10stream.npy",
                    "group_a_test_10stream.npy",
                    "oof_and_test_tabm32_463_dae.npz",
                    "oof_and_test_lgbm_dae491.npz",
                ],
                "raw_data_training_reproducible": False,
                "independent_leakage_safe_oof_available": False,
                "classification": "missing exact artifacts; no substitution",
                "blocking_dependencies": [
                    "Exact named prediction archives and their configurations are unavailable",
                    "dae_pt_v1.npz contains embeddings only, with no encoder/training provenance",
                ],
            },
        ],
        "historical_role": {
            "suite_weight": 0.2596,
            "reported_historical_oof_auc": 0.774331,
            "reported_historical_feature_count": 239,
            "measured_historical_source_feature_count": 173,
            "lineage_discrepancy": (
                "The available full-data generator emits 173 features, while lineage "
                "documentation claims 239. EXP11 pins the available generator and "
                "does not silently invent the missing 66 features."
            ),
            "historical_folds": 10,
            "historical_fold_seed": 42,
        },
        "original_training_entrypoint": source_file_record(
            HISTORICAL_SOURCE, REPOSITORY_ROOT
        ),
        "historical_cache": hist,
        "classification": {
            "historical_cache": "historical cache replay evidence, not fresh training",
            "exp11_predictions": "independent raw-data training, five matched OOF folds",
            "missing_neural_streams": [
                {
                    "artifact": "group_a_oof_10stream.npy",
                    "status": "NOT_FOUND",
                    "meaning": "No exact historical OOF file available for independent evaluation.",
                },
                {
                    "artifact": "group_a_test_10stream.npy",
                    "status": "NOT_FOUND",
                    "meaning": "No exact historical test prediction file available.",
                },
                {
                    "artifact": "oof_and_test_tabm32_463_dae.npz",
                    "status": "NOT_FOUND",
                    "meaning": "No exact TabM/DAE prediction archive available.",
                },
                {
                    "artifact": "oof_and_test_lgbm_dae491.npz",
                    "status": "NOT_FOUND",
                    "meaning": "No exact LightGBM/DAE prediction archive available.",
                },
                {
                    "artifact": "dae_pt_v1.npz",
                    "status": "PRESENT_EMBEDDINGS_ONLY",
                    "meaning": "Contains E_tr/E_te embeddings, not the named OOF/test streams or encoder provenance.",
                },
            ],
        },
        "preprocessing": {
            "rowwise_features": "XGB10 financial feature and raw numeric recipe from the pinned source",
            "historical_feature_count_mismatch": "Available source emits 173 full-data features; lineage documentation claims 239",
            "categorical_mapping": "fit-partition one-hot levels plus explicit unseen and missing indicators",
            "early_stopping": "inner 10% stratified split of outer-training rows; refit all outer-training rows for selected rounds",
        },
    }
    write_json(run_dir / "component_lineage.json", lineage)

    summary = {
        "experiment_id": "exp11_xgb10_seed20261102",
        "decision": (
            "OOF_THRESHOLD_PASS_PENDING_INDEPENDENT_VERIFICATION"
            if gate_passed
            else "REJECTED"
        ),
        "official_leaderboard_score": None,
        "official_score_claim": False,
        "candidate_submission_created": False,
        "fold_seed": FOLD_SEED,
        "model_seed": MODEL_SEED,
        "folds": FOLD_COUNT,
        "xgboost_version": xgb.__version__,
        "feature_count_by_fold": [
            record["feature_count"] for record in fold_records
        ],
        "historical_full_data_feature_count": 173,
        "model_configuration": {
            "historical_fixed_parameters": {
                "learning_rate": 0.035,
                "max_depth": 6,
                "subsample": 0.80,
                "colsample_bytree": 0.55,
                "reg_alpha": 3.0,
                "reg_lambda": 8.0,
                "min_child_weight": 30,
                "max_estimators": MAX_ESTIMATORS,
                "tree_method": "hist",
                "device": "cuda:0",
                "eval_metric": "auc",
            },
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "inner_stop_fraction": 0.10,
            "fold_model_seeds": [MODEL_SEED + fold for fold in range(1, 6)],
        },
        "baseline": {
            "source": "EXP10 independently verified shared Phase 2 contract",
            "sha256": EXPECTED_BASELINE_SHA256,
            "fold_seed": FOLD_SEED,
            "pooled_oof_auc": phase2_auc,
            "oof_sha256": sha256_array(contract["phase2_oof"]),
            "fold_membership_sha256": sha256_array(fold_membership),
        },
        "metrics": {
            "phase2_pooled_oof_auc": phase2_auc,
            "xgb10_strict_pooled_oof_auc": candidate_auc,
            "absolute_delta_vs_phase2": delta,
            "positive_folds": positive_folds,
            "fold_metrics": fold_records,
            "prediction_correlation": float(corr),
            "residual_correlation": float(residual_corr),
            "absolute_error_correlation": float(absolute_error_corr),
            "ext_source_missingness_subgroups": subgroup_metrics,
            "residual_distribution": {
                "phase2_mae": float(np.mean(np.abs(residual_phase2))),
                "candidate_mae": float(np.mean(np.abs(residual_candidate))),
                "phase2_mean_signed_error": float(np.mean(residual_phase2)),
                "candidate_mean_signed_error": float(np.mean(residual_candidate)),
            },
        },
        "advancement_gate": {
            "minimum_pooled_auc_delta": MINIMUM_POOLED_DELTA,
            "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
            "pooled_auc_delta": delta,
            "positive_folds": positive_folds,
            "passed": gate_passed,
            "full_gate_pending_independent_model_replay": gate_passed,
        },
        "integrity": {
            "raw_inputs_hash_verified": True,
            "ordered_train_ids_sha256": sha256_array(contract["train_ids"]),
            "ordered_test_ids_sha256": sha256_array(contract["test_ids"]),
            "labels_sha256": sha256_array(labels),
            "fold_membership_sha256": sha256_array(fold_membership),
            "baseline_contract_sha256": EXPECTED_BASELINE_SHA256,
            "oof_prediction_sha256": sha256_array(oof),
            "test_prediction_sha256": sha256_array(test_prediction),
            "all_outer_validation_rows_predicted_once": True,
            "outer_validation_used_for_early_stopping": False,
            "shared_baseline_exactly_aligned": True,
        },
        "gpu_evidence": gpu,
        "runtime_seconds": runtime,
        "component_lineage": "component_lineage.json",
        "historical_component_cache_reference": hist,
        "source_hashes": {
            "exp11_runner_sha256": sha256_file(Path(__file__).resolve()),
            "historical_xgb10_source_sha256": sha256_file(HISTORICAL_SOURCE),
        },
    }
    write_json(run_dir / "experiment_summary.json", summary)

    lines = [
        "# EXP11 — Historical XGB10 Component Recovery",
        "",
        f"- Decision: **{summary['decision']}**.",
        "- This is an OOF research result, not an official leaderboard score.",
        "- No submission was created and the frozen historical champion was not modified.",
        "",
        "## Lineage and method",
        "",
        "- Selected stream: EXP85 `xgb10`, historically weighted 25.96% within its 8-engine suite.",
        "- Historical reference: cached OOF AUC recomputes to 0.77433064; it is provenance only and its fold IDs are unavailable.",
        "- EXP11: raw-data XGBoost refit using the available historical generator and five matched folds.",
        "- Lineage discrepancy: the available source emits 173 full-data features, although documentation claims 239; per-fold feature counts are recorded in the summary.",
        "- Fixed historical tree parameters; fold-local categorical mappings and nested early stopping prevent outer-validation contamination.",
        "- RTX 4050 CUDA was required; CPU fallback is rejected.",
        "",
        "## Pooled OOF comparison",
        "",
        "| Stream | Pooled OOF AUC | Delta vs Phase 2 |",
        "|---|---:|---:|",
        f"| Shared Phase 2 | {phase2_auc:.8f} | +0.00000000 |",
        f"| Strict XGB10 refit | {candidate_auc:.8f} | {delta:+.8f} |",
        "",
        "## Per-fold metrics",
        "",
        "| Fold | Phase 2 AUC | XGB10 AUC | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for item in fold_records:
        lines.append(
            f"| {item['fold']} | {item['phase2_auc']:.8f} | "
            f"{item['candidate_auc']:.8f} | {item['delta_vs_phase2']:+.8f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Prediction correlation with Phase 2: {corr:.6f}.",
            f"- Residual correlation: {residual_corr:.6f}.",
            f"- Absolute-error correlation: {absolute_error_corr:.6f}.",
            f"- Phase 2 / XGB10 MAE: {np.mean(np.abs(residual_phase2)):.6f} / {np.mean(np.abs(residual_candidate)):.6f}.",
            "- EXT_SOURCE missingness subgroup AUCs are recorded in `experiment_summary.json`.",
            f"- RTX 4050 peak utilization / VRAM: {gpu['maximum_utilization_percent']}% / {gpu['maximum_memory_used_mib']} MiB; runtime {runtime:.1f}s.",
            "",
            "## Advancement decision",
            "",
            f"- Required gain: +{MINIMUM_POOLED_DELTA:.6f}; observed: {delta:+.8f}.",
            f"- Required positive folds: {MINIMUM_POSITIVE_FOLDS}/5; observed: {positive_folds}/5.",
            f"- OOF threshold gate: **{'PASS' if gate_passed else 'FAIL'}**; the final research gate also requires independent saved-model replay.",
            "- See `experiment_summary.json`, `component_lineage.json`, `SHA256_MANIFEST.json`, and `VERIFICATION.json` for machine-readable evidence.",
            "",
            "EXP11 does not establish an official public score or prove the preserved 0.801340 champion has been beaten.",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    write_fold_metrics(run_dir / "FOLD_METRICS.csv", fold_records)
    log(
        log_path,
        f"Completed: pooled AUC={candidate_auc:.8f}, "
        f"delta={delta:+.8f}, gate={'PASS' if gate_passed else 'FAIL'}",
    )
    make_manifest(run_dir, data_dir, baseline_path, HISTORICAL_SOURCE)
    return summary


def write_fold_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "fold",
        "outer_train_rows",
        "outer_validation_rows",
        "inner_fit_rows",
        "inner_stop_rows",
        "early_stopping_best_iteration",
        "final_rounds",
        "phase2_auc",
        "candidate_auc",
        "delta_vs_phase2",
        "positive_fold",
        "training_runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in fields} for record in records)


def make_manifest(
    run_dir: Path,
    data_dir: Path,
    baseline_path: Path,
    historical_source: Path,
) -> None:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256_MANIFEST.json":
            continue
        records[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "experiment_id": "exp11_xgb10_seed20261102",
        "inputs": {
            "train.csv": {
                "path": str((data_dir / "train.csv").resolve()),
                "sha256": sha256_file(data_dir / "train.csv"),
                "size_bytes": (data_dir / "train.csv").stat().st_size,
            },
            "test.csv": {
                "path": str((data_dir / "test.csv").resolve()),
                "sha256": sha256_file(data_dir / "test.csv"),
                "size_bytes": (data_dir / "test.csv").stat().st_size,
            },
            "shared_baseline_contract.npz": {
                "path": str(baseline_path.resolve()),
                "sha256": sha256_file(baseline_path),
                "size_bytes": baseline_path.stat().st_size,
            },
        },
        "sources": {
            "exp11_runner.py": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "size_bytes": Path(__file__).stat().st_size,
            },
            "historical_xgb10_source.py": {
                "path": str(historical_source.resolve()),
                "sha256": sha256_file(historical_source),
                "size_bytes": historical_source.stat().st_size,
            },
        },
        "outputs": records,
    }
    write_json(run_dir / "SHA256_MANIFEST.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--baseline-contract", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["advancement_gate"], indent=2))


if __name__ == "__main__":
    main()
