#!/usr/bin/env python3
"""Run a new-seed validation of Phase 2 and fold-local fee frequencies."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
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
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
PHASE_01_DIR = PROJECT_DIR / "experiments" / "phase_01_fee_product_features"
PHASE_02_DIR = PROJECT_DIR / "experiments" / "phase_02_fold_safe_key_encodings"
sys.path.insert(0, str(SOURCE_DIR))

from run_independent_baseline import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    MODEL_CONFIG,
    TARGET_COLUMN,
    ID_COLUMN,
    build_rowwise_features,
    gpu_snapshot,
    make_lightgbm_frames,
    sha256_file,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase01 = load_module(
    "phase01_for_final_gate",
    PHASE_01_DIR / "run_experiment.py",
)
phase02 = load_module(
    "phase02_for_final_gate",
    PHASE_02_DIR / "run_experiment.py",
)

DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "final_gate_seed20261101"
FEE_AMOUNT = phase01.FEE_AMOUNT
PRODUCT_RATE = phase01.PRODUCT_RATE
NEW_OUTER_SEED = 20261101
INNER_SEED = phase02.INNER_SEED
N_SPLITS = phase02.N_SPLITS
FEATURE_VARIANTS = ("raw_baseline", "phase2_pooled", "phase2_fold_local")
LOCAL_COUNT_COLUMNS = (
    "FEE_DECODED",
    "FEE_RAW_RATIO_COUNT",
    "FEE_ADJUSTED_RATIO_COUNT",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def ratio_keys(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    raw_ratio = np.full(len(raw), np.nan, dtype=np.float64)
    fee_ratio = np.full(len(raw), np.nan, dtype=np.float64)
    valid_credit = np.isfinite(credit) & (credit != 0.0)
    raw_ratio[valid_credit] = annuity[valid_credit] / credit[valid_credit]
    fee_ratio[valid_credit] = (
        annuity[valid_credit] - FEE_AMOUNT
    ) / credit[valid_credit]
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").to_numpy()
    return raw_ratio, fee_ratio, contract


def build_decoded_features(
    count_reference_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply frequency-derived fee decoding using reference covariates only."""
    reference = count_reference_raw.drop(columns=[TARGET_COLUMN], errors="ignore")
    apply = apply_raw.drop(columns=[TARGET_COLUMN], errors="ignore")
    reference_raw_ratio, reference_fee_ratio, _ = ratio_keys(reference)
    apply_raw_ratio, apply_fee_ratio, apply_contract = ratio_keys(apply)

    reference_raw_keys = pd.Series(reference_raw_ratio).round(5)
    reference_fee_keys = pd.Series(reference_fee_ratio).round(5)
    raw_count_map = reference_raw_keys.value_counts(dropna=True)
    fee_count_map = reference_fee_keys.value_counts(dropna=True)
    apply_raw_keys = pd.Series(apply_raw_ratio).round(5)
    apply_fee_keys = pd.Series(apply_fee_ratio).round(5)
    raw_counts = (
        apply_raw_keys.map(raw_count_map).fillna(0).to_numpy(dtype=np.float64)
    )
    fee_counts = (
        apply_fee_keys.map(fee_count_map).fillna(0).to_numpy(dtype=np.float64)
    )
    fee_decoded = (
        (apply_contract == "Cash loans")
        & (fee_counts > raw_counts)
        & np.isfinite(apply_raw_ratio)
    )
    decoded_ratio = np.where(fee_decoded, apply_fee_ratio, apply_raw_ratio)

    credit = apply["AMT_CREDIT"].to_numpy(dtype=np.float64)
    goods = apply["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    goods_valid = np.isfinite(goods) & (goods != 0.0)
    credit_goods_ratio = np.full(len(apply), np.nan, dtype=np.float64)
    credit_goods_ratio[goods_valid] = credit[goods_valid] / goods[goods_valid]
    implied_product = (credit_goods_ratio - 1.0) / PRODUCT_RATE
    nearest_product = np.rint(implied_product)
    product_valid = (
        (apply_contract == "Cash loans")
        & np.isfinite(implied_product)
        & (credit > goods)
        & (np.abs(implied_product - nearest_product) < 0.1)
    )

    features = pd.DataFrame(
        {
            "FEE_DECODED": fee_decoded.astype(np.float32),
            "FEE_RAW_RATIO_COUNT": np.log1p(raw_counts).astype(np.float32),
            "FEE_ADJUSTED_RATIO_COUNT": np.log1p(fee_counts).astype(np.float32),
            "ANNUITY_CREDIT_RATIO_FEE_ADJUSTED": apply_fee_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED": decoded_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3": np.round(decoded_ratio, 3),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4": np.round(decoded_ratio, 4),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5": np.round(decoded_ratio, 5),
            "CREDIT_GOODS_RATIO": credit_goods_ratio,
            "PRODUCT_KEY_VALID": product_valid.astype(np.float32),
            "PRODUCT_KEY_K": np.where(product_valid, nearest_product, np.nan),
            "PRODUCT_KEY_DISTANCE": np.where(
                product_valid, implied_product - nearest_product, np.nan
            ),
        }
    )
    require(
        not np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any(),
        "Decoded features contain infinity",
    )
    require(ID_COLUMN not in features and TARGET_COLUMN not in features, "ID/target leakage")
    diagnostics = {
        "count_reference_rows": int(len(reference)),
        "apply_rows": int(len(apply)),
        "count_reference_id_sha256": hashlib.sha256(
            np.asarray(reference[ID_COLUMN]).tobytes()
        ).hexdigest(),
        "reference_contains_validation_or_test_rows": False,
        "target_used": False,
        "fee_decoded_rows": int(fee_decoded.sum()),
        "product_key_valid_rows": int(product_valid.sum()),
    }
    return features, diagnostics


def make_pooled_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference = pd.concat(
        [
            train_raw.drop(columns=[TARGET_COLUMN]),
            test_raw,
        ],
        ignore_index=True,
    )
    combined_features, diagnostics = build_decoded_features(reference, reference)
    return (
        combined_features.iloc[: len(train_raw)].reset_index(drop=True),
        combined_features.iloc[len(train_raw) :].reset_index(drop=True),
        diagnostics,
    )


def make_fold_local_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference = train_raw.iloc[fit_indices].drop(
        columns=[TARGET_COLUMN], errors="ignore"
    )
    fit_features, fit_diagnostics = build_decoded_features(
        reference, reference
    )
    valid_features, valid_diagnostics = build_decoded_features(
        reference,
        train_raw.iloc[valid_indices].drop(
            columns=[TARGET_COLUMN], errors="ignore"
        ),
    )
    test_features, test_diagnostics = build_decoded_features(reference, test_raw)
    full_train = pd.DataFrame(
        index=np.arange(len(train_raw)),
        columns=fit_features.columns,
        dtype=np.float64,
    )
    full_train.iloc[fit_indices] = fit_features.to_numpy()
    full_train.iloc[valid_indices] = valid_features.to_numpy()
    diagnostics = {
        "fit": fit_diagnostics,
        "validation": valid_diagnostics,
        "test": test_diagnostics,
        "fit_validation_ids_disjoint": not bool(
            np.intersect1d(
                train_raw.iloc[fit_indices][ID_COLUMN].to_numpy(),
                train_raw.iloc[valid_indices][ID_COLUMN].to_numpy(),
            ).size
        ),
        "test_ids_used_in_frequency_reference": False,
    }
    require(
        diagnostics["fit_validation_ids_disjoint"],
        "Fold-local frequency reference overlaps validation rows",
    )
    return (
        full_train,
        test_features.reset_index(drop=True),
        diagnostics,
    )


class GpuMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                timestamp = datetime.now(timezone.utc).isoformat()
                for line in result.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) != 4:
                        raise ValueError(
                            f"Unexpected nvidia-smi sample: {line!r}"
                        )
                    self.samples.append(
                        {
                            "timestamp_utc": timestamp,
                            "index": int(fields[0]),
                            "name": fields[1],
                            "memory_used_mib": int(fields[2]),
                            "utilization_percent": int(fields[3]),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12)


def train_lightgbm_fold(
    X_train: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold: int,
    outer_seed: int,
) -> tuple[np.ndarray, int, float]:
    settings = copy.deepcopy(MODEL_CONFIG["lightgbm"])
    rounds = settings.pop("num_boost_round")
    early_stopping = settings.pop("early_stopping_rounds")
    settings.pop("device")
    settings["seed"] = outer_seed + fold
    settings["feature_fraction_seed"] = outer_seed + fold
    settings["bagging_seed"] = outer_seed + fold
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
        settings,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    prediction = model.predict(
        X_train.iloc[valid_indices],
        num_iteration=model.best_iteration,
    ).astype(np.float32)
    return (
        prediction,
        int(model.best_iteration),
        float(roc_auc_score(y[valid_indices], prediction)),
    )


def train_catboost_fold(
    X_train: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold: int,
    outer_seed: int,
) -> tuple[np.ndarray, int, float]:
    settings = copy.deepcopy(MODEL_CONFIG["catboost"])
    early_stopping = settings.pop("early_stopping_rounds")
    settings.update(
        {
            "random_seed": outer_seed + fold,
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    model = CatBoostClassifier(**settings)
    with GpuMonitor() as monitor:
        model.fit(
            X_train.iloc[train_indices],
            y[train_indices],
            cat_features=list(CATEGORICAL_COLUMNS),
            eval_set=(X_train.iloc[valid_indices], y[valid_indices]),
            early_stopping_rounds=early_stopping,
            verbose=False,
        )
    prediction = model.predict_proba(
        X_train.iloc[valid_indices]
    )[:, 1].astype(np.float32)
    require(
        not monitor.errors,
        "GPU utilization monitor had errors: " + "; ".join(monitor.errors),
    )
    return (
        prediction,
        int(model.best_iteration_),
        float(roc_auc_score(y[valid_indices], prediction)),
    )


def target_encoding_fold(
    raw_features: pd.DataFrame,
    test_features: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    n_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    require(
        np.intersect1d(fit_indices, valid_indices).size == 0,
        "Outer training and validation indices overlap",
    )
    keys_by_name = phase02.key_arrays(
        raw_features,
        test_features,
    )
    train_te = phase02.nested_training_encodings(
        keys_by_name, labels, fit_indices
    )
    valid_te = phase02.apply_training_encodings(
        keys_by_name, labels, fit_indices, valid_indices
    )
    test_indices = np.arange(
        len(labels), len(labels) + n_test, dtype=np.int64
    )
    test_te = phase02.apply_training_encodings(
        keys_by_name, labels, fit_indices, test_indices
    )
    require(len(train_te) == len(fit_indices), "Incomplete inner OOF encodings")
    require(len(valid_te) == len(valid_indices), "Incomplete validation encodings")
    require(len(test_te) == n_test, "Incomplete test encodings")
    for frame in (train_te, valid_te, test_te):
        require(np.isfinite(frame.to_numpy()).all(), "Non-finite target encodings")
    audit = {
        "outer_fit_valid_overlap": 0,
        "fit_rows_encoded_by_inner_oof": len(train_te),
        "validation_encoded_from_outer_fit_only": True,
        "test_encoded_from_outer_fit_only": True,
        "apply_row_targets_used_by_mapping": False,
        "inner_splits": phase02.INNER_SPLITS,
        "inner_seed": INNER_SEED,
    }
    return train_te, valid_te, test_te, audit


def add_target_encodings(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_te: pd.DataFrame,
    valid_te: pd.DataFrame,
    test_te: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_features.copy()
    X_test = pd.concat([test_features, test_te], axis=1)
    X_train.loc[fit_indices, train_te.columns] = train_te.to_numpy()
    X_train.loc[valid_indices, valid_te.columns] = valid_te.to_numpy()
    require(
        ID_COLUMN not in X_train.columns
        and TARGET_COLUMN not in X_train.columns
        and ID_COLUMN not in X_test.columns
        and TARGET_COLUMN not in X_test.columns,
        "ID or target reached model feature matrix",
    )
    return X_train, X_test


def model_frames(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    rowwise_train: pd.DataFrame,
    rowwise_test: pd.DataFrame,
    fee_train: pd.DataFrame | None = None,
    fee_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fee_train is None or fee_test is None:
        cat_train = rowwise_train.copy()
        cat_test = rowwise_test.copy()
    else:
        cat_train = pd.concat(
            [rowwise_train.reset_index(drop=True), fee_train.reset_index(drop=True)],
            axis=1,
        )
        cat_test = pd.concat(
            [rowwise_test.reset_index(drop=True), fee_test.reset_index(drop=True)],
            axis=1,
        )
    for frame in (cat_train, cat_test):
        require(ID_COLUMN not in frame.columns, "ID leaked into model features")
        require(TARGET_COLUMN not in frame.columns, "Target leaked into model features")
        require(frame.columns.is_unique, "Duplicate feature columns")
    lgb_train, lgb_test = make_lightgbm_frames(cat_train, cat_test)
    return cat_train, cat_test, lgb_train, lgb_test


def make_variant_fold(
    variant: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    rowwise_train: pd.DataFrame,
    rowwise_test: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    pooled_fee_train: pd.DataFrame,
    pooled_fee_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if variant == "raw_baseline":
        cat_train, cat_test, lgb_train, lgb_test = model_frames(
            train_raw,
            test_raw,
            rowwise_train,
            rowwise_test,
        )
        return cat_train, cat_test, lgb_train, lgb_test, {}

    if variant == "phase2_pooled":
        fee_train = pooled_fee_train
        fee_test = pooled_fee_test
        frequency = {
            "mode": "pooled_train_test_unlabeled",
            "target_used": False,
            "validation_test_covariates_in_frequency_reference": True,
        }
    elif variant == "phase2_fold_local":
        fee_train, fee_test, frequency = make_fold_local_features(
            train_raw,
            test_raw,
            fit_indices,
            valid_indices,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    cat_train, cat_test, _, _ = model_frames(
        train_raw,
        test_raw,
        rowwise_train,
        rowwise_test,
        fee_train,
        fee_test,
    )
    train_te, valid_te, test_te, te_audit = target_encoding_fold(
        fee_train,
        fee_test,
        train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8),
        fit_indices,
        valid_indices,
        len(test_raw),
    )
    cat_train, cat_test = add_target_encodings(
        cat_train,
        cat_test,
        train_te,
        valid_te,
        test_te,
        fit_indices,
        valid_indices,
    )
    lgb_train, lgb_test = make_lightgbm_frames(cat_train, cat_test)
    return (
        cat_train,
        cat_test,
        lgb_train,
        lgb_test,
        {"frequency": frequency, "target_encoding": te_audit},
    )


def hash_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run(
    data_dir: Path,
    run_dir: Path,
    outer_seed: int = NEW_OUTER_SEED,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    if outer_seed == phase02.FOLD_SEED:
        raise ValueError("Final gate must use a new outer seed")
    input_paths = {
        name: data_dir / filename
        for name, filename in (
            ("train", "train.csv"),
            ("test", "test.csv"),
            ("sample", "sample_submission.csv"),
        )
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required raw input missing: {path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_raw = pd.read_csv(input_paths["train"])
    test_raw = pd.read_csv(input_paths["test"])
    sample = pd.read_csv(input_paths["sample"])
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Unexpected sample-submission schema",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()),
        "Test/sample IDs or ordering differ",
    )
    require(not train_raw[ID_COLUMN].duplicated().any(), "Duplicate train IDs")
    require(not test_raw[ID_COLUMN].duplicated().any(), "Duplicate test IDs")
    require(
        set(train_raw[ID_COLUMN]).isdisjoint(set(test_raw[ID_COLUMN])),
        "Train/test ID sets overlap",
    )
    labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(set(np.unique(labels)).issubset({0, 1}), "Unexpected target values")

    rowwise_train = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    rowwise_test = build_rowwise_features(test_raw)
    require(
        ID_COLUMN not in rowwise_train and TARGET_COLUMN not in rowwise_train,
        "ID/target in rowwise training features",
    )
    require(
        list(rowwise_train.columns) == list(rowwise_test.columns),
        "Raw baseline train/test feature schema mismatch",
    )
    pooled_fee_train, pooled_fee_test, pooled_diagnostics = make_pooled_features(
        train_raw, test_raw
    )
    require(
        list(pooled_fee_train.columns) == list(pooled_fee_test.columns),
        "Pooled fee feature schema mismatch",
    )

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=outer_seed,
        ).split(np.zeros(len(labels)), labels)
    )
    fold_assignments = {}
    predictions = {
        variant: {
            "lightgbm": np.full(len(labels), np.nan, dtype=np.float32),
            "catboost": np.full(len(labels), np.nan, dtype=np.float32),
        }
        for variant in FEATURE_VARIANTS
    }
    test_fold_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        variant: {"lightgbm": [], "catboost": []}
        for variant in FEATURE_VARIANTS
    }
    model_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        variant: {"lightgbm": [], "catboost": []}
        for variant in FEATURE_VARIANTS
    }
    fold_records = []
    gpu_before = gpu_snapshot()

    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        require(
            np.intersect1d(fit_indices, valid_indices).size == 0,
            f"Fold {fold_number} fit/validation overlap",
        )
        fold_assignments[f"fit_{fold_number}"] = fit_indices.astype(np.int32)
        fold_assignments[f"valid_{fold_number}"] = valid_indices.astype(np.int32)
        current_fold_record: dict[str, Any] = {
            "fold": fold_number,
            "fit_rows": int(len(fit_indices)),
            "valid_rows": int(len(valid_indices)),
            "fit_validation_ids_disjoint": True,
            "variants": {},
        }
        for variant in FEATURE_VARIANTS:
            variant_started = time.perf_counter()
            (
                cat_train,
                cat_test,
                lgb_train,
                lgb_test,
                variant_audit,
            ) = make_variant_fold(
                variant,
                train_raw,
                test_raw,
                rowwise_train,
                rowwise_test,
                fit_indices,
                valid_indices,
                pooled_fee_train,
                pooled_fee_test,
            )
            require(
                list(lgb_train.columns) == list(lgb_test.columns),
                f"{variant} LightGBM train/test schema mismatch",
            )
            for frame in (cat_train, cat_test, lgb_train, lgb_test):
                numeric = frame.select_dtypes(include=[np.number])
                require(
                    not np.isinf(
                        numeric.to_numpy(dtype=np.float64, copy=False)
                    ).any(),
                    f"{variant} contains infinite values",
                )

            lgb_settings = copy.deepcopy(MODEL_CONFIG["lightgbm"])
            lgb_rounds = lgb_settings.pop("num_boost_round")
            lgb_early_stopping = lgb_settings.pop("early_stopping_rounds")
            lgb_settings.pop("device")
            lgb_settings["seed"] = outer_seed + fold_number
            lgb_settings["feature_fraction_seed"] = outer_seed + fold_number
            lgb_settings["bagging_seed"] = outer_seed + fold_number
            lgb_fit = lgb.Dataset(
                lgb_train.iloc[fit_indices],
                label=labels[fit_indices],
                free_raw_data=True,
            )
            lgb_valid = lgb.Dataset(
                lgb_train.iloc[valid_indices],
                label=labels[valid_indices],
                reference=lgb_fit,
                free_raw_data=True,
            )
            lgb_started = time.perf_counter()
            lgb_model = lgb.train(
                lgb_settings,
                lgb_fit,
                num_boost_round=lgb_rounds,
                valid_sets=[lgb_valid],
                callbacks=[
                    lgb.early_stopping(lgb_early_stopping, verbose=False)
                ],
            )
            lgb_prediction = lgb_model.predict(
                lgb_train.iloc[valid_indices],
                num_iteration=lgb_model.best_iteration,
            ).astype(np.float32)
            lgb_test_prediction = lgb_model.predict(
                lgb_test,
                num_iteration=lgb_model.best_iteration,
            ).astype(np.float32)
            lgb_auc = float(
                roc_auc_score(labels[valid_indices], lgb_prediction)
            )
            predictions[variant]["lightgbm"][valid_indices] = lgb_prediction
            test_fold_predictions[variant]["lightgbm"].append(
                lgb_test_prediction
            )
            model_records[variant]["lightgbm"].append(
                {
                    "fold": fold_number,
                    "device": "CPU",
                    "seed": outer_seed + fold_number,
                    "best_iteration": int(lgb_model.best_iteration),
                    "auc": lgb_auc,
                    "elapsed_seconds": time.perf_counter() - lgb_started,
                }
            )
            del lgb_model, lgb_fit, lgb_valid
            gc.collect()

            cat_settings = copy.deepcopy(MODEL_CONFIG["catboost"])
            cat_early_stopping = cat_settings.pop("early_stopping_rounds")
            cat_settings.pop("device")
            cat_settings.update(
                {
                    "random_seed": outer_seed + fold_number,
                    "allow_writing_files": False,
                    "verbose": False,
                }
            )
            cat_model = CatBoostClassifier(**cat_settings)
            cat_started = time.perf_counter()
            with GpuMonitor() as monitor:
                cat_model.fit(
                    cat_train.iloc[fit_indices],
                    labels[fit_indices],
                    cat_features=list(CATEGORICAL_COLUMNS),
                    eval_set=(
                        cat_train.iloc[valid_indices],
                        labels[valid_indices],
                    ),
                    early_stopping_rounds=cat_early_stopping,
                    verbose=False,
                )
            require(
                not monitor.errors,
                "GPU monitoring failed during CatBoost fold "
                f"{fold_number}: {'; '.join(monitor.errors)}",
            )
            cat_prediction = cat_model.predict_proba(
                cat_train.iloc[valid_indices]
            )[:, 1].astype(np.float32)
            cat_test_prediction = cat_model.predict_proba(cat_test)[
                :, 1
            ].astype(np.float32)
            cat_auc = float(
                roc_auc_score(labels[valid_indices], cat_prediction)
            )
            predictions[variant]["catboost"][valid_indices] = cat_prediction
            test_fold_predictions[variant]["catboost"].append(
                cat_test_prediction
            )
            monitor_summary = {
                "samples": len(monitor.samples),
                "max_gpu_utilization_percent": max(
                    (
                        item["utilization_percent"]
                        for item in monitor.samples
                    ),
                    default=None,
                ),
                "max_gpu_memory_used_mib": max(
                    (
                        item["memory_used_mib"]
                        for item in monitor.samples
                    ),
                    default=None,
                ),
                "first_samples": monitor.samples[:3],
                "last_samples": monitor.samples[-3:],
                "errors": monitor.errors,
            }
            model_records[variant]["catboost"].append(
                {
                    "fold": fold_number,
                    "device": "GPU:0",
                    "seed": outer_seed + fold_number,
                    "best_iteration": int(cat_model.best_iteration_),
                    "auc": cat_auc,
                    "elapsed_seconds": time.perf_counter() - cat_started,
                    "gpu_monitor": monitor_summary,
                }
            )
            del cat_model
            gc.collect()
            current_fold_record["variants"][variant] = {
                "lightgbm_auc": lgb_auc,
                "catboost_auc": cat_auc,
                "blend_auc": float(
                    roc_auc_score(
                        labels[valid_indices],
                        (
                            lgb_prediction.astype(np.float64)
                            + cat_prediction.astype(np.float64)
                        )
                        / 2.0,
                    )
                ),
                "audit": variant_audit,
                "elapsed_seconds": time.perf_counter() - variant_started,
            }
            print(
                f"[{variant}] fold {fold_number}/{N_SPLITS} "
                f"auc_lgb={lgb_auc:.8f} auc_cat={cat_auc:.8f} "
                f"auc_blend={current_fold_record['variants'][variant]['blend_auc']:.8f} "
                f"gpu_samples={monitor_summary['samples']} "
                f"gpu_util_max={monitor_summary['max_gpu_utilization_percent']}",
                flush=True,
            )
        fold_records.append(current_fold_record)
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=fit_indices.astype(np.int32),
            valid_indices=valid_indices.astype(np.int32),
        )

    require(
        all(
            np.isfinite(predictions[v][m]).all()
            for v in FEATURE_VARIANTS
            for m in ("lightgbm", "catboost")
        ),
        "One or more OOF vectors are incomplete",
    )
    metric_records = {}
    candidate_records = {}
    for variant in FEATURE_VARIANTS:
        blend_oof = (
            predictions[variant]["lightgbm"].astype(np.float64)
            + predictions[variant]["catboost"].astype(np.float64)
        ) / 2.0
        test_by_model = {
            model_name: np.mean(
                np.stack(test_fold_predictions[variant][model_name]),
                axis=0,
                dtype=np.float64,
            )
            for model_name in ("lightgbm", "catboost")
        }
        blend_test = (
            test_by_model["lightgbm"] + test_by_model["catboost"]
        ) / 2.0
        require(np.isfinite(blend_test).all(), f"Non-finite {variant} test blend")
        require(
            ((blend_test >= 0.0) & (blend_test <= 1.0)).all(),
            f"{variant} test predictions out of range",
        )
        np.save(run_dir / f"oof_{variant}_lightgbm.npy", predictions[variant]["lightgbm"])
        np.save(run_dir / f"oof_{variant}_catboost.npy", predictions[variant]["catboost"])
        np.save(run_dir / f"oof_{variant}_blend.npy", blend_oof)
        np.save(run_dir / f"test_{variant}_lightgbm.npy", test_by_model["lightgbm"])
        np.save(run_dir / f"test_{variant}_catboost.npy", test_by_model["catboost"])
        np.save(run_dir / f"test_{variant}_blend.npy", blend_test)
        candidate = pd.DataFrame(
            {ID_COLUMN: test_raw[ID_COLUMN].to_numpy(), TARGET_COLUMN: blend_test}
        )
        require(
            np.array_equal(candidate[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
            f"{variant} candidate IDs differ from sample submission",
        )
        require(
            candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
            and candidate[ID_COLUMN].is_unique,
            f"{variant} candidate schema/IDs invalid",
        )
        candidate_path = run_dir / f"submission_{variant}.csv"
        candidate.to_csv(candidate_path, index=False, float_format="%.10f")
        parsed = pd.read_csv(candidate_path)
        require(
            np.array_equal(
                parsed[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()
            ),
            f"{variant} candidate ID order changed during serialization",
        )
        require(
            np.max(
                np.abs(
                    parsed[TARGET_COLUMN].to_numpy(dtype=np.float64)
                    - blend_test
                )
            )
            < 1e-9,
            f"{variant} candidate values do not match test blend",
        )
        fold_auc = []
        for _, valid_indices in folds:
            fold_auc.append(
                float(roc_auc_score(labels[valid_indices], blend_oof[valid_indices]))
            )
        metric_records[variant] = {
            "oof_blend_auc": float(roc_auc_score(labels, blend_oof)),
            "mean_fold_blend_auc": float(np.mean(fold_auc)),
            "fold_blend_auc": fold_auc,
            "oof_lightgbm_auc": float(
                roc_auc_score(labels, predictions[variant]["lightgbm"])
            ),
            "oof_catboost_auc": float(
                roc_auc_score(labels, predictions[variant]["catboost"])
            ),
        }
        candidate_records[variant] = {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "rows": int(len(candidate)),
            "columns": candidate.columns.tolist(),
            "unique_ids": int(candidate[ID_COLUMN].nunique()),
            "prediction_min": float(blend_test.min()),
            "prediction_max": float(blend_test.max()),
        }

    comparisons = {}
    baseline_folds = metric_records["raw_baseline"]["fold_blend_auc"]
    for variant in ("phase2_pooled", "phase2_fold_local"):
        candidate_folds = metric_records[variant]["fold_blend_auc"]
        fold_deltas = [
            candidate - baseline
            for candidate, baseline in zip(candidate_folds, baseline_folds)
        ]
        comparisons[variant] = {
            "full_oof_auc_delta_vs_raw_baseline": (
                metric_records[variant]["oof_blend_auc"]
                - metric_records["raw_baseline"]["oof_blend_auc"]
            ),
            "paired_fold_auc_deltas_vs_raw_baseline": fold_deltas,
            "positive_folds_vs_raw_baseline": sum(delta > 0 for delta in fold_deltas),
            "mean_paired_fold_delta_vs_raw_baseline": float(np.mean(fold_deltas)),
        }
    pooled_folds = metric_records["phase2_pooled"]["fold_blend_auc"]
    local_folds = metric_records["phase2_fold_local"]["fold_blend_auc"]
    local_vs_pooled = [local - pooled for local, pooled in zip(local_folds, pooled_folds)]
    comparisons["phase2_fold_local_vs_pooled"] = {
        "full_oof_auc_delta": (
            metric_records["phase2_fold_local"]["oof_blend_auc"]
            - metric_records["phase2_pooled"]["oof_blend_auc"]
        ),
        "paired_fold_auc_deltas": local_vs_pooled,
        "positive_folds": sum(delta > 0 for delta in local_vs_pooled),
    }
    pooled_improved = (
        comparisons["phase2_pooled"]["full_oof_auc_delta_vs_raw_baseline"] > 0
        and comparisons["phase2_pooled"]["positive_folds_vs_raw_baseline"] >= 4
    )
    local_improved = (
        comparisons["phase2_fold_local"]["full_oof_auc_delta_vs_raw_baseline"] > 0
        and comparisons["phase2_fold_local"]["positive_folds_vs_raw_baseline"] >= 4
    )
    decision = "PASS" if pooled_improved and local_improved else "CONDITIONAL"
    gpu_after = gpu_snapshot()
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "decision": decision,
        "outer_seed": outer_seed,
        "outer_folds": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": outer_seed,
            "different_from_previous_seed_42": True,
        },
        "inner_target_encoding": {
            "n_splits": phase02.INNER_SPLITS,
            "random_state": INNER_SEED,
            "nested_within_outer_training_rows": True,
            "self_target_contribution": False,
            "validation_labels_enter_training_encodings": False,
        },
        "lineage": "INDEPENDENT_RAW_ONLY_FINAL_VALIDATION",
        "historical_prediction_inputs": [],
        "historical_champion_read": False,
        "target_used_in_unsupervised_frequency_features": False,
        "fee_frequency_protocol": {
            "pooled": "counts from train and test covariates, without targets",
            "fold_local": (
                "each outer fold's frequency maps use only its fit-row "
                "covariates; validation and test covariates are excluded"
            ),
            "pooled_feature_diagnostics": pooled_diagnostics,
        },
        "metrics": metric_records,
        "comparisons": comparisons,
        "models": model_records,
        "folds": fold_records,
        "candidate_integrity": candidate_records,
        "devices": {
            "lightgbm": "CPU",
            "catboost": "GPU:0 explicitly configured",
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "per_catboost_fold_monitoring": True,
        },
        "raw_inputs": {
            name: {
                "path": str(path),
                **hash_record(path),
            }
            for name, path in input_paths.items()
        },
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                SOURCE_DIR / "run_independent_baseline.py",
                PHASE_01_DIR / "run_experiment.py",
                PHASE_02_DIR / "run_experiment.py",
            )
        },
        "model_configuration": MODEL_CONFIG,
        "official_competition_score": "NOT_EVALUATED",
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_hashes = {
        path.relative_to(run_dir).as_posix(): hash_record(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    manifest = {
        "experiment_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: hash_record(path) for name, path in input_paths.items()
        },
        "sources": {
            str(path): hash_record(path)
            for path in (
                Path(__file__),
                SOURCE_DIR / "run_independent_baseline.py",
                PHASE_01_DIR / "run_experiment.py",
                PHASE_02_DIR / "run_experiment.py",
            )
        },
        "outputs": output_hashes,
    }
    (run_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=NEW_OUTER_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.data_dir, arguments.run_dir, arguments.seed)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
