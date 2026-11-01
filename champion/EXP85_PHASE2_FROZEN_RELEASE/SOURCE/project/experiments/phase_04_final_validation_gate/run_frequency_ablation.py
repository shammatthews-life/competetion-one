#!/usr/bin/env python3
"""Ablate fee-frequency decoding on the final gate's fixed folds."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
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
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(SOURCE_DIR))

from run_final_gate import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DEFAULT_DATA_DIR,
    DEFAULT_RUN_DIR,
    FEE_AMOUNT,
    INNER_SEED,
    MODEL_CONFIG,
    NEW_OUTER_SEED,
    N_SPLITS,
    PRODUCT_RATE,
    GpuMonitor,
    phase02,
    sha256_file,
)
from run_independent_baseline import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    build_rowwise_features,
    make_lightgbm_frames,
)


DEFAULT_RUN_DIR_ABLATION = PROJECT_DIR / "runs" / "final_gate_seed20261101_frequency_ablation"
REFERENCE_GATE_DIR = DEFAULT_RUN_DIR
LOCAL_REFERENCE = REFERENCE_GATE_DIR / "oof_phase2_fold_local_blend.npy"
BASELINE_REFERENCE = REFERENCE_GATE_DIR / "oof_raw_baseline_blend.npy"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_frequency_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep rowwise/product/fee formulas but disable all frequency decoding."""
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
    goods = raw["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    goods_valid = np.isfinite(goods) & (goods != 0.0)
    credit_goods_ratio = np.full(len(raw), np.nan, dtype=np.float64)
    credit_goods_ratio[goods_valid] = credit[goods_valid] / goods[goods_valid]
    implied_product = (credit_goods_ratio - 1.0) / PRODUCT_RATE
    nearest_product = np.rint(implied_product)
    product_valid = (
        (contract == "Cash loans")
        & np.isfinite(implied_product)
        & (credit > goods)
        & (np.abs(implied_product - nearest_product) < 0.1)
    )
    features = pd.DataFrame(
        {
            "FEE_DECODED": np.zeros(len(raw), dtype=np.float32),
            "FEE_RAW_RATIO_COUNT": np.zeros(len(raw), dtype=np.float32),
            "FEE_ADJUSTED_RATIO_COUNT": np.zeros(len(raw), dtype=np.float32),
            "ANNUITY_CREDIT_RATIO_FEE_ADJUSTED": fee_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED": raw_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3": np.round(raw_ratio, 3),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4": np.round(raw_ratio, 4),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5": np.round(raw_ratio, 5),
            "CREDIT_GOODS_RATIO": credit_goods_ratio,
            "PRODUCT_KEY_VALID": product_valid.astype(np.float32),
            "PRODUCT_KEY_K": np.where(product_valid, nearest_product, np.nan),
            "PRODUCT_KEY_DISTANCE": np.where(
                product_valid, implied_product - nearest_product, np.nan
            ),
        }
    )
    require(ID_COLUMN not in features and TARGET_COLUMN not in features, "ID/target leakage")
    require(
        not np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any(),
        "Ablation features contain infinity",
    )
    return features


def file_hash(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run(
    data_dir: Path,
    run_dir: Path,
    reference_run_dir: Path = REFERENCE_GATE_DIR,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    reference_run_dir = reference_run_dir.resolve()
    local_reference = reference_run_dir / "oof_phase2_fold_local_blend.npy"
    baseline_reference = reference_run_dir / "oof_raw_baseline_blend.npy"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Ablation directory must be empty: {run_dir}")
    for path in (
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "sample_submission.csv",
        reference_run_dir / "experiment_summary.json",
        local_reference,
        baseline_reference,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required final-gate input missing: {path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    test_ids = test_raw[ID_COLUMN].to_numpy()
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and np.array_equal(sample[ID_COLUMN].to_numpy(), test_ids),
        "Sample submission schema or ID ordering is invalid",
    )
    rowwise_train = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    rowwise_test = build_rowwise_features(test_raw)
    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=NEW_OUTER_SEED,
        ).split(np.zeros(len(labels)), labels)
    )
    fold_predictions = {
        name: np.full(len(labels), np.nan, dtype=np.float32)
        for name in ("lightgbm", "catboost")
    }
    test_fold_predictions = {"lightgbm": [], "catboost": []}
    fold_records = []
    model_records = {"lightgbm": [], "catboost": []}
    assignment_arrays = {}

    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        require(
            np.intersect1d(fit_indices, valid_indices).size == 0,
            f"Fold {fold_number} fit/validation overlap",
        )
        reference_raw = train_raw.iloc[fit_indices].drop(
            columns=[TARGET_COLUMN], errors="ignore"
        )
        fit_fee = no_frequency_features(reference_raw)
        valid_fee = no_frequency_features(
            train_raw.iloc[valid_indices].drop(
                columns=[TARGET_COLUMN], errors="ignore"
            )
        )
        test_fee = no_frequency_features(test_raw)
        fee_train = pd.DataFrame(
            index=np.arange(len(train_raw)),
            columns=fit_fee.columns,
            dtype=np.float64,
        )
        fee_train.iloc[fit_indices] = fit_fee.to_numpy()
        fee_train.iloc[valid_indices] = valid_fee.to_numpy()
        cat_train = pd.concat(
            [rowwise_train.reset_index(drop=True), fee_train.reset_index(drop=True)],
            axis=1,
        )
        cat_test = pd.concat(
            [rowwise_test.reset_index(drop=True), test_fee.reset_index(drop=True)],
            axis=1,
        )
        keys = phase02.key_arrays(fee_train, test_fee)
        train_te = phase02.nested_training_encodings(keys, labels, fit_indices)
        valid_te = phase02.apply_training_encodings(
            keys, labels, fit_indices, valid_indices
        )
        test_indices = np.arange(
            len(labels), len(labels) + len(test_raw), dtype=np.int64
        )
        test_te = phase02.apply_training_encodings(
            keys, labels, fit_indices, test_indices
        )
        cat_train.loc[fit_indices, train_te.columns] = train_te.to_numpy()
        cat_train.loc[valid_indices, valid_te.columns] = valid_te.to_numpy()
        cat_test = pd.concat([cat_test, test_te], axis=1)
        require(
            ID_COLUMN not in cat_train.columns
            and TARGET_COLUMN not in cat_train.columns
            and ID_COLUMN not in cat_test.columns
            and TARGET_COLUMN not in cat_test.columns,
            "ID or target reached model features",
        )
        lgb_train, lgb_test = make_lightgbm_frames(cat_train, cat_test)
        require(
            list(lgb_train.columns) == list(lgb_test.columns),
            "LightGBM feature schema mismatch",
        )
        for frame in (cat_train, cat_test, lgb_train, lgb_test):
            numeric = frame.select_dtypes(include=[np.number])
            require(
                not np.isinf(
                    numeric.to_numpy(dtype=np.float64, copy=False)
                ).any(),
                "Feature matrix contains infinite values",
            )

        lgb_settings = copy.deepcopy(MODEL_CONFIG["lightgbm"])
        rounds = lgb_settings.pop("num_boost_round")
        early_stopping = lgb_settings.pop("early_stopping_rounds")
        lgb_settings.pop("device")
        for seed_name in ("seed", "feature_fraction_seed", "bagging_seed"):
            lgb_settings[seed_name] = NEW_OUTER_SEED + fold_number
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
        start = time.perf_counter()
        lgb_model = lgb.train(
            lgb_settings,
            lgb_fit,
            num_boost_round=rounds,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        lgb_prediction = lgb_model.predict(
            lgb_train.iloc[valid_indices],
            num_iteration=lgb_model.best_iteration,
        ).astype(np.float32)
        lgb_test_prediction = lgb_model.predict(
            lgb_test,
            num_iteration=lgb_model.best_iteration,
        ).astype(np.float32)
        lgb_auc = float(roc_auc_score(labels[valid_indices], lgb_prediction))
        fold_predictions["lightgbm"][valid_indices] = lgb_prediction
        test_fold_predictions["lightgbm"].append(lgb_test_prediction)
        model_records["lightgbm"].append(
            {
                "fold": fold_number,
                "device": "CPU",
                "seed": NEW_OUTER_SEED + fold_number,
                "best_iteration": int(lgb_model.best_iteration),
                "auc": lgb_auc,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        del lgb_model, lgb_fit, lgb_valid
        gc.collect()

        cat_settings = copy.deepcopy(MODEL_CONFIG["catboost"])
        cat_early_stopping = cat_settings.pop("early_stopping_rounds")
        cat_settings.pop("device")
        cat_settings.update(
            {
                "random_seed": NEW_OUTER_SEED + fold_number,
                "allow_writing_files": False,
                "verbose": False,
            }
        )
        cat_model = CatBoostClassifier(**cat_settings)
        start = time.perf_counter()
        with GpuMonitor() as monitor:
            cat_model.fit(
                cat_train.iloc[fit_indices],
                labels[fit_indices],
                cat_features=list(CATEGORICAL_COLUMNS),
                eval_set=(cat_train.iloc[valid_indices], labels[valid_indices]),
                early_stopping_rounds=cat_early_stopping,
                verbose=False,
            )
        require(
            not monitor.errors and monitor.samples,
            f"GPU monitoring failed in fold {fold_number}: {monitor.errors}",
        )
        cat_prediction = cat_model.predict_proba(
            cat_train.iloc[valid_indices]
        )[:, 1].astype(np.float32)
        cat_test_prediction = cat_model.predict_proba(cat_test)[:, 1].astype(
            np.float32
        )
        cat_auc = float(roc_auc_score(labels[valid_indices], cat_prediction))
        fold_predictions["catboost"][valid_indices] = cat_prediction
        test_fold_predictions["catboost"].append(cat_test_prediction)
        gpu_monitor = {
            "samples": len(monitor.samples),
            "max_utilization_percent": max(
                sample["utilization_percent"] for sample in monitor.samples
            ),
            "max_memory_used_mib": max(
                sample["memory_used_mib"] for sample in monitor.samples
            ),
            "errors": monitor.errors,
        }
        model_records["catboost"].append(
            {
                "fold": fold_number,
                "device": "GPU:0",
                "seed": NEW_OUTER_SEED + fold_number,
                "best_iteration": int(cat_model.best_iteration_),
                "auc": cat_auc,
                "elapsed_seconds": time.perf_counter() - start,
                "gpu_monitor": gpu_monitor,
            }
        )
        del cat_model
        gc.collect()
        assignment_arrays[f"fit_{fold_number}"] = fit_indices.astype(np.int32)
        assignment_arrays[f"valid_{fold_number}"] = valid_indices.astype(np.int32)
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=fit_indices.astype(np.int32),
            valid_indices=valid_indices.astype(np.int32),
        )
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
                "fee_decoded_rows": 0,
                "frequency_count_values": "all zero",
                "nested_training_encoding_rows": int(len(train_te)),
                "validation_labels_used_for_training_encodings": False,
            }
        )
        print(
            f"[frequency_disabled] fold {fold_number}/{N_SPLITS} "
            f"auc_lgb={lgb_auc:.8f} auc_cat={cat_auc:.8f} "
            f"gpu_samples={gpu_monitor['samples']} "
            f"gpu_util_max={gpu_monitor['max_utilization_percent']}",
            flush=True,
        )

    require(
        all(np.isfinite(values).all() for values in fold_predictions.values()),
        "Incomplete OOF prediction vector",
    )
    oof_blend = (
        fold_predictions["lightgbm"].astype(np.float64)
        + fold_predictions["catboost"].astype(np.float64)
    ) / 2.0
    test_predictions = {
        model: np.mean(
            np.stack(values), axis=0, dtype=np.float64
        ).astype(np.float64)
        for model, values in test_fold_predictions.items()
    }
    test_blend = (test_predictions["lightgbm"] + test_predictions["catboost"]) / 2.0
    np.save(run_dir / "oof_frequency_disabled_lightgbm.npy", fold_predictions["lightgbm"])
    np.save(run_dir / "oof_frequency_disabled_catboost.npy", fold_predictions["catboost"])
    np.save(run_dir / "oof_frequency_disabled_blend.npy", oof_blend)
    np.save(run_dir / "test_frequency_disabled_lightgbm.npy", test_predictions["lightgbm"])
    np.save(run_dir / "test_frequency_disabled_catboost.npy", test_predictions["catboost"])
    np.save(run_dir / "test_frequency_disabled_blend.npy", test_blend)
    candidate = pd.DataFrame(
        {ID_COLUMN: test_ids, TARGET_COLUMN: test_blend}
    )
    candidate_path = run_dir / "submission_frequency_disabled.csv"
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    local_oof = np.load(local_reference, allow_pickle=False).astype(np.float64)
    baseline_oof = np.load(baseline_reference, allow_pickle=False).astype(np.float64)
    local_fold_auc = []
    ablated_fold_auc = []
    baseline_fold_auc = []
    for (_, valid_indices), local_prediction in zip(folds, fold_records):
        local_fold_auc.append(
            float(roc_auc_score(labels[valid_indices], local_oof[valid_indices]))
        )
        ablated_fold_auc.append(
            float(
                roc_auc_score(
                    labels[valid_indices],
                    oof_blend[valid_indices],
                )
            )
        )
        baseline_fold_auc.append(
            float(
                roc_auc_score(
                    labels[valid_indices],
                    baseline_oof[valid_indices],
                )
            )
        )
    frequency_deltas = [
        local_auc - disabled_auc
        for local_auc, disabled_auc in zip(local_fold_auc, ablated_fold_auc)
    ]
    baseline_deltas = [
        disabled_auc - baseline_auc
        for disabled_auc, baseline_auc in zip(
            ablated_fold_auc, baseline_fold_auc
        )
    ]
    local_full_auc = float(roc_auc_score(labels, local_oof))
    ablated_full_auc = float(roc_auc_score(labels, oof_blend))
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "outer_seed": NEW_OUTER_SEED,
        "folds_identical_to_final_gate_run": True,
        "feature_ablation": (
            "Disable fee-frequency decoder: FEE_DECODED and both frequency "
            "counts are zero; decoded ratio/roundings revert to raw "
            "annuity-credit ratio. Preserve rowwise features, product-key "
            "features, fixed fee-adjusted ratio, and nested target encoding."
        ),
        "historical_prediction_inputs": [],
        "posthoc_comparison_inputs_only": [
            str(local_reference),
            str(baseline_reference),
        ],
        "metrics": {
            "frequency_disabled_oof_blend_auc": ablated_full_auc,
            "frequency_local_enabled_oof_blend_auc": local_full_auc,
            "new_seed_raw_baseline_oof_blend_auc": float(
                roc_auc_score(labels, baseline_oof)
            ),
            "paired_fold_auc": {
                "frequency_disabled": ablated_fold_auc,
                "frequency_local_enabled": local_fold_auc,
                "raw_baseline": baseline_fold_auc,
            },
            "fee_frequency_decoder_delta_vs_disabled": {
                "full_oof_auc_delta": local_full_auc - ablated_full_auc,
                "paired_fold_deltas": frequency_deltas,
                "positive_folds": sum(value > 0 for value in frequency_deltas),
            },
            "frequency_disabled_delta_vs_raw_baseline": {
                "full_oof_auc_delta": ablated_full_auc
                - float(roc_auc_score(labels, baseline_oof)),
                "paired_fold_deltas": baseline_deltas,
                "positive_folds": sum(value > 0 for value in baseline_deltas),
            },
        },
        "models": model_records,
        "folds": fold_records,
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "rows": len(candidate),
            "columns": candidate.columns.tolist(),
            "unique_ids": int(candidate[ID_COLUMN].nunique()),
            "ids_match_test_order": bool(
                np.array_equal(candidate[ID_COLUMN].to_numpy(), test_ids)
            ),
            "prediction_min": float(test_blend.min()),
            "prediction_max": float(test_blend.max()),
        },
        "raw_inputs": {
            name: {"path": str(path), **file_hash(path)}
            for name, path in paths.items()
        },
        "source_hashes": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                EXPERIMENT_DIR / "run_final_gate.py",
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
    outputs = {
        path.relative_to(run_dir).as_posix(): file_hash(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    manifest = {
        "experiment_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: file_hash(path) for name, path in paths.items()},
        "analysis_only_inputs": {
            str(path): file_hash(path)
            for path in (local_reference, baseline_reference)
        },
        "sources": {
            str(path): file_hash(path)
            for path in (
                Path(__file__),
                EXPERIMENT_DIR / "run_final_gate.py",
                SOURCE_DIR / "run_independent_baseline.py",
                PHASE_01_DIR / "run_experiment.py",
                PHASE_02_DIR / "run_experiment.py",
            )
        },
        "outputs": outputs,
    }
    (run_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR_ABLATION
    )
    parser.add_argument(
        "--reference-run-dir", type=Path, default=REFERENCE_GATE_DIR
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.data_dir, args.run_dir, args.reference_run_dir)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
