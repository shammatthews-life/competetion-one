#!/usr/bin/env python3
"""Evaluate CUDA XGBoost alongside the nested-key-encoding ensemble."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
PHASE_01_DIR = PROJECT_DIR / "experiments" / "phase_01_fee_product_features"
PHASE_02_DIR = PROJECT_DIR / "experiments" / "phase_02_fold_safe_key_encodings"
PHASE_02_RUN = PROJECT_DIR / "runs" / "phase02_key_te_20261101_181500"
sys.path.insert(0, str(SOURCE_DIR))

from run_independent_baseline import (  # noqa: E402
    FOLD_SEED,
    ID_COLUMN,
    N_SPLITS,
    TARGET_COLUMN,
    build_rowwise_features,
    canonical_hash,
    gpu_snapshot,
    make_lightgbm_frames,
    sha256_file,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase_01 = load_module(
    "phase_01_fee_product_experiment",
    PHASE_01_DIR / "run_experiment.py",
)
phase_02 = load_module(
    "phase_02_nested_target_experiment",
    PHASE_02_DIR / "run_experiment.py",
)

XGB_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 8.0,
    "alpha": 0.1,
    "max_bin": 256,
    "nthread": 8,
    "verbosity": 0,
    "num_boost_round": 2500,
    "early_stopping_rounds": 150,
}


def output_hash(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Experiment directory must be empty: {run_dir}")
    if not (PHASE_02_RUN / "oof_50_50_blend.npy").is_file():
        raise FileNotFoundError(f"Nested target encoding run missing: {PHASE_02_RUN}")
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    y = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample schema: {sample.columns.tolist()}")
    if not np.array_equal(sample[ID_COLUMN], test_raw[ID_COLUMN]):
        raise ValueError("Sample and test IDs/order differ")

    base_train = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    base_test = build_rowwise_features(test_raw)
    fee_train, fee_test, feature_diagnostics = (
        phase_01.build_decoded_loan_features(train_raw, test_raw)
    )
    base_train = pd.concat([base_train, fee_train], axis=1)
    base_test = pd.concat([base_test, fee_test], axis=1)
    keys_by_name = phase_02.key_arrays(fee_train, fee_test)
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(
            np.zeros(len(y)), y
        )
    )
    gpu_before = gpu_snapshot()
    oof = np.full(len(y), np.nan, dtype=np.float32)
    test_fold_predictions = []
    fold_records = []
    fold_assignments = {}

    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        fold_started = time.perf_counter()
        train_te = phase_02.nested_training_encodings(
            keys_by_name, y, fit_indices
        )
        valid_te = phase_02.apply_training_encodings(
            keys_by_name, y, fit_indices, valid_indices
        )
        test_indices = np.arange(
            len(y), len(y) + len(test_raw), dtype=np.int64
        )
        test_te = phase_02.apply_training_encodings(
            keys_by_name, y, fit_indices, test_indices
        )
        X_train_fold = base_train.copy()
        X_train_fold.loc[fit_indices, train_te.columns] = train_te.to_numpy()
        X_train_fold.loc[valid_indices, valid_te.columns] = valid_te.to_numpy()
        X_test_fold = pd.concat([base_test, test_te], axis=1)
        X_lgb_train, X_lgb_test = make_lightgbm_frames(
            X_train_fold, X_test_fold
        )
        if not X_lgb_train.columns.is_unique or list(
            X_lgb_train.columns
        ) != list(X_lgb_test.columns):
            raise ValueError("XGBoost train/test schemas do not match")
        for frame in (X_lgb_train, X_lgb_test):
            if np.isinf(frame.to_numpy(dtype=np.float64, copy=False)).any():
                raise ValueError("XGBoost input contains infinite values")

        dtrain = xgb.DMatrix(
            X_lgb_train.iloc[fit_indices],
            label=y[fit_indices],
            feature_names=list(X_lgb_train.columns),
        )
        dvalid = xgb.DMatrix(
            X_lgb_train.iloc[valid_indices],
            label=y[valid_indices],
            feature_names=list(X_lgb_train.columns),
        )
        dtest = xgb.DMatrix(
            X_lgb_test,
            feature_names=list(X_lgb_test.columns),
        )
        params = dict(XGB_CONFIG)
        params.pop("num_boost_round")
        params.pop("early_stopping_rounds")
        params["seed"] = FOLD_SEED + fold_number
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=XGB_CONFIG["num_boost_round"],
            evals=[(dvalid, "valid")],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=XGB_CONFIG["early_stopping_rounds"],
                    save_best=True,
                    maximize=True,
                    data_name="valid",
                    metric_name="auc",
                )
            ],
            verbose_eval=False,
        )
        valid_prediction = model.predict(dvalid)
        test_prediction = model.predict(dtest)
        fold_auc = float(roc_auc_score(y[valid_indices], valid_prediction))
        oof[valid_indices] = valid_prediction.astype(np.float32)
        test_fold_predictions.append(test_prediction.astype(np.float32))
        model_path = run_dir / f"xgboost_fold_{fold_number}.json"
        model.save_model(model_path)
        fold_records.append(
            {
                "fold": fold_number,
                "device": "CUDA (explicit XGBoost device=cuda)",
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
                "best_iteration": int(model.best_iteration),
                "auc": fold_auc,
                "elapsed_seconds": time.perf_counter() - fold_started,
                "booster_config": json.loads(model.save_config()),
            }
        )
        fold_assignments[f"fit_{fold_number}"] = fit_indices.astype(np.int32)
        fold_assignments[f"valid_{fold_number}"] = valid_indices.astype(np.int32)
        print(
            f"[xgboost] fold {fold_number}/{N_SPLITS} device=CUDA "
            f"auc={fold_auc:.8f} best_iteration={model.best_iteration}",
            flush=True,
        )

    if np.isnan(oof).any():
        raise ValueError("XGBoost OOF predictions are incomplete")
    test_prediction = np.mean(
        np.stack(test_fold_predictions), axis=0, dtype=np.float64
    ).astype(np.float32)
    np.save(run_dir / "oof_xgboost.npy", oof)
    np.save(run_dir / "test_xgboost.npy", test_prediction)
    np.savez_compressed(run_dir / "fold_assignments.npz", **fold_assignments)

    phase_02_oof_lgb = np.load(
        PHASE_02_RUN / "oof_lightgbm.npy", allow_pickle=False
    )
    phase_02_oof_cat = np.load(
        PHASE_02_RUN / "oof_catboost.npy", allow_pickle=False
    )
    phase_02_test_lgb = np.load(
        PHASE_02_RUN / "test_lightgbm.npy", allow_pickle=False
    )
    phase_02_test_cat = np.load(
        PHASE_02_RUN / "test_catboost.npy", allow_pickle=False
    )
    two_way_oof = (
        phase_02_oof_lgb.astype(np.float64)
        + phase_02_oof_cat.astype(np.float64)
    ) / 2.0
    three_way_oof = (
        phase_02_oof_lgb.astype(np.float64)
        + phase_02_oof_cat.astype(np.float64)
        + oof.astype(np.float64)
    ) / 3.0
    three_way_test = (
        phase_02_test_lgb.astype(np.float64)
        + phase_02_test_cat.astype(np.float64)
        + test_prediction.astype(np.float64)
    ) / 3.0
    np.save(run_dir / "oof_equal_3_model_blend.npy", three_way_oof)
    np.save(run_dir / "test_equal_3_model_blend.npy", three_way_test)
    candidate = pd.DataFrame(
        {ID_COLUMN: test_raw[ID_COLUMN].to_numpy(), TARGET_COLUMN: three_way_test}
    )
    if not np.array_equal(candidate[ID_COLUMN], sample[ID_COLUMN]):
        raise ValueError("Candidate IDs/order differ from sample submission")
    if not np.isfinite(three_way_test).all() or (
        (three_way_test < 0.0) | (three_way_test > 1.0)
    ).any():
        raise ValueError("Candidate predictions are invalid")
    candidate_path = run_dir / "submission_equal_3_model.csv"
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    per_fold = []
    for fold_number, (_, valid_indices) in enumerate(folds, 1):
        xgb_auc = float(roc_auc_score(y[valid_indices], oof[valid_indices]))
        two_auc = float(
            roc_auc_score(y[valid_indices], two_way_oof[valid_indices])
        )
        three_auc = float(
            roc_auc_score(y[valid_indices], three_way_oof[valid_indices])
        )
        per_fold.append(
            {
                "fold": fold_number,
                "xgboost_auc": xgb_auc,
                "nested_te_two_model_auc": two_auc,
                "fixed_equal_three_model_auc": three_auc,
                "three_model_delta_vs_two_model": three_auc - two_auc,
            }
        )
    gpu_after = gpu_snapshot()
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "lineage": "INDEPENDENT_RAW_ONLY_WITH_NESTED_TARGET_ENCODINGS_AND_XGBOOST_CUDA",
        "historical_prediction_inputs": [],
        "comparison_only_prediction_inputs": [
            str(PHASE_02_RUN / "oof_lightgbm.npy"),
            str(PHASE_02_RUN / "oof_catboost.npy"),
            str(PHASE_02_RUN / "test_lightgbm.npy"),
            str(PHASE_02_RUN / "test_catboost.npy"),
        ],
        "xgboost_version": xgb.__version__,
        "xgboost_configuration": XGB_CONFIG,
        "folds": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "seed": FOLD_SEED,
            "same_as_previous_experiments": True,
        },
        "xgboost_oof_auc": float(roc_auc_score(y, oof)),
        "two_model_oof_auc": float(roc_auc_score(y, two_way_oof)),
        "fixed_equal_three_model_oof_auc": float(
            roc_auc_score(y, three_way_oof)
        ),
        "oof_prediction_correlations": {
            "xgboost_lightgbm_pearson": float(
                np.corrcoef(oof, phase_02_oof_lgb)[0, 1]
            ),
            "xgboost_catboost_pearson": float(
                np.corrcoef(oof, phase_02_oof_cat)[0, 1]
            ),
        },
        "per_fold": per_fold,
        "fold_models": fold_records,
        "feature_diagnostics": feature_diagnostics,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "official_competition_score": "NOT_EVALUATED",
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    summary["experiment_fingerprint"] = canonical_hash(
        {
            "train_sha256": sha256_file(paths["train"]),
            "test_sha256": sha256_file(paths["test"]),
            "source_sha256": sha256_file(Path(__file__)),
            "phase_02_source_sha256": sha256_file(
                PHASE_02_DIR / "run_experiment.py"
            ),
            "xgboost_configuration": XGB_CONFIG,
            "fold_seed": FOLD_SEED,
        }
    )
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            outputs[path.relative_to(run_dir).as_posix()] = output_hash(path)
    inputs = {
        name: output_hash(path)
        for name, path in paths.items()
    }
    inputs["phase_02_summary"] = output_hash(
        PHASE_02_RUN / "experiment_summary.json"
    )
    inputs["phase_02_source"] = output_hash(
        PHASE_02_DIR / "run_experiment.py"
    )
    inputs["experiment_source"] = output_hash(Path(__file__))
    (run_dir / "sha256_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": inputs,
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
