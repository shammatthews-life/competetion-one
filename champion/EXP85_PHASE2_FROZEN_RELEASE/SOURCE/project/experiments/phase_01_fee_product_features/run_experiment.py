#!/usr/bin/env python3
"""Compare raw-only baseline models with unsupervised decoded-loan features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
sys.path.insert(0, str(SOURCE_DIR))

from run_independent_baseline import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FOLD_SEED,
    ID_COLUMN,
    N_SPLITS,
    TARGET_COLUMN,
    build_rowwise_features,
    canonical_hash,
    fit_model,
    make_lightgbm_frames,
    sha256_file,
)


BASELINE_RUN = PROJECT_DIR / "runs" / "raw_baseline_20261101_154924"
BASELINE_METRICS = BASELINE_RUN / "metrics.json"
FEE_AMOUNT = 129.5
PRODUCT_RATE = 0.0132


def build_decoded_loan_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build label-free fee and product-tier features from train/test covariates."""
    combined = pd.concat(
        [
            train_raw.drop(columns=[TARGET_COLUMN], errors="ignore"),
            test_raw,
        ],
        ignore_index=True,
    )
    credit = combined["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = combined["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = combined["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    contract = combined["NAME_CONTRACT_TYPE"].astype("string").to_numpy()

    raw_ratio = np.full(len(combined), np.nan, dtype=np.float64)
    fee_ratio = np.full(len(combined), np.nan, dtype=np.float64)
    valid_credit = np.isfinite(credit) & (credit != 0.0)
    raw_ratio[valid_credit] = annuity[valid_credit] / credit[valid_credit]
    fee_ratio[valid_credit] = (annuity[valid_credit] - FEE_AMOUNT) / credit[
        valid_credit
    ]

    raw_key = pd.Series(raw_ratio).round(5)
    fee_key = pd.Series(fee_ratio).round(5)
    raw_counts = raw_key.map(raw_key.value_counts()).fillna(0).to_numpy()
    fee_counts = fee_key.map(fee_key.value_counts()).fillna(0).to_numpy()
    fee_decoded = (
        (contract == "Cash loans")
        & (fee_counts > raw_counts)
        & np.isfinite(raw_ratio)
    )
    decoded_ratio = np.where(fee_decoded, fee_ratio, raw_ratio)

    goods_valid = np.isfinite(goods) & (goods != 0.0)
    credit_goods_ratio = np.full(len(combined), np.nan, dtype=np.float64)
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
            "FEE_DECODED": fee_decoded.astype(np.float32),
            "FEE_RAW_RATIO_COUNT": np.log1p(raw_counts).astype(np.float32),
            "FEE_ADJUSTED_RATIO_COUNT": np.log1p(fee_counts).astype(np.float32),
            "ANNUITY_CREDIT_RATIO_FEE_ADJUSTED": fee_ratio,
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
    if np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Decoded loan features contain infinite values")

    n_train = len(train_raw)
    train_features = features.iloc[:n_train].reset_index(drop=True)
    test_features = features.iloc[n_train:].reset_index(drop=True)
    diagnostics = {
        "fee_amount": FEE_AMOUNT,
        "product_rate": PRODUCT_RATE,
        "fee_decoded_rows": int(fee_decoded.sum()),
        "fee_decoded_cash_loan_fraction": float(
            fee_decoded[contract == "Cash loans"].mean()
        ),
        "product_key_valid_rows": int(product_valid.sum()),
        "product_key_valid_fraction_of_cash_loans": float(
            product_valid[contract == "Cash loans"].mean()
        ),
        "feature_names": features.columns.tolist(),
        "frequency_counts_use_train_and_test_covariates": True,
        "frequency_counts_use_target": False,
    }
    return train_features, test_features, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Experiment directory must be empty: {run_dir}")
    if not BASELINE_METRICS.is_file():
        raise FileNotFoundError(f"Frozen baseline metrics missing: {BASELINE_METRICS}")

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
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample schema: {sample.columns.tolist()}")
    if not np.array_equal(sample[ID_COLUMN], test_raw[ID_COLUMN]):
        raise ValueError("Sample and test IDs/order differ")
    if train_raw[ID_COLUMN].duplicated().any() or test_raw[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate IDs in raw input")
    y = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)

    train_features = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    test_features = build_rowwise_features(test_raw)
    fee_train, fee_test, fee_diagnostics = build_decoded_loan_features(
        train_raw, test_raw
    )
    train_features = pd.concat([train_features, fee_train], axis=1)
    test_features = pd.concat([test_features, fee_test], axis=1)
    X_lgb_train, X_lgb_test = make_lightgbm_frames(
        train_features, test_features
    )

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(
            np.zeros(len(y)), y
        )
    )
    baseline_metrics = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    baseline_oof = {
        "lightgbm": np.load(BASELINE_RUN / "oof_lightgbm.npy", allow_pickle=False),
        "catboost": np.load(BASELINE_RUN / "oof_catboost.npy", allow_pickle=False),
        "blend": np.load(
            BASELINE_RUN / "oof_fixed_50_50_probability_blend.npy",
            allow_pickle=False,
        ),
    }
    if any(len(prediction) != len(y) for prediction in baseline_oof.values()):
        raise ValueError("Frozen baseline OOF row counts do not match training data")

    model_results = {}
    for model_name, X_train, X_test in (
        ("lightgbm", X_lgb_train, X_lgb_test),
        ("catboost", train_features, test_features),
    ):
        model_results[model_name] = fit_model(
            run_dir=run_dir,
            model_name=model_name,
            X_train=X_train,
            y=y,
            X_test=X_test,
            folds=folds,
            base_fingerprint_data={
                "experiment": "fee_decoded_product_tier_features",
                "raw_train_sha256": sha256_file(paths["train"]),
                "raw_test_sha256": sha256_file(paths["test"]),
                "baseline_source_sha256": sha256_file(
                    SOURCE_DIR / "run_independent_baseline.py"
                ),
                "feature_names": train_features.columns.tolist(),
                "fee_amount": FEE_AMOUNT,
                "product_rate": PRODUCT_RATE,
            },
            resume=False,
        )

    lgb_oof = model_results["lightgbm"]["oof"]
    cat_oof = model_results["catboost"]["oof"]
    blend_oof = (lgb_oof.astype(np.float64) + cat_oof.astype(np.float64)) / 2.0
    test_blend = (
        model_results["lightgbm"]["test"].astype(np.float64)
        + model_results["catboost"]["test"].astype(np.float64)
    ) / 2.0
    per_fold = []
    for fold_number, (_, valid_idx) in enumerate(folds, 1):
        baseline_fold = {
            model: float(roc_auc_score(y[valid_idx], baseline_oof[model][valid_idx]))
            for model in ("lightgbm", "catboost", "blend")
        }
        candidate_fold = {
            "lightgbm": float(roc_auc_score(y[valid_idx], lgb_oof[valid_idx])),
            "catboost": float(roc_auc_score(y[valid_idx], cat_oof[valid_idx])),
            "blend": float(roc_auc_score(y[valid_idx], blend_oof[valid_idx])),
        }
        per_fold.append(
            {
                "fold": fold_number,
                "baseline": baseline_fold,
                "fee_features": candidate_fold,
                "paired_delta": {
                    model: candidate_fold[model] - baseline_fold[model]
                    for model in candidate_fold
                },
            }
        )

    candidate = pd.DataFrame(
        {ID_COLUMN: test_raw[ID_COLUMN].to_numpy(), TARGET_COLUMN: test_blend}
    )
    if not np.isfinite(test_blend).all() or (
        (test_blend < 0.0) | (test_blend > 1.0)
    ).any():
        raise ValueError("Candidate prediction values are invalid")
    candidate_path = run_dir / "submission_fee_features.csv"
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    np.save(run_dir / "oof_fee_lightgbm.npy", lgb_oof)
    np.save(run_dir / "oof_fee_catboost.npy", cat_oof)
    np.save(run_dir / "oof_fee_fixed_50_50_blend.npy", blend_oof)
    np.save(run_dir / "test_fee_fixed_50_50_blend.npy", test_blend)
    np.savez_compressed(
        run_dir / "fold_assignments.npz",
        **{
            f"fold_{fold_number}_valid_indices": valid_idx.astype(np.int32)
            for fold_number, (_, valid_idx) in enumerate(folds, 1)
        },
    )

    deltas = {
        model: np.asarray(
            [row["paired_delta"][model] for row in per_fold], dtype=np.float64
        )
        for model in ("lightgbm", "catboost", "blend")
    }
    comparison = {
        model: {
            "fold_deltas": values.tolist(),
            "mean_fold_delta": float(values.mean()),
            "std_fold_delta": float(values.std(ddof=1)),
            "positive_folds": int((values > 0).sum()),
            "baseline_mean_fold_auc": float(
                np.mean([row["baseline"][model] for row in per_fold])
            ),
            "fee_feature_mean_fold_auc": float(
                np.mean([row["fee_features"][model] for row in per_fold])
            ),
            "baseline_full_oof_auc": float(
                roc_auc_score(y, baseline_oof[model])
            ),
            "fee_feature_full_oof_auc": float(
                roc_auc_score(
                    y,
                    {
                        "lightgbm": lgb_oof,
                        "catboost": cat_oof,
                        "blend": blend_oof,
                    }[model],
                )
            ),
        }
        for model, values in deltas.items()
    }

    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "lineage": "INDEPENDENT_RAW_ONLY_WITH_LABEL_FREE_FEE_AND_PRODUCT_FEATURES",
        "historical_prediction_inputs": [],
        "transductive_covariates": [
            "Pooled train/test unlabeled ratio-frequency counts for the fee decoder"
        ],
        "target_used_in_feature_construction": False,
        "folds": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "seed": FOLD_SEED,
            "same_as_frozen_baseline": True,
        },
        "baseline_run": str(BASELINE_RUN),
        "baseline_oof_auc": {
            "lightgbm": baseline_metrics["lightgbm"]["auc"],
            "catboost": baseline_metrics["catboost"]["auc"],
            "blend": baseline_metrics["fixed_50_50_probability_blend"]["auc"],
        },
        "fee_feature_oof_auc": {
            "lightgbm": float(roc_auc_score(y, lgb_oof)),
            "catboost": float(roc_auc_score(y, cat_oof)),
            "blend": float(roc_auc_score(y, blend_oof)),
        },
        "paired_cv_comparison": comparison,
        "per_fold": per_fold,
        "feature_diagnostics": fee_diagnostics,
        "model_folds": {
            model: result["folds"] for model, result in model_results.items()
        },
        "training_seconds": {
            model: float(
                sum(fold["elapsed_seconds"] for fold in result["folds"])
            )
            for model, result in model_results.items()
        },
        "elapsed_seconds_total": time.perf_counter() - started,
        "official_competition_score": "NOT_EVALUATED",
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_path": str(candidate_path),
        "fold_assignment_sha256": sha256_file(run_dir / "fold_assignments.npz"),
    }
    summary["experiment_fingerprint"] = canonical_hash(
        {
            "raw_train": sha256_file(paths["train"]),
            "raw_test": sha256_file(paths["test"]),
            "source": sha256_file(Path(__file__)),
            "baseline_source": sha256_file(SOURCE_DIR / "run_independent_baseline.py"),
            "fee_amount": FEE_AMOUNT,
            "product_rate": PRODUCT_RATE,
            "fold_seed": FOLD_SEED,
        }
    )
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    output_hashes = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            output_hashes[path.relative_to(run_dir).as_posix()] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    input_hashes = {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    (run_dir / "sha256_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": run_dir.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": input_hashes,
                "outputs": output_hashes,
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
