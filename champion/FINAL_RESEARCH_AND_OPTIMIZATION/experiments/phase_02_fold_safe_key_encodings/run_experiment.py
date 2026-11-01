#!/usr/bin/env python3
"""Add nested, fold-safe target encodings for decoded loan-product keys."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
PHASE_01_DIR = PROJECT_DIR / "experiments" / "phase_01_fee_product_features"
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(PHASE_01_DIR))

from run_independent_baseline import (  # noqa: E402
    FOLD_SEED,
    ID_COLUMN,
    MODEL_CONFIG,
    N_SPLITS,
    TARGET_COLUMN,
    build_rowwise_features,
    canonical_hash,
    gpu_snapshot,
    make_lightgbm_frames,
    sha256_file,
    train_catboost_fold,
    train_lightgbm_fold,
)

_phase_01_spec = importlib.util.spec_from_file_location(
    "phase_01_fee_product_experiment",
    PHASE_01_DIR / "run_experiment.py",
)
if _phase_01_spec is None or _phase_01_spec.loader is None:
    raise ImportError("Could not load the phase-one feature builder")
_phase_01_module = importlib.util.module_from_spec(_phase_01_spec)
_phase_01_spec.loader.exec_module(_phase_01_module)
build_decoded_loan_features = _phase_01_module.build_decoded_loan_features


BASELINE_RUN = PROJECT_DIR / "runs" / "raw_baseline_20261101_154924"
PHASE_01_RUN = PROJECT_DIR / "runs" / "phase01_fee_product_20261101_171500"
MISSING_KEY = "__MISSING_KEY__"
KEY_SPECS = (
    ("FEE_Q3", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3", 100.0),
    ("FEE_Q4", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4", 30.0),
    ("FEE_Q5", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5", 30.0),
    ("PRODUCT_K", "PRODUCT_KEY_K", 50.0),
)
INNER_SPLITS = 5
INNER_SEED = 7


def key_arrays(
    fee_train: pd.DataFrame, fee_test: pd.DataFrame
) -> dict[str, np.ndarray]:
    combined = pd.concat([fee_train, fee_test], ignore_index=True)
    result = {}
    for name, column, _ in KEY_SPECS:
        values = combined[column].round(0 if name == "PRODUCT_K" else int(column[-1]))
        result[name] = (
            values.astype("string").fillna(MISSING_KEY).to_numpy(dtype=str)
        )
    return result


def encode_keys(
    keys: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    key_name: str,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode apply rows from fit labels only; unknown keys receive the fit prior."""
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    if np.intersect1d(fit_indices, apply_indices).size:
        raise ValueError("Target-encoding fit and apply indices overlap")
    prior = float(np.mean(labels[fit_indices]))
    fit = pd.DataFrame(
        {
            "key": keys[fit_indices],
            "target": labels[fit_indices].astype(np.float64),
        }
    )
    grouped = fit.groupby("key", sort=False)["target"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + smoothing * prior) / (
        grouped["count"] + smoothing
    )
    apply_keys = pd.Series(keys[apply_indices])
    encoded = apply_keys.map(smoothed).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    return encoded, counts


def nested_training_encodings(
    keys_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    outer_train_indices: np.ndarray,
) -> pd.DataFrame:
    """Create inner OOF encodings for outer-training rows."""
    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    output = np.full(
        (len(outer_train_indices), 2 * len(KEY_SPECS)),
        np.nan,
        dtype=np.float32,
    )
    inner = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED
    )
    local_y = labels[outer_train_indices]
    for inner_fit_local, inner_apply_local in inner.split(
        np.zeros(len(outer_train_indices)), local_y
    ):
        inner_fit = outer_train_indices[inner_fit_local]
        inner_apply = outer_train_indices[inner_apply_local]
        for key_index, (name, _, smoothing) in enumerate(KEY_SPECS):
            encoded, count = encode_keys(
                keys_by_name[name],
                labels,
                inner_fit,
                inner_apply,
                name,
                smoothing,
            )
            output[inner_apply_local, 2 * key_index] = encoded
            output[inner_apply_local, 2 * key_index + 1] = np.log1p(count)
    if not np.isfinite(output).all():
        raise ValueError("Nested target encodings are incomplete or non-finite")
    columns = [
        column
        for name, _, _ in KEY_SPECS
        for column in (f"TE_{name}", f"LOG_COUNT_{name}")
    ]
    return pd.DataFrame(output, columns=columns)


def apply_training_encodings(
    keys_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> pd.DataFrame:
    values = {}
    for name, _, smoothing in KEY_SPECS:
        encoded, count = encode_keys(
            keys_by_name[name],
            labels,
            fit_indices,
            apply_indices,
            name,
            smoothing,
        )
        values[f"TE_{name}"] = encoded
        values[f"LOG_COUNT_{name}"] = np.log1p(count)
    return pd.DataFrame(values).reset_index(drop=True)


def file_hash(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Experiment directory must be empty: {run_dir}")
    for path in (
        BASELINE_RUN / "metrics.json",
        PHASE_01_RUN / "experiment_summary.json",
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "sample_submission.csv",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input missing: {path}")
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
    y = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample schema: {sample.columns.tolist()}")
    if not np.array_equal(sample[ID_COLUMN], test_raw[ID_COLUMN]):
        raise ValueError("Sample and test IDs/order differ")

    base_train = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    base_test = build_rowwise_features(test_raw)
    fee_train, fee_test, feature_diagnostics = build_decoded_loan_features(
        train_raw, test_raw
    )
    base_train = pd.concat([base_train, fee_train], axis=1)
    base_test = pd.concat([base_test, fee_test], axis=1)
    keys_by_name = key_arrays(fee_train, fee_test)
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED).split(
            np.zeros(len(y)), y
        )
    )
    gpu_before = gpu_snapshot()
    model_oof = {
        name: np.full(len(y), np.nan, dtype=np.float32)
        for name in ("lightgbm", "catboost")
    }
    model_tests: dict[str, list[np.ndarray]] = {
        name: [] for name in ("lightgbm", "catboost")
    }
    model_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("lightgbm", "catboost")
    }
    importance: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("lightgbm", "catboost")
    }
    fold_records = []

    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        fold_started = time.perf_counter()
        train_te = nested_training_encodings(
            keys_by_name, y, fit_indices
        )
        valid_te = apply_training_encodings(
            keys_by_name, y, fit_indices, valid_indices
        )
        test_indices = np.arange(len(y), len(y) + len(test_raw), dtype=np.int64)
        test_te = apply_training_encodings(
            keys_by_name, y, fit_indices, test_indices
        )
        X_train_fold = base_train.copy()
        X_train_fold.loc[fit_indices, train_te.columns] = train_te.to_numpy()
        X_train_fold.loc[valid_indices, valid_te.columns] = valid_te.to_numpy()
        X_test_fold = pd.concat([base_test, test_te], axis=1)
        if np.isinf(
            X_train_fold.select_dtypes(include=[np.number]).to_numpy(
                dtype=np.float64, copy=False
            )
        ).any():
            raise ValueError("Feature matrix has infinite numeric values")

        X_lgb_train, X_lgb_test = make_lightgbm_frames(
            X_train_fold, X_test_fold
        )
        lgb_result = train_lightgbm_fold(
            X_lgb_train,
            y,
            X_lgb_test,
            fit_indices,
            valid_indices,
            fold_number,
        )
        cat_result = train_catboost_fold(
            X_train_fold,
            y,
            X_test_fold,
            fit_indices,
            valid_indices,
            fold_number,
        )
        for name, result in (
            ("lightgbm", lgb_result),
            ("catboost", cat_result),
        ):
            valid_prediction, test_prediction, best_iteration, fold_auc, imp = (
                result
            )
            model_oof[name][valid_indices] = valid_prediction
            model_tests[name].append(test_prediction)
            model_records[name].append(
                {
                    "fold": fold_number,
                    "device": MODEL_CONFIG[name]["device"],
                    "best_iteration": best_iteration,
                    "auc": fold_auc,
                }
            )
            importance[name].append(imp)
            print(
                f"[{name}] fold {fold_number}/{N_SPLITS} "
                f"device={MODEL_CONFIG[name]['device']} "
                f"auc={fold_auc:.8f} best_iteration={best_iteration}",
                flush=True,
            )

        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
                "nested_encoding_rows": int(len(train_te)),
                "fit_valid_indices_disjoint": not bool(
                    np.intersect1d(fit_indices, valid_indices).size
                ),
                "seconds": time.perf_counter() - fold_started,
            }
        )
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=fit_indices.astype(np.int32),
            valid_indices=valid_indices.astype(np.int32),
        )

    if any(np.isnan(prediction).any() for prediction in model_oof.values()):
        raise ValueError("OOF predictions contain uncovered rows")
    model_test = {
        name: np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(
            np.float32
        )
        for name, predictions in model_tests.items()
    }
    blend_oof = (
        model_oof["lightgbm"].astype(np.float64)
        + model_oof["catboost"].astype(np.float64)
    ) / 2.0
    blend_test = (
        model_test["lightgbm"].astype(np.float64)
        + model_test["catboost"].astype(np.float64)
    ) / 2.0
    for model in ("lightgbm", "catboost"):
        pd.concat(importance[model], ignore_index=True).to_csv(
            run_dir / f"{model}_feature_importance.csv", index=False
        )
        np.save(run_dir / f"oof_{model}.npy", model_oof[model])
        np.save(run_dir / f"test_{model}.npy", model_test[model])
    np.save(run_dir / "oof_50_50_blend.npy", blend_oof)
    np.save(run_dir / "test_50_50_blend.npy", blend_test)
    candidate = pd.DataFrame(
        {ID_COLUMN: test_raw[ID_COLUMN].to_numpy(), TARGET_COLUMN: blend_test}
    )
    if not np.isfinite(blend_test).all() or (
        (blend_test < 0.0) | (blend_test > 1.0)
    ).any():
        raise ValueError("Candidate prediction values are invalid")
    if not np.array_equal(candidate[ID_COLUMN], sample[ID_COLUMN]):
        raise ValueError("Candidate ID order differs from sample")
    candidate_path = run_dir / "submission_key_target_encoding.csv"
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    baseline_oof = {
        "lightgbm": np.load(BASELINE_RUN / "oof_lightgbm.npy", allow_pickle=False),
        "catboost": np.load(BASELINE_RUN / "oof_catboost.npy", allow_pickle=False),
        "blend": np.load(
            BASELINE_RUN / "oof_fixed_50_50_probability_blend.npy",
            allow_pickle=False,
        ),
    }
    fee_oof = {
        "lightgbm": np.load(PHASE_01_RUN / "oof_lightgbm.npy", allow_pickle=False),
        "catboost": np.load(PHASE_01_RUN / "oof_catboost.npy", allow_pickle=False),
        "blend": np.load(
            PHASE_01_RUN / "oof_fee_fixed_50_50_blend.npy",
            allow_pickle=False,
        ),
    }
    candidate_oof = {
        "lightgbm": model_oof["lightgbm"],
        "catboost": model_oof["catboost"],
        "blend": blend_oof,
    }
    paired = []
    for fold_number, (_, valid_indices) in enumerate(folds, 1):
        scores = {
            label: {
                model: float(roc_auc_score(y[valid_indices], predictions[model][valid_indices]))
                for model in ("lightgbm", "catboost", "blend")
            }
            for label, predictions in (
                ("raw_baseline", baseline_oof),
                ("fee_features", fee_oof),
                ("fee_plus_nested_te", candidate_oof),
            )
        }
        paired.append(
            {
                "fold": fold_number,
                "auc": scores,
                "delta_vs_fee_features": {
                    model: scores["fee_plus_nested_te"][model]
                    - scores["fee_features"][model]
                    for model in ("lightgbm", "catboost", "blend")
                },
                "delta_vs_raw_baseline": {
                    model: scores["fee_plus_nested_te"][model]
                    - scores["raw_baseline"][model]
                    for model in ("lightgbm", "catboost", "blend")
                },
            }
        )

    comparison = {}
    for model in ("lightgbm", "catboost", "blend"):
        deltas = np.asarray(
            [row["delta_vs_fee_features"][model] for row in paired],
            dtype=np.float64,
        )
        comparison[model] = {
            "fee_feature_mean_fold_auc": float(
                np.mean([row["auc"]["fee_features"][model] for row in paired])
            ),
            "fee_plus_te_mean_fold_auc": float(
                np.mean(
                    [row["auc"]["fee_plus_nested_te"][model] for row in paired]
                )
            ),
            "mean_fold_delta_vs_fee_features": float(deltas.mean()),
            "std_fold_delta_vs_fee_features": float(deltas.std(ddof=1)),
            "positive_folds_vs_fee_features": int((deltas > 0).sum()),
            "full_oof_auc_raw_baseline": float(
                roc_auc_score(y, baseline_oof[model])
            ),
            "full_oof_auc_fee_features": float(roc_auc_score(y, fee_oof[model])),
            "full_oof_auc_fee_plus_nested_te": float(
                roc_auc_score(y, candidate_oof[model])
            ),
        }

    gpu_after = gpu_snapshot()
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "lineage": "INDEPENDENT_RAW_ONLY_WITH_NESTED_FOLD_SAFE_TARGET_ENCODINGS",
        "historical_prediction_inputs": [],
        "comparison_only_prediction_inputs": [
            str(BASELINE_RUN / "oof_lightgbm.npy"),
            str(BASELINE_RUN / "oof_catboost.npy"),
            str(PHASE_01_RUN / "oof_fee_fixed_50_50_blend.npy"),
        ],
        "target_encoding_fit_rule": (
            "For each outer fold, inner five-fold OOF encodings with seed 7 "
            "for outer-training rows; outer validation and test keys encoded "
            "from outer-training labels only."
        ),
        "target_encoding_keys": [
            {"name": name, "source_feature": column, "smoothing": smoothing}
            for name, column, smoothing in KEY_SPECS
        ],
        "folds": {
            "outer_splits": N_SPLITS,
            "outer_seed": FOLD_SEED,
            "inner_splits": INNER_SPLITS,
            "inner_seed": INNER_SEED,
            "same_outer_folds_as_baseline": True,
        },
        "baseline_run": str(BASELINE_RUN),
        "phase_01_run": str(PHASE_01_RUN),
        "auc_comparison": comparison,
        "paired_cv_by_fold": paired,
        "model_folds": model_records,
        "fold_execution": fold_records,
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
            "baseline_source_sha256": sha256_file(
                SOURCE_DIR / "run_independent_baseline.py"
            ),
            "phase_01_source_sha256": sha256_file(
                PHASE_01_DIR / "run_experiment.py"
            ),
            "outer_seed": FOLD_SEED,
            "inner_seed": INNER_SEED,
            "key_specs": KEY_SPECS,
        }
    )
    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            outputs[path.relative_to(run_dir).as_posix()] = file_hash(path)
    inputs = {
        name: file_hash(path)
        for name, path in paths.items()
    }
    inputs["baseline_metrics"] = file_hash(BASELINE_RUN / "metrics.json")
    inputs["phase_01_summary"] = file_hash(PHASE_01_RUN / "experiment_summary.json")
    inputs["baseline_source"] = file_hash(
        SOURCE_DIR / "run_independent_baseline.py"
    )
    inputs["phase_01_source"] = file_hash(PHASE_01_DIR / "run_experiment.py")
    inputs["experiment_source"] = file_hash(Path(__file__))
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
