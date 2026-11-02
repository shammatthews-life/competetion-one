#!/usr/bin/env python3
"""Screen CatBoost Lossguide against the frozen Phase 2 validation folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


EXPERIMENT_DIR = Path(__file__).resolve().parent
IMPROVEMENT_DIR = EXPERIMENT_DIR.parents[1]
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
FROZEN_PROJECT_DIR = (
    REPOSITORY_ROOT
    / "champion"
    / "EXP85_PHASE2_FROZEN_RELEASE"
    / "SOURCE"
    / "project"
)
PHASE_04_DIR = FROZEN_PROJECT_DIR / "experiments" / "phase_04_final_validation_gate"
REFERENCE_DIR = (
    REPOSITORY_ROOT
    / "champion"
    / "EXP85_PHASE2_FROZEN_RELEASE"
    / "EVIDENCE"
    / "second_seed"
    / "main"
)
DATA_DIR = Path(r"D:\gci\inpuy")
CHAMPION_REFERENCE = (
    IMPROVEMENT_DIR
    / "baseline"
    / "reference"
    / "EXP85_HISTORICAL_CHAMPION.csv"
)
OUTER_SEED = 20261101
N_SPLITS = 5
EXPECTED_CHAMPION_SHA256 = (
    "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
)
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}

sys.path.insert(0, str(FROZEN_PROJECT_DIR / "source"))
sys.path.insert(0, str(PHASE_04_DIR))

from run_final_gate import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    GpuMonitor,
    build_rowwise_features,
    make_variant_fold,
)


MODEL_CONFIG = {
    "iterations": 4000,
    "depth": 8,
    "grow_policy": "Lossguide",
    "max_leaves": 32,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 6.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "border_count": 64,
    "task_type": "GPU",
    "devices": "0",
    "gpu_ram_part": 0.78,
    "early_stopping_rounds": 200,
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
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def validate_probabilities(values: np.ndarray, expected_rows: int, name: str) -> None:
    require(values.ndim == 1, f"{name} must be a one-dimensional array")
    require(len(values) == expected_rows, f"{name} row count is incorrect")
    require(np.isfinite(values).all(), f"{name} contains non-finite values")
    require(
        ((values >= 0.0) & (values <= 1.0)).all(),
        f"{name} contains values outside [0, 1]",
    )


def validate_submission(
    candidate: pd.DataFrame,
    sample: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    require(
        candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Candidate submission schema is invalid",
    )
    require(
        candidate[ID_COLUMN].is_unique,
        "Candidate submission contains duplicate IDs",
    )
    require(
        np.array_equal(
            candidate[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()
        ),
        "Candidate IDs/order differ from sample submission",
    )
    values = candidate[TARGET_COLUMN].to_numpy(dtype=np.float64)
    validate_probabilities(values, len(sample), "Serialized submission")
    require(
        np.max(np.abs(values - predictions)) < 1e-9,
        "Serialized submission predictions differ from saved test predictions",
    )


def model_parameters(seed: int) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in MODEL_CONFIG.items()
        if key != "early_stopping_rounds"
    }
    parameters.update(
        {
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    return parameters


def inner_early_stopping_split(
    labels: np.ndarray,
    outer_fit_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=seed,
    )
    inner_fit_local, inner_valid_local = next(
        split.split(
            np.zeros(len(outer_fit_indices)),
            labels[outer_fit_indices],
        )
    )
    inner_fit = outer_fit_indices[inner_fit_local]
    inner_valid = outer_fit_indices[inner_valid_local]
    require(
        not np.intersect1d(inner_fit, inner_valid).size,
        "Inner early-stopping fit and validation rows overlap",
    )
    require(
        np.isin(inner_fit, outer_fit_indices).all()
        and np.isin(inner_valid, outer_fit_indices).all(),
        "Inner early-stopping split escaped the outer fit rows",
    )
    return inner_fit, inner_valid


def summarize_predictions(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def compare_oof(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    validate_probabilities(candidate, len(labels), "Candidate OOF")
    validate_probabilities(reference, len(labels), "Phase 2 OOF reference")
    candidate_fold_auc = []
    reference_fold_auc = []
    for _, valid_indices in folds:
        candidate_fold_auc.append(
            float(roc_auc_score(labels[valid_indices], candidate[valid_indices]))
        )
        reference_fold_auc.append(
            float(roc_auc_score(labels[valid_indices], reference[valid_indices]))
        )
    deltas = [
        candidate_auc - reference_auc
        for candidate_auc, reference_auc in zip(
            candidate_fold_auc, reference_fold_auc
        )
    ]
    candidate_auc = float(roc_auc_score(labels, candidate))
    reference_auc = float(roc_auc_score(labels, reference))
    pooled_delta = candidate_auc - reference_auc
    advances = pooled_delta >= 0.001 and sum(delta > 0 for delta in deltas) >= 4
    return {
        "candidate_oof_auc": candidate_auc,
        "phase2_reference_oof_auc": reference_auc,
        "pooled_auc_delta": pooled_delta,
        "candidate_fold_auc": candidate_fold_auc,
        "phase2_reference_fold_auc": reference_fold_auc,
        "paired_fold_auc_delta": deltas,
        "positive_folds": sum(delta > 0 for delta in deltas),
        "advancement_gate": {
            "rule": "pooled OOF AUC delta >= 0.001 and positive on >= 4/5 paired folds",
            "result": "ADVANCE_TO_INDEPENDENT_SEED" if advances else "REJECT_SCREEN",
        },
    }


def load_phase2_reference(
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    expected_inputs: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = REFERENCE_DIR / "experiment_summary.json"
    manifest_path = REFERENCE_DIR / "sha256_manifest.json"
    oof_path = REFERENCE_DIR / "oof_phase2_fold_local_blend.npy"
    require(summary_path.is_file(), f"Phase 2 summary is missing: {summary_path}")
    require(manifest_path.is_file(), f"Phase 2 manifest is missing: {manifest_path}")
    require(oof_path.is_file(), f"Phase 2 OOF reference is missing: {oof_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(summary["status"] == "COMPLETED", "Phase 2 reference did not complete")
    require(
        summary["outer_seed"] == OUTER_SEED,
        "Phase 2 comparison reference uses a different seed",
    )
    require(
        summary["historical_champion_read"] is False,
        "Phase 2 reference unexpectedly read the champion during training",
    )
    for name, actual in expected_inputs.items():
        expected = manifest["inputs"][name]
        require(
            actual["sha256"] == expected["sha256"],
            f"Phase 2 reference uses different raw {name} data",
        )
    expected_oof = manifest["outputs"]["oof_phase2_fold_local_blend.npy"]
    require(
        sha256_file(oof_path) == expected_oof["sha256"],
        "Phase 2 OOF reference hash differs from its manifest",
    )
    reference = np.load(oof_path, allow_pickle=False)
    validate_probabilities(reference, len(labels), "Phase 2 OOF reference")

    reference_folds = []
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        fold_path = REFERENCE_DIR / f"fold_{fold_number}_indices.npz"
        expected = manifest["outputs"][fold_path.name]
        require(
            sha256_file(fold_path) == expected["sha256"],
            f"Phase 2 fold {fold_number} hash differs from its manifest",
        )
        with np.load(fold_path, allow_pickle=False) as fold_archive:
            reference_folds.append(
                (
                    fold_archive["fit_indices"],
                    fold_archive["valid_indices"],
                )
            )
        require(
            np.array_equal(reference_folds[-1][0], fit_indices)
            and np.array_equal(reference_folds[-1][1], valid_indices),
            f"Phase 2 fold {fold_number} differs from regenerated split",
        )
    return reference, {
        "path": str(oof_path),
        "sha256": expected_oof["sha256"],
        "classification": "OOF validation comparator only; never a model input",
        "outer_seed": summary["outer_seed"],
        "folds_verified": len(reference_folds),
        "fold_seed_match": True,
    }


def validate_gpu_preflight() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        require(len(fields) == 4, f"Unexpected nvidia-smi output: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_free_mib": int(fields[3]),
            }
        )
    require(rows, "No NVIDIA GPU was reported")
    gpu = next((row for row in rows if row["index"] == 0), None)
    require(gpu is not None, "GPU device 0 is unavailable")
    require(
        gpu["memory_free_mib"] >= 5000,
        f"GPU 0 has insufficient free VRAM for this run: {gpu}",
    )
    require(
        "RTX 4050" in gpu["name"],
        f"Expected RTX 4050 for this experiment, found {gpu['name']}",
    )
    return gpu


def run_experiment(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.perf_counter()
    input_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in input_paths.values():
        require(path.is_file(), f"Required raw input is missing: {path}")
    input_records = {name: file_record(path) for name, path in input_paths.items()}
    for name, record in input_records.items():
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[name],
            f"Unexpected raw {name} input hash: {record['sha256']}",
        )

    require(
        CHAMPION_REFERENCE.is_file(),
        f"Historical champion reference is missing: {CHAMPION_REFERENCE}",
    )
    champion_hash = sha256_file(CHAMPION_REFERENCE)
    require(
        champion_hash == EXPECTED_CHAMPION_SHA256,
        "Frozen historical champion hash differs from the verified reference",
    )
    gpu_before = validate_gpu_preflight()

    train_raw = pd.read_csv(input_paths["train"])
    test_raw = pd.read_csv(input_paths["test"])
    sample = pd.read_csv(input_paths["sample"])
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Unexpected sample-submission columns",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()),
        "Test IDs/order differ from sample-submission IDs/order",
    )
    require(
        train_raw[ID_COLUMN].is_unique and test_raw[ID_COLUMN].is_unique,
        "Duplicate applicant IDs found",
    )
    require(
        set(train_raw[ID_COLUMN]).isdisjoint(set(test_raw[ID_COLUMN])),
        "Training and test IDs overlap",
    )
    labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(
        set(np.unique(labels)).issubset({0, 1}),
        "Training labels are not binary",
    )
    train_features = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    test_features = build_rowwise_features(test_raw)
    require(
        train_features.columns.tolist() == test_features.columns.tolist(),
        "Raw feature train/test schemas differ",
    )
    require(
        ID_COLUMN not in train_features and TARGET_COLUMN not in train_features,
        "ID or target reached raw feature generation",
    )

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED
        ).split(np.zeros(len(labels)), labels)
    )
    phase2_oof, phase2_reference_record = load_phase2_reference(
        labels, folds, input_records
    )
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    test_fold_predictions: list[np.ndarray] = []
    fold_records = []
    fold_arrays: dict[str, np.ndarray] = {}
    importance_frames = []
    telemetry_samples = []
    feature_names: list[str] | None = None

    log_message(
        log_path,
        f"Starting {run_dir.name}: rows={len(labels)} test_rows={len(test_raw)} "
        f"GPU={gpu_before['name']} VRAM_free={gpu_before['memory_free_mib']}MiB",
    )
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        require(
            not np.intersect1d(fit_indices, valid_indices).size,
            f"Fold {fold_number} has overlapping fit/validation rows",
        )
        (
            cat_train,
            cat_test,
            _,
            _,
            feature_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            train_features,
            test_features,
            fit_indices,
            valid_indices,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if feature_names is None:
            feature_names = cat_train.columns.tolist()
        require(
            cat_train.columns.tolist() == feature_names
            and cat_test.columns.tolist() == feature_names,
            f"Fold {fold_number} feature schema differs",
        )
        require(
            ID_COLUMN not in feature_names and TARGET_COLUMN not in feature_names,
            "ID or target leaked into the model feature schema",
        )
        numeric = cat_train.select_dtypes(include=[np.number])
        numeric_values = numeric.to_numpy(dtype=np.float64, copy=False)
        require(
            not np.isinf(numeric_values).any(),
            f"Fold {fold_number} model features contain infinity",
        )
        require(
            all(column in cat_train for column in CATEGORICAL_COLUMNS),
            f"Fold {fold_number} is missing native categorical fields",
        )

        inner_fit_indices, inner_valid_indices = inner_early_stopping_split(
            labels,
            fit_indices,
            OUTER_SEED + 1000 + fold_number,
        )
        (
            inner_cat_train,
            _,
            _,
            _,
            inner_feature_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            train_features,
            test_features,
            inner_fit_indices,
            inner_valid_indices,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        require(
            inner_cat_train.columns.tolist() == feature_names,
            f"Fold {fold_number} inner feature schema differs",
        )
        inner_seed = OUTER_SEED + 100_000 + fold_number
        inner_model = CatBoostClassifier(**model_parameters(inner_seed))
        fold_started = time.perf_counter()
        with GpuMonitor() as monitor:
            inner_model.fit(
                inner_cat_train.iloc[inner_fit_indices],
                labels[inner_fit_indices],
                cat_features=list(CATEGORICAL_COLUMNS),
                eval_set=(
                    inner_cat_train.iloc[inner_valid_indices],
                    labels[inner_valid_indices],
                ),
                early_stopping_rounds=MODEL_CONFIG["early_stopping_rounds"],
                use_best_model=True,
                verbose=False,
            )
            selected_iterations = max(int(inner_model.best_iteration_) + 1, 1)
            final_parameters = model_parameters(OUTER_SEED + fold_number)
            final_parameters["iterations"] = selected_iterations
            final_model = CatBoostClassifier(**final_parameters)
            final_model.fit(
                cat_train.iloc[fit_indices],
                labels[fit_indices],
                cat_features=list(CATEGORICAL_COLUMNS),
                verbose=False,
            )
        require(
            not monitor.errors,
            f"GPU telemetry failed in fold {fold_number}: "
            + "; ".join(monitor.errors),
        )
        require(
            any(sample["name"] == gpu_before["name"] for sample in monitor.samples),
            f"RTX 4050 telemetry was not captured during fold {fold_number}",
        )
        effective = final_model.get_all_params()
        require(
            effective.get("grow_policy") == "Lossguide",
            f"Fold {fold_number} did not use Lossguide: {effective.get('grow_policy')}",
        )
        require(
            effective.get("task_type") == "GPU",
            f"Fold {fold_number} did not train on GPU: {effective.get('task_type')}",
        )
        require(
            int(effective.get("iterations", 0)) == selected_iterations,
            f"Fold {fold_number} final refit iteration count differs",
        )
        valid_prediction = final_model.predict_proba(
            cat_train.iloc[valid_indices]
        )[:, 1].astype(np.float32)
        test_prediction = final_model.predict_proba(cat_test)[
            :, 1
        ].astype(np.float32)
        validate_probabilities(
            valid_prediction, len(valid_indices), f"Fold {fold_number} OOF"
        )
        validate_probabilities(
            test_prediction, len(test_raw), f"Fold {fold_number} test"
        )
        oof[valid_indices] = valid_prediction
        test_fold_predictions.append(test_prediction)
        fold_arrays[f"fit_{fold_number}"] = fit_indices.astype(np.int32)
        fold_arrays[f"valid_{fold_number}"] = valid_indices.astype(np.int32)
        fold_importance = final_model.get_feature_importance(
            type="FeatureImportance"
        )
        require(
            len(fold_importance) == len(feature_names),
            f"Fold {fold_number} feature-importance schema mismatch",
        )
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": np.asarray(fold_importance, dtype=np.float64),
                }
            )
        )
        telemetry_samples.extend(monitor.samples)
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(valid_indices)),
                "fit_validation_ids_disjoint": True,
                "outer_validation_used_for_iteration_selection": False,
                "inner_fit_rows": int(len(inner_fit_indices)),
                "inner_validation_rows": int(len(inner_valid_indices)),
                "inner_fit_outer_validation_overlap": 0,
                "inner_seed": inner_seed,
                "inner_best_iteration": int(inner_model.best_iteration_),
                "final_fit_iterations": selected_iterations,
                "auc": float(roc_auc_score(labels[valid_indices], valid_prediction)),
                "elapsed_seconds": time.perf_counter() - fold_started,
                "gpu": {
                    "device": effective.get("task_type"),
                    "name": gpu_before["name"],
                    "telemetry_samples": len(monitor.samples),
                    "max_utilization_percent": max(
                        sample["utilization_percent"] for sample in monitor.samples
                    ),
                    "max_memory_used_mib": max(
                        sample["memory_used_mib"] for sample in monitor.samples
                    ),
                },
                "feature_protocol": feature_audit,
                "inner_feature_protocol": inner_feature_audit,
            }
        )
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=fit_indices.astype(np.int32),
            valid_indices=valid_indices.astype(np.int32),
        )
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_predictions.npz",
            valid_indices=valid_indices.astype(np.int32),
            oof=valid_prediction,
            test=test_prediction,
        )
        log_message(
            log_path,
            f"Fold {fold_number}/{N_SPLITS}: AUC="
            f"{fold_records[-1]['auc']:.8f} inner_best_iteration="
            f"{fold_records[-1]['inner_best_iteration']} final_iterations="
            f"{selected_iterations} gpu_max="
            f"{fold_records[-1]['gpu']['max_utilization_percent']}% "
            f"vram_max={fold_records[-1]['gpu']['max_memory_used_mib']}MiB",
        )
        del inner_model, final_model

    validate_probabilities(oof, len(labels), "Complete candidate OOF")
    test_prediction = np.mean(
        np.stack(test_fold_predictions), axis=0, dtype=np.float64
    ).astype(np.float32)
    validate_probabilities(test_prediction, len(test_raw), "Candidate test")
    np.save(run_dir / "oof_catboost_lossguide.npy", oof)
    np.save(run_dir / "test_catboost_lossguide.npy", test_prediction)
    np.savez_compressed(run_dir / "fold_indices.npz", **fold_arrays)

    comparison = compare_oof(labels, oof, phase2_oof, folds)
    importance = (
        pd.concat(importance_frames)
        .groupby("feature", sort=False)["importance"]
        .mean()
        .sort_values(ascending=False)
        .rename("mean_importance")
        .reset_index()
    )
    importance.to_csv(run_dir / "feature_importance.csv", index=False)
    feature_list_path = run_dir / "feature_columns.json"
    write_json(feature_list_path, feature_names or [])

    candidate = pd.DataFrame(
        {
            ID_COLUMN: test_raw[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: test_prediction,
        }
    )
    validate_submission(candidate, sample, test_prediction)
    submission_path = run_dir / "submission_catboost_lossguide.csv"
    candidate.to_csv(submission_path, index=False, float_format="%.10f")
    parsed_candidate = pd.read_csv(submission_path)
    validate_submission(parsed_candidate, sample, test_prediction)

    champion = pd.read_csv(CHAMPION_REFERENCE)
    require(
        champion.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Frozen champion reference schema differs from submission",
    )
    require(
        np.array_equal(champion[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()),
        "Frozen champion reference IDs/order differ from test data",
    )
    champion_predictions = champion[TARGET_COLUMN].to_numpy(dtype=np.float64)
    validate_probabilities(
        champion_predictions, len(test_raw), "Historical champion reference"
    )
    pearson = float(np.corrcoef(test_prediction, champion_predictions)[0, 1])
    spearman = float(
        spearmanr(test_prediction, champion_predictions).statistic
    )
    rank_displacement = np.abs(
        rankdata(test_prediction, method="average")
        - rankdata(champion_predictions, method="average")
    )

    telemetry_path = run_dir / "gpu_telemetry.json"
    write_json(telemetry_path, telemetry_samples)
    source_paths = {
        "experiment_runner": Path(__file__).resolve(),
        "phase2_raw_feature_builder": FROZEN_PROJECT_DIR
        / "source"
        / "run_independent_baseline.py",
        "phase1_fee_product_features": FROZEN_PROJECT_DIR
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py",
        "phase2_target_encodings": FROZEN_PROJECT_DIR
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py",
        "phase2_fold_local_pipeline": PHASE_04_DIR / "run_final_gate.py",
    }
    for name, path in source_paths.items():
        require(path.is_file(), f"Source file missing for hash record: {path}")
        source_paths[name] = path.resolve()
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "lineage": "INDEPENDENT_RAW_ONLY_CATBOOST_LOSSGUIDE",
        "official_competition_score": "NOT_EVALUATED",
        "historical_public_score_reference": 0.801340,
        "historical_public_score_verified_for_candidate": False,
        "raw_data_used_for_training": True,
        "historical_predictions_used_for_training": False,
        "historical_predictions_used_as_labels": False,
        "champion_reference_read_for_post_training_diagnostic_only": True,
        "champion_reference_sha256": champion_hash,
        "inputs": input_records,
        "phase2_comparison_reference": phase2_reference_record,
        "folds": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "seed": OUTER_SEED,
            "matched_phase2_seed_and_indices": True,
            "fit_validation_ids_disjoint": True,
        },
        "validation_protocol": {
            "outer_validation_used_for_iteration_selection": False,
            "iteration_selection": (
                "Stratified 15% inner split of outer-fit rows; final model refit "
                "on all outer-fit rows at the selected iteration count"
            ),
            "outer_validation_labels_used_for_training_or_feature_generation": False,
        },
        "feature_generation": {
            "base": "Frozen Phase 2 rowwise features; no labels or other-row values",
            "additional_features": "Fold-local frequency/product features and nested target encodings",
            "validation_frequency_reference": "Outer-fit rows only",
            "training_target_encodings": "Nested inner OOF; no self-target contribution",
            "validation_test_target_encodings": "Outer-fit labels only",
            "transductive_test_covariates_in_frequency_reference": False,
            "feature_count": len(feature_names or []),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "feature_names_sha256": hashlib.sha256(
                json.dumps(feature_names, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "model": {
            "family": "CatBoost GPU, asymmetric Lossguide trees",
            "parameters": MODEL_CONFIG,
            "fold_seeds": [OUTER_SEED + fold for fold in range(1, N_SPLITS + 1)],
            "gpu_preflight": gpu_before,
            "gpu_training_confirmed_by_catboost": all(
                item["gpu"]["device"] == "GPU" for item in fold_records
            ),
            "gpu_telemetry": {
                "sample_count": len(telemetry_samples),
                "max_utilization_percent": max(
                    sample["utilization_percent"] for sample in telemetry_samples
                ),
                "max_memory_used_mib": max(
                    sample["memory_used_mib"] for sample in telemetry_samples
                ),
            },
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
        "fold_results": fold_records,
        "validation": comparison,
        "prediction_diagnostics": {
            "test": summarize_predictions(test_prediction),
            "historical_champion_similarity": {
                "classification": "Post-training diagnostic only; not a tuning target",
                "pearson": pearson,
                "spearman": spearman,
                "mean_rank_displacement": float(np.mean(rank_displacement)),
                "maximum_rank_displacement": float(np.max(rank_displacement)),
            },
        },
        "candidate_submission": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
            "rows": int(len(candidate)),
            "columns": candidate.columns.tolist(),
            "unique_ids": int(candidate[ID_COLUMN].nunique()),
            "prediction_min": float(test_prediction.min()),
            "prediction_max": float(test_prediction.max()),
        },
        "outputs": {},
        "source_hashes": {
            name: {"path": str(path), **file_record(path)}
            for name, path in source_paths.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name in (
        "oof_catboost_lossguide.npy",
        "test_catboost_lossguide.npy",
        "fold_indices.npz",
        "feature_importance.csv",
        "feature_columns.json",
        "gpu_telemetry.json",
        "submission_catboost_lossguide.csv",
    ):
        summary["outputs"][name] = file_record(run_dir / name)
    write_json(run_dir / "experiment_summary.json", summary)
    log_message(
        log_path,
        f"Completed: candidate_auc={comparison['candidate_oof_auc']:.8f} "
        f"phase2_auc={comparison['phase2_reference_oof_auc']:.8f} "
        f"delta={comparison['pooled_auc_delta']:+.8f} "
        f"gate={comparison['advancement_gate']['result']}",
    )

    manifest = {
        "experiment_id": run_dir.name,
        "inputs": {
            **input_records,
            "phase2_oof_reference": file_record(
                REFERENCE_DIR / "oof_phase2_fold_local_blend.npy"
            ),
            "historical_champion_diagnostic_reference": file_record(
                CHAMPION_REFERENCE
            ),
        },
        "sources": {
            name: {"path": str(path), **file_record(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            str(path.relative_to(run_dir)): file_record(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    }
    write_json(run_dir / "sha256_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    default_run_dir = EXPERIMENT_DIR / "runs" / (
        "exp04_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_catboost_lossguide_seed20261101"
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()
