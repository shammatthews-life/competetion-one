#!/usr/bin/env python3
"""Verify saved raw-only experiment outputs without fitting any models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RUNS = {
    "baseline": "raw_baseline_20261101_154924",
    "phase01": "phase01_fee_product_20261101_171500",
    "phase02": "phase02_key_te_20261101_181500",
    "phase03": "phase03_xgb_cuda_20261101_190000",
}
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
FOLD_SEED = 42
N_SPLITS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_array(path: Path, expected_rows: int) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    require(values.ndim == 1, f"Expected a vector at {path}")
    require(len(values) == expected_rows, f"Unexpected row count at {path}")
    require(np.isfinite(values).all(), f"Non-finite predictions at {path}")
    require(
        ((values >= 0.0) & (values <= 1.0)).all(),
        f"Out-of-range probabilities at {path}",
    )
    return values.astype(np.float64, copy=False)


def verify_run_manifest(
    run_dir: Path,
    data_paths: dict[str, Path],
    dependency_paths: dict[str, Path],
) -> dict[str, Any]:
    path = run_dir / "sha256_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked_outputs = 0
    for relative_path, expected in manifest["outputs"].items():
        file_path = run_dir / relative_path
        require(file_path.is_file(), f"Manifest output missing: {file_path}")
        require(
            sha256_file(file_path) == expected["sha256"],
            f"Output SHA-256 mismatch: {file_path}",
        )
        require(
            file_path.stat().st_size == expected["size_bytes"],
            f"Output size mismatch: {file_path}",
        )
        checked_outputs += 1

    checked_inputs = 0
    for name, expected in manifest["inputs"].items():
        input_path = data_paths.get(name) or dependency_paths.get(name)
        require(input_path is not None, f"Unknown manifest input: {name}")
        require(input_path.is_file(), f"Required input missing: {input_path}")
        require(
            sha256_file(input_path) == expected["sha256"],
            f"Input SHA-256 mismatch: {input_path}",
        )
        require(
            input_path.stat().st_size == expected["size_bytes"],
            f"Input size mismatch: {input_path}",
        )
        checked_inputs += 1
    return {
        "output_records_checked": checked_outputs,
        "input_records_checked": checked_inputs,
        "manifest_sha256": sha256_file(path),
    }


def verify_baseline_manifest(
    run_dir: Path,
    data_paths: dict[str, Path],
    source_path: Path,
) -> dict[str, Any]:
    path = run_dir / "dependency_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_raw = {
        str(data_paths["train"]): data_paths["train"],
        str(data_paths["test"]): data_paths["test"],
        str(data_paths["sample"]): data_paths["sample"],
    }
    for raw_path, input_path in expected_raw.items():
        expected = manifest["raw_inputs"][raw_path]
        require(
            sha256_file(input_path) == expected["sha256"],
            f"Baseline raw-input SHA-256 mismatch: {input_path}",
        )
        require(
            input_path.stat().st_size == expected["size_bytes"],
            f"Baseline raw-input size mismatch: {input_path}",
        )
    expected_source = manifest["source"][str(source_path)]
    require(
        sha256_file(source_path) == expected_source["sha256"],
        "Baseline source SHA-256 mismatch",
    )
    for relative_path, expected in manifest["generated_outputs"].items():
        output_path = run_dir / relative_path
        require(output_path.is_file(), f"Baseline output missing: {output_path}")
        require(
            sha256_file(output_path) == expected["sha256"],
            f"Baseline output SHA-256 mismatch: {output_path}",
        )
        require(
            output_path.stat().st_size == expected["size_bytes"],
            f"Baseline output size mismatch: {output_path}",
        )
    return {
        "output_records_checked": len(manifest["generated_outputs"]),
        "input_records_checked": len(manifest["raw_inputs"])
        + len(manifest["source"]),
        "manifest_sha256": sha256_file(path),
    }


def verify_candidate(
    path: Path,
    test_ids: np.ndarray,
    expected_prediction: np.ndarray,
) -> dict[str, Any]:
    candidate = pd.read_csv(path)
    require(
        candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        f"Unexpected candidate columns: {path}",
    )
    ids = candidate[ID_COLUMN].to_numpy()
    predictions = candidate[TARGET_COLUMN].to_numpy(dtype=np.float64)
    require(len(ids) == len(test_ids), f"Candidate row count mismatch: {path}")
    require(ids.dtype == test_ids.dtype, f"Candidate ID dtype mismatch: {path}")
    require(np.array_equal(ids, test_ids), f"Candidate ID/order mismatch: {path}")
    require(len(np.unique(ids)) == len(ids), f"Duplicate candidate IDs: {path}")
    require(np.isfinite(predictions).all(), f"Non-finite CSV values: {path}")
    require(
        ((predictions >= 0.0) & (predictions <= 1.0)).all(),
        f"Out-of-range CSV values: {path}",
    )
    max_csv_rounding_error = float(np.max(np.abs(predictions - expected_prediction)))
    require(
        max_csv_rounding_error < 1e-9,
        f"CSV predictions do not match saved blend: {path}",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(candidate),
        "columns": candidate.columns.tolist(),
        "unique_ids": int(pd.Series(ids).nunique()),
        "max_csv_rounding_error": max_csv_rounding_error,
    }


def verify_fold_assignments(
    phase02_dir: Path,
    phase03_dir: Path,
    labels: np.ndarray,
) -> dict[str, Any]:
    expected_folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED
        ).split(np.zeros(len(labels)), labels)
    )
    phase03 = np.load(phase03_dir / "fold_assignments.npz", allow_pickle=False)
    for fold_number, (expected_fit, expected_valid) in enumerate(expected_folds, 1):
        phase02 = np.load(
            phase02_dir / f"fold_{fold_number}_indices.npz",
            allow_pickle=False,
        )
        require(
            np.array_equal(expected_fit, phase02["fit_indices"]),
            f"Phase 2 fit-fold mismatch in fold {fold_number}",
        )
        require(
            np.array_equal(expected_valid, phase02["valid_indices"]),
            f"Phase 2 validation-fold mismatch in fold {fold_number}",
        )
        require(
            np.array_equal(expected_fit, phase03[f"fit_{fold_number}"]),
            f"Phase 3 fit-fold mismatch in fold {fold_number}",
        )
        require(
            np.array_equal(expected_valid, phase03[f"valid_{fold_number}"]),
            f"Phase 3 validation-fold mismatch in fold {fold_number}",
        )
    return {
        "splits": N_SPLITS,
        "seed": FOLD_SEED,
        "phase02_and_phase03_match_expected": True,
    }


def verify_device_evidence(
    baseline_config: dict[str, Any],
    phase02_summary: dict[str, Any],
    phase03_summary: dict[str, Any],
) -> dict[str, Any]:
    models = baseline_config["models"]
    require(
        models["lightgbm"]["device"] == "CPU",
        "Baseline LightGBM config does not match the recorded CPU device",
    )
    require(
        models["catboost"]["task_type"] == "GPU"
        and models["catboost"]["devices"] == "0",
        "Baseline CatBoost config does not explicitly request GPU:0",
    )
    for model_name, expected_device in (
        ("lightgbm", "CPU"),
        ("catboost", "GPU:0"),
    ):
        records = phase02_summary["model_folds"][model_name]
        require(len(records) == N_SPLITS, f"Missing {model_name} fold records")
        require(
            all(record.get("device") == expected_device for record in records),
            f"Unexpected {model_name} device in Phase 2 fold records",
        )

    xgb_devices = []
    xgb_updaters = []
    for record in phase03_summary["fold_models"]:
        config = record["booster_config"]["learner"]
        device = config["generic_param"]["device"]
        updater = config["gradient_booster"]["gbtree_train_param"]["updater_seq"]
        require(device.startswith("cuda"), "XGBoost fold did not record CUDA")
        require("gpu" in updater, "XGBoost fold did not record GPU histogram updater")
        xgb_devices.append(device)
        xgb_updaters.append(updater)
    require(
        len(xgb_devices) == N_SPLITS,
        "Missing XGBoost fold device configuration records",
    )
    return {
        "lightgbm": "CPU",
        "catboost": "GPU:0",
        "xgboost_devices": xgb_devices,
        "xgboost_updaters": xgb_updaters,
        "gpu_before": phase03_summary.get("gpu_before"),
        "gpu_after": phase03_summary.get("gpu_after"),
    }


def run(project_dir: Path, data_dir: Path, output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    project_dir = project_dir.resolve()
    data_dir = data_dir.resolve()
    runs_dir = project_dir / "runs"
    run_dirs = {name: runs_dir / path for name, path in RUNS.items()}
    data_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    source_dir = project_dir / "source"
    phase01_source = (
        project_dir
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py"
    )
    phase02_source = (
        project_dir
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py"
    )
    phase03_source = (
        project_dir
        / "experiments"
        / "phase_03_xgboost_cuda"
        / "run_experiment.py"
    )
    dependency_paths = {
        "phase01": {},
        "phase02": {
            "baseline_metrics": run_dirs["baseline"] / "metrics.json",
            "phase_01_summary": run_dirs["phase01"] / "experiment_summary.json",
            "baseline_source": source_dir / "run_independent_baseline.py",
            "phase_01_source": phase01_source,
            "experiment_source": project_dir
            / "experiments"
            / "phase_02_fold_safe_key_encodings"
            / "run_experiment.py",
        },
        "phase03": {
            "phase_02_summary": run_dirs["phase02"] / "experiment_summary.json",
            "phase_02_source": phase02_source,
            "experiment_source": phase03_source,
        },
    }
    manifests = {
        "baseline": verify_baseline_manifest(
            run_dirs["baseline"],
            data_paths,
            source_dir / "run_independent_baseline.py",
        ),
        **{
            name: verify_run_manifest(
                run_dirs[name], data_paths, dependency_paths[name]
            )
            for name in ("phase01", "phase02", "phase03")
        },
    }

    train = pd.read_csv(data_paths["train"], usecols=[ID_COLUMN, TARGET_COLUMN])
    test = pd.read_csv(data_paths["test"], usecols=[ID_COLUMN])
    sample = pd.read_csv(data_paths["sample"])
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    test_ids = test[ID_COLUMN].to_numpy()
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Unexpected sample-submission schema",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_ids),
        "Sample and test IDs/order differ",
    )

    baseline_dir = run_dirs["baseline"]
    phase01_dir = run_dirs["phase01"]
    phase02_dir = run_dirs["phase02"]
    phase03_dir = run_dirs["phase03"]
    baseline_oof = load_array(
        baseline_dir / "oof_fixed_50_50_probability_blend.npy", len(labels)
    )
    phase01_lgb = load_array(phase01_dir / "oof_fee_lightgbm.npy", len(labels))
    phase01_cat = load_array(phase01_dir / "oof_fee_catboost.npy", len(labels))
    phase01_oof = load_array(
        phase01_dir / "oof_fee_fixed_50_50_blend.npy", len(labels)
    )
    phase01_expected = (phase01_lgb + phase01_cat) / 2.0
    require(
        np.allclose(phase01_oof, phase01_expected, rtol=0.0, atol=1e-7),
        "Phase 1 saved blend does not match model predictions",
    )

    phase02_lgb = load_array(phase02_dir / "oof_lightgbm.npy", len(labels))
    phase02_cat = load_array(phase02_dir / "oof_catboost.npy", len(labels))
    phase02_oof = load_array(phase02_dir / "oof_50_50_blend.npy", len(labels))
    phase02_expected = (phase02_lgb + phase02_cat) / 2.0
    require(
        np.allclose(phase02_oof, phase02_expected, rtol=0.0, atol=1e-7),
        "Phase 2 saved blend does not match model predictions",
    )

    phase03_xgb_oof = load_array(phase03_dir / "oof_xgboost.npy", len(labels))
    phase03_oof = load_array(
        phase03_dir / "oof_equal_3_model_blend.npy", len(labels)
    )
    phase03_expected = (
        phase02_lgb.astype(np.float64)
        + phase02_cat.astype(np.float64)
        + phase03_xgb_oof.astype(np.float64)
    ) / 3.0
    require(
        np.allclose(phase03_oof, phase03_expected, rtol=0.0, atol=1e-7),
        "Phase 3 OOF blend does not match its three component models",
    )

    phase01_lgb_test = load_array(
        phase01_dir / "test_lightgbm.npy", len(test_ids)
    )
    phase01_cat_test = load_array(
        phase01_dir / "test_catboost.npy", len(test_ids)
    )
    phase01_test = load_array(
        phase01_dir / "test_fee_fixed_50_50_blend.npy", len(test_ids)
    )
    phase01_test_expected = (phase01_lgb_test + phase01_cat_test) / 2.0
    require(
        np.allclose(phase01_test, phase01_test_expected, rtol=0.0, atol=1e-7),
        "Phase 1 saved test blend does not match model predictions",
    )

    phase02_lgb_test = load_array(
        phase02_dir / "test_lightgbm.npy", len(test_ids)
    )
    phase02_cat_test = load_array(
        phase02_dir / "test_catboost.npy", len(test_ids)
    )
    phase02_test = load_array(
        phase02_dir / "test_50_50_blend.npy", len(test_ids)
    )
    phase02_test_expected = (phase02_lgb_test + phase02_cat_test) / 2.0
    require(
        np.allclose(phase02_test, phase02_test_expected, rtol=0.0, atol=1e-7),
        "Phase 2 saved test blend does not match model predictions",
    )

    phase03_xgb_test = load_array(
        phase03_dir / "test_xgboost.npy", len(test_ids)
    )
    phase03_test = load_array(
        phase03_dir / "test_equal_3_model_blend.npy", len(test_ids)
    )
    phase03_test_expected = (
        phase02_lgb_test.astype(np.float64)
        + phase02_cat_test.astype(np.float64)
        + phase03_xgb_test.astype(np.float64)
    ) / 3.0
    require(
        np.allclose(phase03_test, phase03_test_expected, rtol=0.0, atol=1e-7),
        "Phase 3 test blend does not match its three component models",
    )

    candidates = [
        verify_candidate(
            phase01_dir / "submission_fee_features.csv",
            test_ids,
            phase01_test,
        ),
        verify_candidate(
            phase02_dir / "submission_key_target_encoding.csv",
            test_ids,
            phase02_test,
        ),
        verify_candidate(
            phase03_dir / "submission_equal_3_model.csv",
            test_ids,
            phase03_test,
        ),
    ]
    folds = verify_fold_assignments(phase02_dir, phase03_dir, labels)
    aucs = {
        "raw_baseline_50_50": float(roc_auc_score(labels, baseline_oof)),
        "phase01_fee_product_50_50": float(roc_auc_score(labels, phase01_oof)),
        "phase02_nested_te_50_50": float(roc_auc_score(labels, phase02_oof)),
        "phase03_xgboost_only": float(roc_auc_score(labels, phase03_xgb_oof)),
        "phase03_three_model_blend": float(roc_auc_score(labels, phase03_oof)),
    }
    baseline_config_path = baseline_dir / "config.json"
    baseline_config = json.loads(
        baseline_config_path.read_text(encoding="utf-8")
    )
    phase_summaries = {
        name: json.loads(
            (run_dirs[name] / "experiment_summary.json").read_text(
                encoding="utf-8"
            )
        )
        for name in ("phase01", "phase02", "phase03")
    }
    phase02_summary = phase_summaries["phase02"]
    phase03_summary = phase_summaries["phase03"]
    device_evidence = verify_device_evidence(
        baseline_config,
        phase02_summary,
        phase03_summary,
    )
    output_names = {
        "baseline": [
            "oof_fixed_50_50_probability_blend.npy",
            "test_fixed_50_50_probability_blend.npy",
            "submission_independent_baseline.csv",
        ],
        "phase01": [
            "oof_fee_fixed_50_50_blend.npy",
            "test_fee_fixed_50_50_blend.npy",
            "submission_fee_features.csv",
        ],
        "phase02": [
            "oof_50_50_blend.npy",
            "test_50_50_blend.npy",
            "submission_key_target_encoding.csv",
        ],
        "phase03": [
            "oof_xgboost.npy",
            "test_xgboost.npy",
            "oof_equal_3_model_blend.npy",
            "test_equal_3_model_blend.npy",
            "submission_equal_3_model.csv",
        ],
    }
    lineage_phases = []
    for name, run_dir in run_dirs.items():
        summary = phase_summaries.get(name, {})
        run_manifest_path = run_dir / (
            "dependency_manifest.json"
            if name == "baseline"
            else "sha256_manifest.json"
        )
        outputs = {
            relative: {
                "sha256": sha256_file(run_dir / relative),
                "size_bytes": (run_dir / relative).stat().st_size,
            }
            for relative in output_names[name]
        }
        source_paths = {
            "baseline": [source_dir / "run_independent_baseline.py"],
            "phase01": [phase01_source, source_dir / "run_independent_baseline.py"],
            "phase02": [
                phase02_source,
                phase01_source,
                source_dir / "run_independent_baseline.py",
            ],
            "phase03": [
                phase03_source,
                phase02_source,
                phase01_source,
                source_dir / "run_independent_baseline.py",
            ],
        }[name]
        runtime_seconds = summary.get("elapsed_seconds_total")
        if runtime_seconds is None:
            execution_path = run_dir / "execution.json"
            runtime_seconds = json.loads(
                execution_path.read_text(encoding="utf-8")
            ).get("elapsed_seconds")
        lineage_phases.append(
            {
                "run_id": run_dir.name,
                "lineage": summary.get(
                    "lineage", "INDEPENDENT_RAW_ONLY_BASELINE"
                ),
                "source_files": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in source_paths
                ],
                "historical_prediction_inputs": [],
                "candidate_outputs": outputs,
                "run_manifest_sha256": (
                    sha256_file(run_manifest_path)
                    if run_manifest_path.is_file()
                    else None
                ),
                "runtime_seconds": runtime_seconds,
                "official_competition_score": "NOT_EVALUATED",
            }
        )
    lineage = {
        "manifest_version": 1,
        "created_by": "source/validate_experiment_artifacts.py",
        "classification": (
            "INDEPENDENT_RAW_DATA_EXPERIMENT; NOT A HISTORICAL EXP85 REBUILD"
        ),
        "raw_inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in data_paths.items()
        },
        "folds": folds,
        "models": {
            "lightgbm": baseline_config["models"]["lightgbm"],
            "catboost": baseline_config["models"]["catboost"],
            "xgboost": phase03_summary["xgboost_configuration"],
        },
        "device_evidence": device_evidence,
        "phases": lineage_phases,
        "selected_candidate": {
            "phase": "phase02",
            "path": str(
                phase02_dir / "submission_key_target_encoding.csv"
            ),
            "sha256": sha256_file(
                phase02_dir / "submission_key_target_encoding.csv"
            ),
            "oof_auc": aucs["phase02_nested_te_50_50"],
            "prediction_dependencies": [
                "Five-fold LightGBM trained from raw-derived features",
                "Five-fold CatBoost trained from raw-derived features",
            ],
            "historical_prediction_dependencies": [],
        },
        "phase03_dependency_correction": {
            "run_summary_field": "comparison_only_prediction_inputs",
            "field_is_mislabeled_for_candidate_blend": True,
            "actual_use": (
                "Phase 2 LightGBM/CatBoost OOF and test predictions are "
                "averaged with XGBoost predictions to create the Phase 3 "
                "three-model candidate. They are raw-only Phase 2 outputs, "
                "not historical caches."
            ),
        },
        "official_competition_score": "NOT_EVALUATED",
    }
    lineage_path = project_dir / "MODEL_LINEAGE_MANIFEST.json"
    lineage_path.write_text(
        json.dumps(lineage, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evidence = {
        "status": "PASS",
        "models_retrained": False,
        "official_competition_score": "NOT_EVALUATED",
        "raw_rows": {"train": len(train), "test": len(test)},
        "folds": folds,
        "run_manifests": manifests,
        "device_evidence": device_evidence,
        "model_lineage_manifest": {
            "path": str(lineage_path),
            "sha256": sha256_file(lineage_path),
        },
        "candidates": candidates,
        "recomputed_oof_auc": aucs,
        "phase03_lineage_note": (
            "The XGBoost fold models use raw-derived Phase 2 features. "
            "The three-model candidate also consumes Phase 2 LightGBM/CatBoost "
            "OOF and test predictions; those are generated by the separate "
            "raw-only Phase 2 experiment, not historical prediction caches."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "VALIDATION_EVIDENCE.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.project_dir, args.data_dir, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
