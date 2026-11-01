from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_validation import load_inputs, sha256_file
from folds import build_folds, fold_id_vector
from gpu_telemetry import gpu_snapshot, require_gpu_preflight
from phase1_features import build_decoded_loan_features
from phase2_target_encoding import (
    INNER_SEED,
    INNER_SPLITS,
    KEY_SPECS,
    apply_training_encodings,
    key_arrays,
    nested_training_encodings,
)
from prediction_blender import combine_phase2_predictions, mean_fold_predictions
from preprocessing import CATEGORICAL_COLUMNS, ID_COLUMN, TARGET_COLUMN, build_rowwise_features, make_lightgbm_frames
from submission_builder import build_submission, validate_submission
from train_catboost import train_catboost_fold
from train_lightgbm import train_lightgbm_fold


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
FROZEN_DIR = PACKAGE_DIR.parent / "EXP85_PHASE2_FROZEN_RELEASE"
REFERENCE_DIR = FROZEN_DIR / "EVIDENCE" / "original_runs" / "phase_02"
REFERENCE_CANDIDATE = FROZEN_DIR / "CANDIDATE" / "submission_key_target_encoding.csv"
EXPECTED_CANDIDATE_SHA256 = (
    "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_inventory(paths: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(paths):
        result[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "scipy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "MISSING"
    return result


def git_evidence() -> dict[str, Any]:
    def command(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return f"ERROR: {type(error).__name__}: {error}"
        return result.stdout.strip()

    return {
        "repository": str(REPO_ROOT),
        "commit": command(["rev-parse", "HEAD"]),
        "branch": command(["branch", "--show-current"]),
        "status": command(["status", "--short"]),
    }


def verify_frozen_reference(data_dir: Path) -> dict[str, Any]:
    if not REFERENCE_DIR.is_dir() or not REFERENCE_CANDIDATE.is_file():
        raise FileNotFoundError(
            "Frozen Phase 2 reference artifacts are missing; reproduction gate cannot run."
        )
    release_manifest = read_json(FROZEN_DIR / "FREEZE_MANIFEST.json")
    candidate_hash = sha256_file(REFERENCE_CANDIDATE)
    if candidate_hash != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Frozen Phase 2 candidate hash changed.")

    reference_manifest_path = REFERENCE_DIR / "sha256_manifest.json"
    reference_manifest = read_json(reference_manifest_path)
    for relative, record in reference_manifest["outputs"].items():
        path = REFERENCE_DIR / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen Phase 2 reference artifact hash mismatch: {path}")
    expected_input = release_manifest["raw_inputs"]
    for filename, path in (
        ("train.csv", data_dir / "train.csv"),
        ("test.csv", data_dir / "test.csv"),
        ("sample_submission.csv", data_dir / "sample_submission.csv"),
    ):
        expected = expected_input[filename]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Raw {filename} differs from the frozen Phase 2 input hash."
            )
    summary = read_json(REFERENCE_DIR / "experiment_summary.json")
    return {
        "frozen_candidate_sha256": candidate_hash,
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "frozen_release_manifest_sha256": sha256_file(FROZEN_DIR / "FREEZE_MANIFEST.json"),
        "candidate_oof_auc": summary["auc_comparison"]["blend"][
            "full_oof_auc_fee_plus_nested_te"
        ],
        "candidate_oof_by_fold": [
            row["auc"]["fee_plus_nested_te"]["blend"]
            for row in summary["paired_cv_by_fold"]
        ],
        "reference_files_verified": len(reference_manifest["outputs"]),
        "raw_inputs_match_frozen_hashes": True,
    }


def load_reference_arrays() -> tuple[dict[str, np.ndarray], np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    arrays = {
        name: np.load(REFERENCE_DIR / f"oof_{name}.npy", allow_pickle=False)
        for name in ("lightgbm", "catboost")
    }
    arrays["blend"] = np.load(
        REFERENCE_DIR / "oof_50_50_blend.npy", allow_pickle=False
    )
    reference_test = np.load(
        REFERENCE_DIR / "test_50_50_blend.npy", allow_pickle=False
    )
    folds = []
    for fold_number in range(1, 6):
        with np.load(REFERENCE_DIR / f"fold_{fold_number}_indices.npz", allow_pickle=False) as archive:
            folds.append((archive["fit_indices"].copy(), archive["valid_indices"].copy()))
    return arrays, reference_test, folds


def prediction_comparison(
    current: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape or current.ndim != 1:
        raise ValueError("Reference and current prediction shapes differ.")
    delta = current - reference
    correlation = float(np.corrcoef(current, reference)[0, 1])
    return {
        "pearson": correlation,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "maximum_absolute_difference": float(np.max(np.abs(delta))),
        "current_mean": float(current.mean()),
        "reference_mean": float(reference.mean()),
        "current_std": float(current.std()),
        "reference_std": float(reference.std()),
    }


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    mode: str = "full",
    run_id: str | None = None,
) -> dict[str, Any]:
    if mode not in {"full", "reproduce"}:
        raise ValueError("Mode must be 'full' or 'reproduce'.")
    data_dir = data_dir.resolve()
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_phase2_complete")
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Run ID may contain only letters, digits, underscore, and hyphen.")
    run_dir = PACKAGE_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in ("models", "predictions", "submissions", "reports"):
        (run_dir / relative).mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    source_files = sorted((PACKAGE_DIR / "source").glob("*.py"))
    config_files = sorted((PACKAGE_DIR / "config").glob("*.json"))
    authoritative_sources = [
        FROZEN_DIR / "SOURCE" / "project" / "source" / "run_independent_baseline.py",
        FROZEN_DIR
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py",
        FROZEN_DIR
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py",
    ]
    run_state: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "status": "RUNNING",
        "started_at_utc": started_at,
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "git": git_evidence(),
        "software_versions": package_versions(),
        "source_hashes": file_inventory(source_files, PACKAGE_DIR),
        "authoritative_source_hashes": file_inventory(
            authoritative_sources, FROZEN_DIR / "SOURCE" / "project"
        ),
        "config_hashes": file_inventory(config_files, PACKAGE_DIR),
    }
    try:
        model_config = read_json(PACKAGE_DIR / "config" / "model_config.json")
        feature_config = read_json(PACKAGE_DIR / "config" / "feature_config.json")
        ensemble_config = read_json(PACKAGE_DIR / "config" / "ensemble_config.json")
        expected_hashes = {
            name: value["sha256"]
            for name, value in read_json(FROZEN_DIR / "HASHES" / "INPUT_HASHES.json")[
                "inputs"
            ].items()
        }
        reference_meta = verify_frozen_reference(data_dir)
        gpu_before = gpu_snapshot()
        if model_config["catboost"]["task_type"] == "GPU":
            gpu_preflight = require_gpu_preflight()
        else:
            raise ValueError("Frozen Phase 2 requires CatBoost GPU training.")

        train_raw, test_raw, sample, labels, input_metadata = load_inputs(
            data_dir, expected_hashes
        )
        if feature_config["fee_decoder"]["target_used"] is not False:
            raise ValueError("Fee feature configuration must be label-free.")
        if feature_config["fee_decoder"]["frequency_counts_use_train_and_test_covariates"] is not True:
            raise ValueError("Fee-frequency configuration differs from selected Phase 2.")
        expected_keys = [
            (item["name"], item["source_feature"], float(item["smoothing"]))
            for item in feature_config["target_encoding"]["keys"]
        ]
        if expected_keys != list(KEY_SPECS):
            raise ValueError("Target-encoding configuration differs from source implementation.")
        if tuple(feature_config["categorical_columns"]) != CATEGORICAL_COLUMNS:
            raise ValueError("Categorical feature configuration differs from source implementation.")
        if float(ensemble_config["lightgbm_weight"]) != 0.5 or float(
            ensemble_config["catboost_weight"]
        ) != 0.5:
            raise ValueError("The frozen Phase 2 ensemble is a 50/50 probability blend.")

        base_train = build_rowwise_features(train_raw.drop(columns=[TARGET_COLUMN]))
        base_test = build_rowwise_features(test_raw)
        fee_train, fee_test, feature_diagnostics = build_decoded_loan_features(
            train_raw,
            test_raw,
            pooled_frequency_reference=True,
        )
        base_train = pd.concat([base_train, fee_train], axis=1)
        base_test = pd.concat([base_test, fee_test], axis=1)
        keys_by_name = key_arrays(fee_train, fee_test)
        folds = build_folds(
            labels,
            n_splits=int(model_config["outer_cv"]["n_splits"]),
            seed=int(model_config["outer_cv"]["seed"]),
        )
        reference_oof, reference_test, reference_folds = load_reference_arrays()
        fold_identity = True
        for current, reference in zip(folds, reference_folds):
            if not (
                np.array_equal(current[0], reference[0])
                and np.array_equal(current[1], reference[1])
            ):
                fold_identity = False
                break
        if not fold_identity:
            raise ValueError("New fold assignments do not exactly match frozen Phase 2.")
        n_train = len(train_raw)
        test_indices = np.arange(n_train, n_train + len(test_raw), dtype=np.int64)
        fold_ids = fold_id_vector(labels, folds)
        np.savez_compressed(
            run_dir / "fold_assignments.npz",
            fold_id=fold_ids,
            **{
                f"fold_{index}_fit": fit.astype(np.int32)
                for index, (fit, _) in enumerate(folds, 1)
            },
            **{
                f"fold_{index}_valid": valid.astype(np.int32)
                for index, (_, valid) in enumerate(folds, 1)
            },
        )

        oof = {
            name: np.full(n_train, np.nan, dtype=np.float32)
            for name in ("lightgbm", "catboost")
        }
        fold_test_predictions: dict[str, list[np.ndarray]] = {
            name: [] for name in ("lightgbm", "catboost")
        }
        fold_records = []
        feature_schema: dict[str, Any] | None = None
        for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
            fold_started = time.perf_counter()
            train_te = nested_training_encodings(keys_by_name, labels, fit_indices)
            valid_te = apply_training_encodings(
                keys_by_name, labels, fit_indices, valid_indices
            )
            test_te = apply_training_encodings(
                keys_by_name, labels, fit_indices, test_indices
            )
            if np.intersect1d(fit_indices, valid_indices).size:
                raise RuntimeError("Outer target-encoding fit/validation indices overlap.")
            if not train_te.index.equals(pd.RangeIndex(len(fit_indices))):
                raise RuntimeError("Nested encodings are not aligned to outer fit indices.")
            X_train_fold = base_train.copy()
            X_train_fold.loc[fit_indices, train_te.columns] = train_te.to_numpy()
            X_train_fold.loc[valid_indices, valid_te.columns] = valid_te.to_numpy()
            X_test_fold = pd.concat([base_test, test_te], axis=1)
            if np.isinf(
                X_train_fold.select_dtypes(include=[np.number]).to_numpy(
                    dtype=np.float64, copy=False
                )
            ).any():
                raise ValueError("Fold feature matrix contains infinite values.")
            X_lgb_train, X_lgb_test = make_lightgbm_frames(X_train_fold, X_test_fold)
            if feature_schema is None:
                feature_schema = {
                    "catboost_features": [
                        {"name": name, "dtype": str(dtype)}
                        for name, dtype in X_train_fold.dtypes.items()
                    ],
                    "lightgbm_features": [
                        {"name": name, "dtype": str(dtype)}
                        for name, dtype in X_lgb_train.dtypes.items()
                    ],
                }
                write_json(run_dir / "feature_schema.json", feature_schema)

            lgb_settings = dict(model_config["lightgbm"])
            lgb_settings["num_threads"] = min(
                int(model_config["cpu_threads"]), os.cpu_count() or 1
            )
            cat_settings = dict(model_config["catboost"])
            cat_settings["num_threads"] = min(
                int(model_config["cpu_threads"]), os.cpu_count() or 1
            )
            cat_settings["categorical_columns"] = list(CATEGORICAL_COLUMNS)
            lgb_result = train_lightgbm_fold(
                X_lgb_train,
                labels,
                X_lgb_test,
                fit_indices,
                valid_indices,
                fold_number,
                lgb_settings,
                int(model_config["outer_cv"]["seed"]) + fold_number,
                run_dir / "models" / "lightgbm" / f"fold_{fold_number}.txt",
            )
            cat_result = train_catboost_fold(
                X_train_fold,
                labels,
                X_test_fold,
                fit_indices,
                valid_indices,
                fold_number,
                cat_settings,
                int(model_config["outer_cv"]["seed"]) + fold_number,
                run_dir / "models" / "catboost" / f"fold_{fold_number}.cbm",
            )
            for name, result in (("lightgbm", lgb_result), ("catboost", cat_result)):
                oof[name][valid_indices] = result["valid_prediction"]
                fold_test_predictions[name].append(result["test_prediction"])
            fold_records.append(
                {
                    "fold": fold_number,
                    "fit_rows": int(len(fit_indices)),
                    "valid_rows": int(len(valid_indices)),
                    "fit_validation_disjoint": True,
                    "nested_te_rows": int(len(train_te)),
                    "lightgbm": {
                        "auc": lgb_result["auc"],
                        "best_iteration": lgb_result["best_iteration"],
                        "device": lgb_result["device"],
                    },
                    "catboost": {
                        "auc": cat_result["auc"],
                        "best_iteration": cat_result["best_iteration"],
                        "device": cat_result["device"],
                        "gpu_telemetry": cat_result["gpu_telemetry"],
                    },
                    "elapsed_seconds": time.perf_counter() - fold_started,
                }
            )
            pd.concat(
                [lgb_result["importance"]], ignore_index=True
            ).to_csv(run_dir / f"lightgbm_importance_fold_{fold_number}.csv", index=False)
            cat_result["importance"].to_csv(
                run_dir / f"catboost_importance_fold_{fold_number}.csv", index=False
            )
            np.savez_compressed(
                run_dir / f"fold_{fold_number}_indices.npz",
                fit_indices=fit_indices.astype(np.int32),
                valid_indices=valid_indices.astype(np.int32),
            )
            print(
                f"fold={fold_number}/5 lgb_auc={lgb_result['auc']:.8f} "
                f"cat_auc={cat_result['auc']:.8f} "
                f"best_iter={lgb_result['best_iteration']}/"
                f"{cat_result['best_iteration']}",
                flush=True,
            )

        if any(not np.isfinite(values).all() for values in oof.values()):
            raise ValueError("At least one OOF prediction is missing or non-finite.")
        test_predictions = {
            name: mean_fold_predictions(values)
            for name, values in fold_test_predictions.items()
        }
        blend_oof = combine_phase2_predictions(oof["lightgbm"], oof["catboost"])
        blend_test = combine_phase2_predictions(
            test_predictions["lightgbm"], test_predictions["catboost"]
        )
        for name in ("lightgbm", "catboost"):
            np.save(run_dir / "predictions" / f"oof_{name}.npy", oof[name])
            np.save(run_dir / "predictions" / f"test_{name}.npy", test_predictions[name])
        np.save(run_dir / "predictions" / "oof_phase2_blend.npy", blend_oof)
        np.save(run_dir / "predictions" / "test_phase2_blend.npy", blend_test)
        importance_frames = {
            name: pd.concat(
                [
                    pd.read_csv(run_dir / f"{name}_importance_fold_{fold}.csv")
                    for fold in range(1, 6)
                ],
                ignore_index=True,
            )
            for name in ("lightgbm", "catboost")
        }
        for name, frame in importance_frames.items():
            frame.to_csv(run_dir / f"{name}_feature_importance.csv", index=False)
        candidate_path = run_dir / "submissions" / "submission_phase2_complete.csv"
        submission_check = build_submission(test_raw, sample, blend_test, candidate_path)
        candidate_sha = sha256_file(candidate_path)

        reference_frame = pd.read_csv(REFERENCE_CANDIDATE)
        candidate_frame = pd.read_csv(candidate_path)
        reference_submission_check = validate_submission(
            reference_frame, test_raw, sample
        )
        current_auc = float(roc_auc_score(labels, blend_oof))
        current_fold_auc = [
            float(roc_auc_score(labels[valid], blend_oof[valid]))
            for _, valid in folds
        ]
        reference_fold_auc = [
            float(roc_auc_score(labels[valid], reference_oof["blend"][valid]))
            for _, valid in folds
        ]
        fold_auc_deltas = [
            current - reference
            for current, reference in zip(current_fold_auc, reference_fold_auc)
        ]
        oof_compare = prediction_comparison(blend_oof, reference_oof["blend"])
        test_compare = prediction_comparison(
            blend_test, reference_test.astype(np.float64)
        )
        candidate_values = candidate_frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        frozen_values = reference_frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        candidate_csv_compare = prediction_comparison(candidate_values, frozen_values)
        validation_limits = {
            "maximum_absolute_global_oof_auc_delta": 0.001,
            "maximum_absolute_per_fold_auc_delta": 0.003,
            "minimum_oof_blend_pearson": 0.99,
            "minimum_test_blend_pearson": 0.99,
            "minimum_submission_pearson": 0.99,
        }
        reproduction_checks = {
            "fold_assignments_exact": fold_identity,
            "global_oof_auc_within_tolerance": abs(
                current_auc - reference_meta["candidate_oof_auc"]
            )
            <= validation_limits["maximum_absolute_global_oof_auc_delta"],
            "per_fold_auc_within_tolerance": max(
                abs(value) for value in fold_auc_deltas
            )
            <= validation_limits["maximum_absolute_per_fold_auc_delta"],
            "oof_prediction_correlation_within_tolerance": oof_compare["pearson"]
            >= validation_limits["minimum_oof_blend_pearson"],
            "test_prediction_correlation_within_tolerance": test_compare["pearson"]
            >= validation_limits["minimum_test_blend_pearson"],
            "submission_correlation_within_tolerance": candidate_csv_compare["pearson"]
            >= validation_limits["minimum_submission_pearson"],
        }
        reproduction_pass = all(reproduction_checks.values())
        if not reproduction_pass:
            raise RuntimeError(
                "Reproduction validation failed; submission was not promoted. "
                + json.dumps(reproduction_checks, sort_keys=True)
            )
        promotion_path = None
        if mode == "full":
            promotion_path = PACKAGE_DIR / "submissions" / run_id / candidate_path.name
            if promotion_path.exists():
                raise FileExistsError(f"Refusing to overwrite submission: {promotion_path}")
            promotion_path.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(candidate_path, promotion_path)
            if sha256_file(promotion_path) != candidate_sha:
                raise IOError("Promoted submission copy failed SHA-256 verification.")

        gpu_after = gpu_snapshot()
        report = {
            "run_id": run_id,
            "status": "PASS",
            "mode": mode,
            "classification": "INDEPENDENT_RAW_DATA_PHASE2_PIPELINE",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - timer,
            "input_validation": input_metadata,
            "feature_diagnostics": feature_diagnostics,
            "feature_schema": feature_schema,
            "target_encoding": {
                "key_specs": [
                    {"name": name, "source_feature": column, "smoothing": smoothing}
                    for name, column, smoothing in KEY_SPECS
                ],
                "smoothing_formula": "(sum(target) + smoothing * fit_prior) / (count + smoothing)",
                "inner_folds": INNER_SPLITS,
                "inner_seed": INNER_SEED,
                "training_rows_inner_oof": True,
                "outer_validation_encoded_from_outer_fit_only": True,
                "test_encoded_from_outer_fit_only": True,
                "validation_labels_excluded_from_all_fold_encodings": True,
                "self_target_excluded_by_disjoint_fit_apply_indices": True,
            },
            "folds": {
                "outer_count": len(folds),
                "outer_seed": int(model_config["outer_cv"]["seed"]),
                "fold_assignments_match_frozen": fold_identity,
                "fold_records": fold_records,
            },
            "model_configuration": model_config,
            "models": {
                "lightgbm_device": "CPU",
                "catboost_device": "GPU:0",
                "lightgbm_fold_auc": [row["lightgbm"]["auc"] for row in fold_records],
                "catboost_fold_auc": [row["catboost"]["auc"] for row in fold_records],
                "lightgbm_best_iterations": [
                    row["lightgbm"]["best_iteration"] for row in fold_records
                ],
                "catboost_best_iterations": [
                    row["catboost"]["best_iteration"] for row in fold_records
                ],
            },
            "ensemble": {
                "formula": "0.5 * LightGBM raw probability + 0.5 * CatBoost raw probability",
                "post_processing": "none",
                "oof_auc": current_auc,
                "per_fold_auc": current_fold_auc,
                "reference_oof_auc": reference_meta["candidate_oof_auc"],
                "oof_auc_delta": current_auc - reference_meta["candidate_oof_auc"],
            },
            "reproduction_validation": {
                "status": "PASS",
                "reference_candidate_sha256": EXPECTED_CANDIDATE_SHA256,
                "limits": validation_limits,
                "checks": reproduction_checks,
                "fold_auc_delta": fold_auc_deltas,
                "oof_prediction_comparison": oof_compare,
                "test_prediction_comparison": test_compare,
                "submission_comparison": candidate_csv_compare,
                "frozen_submission_checks": reference_submission_check,
            },
            "submission": {
                "run_path": str(candidate_path.resolve()),
                "sha256": candidate_sha,
                "verification": submission_check,
                "promoted_path": str(promotion_path.resolve()) if promotion_path else None,
                "promoted_sha256": (
                    sha256_file(promotion_path) if promotion_path else None
                ),
            },
            "gpu": {
                "before": gpu_before,
                "preflight": gpu_preflight,
                "after": gpu_after,
                "catboost_fold_telemetry": [
                    row["catboost"]["gpu_telemetry"] for row in fold_records
                ],
            },
            "reference": reference_meta,
            "software_versions": package_versions(),
            "authoritative_source_hashes": run_state[
                "authoritative_source_hashes"
            ],
            "git": git_evidence(),
            "official_competition_score": "NOT_EVALUATED",
        }
        report_path = run_dir / "reports" / "implementation_report.json"
        write_json(report_path, report)
        write_markdown_report(run_dir / "reports" / "implementation_report.md", report)
        write_json(PACKAGE_DIR / "IMPLEMENTATION_REPORT.json", report)
        write_markdown_report(PACKAGE_DIR / "IMPLEMENTATION_REPORT.md", report)
        run_state.update(
            {
                "status": "PASS",
                "completed_at_utc": report["completed_at_utc"],
                "elapsed_seconds": report["elapsed_seconds"],
                "inputs": input_metadata["input_hashes"],
                "frozen_reference": reference_meta,
                "effective_cpu_threads": min(
                    int(model_config["cpu_threads"]), os.cpu_count() or 1
                ),
                "outputs": None,
            }
        )
        write_json(run_dir / "run_manifest.json", run_state)
        outputs = {
            path.relative_to(run_dir).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
            and path.name not in {"sha256_manifest.json", "run_manifest.json"}
        }
        run_state["outputs"] = outputs
        write_json(run_dir / "run_manifest.json", run_state)
        manifest_files = {
            path.relative_to(run_dir).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "sha256_manifest.json"
        }
        write_json(
            run_dir / "sha256_manifest.json",
            {
                "run_id": run_id,
                "algorithm": "SHA-256",
                "files": manifest_files,
                "manifest_excludes_itself": True,
            },
        )
        global_manifest = PACKAGE_DIR / "runs" / "manifests" / f"{run_id}.json"
        write_json(global_manifest, run_state)
        global_report = PACKAGE_DIR / "runs" / "reports" / f"{run_id}.md"
        global_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "reports" / "implementation_report.md", global_report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return report
    except Exception as error:
        failure = {
            "run_id": run_id,
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "run_state": run_state,
        }
        write_json(run_dir / "failure_report.json", failure)
        raise


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    ensemble = report["ensemble"]
    submission = report["submission"]
    reproduction = report["reproduction_validation"]
    lines = [
        "# EXP85 Phase 2 Complete Pipeline — Execution Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Mode: `{report['mode']}`",
        f"- Run ID: `{report['run_id']}`",
        f"- Runtime seconds: `{report['elapsed_seconds']:.3f}`",
        f"- OOF AUC: `{ensemble['oof_auc']:.10f}`",
        f"- Frozen OOF AUC: `{ensemble['reference_oof_auc']:.10f}`",
        f"- OOF AUC delta: `{ensemble['oof_auc_delta']:+.10f}`",
        f"- Submission SHA-256: `{submission['sha256']}`",
        f"- Reproduction gate: **{reproduction['status']}**",
        f"- Official competition score: **NOT EVALUATED**",
        "",
        "## Per-fold validation",
        "",
        "| Fold | LightGBM AUC | CatBoost AUC | Blend AUC | Frozen blend AUC | Delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for index, (row, current, reference) in enumerate(
        zip(
            report["folds"]["fold_records"],
            ensemble["per_fold_auc"],
            report["reference"]["candidate_oof_by_fold"],
        ),
        1,
    ):
        lines.append(
            f"| {index} | {row['lightgbm']['auc']:.8f} | "
            f"{row['catboost']['auc']:.8f} | {current:.8f} | "
            f"{reference:.8f} | {current - reference:+.8f} |"
        )
    lines.extend(
        [
            "",
            "## Device evidence",
            "",
            f"- LightGBM: `{report['models']['lightgbm_device']}`",
            f"- CatBoost: `{report['models']['catboost_device']}`",
            f"- GPU preflight: `{report['gpu']['preflight']['catboost_gpu_smoke_test']}`",
            f"- GPU telemetry samples: "
            f"`{sum(item['sample_count'] for item in report['gpu']['catboost_fold_telemetry'])}`",
            "",
            "## Reproduction comparison",
            "",
            f"- OOF blend correlation: `{reproduction['oof_prediction_comparison']['pearson']:.8f}`",
            f"- Test blend correlation: `{reproduction['test_prediction_comparison']['pearson']:.8f}`",
            f"- Submission correlation: `{reproduction['submission_comparison']['pearson']:.8f}`",
            f"- Promoted submission: `{submission['promoted_path']}`",
            "",
            "## Provenance",
            "",
            "- Raw training/test/sample inputs were hash-verified and no historical EXP85 predictions were used.",
            "- The frozen Phase 2 candidate and its saved OOF/test outputs were read-only comparison references.",
            "- Authoritative frozen baseline/Phase 1/Phase 2 source SHA-256 values are recorded in the JSON report and run manifest.",
            "- Full raw-data retraining is independent of historical-cache reconstruction.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
