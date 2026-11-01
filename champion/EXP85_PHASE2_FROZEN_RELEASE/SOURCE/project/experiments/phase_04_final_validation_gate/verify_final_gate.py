#!/usr/bin/env python3
"""Independently verify final-gate saved predictions, folds and integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


VARIANTS = ("raw_baseline", "phase2_pooled", "phase2_fold_local")
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_prediction(path: Path, expected_rows: int) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    require(values.ndim == 1, f"Prediction array is not a vector: {path}")
    require(len(values) == expected_rows, f"Prediction row-count mismatch: {path}")
    require(np.isfinite(values).all(), f"Non-finite predictions: {path}")
    require(
        ((values >= 0.0) & (values <= 1.0)).all(),
        f"Out-of-range predictions: {path}",
    )
    return values.astype(np.float64, copy=False)


def close(actual: float, expected: float, name: str) -> None:
    require(
        abs(actual - expected) <= 1e-12,
        f"Recomputed {name} differs from summary: {actual} != {expected}",
    )


def verify(
    run_dir: Path,
    data_dir: Path,
    original_candidate: Path,
    output_path: Path,
    frequency_ablation_dir: Path,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    data_dir = data_dir.resolve()
    frequency_ablation_dir = frequency_ablation_dir.resolve()
    output_path = output_path.resolve()
    summary = json.loads(
        (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )
    manifest_path = run_dir / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(summary["status"] == "COMPLETED", "Final gate did not complete")
    require(summary["outer_seed"] != 42, "Final gate reused the original seed")

    output_hashes_verified = 0
    for relative, expected in manifest["outputs"].items():
        path = run_dir / relative
        require(path.is_file(), f"Run-manifest output missing: {relative}")
        require(
            sha256_file(path) == expected["sha256"],
            f"Run-manifest SHA mismatch: {relative}",
        )
        require(
            path.stat().st_size == expected["size_bytes"],
            f"Run-manifest size mismatch: {relative}",
        )
        output_hashes_verified += 1

    input_hashes_verified = 0
    data_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for name, path in data_paths.items():
        expected = manifest["inputs"][name]
        require(sha256_file(path) == expected["sha256"], f"Raw-input hash mismatch: {name}")
        require(
            path.stat().st_size == expected["size_bytes"],
            f"Raw-input size mismatch: {name}",
        )
        input_hashes_verified += 1
    source_hashes_verified = 0
    for path_text, expected in manifest["sources"].items():
        path = Path(path_text)
        require(path.is_file(), f"Source file missing: {path}")
        require(
            sha256_file(path) == expected["sha256"],
            f"Source hash mismatch: {path}",
        )
        source_hashes_verified += 1

    train = pd.read_csv(data_paths["train"], usecols=[ID_COLUMN, TARGET_COLUMN])
    test = pd.read_csv(data_paths["test"], usecols=[ID_COLUMN])
    train_columns = pd.read_csv(data_paths["train"], nrows=0).columns.tolist()
    test_columns = pd.read_csv(data_paths["test"], nrows=0).columns.tolist()
    sample = pd.read_csv(data_paths["sample"])
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    test_ids = test[ID_COLUMN].to_numpy()
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Sample submission schema mismatch",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_ids),
        "Sample/test ID order mismatch",
    )
    require(
        set(train[ID_COLUMN]).isdisjoint(set(test_ids)),
        "Train and test IDs overlap",
    )
    require(
        [column for column in train_columns if column != TARGET_COLUMN]
        == test_columns,
        "Train/test predictor columns differ beyond TARGET",
    )
    require(
        train_columns.count(TARGET_COLUMN) == 1
        and TARGET_COLUMN not in test_columns,
        "Target appears outside the labeled training input",
    )
    forbidden_predictor_tokens = (
        "PAYMENT",
        "REPAY",
        "DEFAULT",
        "DELINQU",
        "OVERDUE",
        "OUTCOME",
        "LABEL",
    )
    forbidden_columns = [
        column
        for column in test_columns
        if any(token in column.upper() for token in forbidden_predictor_tokens)
    ]
    require(
        not forbidden_columns,
        f"Potential post-outcome predictor fields found: {forbidden_columns}",
    )
    require(
        summary["historical_champion_read"] is False
        and summary["historical_prediction_inputs"] == [],
        "Final gate accessed historical predictions",
    )
    config = summary["model_configuration"]
    require(
        config["lightgbm"]["device"] == "CPU",
        "LightGBM configuration differs from the frozen baseline device",
    )
    require(
        config["catboost"]["task_type"] == "GPU"
        and config["catboost"]["devices"] == "0",
        "CatBoost was not explicitly configured for GPU:0",
    )

    seed = int(summary["outer_seed"])
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels
        )
    )
    saved_folds = np.load(run_dir / "fold_1_indices.npz", allow_pickle=False)
    require(
        np.array_equal(saved_folds["fit_indices"], folds[0][0])
        and np.array_equal(saved_folds["valid_indices"], folds[0][1]),
        "Fold 1 does not match reconstructed StratifiedKFold",
    )
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        stored = np.load(
            run_dir / f"fold_{fold_number}_indices.npz",
            allow_pickle=False,
        )
        require(
            np.array_equal(stored["fit_indices"], fit_indices)
            and np.array_equal(stored["valid_indices"], valid_indices),
            f"Fold {fold_number} differs from reconstructed split",
        )

    metrics: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    for variant in VARIANTS:
        lightgbm_oof = load_prediction(
            run_dir / f"oof_{variant}_lightgbm.npy", len(labels)
        )
        catboost_oof = load_prediction(
            run_dir / f"oof_{variant}_catboost.npy", len(labels)
        )
        blend_oof = load_prediction(
            run_dir / f"oof_{variant}_blend.npy", len(labels)
        )
        expected_oof_blend = (lightgbm_oof + catboost_oof) / 2.0
        require(
            np.allclose(blend_oof, expected_oof_blend, rtol=0.0, atol=1e-7),
            f"{variant} OOF blend does not match the two saved models",
        )
        overall_auc = float(roc_auc_score(labels, blend_oof))
        fold_aucs = [
            float(roc_auc_score(labels[valid], blend_oof[valid]))
            for _, valid in folds
        ]
        close(
            overall_auc,
            summary["metrics"][variant]["oof_blend_auc"],
            f"{variant} OOF AUC",
        )
        for fold_index, (actual, expected) in enumerate(
            zip(fold_aucs, summary["metrics"][variant]["fold_blend_auc"]), 1
        ):
            close(actual, expected, f"{variant} fold {fold_index} AUC")

        lightgbm_test = load_prediction(
            run_dir / f"test_{variant}_lightgbm.npy", len(test_ids)
        )
        catboost_test = load_prediction(
            run_dir / f"test_{variant}_catboost.npy", len(test_ids)
        )
        blend_test = load_prediction(
            run_dir / f"test_{variant}_blend.npy", len(test_ids)
        )
        require(
            np.allclose(
                blend_test,
                (lightgbm_test + catboost_test) / 2.0,
                rtol=0.0,
                atol=1e-7,
            ),
            f"{variant} test blend does not match the two saved models",
        )
        candidate_path = run_dir / f"submission_{variant}.csv"
        candidate = pd.read_csv(candidate_path)
        require(
            candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
            f"{variant} candidate schema mismatch",
        )
        candidate_ids = candidate[ID_COLUMN].to_numpy()
        candidate_values = candidate[TARGET_COLUMN].to_numpy(dtype=np.float64)
        require(
            len(candidate) == len(test_ids)
            and candidate[ID_COLUMN].is_unique
            and np.array_equal(candidate_ids, test_ids),
            f"{variant} candidate IDs/count/order are invalid",
        )
        require(np.isfinite(candidate_values).all(), f"{variant} has NaN/Inf")
        require(
            ((candidate_values >= 0.0) & (candidate_values <= 1.0)).all(),
            f"{variant} candidate probabilities out of range",
        )
        csv_error = float(np.max(np.abs(candidate_values - blend_test)))
        require(csv_error < 1e-9, f"{variant} candidate blend mismatch")
        metrics[variant] = {
            "oof_blend_auc_recomputed": overall_auc,
            "fold_blend_auc_recomputed": fold_aucs,
            "csv_max_abs_difference_from_test_blend": csv_error,
        }
        candidates[variant] = {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "rows": len(candidate),
            "unique_ids": int(candidate[ID_COLUMN].nunique()),
            "schema": candidate.columns.tolist(),
            "prediction_min": float(candidate_values.min()),
            "prediction_max": float(candidate_values.max()),
        }

    raw_ids = pd.read_csv(data_paths["train"], usecols=[ID_COLUMN])[ID_COLUMN].to_numpy()
    leakage_audit = []
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        record = summary["folds"][fold_number - 1]
        local_audit = record["variants"]["phase2_fold_local"]["audit"]
        frequency = local_audit["frequency"]
        expected_reference_ids_hash = hashlib.sha256(
            raw_ids[fit_indices].tobytes()
        ).hexdigest()
        for partition in ("fit", "validation", "test"):
            partition_audit = frequency[partition]
            require(
                partition_audit["count_reference_id_sha256"]
                == expected_reference_ids_hash,
                f"Fold {fold_number} {partition} frequency reference IDs differ",
            )
            require(
                partition_audit["count_reference_rows"] == len(fit_indices),
                f"Fold {fold_number} {partition} frequency row count differs",
            )
            require(
                partition_audit["reference_contains_validation_or_test_rows"]
                is False
                and partition_audit["target_used"] is False,
                f"Fold {fold_number} frequency leakage flag is invalid",
            )
        require(
            frequency["test_ids_used_in_frequency_reference"] is False,
            f"Fold {fold_number} used test covariates in local frequencies",
        )
        te = local_audit["target_encoding"]
        require(
            te["outer_fit_valid_overlap"] == 0
            and te["fit_rows_encoded_by_inner_oof"] == len(fit_indices)
            and te["validation_encoded_from_outer_fit_only"] is True
            and te["test_encoded_from_outer_fit_only"] is True
            and te["apply_row_targets_used_by_mapping"] is False,
            f"Fold {fold_number} target-encoding audit failed",
        )
        leakage_audit.append(
            {
                "fold": fold_number,
                "outer_fit_rows": len(fit_indices),
                "validation_rows": len(valid_indices),
                "local_frequency_reference_id_hash": expected_reference_ids_hash,
                "target_encodings_inner_oof_and_outer_fit_only": True,
            }
        )

    stability = {}
    baseline_auc = metrics["raw_baseline"]["oof_blend_auc_recomputed"]
    baseline_folds = metrics["raw_baseline"]["fold_blend_auc_recomputed"]
    for variant in ("phase2_pooled", "phase2_fold_local"):
        variant_auc = metrics[variant]["oof_blend_auc_recomputed"]
        fold_deltas = [
            candidate - baseline
            for candidate, baseline in zip(
                metrics[variant]["fold_blend_auc_recomputed"],
                baseline_folds,
            )
        ]
        stability[variant] = {
            "oof_auc_delta_vs_new_seed_baseline": variant_auc - baseline_auc,
            "paired_fold_deltas": fold_deltas,
            "positive_folds": sum(value > 0 for value in fold_deltas),
        }

    original_candidate = original_candidate.resolve()
    require(
        sha256_file(original_candidate)
        == "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa",
        "Frozen original Phase 2 candidate SHA-256 changed",
    )
    require(
        sha256_file(original_candidate)
        == json.loads(
            (
                run_dir.parents[1] / "runs" / "phase02_key_te_20261101_181500"
                / "experiment_summary.json"
            ).read_text(encoding="utf-8")
        )["candidate_sha256"],
        "Frozen original candidate differs from its original run record",
    )
    gpu_records = []
    for variant in VARIANTS:
        for model in summary["models"][variant]["catboost"]:
            monitor = model["gpu_monitor"]
            require(
                monitor["samples"] > 0
                and monitor["max_gpu_utilization_percent"] is not None
                and monitor["max_gpu_utilization_percent"] > 0
                and not monitor["errors"],
                f"GPU evidence missing for {variant} fold {model['fold']}",
            )
            gpu_records.append(
                {
                    "variant": variant,
                    "fold": model["fold"],
                    "samples": monitor["samples"],
                    "max_utilization_percent": monitor[
                        "max_gpu_utilization_percent"
                    ],
                    "max_memory_used_mib": monitor["max_gpu_memory_used_mib"],
                }
            )

    ablation_summary = json.loads(
        (
            frequency_ablation_dir / "experiment_summary.json"
        ).read_text(encoding="utf-8")
    )
    ablation_manifest_path = frequency_ablation_dir / "sha256_manifest.json"
    ablation_manifest = json.loads(
        ablation_manifest_path.read_text(encoding="utf-8")
    )
    ablation_hash_counts = {"outputs": 0, "inputs": 0, "analysis_inputs": 0, "sources": 0}
    for relative, expected in ablation_manifest["outputs"].items():
        path = frequency_ablation_dir / relative
        require(path.is_file(), f"Frequency-ablation output missing: {relative}")
        require(
            sha256_file(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"],
            f"Frequency-ablation output hash mismatch: {relative}",
        )
        ablation_hash_counts["outputs"] += 1
    for name, path in data_paths.items():
        expected = ablation_manifest["inputs"][name]
        require(
            sha256_file(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"],
            f"Frequency-ablation raw-input hash mismatch: {name}",
        )
        ablation_hash_counts["inputs"] += 1
    for path_text, expected in ablation_manifest["analysis_only_inputs"].items():
        path = Path(path_text)
        require(
            path.is_file()
            and sha256_file(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"],
            f"Frequency-ablation comparison input hash mismatch: {path}",
        )
        ablation_hash_counts["analysis_inputs"] += 1
    for path_text, expected in ablation_manifest["sources"].items():
        path = Path(path_text)
        require(
            path.is_file()
            and sha256_file(path) == expected["sha256"]
            and path.stat().st_size == expected["size_bytes"],
            f"Frequency-ablation source hash mismatch: {path}",
        )
        ablation_hash_counts["sources"] += 1
    ablation_fold_aucs = []
    ablation_blend = load_prediction(
        frequency_ablation_dir / "oof_frequency_disabled_blend.npy",
        len(labels),
    )
    ablation_lgb = load_prediction(
        frequency_ablation_dir / "oof_frequency_disabled_lightgbm.npy",
        len(labels),
    )
    ablation_cat = load_prediction(
        frequency_ablation_dir / "oof_frequency_disabled_catboost.npy",
        len(labels),
    )
    require(
        np.allclose(
            ablation_blend,
            (ablation_lgb + ablation_cat) / 2.0,
            rtol=0.0,
            atol=1e-7,
        ),
        "Frequency-disabled OOF blend differs from its models",
    )
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        stored = np.load(
            frequency_ablation_dir / f"fold_{fold_number}_indices.npz",
            allow_pickle=False,
        )
        require(
            np.array_equal(stored["fit_indices"], fit_indices)
            and np.array_equal(stored["valid_indices"], valid_indices),
            f"Frequency-ablation fold {fold_number} differs",
        )
        ablation_fold_aucs.append(
            float(roc_auc_score(labels[valid_indices], ablation_blend[valid_indices]))
        )
        require(
            ablation_summary["folds"][fold_number - 1][
                "validation_labels_used_for_training_encodings"
            ]
            is False
            and ablation_summary["folds"][fold_number - 1][
                "nested_training_encoding_rows"
            ]
            == len(fit_indices),
            f"Frequency-ablation fold {fold_number} target-encoding audit failed",
        )
    ablated_overall_auc = float(roc_auc_score(labels, ablation_blend))
    close(
        ablated_overall_auc,
        ablation_summary["metrics"]["frequency_disabled_oof_blend_auc"],
        "frequency-disabled OOF AUC",
    )
    for fold_number, (actual, expected) in enumerate(
        zip(
            ablation_fold_aucs,
            ablation_summary["metrics"]["paired_fold_auc"]["frequency_disabled"],
        ),
        1,
    ):
        close(actual, expected, f"frequency-disabled fold {fold_number} AUC")
    local_values = metrics["phase2_fold_local"]["fold_blend_auc_recomputed"]
    frequency_deltas = [
        local - disabled
        for local, disabled in zip(local_values, ablation_fold_aucs)
    ]
    ablation_candidate_path = frequency_ablation_dir / "submission_frequency_disabled.csv"
    ablation_candidate = pd.read_csv(ablation_candidate_path)
    ablation_candidate_values = ablation_candidate[TARGET_COLUMN].to_numpy(
        dtype=np.float64
    )
    require(
        ablation_candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and len(ablation_candidate) == len(test_ids)
        and ablation_candidate[ID_COLUMN].is_unique
        and np.array_equal(ablation_candidate[ID_COLUMN].to_numpy(), test_ids)
        and np.isfinite(ablation_candidate_values).all()
        and ((ablation_candidate_values >= 0.0) & (ablation_candidate_values <= 1.0)).all(),
        "Frequency-disabled candidate integrity failed",
    )
    ablation_test_blend = load_prediction(
        frequency_ablation_dir / "test_frequency_disabled_blend.npy",
        len(test_ids),
    )
    ablation_csv_error = float(
        np.max(np.abs(ablation_candidate_values - ablation_test_blend))
    )
    require(ablation_csv_error < 1e-9, "Frequency-disabled candidate blend mismatch")
    for fold_model in ablation_summary["models"]["catboost"]:
        monitor = fold_model["gpu_monitor"]
        require(
            monitor["samples"] > 0
            and monitor["max_utilization_percent"] > 0
            and not monitor["errors"],
            f"Frequency-ablation GPU evidence missing in fold {fold_model['fold']}",
        )
    full_local_auc = metrics["phase2_fold_local"]["oof_blend_auc_recomputed"]
    frequency_enabled_local_auc = full_local_auc
    frequency_disabled_auc = ablated_overall_auc
    frequency_enabled_delta = frequency_enabled_local_auc - frequency_disabled_auc
    frequency_enabled_pass = (
        frequency_enabled_delta > 0
        and sum(delta > 0 for delta in frequency_deltas) >= 4
    )
    require(
        frequency_enabled_pass
        == (
            ablation_summary["metrics"][
                "fee_frequency_decoder_delta_vs_disabled"
            ]["full_oof_auc_delta"]
            > 0
            and ablation_summary["metrics"][
                "fee_frequency_decoder_delta_vs_disabled"
            ]["positive_folds"]
            >= 4
        ),
        "Frequency ablation decision differs from summary",
    )

    pooled_pass = (
        stability["phase2_pooled"]["oof_auc_delta_vs_new_seed_baseline"] > 0
        and stability["phase2_pooled"]["positive_folds"] >= 4
    )
    local_pass = (
        stability["phase2_fold_local"]["oof_auc_delta_vs_new_seed_baseline"] > 0
        and stability["phase2_fold_local"]["positive_folds"] >= 4
    )
    decision = (
        "PASS"
        if pooled_pass and local_pass and frequency_enabled_pass
        else "CONDITIONAL"
    )
    require(summary["decision"] == decision, "Summary decision does not recompute")
    require(
        frequency_enabled_pass,
        "Fold-local frequency decoder did not beat the disabled control",
    )
    evidence = {
        "status": "PASS",
        "final_decision": decision,
        "official_competition_score": "NOT_EVALUATED",
        "models_retrained_by_verifier": False,
        "run_id": run_dir.name,
        "outer_seed": seed,
        "raw_schema_audit": {
            "train_columns": len(train_columns),
            "test_columns": len(test_columns),
            "target_only_in_train": True,
            "predictor_columns_match": True,
            "potential_post_outcome_columns": forbidden_columns,
            "all_predictors_are_from_raw_application_train_test_csvs": True,
        },
        "hash_verification": {
            "run_output_records": output_hashes_verified,
            "raw_input_records": input_hashes_verified,
            "source_records": source_hashes_verified,
            "run_manifest_sha256": sha256_file(manifest_path),
        },
        "oof_auc_recomputed_from_saved_arrays": metrics,
        "paired_fold_stability": stability,
        "fee_frequency_feature_ablation": {
            "status": "PASS",
            "frequency_disabled_oof_auc_recomputed": frequency_disabled_auc,
            "frequency_enabled_fold_local_oof_auc": frequency_enabled_local_auc,
            "full_oof_auc_delta_enabled_vs_disabled": frequency_enabled_delta,
            "paired_fold_auc_deltas_enabled_vs_disabled": frequency_deltas,
            "positive_folds": sum(value > 0 for value in frequency_deltas),
            "positive_folds_required": 4,
            "ablation_hash_verification": ablation_hash_counts,
            "candidate_sha256": sha256_file(ablation_candidate_path),
            "candidate_csv_max_abs_difference": ablation_csv_error,
        },
        "candidate_integrity": candidates,
        "fold_local_frequency_and_target_encoding_audit": leakage_audit,
        "gpu_per_fold_observations": gpu_records,
        "frozen_original_phase2_candidate": {
            "path": str(original_candidate),
            "sha256": sha256_file(original_candidate),
            "preserved": True,
        },
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
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "runs"
        / "final_gate_seed20261101",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument(
        "--original-candidate",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "runs"
        / "phase02_key_te_20261101_181500"
        / "submission_key_target_encoding.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "FINAL_PHASE2_GATE_EVIDENCE.json",
    )
    parser.add_argument(
        "--frequency-ablation-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "runs"
        / "final_gate_seed20261101_frequency_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = verify(
        args.run_dir,
        args.data_dir,
        args.original_candidate,
        args.output,
        args.frequency_ablation_dir,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
