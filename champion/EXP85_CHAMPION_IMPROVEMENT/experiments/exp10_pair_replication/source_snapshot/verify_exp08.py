#!/usr/bin/env python3
"""Independently verify EXP08 hashes, fold coverage, metrics, and submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
FAMILY_NAMES = (
    "a_financial",
    "b_risk",
    "c_temporal",
    "d_demographic",
    "e_inquiry",
    "f_external",
    "h_interactions",
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


def verify_records(records: dict[str, Any], label: str) -> None:
    for name, record in records.items():
        path = Path(record["path"])
        require(path.is_file(), f"{label} file is missing: {name} ({path})")
        require(
            sha256_file(path) == record["sha256"],
            f"{label} SHA-256 mismatch: {name}",
        )
        require(
            path.stat().st_size == record["size_bytes"],
            f"{label} size mismatch: {name}",
        )


def verify_run(run_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "sha256_manifest.json"
    summary_path = run_dir / "experiment_summary.json"
    require(manifest_path.is_file(), f"Manifest is missing: {manifest_path}")
    require(summary_path.is_file(), f"Summary is missing: {summary_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["status"] == "COMPLETED", "Run is not marked completed")
    verify_records(manifest["inputs"], "Input")
    verify_records(manifest["references"], "Reference")
    verify_records(manifest["sources"], "Source")
    verify_records(manifest["outputs"], "Output")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    for key, path in (
        ("train", data_dir / "train.csv"),
        ("test", data_dir / "test.csv"),
        ("sample", data_dir / "sample_submission.csv"),
    ):
        require(
            sha256_file(path) == manifest["inputs"][key]["sha256"],
            f"Verifier input differs from run input: {key}",
        )
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    folds = list(
        StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(summary["outer_seed"]),
        ).split(np.zeros(len(labels)), labels)
    )
    coverage = np.zeros(len(labels), dtype=np.int8)
    for fold_number, (fit_expected, valid_expected) in enumerate(folds, 1):
        fold_path = run_dir / f"fold_{fold_number}_indices.npz"
        with np.load(fold_path, allow_pickle=False) as archive:
            fit_actual = archive["fit_indices"]
            valid_actual = archive["valid_indices"]
        require(
            np.array_equal(fit_actual, fit_expected)
            and np.array_equal(valid_actual, valid_expected),
            f"Fold {fold_number} differs from the declared stratified split",
        )
        require(
            not np.intersect1d(fit_actual, valid_actual).size,
            f"Fold {fold_number} fit and validation rows overlap",
        )
        coverage[valid_actual] += 1
    require(np.all(coverage == 1), "Outer folds do not cover each row exactly once")

    predictions: dict[str, np.ndarray] = {}
    for name in (
        "oof_phase2_matched.npy",
        "oof_historical_candidate.npy",
        "oof_nested_blend.npy",
    ):
        array = np.load(run_dir / name, allow_pickle=False)
        require(
            array.shape == labels.shape and np.isfinite(array).all(),
            f"Invalid OOF prediction vector: {name}",
        )
        predictions[name] = array
    phase2_reference_record = manifest["references"].get(
        "oof_phase2_strict_matched.npy"
    )
    if phase2_reference_record is not None:
        reference_oof = np.load(
            Path(phase2_reference_record["path"]),
            allow_pickle=False,
        )
        require(
            np.array_equal(
                predictions["oof_phase2_matched.npy"],
                reference_oof,
            ),
            "Saved Phase 2 OOF does not match the hashed strict reference",
        )
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    require(len(fold_metrics) == 5, "Fold metrics do not have five rows")
    for key, filename in (
        ("strict_phase2_matched", "oof_phase2_matched.npy"),
        ("historical_candidate", "oof_historical_candidate.npy"),
        ("nested_blend", "oof_nested_blend.npy"),
    ):
        expected_auc = float(summary["metrics"][key]["pooled_oof_auc"])
        actual_auc = float(roc_auc_score(labels, predictions[filename]))
        require(
            abs(actual_auc - expected_auc) < 1e-12,
            f"{key} pooled AUC does not reproduce from saved OOF predictions",
        )
    for column, filename in (
        ("phase2_auc", "oof_phase2_matched.npy"),
        ("historical_candidate_auc", "oof_historical_candidate.npy"),
        ("nested_blend_auc", "oof_nested_blend.npy"),
    ):
        actual = [
            float(roc_auc_score(labels[valid], predictions[filename][valid]))
            for _, valid in folds
        ]
        require(
            np.allclose(fold_metrics[column].to_numpy(), actual, atol=1e-12),
            f"Saved fold metric does not reproduce: {column}",
        )
    for family in FAMILY_NAMES:
        family_path = run_dir / f"oof_{family}.npy"
        family_oof = np.load(family_path, allow_pickle=False)
        require(
            family_oof.shape == labels.shape and np.isfinite(family_oof).all(),
            f"Invalid family OOF predictions: {family}",
        )
        require(
            abs(
                roc_auc_score(labels, family_oof)
                - summary["metrics"]["historical_families"][
                    family.upper()
                ]["pooled_oof_auc"]
            )
            < 1e-12,
            f"Family AUC does not reproduce: {family}",
        )

    submission_path = Path(summary["submission"]["path"])
    submission = pd.read_csv(submission_path)
    require(
        submission.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and len(submission) == len(test) == len(sample),
        "Submission columns or row count are incorrect",
    )
    require(
        submission[ID_COLUMN].is_unique
        and np.array_equal(
            submission[ID_COLUMN].to_numpy(),
            test[ID_COLUMN].to_numpy(),
        )
        and np.array_equal(
            submission[ID_COLUMN].to_numpy(),
            sample[ID_COLUMN].to_numpy(),
        ),
        "Submission IDs are not in exact test/sample order",
    )
    target_values = submission[TARGET_COLUMN].to_numpy(dtype=np.float64)
    require(
        np.isfinite(target_values).all()
        and np.all((target_values >= 0.0) & (target_values <= 1.0)),
        "Submission probabilities are invalid",
    )
    saved_test_blend = np.load(
        run_dir / "test_nested_blend.npy",
        allow_pickle=False,
    )
    require(
        np.array_equal(target_values.astype(np.float32), saved_test_blend),
        "Submission targets differ from the saved test blend",
    )
    require(
        sha256_file(submission_path) == summary["submission"]["sha256"],
        "Submission SHA-256 does not match the summary",
    )
    inventory = pd.read_csv(run_dir.parent.parent / "FEATURE_INVENTORY.csv")
    require(len(inventory) == 669, "Feature inventory does not contain 669 rows")
    require(
        int(inventory["evaluation_status"].str.contains("excluded").sum()) == 10,
        "Inventory should include nine excluded formulas plus the infinity-excluded feature",
    )
    computed_delta = (
        summary["metrics"]["nested_blend"]["pooled_oof_auc"]
        - summary["metrics"]["strict_phase2_matched"]["pooled_oof_auc"]
    )
    computed_fold_deltas = (
        fold_metrics["nested_blend_delta_vs_phase2"].to_numpy(dtype=np.float64)
    )
    computed_positive_folds = int(np.sum(computed_fold_deltas > 0.0))
    mean_delta = float(np.mean(computed_fold_deltas))
    delta_se = float(
        np.std(computed_fold_deltas, ddof=1) / math.sqrt(len(computed_fold_deltas))
    )
    gate = summary["advancement_gate"]
    expected_pass = (
        computed_delta >= gate["minimum_pooled_auc_delta"]
        and computed_positive_folds >= gate["minimum_positive_folds"]
        and mean_delta
        >= gate["minimum_mean_delta_standard_errors"] * delta_se
    )
    require(
        abs(computed_delta - gate["pooled_auc_delta"]) < 1e-12
        and computed_positive_folds == gate["positive_folds"]
        and abs(mean_delta - gate["mean_delta"]) < 1e-12
        and abs(delta_se - gate["fold_delta_standard_error"]) < 1e-12
        and expected_pass == gate["passed"],
        "Advancement gate summary does not match the saved predictions",
    )
    return {
        "status": "VERIFIED",
        "run_id": manifest["experiment_id"],
        "outer_folds": len(folds),
        "feature_inventory_rows": len(inventory),
        "submission_rows": len(submission),
        "submission_sha256": sha256_file(submission_path),
        "pooled_auc": {
            key: float(value["pooled_oof_auc"])
            for key, value in summary["metrics"].items()
            if isinstance(value, dict) and "pooled_oof_auc" in value
        },
        "outputs_verified": len(manifest["outputs"]),
        "sources_verified": len(manifest["sources"]),
        "references_verified": len(manifest["references"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    args = parser.parse_args()
    result = verify_run(args.run_dir.resolve(), args.data_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
