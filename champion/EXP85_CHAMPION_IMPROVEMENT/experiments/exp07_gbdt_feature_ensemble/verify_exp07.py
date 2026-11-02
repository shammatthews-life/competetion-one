#!/usr/bin/env python3
"""Independently verify hashes, fold coverage, metrics, and submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp07_signature_seed20261101_v1"
DATA_DIR = Path(r"D:\gci\inpuy")
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


def verify(run_dir: Path, data_dir: Path) -> dict[str, object]:
    run_dir, data_dir = run_dir.resolve(), data_dir.resolve()
    summary = json.loads(
        (run_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "sha256_manifest.json").read_text(encoding="utf-8")
    )
    require(summary["status"] == "COMPLETED_SCREEN", "Run is not complete")
    require(
        summary["fold_protocol"].get("outer_seed", summary["outer_seed"])
        == summary["outer_seed"],
        "Summary outer seed is inconsistent",
    )
    gpu = summary["model"]
    telemetry = gpu["gpu_telemetry"]
    require(
        "RTX 4050" in gpu["gpu_preflight"]["name"]
        and gpu["catboost_gpu_task_type_verified"] is True
        and telemetry["sample_count"] > 0
        and telemetry["max_utilization_percent"] > 0
        and telemetry["max_memory_used_mib"]
        <= gpu["gpu_preflight"]["total_memory_mib"],
        "Verified RTX 4050 CUDA training evidence is missing or invalid",
    )
    for group in ("inputs", "sources"):
        for name, record in manifest[group].items():
            path = Path(record["path"])
            require(path.is_file(), f"{group} file is missing: {name}")
            require(
                sha256_file(path) == record["sha256"],
                f"{group} hash mismatch: {name}",
            )
            if "size_bytes" in record:
                require(
                    path.stat().st_size == record["size_bytes"],
                    f"{group} size mismatch: {name}",
                )
    for relative, record in manifest["outputs"].items():
        path = run_dir / relative
        require(path.is_file(), f"Output is missing: {relative}")
        require(
            sha256_file(path) == record["sha256"],
            f"Output hash mismatch: {relative}",
        )
        require(
            path.stat().st_size == record["size_bytes"],
            f"Output size mismatch: {relative}",
        )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    outer_seed = int(summary["outer_seed"])
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed).split(
            np.zeros(len(labels)), labels
        )
    )
    oof = np.load(run_dir / "oof_signature_model.npy", allow_pickle=False)
    blended = np.load(run_dir / "oof_nested_blend.npy", allow_pickle=False)
    baseline = np.load(
        run_dir / "oof_phase2_strict_matched.npy", allow_pickle=False
    )
    require(
        oof.shape == blended.shape == baseline.shape == (len(train),),
        "OOF prediction shape mismatch",
    )
    for name, values in (
        ("signature", oof),
        ("nested blend", blended),
        ("Phase 2", baseline),
    ):
        require(np.isfinite(values).all(), f"{name} OOF is non-finite")
        require(
            np.all((values >= 0.0) & (values <= 1.0)),
            f"{name} OOF is outside [0,1]",
        )
    fold_records = json.loads(
        (run_dir / "fold_metrics.json").read_text(encoding="utf-8")
    )
    require(len(fold_records) == len(folds), "Fold metric count is not five")
    fold_coverage = np.zeros(len(train), dtype=np.int8)
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        fold_coverage[valid_indices] += 1
        require(
            not np.intersect1d(fit_indices, valid_indices).size,
            f"Fold {fold_number} fit/valid overlap",
        )
        with np.load(
            run_dir / f"fold_{fold_number}_indices.npz",
            allow_pickle=False,
        ) as index_archive:
            require(
                np.array_equal(index_archive["fit_indices"], fit_indices)
                and np.array_equal(index_archive["valid_indices"], valid_indices),
                f"Fold {fold_number} differs from fixed outer indices",
            )
            inner_fit = index_archive["inner_fit_indices"]
            inner_valid = index_archive["inner_valid_indices"]
            require(
                not np.intersect1d(inner_fit, inner_valid).size
                and np.isin(inner_fit, fit_indices).all()
                and np.isin(inner_valid, fit_indices).all(),
                f"Fold {fold_number} inner indices leak outer validation",
            )
        metrics = fold_records[fold_number - 1]
        require(
            metrics["fold"] == fold_number
            and metrics["catboost_task_type"] == "GPU",
            f"Fold {fold_number} GPU or metric record is invalid",
        )
        baseline_auc = roc_auc_score(labels[valid_indices], baseline[valid_indices])
        candidate_auc = roc_auc_score(labels[valid_indices], oof[valid_indices])
        blend_auc = roc_auc_score(
            labels[valid_indices], blended[valid_indices]
        )
        require(
            np.isclose(metrics["baseline_outer_auc"], baseline_auc, atol=1e-10)
            and np.isclose(metrics["candidate_outer_auc"], candidate_auc, atol=1e-10)
            and np.isclose(metrics["nested_blend_outer_auc"], blend_auc, atol=1e-10),
            f"Fold {fold_number} AUC does not match the saved predictions",
        )
    require(np.all(fold_coverage == 1), "Outer folds do not cover each row once")
    for name, filename in (
        ("submission_signature_model.csv", "test_signature_model.npy"),
        ("submission_nested_blend.csv", "test_nested_blend.npy"),
    ):
        submission = pd.read_csv(run_dir / name)
        predictions = np.load(run_dir / filename, allow_pickle=False)
        require(
            submission.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
            and len(submission) == len(test)
            and submission[ID_COLUMN].is_unique
            and np.array_equal(submission[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy())
            and np.array_equal(submission[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
            f"{name} schema/ordered IDs are invalid",
        )
        require(
            predictions.shape == (len(test),)
            and np.isfinite(predictions).all()
            and np.all((predictions >= 0.0) & (predictions <= 1.0))
            and np.array_equal(
                predictions.astype(np.float64),
                submission[TARGET_COLUMN].to_numpy(dtype=np.float64),
            ),
            f"{name} predictions are invalid or differ from saved array",
        )
    result = {
        "run": str(run_dir),
        "decision": summary["decision"],
        "candidate_auc": roc_auc_score(labels, oof),
        "nested_blend_auc": roc_auc_score(labels, blended),
        "phase2_auc": roc_auc_score(labels, baseline),
        "all_input_source_output_hashes_verified": True,
        "outer_and_inner_fold_indices_verified": True,
        "submission_schemas_and_id_orders_verified": True,
        "output_count": len(manifest["outputs"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
