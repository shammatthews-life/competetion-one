#!/usr/bin/env python3
"""Independently verify the EXP09 artifact manifest and core AUC result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "SHA256_MANIFEST.json"
    summary_path = run_dir / "experiment_summary.json"
    require(manifest_path.is_file() and summary_path.is_file(), "EXP09 summary or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for label, record in manifest["inputs"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Input missing: {label}")
        require(sha256_file(path) == record["sha256"], f"Input SHA-256 mismatch: {label}")
    for label, record in manifest["sources"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Source missing: {label}")
        require(sha256_file(path) == record["sha256"], f"Source SHA-256 mismatch: {label}")
    for label, record in manifest["source_experiment_artifact_manifests"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Source experiment manifest missing: {label}")
        require(sha256_file(path) == record["sha256"], f"Source experiment manifest changed: {label}")
    for label, record in manifest["outputs"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Output missing: {label}")
        require(sha256_file(path) == record["sha256"], f"Output SHA-256 mismatch: {label}")
        require(path.stat().st_size == record["size_bytes"], f"Output size mismatch: {label}")

    train = pd.read_csv(manifest["inputs"]["train"]["path"], usecols=["TARGET"])
    labels = train["TARGET"].to_numpy()
    prediction_path = run_dir / "fixed_blend_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as predictions:
        baseline = np.asarray(predictions["phase2"], dtype=np.float64)
        best_key = summary["metrics"]["best_first_seed_screen"]["strategy"]
        require(best_key in predictions.files, f"Best strategy predictions are missing: {best_key}")
        best = np.asarray(predictions[best_key], dtype=np.float64)
    require(len(baseline) == len(labels) and len(best) == len(labels), "OOF row count mismatch")
    require(np.isfinite(baseline).all() and np.isfinite(best).all(), "Non-finite prediction values")
    measured_baseline_auc = float(roc_auc_score(labels, baseline))
    measured_best_auc = float(roc_auc_score(labels, best))
    reported_baseline_auc = float(summary["metrics"]["main_seed"]["phase2"]["pooled_oof_auc"])
    reported_best_auc = float(summary["metrics"]["best_first_seed_screen"]["pooled_oof_auc"])
    require(abs(measured_baseline_auc - reported_baseline_auc) <= 1e-12, "Phase 2 OOF AUC mismatch")
    require(abs(measured_best_auc - reported_best_auc) <= 1e-12, "Best strategy OOF AUC mismatch")

    folds_dir = Path(manifest["source_experiment_artifact_manifests"]["phase2_main"]["path"]).parent
    folds: list[np.ndarray] = []
    assignment = np.full(len(labels), -1, dtype=np.int8)
    for fold_number in range(1, 6):
        with np.load(folds_dir / f"fold_{fold_number}_indices.npz", allow_pickle=False) as values:
            indices = np.asarray(values["valid_indices"], dtype=np.int64)
        require(np.all(assignment[indices] == -1), "Phase 2 validation folds overlap")
        assignment[indices] = fold_number
        folds.append(indices)
    require(np.all(assignment >= 1), "Phase 2 folds do not cover every training row")
    fold_deltas = [
        float(
            roc_auc_score(labels[indices], best[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
        for indices in folds
    ]
    require(
        all(abs(left - right) <= 1e-12 for left, right in zip(
            fold_deltas,
            summary["metrics"]["best_first_seed_screen"]["fold_delta_vs_phase2"],
        )),
        "Best-strategy fold AUC deltas mismatch",
    )
    require(summary["promotion_eligible"] is False, "EXP09 unexpectedly marked eligible")
    require(summary["submission"]["created"] is False, "Rejected run must not create a submission")
    require(not any(run_dir.glob("submission*.csv")), "Rejected run contains a submission CSV")

    inventory = pd.read_csv(run_dir / "ARTIFACT_INVENTORY.csv")
    require(
        not inventory.duplicated(subset=["path", "prediction_role"]).any(),
        "Artifact inventory contains duplicate path/role rows",
    )
    for row in inventory.to_dict(orient="records"):
        artifact = Path(row["path"])
        require(artifact.is_file(), f"Inventoried source artifact missing: {artifact}")
        expected_hash = row.get("output_sha256")
        require(
            isinstance(expected_hash, str) and sha256_file(artifact) == expected_hash,
            f"Inventoried artifact SHA-256 mismatch: {artifact}",
        )

    eligible_oof = inventory[inventory["eligible_for_oof_comparison"].fillna(False)]
    require(not eligible_oof.empty, "No OOF artifacts are marked eligible")
    require(
        eligible_oof["row_alignment_verified"].fillna(False).all()
        and eligible_oof["folds_match_seed_reference"].fillna(False).all()
        and eligible_oof["input_hashes_verified"].fillna(False).all()
        and eligible_oof["source_manifest_hash_verified"].fillna(False).all(),
        "An eligible OOF artifact is missing a provenance or alignment check",
    )
    require(
        set(eligible_oof["prediction_id_sha256"].dropna().astype(str))
        .issubset({summary["prediction_id_records"]["train_prediction_id_sha256"]}),
        "Eligible OOF predictions do not share the verified train ID hash",
    )
    return {
        "status": "PASS",
        "main_phase2_auc": measured_baseline_auc,
        "best_first_screen_auc": measured_best_auc,
        "best_first_screen_delta": measured_best_auc - measured_baseline_auc,
        "fold_deltas": fold_deltas,
        "eligible_oof_artifacts": int(len(eligible_oof)),
        "outputs_verified": len(manifest["outputs"]),
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
