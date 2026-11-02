#!/usr/bin/env python3
"""Independent integrity and metric verification for an EXP10 replication."""

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

from run_exp10 import (
    ADVANCEMENT_GATE,
    EXPECTED_EXP08_BASE_SOURCE_HASHES,
    EXPECTED_INPUT_SHA256,
    FOLD_COUNT,
    ID_COLUMN,
    OUTER_SEED,
    TARGET_COLUMN,
    baseline_discrepancy,
    canonical_array_hash,
    canonical_float64_hash,
    canonical_id_hash,
    fold_indices,
    passes_gate,
    sha256_file,
    verify_fold_oof_stream,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_component_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "sha256_manifest.json"
    require(manifest_path.is_file(), f"Component manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in ("inputs", "references", "sources", "outputs"):
        for label, record in manifest.get(group, {}).items():
            path = Path(record["path"])
            require(
                path.is_file()
                and path.stat().st_size == record["size_bytes"]
                and sha256_file(path) == record["sha256"],
                f"Component {group} hash/size mismatch: {label}",
            )


def verify(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "SHA256_MANIFEST.json"
    summary_path = run_dir / "experiment_summary.json"
    predictions_path = run_dir / "replication_predictions.npz"
    shared_baseline_path = run_dir / "shared_baseline_contract.npz"
    require(manifest_path.is_file(), "EXP10 SHA256 manifest is missing")
    require(summary_path.is_file(), "EXP10 summary is missing")
    require(shared_baseline_path.is_file(), "Shared Phase 2 baseline contract is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    require(summary["seed"] == OUTER_SEED, "Unexpected replication seed")
    require(summary["pair"]["exact_exp09_strategy"] == "equal_exp07_signature__exp08_lgbcat",
            "The exact EXP09 selected pair changed")
    require(summary["pair"]["components"] == ["exp07_signature", "exp08_lgbcat"],
            "EXP10 components differ from the selected EXP09 pair")
    require(summary["pair"]["blend_rule"].startswith("fixed 50:50"),
            "The pair weights are not the predeclared equal weights")
    require(
        summary["discovery_reference"]["exp09_best_two_strategy"]
        == "equal_exp07_signature__exp08_lgbcat"
        and summary["discovery_reference"]["exp09_main_seed"] == 20261101,
        "EXP10 does not reference the exact EXP09 selected pair and discovery seed",
    )

    for group in ("inputs", "source_snapshots", "source_lineage", "outputs"):
        require(group in manifest and manifest[group], f"Manifest section is missing: {group}")
    for label, record in manifest["inputs"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Input is missing: {label}")
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[label],
            f"Input does not match the predeclared raw data hash: {label}",
        )
        require(sha256_file(path) == record["sha256"], f"Input hash mismatch: {label}")
        require(path.stat().st_size == record["size_bytes"], f"Input size mismatch: {label}")
    for group in ("source_snapshots", "source_lineage", "outputs"):
        for label, record in manifest[group].items():
            path = Path(record["path"])
            require(path.is_file(), f"{group} artifact is missing: {label}")
            require(sha256_file(path) == record["sha256"], f"{group} hash mismatch: {label}")
            require(path.stat().st_size == record["size_bytes"], f"{group} size mismatch: {label}")
    for name, expected_hash in EXPECTED_EXP08_BASE_SOURCE_HASHES.items():
        record = manifest["source_lineage"][f"exp08_original_{'runner' if name == 'run_exp08.py' else 'feature_builder'}_snapshot"]
        require(
            record["sha256"] == expected_hash,
            f"Archived original EXP08 source hash differs from its pinned hash: {name}",
        )
    require(
        manifest["source_snapshots"]["source_snapshot/historical_features.py"][
            "sha256"
        ]
        == EXPECTED_EXP08_BASE_SOURCE_HASHES["historical_features.py"],
        "EXP08 feature-builder execution snapshot differs from the original source",
    )
    require(
        "source_snapshot/FEATURE_INVENTORY.md" in manifest["source_snapshots"]
        and summary["source_lineage"]["exp07_feature_inventory_sha256"]
        == manifest["source_snapshots"]["source_snapshot/FEATURE_INVENTORY.md"][
            "sha256"
        ],
        "Required EXP07 feature inventory snapshot is missing or changed",
    )
    for label, summary_key in (
        ("exp10_runner", "exp10_runner_sha256"),
        ("exp10_verifier", "exp10_verifier_sha256"),
        ("exp10_readme", "exp10_readme_sha256"),
    ):
        require(
            manifest["source_lineage"][label]["sha256"]
            == summary["source_lineage"][summary_key],
            f"EXP10 source hash differs from its manifest record: {label}",
        )
    for test_name, expected_hash in summary["source_lineage"][
        "exp10_unit_tests_sha256"
    ].items():
        require(
            manifest["source_lineage"][f"exp10_test_{test_name}"]["sha256"]
            == expected_hash,
            f"EXP10 test source hash differs from its manifest: {test_name}",
        )
    require(
        sha256_file(predictions_path)
        == summary["integrity"]["prediction_archive_sha256"],
        "Prediction archive hash differs from the summary",
    )

    train_path = Path(manifest["inputs"]["train"]["path"])
    test_path = Path(manifest["inputs"]["test"]["path"])
    sample_path = Path(manifest["inputs"]["sample"]["path"])
    train = pd.read_csv(train_path, usecols=[ID_COLUMN, TARGET_COLUMN])
    test = pd.read_csv(test_path, usecols=[ID_COLUMN])
    sample = pd.read_csv(sample_path, usecols=[ID_COLUMN])
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    train_ids = train[ID_COLUMN].to_numpy()
    test_ids = test[ID_COLUMN].to_numpy()
    require(
        np.array_equal(test_ids, sample[ID_COLUMN].to_numpy()),
        "Test IDs do not match sample submission order",
    )

    with np.load(predictions_path, allow_pickle=False) as archive:
        expected_arrays = {
            "train_ids",
            "test_ids",
            "labels",
            "phase2_oof",
            "exp08_local_phase2_oof",
            "exp07_signature_oof",
            "exp08_lgbcat_oof",
            "pair_oof",
            "phase2_test",
            "exp08_local_phase2_test",
            "exp07_signature_test",
            "exp08_lgbcat_test",
            "pair_test",
        }
        require(expected_arrays.issubset(archive.files), "Prediction archive is incomplete")
        values = {key: np.asarray(archive[key]) for key in expected_arrays}
    require(np.array_equal(values["train_ids"], train_ids), "Stored train IDs are misaligned")
    require(np.array_equal(values["test_ids"], test_ids), "Stored test IDs are misaligned")
    require(np.array_equal(values["labels"], labels), "Stored labels differ from the input")
    require(
        summary["integrity"]["ordered_train_ids_sha256"] == canonical_id_hash(train_ids)
        and summary["integrity"]["ordered_test_ids_sha256"] == canonical_id_hash(test_ids),
        "Ordered row-ID hashes differ from the input data",
    )
    require(
        summary["integrity"]["test_rows"] == len(test_ids)
        and summary["integrity"]["test_pair_prediction_values_sha256"]
        == canonical_float64_hash(values["pair_test"]),
        "Test prediction count or canonical value hash differs from the saved predictions",
    )
    with np.load(shared_baseline_path, allow_pickle=False) as archive:
        contract = {key: np.asarray(archive[key]) for key in archive.files}
    expected_contract_arrays = {
        "train_ids",
        "test_ids",
        "labels",
        "fold_membership",
        "phase2_oof",
        "phase2_test",
    }
    require(
        expected_contract_arrays.issubset(contract),
        "Shared baseline contract archive is incomplete",
    )
    require(
        np.array_equal(contract["train_ids"], train_ids)
        and np.array_equal(contract["test_ids"], test_ids)
        and np.array_equal(contract["labels"], labels)
        and np.array_equal(contract["phase2_oof"], values["phase2_oof"])
        and np.array_equal(contract["phase2_test"], values["phase2_test"]),
        "Shared baseline contract differs from raw IDs, labels, or predictions",
    )
    shared_baseline = summary["pair"]["shared_phase2_baseline"]
    require(
        shared_baseline["file"]["sha256"] == sha256_file(shared_baseline_path)
        and shared_baseline["file"]["size_bytes"] == shared_baseline_path.stat().st_size
        and manifest["outputs"]["shared_baseline_contract.npz"]["sha256"]
        == sha256_file(shared_baseline_path),
        "Shared baseline contract file hash differs across its records",
    )
    expected_array_hashes = {
        "train_ids_sha256": canonical_id_hash(train_ids),
        "test_ids_sha256": canonical_id_hash(test_ids),
        "labels_sha256": canonical_array_hash(labels),
        "phase2_oof_sha256": canonical_array_hash(values["phase2_oof"]),
        "phase2_test_sha256": canonical_array_hash(values["phase2_test"]),
    }
    require(
        all(
            shared_baseline["arrays"][name] == digest
            for name, digest in expected_array_hashes.items()
        ),
        "Shared baseline array hashes differ from the saved contract",
    )
    for key in expected_arrays - {"train_ids", "test_ids", "labels"}:
        require(np.isfinite(values[key]).all(), f"Non-finite prediction values: {key}")
        require(
            np.all((values[key] >= 0.0) & (values[key] <= 1.0)),
            f"Prediction outside [0,1]: {key}",
        )
    exp07_run = Path(summary["component_runs"]["exp07"]["path"])
    exp08_run = Path(summary["component_runs"]["exp08"]["path"])
    require(
        exp07_run.is_dir() and exp08_run.is_dir(),
        "A component training run is missing",
    )
    verify_component_manifest(exp07_run)
    verify_component_manifest(exp08_run)
    for label, record in {
        "shared baseline OOF": shared_baseline["exp07_baseline_source"]["oof"],
        "shared baseline test": shared_baseline["exp07_baseline_source"]["test"],
        "EXP08 diagnostic OOF": shared_baseline[
            "exp08_local_baseline_diagnostic"
        ]["oof"],
        "EXP08 diagnostic test": shared_baseline[
            "exp08_local_baseline_diagnostic"
        ]["test"],
    }.items():
        path = Path(record["path"])
        require(
            path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and sha256_file(path) == record["sha256"],
            f"Baseline provenance artifact failed hash verification: {label}",
        )
    require(
        np.array_equal(
            values["exp08_local_phase2_oof"],
            np.load(exp08_run / "oof_phase2_matched.npy", allow_pickle=False),
        )
        and np.array_equal(
            values["exp08_local_phase2_test"],
            np.load(exp08_run / "test_phase2_matched.npy", allow_pickle=False),
        ),
        "Recorded EXP08 local Phase 2 diagnostics differ from their source artifacts",
    )
    local_baseline_diagnostics = shared_baseline["exp08_local_baseline_diagnostic"]
    recomputed_oof_discrepancy = baseline_discrepancy(
        values["phase2_oof"], values["exp08_local_phase2_oof"]
    )
    recomputed_test_discrepancy = baseline_discrepancy(
        values["phase2_test"], values["exp08_local_phase2_test"]
    )
    require(
        local_baseline_diagnostics["used_for_pair_scoring"] is False
        and local_baseline_diagnostics["oof_difference"]
        == recomputed_oof_discrepancy
        and local_baseline_diagnostics["test_difference"]
        == recomputed_test_discrepancy
        and summary["integrity"][
            "one_authoritative_phase2_baseline_used_for_all_reported_metrics"
        ]
        is True,
        "Independent component baseline discrepancy was hidden or used for scoring",
    )

    require(
        np.allclose(
            values["pair_oof"],
            (values["exp07_signature_oof"] + values["exp08_lgbcat_oof"]) / 2.0,
            rtol=0.0,
            atol=1e-7,
        ),
        "OOF pair does not equal the fixed 50:50 mean",
    )
    require(
        np.allclose(
            values["pair_test"],
            (values["exp07_signature_test"] + values["exp08_lgbcat_test"]) / 2.0,
            rtol=0.0,
            atol=1e-7,
        ),
        "Test pair does not equal the fixed 50:50 mean",
    )
    exp07_summary = json.loads(
        (exp07_run / "experiment_summary.json").read_text(encoding="utf-8")
    )
    exp08_summary = json.loads(
        (exp08_run / "experiment_summary.json").read_text(encoding="utf-8")
    )
    require(
        exp07_summary["outer_seed"] == OUTER_SEED
        and exp08_summary["outer_seed"] == OUTER_SEED,
        "A model component was not trained with the fresh EXP10 seed",
    )
    require(
        exp07_summary["model"]["catboost_gpu_task_type_verified"] is True
        and "GPU" in exp08_summary["models"]["catboost_device"].upper()
        and exp08_summary["gpu"]["telemetry"]["monitor_errors"] == [],
        "GPU-supported CatBoost training lacks confirmed GPU evidence",
    )
    require(
        exp08_summary["models"]["xgboost_gpu_preflight"]["cpu_fallback_used"]
        is False,
        "EXP08 XGBoost unexpectedly fell back to CPU",
    )
    folds: list[np.ndarray] = []
    expected_folds = list(
        StratifiedKFold(
            n_splits=FOLD_COUNT, shuffle=True, random_state=OUTER_SEED
        ).split(np.zeros(len(labels)), labels)
    )
    expected_fold_membership = np.full(len(labels), -1, dtype=np.int8)
    fold_contracts = shared_baseline["folds"]
    for fold in range(1, FOLD_COUNT + 1):
        fit_07, valid_07 = fold_indices(exp07_run, fold)
        fit_08, valid_08 = fold_indices(exp08_run, fold)
        expected_fit, expected_valid = expected_folds[fold - 1]
        require(
            np.array_equal(fit_07, expected_fit)
            and np.array_equal(valid_07, expected_valid)
            and np.array_equal(fit_08, expected_fit)
            and np.array_equal(valid_08, expected_valid),
            f"Fold {fold} does not match the declared StratifiedKFold seed",
        )
        expected_fold_membership[expected_valid] = fold
        fold_record = fold_contracts[fold - 1]
        require(
            fold_record["fold"] == fold
            and fold_record["fit_indices_sha256"] == canonical_array_hash(fit_07)
            and fold_record["valid_indices_sha256"]
            == canonical_array_hash(valid_07),
            f"Shared fold hash differs from component fold {fold}",
        )
        require(
            fold_record["phase2_valid_sha256"]
            == canonical_array_hash(values["phase2_oof"][valid_07]),
            f"Shared Phase 2 validation prediction hash differs in fold {fold}",
        )
        folds.append(valid_07)
    require(
        np.array_equal(contract["fold_membership"], expected_fold_membership)
        and shared_baseline["arrays"]["fold_membership_sha256"]
        == canonical_array_hash(expected_fold_membership),
        "Shared fold assignment differs from the declared five-fold partition",
    )
    verify_fold_oof_stream(
        exp07_run,
        "Phase 2 baseline",
        "phase2_reference",
        values["phase2_oof"].astype(np.float64),
    )
    verify_fold_oof_stream(
        exp07_run,
        "EXP07 signature",
        "candidate_signature",
        values["exp07_signature_oof"].astype(np.float64),
    )
    verify_fold_oof_stream(
        exp08_run,
        "EXP08 local Phase 2 diagnostic",
        "phase2_valid",
        values["exp08_local_phase2_oof"].astype(np.float64),
    )
    verify_fold_oof_stream(
        exp08_run,
        "EXP08 LightGBM/CatBoost",
        "historical_lgb_cat_valid",
        values["exp08_lgbcat_oof"].astype(np.float64),
    )

    metric_streams = {
        "phase2": values["phase2_oof"],
        "exp07_signature": values["exp07_signature_oof"],
        "exp08_lgbcat": values["exp08_lgbcat_oof"],
        "fixed_equal_pair": values["pair_oof"],
    }
    recomputed_metrics: dict[str, dict[str, Any]] = {}
    for name, predictions in metric_streams.items():
        fold_aucs = [
            float(roc_auc_score(labels[indices], predictions[indices]))
            for indices in folds
        ]
        recomputed_metrics[name] = {
            "pooled_oof_auc": float(roc_auc_score(labels, predictions)),
            "fold_auc": fold_aucs,
        }
    baseline_auc = recomputed_metrics["phase2"]["pooled_oof_auc"]
    pair_auc = recomputed_metrics["fixed_equal_pair"]["pooled_oof_auc"]
    baseline_fold_aucs = recomputed_metrics["phase2"]["fold_auc"]
    pair_fold_aucs = recomputed_metrics["fixed_equal_pair"]["fold_auc"]
    fold_deltas = [
        pair_fold_auc - baseline_fold_auc
        for pair_fold_auc, baseline_fold_auc in zip(
            pair_fold_aucs, baseline_fold_aucs
        )
    ]
    pooled_delta = pair_auc - baseline_auc
    positive_folds = sum(delta > 0.0 for delta in fold_deltas)
    for name, predictions in metric_streams.items():
        baseline = recomputed_metrics["phase2"]
        expected_deltas = [
            auc - phase2_auc
            for auc, phase2_auc in zip(
                recomputed_metrics[name]["fold_auc"],
                baseline["fold_auc"],
            )
        ]
        expected_positive_folds = sum(delta > 0.0 for delta in expected_deltas)
        reported = summary["metrics"][name]
        require(
            abs(
                reported["pooled_oof_auc"]
                - recomputed_metrics[name]["pooled_oof_auc"]
            )
            <= 1e-12
            and abs(
                reported["pooled_delta_vs_phase2"]
                - (
                    recomputed_metrics[name]["pooled_oof_auc"]
                    - baseline["pooled_oof_auc"]
                )
            )
            <= 1e-12
            and np.allclose(
                reported["fold_auc"],
                recomputed_metrics[name]["fold_auc"],
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                reported["fold_delta_vs_phase2"],
                expected_deltas,
                rtol=0.0,
                atol=1e-12,
            )
            and reported["positive_folds"] == expected_positive_folds,
            f"Reported AUC metrics do not match recomputed stream: {name}",
        )
    gate_passed = passes_gate(pooled_delta, positive_folds)
    require(
        abs(summary["advancement_gate"]["pooled_auc_delta"] - pooled_delta) <= 1e-12
        and summary["advancement_gate"]["positive_folds"] == positive_folds
        and summary["promotion_eligible"] is gate_passed,
        "Advancement gate in summary is inconsistent",
    )

    candidate_path = run_dir / "candidate_submission.csv"
    if gate_passed:
        require(candidate_path.is_file(), "Passing gate must create a candidate submission")
        candidate = pd.read_csv(candidate_path, usecols=[ID_COLUMN, TARGET_COLUMN])
        require(
            np.array_equal(candidate[ID_COLUMN].to_numpy(), test_ids)
            and np.allclose(
                candidate[TARGET_COLUMN].to_numpy(dtype=np.float64),
                values["pair_test"],
                rtol=0.0,
                atol=1e-7,
            ),
            "Candidate submission IDs or probabilities are misaligned",
        )
        require(
            sha256_file(candidate_path)
            == summary["candidate_submission"]["sha256"],
            "Candidate submission hash differs from summary",
        )
    else:
        require(
            not candidate_path.exists()
            and summary["candidate_submission_created"] is False,
            "Rejected replication must not create a submission",
        )
    require(
        not any(run_dir.rglob("submission_*.csv"))
        and not any(run_dir.rglob("submission_exp08_*.csv")),
        "Component run unexpectedly emitted a submission CSV",
    )

    return {
        "status": "PASS",
        "seed": OUTER_SEED,
        "phase2_oof_auc": baseline_auc,
        "pair_oof_auc": pair_auc,
        "pooled_delta": pooled_delta,
        "fold_deltas": fold_deltas,
        "positive_folds": positive_folds,
        "advancement_gate_passed": gate_passed,
        "candidate_submission_created": candidate_path.is_file(),
        "outputs_verified": len(manifest["outputs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "VERIFICATION.json"

    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def record_output(path: Path) -> None:
        manifest_path = run_dir / "SHA256_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["outputs"]["VERIFICATION.json"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        write_json(manifest_path, manifest)

    result = verify(run_dir)
    write_json(report_path, result)
    record_output(report_path)
    result = verify(run_dir)
    write_json(report_path, result)
    record_output(report_path)
    final_result = verify(run_dir)
    require(final_result == result, "Saved verifier report does not match final verification")
    print(json.dumps(final_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
