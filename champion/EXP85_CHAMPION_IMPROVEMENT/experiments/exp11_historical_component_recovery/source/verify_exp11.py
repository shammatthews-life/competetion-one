#!/usr/bin/env python3
"""Independently verify EXP11 artifacts, folds, hashes, and saved-model replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run_exp11 import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_HISTORICAL_SOURCE_SHA256,
    EXPECTED_INPUT_SHA256,
    FOLD_COUNT,
    FOLD_SEED,
    ID_COLUMN,
    TARGET_COLUMN,
    build_xgb_features,
    sha256_array,
    sha256_file,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp11_xgb10_seed20261102"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(run_dir: Path, replay_models: bool = True) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "SHA256_MANIFEST.json"
    summary_path = run_dir / "experiment_summary.json"
    require(manifest_path.is_file(), "SHA256 manifest is missing")
    require(summary_path.is_file(), "Experiment summary is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["experiment_id"] == "exp11_xgb10_seed20261102", "Wrong run ID")
    require(summary["fold_seed"] == FOLD_SEED, "Fold seed mismatch")
    require(summary["folds"] == FOLD_COUNT, "Fold count mismatch")
    require(summary["candidate_submission_created"] is False, "Unexpected submission")

    for name, record in manifest["inputs"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Input missing: {name}")
        require(
            record["sha256"] == sha256_file(path)
            and record["size_bytes"] == path.stat().st_size,
            f"Input hash/size mismatch: {name}",
        )
    require(
        manifest["inputs"]["train.csv"]["sha256"]
        == EXPECTED_INPUT_SHA256["train.csv"],
        "Training input is not the pinned raw dataset",
    )
    require(
        manifest["inputs"]["test.csv"]["sha256"]
        == EXPECTED_INPUT_SHA256["test.csv"],
        "Test input is not the pinned raw dataset",
    )
    require(
        manifest["inputs"]["shared_baseline_contract.npz"]["sha256"]
        == EXPECTED_BASELINE_SHA256,
        "Shared baseline contract hash is not the verified EXP10 artifact",
    )
    require(
        sha256_file(run_dir / "shared_baseline_contract.npz")
        == EXPECTED_BASELINE_SHA256,
        "Copied shared baseline contract hash differs from the verified artifact",
    )
    for label, record in manifest["sources"].items():
        path = Path(record["path"])
        require(
            path.is_file()
            and sha256_file(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            f"Source hash/size mismatch: {label}",
        )
    require(
        manifest["sources"]["historical_xgb10_source.py"]["sha256"]
        == EXPECTED_HISTORICAL_SOURCE_SHA256,
        "Pinned historical XGB10 source checksum differs",
    )
    for name, record in manifest["outputs"].items():
        path = run_dir / name
        require(
            path.is_file()
            and sha256_file(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            f"Output hash/size mismatch: {name}",
        )

    data_dir = Path(manifest["inputs"]["train.csv"]["path"]).parent
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    with np.load(run_dir / "shared_baseline_contract.npz", allow_pickle=False) as z:
        ids = z["train_ids"]
        test_ids = z["test_ids"]
        baseline_labels = z["labels"]
        folds = z["fold_membership"].astype(np.int8)
        phase2_oof = z["phase2_oof"]
    require(np.array_equal(ids, train[ID_COLUMN].to_numpy()), "Train ID order mismatch")
    require(
        np.array_equal(test_ids, test[ID_COLUMN].to_numpy()),
        "Test ID order mismatch",
    )
    require(np.array_equal(labels, baseline_labels), "Label order mismatch")
    expected_folds = np.zeros(len(labels), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=FOLD_SEED)
    for fold_number, (_, validation) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        expected_folds[validation] = fold_number
    require(np.array_equal(folds, expected_folds), "Fold membership differs from seed")

    oof = np.load(run_dir / "oof_xgb10_strict.npy", allow_pickle=False)
    test_prediction = np.load(
        run_dir / "test_xgb10_strict.npy", allow_pickle=False
    )
    test_by_fold = np.load(
        run_dir / "test_xgb10_by_fold.npy", allow_pickle=False
    )
    require(oof.shape == labels.shape, "OOF shape mismatch")
    require(test_prediction.shape == (len(test),), "Test shape mismatch")
    require(test_by_fold.shape == (FOLD_COUNT, len(test)), "Fold test shape mismatch")
    require(
        np.isfinite(oof).all()
        and np.isfinite(test_prediction).all()
        and np.all((oof >= 0) & (oof <= 1))
        and np.all((test_prediction >= 0) & (test_prediction <= 1)),
        "Prediction values are invalid",
    )
    require(
        np.array_equal(
            test_prediction,
            np.mean(test_by_fold, axis=0, dtype=np.float64).astype(np.float32),
        ),
        "Test aggregation does not match the five fold streams",
    )

    fold_results = []
    for fold_number in range(1, FOLD_COUNT + 1):
        index_path = run_dir / f"fold_{fold_number}_indices.npz"
        with np.load(index_path, allow_pickle=False) as index_archive:
            outer_train = index_archive["outer_train"]
            outer_valid = index_archive["outer_valid"]
            inner_fit = index_archive["inner_fit"]
            inner_stop = index_archive["inner_stop"]
        expected_valid = np.flatnonzero(folds == fold_number)
        expected_train = np.flatnonzero(folds != fold_number)
        require(np.array_equal(outer_valid, expected_valid), f"Fold {fold_number} valid mismatch")
        require(np.array_equal(outer_train, expected_train), f"Fold {fold_number} train mismatch")
        require(
            not np.intersect1d(outer_train, outer_valid).size,
            f"Outer train/valid overlap in fold {fold_number}",
        )
        require(
            not np.intersect1d(outer_valid, inner_fit).size
            and not np.intersect1d(outer_valid, inner_stop).size
            and not np.intersect1d(inner_fit, inner_stop).size,
            f"Fold {fold_number} outer validation contaminated early stopping",
        )
        require(
            np.array_equal(np.sort(np.concatenate([inner_fit, inner_stop])), outer_train),
            f"Fold {fold_number} inner split does not partition outer training rows",
        )
        candidate_auc = float(roc_auc_score(labels[outer_valid], oof[outer_valid]))
        phase2_auc = float(
            roc_auc_score(labels[outer_valid], phase2_oof[outer_valid])
        )
        fold_results.append(
            {
                "fold": fold_number,
                "candidate_auc": candidate_auc,
                "phase2_auc": phase2_auc,
                "delta": candidate_auc - phase2_auc,
            }
        )

        if replay_models:
            model = xgb.XGBClassifier()
            model.load_model(run_dir / f"fold_{fold_number}_xgb.ubj")
            model.set_params(device="cuda:0")
            apply_rows = pd.concat(
                [train.iloc[outer_valid], test], ignore_index=True
            )
            x_fit, x_apply = build_xgb_features(train.iloc[outer_train], apply_rows)
            x_valid = x_apply.iloc[: len(outer_valid)].reset_index(drop=True)
            x_test = x_apply.iloc[len(outer_valid) :].reset_index(drop=True)
            replay_valid = model.predict_proba(x_valid)[:, 1].astype(np.float32)
            replay_test = model.predict_proba(x_test)[:, 1].astype(np.float32)
            require(
                np.allclose(replay_valid, oof[outer_valid], rtol=0.0, atol=1e-7),
                f"Saved model validation replay differs in fold {fold_number}",
            )
            require(
                np.allclose(replay_test, test_by_fold[fold_number - 1], rtol=0.0, atol=1e-7),
                f"Saved model test replay differs in fold {fold_number}",
            )

    pooled_baseline = float(roc_auc_score(labels, phase2_oof))
    pooled_candidate = float(roc_auc_score(labels, oof))
    delta = pooled_candidate - pooled_baseline
    positive_folds = sum(item["delta"] > 0.0 for item in fold_results)
    gate_passed = delta >= 0.001 and positive_folds >= 4
    require(
        math_isclose(summary["metrics"]["phase2_pooled_oof_auc"], pooled_baseline),
        "Summary Phase 2 AUC mismatch",
    )
    require(
        math_isclose(summary["metrics"]["xgb10_strict_pooled_oof_auc"], pooled_candidate),
        "Summary XGB10 AUC mismatch",
    )
    require(math_isclose(summary["metrics"]["absolute_delta_vs_phase2"], delta), "Summary delta mismatch")
    require(summary["advancement_gate"]["passed"] == gate_passed, "Gate decision mismatch")

    verification = {
        "experiment_id": summary["experiment_id"],
        "passed": True,
        "replayed_saved_models": replay_models,
        "model_replay_device": "cuda:0" if replay_models else None,
        "verified_output_count": len(manifest["outputs"]),
        "verified_outer_folds": FOLD_COUNT,
        "phase2_pooled_oof_auc": pooled_baseline,
        "candidate_pooled_oof_auc": pooled_candidate,
        "delta_vs_phase2": delta,
        "positive_folds": positive_folds,
        "gate_passed": gate_passed,
        "train_ids_sha256": sha256_array(ids),
        "test_ids_sha256": sha256_array(test_ids),
        "labels_sha256": sha256_array(labels),
        "fold_membership_sha256": sha256_array(folds),
        "oof_sha256": sha256_array(oof),
        "test_sha256": sha256_array(test_prediction),
        "fold_metrics": fold_results,
        "decision": (
            "OOF_GATE_PASSED_RESEARCH_ONLY"
            if gate_passed and replay_models
            else "REJECTED"
            if not gate_passed
            else "VERIFICATION_INCOMPLETE"
        ),
    }
    verification_path = run_dir / "VERIFICATION.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest["sources"]["verify_exp11.py"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
        "size_bytes": Path(__file__).stat().st_size,
    }
    manifest["outputs"]["VERIFICATION.json"] = {
        "sha256": sha256_file(verification_path),
        "size_bytes": verification_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return verification


def math_isclose(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--skip-model-replay", action="store_true")
    args = parser.parse_args()
    result = verify(args.run_dir, replay_models=not args.skip_model_replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
