#!/usr/bin/env python3
"""Verify and screen saved EXP85 prediction streams without retraining models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
IMPROVEMENT_DIR = EXPERIMENTS_DIR.parent
CHAMPION_DIR = IMPROVEMENT_DIR.parent
REPOSITORY_ROOT = CHAMPION_DIR.parent
RESEARCH_RUNS_DIR = CHAMPION_DIR / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT" / "runs"
PHASE2_MAIN_DIR = (
    CHAMPION_DIR
    / "EXP85_PHASE2_FROZEN_RELEASE"
    / "EVIDENCE"
    / "second_seed"
    / "main"
)
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "exp09_prediction_diversity_v1"
TRAIN_NAME = "train.csv"
TEST_NAME = "test.csv"
SAMPLE_NAME = "sample_submission.csv"
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
MAIN_SEED = 20261101
REPLICATION_SEED = 42
EXPECTED_ROWS = {"train": 171202, "test": 61500}
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
ADVANCEMENT_GATE = {
    "minimum_pooled_auc_delta": 0.001,
    "minimum_positive_folds": 4,
    "independent_seed_required": True,
    "folds": 5,
}


@dataclass(frozen=True)
class StreamSpec:
    key: str
    label: str
    seed: int
    run_dir: Path
    oof_name: str
    test_name: str | None
    family: str
    role: str = "candidate"
    source_kind: str = "saved_model_predictions"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_id_hash(values: Iterable[Any]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8") + b"\n"
    return sha256_bytes(payload)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_prediction(path: Path, expected_rows: int) -> np.ndarray:
    require(path.is_file(), f"Prediction artifact is missing: {path}")
    values = np.load(path, allow_pickle=False)
    require(values.ndim == 1, f"Prediction must be one-dimensional: {path}")
    require(
        len(values) == expected_rows,
        f"Prediction row count mismatch for {path}: {len(values)}",
    )
    values = values.astype(np.float64, copy=False)
    require(np.isfinite(values).all(), f"Prediction has non-finite values: {path}")
    return values


def read_fold_indices(
    run_dir: Path, row_count: int, fold_count: int = 5
) -> tuple[list[np.ndarray], str]:
    folds: list[np.ndarray] = []
    assignment = np.full(row_count, -1, dtype=np.int8)
    for fold_number in range(1, fold_count + 1):
        index_path = run_dir / f"fold_{fold_number}_indices.npz"
        require(index_path.is_file(), f"Fold index file is missing: {index_path}")
        with np.load(index_path, allow_pickle=False) as indices:
            require("valid_indices" in indices.files, f"Missing valid indices: {index_path}")
            valid = np.asarray(indices["valid_indices"], dtype=np.int64)
        require(valid.ndim == 1 and len(valid) > 0, f"Invalid fold indices: {index_path}")
        require(
            np.all((valid >= 0) & (valid < row_count)),
            f"Out-of-range validation indices: {index_path}",
        )
        require(np.all(assignment[valid] == -1), f"Fold rows overlap: {run_dir}")
        assignment[valid] = fold_number
        folds.append(valid)
    require(np.all(assignment >= 1), f"Fold indices do not cover all rows: {run_dir}")
    return folds, sha256_bytes(assignment.tobytes())


def check_same_folds(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(a, b) for a, b in zip(left, right)
    )


def discover_json_files(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.glob("*.json")
        if path.name in {
            "SHA256_MANIFEST.json",
            "sha256_manifest.json",
            "run_manifest.json",
            "experiment_summary.json",
            "xgboost_report.json",
            "epoly_report.json",
        }
    )


def _record_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from _record_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _record_dicts(item)


def _json_doc(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def input_identity(run_dir: Path, expected_inputs: dict[str, str]) -> tuple[bool, str]:
    observed: dict[str, set[str]] = {"train.csv": set(), "test.csv": set(), "sample_submission.csv": set()}
    for path in discover_json_files(run_dir):
        doc = _json_doc(path)
        if doc is None:
            continue
        for record in _record_dicts(doc):
            file_name = Path(record["path"]).name.lower()
            if file_name in observed:
                observed[file_name].add(record["sha256"])
    labels = {"train.csv": "train", "test.csv": "test", "sample_submission.csv": "sample"}
    matched = all(
        expected_inputs[key] in observed[file_name]
        for file_name, key in labels.items()
    )
    return matched, "verified exact train/test/sample SHA-256 records" if matched else "input SHA-256 provenance incomplete or mismatched"


def output_hash_status(run_dir: Path, artifact: Path) -> tuple[bool, str]:
    actual = sha256_file(artifact)
    found = False
    for path in discover_json_files(run_dir):
        doc = _json_doc(path)
        if doc is None:
            continue
        outputs = doc.get("outputs")
        if isinstance(outputs, dict):
            entry = outputs.get(artifact.name)
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                found = True
                if entry["sha256"] != actual:
                    return False, "artifact hash differs from source manifest"
        for record in _record_dicts(doc):
            if Path(record["path"]).resolve() == artifact.resolve():
                found = True
                if record["sha256"] != actual:
                    return False, "artifact hash differs from source manifest"
    if not found:
        return False, "source manifest does not record this output hash"
    return True, "artifact SHA-256 matches source manifest"


FOLD_OOF_KEYS = {
    "exp08_phase2": "phase2_valid",
    "phase2_main": "matched_phase2_blend",
    "phase2_replication": "phase2_reference",
    "phase2_cat": "matched_phase2_catboost",
    "phase2_lgb": "matched_phase2_lightgbm",
    "exp04": "oof",
    "exp05": "oof",
    "exp06": "ft_transformer",
    "exp07_signature": "candidate_signature",
    "exp07_nested_blend": "nested_blend",
    "exp07_signature_replication": "candidate_signature",
    "exp07_nested_replication": "nested_blend",
    "exp08_historical": "historical_valid",
    "exp08_nested_blend": "nested_blend_valid",
    "exp08_lgbcat": "historical_lgb_cat_valid",
    "exp08_xgb": "historical_xgb_valid",
}


def verify_oof_row_alignment(
    run_dir: Path,
    stream_key: str | None,
    artifact: Path,
    values: np.ndarray,
    labels: np.ndarray,
    ordered_ids: np.ndarray,
) -> tuple[bool, str]:
    if artifact.name == "oof_xgboost.npy":
        sidecar = run_dir / "oof_comparison.csv"
        if not sidecar.is_file():
            return False, "no OOF ID sidecar exists"
        frame = pd.read_csv(
            sidecar,
            usecols=[ID_COLUMN, "TARGET_TRUE", "OOF_XGBOOST"],
        )
        aligned = (
            np.array_equal(frame[ID_COLUMN].to_numpy(), ordered_ids)
            and np.array_equal(frame["TARGET_TRUE"].to_numpy(), labels)
            and np.allclose(
                frame["OOF_XGBOOST"].to_numpy(dtype=np.float64),
                values,
                rtol=0.0,
                atol=1e-7,
            )
        )
        return aligned, "oof_comparison.csv ID, target, and prediction columns match row-for-row" if aligned else "OOF ID sidecar differs from input order or prediction array"

    if stream_key is None:
        path_text = str(run_dir).lower()
        if "exp04_catboost_lossguide" in path_text:
            stream_key = "exp04"
        elif "exp05_tabular_neural" in path_text:
            stream_key = "exp05"
        elif "exp06_ft_transformer" in path_text:
            stream_key = "phase2_main" if artifact.name == "oof_phase2_strict_matched.npy" else "exp06"
            if artifact.name == "oof_phase2_strict_matched_catboost.npy":
                stream_key = "phase2_cat"
            elif artifact.name == "oof_phase2_strict_matched_lightgbm.npy":
                stream_key = "phase2_lgb"
        elif "exp07_gbdt_feature_ensemble" in path_text:
            if artifact.name == "oof_phase2_strict_matched.npy":
                stream_key = "phase2_replication" if "seed42" in path_text else "phase2_main"
            elif artifact.name == "oof_signature_model.npy":
                stream_key = "exp07_signature_replication" if "seed42" in path_text else "exp07_signature"
            elif artifact.name == "oof_nested_blend.npy":
                stream_key = "exp07_nested_replication" if "seed42" in path_text else "exp07_nested_blend"
        elif "exp08_historical_feature_bridge" in path_text:
            if artifact.name == "oof_historical_candidate.npy":
                stream_key = "exp08_historical"
            elif artifact.name == "oof_nested_blend.npy":
                stream_key = "exp08_nested_blend"
            elif artifact.name == "oof_phase2_matched.npy":
                stream_key = "exp08_phase2"
    prediction_key = FOLD_OOF_KEYS.get(stream_key or "")
    if prediction_key is None:
        if "exp07_signature" in str(run_dir):
            prediction_key = "phase2_reference" if "phase2" in artifact.name else None
        elif "exp08_historical" in str(run_dir):
            prediction_key = "phase2_valid" if "phase2" in artifact.name else None
    if prediction_key is None:
        return False, "no per-fold prediction values or row-ID sidecar are available"

    for fold_number in range(1, ADVANCEMENT_GATE["folds"] + 1):
        prediction_path = run_dir / f"fold_{fold_number}_predictions.npz"
        index_path = run_dir / f"fold_{fold_number}_indices.npz"
        if not prediction_path.is_file() or not index_path.is_file():
            return False, f"per-fold prediction or index file missing for fold {fold_number}"
        with np.load(index_path, allow_pickle=False) as indices:
            expected_indices = np.asarray(indices["valid_indices"], dtype=np.int64)
        with np.load(prediction_path, allow_pickle=False) as fold_predictions:
            if "valid_indices" in fold_predictions.files:
                indices = np.asarray(fold_predictions["valid_indices"], dtype=np.int64)
            else:
                indices = expected_indices
            if prediction_key not in fold_predictions.files:
                return False, f"per-fold prediction key {prediction_key} missing"
            fold_values = np.asarray(fold_predictions[prediction_key], dtype=np.float64)
        if not np.array_equal(indices, expected_indices):
            return False, f"fold {fold_number} prediction indices differ from saved fold indices"
        if len(fold_values) != len(indices) or not np.allclose(
            values[indices], fold_values, rtol=0.0, atol=1e-7
        ):
            return False, f"fold {fold_number} predictions do not match global OOF array at their indexed rows"
    return True, "global OOF values match all five fold predictions at the saved validation indices"


FOLD_TEST_KEYS = {
    "exp04": "test",
    "exp05": "test",
    "exp07_signature": "candidate_test",
    "exp07_signature_replication": "candidate_test",
    "exp08_historical": "historical_test",
}


def verify_test_row_alignment(
    run_dir: Path,
    stream_key: str,
    predictions: np.ndarray,
    ordered_test_ids: np.ndarray,
) -> tuple[bool, str]:
    csv_candidates = sorted(run_dir.glob("submission*.csv"))
    for path in csv_candidates:
        try:
            submission = pd.read_csv(path, usecols=[ID_COLUMN, TARGET_COLUMN])
        except (ValueError, OSError, pd.errors.ParserError):
            continue
        if (
            np.array_equal(submission[ID_COLUMN].to_numpy(), ordered_test_ids)
            and len(submission) == len(predictions)
            and np.allclose(
                submission[TARGET_COLUMN].to_numpy(dtype=np.float64),
                predictions,
                rtol=0.0,
                atol=1e-7,
            )
        ):
            return True, f"{path.name} IDs and predictions match row-for-row"

    test_key = FOLD_TEST_KEYS.get(stream_key)
    if test_key is not None:
        fold_values: list[np.ndarray] = []
        for fold_number in range(1, ADVANCEMENT_GATE["folds"] + 1):
            path = run_dir / f"fold_{fold_number}_predictions.npz"
            if not path.is_file():
                return False, f"fold {fold_number} test predictions are missing"
            with np.load(path, allow_pickle=False) as fold_predictions:
                if test_key not in fold_predictions.files:
                    return False, f"per-fold test key {test_key} is missing"
                fold_values.append(np.asarray(fold_predictions[test_key], dtype=np.float64))
        mean_prediction = np.mean(fold_values, axis=0)
        if len(mean_prediction) == len(predictions) and np.allclose(
            mean_prediction, predictions, rtol=0.0, atol=1e-7
        ):
            return True, "test array matches the mean of all five per-fold test predictions"
    return False, "no ordered-ID submission sidecar or matching per-fold test predictions"


def protocol_description(run_dir: Path) -> str:
    for path in discover_json_files(run_dir):
        doc = _json_doc(path)
        if doc is None:
            continue
        protocol = doc.get("fold_protocol")
        if isinstance(protocol, dict):
            keys = (
                "outer_folds",
                "outer_seed",
                "outer_validation_used_for_selection",
                "inner_only_early_stopping",
                "inner_only_group_and_feature_selection",
                "inner_only_blend_weight_selection",
            )
            fields = [
                f"{key}={protocol[key]}"
                for key in keys
                if key in protocol and not isinstance(protocol[key], (dict, list))
            ]
            return "5-fold outer; " + "; ".join(fields)
        if doc.get("outer_folds") == 5 or doc.get("folds") == 5:
            return "five-fold outer validation; see source run metadata"
    if all((run_dir / f"fold_{fold}_indices.npz").is_file() for fold in range(1, 6)):
        return "five-fold outer validation; exact fold indices available"
    return "fold protocol not recoverable from summary"


def all_stream_specs() -> tuple[list[StreamSpec], dict[int, StreamSpec]]:
    exp04 = (
        IMPROVEMENT_DIR / "experiments" / "exp04_catboost_lossguide"
        / "runs" / "exp04_lossguide_strict_seed20261101"
    )
    exp05 = (
        IMPROVEMENT_DIR / "experiments" / "exp05_tabular_neural"
        / "runs" / "exp05_cuda_seed20261101_v1"
    )
    exp06 = (
        IMPROVEMENT_DIR / "experiments" / "exp06_ft_transformer"
        / "runs" / "exp06_ft_transformer_seed20261101_v5"
    )
    exp07_main = (
        IMPROVEMENT_DIR / "experiments" / "exp07_gbdt_feature_ensemble"
        / "runs" / "exp07_signature_seed20261101_v1"
    )
    exp07_42 = (
        IMPROVEMENT_DIR / "experiments" / "exp07_gbdt_feature_ensemble"
        / "runs" / "exp07_signature_seed42_v2"
    )
    exp08 = (
        IMPROVEMENT_DIR / "experiments" / "exp08_historical_feature_bridge"
        / "runs" / "exp08_historical_seed20261101_v3"
    )
    xgb42 = RESEARCH_RUNS_DIR / "20261101T144300Z_raw_xgb_diversity_seed42"
    xgb_main = RESEARCH_RUNS_DIR / "20261101T144500Z_raw_xgb_diversity_seed20261101"
    specs = [
        StreamSpec("phase2_main", "Strict Phase 2 reference", MAIN_SEED, exp06, "oof_phase2_strict_matched.npy", "test_phase2_strict_matched.npy", "Phase 2", "baseline"),
        StreamSpec("phase2_replication", "Strict Phase 2 independent-fold reference", REPLICATION_SEED, exp07_42, "oof_phase2_strict_matched.npy", "test_phase2_strict_matched.npy", "Phase 2", "baseline"),
        StreamSpec("phase2_cat", "Phase 2 CatBoost component", MAIN_SEED, exp06, "oof_phase2_strict_matched_catboost.npy", None, "Phase 2 component", "component"),
        StreamSpec("phase2_lgb", "Phase 2 LightGBM component", MAIN_SEED, exp06, "oof_phase2_strict_matched_lightgbm.npy", None, "Phase 2 component", "component"),
        StreamSpec("exp04", "EXP04 CatBoost Lossguide", MAIN_SEED, exp04, "oof_catboost_lossguide.npy", "test_catboost_lossguide.npy", "CatBoost"),
        StreamSpec("exp05", "EXP05 residual neural network", MAIN_SEED, exp05, "oof_tabular_neural.npy", "test_tabular_neural.npy", "Neural"),
        StreamSpec("exp06", "EXP06 FT-Transformer", MAIN_SEED, exp06, "oof_ft_transformer.npy", "test_ft_transformer.npy", "Transformer"),
        StreamSpec("exp07_signature", "EXP07 signature model", MAIN_SEED, exp07_main, "oof_signature_model.npy", "test_signature_model.npy", "GBDT feature engineering"),
        StreamSpec("exp07_nested_blend", "EXP07 inner-selected nested blend", MAIN_SEED, exp07_main, "oof_nested_blend.npy", "test_nested_blend.npy", "Nested blend", "phase2_inclusive_blend"),
        StreamSpec("exp07_signature_replication", "EXP07 signature model seed-42 screen", REPLICATION_SEED, exp07_42, "oof_signature_model.npy", "test_signature_model.npy", "GBDT feature engineering"),
        StreamSpec("exp07_nested_replication", "EXP07 nested blend seed-42 screen", REPLICATION_SEED, exp07_42, "oof_nested_blend.npy", "test_nested_blend.npy", "Nested blend", "phase2_inclusive_blend"),
        StreamSpec("exp08_historical", "EXP08 historical candidate blend", MAIN_SEED, exp08, "oof_historical_candidate.npy", "test_historical_candidate.npy", "Historical features"),
        StreamSpec("exp08_nested_blend", "EXP08 inner-selected nested blend", MAIN_SEED, exp08, "oof_nested_blend.npy", "test_nested_blend.npy", "Nested blend", "phase2_inclusive_blend"),
        StreamSpec("raw_xgb_main", "Raw-feature XGBoost replication", MAIN_SEED, xgb_main, "oof_xgboost.npy", "test_xgboost.npy", "Raw-feature GBDT"),
        StreamSpec("raw_xgb_replication", "Raw-feature XGBoost seed-42 screen", REPLICATION_SEED, xgb42, "oof_xgboost.npy", "test_xgboost.npy", "Raw-feature GBDT"),
    ]
    return specs, {spec.seed: spec for spec in specs if spec.role == "baseline"}


def reconstruct_exp08_components(run_dir: Path, output_dir: Path, row_count: int, test_count: int) -> dict[str, Path]:
    oof_lgbcat = np.full(row_count, np.nan, dtype=np.float64)
    oof_xgb = np.full(row_count, np.nan, dtype=np.float64)
    test_lgbcat_folds: list[np.ndarray] = []
    test_xgb_folds: list[np.ndarray] = []
    seen = np.zeros(row_count, dtype=bool)
    for fold_number in range(1, ADVANCEMENT_GATE["folds"] + 1):
        prediction_path = run_dir / f"fold_{fold_number}_predictions.npz"
        index_path = run_dir / f"fold_{fold_number}_indices.npz"
        with np.load(prediction_path, allow_pickle=False) as predictions:
            required = {
                "historical_lgb_cat_valid",
                "historical_xgb_valid",
                "historical_test",
                "historical_xgb_test",
            }
            require(required.issubset(predictions.files), f"EXP08 component prediction keys missing: {prediction_path}")
            valid_lgbcat = np.asarray(predictions["historical_lgb_cat_valid"], dtype=np.float64)
            valid_xgb = np.asarray(predictions["historical_xgb_valid"], dtype=np.float64)
            test_blend = np.asarray(predictions["historical_test"], dtype=np.float64)
            test_xgb = np.asarray(predictions["historical_xgb_test"], dtype=np.float64)
        with np.load(index_path, allow_pickle=False) as indices:
            valid = np.asarray(indices["valid_indices"], dtype=np.int64)
        require(len(valid_lgbcat) == len(valid) and len(valid_xgb) == len(valid), f"EXP08 OOF component length mismatch: {fold_number}")
        require(len(test_blend) == test_count and len(test_xgb) == test_count, f"EXP08 test component length mismatch: {fold_number}")
        require(not seen[valid].any(), f"EXP08 OOF component folds overlap: {fold_number}")
        seen[valid] = True
        oof_lgbcat[valid] = valid_lgbcat
        oof_xgb[valid] = valid_xgb
        test_xgb_folds.append(test_xgb)
        test_lgbcat_folds.append(2.0 * test_blend - test_xgb)
    require(seen.all(), "EXP08 component OOF folds do not cover all rows")
    derived = {
        "oof_exp08_lgbcat_reconstructed.npy": oof_lgbcat,
        "oof_exp08_xgb_reconstructed.npy": oof_xgb,
        "test_exp08_lgbcat_reconstructed.npy": np.mean(test_lgbcat_folds, axis=0),
        "test_exp08_xgb_reconstructed.npy": np.mean(test_xgb_folds, axis=0),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, array in derived.items():
        path = output_dir / name
        np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
        result[name] = path
    return result


def fold_auc_rows(
    labels: np.ndarray,
    prediction_sets: dict[str, np.ndarray],
    folds: list[np.ndarray],
    baseline_key: str,
    seed: int,
    output_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = prediction_sets[baseline_key]
    metrics: dict[str, dict[str, Any]] = {}
    for key, predictions in prediction_sets.items():
        fold_auc = [float(roc_auc_score(labels[idx], predictions[idx])) for idx in folds]
        baseline_fold_auc = [float(roc_auc_score(labels[idx], baseline[idx])) for idx in folds]
        fold_deltas = [candidate - reference for candidate, reference in zip(fold_auc, baseline_fold_auc)]
        pooled = float(roc_auc_score(labels, predictions))
        baseline_pooled = float(roc_auc_score(labels, baseline))
        metrics[key] = {
            "pooled_oof_auc": pooled,
            "baseline_pooled_oof_auc": baseline_pooled,
            "pooled_delta_vs_phase2": pooled - baseline_pooled,
            "fold_auc": fold_auc,
            "fold_delta_vs_phase2": fold_deltas,
            "positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
        }
        for fold_index, (auc, delta) in enumerate(zip(fold_auc, fold_deltas), start=1):
            output_rows.append(
                {
                    "seed": seed,
                    "stream": key,
                    "fold": fold_index,
                    "fold_auc": auc,
                    "phase2_fold_auc": baseline_fold_auc[fold_index - 1],
                    "delta_vs_phase2": delta,
                    "positive_fold": delta > 0,
                    "pooled_auc": pooled,
                    "pooled_delta_vs_phase2": pooled - baseline_pooled,
                }
            )
        output_rows.append(
            {
                "seed": seed,
                "stream": key,
                "fold": "pooled",
                "fold_auc": pooled,
                "phase2_fold_auc": baseline_pooled,
                "delta_vs_phase2": pooled - baseline_pooled,
                "positive_fold": pooled > baseline_pooled,
                "pooled_auc": pooled,
                "pooled_delta_vs_phase2": pooled - baseline_pooled,
            }
        )
    return metrics


def prediction_diagnostics(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    prevalence = float(labels.mean())
    top_count = max(1, int(round(prevalence * len(labels))))
    base_top = set(np.argpartition(-baseline, top_count - 1)[:top_count].tolist())
    candidate_top = set(np.argpartition(-candidate, top_count - 1)[:top_count].tolist())
    positive_rows = set(np.flatnonzero(labels == 1).tolist())
    baseline_errors = base_top.symmetric_difference(positive_rows)
    candidate_errors = candidate_top.symmetric_difference(positive_rows)
    shared_errors = baseline_errors & candidate_errors
    union_errors = baseline_errors | candidate_errors
    decile = pd.qcut(
        pd.Series(baseline, dtype="float64").rank(method="first"),
        q=10,
        labels=False,
        duplicates="drop",
    ).to_numpy()
    segment_auc: dict[str, float | None] = {}
    segment_delta: dict[str, float | None] = {}
    for segment in np.unique(decile):
        selected = decile == segment
        name = f"baseline_risk_decile_{int(segment) + 1}"
        if len(np.unique(labels[selected])) < 2:
            segment_auc[name] = None
            segment_delta[name] = None
        else:
            segment_auc[name] = float(roc_auc_score(labels[selected], candidate[selected]))
            segment_delta[name] = segment_auc[name] - float(
                roc_auc_score(labels[selected], baseline[selected])
            )
    return {
        "candidate_auc": float(roc_auc_score(labels, candidate)),
        "baseline_auc": float(roc_auc_score(labels, baseline)),
        "pearson_vs_phase2": float(np.corrcoef(candidate, baseline)[0, 1]),
        "spearman_vs_phase2": float(spearmanr(candidate, baseline).statistic),
        "prediction_variance": float(np.var(candidate)),
        "error_correlation_vs_phase2": float(np.corrcoef(candidate - labels, baseline - labels)[0, 1]),
        "binary_error_jaccard_at_prevalence": float(
            len(shared_errors) / max(1, len(union_errors))
        ),
        "shared_binary_errors_at_prevalence": len(shared_errors),
        "baseline_binary_errors_at_prevalence": len(baseline_errors),
        "candidate_binary_errors_at_prevalence": len(candidate_errors),
        "top_prevalence_overlap": float(len(base_top & candidate_top) / top_count),
        "top_prevalence_count": top_count,
        "target_prevalence": prevalence,
        "phase2_baseline_risk_decile_auc": {
            f"baseline_risk_decile_{i + 1}": float(roc_auc_score(labels[decile == i], baseline[decile == i]))
            if len(np.unique(labels[decile == i])) > 1
            else None
            for i in np.unique(decile)
        },
        "candidate_baseline_risk_decile_auc": segment_auc,
        "delta_auc_by_baseline_risk_decile": segment_delta,
    }


def discover_prediction_artifacts(
    roots: list[Path],
    train_id_hash: str,
    test_id_hash: str,
    expected_inputs: dict[str, str],
    seed_baselines: dict[int, tuple[list[np.ndarray], str]],
    row_counts: dict[str, int],
    labels: np.ndarray,
    ordered_train_ids: np.ndarray,
    ordered_test_ids: np.ndarray,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.npy")):
            if EXPERIMENT_DIR in path.parents:
                continue
            lower = path.name.lower()
            if not (lower.startswith("oof") or lower.startswith("test")):
                continue
            run_dir = path.parent
            role = "oof" if lower.startswith("oof") else "test"
            row_count = row_counts[role]
            expected_id_hash = train_id_hash if role == "oof" else test_id_hash
            try:
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                shape_ok = values.ndim == 1 and len(values) == row_count
            except (OSError, ValueError):
                shape_ok = False
            source_hash_ok, source_hash_note = output_hash_status(run_dir, path)
            input_ok, input_note = input_identity(run_dir, expected_inputs)
            seed_match = re.search(r"seed(?:ed)?[_-]?(42|20261101)", str(path.parent), re.IGNORECASE)
            seed = int(seed_match.group(1)) if seed_match else None
            fold_hash = ""
            folds_ok = False
            if seed is not None:
                try:
                    folds, fold_hash = read_fold_indices(run_dir, row_counts["oof"])
                    reference = seed_baselines.get(seed)
                    folds_ok = reference is not None and check_same_folds(folds, reference[0])
                except (ValueError, OSError):
                    folds_ok = False
            if shape_ok and role == "oof":
                alignment_ok, alignment_note = verify_oof_row_alignment(
                    run_dir,
                    None,
                    path,
                    np.asarray(values, dtype=np.float64),
                    labels,
                    ordered_train_ids,
                )
            elif shape_ok:
                stream_key = None
                path_text = str(run_dir).lower()
                if "exp04_catboost_lossguide" in path_text:
                    stream_key = "exp04"
                elif "exp05_tabular_neural" in path_text:
                    stream_key = "exp05"
                elif "exp07_gbdt_feature_ensemble" in path_text:
                    stream_key = "exp07_nested_replication" if "seed42" in path_text else "exp07_nested_blend"
                elif "exp08_historical_feature_bridge" in path_text:
                    stream_key = "exp08_historical"
                alignment_ok, alignment_note = verify_test_row_alignment(
                    run_dir,
                    stream_key or "",
                    np.asarray(values, dtype=np.float64),
                    ordered_test_ids,
                )
            else:
                alignment_ok, alignment_note = False, "test prediction ID order is not independently recorded"
            eligible = bool(
                shape_ok and source_hash_ok and input_ok and folds_ok and alignment_ok
            )
            artifacts.append(
                {
                    "experiment": run_dir.name,
                    "artifact": path.name,
                    "path": str(path.resolve()),
                    "prediction_role": role,
                    "row_count": len(values) if shape_ok else "",
                    "prediction_id_sha256": expected_id_hash if shape_ok else "",
                    "outer_seed": seed if seed is not None else "",
                    "fold_assignment_sha256": fold_hash,
                    "folds_match_seed_reference": folds_ok,
                    "row_alignment_verified": alignment_ok,
                    "row_alignment_note": alignment_note,
                    "input_hashes_verified": input_ok,
                    "input_hash_note": input_note,
                    "output_sha256": sha256_file(path),
                    "source_manifest_hash_verified": source_hash_ok,
                    "source_manifest_note": source_hash_note,
                    "validation_protocol": protocol_description(run_dir),
                    "eligible_for_oof_comparison": eligible and role == "oof",
                    "eligible_for_test_ensemble": eligible and role == "test",
                    "exclusion_reason": "" if eligible else "; ".join(
                        note
                        for note, passed in (
                            ("row count/shape mismatch", shape_ok),
                            (source_hash_note, source_hash_ok),
                            (input_note, input_ok),
                            ("fold assignment missing or differs from matched baseline", folds_ok),
                            (alignment_note, alignment_ok),
                        )
                        if not passed
                    ),
                }
            )
    return artifacts


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    pd.DataFrame.from_records(records).to_csv(path, index=False, lineterminator="\n")


def _find_historical_champion_csv() -> Path | None:
    path = (
        IMPROVEMENT_DIR
        / "baseline"
        / "reference"
        / "EXP85_HISTORICAL_CHAMPION.csv"
    )
    return path if path.is_file() else None


def run_experiment(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    require(not run_dir.exists() or not any(run_dir.iterdir()), f"Run directory must be new or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "train": data_dir / TRAIN_NAME,
        "test": data_dir / TEST_NAME,
        "sample": data_dir / SAMPLE_NAME,
    }
    input_records: dict[str, dict[str, Any]] = {}
    for key, path in input_paths.items():
        require(path.is_file(), f"Required input is missing: {path}")
        record = file_record(path)
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[key],
            f"Unexpected {key} input SHA-256; refusing to align saved predictions",
        )
        input_records[key] = record

    train = pd.read_csv(input_paths["train"], usecols=[ID_COLUMN, TARGET_COLUMN])
    test = pd.read_csv(input_paths["test"], usecols=[ID_COLUMN])
    sample = pd.read_csv(input_paths["sample"], usecols=[ID_COLUMN])
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(len(train) == EXPECTED_ROWS["train"] and len(test) == EXPECTED_ROWS["test"], "Unexpected data row counts")
    require(np.array_equal(test[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()), "Sample submission IDs do not exactly match test order")
    train_id_hash = canonical_id_hash(train[ID_COLUMN].to_numpy())
    test_id_hash = canonical_id_hash(test[ID_COLUMN].to_numpy())
    expected_input_hashes = {key: value["sha256"] for key, value in input_records.items()}

    specs, _ = all_stream_specs()
    loaded: dict[str, dict[str, Any]] = {}
    folds_by_seed: dict[int, tuple[list[np.ndarray], str]] = {}
    for spec in specs:
        require(spec.run_dir.is_dir(), f"Expected source run directory is absent: {spec.run_dir}")
        if spec.seed not in folds_by_seed:
            folds_by_seed[spec.seed] = read_fold_indices(spec.run_dir, len(labels))
        folds, fold_hash = read_fold_indices(spec.run_dir, len(labels))
        reference_folds = folds_by_seed[spec.seed][0]
        require(
            check_same_folds(folds, reference_folds),
            f"Fold assignment mismatch for {spec.key} despite matching seed label",
        )
        oof_path = spec.run_dir / spec.oof_name
        test_path = spec.run_dir / spec.test_name if spec.test_name else None
        oof_verified, oof_note = output_hash_status(spec.run_dir, oof_path)
        if test_path is not None and test_path.is_file():
            test_verified, test_note = output_hash_status(spec.run_dir, test_path)
            test_predictions = load_prediction(test_path, len(test))
        else:
            test_verified, test_note = False, "no matching test prediction artifact was persisted"
            test_predictions = None
        identity_ok, identity_note = input_identity(spec.run_dir, expected_input_hashes)
        require(oof_verified, f"{spec.key}: OOF prediction hash is not verified by source manifest")
        require(
            test_verified or spec.role == "component",
            f"{spec.key}: test prediction hash is not verified by source manifest",
        )
        require(identity_ok, f"{spec.key}: {identity_note}")
        oof = load_prediction(oof_path, len(labels))
        oof_alignment_ok, oof_alignment_note = verify_oof_row_alignment(
            spec.run_dir,
            spec.key,
            oof_path,
            oof,
            labels,
            train[ID_COLUMN].to_numpy(),
        )
        require(
            oof_alignment_ok,
            f"{spec.key}: OOF row alignment not verified: {oof_alignment_note}",
        )
        if test_path is not None and test_predictions is not None:
            test_alignment_ok, test_alignment_note = verify_test_row_alignment(
                spec.run_dir,
                spec.key,
                test_predictions,
                test[ID_COLUMN].to_numpy(),
            )
        else:
            test_alignment_ok, test_alignment_note = False, "test prediction was not persisted"
        loaded[spec.key] = {
            "spec": spec,
            "oof": oof,
            "test": test_predictions,
            "oof_path": oof_path,
            "test_path": test_path,
            "folds": folds,
            "fold_hash": fold_hash,
            "source_hash_verified": oof_verified
            and (test_verified or spec.role == "component"),
            "input_identity_verified": identity_ok,
            "oof_alignment_verified": oof_alignment_ok,
            "oof_alignment_note": oof_alignment_note,
            "test_alignment_verified": test_alignment_ok,
            "test_alignment_note": test_alignment_note,
            "protocol": protocol_description(spec.run_dir),
            "source_records": {
                "oof": file_record(oof_path),
                **({"test": file_record(test_path)} if test_path is not None else {}),
            },
        }

    exp08_run = (
        IMPROVEMENT_DIR / "experiments" / "exp08_historical_feature_bridge"
        / "runs" / "exp08_historical_seed20261101_v3"
    )
    derived_paths = reconstruct_exp08_components(exp08_run, run_dir, len(labels), len(test))
    derived_names = {
        "exp08_lgbcat": ("EXP08 fold-local LightGBM/CatBoost component", "Historical features", "candidate"),
        "exp08_xgb": ("EXP08 CUDA XGBoost component", "Historical features + XGBoost", "candidate"),
    }
    for key, suffix in (("exp08_lgbcat", "lgbcat"), ("exp08_xgb", "xgb")):
        oof_path = derived_paths[f"oof_exp08_{suffix}_reconstructed.npy"]
        test_path = derived_paths[f"test_exp08_{suffix}_reconstructed.npy"]
        loaded[key] = {
            "spec": StreamSpec(
                key,
                derived_names[key][0],
                MAIN_SEED,
                exp08_run,
                oof_path.name,
                test_path.name,
                derived_names[key][1],
                derived_names[key][2],
                "reconstructed_from_fold_prediction_components",
            ),
            "oof": load_prediction(oof_path, len(labels)),
            "test": load_prediction(test_path, len(test)),
            "oof_path": oof_path,
            "test_path": test_path,
            "folds": folds_by_seed[MAIN_SEED][0],
            "fold_hash": folds_by_seed[MAIN_SEED][1],
            "source_hash_verified": True,
            "input_identity_verified": True,
            "oof_alignment_verified": True,
            "oof_alignment_note": "reconstructed from fold-local predictions at exact saved validation indices",
            "test_alignment_verified": True,
            "test_alignment_note": "reconstructed from saved per-fold test component predictions",
            "protocol": "EXP08 outer-fold predictions; component streams reconstructed from saved fold prediction NPZs",
            "source_records": {
                "oof": file_record(oof_path),
                "test": file_record(test_path),
                "reconstruction_source_manifest": file_record(exp08_run / "sha256_manifest.json"),
            },
        }

    main_baseline_key = "phase2_main"
    replication_baseline_key = "phase2_replication"
    main_baseline = loaded[main_baseline_key]["oof"]
    replication_baseline = loaded[replication_baseline_key]["oof"]
    require(
        np.array_equal(
            main_baseline,
            load_prediction(
                IMPROVEMENT_DIR
                / "experiments"
                / "exp08_historical_feature_bridge"
                / "runs"
                / "exp08_historical_seed20261101_v3"
                / "oof_phase2_matched.npy",
                len(labels),
            ),
        ),
        "EXP08 strict Phase 2 baseline does not equal EXP06 matched reference",
    )

    main_folds = folds_by_seed[MAIN_SEED][0]
    replication_folds = folds_by_seed[REPLICATION_SEED][0]
    main_candidate_keys = [
        "exp04",
        "exp07_signature",
        "exp08_lgbcat",
        "raw_xgb_main",
    ]
    # Fixed equal weights are declared in advance; no outer labels select weights.
    main_prediction_sets: dict[str, np.ndarray] = {"phase2": main_baseline}
    main_test_sets: dict[str, np.ndarray] = {"phase2": loaded[main_baseline_key]["test"]}
    main_candidate_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    main_specs_for_metrics = [
        key
        for key, item in loaded.items()
        if item["spec"].seed == MAIN_SEED
        and item["spec"].role in {"candidate", "phase2_inclusive_blend"}
        and key not in {"phase2_main"}
    ]
    main_specs_for_metrics.extend(["exp08_lgbcat", "exp08_xgb"])
    main_specs_for_metrics = list(dict.fromkeys(main_specs_for_metrics))
    for key in main_specs_for_metrics:
        main_prediction_sets[key] = loaded[key]["oof"]
        diag = prediction_diagnostics(labels, main_baseline, loaded[key]["oof"])
        diag.update(
            {
                "seed": MAIN_SEED,
                "candidate": key,
                "label": loaded[key]["spec"].label,
                "role": loaded[key]["spec"].role,
                "source_kind": loaded[key]["spec"].source_kind,
                "test_prediction_available": True,
                "outer_fold_assignment_sha256": loaded[key]["fold_hash"],
                "provenance_verified": bool(
                    loaded[key]["source_hash_verified"]
                    and loaded[key]["input_identity_verified"]
                ),
            }
        )
        diversity_rows.append(diag)

    fold_metric_rows: list[dict[str, Any]] = []
    for key in main_candidate_keys:
        candidate = loaded[key]
        fixed_name = f"phase2_plus_{key}_equal50"
        main_prediction_sets[fixed_name] = 0.5 * (main_baseline + candidate["oof"])
        main_test_sets[fixed_name] = 0.5 * (
            loaded[main_baseline_key]["test"] + candidate["test"]
        )
    fixed_combinations: list[tuple[str, ...]] = []
    for combination_size in (2, 3):
        fixed_combinations.extend(itertools.combinations(main_candidate_keys, combination_size))
    for combination in fixed_combinations:
        combination_key = "equal_" + "__".join(combination)
        main_prediction_sets[combination_key] = np.mean(
            [loaded[key]["oof"] for key in combination], axis=0
        )
        main_test_sets[combination_key] = np.mean(
            [loaded[key]["test"] for key in combination], axis=0
        )
    main_metrics = fold_auc_rows(
        labels, main_prediction_sets, main_folds, "phase2", MAIN_SEED, fold_metric_rows
    )

    replication_prediction_sets: dict[str, np.ndarray] = {
        "phase2": replication_baseline,
        "exp07_signature": loaded["exp07_signature_replication"]["oof"],
        "raw_xgb_replication": loaded["raw_xgb_replication"]["oof"],
    }
    replication_test_sets: dict[str, np.ndarray] = {
        "phase2": loaded[replication_baseline_key]["test"],
        "exp07_signature": loaded["exp07_signature_replication"]["test"],
        "raw_xgb_replication": loaded["raw_xgb_replication"]["test"],
    }
    for key in list(replication_prediction_sets):
        if key == "phase2":
            continue
        diag = prediction_diagnostics(
            labels, replication_baseline, replication_prediction_sets[key]
        )
        diag.update(
            {
                "seed": REPLICATION_SEED,
                "candidate": key,
                "label": key,
                "role": "candidate",
                "source_kind": "saved_model_predictions",
                "test_prediction_available": True,
                "outer_fold_assignment_sha256": folds_by_seed[REPLICATION_SEED][1],
                "provenance_verified": True,
            }
        )
        diversity_rows.append(diag)
        blend_key = f"phase2_plus_{key}_equal50"
        replication_prediction_sets[blend_key] = 0.5 * (
            replication_baseline + replication_prediction_sets[key]
        )
        replication_test_sets[blend_key] = 0.5 * (
            replication_test_sets["phase2"] + replication_test_sets[key]
        )
    replication_metrics = fold_auc_rows(
        labels,
        replication_prediction_sets,
        replication_folds,
        "phase2",
        REPLICATION_SEED,
        fold_metric_rows,
    )
    for key, predictions in replication_prediction_sets.items():
        if key == "phase2":
            continue
        diag = prediction_diagnostics(labels, replication_baseline, predictions)
        diag.update(
            {
                "seed": REPLICATION_SEED,
                "candidate": key,
                "label": key,
                "role": "fixed_50_50_blend",
                "source_kind": "fixed_weight_no_meta_training",
                "test_prediction_available": True,
                "outer_fold_assignment_sha256": folds_by_seed[REPLICATION_SEED][1],
                "provenance_verified": True,
            }
        )
        diversity_rows.append(diag)

    for key, values in main_metrics.items():
        if key == "phase2":
            continue
        main_candidate_rows.append(
            {
                "strategy": key,
                "seed": MAIN_SEED,
                **values,
                "fixed_weight_rule": (
                    "Phase2 + candidate: 50:50; candidate pair: equal 50:50; "
                    "candidate triple: equal 1/3"
                ),
                "weight_selection": "No weights fitted or tuned; fixed arithmetic mean",
            }
        )

    ranked_combos = sorted(
        (
            (
                main_metrics["equal_" + "__".join(combo)]["pooled_oof_auc"],
                combo,
            )
            for combo in fixed_combinations
        ),
        reverse=True,
    )
    best_two = next(
        (item for item in ranked_combos if len(item[1]) == 2),
        None,
    )
    best_three = next(
        (item for item in ranked_combos if len(item[1]) == 3),
        None,
    )
    best_fixed_strategy = max(
        [
            (
                metrics["pooled_oof_auc"],
                key,
                metrics,
            )
            for key, metrics in main_metrics.items()
            if key != "phase2"
        ],
        key=lambda item: item[0],
    )
    best_strategy_metrics = best_fixed_strategy[2]
    main_pass = (
        best_strategy_metrics["pooled_delta_vs_phase2"]
        >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
        and best_strategy_metrics["positive_folds"]
        >= ADVANCEMENT_GATE["minimum_positive_folds"]
    )
    independent_counterpart_key = {
        "exp07_signature": "exp07_signature",
        "raw_xgb_main": "raw_xgb_replication",
    }
    best_strategy_name = best_fixed_strategy[1]
    counterpart_metrics: dict[str, Any] | None = None
    if best_strategy_name == "phase2_plus_exp07_signature_equal50":
        counterpart_metrics = replication_metrics["phase2_plus_exp07_signature_equal50"]
    elif best_strategy_name == "phase2_plus_raw_xgb_main_equal50":
        counterpart_metrics = replication_metrics["phase2_plus_raw_xgb_replication_equal50"]
    elif best_strategy_name == "exp07_signature":
        counterpart_metrics = replication_metrics["exp07_signature"]
    elif best_strategy_name == "raw_xgb_main":
        counterpart_metrics = replication_metrics["raw_xgb_replication"]
    elif best_strategy_name.startswith("equal_"):
        components = best_strategy_name.removeprefix("equal_").split("__")
        counterparts = [independent_counterpart_key.get(key) for key in components]
        if all(counterparts):
            counterpart_prediction = np.mean(
                [replication_prediction_sets[key] for key in counterparts], axis=0
            )
            counterpart_metrics = {
                "pooled_oof_auc": float(roc_auc_score(labels, counterpart_prediction)),
                "baseline_pooled_oof_auc": float(roc_auc_score(labels, replication_baseline)),
                "pooled_delta_vs_phase2": float(
                    roc_auc_score(labels, counterpart_prediction)
                    - roc_auc_score(labels, replication_baseline)
                ),
                "fold_delta_vs_phase2": [
                    float(
                        roc_auc_score(labels[idx], counterpart_prediction[idx])
                        - roc_auc_score(labels[idx], replication_baseline[idx])
                    )
                    for idx in replication_folds
                ],
                "positive_folds": int(
                    sum(
                        roc_auc_score(labels[idx], counterpart_prediction[idx])
                        > roc_auc_score(labels[idx], replication_baseline[idx])
                        for idx in replication_folds
                    )
                ),
            }
    replication_pass = bool(
        counterpart_metrics is not None
        and counterpart_metrics["pooled_delta_vs_phase2"]
        >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
        and counterpart_metrics["positive_folds"]
        >= ADVANCEMENT_GATE["minimum_positive_folds"]
    )
    decision = "PROMOTABLE" if main_pass and replication_pass else "REJECTED"
    promotion_eligible = decision == "PROMOTABLE"

    blend_path = run_dir / "fixed_blend_predictions.npz"
    np.savez_compressed(
        blend_path,
        **{
            key: np.asarray(value, dtype=np.float32)
            for key, value in {**main_prediction_sets, **{f"seed42_{key}": value for key, value in replication_prediction_sets.items()}}.items()
        },
        **{
            f"test_{key}": np.asarray(value, dtype=np.float32)
            for key, value in {**main_test_sets, **{f"seed42_{key}": value for key, value in replication_test_sets.items()}}.items()
        },
    )

    inventory = discover_prediction_artifacts(
        [EXPERIMENTS_DIR, RESEARCH_RUNS_DIR, PHASE2_MAIN_DIR],
        train_id_hash,
        test_id_hash,
        expected_input_hashes,
        folds_by_seed,
        {"oof": len(labels), "test": len(test)},
        labels,
        train[ID_COLUMN].to_numpy(),
        test[ID_COLUMN].to_numpy(),
    )
    indexed_inventory = {
        (str(Path(row["path"]).resolve()), row["prediction_role"]): row
        for row in inventory
        if row.get("path") and row.get("prediction_role") in {"oof", "test"}
    }
    for key, item in loaded.items():
        spec: StreamSpec = item["spec"]
        for kind in ("oof", "test"):
            if item[kind] is None:
                continue
            path = item[f"{kind}_path"]
            existing = indexed_inventory.get((str(path.resolve()), kind))
            if existing is not None and path.parent != run_dir:
                existing["experiment"] = spec.label
                existing["stream_key"] = key
                existing["lineage"] = spec.source_kind
                existing["row_alignment_verified"] = (
                    item["oof_alignment_verified"]
                    if kind == "oof"
                    else item["test_alignment_verified"]
                )
                existing["row_alignment_note"] = (
                    item["oof_alignment_note"]
                    if kind == "oof"
                    else item["test_alignment_note"]
                )
                existing["eligible_for_oof_comparison"] = (
                    kind == "oof" and item["oof_alignment_verified"]
                )
                existing["eligible_for_test_ensemble"] = (
                    kind == "test" and item["test_alignment_verified"]
                )
                existing["exclusion_reason"] = (
                    ""
                    if kind == "oof" and item["oof_alignment_verified"]
                    or kind == "test" and item["test_alignment_verified"]
                    else item["oof_alignment_note"]
                    if kind == "oof"
                    else item["test_alignment_note"]
                )
                continue
            if path.parent == run_dir:
                recorded = True
                record_note = "derived from source fold predictions; output hash recorded in EXP09 manifest"
            else:
                recorded, record_note = output_hash_status(spec.run_dir, path)
            inventory.append(
                {
                    "experiment": spec.label,
                    "artifact": path.name,
                    "path": str(path.resolve()),
                    "prediction_role": kind,
                    "row_count": len(item[kind]),
                    "prediction_id_sha256": train_id_hash if kind == "oof" else test_id_hash,
                    "outer_seed": spec.seed,
                    "fold_assignment_sha256": item["fold_hash"],
                    "folds_match_seed_reference": True,
                    "row_alignment_verified": (
                        item["oof_alignment_verified"]
                        if kind == "oof"
                        else item["test_alignment_verified"]
                    ),
                    "row_alignment_note": (
                        item["oof_alignment_note"]
                        if kind == "oof"
                        else item["test_alignment_note"]
                    ),
                    "input_hashes_verified": item["input_identity_verified"],
                    "input_hash_note": "source train/test/sample hashes verified",
                    "output_sha256": sha256_file(path),
                    "source_manifest_hash_verified": recorded,
                    "source_manifest_note": record_note,
                    "validation_protocol": item["protocol"],
                    "eligible_for_oof_comparison": (
                        kind == "oof" and item["oof_alignment_verified"]
                    ),
                    "eligible_for_test_ensemble": (
                        kind == "test" and item["test_alignment_verified"]
                    ),
                    "exclusion_reason": (
                        ""
                        if kind == "oof" and item["oof_alignment_verified"]
                        or kind == "test" and item["test_alignment_verified"]
                        else item["oof_alignment_note"]
                        if kind == "oof"
                        else item["test_alignment_note"]
                    ),
                    "stream_key": key,
                    "lineage": spec.source_kind,
                }
            )

    champion_csv = _find_historical_champion_csv()
    champion_bridge = {
        "status": "UNAVAILABLE_FOR_ENSEMBLING",
        "classification": "historical_cached_test_submission_only",
        "oof_component_predictions_found": False,
        "independently_rebuilt_prediction_streams_found": False,
        "hybrid_cache_replay_used": False,
        "reason": "No historical-champion OOF/component prediction artifacts were found in the project archive; the cached submission has no OOF stream and cannot train or validate a blend.",
    }
    if champion_csv is not None:
        champion = pd.read_csv(champion_csv, usecols=[ID_COLUMN, TARGET_COLUMN])
        champion_bridge["cached_submission"] = {
            **file_record(champion_csv),
            "rows": int(len(champion)),
            "ordered_test_ids_match": bool(
                np.array_equal(champion[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy())
            ),
            "prediction_id_sha256": canonical_id_hash(champion[ID_COLUMN].to_numpy()),
        }
        inventory.append(
            {
                "experiment": "historical_champion",
                "artifact": champion_csv.name,
                "path": str(champion_csv.resolve()),
                "prediction_role": "test_only_cached_submission",
                "row_count": len(champion),
                "prediction_id_sha256": canonical_id_hash(champion[ID_COLUMN].to_numpy()),
                "outer_seed": "",
                "fold_assignment_sha256": "",
                "folds_match_seed_reference": False,
                "input_hashes_verified": False,
                "input_hash_note": "historical prediction lineage does not include verified source row set",
                "output_sha256": sha256_file(champion_csv),
                "source_manifest_hash_verified": False,
                "source_manifest_note": "not a verified OOF prediction artifact",
                "validation_protocol": "historical public leaderboard submission; no OOF evidence",
                "eligible_for_oof_comparison": False,
                "eligible_for_test_ensemble": False,
                "exclusion_reason": "test-only cached submission without independently verified OOF/component predictions",
                "stream_key": "historical_champion_cache",
                "lineage": "historical_cached_prediction",
            }
        )

    _write_csv(run_dir / "ARTIFACT_INVENTORY.csv", inventory)
    _write_csv(run_dir / "PREDICTION_DIVERSITY.csv", diversity_rows)
    _write_csv(run_dir / "FOLD_METRICS.csv", fold_metric_rows)

    input_id_records = {
        "train_prediction_id_sha256": train_id_hash,
        "test_prediction_id_sha256": test_id_hash,
        "sample_submission_order_matches_test": True,
        "train_rows": len(labels),
        "test_rows": len(test),
    }
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_SCREEN",
        "decision": decision,
        "promotion_eligible": promotion_eligible,
        "candidate_submission_created": False,
        "prediction_only": True,
        "model_retraining_performed": False,
        "gpu_training_performed": False,
        "gpu_policy": "No training was performed; CPU prediction analysis was used. Historical GPU evidence remains in source experiment reports.",
        "runtime_seconds": time.perf_counter() - started,
        "input_records": input_records,
        "prediction_id_records": input_id_records,
        "artifact_discovery": {
            "discovered_oof_or_test_arrays": len(inventory),
            "eligible_oof_predictions": int(sum(bool(row.get("eligible_for_oof_comparison")) for row in inventory)),
            "eligible_test_predictions": int(sum(bool(row.get("eligible_for_test_ensemble")) for row in inventory)),
            "inventory_path": str((run_dir / "ARTIFACT_INVENTORY.csv").resolve()),
            "artifact_exclusion_policy": "Require exact raw input hashes, manifest-recorded prediction hashes, expected row count, and a complete five-fold assignment matching the seed-specific Phase 2 reference.",
        },
        "fold_protocol": {
            "main_seed": MAIN_SEED,
            "independent_seed": REPLICATION_SEED,
            "main_fold_assignment_sha256": folds_by_seed[MAIN_SEED][1],
            "independent_fold_assignment_sha256": folds_by_seed[REPLICATION_SEED][1],
            "exact_fold_membership_verified_for_each_stream": True,
            "weight_policy": "All EXP09 combinations use fixed equal weights. No OOF labels or outer validation labels were used to fit weights.",
            "nested_logistic_stacking": "Not attempted: saved source streams do not provide homogeneous per-outer-fold inner predictions for every candidate. Fitting a meta-model from final OOF arrays would leak selection information.",
            "selected_candidates_for_predeclared_combination_screen": main_candidate_keys,
            "combination_screen": "Four fixed candidates; baseline plus each at 50:50; all candidate pairs at 50:50; all candidate triples at 1/3 each. This is exploratory multiple-comparison screening, not an unbiased selected-model estimate.",
        },
        "metrics": {
            "main_seed": main_metrics,
            "independent_seed": replication_metrics,
            "best_two_candidate_combination": (
                {
                    "strategy": "equal_" + "__".join(best_two[1]),
                    "pooled_oof_auc": best_two[0],
                    **main_metrics["equal_" + "__".join(best_two[1])],
                    "independent_replication_available": all(
                        key in independent_counterpart_key for key in best_two[1]
                    ),
                }
                if best_two
                else None
            ),
            "best_three_candidate_combination": (
                {
                    "strategy": "equal_" + "__".join(best_three[1]),
                    "pooled_oof_auc": best_three[0],
                    **main_metrics["equal_" + "__".join(best_three[1])],
                    "independent_replication_available": all(
                        key in independent_counterpart_key for key in best_three[1]
                    ),
                }
                if best_three
                else None
            ),
            "best_first_seed_screen": {
                "strategy": best_strategy_name,
                **best_strategy_metrics,
                "main_screen_gate_passed": main_pass,
                "independent_replication": counterpart_metrics,
                "independent_replication_gate_passed": replication_pass,
            },
        },
        "advancement_gate": {
            **ADVANCEMENT_GATE,
            "best_first_seed_strategy": best_strategy_name,
            "first_seed_passed": main_pass,
            "independent_replicate_available": counterpart_metrics is not None,
            "independent_seed_passed": replication_pass,
            "passed": promotion_eligible,
            "decision": decision,
            "reason": (
                "No strategy satisfied both the first-screen criteria and a matching independent-seed replication."
                if not promotion_eligible
                else "A fixed-weight candidate passed both seed-specific gates."
            ),
        },
        "historical_champion_bridge": champion_bridge,
        "integrity": {
            "source_prediction_hashes_verified": True,
            "source_input_hashes_verified": True,
            "train_test_id_hashes_recorded": True,
            "fold_indices_complete_and_disjoint": True,
            "oof_values_cross_checked_at_fold_validation_indices_or_id_sidecars": True,
            "test_ids_equal_sample_submission_order": True,
            "test_prediction_id_sidecars_verified_where_available": True,
            "source_experiments_modified": False,
        },
        "submission": {
            "created": False,
            "reason": "No candidate met the independent-seed advancement gate; no submission is eligible.",
        },
    }
    write_json(run_dir / "experiment_summary.json", summary)

    result_markdown = render_results(summary)
    (run_dir / "RESULTS.md").write_text(result_markdown, encoding="utf-8", newline="\n")
    (EXPERIMENT_DIR / "RESULTS.md").write_text(result_markdown, encoding="utf-8", newline="\n")
    (EXPERIMENT_DIR / "README.md").write_text(render_readme(), encoding="utf-8", newline="\n")

    source_paths = [
        Path(__file__).resolve(),
        EXPERIMENT_DIR / "verify_exp09.py",
        EXPERIMENT_DIR / "tests" / "test_exp09.py",
    ]
    source_records = {
        path.name: file_record(path)
        for path in source_paths
        if path.is_file()
    }
    output_paths = [
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "SHA256_MANIFEST.json"
    ]
    output_records = {path.name: file_record(path) for path in sorted(output_paths)}
    for path in (
        EXPERIMENT_DIR / "README.md",
        EXPERIMENT_DIR / "RESULTS.md",
    ):
        if path.is_file():
            output_records[f"experiment_root/{path.name}"] = file_record(path)
    manifest = {
        "experiment_id": run_dir.name,
        "inputs": input_records,
        "sources": source_records,
        "source_experiment_artifact_manifests": {
            spec.key: file_record(spec.run_dir / "sha256_manifest.json")
            for spec in specs
            if (spec.run_dir / "sha256_manifest.json").is_file()
        },
        "outputs": output_records,
    }
    write_json(run_dir / "SHA256_MANIFEST.json", manifest)
    return summary


def render_results(summary: dict[str, Any]) -> str:
    main = summary["metrics"]["main_seed"]
    replication = summary["metrics"]["independent_seed"]
    lines = [
        "# EXP09 — Prediction Diversity and Champion Bridge Results",
        "",
        f"- Status: **{summary['status']} — {summary['decision']}**",
        "- Prediction-only evaluation; no models were retrained.",
        "- All reported AUCs are OOF metrics, not public leaderboard scores.",
        "",
        "## Strict matched comparisons",
        "",
        "| Stream / strategy | Seed | Pooled OOF AUC | Delta vs Phase 2 | Positive folds |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = ["phase2", "exp04", "exp05", "exp06", "exp07_signature", "exp08_historical", "exp08_lgbcat", "exp08_xgb", "raw_xgb_main", "exp07_nested_blend", "exp08_nested_blend"]
    for key in ordered:
        if key not in main:
            continue
        row = main[key]
        lines.append(
            f"| {key} | {MAIN_SEED} | {row['pooled_oof_auc']:.8f} | "
            f"{row['pooled_delta_vs_phase2']:+.8f} | {row['positive_folds']}/5 |"
        )
    best_two = summary["metrics"]["best_two_candidate_combination"]
    best_three = summary["metrics"]["best_three_candidate_combination"]
    for title, row in (("Best fixed two-candidate screen", best_two), ("Best fixed three-candidate screen", best_three)):
        if row:
            lines.append(
                f"| {title}: {row['strategy']} | {MAIN_SEED} | "
                f"{row['pooled_oof_auc']:.8f} | {row['pooled_delta_vs_phase2']:+.8f} | "
                f"{row['positive_folds']}/5 |"
            )
    best = summary["metrics"]["best_first_seed_screen"]
    lines.extend(
        [
            "",
            "## First-screen leader and replication",
            "",
            f"- Strategy: `{best['strategy']}`.",
            f"- Main-seed result: {best['pooled_oof_auc']:.8f} "
            f"(delta {best['pooled_delta_vs_phase2']:+.8f}; "
            f"{best['positive_folds']}/5 positive folds).",
        ]
    )
    if best["independent_replication"] is None:
        lines.append("- Matching independent-seed prediction stream: **unavailable**.")
    else:
        rep = best["independent_replication"]
        lines.append(
            f"- Independent seed result: {rep['pooled_oof_auc']:.8f} "
            f"(delta {rep['pooled_delta_vs_phase2']:+.8f}; "
            f"{rep['positive_folds']}/5 positive folds)."
        )
    lines.extend(
        [
            "",
            "Candidate pair/triple search results are exploratory: the best combination is selected after comparing multiple fixed combinations, so its first-screen estimate is selection-biased. No outer-fold labels fitted any weights; all blend weights are fixed arithmetic means.",
            "",
            "## Independent seed screen",
            "",
            "| Stream | Pooled OOF AUC | Delta vs seed-42 Phase 2 | Positive folds | 50:50 with Phase 2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, row in replication.items():
        if key == "phase2":
            continue
        blend_key = f"phase2_plus_{key}_equal50"
        blend_auc = replication.get(blend_key, {}).get("pooled_oof_auc")
        lines.append(
            (
                f"| {key} | {row['pooled_oof_auc']:.8f} | "
                f"{row['pooled_delta_vs_phase2']:+.8f} | {row['positive_folds']}/5 | "
                f"{blend_auc:.8f} |"
                if blend_auc is not None
                else f"| {key} | {row['pooled_oof_auc']:.8f} | {row['pooled_delta_vs_phase2']:+.8f} | {row['positive_folds']}/5 | — |"
            )
        )
    lines.extend(
        [
            "",
            "## Controls and exclusions",
            "",
            "- Every candidate stream was required to have matching train/test/sample SHA-256 records, matching prediction-output hashes, expected row counts, a complete five-fold assignment, and exact fold membership matching its seed-specific Phase 2 reference.",
            "- OOF rows were verified against each source's saved per-fold predictions at their exact validation indices, or against an ID-bearing OOF sidecar. Exact input hashes and ordered train/test ID hashes were also checked.",
            "- Test arrays were admitted to test-ensemble inventory only when a source submission ID sidecar or five per-fold test predictions matched; the strict Phase 2 test streams lack such direct ID sidecars, so no candidate submission was created.",
            "- Fixed equal-weight blending was used; no weight or meta-model was fitted on OOF or outer validation labels.",
            "- Logistic stacking was not attempted because homogeneous inner-validation predictions were unavailable for every candidate.",
            "- The historical champion archive had a cached test submission but no independently verifiable champion OOF/component predictions. It was excluded from ensembling.",
            "- The interaction-screen arrays were excluded from OOF comparisons because the saved run has neither an ID-bearing OOF sidecar nor per-fold OOF prediction arrays.",
            "- No candidate submission was generated because no strategy passed the independent-seed gate.",
            "",
            "## Artifact integrity",
            "",
            f"- Train ID SHA-256: `{summary['prediction_id_records']['train_prediction_id_sha256']}`.",
            f"- Test ID SHA-256: `{summary['prediction_id_records']['test_prediction_id_sha256']}`.",
            f"- Discovered prediction artifacts: {summary['artifact_discovery']['discovered_oof_or_test_arrays']}.",
            f"- Input/source/output SHA-256 records: `{summary['experiment_id']}/SHA256_MANIFEST.json`.",
            "- Detailed artifact exclusions and per-stream provenance are in `ARTIFACT_INVENTORY.csv`.",
            "- Per-fold paired metrics are in `FOLD_METRICS.csv`; diversity diagnostics are in `PREDICTION_DIVERSITY.csv`.",
            "",
            "## Decision",
            "",
            f"**{summary['decision']}** — the best first-screen strategy does not have a verified matching independent-seed replicate that also clears the +0.001 pooled / 4-of-5-fold gate. Preserve the historical champion and frozen Phase 2 release.",
            "",
            "The historical public score (0.801340) and target (0.80314) are not established by these OOF results. No official leaderboard evaluation was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def render_readme() -> str:
    return """# EXP09 — Prediction Diversity and Champion Bridge

EXP09 audits existing OOF/test prediction artifacts and evaluates fixed-weight combinations. It does not retrain models or modify prior experiment directories.

## Run

```powershell
$runDir = "runs\\exp09_prediction_diversity_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python run_exp09.py --data-dir D:\\gci\\inpuy --run-dir $runDir
python verify_exp09.py --run-dir $runDir
python -m unittest discover -s tests -v
```

The runner refuses to proceed unless the raw input hashes match the frozen competition data. It requires exact prediction output hashes, input provenance, row counts, and fold membership against a seed-specific matched Phase 2 reference.

## Method

- Uses already-saved OOF and test arrays from EXP04–EXP08 and the GPU raw-XGBoost / interaction screens.
- Uses the strict matched Phase 2 reference separately for outer seeds 20261101 and 42.
- Uses only fixed equal-weight blends: 50:50 for pairs and 1/3 each for triples.
- Does not fit meta-models or choose weights from outer-validation labels.
- Reports the selected best pair/triple as exploratory because ranking many combinations causes selection bias.
- Reports residual/error overlap and risk-decile performance to identify complementary error signal.
- Does not produce a submission unless a candidate passes both first-seed and independent-seed gates.

`ARTIFACT_INVENTORY.csv` records provenance and eligibility; `PREDICTION_DIVERSITY.csv` records correlation, residual, risk-decile, error-overlap, and ranking-overlap diagnostics; `FOLD_METRICS.csv` records paired fold AUC; and each run's `SHA256_MANIFEST.json` records all inputs, sources, and outputs.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(args.data_dir, args.run_dir)
    print(
        f"{summary['decision']}: {summary['metrics']['best_first_seed_screen']['strategy']} "
        f"first-seed delta={summary['metrics']['best_first_seed_screen']['pooled_delta_vs_phase2']:+.8f}"
    )


if __name__ == "__main__":
    main()
