#!/usr/bin/env python3
"""Fresh, fixed-pair replication of the EXP09 prediction-pair screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
CHAMPION_DIR = REPOSITORY_ROOT / "champion"
IMPROVEMENT_DIR = CHAMPION_DIR / "EXP85_CHAMPION_IMPROVEMENT"
EXPERIMENTS_DIR = IMPROVEMENT_DIR / "experiments"
SOURCE_SNAPSHOT_DIR = EXPERIMENT_DIR / "source_snapshot"
EXP07_DIR = EXPERIMENTS_DIR / "exp07_gbdt_feature_ensemble"
EXP09_RUN_DIR = (
    EXPERIMENTS_DIR
    / "exp09_prediction_diversity"
    / "runs"
    / "exp09_prediction_diversity_v5"
)
EXP06_DIR = EXPERIMENTS_DIR / "exp06_ft_transformer"
FROZEN_DIR = CHAMPION_DIR / "EXP85_PHASE2_FROZEN_RELEASE"
DATA_DIR_DEFAULT = Path(r"D:\gci\inpuy")
OUTER_SEED = 20261102
FOLD_COUNT = 5
PAIR_COMPONENTS = ("exp07_signature", "exp08_lgbcat")
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
EXPECTED_EXP08_BASE_SOURCE_HASHES = {
    "run_exp08.py": "a76fd981594623a5d9940f6a86ca5147c88818af09145d3a34535af19e2af188",
    "historical_features.py": "081b373183f3860194acbd2fe8a72e903ca4899c84835440b121aa2299d7d18f",
}
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
ADVANCEMENT_GATE = {
    "minimum_pooled_auc_delta": 0.001,
    "minimum_positive_folds": 4,
    "folds": FOLD_COUNT,
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
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def canonical_id_hash(ids: np.ndarray) -> str:
    values = np.asarray(ids)
    require(values.ndim == 1, "ID hash requires a one-dimensional array")
    return hashlib.sha256(
        "\n".join(str(value) for value in values.tolist()).encode("utf-8")
    ).hexdigest()


def canonical_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_float64_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    require(array.ndim == 1 and np.isfinite(array).all(), "Invalid prediction hash input")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def import_source_snapshot(name: str, path: Path) -> Any:
    require(path.is_file(), f"Required EXP10 source snapshot is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_shared_paths(module: Any, source_dir: Path) -> None:
    """Keep model code in snapshots while wiring dependencies to the project."""
    module.EXPERIMENT_DIR = source_dir
    module.REPOSITORY_ROOT = REPOSITORY_ROOT
    module.CHAMPION_DIR = CHAMPION_DIR
    module.IMPROVEMENT_DIR = IMPROVEMENT_DIR
    module.FROZEN_DIR = FROZEN_DIR
    module.EXP06_DIR = EXP06_DIR
    module.EXP06_RUN_DIR = (
        EXP06_DIR / "runs" / "exp06_ft_transformer_seed20261101_v5"
    )
    module.EXP06_RUNNER = EXP06_DIR / "run_experiment.py"
    module.PHASE2_SOURCE = (
        FROZEN_DIR / "SOURCE" / "project" / "source" / "run_independent_baseline.py"
    )
    module.PHASE4_SOURCE = (
        FROZEN_DIR
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_04_final_validation_gate"
        / "run_final_gate.py"
    )
    module.HISTORICAL_FEATURE_SOURCE = (
        CHAMPION_DIR
        / "05_FEATURE_ENGINEERING"
        / "FEATURE_BUILDERS"
        / "build_hasan_600_features.py"
    )
    module.HISTORICAL_SOURCE = module.HISTORICAL_FEATURE_SOURCE
    module.HISTORICAL_SPEC = (
        CHAMPION_DIR
        / "05_FEATURE_ENGINEERING"
        / "FEATURES_669"
        / "FEATURES_669_MATHEMATICAL_SPECIFICATION.md"
    )
    module.HISTORICAL_PARITY_AUDIT = (
        CHAMPION_DIR
        / "05_FEATURE_ENGINEERING"
        / "FEATURE_VALIDATION"
        / "FEATURE_MATRIX_PARITY_AUDIT.md"
    )
    module.HISTORICAL_LINEAGE = (
        CHAMPION_DIR / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT" / "FEATURE_LINEAGE.md"
    )
    module.HISTORICAL_ARCHITECTURE = (
        CHAMPION_DIR
        / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT"
        / "HISTORICAL_ARCHITECTURE.md"
    )
    module.DEFAULT_DATA_DIR = DATA_DIR_DEFAULT


def load_experiment_runners() -> tuple[Any, Any]:
    require(SOURCE_SNAPSHOT_DIR.is_dir(), "EXP10 source snapshot directory is missing")
    sys.path.insert(0, str(SOURCE_SNAPSHOT_DIR))
    exp07 = import_source_snapshot(
        "exp10_exp07_snapshot", SOURCE_SNAPSHOT_DIR / "run_exp07.py"
    )
    configure_shared_paths(exp07, SOURCE_SNAPSHOT_DIR)
    exp07.EXP04_RUN_DIR = (
        IMPROVEMENT_DIR
        / "experiments"
        / "exp04_catboost_lossguide"
        / "runs"
        / "exp04_lossguide_strict_seed20261101"
    )
    exp07.EXP05_RUN_DIR = (
        IMPROVEMENT_DIR
        / "experiments"
        / "exp05_tabular_neural"
        / "runs"
        / "exp05_cuda_seed20261101_v1"
    )
    exp07.REFERENCE_DIR = FROZEN_DIR / "EVIDENCE" / "second_seed" / "main"
    exp07.SEED42_REFERENCE_DIR = (
        FROZEN_DIR / "EVIDENCE" / "original_runs" / "phase_02"
    )
    exp07.EXP07_DIR = EXP07_DIR

    exp08 = import_source_snapshot(
        "exp10_exp08_snapshot", SOURCE_SNAPSHOT_DIR / "run_exp08.py"
    )
    configure_shared_paths(exp08, SOURCE_SNAPSHOT_DIR)
    exp08.EXP07_DIR = EXP07_DIR
    exp08.EXP07_RUNNER = SOURCE_SNAPSHOT_DIR / "run_exp07.py"
    exp08.HISTORICAL_SOURCE = exp08.HISTORICAL_FEATURE_SOURCE
    exp08.HISTORICAL_SPEC = (
        CHAMPION_DIR
        / "05_FEATURE_ENGINEERING"
        / "FEATURES_669"
        / "FEATURES_669_MATHEMATICAL_SPECIFICATION.md"
    )
    exp08.HISTORICAL_PARITY_AUDIT = (
        CHAMPION_DIR
        / "05_FEATURE_ENGINEERING"
        / "FEATURE_VALIDATION"
        / "FEATURE_MATRIX_PARITY_AUDIT.md"
    )
    exp08.HISTORICAL_LINEAGE = (
        CHAMPION_DIR / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT" / "FEATURE_LINEAGE.md"
    )
    exp08.HISTORICAL_ARCHITECTURE = (
        CHAMPION_DIR
        / "EXP85_DEEP_RESEARCH_AND_IMPROVEMENT"
        / "HISTORICAL_ARCHITECTURE.md"
    )

    original_load_module = exp08.load_module

    def load_configured_helper(name: str, path: Path) -> Any:
        helper = original_load_module(name, path)
        if name == "exp08_exp07_helpers":
            configure_shared_paths(helper, SOURCE_SNAPSHOT_DIR)
            helper.EXP04_RUN_DIR = exp07.EXP04_RUN_DIR
            helper.EXP05_RUN_DIR = exp07.EXP05_RUN_DIR
            helper.REFERENCE_DIR = exp07.REFERENCE_DIR
            helper.SEED42_REFERENCE_DIR = exp07.SEED42_REFERENCE_DIR
        return helper

    exp08.load_module = load_configured_helper
    return exp07, exp08


def verify_inputs(data_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    inputs: dict[str, Any] = {}
    for name, path in paths.items():
        require(path.is_file(), f"Required {name} CSV is missing: {path}")
        record = file_record(path)
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[name],
            f"Raw {name} SHA-256 differs from the frozen source dataset",
        )
        inputs[name] = record

    train = pd.read_csv(paths["train"], usecols=[ID_COLUMN, TARGET_COLUMN])
    test = pd.read_csv(paths["test"], usecols=[ID_COLUMN])
    sample = pd.read_csv(paths["sample"], usecols=[ID_COLUMN])
    require(train[ID_COLUMN].is_unique, "Train IDs are not unique")
    require(test[ID_COLUMN].is_unique, "Test IDs are not unique")
    require(
        set(train[ID_COLUMN]).isdisjoint(set(test[ID_COLUMN])),
        "Train and test IDs overlap",
    )
    require(
        np.array_equal(test[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
        "Sample submission IDs/order differ from test.csv",
    )
    require(
        set(train[TARGET_COLUMN].unique()).issubset({0, 1}),
        "Training target is not binary",
    )
    return inputs, train, test


def load_probability(path: Path, expected_rows: int) -> np.ndarray:
    require(path.is_file(), f"Prediction array is missing: {path}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    require(
        values.shape == (expected_rows,)
        and np.isfinite(values).all()
        and np.all((values >= 0.0) & (values <= 1.0)),
        f"Invalid probability array: {path}",
    )
    return values


def fold_indices(run_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / f"fold_{fold}_indices.npz"
    require(path.is_file(), f"Fold index file is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["fit_indices"], dtype=np.int64),
            np.asarray(archive["valid_indices"], dtype=np.int64),
        )


def reconstruct_exp08_lgbcat(
    run_dir: Path, row_count: int, test_count: int
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(row_count, np.nan, dtype=np.float64)
    test_folds: list[np.ndarray] = []
    seen = np.zeros(row_count, dtype=bool)
    for fold in range(1, FOLD_COUNT + 1):
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        require(prediction_path.is_file(), f"EXP08 fold predictions missing: {fold}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            required = {"historical_lgb_cat_valid", "historical_test"}
            require(required.issubset(archive.files), f"EXP08 pair stream missing: fold {fold}")
            valid_prediction = np.asarray(
                archive["historical_lgb_cat_valid"], dtype=np.float64
            )
            test_prediction = np.asarray(archive["historical_test"], dtype=np.float64)
            if "historical_xgb_test" in archive.files:
                xgb_test = np.asarray(archive["historical_xgb_test"], dtype=np.float64)
                test_prediction = 2.0 * test_prediction - xgb_test
        _, valid = fold_indices(run_dir, fold)
        require(
            len(valid_prediction) == len(valid)
            and len(test_prediction) == test_count
            and not seen[valid].any(),
            f"EXP08 fold {fold} predictions or validation membership are invalid",
        )
        seen[valid] = True
        oof[valid] = valid_prediction
        test_folds.append(test_prediction)
    require(seen.all() and np.isfinite(oof).all(), "EXP08 OOF folds are incomplete")
    require(
        np.all((oof >= 0.0) & (oof <= 1.0)),
        "Reconstructed EXP08 OOF probabilities are outside [0,1]",
    )
    test_prediction = np.mean(test_folds, axis=0)
    require(
        np.isfinite(test_prediction).all()
        and np.all((test_prediction >= 0.0) & (test_prediction <= 1.0)),
        "Reconstructed EXP08 LightGBM/CatBoost test probabilities are invalid",
    )
    return oof, test_prediction


def verify_fold_oof_stream(
    run_dir: Path,
    stream_name: str,
    prediction_key: str,
    predictions: np.ndarray,
) -> None:
    seen = np.zeros(len(predictions), dtype=bool)
    for fold in range(1, FOLD_COUNT + 1):
        prediction_path = run_dir / f"fold_{fold}_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            require(
                prediction_key in archive.files,
                f"{stream_name} fold {fold} lacks indexed OOF values",
            )
            if "valid_indices" in archive.files:
                indices = np.asarray(archive["valid_indices"], dtype=np.int64)
            else:
                _, indices = fold_indices(run_dir, fold)
            fold_values = np.asarray(archive[prediction_key], dtype=np.float64)
        require(
            not seen[indices].any()
            and len(fold_values) == len(indices)
            and np.allclose(
                predictions[indices], fold_values, rtol=0.0, atol=1e-7
            ),
            f"{stream_name} OOF values differ from fold {fold} predictions",
        )
        seen[indices] = True
    require(seen.all(), f"{stream_name} OOF validation folds are incomplete")


def verify_component_outputs(run_dir: Path, names: list[str]) -> None:
    manifest_path = run_dir / "sha256_manifest.json"
    require(manifest_path.is_file(), f"Component manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in ("inputs", "references", "sources", "outputs"):
        records = manifest.get(group, {})
        for label, record in records.items():
            path = Path(record["path"])
            require(
                path.is_file()
                and path.stat().st_size == record["size_bytes"]
                and sha256_file(path) == record["sha256"],
                f"Completed component {group} artifact failed verification: {label}",
            )
    outputs = manifest.get("outputs", {})
    for name in names:
        record = outputs.get(name)
        path = run_dir / name
        require(
            record is not None and path.is_file(),
            f"Completed component output is missing from its manifest: {name}",
        )
        require(
            Path(record["path"]).resolve() == path.resolve()
            and path.stat().st_size == record["size_bytes"]
            and sha256_file(path) == record["sha256"],
            f"Completed component output failed manifest verification: {name}",
        )


def passes_gate(pooled_delta: float, positive_folds: int) -> bool:
    return (
        pooled_delta >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
        and positive_folds >= ADVANCEMENT_GATE["minimum_positive_folds"]
    )


def baseline_discrepancy(
    authoritative: np.ndarray, diagnostic: np.ndarray
) -> dict[str, Any]:
    authoritative = np.asarray(authoritative, dtype=np.float64)
    diagnostic = np.asarray(diagnostic, dtype=np.float64)
    require(
        authoritative.shape == diagnostic.shape
        and np.isfinite(authoritative).all()
        and np.isfinite(diagnostic).all(),
        "Phase 2 baseline diagnostics have incompatible shapes or non-finite values",
    )
    absolute_difference = np.abs(authoritative - diagnostic)
    return {
        "identical": bool(np.array_equal(authoritative, diagnostic)),
        "max_absolute_difference": float(np.max(absolute_difference)),
        "mean_absolute_difference": float(np.mean(absolute_difference)),
    }


def prediction_diagnostics(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    prevalence = float(np.mean(labels))
    top_count = max(1, int(round(prevalence * len(labels))))
    base_top = set(np.argpartition(-baseline, top_count - 1)[:top_count].tolist())
    candidate_top = set(
        np.argpartition(-candidate, top_count - 1)[:top_count].tolist()
    )
    positive_rows = set(np.flatnonzero(labels == 1).tolist())
    base_errors = base_top.symmetric_difference(positive_rows)
    candidate_errors = candidate_top.symmetric_difference(positive_rows)
    union = base_errors | candidate_errors
    return {
        "pearson_correlation": float(np.corrcoef(baseline, candidate)[0, 1]),
        "spearman_correlation": float(
            spearmanr(baseline, candidate).statistic
        ),
        "residual_correlation": float(
            np.corrcoef(baseline - labels, candidate - labels)[0, 1]
        ),
        "top_prevalence_overlap": float(
            len(base_top & candidate_top) / top_count
        ),
        "binary_error_jaccard_at_prevalence": float(
            len(base_errors & candidate_errors) / max(1, len(union))
        ),
        "baseline_binary_error_count": len(base_errors),
        "candidate_binary_error_count": len(candidate_errors),
        "shared_binary_error_count": len(base_errors & candidate_errors),
    }


def build_fold_metrics(
    labels: np.ndarray,
    baseline: np.ndarray,
    signature: np.ndarray,
    historical: np.ndarray,
    pair: np.ndarray,
    folds: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = {
        "phase2": baseline,
        "exp07_signature": signature,
        "exp08_lgbcat": historical,
        "fixed_equal_pair": pair,
    }
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    baseline_pooled_auc = float(roc_auc_score(labels, baseline))
    for name, values in predictions.items():
        aucs = [float(roc_auc_score(labels[idx], values[idx])) for idx in folds]
        pooled = float(roc_auc_score(labels, values))
        deltas = [
            auc - float(roc_auc_score(labels[idx], baseline[idx]))
            for idx, auc in zip(folds, aucs)
        ]
        metrics[name] = {
            "pooled_oof_auc": pooled,
            "pooled_delta_vs_phase2": pooled - baseline_pooled_auc,
            "fold_auc": aucs,
            "fold_delta_vs_phase2": deltas,
            "positive_folds": int(sum(delta > 0.0 for delta in deltas)),
        }
        for fold, (auc, delta) in enumerate(zip(aucs, deltas), start=1):
            rows.append(
                {
                    "seed": OUTER_SEED,
                    "stream": name,
                    "fold": fold,
                    "auc": auc,
                    "phase2_auc": float(
                        roc_auc_score(labels[folds[fold - 1]], baseline[folds[fold - 1]])
                    ),
                    "delta_vs_phase2": delta,
                }
            )
        rows.append(
            {
                "seed": OUTER_SEED,
                "stream": name,
                "fold": "pooled",
                "auc": pooled,
                "phase2_auc": baseline_pooled_auc,
                "delta_vs_phase2": pooled - baseline_pooled_auc,
            }
        )
    return rows, metrics


def run_experiment(
    data_dir: Path,
    run_dir: Path,
    reuse_components_from: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    require(
        not run_dir.exists() or not any(run_dir.iterdir()),
        f"EXP10 output directory must be new or empty: {run_dir}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    component_root = (
        reuse_components_from.resolve() / "components"
        if reuse_components_from is not None
        else run_dir / "components"
    )
    exp07_run = component_root / "exp07_signature_seed20261102"
    exp08_run = component_root / "exp08_historical_lgbcat_seed20261102"

    inputs, train, test = verify_inputs(data_dir)
    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    train_ids = train[ID_COLUMN].to_numpy()
    test_ids = test[ID_COLUMN].to_numpy()

    exp09_required = (
        "experiment_summary.json",
        "ARTIFACT_INVENTORY.csv",
        "PREDICTION_DIVERSITY.csv",
        "SHA256_MANIFEST.json",
    )
    for name in exp09_required:
        require((EXP09_RUN_DIR / name).is_file(), f"Required EXP09 evidence is missing: {name}")
    exp09_summary_path = EXP09_RUN_DIR / "experiment_summary.json"
    exp09_summary = json.loads(exp09_summary_path.read_text(encoding="utf-8"))
    exp09_pair = exp09_summary["metrics"]["best_two_candidate_combination"]
    require(
        exp09_pair["strategy"] == "equal_exp07_signature__exp08_lgbcat",
        "EXP09's recorded best pair differs from the declared EXP10 replication pair",
    )
    exp08_original_snapshot = SOURCE_SNAPSHOT_DIR / "original"
    for name, expected_hash in EXPECTED_EXP08_BASE_SOURCE_HASHES.items():
        original_path = exp08_original_snapshot / name
        require(
            original_path.is_file()
            and sha256_file(original_path) == expected_hash,
            f"Archived original EXP08 source hash mismatch: {name}",
        )

    exp07_original_hash = sha256_file(EXP07_DIR / "run_exp07.py")
    snapshot_records = {
        path.name: file_record(path)
        for path in sorted(SOURCE_SNAPSHOT_DIR.glob("*.py"))
    }
    snapshot_records.update(
        {
            str(path.relative_to(SOURCE_SNAPSHOT_DIR)): file_record(path)
            for path in sorted((SOURCE_SNAPSHOT_DIR / "tests").glob("*.py"))
        }
    )
    require(
        snapshot_records["historical_features.py"]["sha256"]
        == EXPECTED_EXP08_BASE_SOURCE_HASHES["historical_features.py"],
        "EXP08 feature-builder snapshot differs from the original historical source",
    )
    require(
        (SOURCE_SNAPSHOT_DIR / "README.md").is_file()
        and (SOURCE_SNAPSHOT_DIR / "FEATURE_INVENTORY.md").is_file()
        and (SOURCE_SNAPSHOT_DIR / "FEATURE_PARITY.md").is_file()
        and (SOURCE_SNAPSHOT_DIR / "tests" / "test_exp08_historical_features.py").is_file(),
        "EXP08 supporting source snapshots are incomplete",
    )

    exp07, exp08 = load_experiment_runners()
    if reuse_components_from is not None:
        require(
            (exp07_run / "experiment_summary.json").is_file()
            and (exp08_run / "experiment_summary.json").is_file(),
            "Cannot reuse components until both component runs have completed",
        )
        exp07_summary = json.loads(
            (exp07_run / "experiment_summary.json").read_text(encoding="utf-8")
        )
        exp08_summary = json.loads(
            (exp08_run / "experiment_summary.json").read_text(encoding="utf-8")
        )
    else:
        exp07_summary = exp07.run_experiment(data_dir, exp07_run, OUTER_SEED)
        exp08_summary = exp08.run_experiment(data_dir, exp08_run, OUTER_SEED)
    require(
        exp07_summary.get("status") == "COMPLETED_SCREEN",
        f"EXP07 component run failed: {exp07_summary.get('status')}",
    )
    require(
        exp08_summary.get("status") == "COMPLETED",
        f"EXP08 component run failed: {exp08_summary.get('status')}",
    )
    verify_component_outputs(
        exp07_run,
        [
            "experiment_summary.json",
            "oof_phase2_strict_matched.npy",
            "test_phase2_strict_matched.npy",
            "oof_signature_model.npy",
            "test_signature_model.npy",
            *[
                name
                for fold in range(1, FOLD_COUNT + 1)
                for name in (
                    f"fold_{fold}_indices.npz",
                    f"fold_{fold}_predictions.npz",
                )
            ],
        ],
    )
    verify_component_outputs(
        exp08_run,
        [
            "experiment_summary.json",
            "oof_phase2_matched.npy",
            "test_phase2_matched.npy",
            *[
                name
                for fold in range(1, FOLD_COUNT + 1)
                for name in (
                    f"fold_{fold}_indices.npz",
                    f"fold_{fold}_predictions.npz",
                )
            ],
        ],
    )

    baseline_07 = load_probability(
        exp07_run / "oof_phase2_strict_matched.npy", len(labels)
    )
    signature_oof = load_probability(
        exp07_run / "oof_signature_model.npy", len(labels)
    )
    baseline_test_07 = load_probability(
        exp07_run / "test_phase2_strict_matched.npy", len(test)
    )
    signature_test = load_probability(
        exp07_run / "test_signature_model.npy", len(test)
    )
    historical_oof, historical_test = reconstruct_exp08_lgbcat(
        exp08_run, len(labels), len(test)
    )
    baseline_08 = load_probability(exp08_run / "oof_phase2_matched.npy", len(labels))
    baseline_test_08 = load_probability(
        exp08_run / "test_phase2_matched.npy", len(test)
    )
    baseline_oof_diagnostic = baseline_discrepancy(baseline_07, baseline_08)
    baseline_test_diagnostic = baseline_discrepancy(
        baseline_test_07, baseline_test_08
    )
    verify_fold_oof_stream(
        exp07_run, "Phase 2 baseline", "phase2_reference", baseline_07
    )
    verify_fold_oof_stream(
        exp07_run, "EXP07 signature", "candidate_signature", signature_oof
    )
    verify_fold_oof_stream(
        exp08_run, "EXP08 local Phase 2 diagnostic", "phase2_valid", baseline_08
    )
    verify_fold_oof_stream(
        exp08_run, "EXP08 LightGBM/CatBoost", "historical_lgb_cat_valid", historical_oof
    )
    baseline = baseline_07
    baseline_test = baseline_test_07
    pair_oof = (signature_oof + historical_oof) / 2.0
    pair_test = (signature_test + historical_test) / 2.0
    require(
        np.all(np.isfinite(pair_oof))
        and np.all(np.isfinite(pair_test))
        and np.all((pair_oof >= 0.0) & (pair_oof <= 1.0))
        and np.all((pair_test >= 0.0) & (pair_test <= 1.0)),
        "Fixed pair probabilities are invalid",
    )

    folds: list[np.ndarray] = []
    split = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=OUTER_SEED)
    expected_folds = list(split.split(np.zeros(len(labels)), labels))
    for fold in range(1, FOLD_COUNT + 1):
        fit_07, valid_07 = fold_indices(exp07_run, fold)
        fit_08, valid_08 = fold_indices(exp08_run, fold)
        expected_fit, expected_valid = expected_folds[fold - 1]
        require(
            np.array_equal(valid_07, valid_08)
            and np.array_equal(fit_07, fit_08)
            and np.array_equal(valid_07, expected_valid)
            and np.array_equal(fit_07, expected_fit),
            f"Fold {fold} differs from fresh seed-{OUTER_SEED} partition",
        )
        require(
            not np.intersect1d(fit_07, valid_07).size,
            f"Fold {fold} training/validation rows overlap",
        )
        folds.append(valid_07)

    fold_membership = np.full(len(labels), -1, dtype=np.int8)
    fold_contracts: list[dict[str, Any]] = []
    for fold_number, valid_indices in enumerate(folds, start=1):
        require(
            np.all(fold_membership[valid_indices] == -1),
            f"Fold {fold_number} overlaps a prior validation fold",
        )
        fold_membership[valid_indices] = fold_number
        fit_indices, saved_valid_indices = fold_indices(exp07_run, fold_number)
        fold_contracts.append(
            {
                "fold": fold_number,
                "fit_indices_sha256": canonical_array_hash(fit_indices),
                "valid_indices_sha256": canonical_array_hash(saved_valid_indices),
                "phase2_valid_sha256": canonical_array_hash(
                    baseline[saved_valid_indices]
                ),
            }
        )
    require(np.all(fold_membership > 0), "Shared fold assignment does not cover every row")
    shared_baseline_path = run_dir / "shared_baseline_contract.npz"
    np.savez_compressed(
        shared_baseline_path,
        train_ids=train_ids,
        test_ids=test_ids,
        labels=labels,
        fold_membership=fold_membership,
        phase2_oof=baseline,
        phase2_test=baseline_test,
    )
    shared_baseline_contract = {
        "file": file_record(shared_baseline_path),
        "authority": "fresh EXP07 seed-20261102 matched Phase 2 stream",
        "comparison_streams": [
            "phase2",
            "exp07_signature",
            "exp08_lgbcat",
            "fixed_equal_pair",
        ],
        "arrays": {
            "train_ids_sha256": canonical_id_hash(train_ids),
            "test_ids_sha256": canonical_id_hash(test_ids),
            "labels_sha256": canonical_array_hash(labels),
            "fold_membership_sha256": canonical_array_hash(fold_membership),
            "phase2_oof_sha256": canonical_array_hash(baseline),
            "phase2_test_sha256": canonical_array_hash(baseline_test),
        },
        "folds": fold_contracts,
        "exp07_baseline_source": {
            "oof": file_record(exp07_run / "oof_phase2_strict_matched.npy"),
            "test": file_record(exp07_run / "test_phase2_strict_matched.npy"),
        },
        "exp08_local_baseline_diagnostic": {
            "oof": file_record(exp08_run / "oof_phase2_matched.npy"),
            "test": file_record(exp08_run / "test_phase2_matched.npy"),
            "oof_difference": baseline_oof_diagnostic,
            "test_difference": baseline_test_diagnostic,
            "used_for_pair_scoring": False,
        },
    }

    fold_rows, metrics = build_fold_metrics(
        labels, baseline, signature_oof, historical_oof, pair_oof, folds
    )
    baseline_auc = metrics["phase2"]["pooled_oof_auc"]
    pair_auc = metrics["fixed_equal_pair"]["pooled_oof_auc"]
    pair_delta = pair_auc - baseline_auc
    positive_folds = metrics["fixed_equal_pair"]["positive_folds"]
    gate_passed = passes_gate(pair_delta, positive_folds)
    no_submission_reason = (
        "Created only because the exact fixed pair passed the EXP10 advancement gate."
        if gate_passed
        else "Pair failed the predeclared advancement gate; no candidate submission was created."
    )

    predictions_path = run_dir / "replication_predictions.npz"
    np.savez_compressed(
        predictions_path,
        train_ids=train_ids,
        test_ids=test_ids,
        labels=labels,
        phase2_oof=baseline.astype(np.float64),
        exp08_local_phase2_oof=baseline_08.astype(np.float64),
        exp07_signature_oof=signature_oof.astype(np.float64),
        exp08_lgbcat_oof=historical_oof.astype(np.float64),
        pair_oof=pair_oof.astype(np.float64),
        phase2_test=baseline_test.astype(np.float64),
        exp08_local_phase2_test=baseline_test_08.astype(np.float64),
        exp07_signature_test=signature_test.astype(np.float64),
        exp08_lgbcat_test=historical_test.astype(np.float64),
        pair_test=pair_test.astype(np.float64),
    )
    pd.DataFrame(fold_rows).to_csv(
        run_dir / "FOLD_METRICS.csv", index=False, lineterminator="\n"
    )

    correlation = prediction_diagnostics(labels, signature_oof, historical_oof)
    pair_vs_baseline = prediction_diagnostics(labels, baseline, pair_oof)
    summary: dict[str, Any] = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "decision": "PROMOTABLE" if gate_passed else "REJECTED",
        "promotion_eligible": gate_passed,
        "candidate_submission_created": False,
        "candidate_submission_reason": no_submission_reason,
        "seed": OUTER_SEED,
        "folds": FOLD_COUNT,
        "pair": {
            "exact_exp09_strategy": "equal_exp07_signature__exp08_lgbcat",
            "components": list(PAIR_COMPONENTS),
            "blend_rule": "fixed 50:50 arithmetic mean; no weight fitting",
            "exp07_signature_config": json.loads(
                (exp07_run / "feature_config.json").read_text(encoding="utf-8")
            ),
            "exp07_model_configuration": exp07_summary["model"]["configuration"],
            "exp08_component_config": json.loads(
                (exp08_run / "feature_config.json").read_text(encoding="utf-8")
            ),
            "exp08_model_configuration": exp08_summary["models"],
            "exp08_selected_groups_by_fold": [
                {
                    "fold": fold["fold"],
                    "selected_groups": fold["selected_groups"],
                    "selected_feature_count": fold["selected_feature_count"],
                    "candidate_lgb_best_iteration": fold[
                        "candidate_lgb_best_iteration"
                    ],
                    "candidate_catboost_best_iteration": fold[
                        "candidate_catboost_best_iteration"
                    ],
                }
                for fold in exp08_summary["folds"]
            ],
            "exp08_component_lineage": (
                "historical_lgb_cat_valid from fold-local EXP08 candidate "
                "LightGBM/CatBoost predictions; XGBoost was removed from the "
                "three-model EXP08 candidate average using its saved fold prediction "
                "as 2*mean(LGB/Cat/XGB)-XGB."
            ),
            "shared_phase2_baseline": shared_baseline_contract,
        },
        "advancement_gate": {
            **ADVANCEMENT_GATE,
            "pooled_auc_delta": pair_delta,
            "positive_folds": positive_folds,
            "passed": gate_passed,
            "independent_replication": True,
            "fresh_models_and_phase2": True,
            "new_seed_fold_membership_verified": True,
        },
        "metrics": metrics,
        "prediction_diversity": {
            "between_pair_components": correlation,
            "pair_vs_phase2": pair_vs_baseline,
        },
        "integrity": {
            "raw_inputs_hash_verified": True,
            "ordered_train_ids_sha256": canonical_id_hash(train_ids),
            "ordered_test_ids_sha256": canonical_id_hash(test_ids),
            "sample_ids_exactly_match_test": True,
            "oof_fold_rows_verified_against_seed": True,
            "one_authoritative_phase2_baseline_used_for_all_reported_metrics": True,
            "component_local_exp08_baseline_used_for_pair_scoring": False,
            "phase2_oof_comparison_max_abs_difference": baseline_oof_diagnostic[
                "max_absolute_difference"
            ],
            "phase2_test_comparison_max_abs_difference": baseline_test_diagnostic[
                "max_absolute_difference"
            ],
            "shared_baseline_contract_sha256": shared_baseline_contract["file"][
                "sha256"
            ],
            "no_outer_validation_used_for_model_or_pair_weight_selection": True,
            "component_models_trained_with_original_feature_logic": True,
            "historical_champion_or_prior_runs_modified": False,
            "test_rows": len(test_ids),
            "test_pair_prediction_values_sha256": canonical_float64_hash(pair_test),
            "prediction_archive_sha256": sha256_file(predictions_path),
        },
        "discovery_reference": {
            "exp09_best_two_strategy": exp09_pair["strategy"],
            "exp09_main_seed": exp09_summary["fold_protocol"]["main_seed"],
            "exp09_pooled_oof_auc": exp09_pair["pooled_oof_auc"],
            "exp09_delta_vs_phase2": exp09_pair["pooled_delta_vs_phase2"],
            "exp09_positive_folds": exp09_pair["positive_folds"],
            "exp09_baseline_pooled_oof_auc": exp09_pair[
                "baseline_pooled_oof_auc"
            ],
        },
        "gpu_evidence": {
            "exp07": {
                "device": exp07_summary["model"]["catboost_device"],
                "preflight": exp07_summary["model"]["gpu_preflight"],
                "telemetry": exp07_summary["model"]["gpu_telemetry"],
            },
            "exp07_telemetry_path": str((exp07_run / "gpu_telemetry.json").resolve()),
            "exp08": exp08_summary.get("gpu"),
            "exp08_telemetry_path": str((exp08_run / "gpu_telemetry.csv").resolve()),
            "training_device_policy": (
                "GPU CatBoost/XGBoost required by source preflights; LightGBM remains "
                "CPU because the installed build is CPU-only. No silent fallback."
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "input_records": inputs,
        "source_lineage": {
            "exp10_runner_sha256": sha256_file(EXPERIMENT_DIR / "run_exp10.py"),
            "exp10_verifier_sha256": sha256_file(EXPERIMENT_DIR / "verify_exp10.py"),
            "exp10_unit_tests_sha256": {
                path.name: sha256_file(path)
                for path in sorted((EXPERIMENT_DIR / "tests").glob("*.py"))
            },
            "exp10_readme_sha256": sha256_file(EXPERIMENT_DIR / "README.md"),
            "exp07_original_runner_sha256": exp07_original_hash,
            "exp07_snapshot_sha256": snapshot_records["run_exp07.py"]["sha256"],
            "exp08_original_runner_sha256": EXPECTED_EXP08_BASE_SOURCE_HASHES[
                "run_exp08.py"
            ],
            "exp08_snapshot_sha256": snapshot_records["run_exp08.py"]["sha256"],
            "exp08_original_feature_builder_sha256": EXPECTED_EXP08_BASE_SOURCE_HASHES[
                "historical_features.py"
            ],
            "exp08_original_sources_archived_inside_exp10": True,
            "exp07_feature_inventory_sha256": file_record(
                SOURCE_SNAPSHOT_DIR / "FEATURE_INVENTORY.md"
            )["sha256"],
            "exp08_snapshot_feature_builder_sha256": snapshot_records[
                "historical_features.py"
            ]["sha256"],
            "seed_only_source_changes": (
                "Snapshots add explicit seed-20261102 fold support and record "
                "generated folds instead of comparing to a frozen old seed. "
                "Model/feature code and hyperparameters are unchanged. Component "
                "submission CSV writes are disabled; all ID/order checks remain."
            ),
            "exp09_discovery_run": str(
                EXP09_RUN_DIR.resolve()
            ),
        },
        "component_runs": {
            "component_source_run": str(
                (reuse_components_from.resolve() if reuse_components_from else run_dir.resolve())
            ),
            "exp07": {
                "path": str(exp07_run.resolve()),
                "status": exp07_summary["status"],
                "runtime_seconds": exp07_summary["runtime_seconds"],
                "training_seconds": exp07_summary["training_seconds"],
                "source_summary_sha256": sha256_file(
                    exp07_run / "experiment_summary.json"
                ),
            },
            "exp08": {
                "path": str(exp08_run.resolve()),
                "status": exp08_summary["status"],
                "runtime_seconds": exp08_summary["runtime_seconds"],
                "training_seconds": exp08_summary["training_seconds"],
                "source_summary_sha256": sha256_file(
                    exp08_run / "experiment_summary.json"
                ),
            },
        },
    }

    if gate_passed:
        sample = pd.read_csv(data_dir / "sample_submission.csv", usecols=[ID_COLUMN])
        require(np.array_equal(sample[ID_COLUMN].to_numpy(), test_ids), "Submission IDs changed")
        pd.DataFrame(
            {ID_COLUMN: test_ids, TARGET_COLUMN: pair_test}
        ).to_csv(run_dir / "candidate_submission.csv", index=False, lineterminator="\n")
        summary["candidate_submission_created"] = True
        summary["candidate_submission_reason"] = no_submission_reason
        summary["candidate_submission"] = file_record(
            run_dir / "candidate_submission.csv"
        )

    write_json(run_dir / "experiment_summary.json", summary)
    write_results(run_dir / "RESULTS.md", summary)
    source_records = {
        f"source_snapshot/{path.relative_to(SOURCE_SNAPSHOT_DIR)}": file_record(path)
        for path in sorted(SOURCE_SNAPSHOT_DIR.rglob("*"))
        if path.is_file()
    }
    source_lineage_records = {
        "exp10_runner": file_record(EXPERIMENT_DIR / "run_exp10.py"),
        "exp10_verifier": file_record(EXPERIMENT_DIR / "verify_exp10.py"),
        "exp10_readme": file_record(EXPERIMENT_DIR / "README.md"),
        **{
            f"exp10_test_{path.name}": file_record(path)
            for path in sorted((EXPERIMENT_DIR / "tests").glob("*.py"))
        },
        "exp07_original_runner": file_record(EXP07_DIR / "run_exp07.py"),
        "exp07_original_run_manifest": file_record(
            EXP07_DIR
            / "runs"
            / "exp07_signature_seed20261101_v1"
            / "sha256_manifest.json"
        ),
        "exp08_original_runner_snapshot": file_record(
            SOURCE_SNAPSHOT_DIR / "original" / "run_exp08.py"
        ),
        "exp08_original_feature_builder_snapshot": file_record(
            SOURCE_SNAPSHOT_DIR / "original" / "historical_features.py"
        ),
        "exp09_selected_pair_summary": file_record(exp09_summary_path),
        "exp09_inventory": file_record(EXP09_RUN_DIR / "ARTIFACT_INVENTORY.csv"),
        "exp09_diversity": file_record(EXP09_RUN_DIR / "PREDICTION_DIVERSITY.csv"),
        "exp09_sha256_manifest": file_record(EXP09_RUN_DIR / "SHA256_MANIFEST.json"),
        "phase2_raw_feature_builder": file_record(exp07.PHASE2_SOURCE),
        "phase2_fold_local_pipeline": file_record(exp07.PHASE4_SOURCE),
        "exp06_protocol_runner": file_record(exp07.EXP06_RUNNER),
    }
    output_records = {
        str(path.relative_to(run_dir)): file_record(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name != "SHA256_MANIFEST.json"
    }
    if component_root.resolve() != (run_dir / "components").resolve():
        for component_name, component_run in (
            ("exp07_signature", exp07_run),
            ("exp08_historical_lgbcat", exp08_run),
        ):
            for path in sorted(component_run.rglob("*")):
                if path.is_file():
                    label = (
                        f"reused_components/{component_name}/"
                        f"{path.relative_to(component_run)}"
                    )
                    output_records[label] = file_record(path)
    manifest = {
        "experiment_id": run_dir.name,
        "seed": OUTER_SEED,
        "inputs": inputs,
        "source_snapshots": source_records,
        "source_lineage": source_lineage_records,
        "replication_metadata": summary["source_lineage"],
        "outputs": output_records,
    }
    write_json(run_dir / "SHA256_MANIFEST.json", manifest)
    return summary


def write_results(path: Path, summary: dict[str, Any]) -> None:
    pair = summary["metrics"]["fixed_equal_pair"]
    baseline = summary["metrics"]["phase2"]
    signature = summary["metrics"]["exp07_signature"]
    historical = summary["metrics"]["exp08_lgbcat"]
    exp07_gpu = summary["gpu_evidence"]["exp07"]
    exp08_gpu = summary["gpu_evidence"]["exp08"]
    exp07_telemetry = exp07_gpu["telemetry"]
    exp08_telemetry = exp08_gpu["telemetry"]
    exp07_peak_utilization = exp07_telemetry.get(
        "max_utilization_percent",
        exp07_telemetry.get("maximum_utilization_percent"),
    )
    exp07_peak_memory = exp07_telemetry.get(
        "max_memory_used_mib",
        exp07_telemetry.get("maximum_memory_used_mib"),
    )
    exp08_peak_utilization = exp08_telemetry.get(
        "max_utilization_percent",
        exp08_telemetry.get("maximum_utilization_percent"),
    )
    exp08_peak_memory = exp08_telemetry.get(
        "max_memory_used_mib",
        exp08_telemetry.get("maximum_memory_used_mib"),
    )
    lines = [
        "# EXP10 — Independent Replication of the EXP09 Best Pair",
        "",
        f"- Decision: **{summary['decision']}**.",
        f"- Outer seed: `{summary['seed']}`; folds: `{summary['folds']}`.",
        "- Prediction streams were trained fresh; EXP09 OOF arrays were not reused.",
        "- All AUC values below are OOF metrics, not public leaderboard scores.",
        "",
        "## Exact predeclared pair",
        "",
        "- EXP09 strategy: `equal_exp07_signature__exp08_lgbcat`.",
        "- Components: EXP07 application-signature model and EXP08 fold-local LightGBM/CatBoost component.",
        "- Pair rule: fixed arithmetic mean, 50:50; no weights were fit or tuned.",
        f"- EXP09 discovery reference (seed {summary['discovery_reference']['exp09_main_seed']}): pooled AUC {summary['discovery_reference']['exp09_pooled_oof_auc']:.8f}, delta {summary['discovery_reference']['exp09_delta_vs_phase2']:+.8f}, positive folds {summary['discovery_reference']['exp09_positive_folds']}/5.",
        "",
        "| Stream | Pooled OOF AUC | Delta vs fresh Phase 2 | Positive folds |",
        "|---|---:|---:|---:|",
        f"| Fresh matched Phase 2 | {baseline['pooled_oof_auc']:.8f} | +0.00000000 | 0/5 |",
        f"| EXP07 signature | {signature['pooled_oof_auc']:.8f} | {signature['pooled_delta_vs_phase2']:+.8f} | {signature['positive_folds']}/5 |",
        f"| EXP08 LGB/Cat | {historical['pooled_oof_auc']:.8f} | {historical['pooled_delta_vs_phase2']:+.8f} | {historical['positive_folds']}/5 |",
        f"| Fixed pair | {pair['pooled_oof_auc']:.8f} | {pair['pooled_delta_vs_phase2']:+.8f} | {pair['positive_folds']}/5 |",
        "",
        "### Paired fold results",
        "",
        "| Fold | Phase 2 AUC | Pair AUC | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for fold, (base_auc, pair_auc, delta) in enumerate(
        zip(
            baseline["fold_auc"],
            pair["fold_auc"],
            pair["fold_delta_vs_phase2"],
        ),
        start=1,
    ):
        lines.append(
            f"| {fold} | {base_auc:.8f} | {pair_auc:.8f} | {delta:+.8f} |"
        )
    diversity = summary["prediction_diversity"]
    components = diversity["between_pair_components"]
    lines.extend(
        [
            "",
            "## Complementarity",
            "",
            f"- Component Pearson correlation: {components['pearson_correlation']:.6f}.",
            f"- Component Spearman correlation: {components['spearman_correlation']:.6f}.",
            f"- Component residual correlation: {components['residual_correlation']:.6f}.",
            f"- Component top-prevalence overlap: {components['top_prevalence_overlap']:.6f}.",
            f"- Component binary-error Jaccard: {components['binary_error_jaccard_at_prevalence']:.6f}.",
            "",
            "## GPU and runtime evidence",
            "",
            f"- EXP07: {exp07_gpu['device']}; peak utilization {exp07_peak_utilization}%; peak VRAM {exp07_peak_memory} MiB; training {summary['component_runs']['exp07']['training_seconds']:.1f}s.",
            f"- EXP08: {summary['component_runs']['exp08']['status']}; {exp08_gpu['preflight']['name']}; peak utilization {exp08_peak_utilization}%; peak VRAM {exp08_peak_memory} MiB; training {summary['component_runs']['exp08']['training_seconds']:.1f}s.",
            "- LightGBM ran on CPU because the installed build is CPU-only; CatBoost GPU training was verified. CUDA XGBoost preflight reported no CPU fallback.",
            "",
            "## Integrity and runtime",
            "",
            "- Every reported comparison uses one authoritative Phase 2 OOF/test stream from the fresh EXP07 seed-20261102 run, saved with IDs, labels, and fold membership in `shared_baseline_contract.npz`.",
            f"- EXP08 independently produced a different local Phase 2 diagnostic stream (max OOF difference {summary['integrity']['phase2_oof_comparison_max_abs_difference']:.6f}; max test difference {summary['integrity']['phase2_test_comparison_max_abs_difference']:.6f}); it was not used for pair scoring.",
            "- Fold membership was independently reconstructed from StratifiedKFold(seed=20261102), and each component's saved fit/validation indices were compared exactly.",
            "- OOF rows and test IDs were aligned to the hashed raw train/test inputs. See `SHA256_MANIFEST.json`.",
            f"- Test prediction rows: {summary['integrity']['test_rows']}; ordered test-ID SHA-256: `{summary['integrity']['ordered_test_ids_sha256']}`.",
            f"- Test pair prediction values SHA-256: `{summary['integrity']['test_pair_prediction_values_sha256']}`; compressed prediction archive SHA-256: `{summary['integrity']['prediction_archive_sha256']}`.",
            "- GPU-supported model training was required to pass the source CUDA preflights; LightGBM used CPU because the installed build is CPU-only.",
            "- The only source snapshot changes add this new seed/fold evidence path and suppress component submission files; model configurations and feature logic are preserved.",
            "",
            "## Advancement decision",
            "",
            f"- Required pooled improvement: +{ADVANCEMENT_GATE['minimum_pooled_auc_delta']:.3f}.",
            f"- Observed pooled improvement: {pair['pooled_delta_vs_phase2']:+.8f}.",
            f"- Required positive folds: {ADVANCEMENT_GATE['minimum_positive_folds']}/5; observed {pair['positive_folds']}/5.",
            f"- Candidate submission created: `{summary['candidate_submission_created']}`.",
            f"- Decision: **{summary['decision']}**.",
            "",
            "EXP10 does not establish an official public leaderboard score or prove the 0.80314 target.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=EXPERIMENT_DIR / "runs" / "exp10_pair_seed20261102",
    )
    parser.add_argument(
        "--reuse-components-from",
        type=Path,
        help="Finalize into a new run directory using components from a completed run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        args.data_dir, args.run_dir, args.reuse_components_from
    )
    pair = summary["metrics"]["fixed_equal_pair"]
    print(
        f"{summary['decision']}: pair delta={pair['pooled_delta_vs_phase2']:+.8f}; "
        f"positive folds={pair['positive_folds']}/5"
    )


if __name__ == "__main__":
    main()
