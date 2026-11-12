from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
FROZEN_ROOT = PROJECT_ROOT / "EXP85_PHASE2_FROZEN_RELEASE"
BASELINE_SOURCE = (
    FROZEN_ROOT
    / "SOURCE"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE2_REFERENCE = FROZEN_ROOT / "EVIDENCE" / "original_runs" / "phase_02"
EXPECTED_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
    "phase2_candidate": "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa",
}
MODEL_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda:0",
    "n_estimators": 1400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 20.0,
    "reg_lambda": 20.0,
    "reg_alpha": 0.1,
    "max_bin": 256,
    "early_stopping_rounds": 100,
    "n_jobs": 4,
}
CATEGORICAL_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_group_context_experiment as runtime_helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_builder():
    if not BASELINE_SOURCE.is_file():
        raise FileNotFoundError(f"Frozen rowwise feature source is missing: {BASELINE_SOURCE}")
    spec = importlib.util.spec_from_file_location(
        "frozen_phase2_rowwise_feature_builder", BASELINE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen feature source: {BASELINE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_fold_matrices(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, dict[str, int]]:
    if "TARGET" in train_features or "SK_ID_CURR" in train_features:
        raise ValueError("Model features must exclude target and applicant ID")
    if "TARGET" in test_features or "SK_ID_CURR" in test_features:
        raise ValueError("Test features must exclude target and applicant ID")
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train/test feature schemas differ")
    missing_cats = set(CATEGORICAL_COLUMNS) - set(train_features.columns)
    if missing_cats:
        raise ValueError(f"Missing categorical features: {sorted(missing_cats)}")

    numeric_columns = [
        column for column in train_features.columns
        if column not in CATEGORICAL_COLUMNS
    ]
    numeric_train = train_features.loc[:, numeric_columns].to_numpy(
        dtype=np.float32, copy=False
    )
    numeric_test = test_features.loc[:, numeric_columns].to_numpy(
        dtype=np.float32, copy=False
    )
    if np.isinf(numeric_train).any() or np.isinf(numeric_test).any():
        raise ValueError("Numeric model features contain infinity")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    fit_categories = train_features.iloc[fit_indices].loc[
        :, list(CATEGORICAL_COLUMNS)
    ].astype("string").fillna("MISSING")
    encoder.fit(fit_categories)

    def make_matrix(numeric: np.ndarray, encoded) -> csr_matrix:
        matrix = hstack(
            [csr_matrix(numeric, dtype=np.float32), encoded],
            format="csr",
            dtype=np.float32,
        )
        matrix.sort_indices()
        return matrix

    fit_cat = encoder.transform(fit_categories)
    valid_categories = train_features.iloc[valid_indices].loc[
        :, list(CATEGORICAL_COLUMNS)
    ].astype("string").fillna("MISSING")
    valid_cat = encoder.transform(valid_categories)
    test_categories = test_features.loc[:, list(CATEGORICAL_COLUMNS)].astype(
        "string"
    ).fillna("MISSING")
    test_cat = encoder.transform(test_categories)

    matrices = (
        make_matrix(numeric_train[fit_indices], fit_cat),
        make_matrix(numeric_train[valid_indices], valid_cat),
        make_matrix(numeric_test, test_cat),
    )
    if len({matrix.shape[1] for matrix in matrices}) != 1:
        raise ValueError("Fold matrix feature counts differ")
    diagnostics = {
        "numeric_feature_count": len(numeric_columns),
        "onehot_feature_count": int(sum(len(levels) for levels in encoder.categories_)),
        "fit_nonzero": int(matrices[0].nnz),
        "valid_nonzero": int(matrices[1].nnz),
        "test_nonzero": int(matrices[2].nnz),
        "matrix_feature_count": int(matrices[0].shape[1]),
    }
    return (*matrices, diagnostics)


def windows_memory_snapshot() -> dict[str, int | str]:
    if sys.platform != "win32":
        return {"available": "false", "reason": "Windows process API unavailable"}

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not success:
        raise ctypes.WinError()
    return {
        "available": "true",
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def load_reference_fold(fold_number: int) -> tuple[np.ndarray, np.ndarray]:
    path = PHASE2_REFERENCE / f"fold_{fold_number}_indices.npz"
    with np.load(path, allow_pickle=False) as data:
        return data["fit_indices"], data["valid_indices"]


def rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)


def run(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be new or empty: {run_dir}")
    input_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} CSV is missing: {path}")
        observed = sha256_file(path)
        if observed != EXPECTED_SHA256[name]:
            raise ValueError(f"{name} input SHA-256 mismatch: {observed}")
    frozen_candidate = (
        FROZEN_ROOT / "CANDIDATE" / "submission_key_target_encoding.csv"
    )
    if sha256_file(frozen_candidate) != EXPECTED_SHA256["phase2_candidate"]:
        raise ValueError("Frozen Phase 2 candidate hash changed")
    if not (PHASE2_REFERENCE / "oof_50_50_blend.npy").is_file():
        raise FileNotFoundError("Frozen Phase 2 OOF reference is missing")

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    module = load_feature_builder()
    train_raw = pd.read_csv(input_paths["train"])
    test_raw = pd.read_csv(input_paths["test"])
    sample = pd.read_csv(input_paths["sample"])
    if list(sample.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"Unexpected sample schema: {list(sample.columns)}")
    if not sample["SK_ID_CURR"].equals(test_raw["SK_ID_CURR"]):
        raise ValueError("Sample and raw test IDs/order differ")
    if "TARGET" not in train_raw or "TARGET" in test_raw:
        raise ValueError("Raw train/test target columns are invalid")
    labels = train_raw["TARGET"].to_numpy(dtype=np.int8)
    train_ids = train_raw["SK_ID_CURR"].to_numpy()
    test_ids = test_raw["SK_ID_CURR"].to_numpy()
    train_features = module.build_rowwise_features(
        train_raw.drop(columns=["TARGET"])
    )
    test_features = module.build_rowwise_features(test_raw)
    if "TARGET" in train_features or "SK_ID_CURR" in train_features:
        raise ValueError("Target/ID leaked into built training features")
    if "TARGET" in test_features or "SK_ID_CURR" in test_features:
        raise ValueError("Target/ID leaked into built test features")

    reference_oof_path = PHASE2_REFERENCE / "oof_50_50_blend.npy"
    phase2_oof = np.load(reference_oof_path, allow_pickle=False).astype(
        np.float64
    )
    if len(phase2_oof) != len(labels):
        raise ValueError("Frozen Phase 2 OOF vector does not align with raw training")

    monitor_path = run_dir.parent / f".{run_dir.name}.gpu_samples.tmp"
    log_path = run_dir.parent / f".{run_dir.name}.training.log.tmp"
    monitor = runtime_helpers.GpuMonitor(monitor_path)
    tee = runtime_helpers.Tee(sys.stdout, log_path)
    original_stdout = sys.stdout
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    test_fold_predictions: list[np.ndarray] = []
    fold_records = []
    max_matrix_features = 0
    process_memory_before = windows_memory_snapshot()
    sys.stdout = tee
    monitor.start()
    try:
        for fold_number in range(1, 6):
            fold_started = time.perf_counter()
            fit_indices, valid_indices = load_reference_fold(fold_number)
            fit_matrix, valid_matrix, test_matrix, matrix_info = build_fold_matrices(
                train_features,
                test_features,
                fit_indices,
                valid_indices,
            )
            max_matrix_features = max(
                max_matrix_features, matrix_info["matrix_feature_count"]
            )
            model = xgb.XGBClassifier(
                **MODEL_CONFIG,
                random_state=42 + fold_number,
                verbosity=0,
            )
            model.fit(
                fit_matrix,
                labels[fit_indices],
                eval_set=[(valid_matrix, labels[valid_indices])],
                verbose=100,
            )
            booster = model.get_booster()
            booster_config = json.loads(booster.save_config())
            actual_device = booster_config["learner"]["generic_param"]["device"]
            if not str(actual_device).startswith("cuda"):
                raise RuntimeError(
                    f"XGBoost device mismatch: requested CUDA, got {actual_device}"
                )
            best_iteration = int(model.best_iteration)
            prediction_range = (0, best_iteration + 1)
            valid_prediction = booster.predict(
                xgb.DMatrix(valid_matrix),
                iteration_range=prediction_range,
            ).astype(np.float32)
            test_prediction = booster.predict(
                xgb.DMatrix(test_matrix),
                iteration_range=prediction_range,
            ).astype(np.float32)
            if (
                not np.isfinite(valid_prediction).all()
                or not np.isfinite(test_prediction).all()
                or np.any((valid_prediction < 0) | (valid_prediction > 1))
                or np.any((test_prediction < 0) | (test_prediction > 1))
            ):
                raise ValueError(f"Fold {fold_number} emitted invalid probabilities")
            oof[valid_indices] = valid_prediction
            test_fold_predictions.append(test_prediction)
            fold_auc = float(
                roc_auc_score(labels[valid_indices], valid_prediction)
            )
            fold_records.append(
                {
                    "fold": fold_number,
                    "fit_rows": int(len(fit_indices)),
                    "valid_rows": int(len(valid_indices)),
                    "best_iteration": best_iteration,
                    "auc": fold_auc,
                    "device": actual_device,
                    "seconds": time.perf_counter() - fold_started,
                    **matrix_info,
                }
            )
            np.savez_compressed(
                run_dir / f"fold_{fold_number}_indices.npz",
                fit_indices=fit_indices.astype(np.int32),
                valid_indices=valid_indices.astype(np.int32),
            )
            print(
                f"[xgboost] fold {fold_number}/5 device={actual_device} "
                f"auc={fold_auc:.8f} best_iteration={best_iteration} "
                f"seconds={fold_records[-1]['seconds']:.1f}",
                flush=True,
            )
            del fit_matrix, valid_matrix, test_matrix, model, booster
    except Exception as error:
        (run_dir / "failure_report.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "gpu_samples": monitor.sample_count,
                    "gpu_monitor_errors": monitor.errors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    finally:
        monitor.stop()
        sys.stdout = original_stdout
        tee.close()
        if monitor_path.is_file():
            monitor_path.replace(run_dir / "gpu_samples.csv")
        if log_path.is_file():
            log_path.replace(run_dir / "training.log")

    if monitor.errors or monitor.sample_count < 2:
        raise RuntimeError(
            f"GPU evidence incomplete: samples={monitor.sample_count}, "
            f"errors={monitor.errors}"
        )
    if np.isnan(oof).any() or len(test_fold_predictions) != 5:
        raise ValueError("OOF coverage or test fold prediction count is incomplete")

    xgb_test = np.mean(
        np.stack(test_fold_predictions), axis=0, dtype=np.float64
    ).astype(np.float32)
    candidate_path = run_dir / "submission_raw_xgboost.csv"
    candidate = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": xgb_test})
    if (
        len(candidate) != 61500
        or candidate["SK_ID_CURR"].duplicated().any()
        or not candidate["SK_ID_CURR"].equals(test_raw["SK_ID_CURR"])
        or not candidate["SK_ID_CURR"].equals(sample["SK_ID_CURR"])
    ):
        raise ValueError("Raw XGBoost candidate IDs/order/row count are invalid")
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    phase2_auc = float(roc_auc_score(labels, phase2_oof))
    xgb_auc = float(roc_auc_score(labels, oof))
    rank_blend = (
        0.80 * rank_percentile(phase2_oof)
        + 0.20 * rank_percentile(oof)
    )
    blend_auc = float(roc_auc_score(labels, rank_blend))
    per_fold = []
    for fold_number, fold_record in enumerate(fold_records, 1):
        _, valid_indices = load_reference_fold(fold_number)
        with np.load(run_dir / f"fold_{fold_number}_indices.npz") as saved:
            if not np.array_equal(saved["valid_indices"], valid_indices):
                raise ValueError(f"Saved fold {fold_number} differs from reference")
        per_fold.append(
            {
                "fold": fold_number,
                "phase2_auc": float(
                    roc_auc_score(labels[valid_indices], phase2_oof[valid_indices])
                ),
                "xgboost_auc": float(
                    roc_auc_score(labels[valid_indices], oof[valid_indices])
                ),
                "fixed_rank_blend_auc": float(
                    roc_auc_score(labels[valid_indices], rank_blend[valid_indices])
                ),
                "xgboost_delta_vs_phase2": float(
                    roc_auc_score(labels[valid_indices], oof[valid_indices])
                    - roc_auc_score(labels[valid_indices], phase2_oof[valid_indices])
                ),
                "blend_delta_vs_phase2": float(
                    roc_auc_score(labels[valid_indices], rank_blend[valid_indices])
                    - roc_auc_score(labels[valid_indices], phase2_oof[valid_indices])
                ),
            }
        )
    residual_xgb = labels.astype(np.float64) - oof.astype(np.float64)
    residual_phase2 = labels.astype(np.float64) - phase2_oof
    prediction_corr = float(np.corrcoef(oof, phase2_oof)[0, 1])
    residual_corr = float(np.corrcoef(residual_xgb, residual_phase2)[0, 1])
    squared_error_corr = float(
        np.corrcoef(residual_xgb**2, residual_phase2**2)[0, 1]
    )
    gpu_rows = pd.read_csv(run_dir / "gpu_samples.csv")
    gpu_util = pd.to_numeric(gpu_rows["utilization.gpu"], errors="raise")
    gpu_memory = pd.to_numeric(gpu_rows["memory.used"], errors="raise")
    if int(gpu_util.max()) <= 0:
        raise RuntimeError("No nonzero GPU utilization was observed during training")
    report = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_SCREEN",
        "lineage": "INDEPENDENT_RAW_ONLY_XGBOOST",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "hypothesis": (
            "An independently trained GPU XGBoost model using Phase 2's "
            "rowwise raw application features may provide complementary "
            "OOF signal to the frozen LightGBM/CatBoost blend."
        ),
        "feature_generation": {
            "source": str(BASELINE_SOURCE),
            "target_or_id_in_features": False,
            "rowwise_and_label_free": True,
            "categorical_encoding": (
                "one-hot encoder fitted on each outer-fit fold only; "
                "unseen validation/test categories ignored"
            ),
            "target_encoding_or_historical_prediction_inputs": [],
            "auxiliary_tables_used": [],
            "numeric_feature_count": fold_records[0]["numeric_feature_count"],
            "max_onehot_feature_count": max(
                row["onehot_feature_count"] for row in fold_records
            ),
            "matrix_feature_count": max_matrix_features,
        },
        "model_config": MODEL_CONFIG,
        "runtime": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scipy": scipy.__version__,
            "device_by_fold": [row["device"] for row in fold_records],
            "fold_seconds": [row["seconds"] for row in fold_records],
            "process_memory_before": process_memory_before,
            "process_memory_after": windows_memory_snapshot(),
        },
        "validation": {
            "seed": 42,
            "same_exact_folds_as_frozen_phase2": True,
            "phase2_oof_auc": phase2_auc,
            "xgboost_oof_auc": xgb_auc,
            "xgboost_delta_vs_phase2": xgb_auc - phase2_auc,
            "fixed_diagnostic_rank_blend": {
                "weights": {"phase2": 0.80, "xgboost": 0.20},
                "oof_auc": blend_auc,
                "delta_vs_phase2": blend_auc - phase2_auc,
                "weight_tuned_on_this_oof": False,
                "candidate_promotion": False,
            },
            "prediction_pearson_correlation": prediction_corr,
            "residual_pearson_correlation": residual_corr,
            "squared_error_pearson_correlation": squared_error_corr,
            "per_fold": per_fold,
        },
        "integrity": {
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_rows": int(len(candidate)),
            "candidate_schema": list(candidate.columns),
            "unique_ids": True,
            "matches_raw_test_order": True,
            "matches_sample_order": True,
            "probability_min": float(xgb_test.min()),
            "probability_max": float(xgb_test.max()),
            "gpu_samples": int(len(gpu_rows)),
            "gpu_max_utilization_percent": int(gpu_util.max()),
            "gpu_max_vram_used_mib": int(gpu_memory.max()),
            "official_competition_score": "NOT_EVALUATED",
        },
        "decision": (
            "PROMISING_SCREEN_REQUIRES_INDEPENDENT_SEED"
            if blend_auc > phase2_auc
            else "REJECT_ON_SCREENING"
        ),
    }
    np.save(run_dir / "oof_xgboost.npy", oof)
    np.save(run_dir / "test_xgboost.npy", xgb_test)
    pd.DataFrame(
        {
            "SK_ID_CURR": train_ids,
            "TARGET_TRUE": labels,
            "OOF_XGBOOST": oof,
            "OOF_PHASE2_REFERENCE": phase2_oof,
            "OOF_FIXED_RANK_BLEND": rank_blend,
        }
    ).to_csv(run_dir / "oof_comparison.csv", index=False, float_format="%.10f")
    (run_dir / "xgboost_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Independent Raw XGBoost Diversity Screen",
        "",
        f"- Run: `{report['experiment_id']}`",
        f"- Decision: `{report['decision']}`",
        "- Same exact five seed-42 folds as frozen Phase 2.",
        "- Candidate predictions were generated from raw application train/test CSVs only.",
        "- No auxiliary tables, historical predictions, or target-derived features were used for XGBoost training.",
        "- The fixed 80/20 rank blend is a diagnostic only; its weight was not tuned and it is not promoted.",
        "",
        f"Frozen Phase 2 OOF AUC: `{phase2_auc:.10f}`",
        f"Independent XGBoost OOF AUC: `{xgb_auc:.10f}`",
        f"XGBoost delta: `{xgb_auc - phase2_auc:+.10f}`",
        f"Fixed 80/20 rank-blend OOF AUC: `{blend_auc:.10f}`",
        f"Fixed blend delta: `{blend_auc - phase2_auc:+.10f}`",
        f"OOF prediction Pearson correlation: `{prediction_corr:.6f}`",
        f"OOF residual Pearson correlation: `{residual_corr:.6f}`",
        f"Squared-error Pearson correlation: `{squared_error_corr:.6f}`",
        "",
        "| Fold | Phase 2 AUC | XGBoost AUC | XGB delta | Fixed blend AUC | Blend delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        lines.append(
            f"| {row['fold']} | {row['phase2_auc']:.8f} | "
            f"{row['xgboost_auc']:.8f} | {row['xgboost_delta_vs_phase2']:+.8f} | "
            f"{row['fixed_rank_blend_auc']:.8f} | {row['blend_delta_vs_phase2']:+.8f} |"
        )
    lines.extend(
        [
            "",
            f"- Runtime: `{report['elapsed_seconds']:.1f}` seconds.",
            f"- GPU samples: `{len(gpu_rows)}`; max utilization `{int(gpu_util.max())}%`; peak VRAM `{int(gpu_memory.max())}` MiB.",
            f"- Candidate SHA-256: `{report['integrity']['candidate_sha256']}`",
            "- Official competition score: **NOT EVALUATED**.",
            "",
        ]
    )
    (run_dir / "xgboost_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    source_paths = {
        "experiment_driver": Path(__file__).resolve(),
        "runtime_helpers": Path(__file__).with_name(
            "run_group_context_experiment.py"
        ),
        "feature_builder": BASELINE_SOURCE,
        "phase2_oof_reference": reference_oof_path,
        "phase2_candidate_reference": frozen_candidate,
    }
    for fold_number in range(1, 6):
        source_paths[f"phase2_fold_{fold_number}"] = (
            PHASE2_REFERENCE / f"fold_{fold_number}_indices.npz"
        )
    inputs = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in {**input_paths, **source_paths}.items()
    }
    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            outputs[path.relative_to(run_dir).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": report["experiment_id"],
                "inputs": inputs,
                "outputs": outputs,
                "gpu_samples": int(len(gpu_rows)),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen an independent raw-data XGBoost model against frozen Phase 2."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
