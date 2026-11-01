#!/usr/bin/env python
"""Rebuild the historical five-family expanded zoo in an isolated runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
IMPROVEMENT_DIR = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = IMPROVEMENT_DIR / "baseline" / "source"
CONFIG_PATH = EXPERIMENT_DIR / "config" / "expanded_zoo.json"
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_HISTORICAL_DIR = Path(r"D:\gci\gemini")
ARRAY_KEYS = (
    "oof_lgb_cons",
    "test_lgb_cons",
    "oof_lgb_et",
    "test_lgb_et",
    "oof_xgb",
    "test_xgb",
    "oof_cat",
    "test_cat",
    "oof_lgb_deep",
    "test_lgb_deep",
    "y_true",
)
PREDICTION_PAIRS = (
    ("oof_lgb_cons", "oof_lgb_cons"),
    ("test_lgb_cons", "test_lgb_cons"),
    ("oof_lgb_et", "oof_lgb_et"),
    ("test_lgb_et", "test_lgb_et"),
    ("oof_xgb", "oof_xgb"),
    ("test_xgb", "test_xgb"),
    ("oof_cat", "oof_cat"),
    ("test_cat", "test_cat"),
    ("oof_lgb_deep", "oof_lgb_deep"),
    ("test_lgb_deep", "test_lgb_deep"),
)
CORE_SOURCES = (
    "build_hasan_600_features.py",
    "screen_and_select_features.py",
    "train_hasan_master_zoo.py",
    "train_catboost_and_deep_lgb.py",
)
DOWNSTREAM_SOURCES = (
    "solo_grand_master_apex_080.py",
    "build_v80200_apex.py",
    "build_level2_regime_meta_stacker.py",
    "build_asymmetric_convex_gated_champion.py",
    "build_optimal_transductive_apex_champion.py",
    "build_precision_transductive_apex_champion.py",
    "build_exp85_optimal_macro_apex.py",
)
DOWNSTREAM_OUTPUTS = {
    "v_solo_080041_pure_champion.csv": "v_solo_080041_pure_champion.csv",
    "v_solo_080150_micro_apex_fusion.csv": "v_solo_080150_micro_apex_fusion.csv",
    "v_solo_080200_grand_master_apex.csv": "v_solo_080200_grand_master_apex.csv",
    "v_solo_080300_level2_regime_stacker.csv": "v_solo_080300_level2_regime_stacker.csv",
    "v_20261027_asymmetric_tail_gated_apex.csv": "v_20261027_asymmetric_tail_gated_apex.csv",
    "v_20261027_optimal_transductive_apex.csv": "v_20261027_optimal_transductive_apex.csv",
    "v_20261027_precision_transductive_apex.csv": "v_20261027_precision_transductive_apex.csv",
    "v_20261027_optimal_macro_apex_v85.csv": "v_20261027_optimal_macro_apex_v85.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def verify_sources(config: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, expected in config["source_sha256"].items():
        path = SOURCE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Verified source snapshot is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Source hash mismatch for {path}: expected {expected}, got {actual}")
        hashes[name] = actual
    return hashes


def ensure_junction(junction: Path, target: Path) -> None:
    if junction.exists():
        if not os.path.samefile(junction, target):
            raise FileExistsError(f"Refusing to use pre-existing non-matching raw-data path: {junction}")
        return
    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(f"Could not create raw-data junction {junction}: {result.stdout}\n{result.stderr}")


def check_raw_inputs(data_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for filename, expected_hash in config["raw_inputs"].items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required raw competition input is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Raw input hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")
        report[filename] = {"path": str(path), "sha256": actual_hash, "bytes": path.stat().st_size}

    train_ids = pd.read_csv(data_dir / "train.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy()
    labels = pd.read_csv(data_dir / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
    test_ids = pd.read_csv(data_dir / "test.csv", usecols=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy()
    artifact = config["artifact"]
    if len(train_ids) != artifact["train_rows"] or len(test_ids) != artifact["test_rows"]:
        raise ValueError(f"Unexpected raw row counts: train={len(train_ids)}, test={len(test_ids)}")
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Duplicate SK_ID_CURR values found in raw train/test inputs")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Raw TARGET contains values other than 0 and 1")
    report["row_alignment"] = {
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "train_ids_unique": True,
        "test_ids_unique": True,
        "labels_dtype": str(labels.dtype),
        "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "test_ids_sha256": hashlib.sha256(np.ascontiguousarray(test_ids).tobytes()).hexdigest(),
    }
    return report


def record_fold_indices(labels: np.ndarray, runtime: Path, config: dict[str, Any]) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    fold_config = config["folds"]
    splitter = StratifiedKFold(
        n_splits=fold_config["n_splits"],
        shuffle=fold_config["shuffle"],
        random_state=fold_config["random_state"],
    )
    train_indices: dict[str, np.ndarray] = {}
    validation_indices: dict[str, np.ndarray] = {}
    hashes = []
    dummy = np.zeros(len(labels), dtype=np.uint8)
    for fold_number, (train_idx, validation_idx) in enumerate(splitter.split(dummy, labels)):
        train_idx = train_idx.astype(np.int32)
        validation_idx = validation_idx.astype(np.int32)
        train_indices[f"train_fold_{fold_number}"] = train_idx
        validation_indices[f"validation_fold_{fold_number}"] = validation_idx
        hashes.append(
            {
                "fold": fold_number,
                "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
                "validation_indices_sha256": hashlib.sha256(validation_idx.tobytes()).hexdigest(),
                "train_rows": len(train_idx),
                "validation_rows": len(validation_idx),
            }
        )
    path = runtime / "fold_indices.npz"
    np.savez_compressed(path, **train_indices, **validation_indices)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "splitter": fold_config,
        "folds": hashes,
        "shared_by": ["feature_selection", "base_zoo", "expanded_zoo"],
        "feature_builder_uidy": "same StratifiedKFold seed/config according to pinned historical source",
    }


def assert_prediction_cache_clean(cache_dir: Path) -> None:
    forbidden = (
        "oof_lgb_cons.npy",
        "test_lgb_cons.npy",
        "oof_lgb_et.npy",
        "test_lgb_et.npy",
        "oof_xgb.npy",
        "test_xgb.npy",
        "hasan_zoo_predictions.npz",
        "hasan_expanded_zoo.npz",
    )
    present = [name for name in forbidden if (cache_dir / name).exists()]
    if present:
        raise RuntimeError(f"Refusing to train with pre-existing prediction caches: {present}")


def validate_feature_frame(
    path: Path,
    expected_rows: int,
    expected_columns: int,
    historical_reference: Path | None = None,
) -> dict[str, Any]:
    frame = pd.read_feather(path)
    if frame.shape != (expected_rows, expected_columns):
        raise ValueError(f"{path.name} has shape {frame.shape}; expected {(expected_rows, expected_columns)}")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    infinity_count = int(np.isinf(numeric).sum())
    details = {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(frame.shape),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "nan_count": int(frame.isna().sum().sum()),
        "infinity_count": infinity_count,
    }
    if infinity_count:
        if historical_reference is None or not historical_reference.is_file():
            raise ValueError(
                f"{path.name} contains {infinity_count} infinities and has no historical reference "
                "to establish whether they are expected"
            )
        parity = frame_comparison(path, historical_reference)
        if not parity["exact_values"] or parity["infinity_count_generated"] != parity["infinity_count_reference"]:
            raise ValueError(f"{path.name} contains infinities that do not exactly match its historical reference")
        details["infinity_reference_parity"] = {
            "exact_values": True,
            "historical_sha256": parity["reference_sha256"],
            "historical_infinity_count": parity["infinity_count_reference"],
        }
    return details


def validate_base_zoo(path: Path, train_rows: int, test_rows: int) -> dict[str, Any]:
    expected = (
        "oof_lgb_cons",
        "test_lgb_cons",
        "oof_lgb_et",
        "test_lgb_et",
        "oof_xgb",
        "test_xgb",
        "y_true",
    )
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != expected:
            raise ValueError(f"Base zoo keys/order mismatch: expected {expected}, got {tuple(archive.files)}")
        arrays = {key: archive[key] for key in expected}
    details = {"path": str(path), "sha256": sha256_file(path), "arrays": {}}
    for key, array in arrays.items():
        is_label = key == "y_true"
        expected_length = train_rows if key.startswith("oof_") or is_label else test_rows
        expected_dtype = np.int64 if is_label else np.float64
        if array.shape != (expected_length,) or array.dtype != expected_dtype:
            raise ValueError(f"Base zoo {key}: got shape={array.shape}, dtype={array.dtype}")
        if is_label:
            if not np.isin(array, [0, 1]).all():
                raise ValueError("Base zoo y_true contains invalid labels")
        elif not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"Base zoo {key} has invalid prediction values")
        details["arrays"][key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": int(array.min()) if is_label else float(array.min()),
            "max": int(array.max()) if is_label else float(array.max()),
        }
    return details


class GpuTelemetry:
    def __init__(self, path: Path, interval_seconds: float = 2.0) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    @staticmethod
    def sample() -> str:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("nvidia-smi returned no GPU devices")
        return "\n".join(lines)

    def _collect(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["timestamp", "gpu_name", "utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "power_w"]
            )
            while not self.stop_event.is_set():
                try:
                    sample = self.sample()
                    for line in sample.splitlines():
                        writer.writerow([field.strip() for field in line.split(",")])
                    stream.flush()
                except Exception as exc:
                    self.error = str(exc)
                    return
                self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.sample()
        self.thread = threading.Thread(target=self._collect, name="gpu-telemetry", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)


def gpu_preflight() -> dict[str, Any]:
    import catboost
    import xgboost

    details: dict[str, Any] = {"nvidia_smi": GpuTelemetry.sample()}
    xgb_model = xgboost.XGBClassifier(
        n_estimators=2,
        max_depth=2,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        n_jobs=1,
        verbosity=0,
    )
    x = np.asarray([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    xgb_model.fit(x, y)
    xgb_config = json.loads(xgb_model.get_booster().save_config())
    xgb_device = xgb_config["learner"]["generic_param"]["device"]
    if not str(xgb_device).startswith("cuda"):
        raise RuntimeError(f"XGBoost preflight did not configure CUDA: device={xgb_device}")

    cat_model = catboost.CatBoostClassifier(
        iterations=2,
        depth=2,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )
    cat_model.fit(x, y)
    cat_params = cat_model.get_all_params()
    if cat_params.get("task_type") != "GPU":
        raise RuntimeError(f"CatBoost preflight did not use GPU: task_type={cat_params.get('task_type')}")
    details["xgboost_device"] = xgb_device
    details["catboost_task_type"] = cat_params["task_type"]
    details["xgboost_version"] = xgboost.__version__
    details["catboost_version"] = catboost.__version__
    return details


def run_stage(
    name: str,
    source_name: str,
    cwd: Path,
    log_path: Path,
    stages: list[dict[str, Any]],
    replacements: dict[str, str] | None = None,
) -> None:
    original = SOURCE_DIR / source_name
    script_path = original
    original_hash = sha256_file(original)
    if replacements:
        source_text = original.read_text(encoding="utf-8")
        for old, new in replacements.items():
            count = source_text.count(old)
            if count != 1:
                raise ValueError(f"{name}: expected exactly one occurrence of {old!r}, found {count}")
            source_text = source_text.replace(old, new)
        isolated_dir = cwd / "isolated_source"
        isolated_dir.mkdir(parents=True, exist_ok=True)
        script_path = isolated_dir / source_name
        script_path.write_text(source_text, encoding="utf-8")

    command = [sys.executable, "-u", str(script_path)]
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, errors="replace", check=False)
    elapsed = time.monotonic() - start_clock
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n{'=' * 20} {name} {'=' * 20}\n")
        log.write(f"Command: {json.dumps(command)}\n")
        log.write(f"Exit code: {result.returncode}; runtime_seconds: {elapsed:.3f}\n")
        log.write(result.stdout)
        if result.stderr:
            log.write("\n[stderr]\n")
            log.write(result.stderr)
        log.flush()
    stages.append(
        {
            "name": name,
            "source": str(original),
            "source_sha256": original_hash,
            "executed_script": str(script_path),
            "executed_script_sha256": sha256_file(script_path),
            "command": command,
            "cwd": str(cwd),
            "started_at_utc": started,
            "exit_code": result.returncode,
            "runtime_seconds": round(elapsed, 3),
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }
    )
    if result.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}; see {log_path}")


def frame_comparison(generated: Path, reference: Path) -> dict[str, Any]:
    left = pd.read_feather(generated)
    right = pd.read_feather(reference)
    result: dict[str, Any] = {
        "generated_sha256": sha256_file(generated),
        "reference_sha256": sha256_file(reference),
        "generated_shape": list(left.shape),
        "reference_shape": list(right.shape),
        "column_order_exact": list(left.columns) == list(right.columns),
        "dtype_order_exact": list(left.dtypes.astype(str)) == list(right.dtypes.astype(str)),
        "exact_values": False,
        "nan_count_generated": int(left.isna().sum().sum()),
        "nan_count_reference": int(right.isna().sum().sum()),
        "infinity_count_generated": 0,
        "infinity_count_reference": 0,
        "max_abs_difference": None,
    }
    numeric = left.select_dtypes(include=[np.number])
    reference_numeric = right.select_dtypes(include=[np.number])
    result["infinity_count_generated"] = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
    result["infinity_count_reference"] = int(np.isinf(reference_numeric.to_numpy(dtype=np.float64)).sum())
    if left.shape == right.shape and list(left.columns) == list(right.columns):
        a = left.to_numpy()
        b = right.to_numpy()
        try:
            equal = np.array_equal(a, b, equal_nan=True)
        except TypeError:
            equal = np.array_equal(a, b)
        result["exact_values"] = bool(equal)
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                result["max_abs_difference"] = float(np.max(np.abs(a[finite] - b[finite])))
    return result


def validate_zoo(path: Path, config: dict[str, Any], labels: np.ndarray) -> dict[str, Any]:
    artifact = config["artifact"]
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if keys != list(artifact["keys_in_order"]):
            raise ValueError(f"Expanded zoo keys/order mismatch: expected {artifact['keys_in_order']}, got {keys}")
        arrays = {key: archive[key] for key in keys}

    details: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "keys_in_order": list(arrays),
        "arrays": {},
    }
    for key in ARRAY_KEYS:
        array = arrays[key]
        is_train = key.startswith("oof_") or key == "y_true"
        expected_length = artifact["train_rows"] if is_train else artifact["test_rows"]
        if array.ndim != 1 or len(array) != expected_length:
            raise ValueError(f"{key} has shape {array.shape}; expected ({expected_length},)")
        if key == "y_true":
            if str(array.dtype) != artifact["label_dtype"]:
                raise ValueError(f"y_true dtype is {array.dtype}; expected {artifact['label_dtype']}")
            if not np.array_equal(array, labels):
                raise ValueError("Expanded zoo y_true does not exactly match raw train TARGET order")
        else:
            if str(array.dtype) != artifact["prediction_dtype"]:
                raise ValueError(f"{key} dtype is {array.dtype}; expected {artifact['prediction_dtype']}")
            if not np.isfinite(array).all():
                raise ValueError(f"{key} contains non-finite predictions")
            low, high = artifact["prediction_range"]
            if np.any(array < low) or np.any(array > high):
                raise ValueError(f"{key} predictions fall outside [{low}, {high}]")
        details["arrays"][key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }
    return details


def prediction_comparison(generated: Path, reference: Path, id_column: str = "SK_ID_CURR") -> dict[str, Any]:
    left = pd.read_csv(generated)
    right = pd.read_csv(reference)
    if id_column not in left or id_column not in right or "TARGET" not in left or "TARGET" not in right:
        raise ValueError(f"Cannot compare submission schema for {generated} and {reference}")
    if left[id_column].duplicated().any() or right[id_column].duplicated().any():
        raise ValueError(f"Duplicate IDs in prediction comparison: {generated}, {reference}")
    aligned = left[[id_column, "TARGET"]].merge(
        right[[id_column, "TARGET"]],
        on=id_column,
        how="outer",
        suffixes=("_generated", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    complete = bool((aligned["_merge"] == "both").all() and len(aligned) == 61500)
    result: dict[str, Any] = {
        "generated_sha256": sha256_file(generated),
        "reference_sha256": sha256_file(reference),
        "generated_rows": len(left),
        "reference_rows": len(right),
        "generated_columns": list(left.columns),
        "reference_columns": list(right.columns),
        "id_coverage_exact": complete,
        "id_order_exact": bool(np.array_equal(left[id_column].to_numpy(), right[id_column].to_numpy())),
        "missing_or_extra_ids": int((aligned["_merge"] != "both").sum()),
    }
    if not complete:
        return result
    a = aligned["TARGET_generated"].to_numpy(dtype=np.float64)
    b = aligned["TARGET_reference"].to_numpy(dtype=np.float64)
    result.update(
        {
            "exact_values": bool(np.array_equal(a, b)),
            "pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic),
            "mae": float(np.mean(np.abs(a - b))),
            "rmse": float(np.sqrt(np.mean(np.square(a - b)))),
            "max_abs_difference": float(np.max(np.abs(a - b))),
            "mean_rank_displacement": float(np.mean(np.abs(rankdata(a) - rankdata(b)))),
            "max_rank_displacement": float(np.max(np.abs(rankdata(a) - rankdata(b)))),
            "generated_range": [float(np.min(a)), float(np.max(a))],
            "reference_range": [float(np.min(b)), float(np.max(b))],
        }
    )
    return result


def compare_zoo(generated: Path, reference: Path) -> dict[str, Any]:
    details: dict[str, Any] = {
        "generated_sha256": sha256_file(generated),
        "reference_sha256": sha256_file(reference),
        "alignment": "positional raw train/test row order; neither historical NPZ stores SK_ID_CURR",
        "arrays": {},
    }
    with np.load(generated, allow_pickle=False) as new, np.load(reference, allow_pickle=False) as old:
        details["generated_keys"] = list(new.files)
        details["reference_keys"] = list(old.files)
        for generated_key, reference_key in PREDICTION_PAIRS:
            if generated_key not in new or reference_key not in old:
                details["arrays"][generated_key] = {"status": "MISSING_FROM_ONE_ARCHIVE"}
                continue
            a = new[generated_key]
            b = old[reference_key]
            if a.shape != b.shape:
                details["arrays"][generated_key] = {
                    "status": "SHAPE_MISMATCH",
                    "generated_shape": list(a.shape),
                    "reference_shape": list(b.shape),
                }
                continue
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            details["arrays"][generated_key] = {
                "status": "EXACT" if np.array_equal(a, b) else "NUMERICAL_DIFFERENCE",
                "exact_values": bool(np.array_equal(a, b)),
                "pearson": float(pearsonr(a, b).statistic),
                "spearman": float(spearmanr(a, b).statistic),
                "mae": float(delta.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                "max_abs_difference": float(delta.max()),
                "mean_rank_displacement": float(np.mean(np.abs(rankdata(a) - rankdata(b)))),
                "max_rank_displacement": float(np.max(np.abs(rankdata(a) - rankdata(b)))),
                "generated_dtype": str(a.dtype),
                "reference_dtype": str(b.dtype),
            }
    return details


def python_string(value: Path | str) -> str:
    return json.dumps(str(value))


def downstream_replacements(source_name: str, runtime: Path, historical: Path) -> dict[str, str]:
    if source_name == "build_asymmetric_convex_gated_champion.py":
        return {
            'champ_path = "d:/gci/gemini/v_solo_080300_level2_regime_stacker.csv"':
                'champ_path = "v_solo_080300_level2_regime_stacker.csv"',
            'output_path = "d:/gci/gemini/v_20261027_asymmetric_tail_gated_apex.csv"':
                'output_path = "v_20261027_asymmetric_tail_gated_apex.csv"',
        }
    if source_name == "build_exp85_optimal_macro_apex.py":
        path_values = {
            '"d:/gci/gemini/v_20261027_precision_transductive_apex.csv"': python_string(
                runtime / "v_20261027_precision_transductive_apex.csv"
            ),
            '"d:/gci/gemini/v_20261027_grand_master_macro_stacker_apex.csv"': python_string(
                runtime / "v_20261027_grand_master_macro_stacker_apex.csv"
            ),
            '"d:/gci/gemini/cache/v_17_full_suite_bundle.npz"': python_string(
                runtime / "cache" / "v_17_full_suite_bundle.npz"
            ),
            '"d:/gci/gemini/cache/v55_members.npz"': python_string(runtime / "cache" / "v55_members.npz"),
            '"d:/gci/gemini/cache/oof_10f_xgb.npy"': python_string(runtime / "cache" / "oof_10f_xgb.npy"),
            '"d:/gci/gemini/cache/test_10f_xgb.npy"': python_string(runtime / "cache" / "test_10f_xgb.npy"),
            '"d:/gci/gemini/cache/y_train.npy"': python_string(runtime / "cache" / "y_train.npy"),
            '"d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv"': python_string(
                runtime / "v_20261027_optimal_macro_apex_v85.csv"
            ),
            '"d:/gci/gemini/experminetbygeminioct22.csv"': python_string(
                runtime / "experminetbygeminioct22.csv"
            ),
        }
        return path_values
    return {}


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required historical downstream input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise IOError(f"Copy hash mismatch: {source} -> {destination}")


def setup_historical_downstream_inputs(runtime: Path, historical: Path) -> dict[str, str]:
    copies = {
        historical / "scratch" / "final_sub" / "final_submission" / "submission_final.csv":
            runtime / "scratch" / "final_sub" / "final_submission" / "submission_final.csv",
        historical / "v_gold_grand_master_apex.csv": runtime / "v_gold_grand_master_apex.csv",
        historical / "v_20_apex_grand_champion_submission.csv":
            runtime / "v_20_apex_grand_champion_submission.csv",
        historical / "v_20261027_grand_master_macro_stacker_apex.csv":
            runtime / "v_20261027_grand_master_macro_stacker_apex.csv",
        historical / "cache" / "v_17_full_suite_bundle.npz":
            runtime / "cache" / "v_17_full_suite_bundle.npz",
        historical / "cache" / "v55_members.npz": runtime / "cache" / "v55_members.npz",
        historical / "cache" / "oof_10f_xgb.npy": runtime / "cache" / "oof_10f_xgb.npy",
        historical / "cache" / "test_10f_xgb.npy": runtime / "cache" / "test_10f_xgb.npy",
    }
    hashes = {}
    for source, destination in copies.items():
        copy_required(source, destination)
        hashes[str(source)] = sha256_file(destination)
    return hashes


def collect_downstream_comparisons(runtime: Path, historical: Path) -> dict[str, Any]:
    comparisons = {}
    for filename in sorted(DOWNSTREAM_OUTPUTS):
        generated = runtime / filename
        reference = historical / filename
        if generated.exists() and reference.exists():
            comparisons[filename] = prediction_comparison(generated, reference)
        else:
            comparisons[filename] = {
                "status": "REFERENCE_OR_GENERATED_OUTPUT_MISSING",
                "generated_exists": generated.exists(),
                "reference_exists": reference.exists(),
            }
    return comparisons


def run_downstream(
    runtime: Path,
    historical: Path,
    data_dir: Path,
    log_path: Path,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    # Original scripts use absolute d:/gci/inpuy reads; these resolve to data_dir.
    # Their writes are either relative to runtime or precisely rewired below.
    input_hashes = setup_historical_downstream_inputs(runtime, historical)
    order = (
        "solo_grand_master_apex_080.py",
        "build_v80200_apex.py",
        "build_level2_regime_meta_stacker.py",
        "build_asymmetric_convex_gated_champion.py",
        "build_optimal_transductive_apex_champion.py",
        "build_precision_transductive_apex_champion.py",
        "build_exp85_optimal_macro_apex.py",
    )
    for source_name in order:
        run_stage(
            name=f"downstream:{source_name}",
            source_name=source_name,
            cwd=runtime,
            log_path=log_path,
            stages=stages,
            replacements=downstream_replacements(source_name, runtime, historical),
        )

    return {
        "historical_input_sha256": input_hashes,
        "prediction_comparisons": collect_downstream_comparisons(runtime, historical),
    }


def integrate_repeat_run(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    historical_dir = Path(args.historical_dir).resolve()
    run_dir = Path(args.integration_repeat).resolve()
    root = run_dir / "integration_repeat"
    runtime = root / "runtime"
    cache_dir = runtime / "cache"
    log_path = root / "execution.log"
    report_path = root / "integration_report.json"
    stages: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "experiment": "EXP03_repeat_zoo_downstream_integration",
        "classification": "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "run_directory": str(root),
        "runtime_directory": str(runtime),
        "source_sha256": {},
        "raw_inputs": {},
        "stages": stages,
        "historical_input_sha256": {},
        "prediction_comparisons": {},
        "gpu": "No model training occurs in these downstream scripts.",
    }
    if root.exists():
        raise FileExistsError(f"Integration directory already exists; refusing to overwrite: {root}")
    cache_dir.mkdir(parents=True)
    repeat_report_path = run_dir / "repeat_validation" / "repeat_report.json"
    repeat_zoo = run_dir / "repeat_validation" / "runtime" / "cache" / config["artifact"]["filename"]
    try:
        if not repeat_report_path.is_file() or not repeat_zoo.is_file():
            raise FileNotFoundError("A completed CatBoost/deep repeat run is required")
        repeat_report = json.loads(repeat_report_path.read_text(encoding="utf-8"))
        if repeat_report.get("status") != "COMPLETED":
            raise ValueError("The fixed-seed repeat run did not complete successfully")
        report["repeat_report_sha256"] = sha256_file(repeat_report_path)
        report["repeat_zoo_sha256"] = sha256_file(repeat_zoo)
        report["source_sha256"] = verify_sources(config)
        report["raw_inputs"] = check_raw_inputs(data_dir, config)
        labels = pd.read_csv(data_dir / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
        report["repeat_zoo_schema"] = validate_zoo(repeat_zoo, config, labels)
        copy_required(repeat_zoo, cache_dir / config["artifact"]["filename"])
        np.save(cache_dir / "y_train.npy", labels)
        ensure_junction(root / "inpuy", data_dir)
        report["historical_input_sha256"] = setup_historical_downstream_inputs(runtime, historical_dir)
        report["historical_zoo_parity"] = compare_zoo(
            cache_dir / config["artifact"]["filename"],
            historical_dir / "cache" / config["artifact"]["filename"],
        )
        report["downstream"] = run_downstream(
            runtime=runtime,
            historical=historical_dir,
            data_dir=data_dir,
            log_path=log_path,
            stages=stages,
        )
        report["prediction_comparisons"] = report["downstream"]["prediction_comparisons"]
        report["status"] = "COMPLETED"
    except Exception as exc:
        report["status"] = "FAILED_PARTIAL" if stages else "FAILED"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["prediction_comparisons"] = collect_downstream_comparisons(runtime, historical_dir)
        print(f"EXP03 integration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["integration_runner_sha256"] = sha256_file(Path(__file__).resolve())
        report["execution_log_sha256"] = sha256_file(log_path) if log_path.is_file() else None
        report["generated_runtime_files"] = [
            {"path": str(path.relative_to(runtime)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(runtime.rglob("*"))
            if path.is_file()
        ]
        json_write(report_path, report)
        print(f"Integration report: {report_path}")
    return 0 if report["status"] == "COMPLETED" else 1


def continue_existing_run(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    historical_dir = Path(args.historical_dir).resolve()
    run_dir = Path(args.continue_run).resolve()
    initial_report_path = run_dir / "run_report.json"
    runtime = run_dir / "runtime"
    cache_dir = runtime / "cache"
    log_path = run_dir / "execution.log"
    report_path = run_dir / "continuation_report.json"
    stages: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "experiment": "EXP03_postrun_validation_and_downstream_replay",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "run_directory": str(run_dir),
        "runtime_directory": str(runtime),
        "classification": "UNRESOLVED",
        "component_classification": {
            "expanded_zoo": "UNRESOLVED",
            "downstream_chain": "NOT_RUN",
        },
        "stages": stages,
        "outputs": {},
        "comparisons": {},
        "packages": package_versions(),
    }
    try:
        if not initial_report_path.is_file() or not runtime.is_dir():
            raise FileNotFoundError(f"Run report/runtime not found under {run_dir}")
        initial = json.loads(initial_report_path.read_text(encoding="utf-8"))
        report["initial_run_report"] = str(initial_report_path)
        report["initial_run_report_sha256"] = sha256_file(initial_report_path)
        report["initial_run_status"] = initial.get("status")
        report["initial_validation_failure"] = initial.get("failure")
        if initial.get("status") != "FAILED" or "contains" not in initial.get("failure", {}).get("message", ""):
            raise ValueError("Continuation is allowed only for a preserved run stopped by post-build validation")

        source_hashes = verify_sources(config)
        if source_hashes != initial.get("source_sha256"):
            raise ValueError("Pinned historical source hashes differ from the original generation run")
        report["source_sha256"] = source_hashes
        report["raw_inputs"] = check_raw_inputs(data_dir, config)
        core_stages = {
            stage.get("name"): stage
            for stage in initial.get("stages", [])
            if stage.get("name", "").startswith("raw_rebuild:")
        }
        expected_names = {f"raw_rebuild:{name}" for name in CORE_SOURCES}
        if set(core_stages) != expected_names or any(stage.get("exit_code") != 0 for stage in core_stages.values()):
            raise ValueError("The original run did not complete all four pinned raw-rebuild stages successfully")
        report["original_core_stages"] = list(core_stages.values())

        for item in initial.get("generated_runtime_files", []):
            artifact_path = runtime / item["path"]
            if not artifact_path.is_file() or sha256_file(artifact_path) != item["sha256"]:
                raise ValueError(f"Original generated output is missing or changed: {artifact_path}")
        report["original_generated_runtime_file_count"] = len(initial.get("generated_runtime_files", []))

        if not os.path.samefile(run_dir / "inpuy", data_dir):
            raise ValueError("Original run's raw-data junction no longer points to the pinned raw-data directory")
        labels = pd.read_csv(data_dir / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
        cached_labels = np.load(cache_dir / "y_train.npy", allow_pickle=False)
        if not np.array_equal(cached_labels, labels):
            raise ValueError("Generated runtime y_train.npy does not match raw train.csv TARGET order")

        rows = config["artifact"]["train_rows"]
        test_rows = config["artifact"]["test_rows"]
        feature_rows = config["artifact"]["feature_rows"]
        report["outputs"]["feature_matrices"] = {}
        report["outputs"]["selected_feature_matrices"] = {}
        feature_references = (
            ("hasan_features_tr.feather", "features_669_train", rows, feature_rows, "feature_matrices", "train"),
            ("hasan_features_te.feather", "features_669_test", test_rows, feature_rows, "feature_matrices", "test"),
            (
                "hasan_selected_550_tr.feather",
                "selected_550_train",
                rows,
                config["artifact"]["selected_features"],
                "selected_feature_matrices",
                "train",
            ),
            (
                "hasan_selected_550_te.feather",
                "selected_550_test",
                test_rows,
                config["artifact"]["selected_features"],
                "selected_feature_matrices",
                "test",
            ),
        )
        for filename, comparison_key, expected_rows, expected_columns, output_key, partition in feature_references:
            generated = cache_dir / filename
            reference = historical_dir / "cache" / filename
            interface = validate_feature_frame(generated, expected_rows, expected_columns, reference)
            parity = frame_comparison(generated, reference)
            report["outputs"][output_key][partition] = interface
            report["comparisons"][comparison_key] = parity
            if not parity["exact_values"] or not parity["column_order_exact"] or not parity["dtype_order_exact"]:
                raise ValueError(f"Historical feature parity failed for {filename}")

        report["outputs"]["base_zoo_predictions"] = validate_base_zoo(
            cache_dir / "hasan_zoo_predictions.npz", rows, test_rows
        )
        generated_zoo = cache_dir / config["artifact"]["filename"]
        report["outputs"]["expanded_zoo"] = validate_zoo(generated_zoo, config, labels)
        report["component_classification"]["expanded_zoo"] = "INDEPENDENT_RAW_REBUILD"
        report["classification"] = "INDEPENDENT_RAW_REBUILD"
        reference_zoo = historical_dir / "cache" / config["artifact"]["filename"]
        if not reference_zoo.is_file():
            raise FileNotFoundError(f"Historical comparison reference is missing: {reference_zoo}")
        report["comparisons"]["expanded_zoo"] = compare_zoo(generated_zoo, reference_zoo)
        report["gpu"] = {
            "preflight": initial.get("gpu", {}).get("preflight"),
            "telemetry_path": initial.get("gpu", {}).get("telemetry_path"),
            "telemetry_samples": initial.get("gpu", {}).get("telemetry_samples"),
        }

        if not args.zoo_only:
            report["downstream"] = run_downstream(
                runtime=runtime,
                historical=historical_dir,
                data_dir=data_dir,
                log_path=log_path,
                stages=stages,
            )
            report["classification"] = "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION"
            report["component_classification"]["downstream_chain"] = "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION"
        report["status"] = "COMPLETED"
    except Exception as exc:
        report["status"] = "FAILED"
        if any(stage.get("name", "").startswith("downstream:") for stage in stages):
            report["classification"] = "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION_FAILED_OR_PARTIAL"
            report["component_classification"]["downstream_chain"] = (
                "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION_FAILED_OR_PARTIAL"
            )
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"EXP03 continuation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["continuation_runner_sha256"] = sha256_file(Path(__file__).resolve())
        report["execution_log"] = str(log_path)
        report["execution_log_sha256"] = sha256_file(log_path) if log_path.is_file() else None
        report["generated_runtime_files"] = [
            {
                "path": str(path.relative_to(runtime)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(runtime.rglob("*"))
            if path.is_file()
        ]
        json_write(report_path, report)
        print(f"Continuation report: {report_path}")
    return 0 if report["status"] == "COMPLETED" else 1


def repeat_cuda_run(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.repeat_cuda_run).resolve()
    main_runtime = run_dir / "runtime"
    repeat_dir = run_dir / "repeat_validation"
    repeat_runtime = repeat_dir / "runtime"
    repeat_cache = repeat_runtime / "cache"
    log_path = repeat_dir / "execution.log"
    report_path = repeat_dir / "repeat_report.json"
    report: dict[str, Any] = {
        "experiment": "EXP03_cuda_repeatability_check",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "classification": "REPEAT_USING_INDEPENDENT_RAW_REBUILT_INTERMEDIATES",
        "source_sha256": {},
        "raw_inputs": {},
        "copied_input_sha256": {},
        "stages": [],
        "gpu": {},
    }
    if repeat_dir.exists():
        raise FileExistsError(f"Repeat-check directory already exists; refusing to overwrite: {repeat_dir}")
    repeat_cache.mkdir(parents=True)
    repeat_runtime.mkdir(parents=True, exist_ok=True)
    telemetry = GpuTelemetry(repeat_dir / "gpu_telemetry.csv")
    try:
        initial_path = run_dir / "run_report.json"
        continuation_path = run_dir / "continuation_report.json"
        if not initial_path.is_file() or not continuation_path.is_file():
            raise FileNotFoundError("Run and continuation reports are required before the repeatability check")
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
        if continuation.get("component_classification", {}).get("expanded_zoo") != "INDEPENDENT_RAW_REBUILD":
            raise ValueError("The initial expanded zoo has not passed independent raw-rebuild validation")
        report["reference_run_report_sha256"] = sha256_file(initial_path)
        report["continuation_report_sha256"] = sha256_file(continuation_path)
        report["source_sha256"] = verify_sources(config)
        report["raw_inputs"] = check_raw_inputs(data_dir, config)

        source_cache = main_runtime / "cache"
        required = (
            "hasan_selected_550_tr.feather",
            "hasan_selected_550_te.feather",
            "hasan_zoo_predictions.npz",
        )
        for name in required:
            source = source_cache / name
            destination = repeat_cache / name
            copy_required(source, destination)
            report["copied_input_sha256"][name] = sha256_file(destination)
        report["base_zoo_input_sha256"] = report["copied_input_sha256"]["hasan_zoo_predictions.npz"]
        report["gpu"]["preflight"] = gpu_preflight()
        telemetry.start()
        run_stage(
            name="repeat:train_catboost_and_deep_lgb.py",
            source_name="train_catboost_and_deep_lgb.py",
            cwd=repeat_runtime,
            log_path=log_path,
            stages=report["stages"],
        )
        generated = repeat_cache / config["artifact"]["filename"]
        reference = source_cache / config["artifact"]["filename"]
        labels = pd.read_csv(data_dir / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
        report["output_schema"] = validate_zoo(generated, config, labels)
        report["prediction_comparisons"] = compare_zoo(generated, reference)
        report["deterministic_exact_match"] = all(
            metrics.get("exact_values", False)
            for name, metrics in report["prediction_comparisons"]["arrays"].items()
            if name in {"oof_cat", "test_cat", "oof_lgb_deep", "test_lgb_deep"}
        )
        report["status"] = "COMPLETED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"EXP03 repeat check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        telemetry.stop()
        report["gpu"]["telemetry_path"] = str(telemetry.path)
        report["gpu"]["telemetry_samples"] = (
            max(0, sum(1 for _ in telemetry.path.open(encoding="utf-8")) - 1) if telemetry.path.exists() else 0
        )
        if telemetry.error:
            report["gpu"]["telemetry_error"] = telemetry.error
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["repeat_runner_sha256"] = sha256_file(Path(__file__).resolve())
        report["execution_log_sha256"] = sha256_file(log_path) if log_path.is_file() else None
        report["output_files"] = [
            {"path": str(path.relative_to(repeat_runtime)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(repeat_runtime.rglob("*"))
            if path.is_file()
        ]
        json_write(report_path, report)
        print(f"Repeatability report: {report_path}")
    return 0 if report["status"] == "COMPLETED" else 1


def execute(args: argparse.Namespace) -> int:
    if args.integration_repeat:
        return integrate_repeat_run(args)
    if args.repeat_cuda_run:
        return repeat_cuda_run(args)
    if args.continue_run:
        return continue_existing_run(args)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    historical_dir = Path(args.historical_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else EXPERIMENT_DIR / "runs" / f"exp03_{timestamp}"
    runtime = run_dir / "runtime"
    cache_dir = runtime / "cache"
    log_path = run_dir / "execution.log"
    report_path = run_dir / "run_report.json"
    telemetry_path = run_dir / "gpu_telemetry.csv"
    stages: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "experiment": "EXP03_missing_dependency_rebuild",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "UNRESOLVED",
        "component_classification": {
            "expanded_zoo": "UNRESOLVED",
            "downstream_chain": "NOT_RUN",
        },
        "status": "RUNNING",
        "run_directory": str(run_dir),
        "runtime_directory": str(runtime),
        "packages": package_versions(),
        "source_sha256": {},
        "stages": stages,
        "gpu": {},
        "comparisons": {},
        "outputs": {},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    telemetry = GpuTelemetry(telemetry_path)
    try:
        report["source_sha256"] = verify_sources(config)
        report["raw_inputs"] = check_raw_inputs(data_dir, config)
        ensure_junction(run_dir / "inpuy", data_dir)
        train_labels = pd.read_csv(data_dir / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
        report["fold_indices"] = record_fold_indices(train_labels, runtime, config)
        label_cache = cache_dir / "y_train.npy"
        np.save(label_cache, train_labels)
        report["outputs"]["y_train.npy"] = {
            "path": str(label_cache),
            "sha256": sha256_file(label_cache),
            "dtype": str(train_labels.dtype),
            "shape": list(train_labels.shape),
            "origin": "raw train.csv TARGET column",
        }
        report["gpu"]["preflight"] = gpu_preflight()
        telemetry.start()

        # A fresh cache is essential: the historical master trainer otherwise
        # silently loads pre-existing base-model OOF/test predictions.
        assert_prediction_cache_clean(cache_dir)

        for source_name in CORE_SOURCES:
            run_stage(
                name=f"raw_rebuild:{source_name}",
                source_name=source_name,
                cwd=runtime,
                log_path=log_path,
                stages=stages,
            )

        rows = config["artifact"]["train_rows"]
        test_rows = config["artifact"]["test_rows"]
        features = config["artifact"]["feature_rows"]
        report["outputs"]["feature_matrices"] = {
            "train": validate_feature_frame(
                cache_dir / "hasan_features_tr.feather",
                rows,
                features,
                historical_dir / "cache" / "hasan_features_tr.feather",
            ),
            "test": validate_feature_frame(
                cache_dir / "hasan_features_te.feather",
                test_rows,
                features,
                historical_dir / "cache" / "hasan_features_te.feather",
            ),
        }
        report["outputs"]["selected_feature_matrices"] = {
            "train": validate_feature_frame(
                cache_dir / "hasan_selected_550_tr.feather",
                rows,
                config["artifact"]["selected_features"],
                historical_dir / "cache" / "hasan_selected_550_tr.feather",
            ),
            "test": validate_feature_frame(
                cache_dir / "hasan_selected_550_te.feather",
                test_rows,
                config["artifact"]["selected_features"],
                historical_dir / "cache" / "hasan_selected_550_te.feather",
            ),
        }
        report["outputs"]["base_zoo_predictions"] = validate_base_zoo(
            cache_dir / "hasan_zoo_predictions.npz", rows, test_rows
        )
        generated_zoo = cache_dir / config["artifact"]["filename"]
        report["outputs"]["expanded_zoo"] = validate_zoo(generated_zoo, config, train_labels)
        report["classification"] = "INDEPENDENT_RAW_REBUILD"
        report["component_classification"]["expanded_zoo"] = "INDEPENDENT_RAW_REBUILD"

        reference_zoo = historical_dir / "cache" / config["artifact"]["filename"]
        if reference_zoo.is_file():
            report["comparisons"]["expanded_zoo"] = compare_zoo(generated_zoo, reference_zoo)
        else:
            report["comparisons"]["expanded_zoo"] = {"status": "REFERENCE_NOT_FOUND"}

        feature_refs = (
            ("hasan_features_tr.feather", "hasan_features_tr.feather", "features_669_train"),
            ("hasan_features_te.feather", "hasan_features_te.feather", "features_669_test"),
            ("hasan_selected_550_tr.feather", "hasan_selected_550_tr.feather", "selected_550_train"),
            ("hasan_selected_550_te.feather", "hasan_selected_550_te.feather", "selected_550_test"),
        )
        for filename, reference_name, label in feature_refs:
            generated = cache_dir / filename
            reference = historical_dir / "cache" / reference_name
            if reference.is_file():
                report["comparisons"][label] = frame_comparison(generated, reference)
            else:
                report["comparisons"][label] = {"status": "REFERENCE_NOT_FOUND"}

        if not args.zoo_only:
            report["downstream"] = run_downstream(
                runtime=runtime,
                historical=historical_dir,
                data_dir=data_dir,
                log_path=log_path,
                stages=stages,
            )
            report["classification"] = "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION"
            report["component_classification"]["downstream_chain"] = "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION"
        report["status"] = "COMPLETED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"EXP03 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        telemetry.stop()
        report["gpu"]["telemetry_path"] = str(telemetry_path)
        report["gpu"]["telemetry_samples"] = max(0, sum(1 for _ in telemetry_path.open(encoding="utf-8")) - 1) if telemetry_path.exists() else 0
        if telemetry.error:
            report["gpu"]["telemetry_error"] = telemetry.error
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime_seconds"] = round(
            (datetime.fromisoformat(report["finished_at_utc"]) - datetime.fromisoformat(report["started_at_utc"])).total_seconds(),
            3,
        )
        report["execution_log"] = str(log_path)
        report["execution_log_sha256"] = sha256_file(log_path) if log_path.is_file() else None
        report["runner_sha256"] = sha256_file(Path(__file__).resolve())
        generated_zoo_path = cache_dir / config["artifact"]["filename"]
        if generated_zoo_path.is_file():
            report["outputs"]["expanded_zoo_sha256"] = sha256_file(generated_zoo_path)
        report["generated_runtime_files"] = [
            {
                "path": str(path.relative_to(runtime)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(runtime.rglob("*"))
            if path.is_file()
        ]
        json_write(report_path, report)
        print(f"Run report: {report_path}")
    return 0 if report["status"] == "COMPLETED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing raw train.csv and test.csv")
    parser.add_argument("--historical-dir", default=str(DEFAULT_HISTORICAL_DIR), help="Historical cache/reference root")
    parser.add_argument("--run-dir", help="New run directory; it must not already exist")
    parser.add_argument("--continue-run", help="Validate completed raw stages in an existing run, then resume downstream")
    parser.add_argument("--repeat-cuda-run", help="Repeat the CatBoost/deep-LGB training in an isolated runtime")
    parser.add_argument("--integration-repeat", help="Run the historical downstream chain using a fixed-seed raw repeat")
    parser.add_argument("--zoo-only", action="store_true", help="Skip downstream hybrid integration")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
