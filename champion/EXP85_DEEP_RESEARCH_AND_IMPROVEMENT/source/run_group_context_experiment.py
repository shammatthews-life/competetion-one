from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
from sklearn.metrics import roc_auc_score

from phase2_group_context import (
    GROUP_COLUMNS,
    NUMERIC_COLUMNS,
    build_fold_group_context,
)


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
FROZEN_ROOT = PROJECT_ROOT / "EXP85_PHASE2_FROZEN_RELEASE"
PHASE2_SCRIPT = (
    FROZEN_ROOT
    / "SOURCE"
    / "project"
    / "experiments"
    / "phase_02_fold_safe_key_encodings"
    / "run_experiment.py"
)
PHASE2_REFERENCE = FROZEN_ROOT / "EVIDENCE" / "original_runs" / "phase_02"
EXPECTED_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
    "phase2_candidate": "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tee:
    def __init__(self, stream, log_path: Path) -> None:
        self.stream = stream
        self.log = log_path.open("w", encoding="utf-8", newline="\n")

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


class GpuMonitor:
    def __init__(self, output_path: Path, interval_seconds: float = 5.0) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.sample_count = 0
        self.errors: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("GPU monitoring thread did not stop cleanly")

    def _sample(self) -> None:
        fields = (
            "timestamp,name,driver_version,memory.total,memory.used,"
            "utilization.gpu,utilization.memory"
        )
        with self.output_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(fields + "\n")
            while True:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--query-gpu={fields}",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    line = result.stdout.strip().replace("\r", "")
                    if line:
                        stream.write(line + "\n")
                        stream.flush()
                        self.sample_count += 1
                except (OSError, subprocess.SubprocessError) as error:
                    self.errors.append(f"{type(error).__name__}: {error}")
                if self.stop_event.wait(self.interval_seconds):
                    break


def load_phase2_module():
    if not PHASE2_SCRIPT.is_file():
        raise FileNotFoundError(f"Frozen Phase 2 runner is missing: {PHASE2_SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        "frozen_phase2_group_context_runner", PHASE2_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen Phase 2 runner: {PHASE2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_fold(fold_number: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(PHASE2_REFERENCE / f"fold_{fold_number}_indices.npz") as data:
        return data["fit_indices"], data["valid_indices"]


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
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {EXPECTED_SHA256[name]}, "
                f"got {observed}"
            )
    frozen_candidate = (
        FROZEN_ROOT / "CANDIDATE" / "submission_key_target_encoding.csv"
    )
    if sha256_file(frozen_candidate) != EXPECTED_SHA256["phase2_candidate"]:
        raise ValueError("Frozen Phase 2 candidate hash changed")
    if not PHASE2_REFERENCE.joinpath("oof_50_50_blend.npy").is_file():
        raise FileNotFoundError("Frozen Phase 2 OOF reference is missing")

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    module = load_phase2_module()
    original_builder = module.build_rowwise_features
    captured_raw: dict[str, pd.DataFrame] = {}
    builder_calls = 0
    def capture_builder(raw: pd.DataFrame) -> pd.DataFrame:
        nonlocal builder_calls
        if "TARGET" in raw.columns:
            raise ValueError("Feature builder received TARGET")
        result = original_builder(raw)
        label = "train" if builder_calls == 0 else "test" if builder_calls == 1 else None
        if label is None:
            raise RuntimeError("Unexpected additional rowwise feature-builder call")
        capture_columns = list(dict.fromkeys([*GROUP_COLUMNS, *(
            column
            for column in (
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
            )
        )]))
        missing = set(capture_columns) - set(raw.columns)
        if missing:
            raise ValueError(f"Raw feature frame missing capture fields: {sorted(missing)}")
        captured_raw[label] = raw.loc[:, capture_columns].reset_index(drop=True).copy()
        builder_calls += 1
        return result

    module.build_rowwise_features = capture_builder
    original_trainers = {
        "lightgbm": module.train_lightgbm_fold,
        "catboost": module.train_catboost_fold,
    }
    cached_fold: int | None = None
    cached_train: pd.DataFrame | None = None
    cached_test: pd.DataFrame | None = None

    def train_with_group_context(original_trainer):
        def wrapped(
            X_train,
            y,
            X_test,
            fit_indices,
            valid_indices,
            fold_number,
        ):
            nonlocal cached_fold, cached_train, cached_test
            if set(captured_raw) != {"train", "test"}:
                raise RuntimeError("Raw covariates were not captured before training")
            if cached_fold != fold_number:
                cached_train, cached_test = build_fold_group_context(
                    captured_raw["train"],
                    captured_raw["test"],
                    fit_indices,
                    valid_indices,
                )
                cached_fold = fold_number
            if cached_train is None or cached_test is None:
                raise RuntimeError("Group-context feature cache was not constructed")
            if len(cached_train) != len(X_train) or len(cached_test) != len(X_test):
                raise ValueError("Group-context feature rows do not align with model data")
            train_features = pd.concat(
                [X_train.reset_index(drop=True), cached_train], axis=1
            )
            test_features = pd.concat(
                [X_test.reset_index(drop=True), cached_test], axis=1
            )
            if train_features.columns.has_duplicates or test_features.columns.has_duplicates:
                raise ValueError("Group-context features created duplicate model columns")
            return original_trainer(
                train_features,
                y,
                test_features,
                fit_indices,
                valid_indices,
                fold_number,
            )

        return wrapped

    module.train_lightgbm_fold = train_with_group_context(original_trainers["lightgbm"])
    module.train_catboost_fold = train_with_group_context(original_trainers["catboost"])

    monitor_temp_path = run_dir.parent / f".{run_dir.name}.gpu_samples.tmp"
    log_temp_path = run_dir.parent / f".{run_dir.name}.training.log.tmp"
    monitor = GpuMonitor(monitor_temp_path)
    log_tee = Tee(sys.stdout, log_temp_path)
    old_stdout = sys.stdout
    sys.stdout = log_tee
    monitor.start()
    try:
        summary = module.run(
            argparse.Namespace(data_dir=data_dir, run_dir=run_dir)
        )
        if builder_calls != 2 or set(captured_raw) != {"train", "test"}:
            raise RuntimeError("Raw covariate capture did not complete as expected")
    except Exception as error:
        failure = {
            "status": "FAILED",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "gpu_monitor_errors": monitor.errors,
            "gpu_sample_count": monitor.sample_count,
        }
        (run_dir / "failure_report.json").write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    finally:
        monitor.stop()
        sys.stdout = old_stdout
        log_tee.close()
        if monitor_temp_path.is_file():
            monitor_temp_path.replace(run_dir / "gpu_samples.csv")
        if log_temp_path.is_file():
            log_temp_path.replace(run_dir / "training.log")

    if monitor.errors:
        raise RuntimeError(f"GPU evidence collection failed: {monitor.errors}")
    if monitor.sample_count < 2:
        raise RuntimeError(
            f"Insufficient GPU activity samples: {monitor.sample_count}"
        )

    train_labels = pd.read_csv(input_paths["train"], usecols=["TARGET"])[
        "TARGET"
    ].to_numpy(dtype=np.int8)
    test_ids = pd.read_csv(input_paths["test"], usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy(dtype=np.int64)
    sample = pd.read_csv(input_paths["sample"])
    candidate_path = run_dir / "submission_key_target_encoding.csv"
    candidate = pd.read_csv(candidate_path)
    if list(candidate.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"Candidate has invalid schema: {list(candidate.columns)}")
    if not np.array_equal(candidate["SK_ID_CURR"].to_numpy(), test_ids):
        raise ValueError("Candidate IDs/order differ from raw test")
    if not np.array_equal(candidate["SK_ID_CURR"].to_numpy(), sample["SK_ID_CURR"].to_numpy()):
        raise ValueError("Candidate IDs/order differ from sample submission")
    predictions = candidate["TARGET"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(predictions).all()
        or np.any((predictions < 0.0) | (predictions > 1.0))
    ):
        raise ValueError("Candidate contains invalid probabilities")

    candidate_oof = np.load(run_dir / "oof_50_50_blend.npy", allow_pickle=False)
    phase2_oof = np.load(
        PHASE2_REFERENCE / "oof_50_50_blend.npy", allow_pickle=False
    ).astype(np.float64)
    if len(candidate_oof) != len(train_labels) or len(phase2_oof) != len(train_labels):
        raise ValueError("Candidate or frozen Phase 2 OOF length mismatch")
    per_fold = []
    for fold_number in range(1, 6):
        actual_fit, actual_valid = load_reference_fold(fold_number)
        with np.load(run_dir / f"fold_{fold_number}_indices.npz") as fold:
            fit_indices = fold["fit_indices"]
            valid_indices = fold["valid_indices"]
        if not np.array_equal(actual_fit, fit_indices) or not np.array_equal(
            actual_valid, valid_indices
        ):
            raise ValueError(f"Fold {fold_number} differs from frozen Phase 2 folds")
        candidate_auc = float(
            roc_auc_score(train_labels[valid_indices], candidate_oof[valid_indices])
        )
        reference_auc = float(
            roc_auc_score(train_labels[valid_indices], phase2_oof[valid_indices])
        )
        per_fold.append(
            {
                "fold": fold_number,
                "phase2_reference_auc": reference_auc,
                "group_context_auc": candidate_auc,
                "paired_auc_delta": candidate_auc - reference_auc,
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
            }
        )

    fold_deltas = np.asarray(
        [row["paired_auc_delta"] for row in per_fold], dtype=np.float64
    )
    report = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_DIAGNOSTIC_VALIDATION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "hypothesis": (
            "Training-fold-only category peer means and within-group deviations "
            "add useful context beyond Phase 2 rowwise features and product-key "
            "target encodings."
        ),
        "feature_definition": {
            "group_columns": list(GROUP_COLUMNS),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "aggregate_features": [
                "log group row count",
                "peer mean for each numeric field",
                "row value minus peer mean",
            ],
            "feature_count": int(len(GROUP_COLUMNS) * (1 + 2 * len(NUMERIC_COLUMNS))),
            "group_statistics_fit_on": "outer-fold training rows only",
            "training_row_own_measurement_excluded": True,
            "validation_and_test_covariates_in_fit": False,
            "target_used_in_feature_engineering": False,
            "ID_used_in_feature_engineering": False,
        },
        "validation": {
            "folds": 5,
            "seed": 42,
            "same_exact_folds_as_frozen_phase2": True,
            "phase2_oof_path": str(PHASE2_REFERENCE / "oof_50_50_blend.npy"),
            "phase2_oof_sha256": sha256_file(
                PHASE2_REFERENCE / "oof_50_50_blend.npy"
            ),
            "phase2_reference_full_oof_auc": float(
                roc_auc_score(train_labels, phase2_oof)
            ),
            "group_context_full_oof_auc": float(
                roc_auc_score(train_labels, candidate_oof)
            ),
            "full_oof_auc_delta": float(
                roc_auc_score(train_labels, candidate_oof)
                - roc_auc_score(train_labels, phase2_oof)
            ),
            "mean_fold_auc_delta": float(fold_deltas.mean()),
            "std_fold_auc_delta": float(fold_deltas.std(ddof=1)),
            "positive_folds": int((fold_deltas > 0).sum()),
            "per_fold": per_fold,
        },
        "integrity": {
            "candidate_sha256": sha256_file(candidate_path),
            "test_rows": int(len(candidate)),
            "test_id_order_matches_raw_test": True,
            "test_id_order_matches_sample": True,
            "candidate_schema": list(candidate.columns),
            "finite_probability_range": [
                float(predictions.min()),
                float(predictions.max()),
            ],
            "gpu_samples": monitor.sample_count,
            "gpu_samples_path": str(run_dir / "gpu_samples.csv"),
            "model_devices": summary["model_folds"],
            "official_competition_score": "NOT_EVALUATED",
        },
        "decision": (
            "REJECT_ON_SCREENING"
            if roc_auc_score(train_labels, candidate_oof)
            <= roc_auc_score(train_labels, phase2_oof)
            else "PROMISING_SCREEN_ONLY_SECOND_SEED_REQUIRED"
        ),
    }
    (run_dir / "group_context_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    md = [
        "# Phase 2 Group-Context Feature Experiment",
        "",
        f"- Run: `{report['experiment_id']}`",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        "- Same five seed-42 folds as frozen Phase 2; validation labels were used only to score OOF predictions.",
        "- Fold group statistics use only outer-training rows; each training row's own measurement is left out.",
        "- No historical champion or test labels were read.",
        "",
        "| Model | Full OOF AUC |",
        "|---|---:|",
        f"| Frozen Phase 2 | {report['validation']['phase2_reference_full_oof_auc']:.10f} |",
        f"| Phase 2 + group context | {report['validation']['group_context_full_oof_auc']:.10f} |",
        "",
        f"Full OOF delta: `{report['validation']['full_oof_auc_delta']:+.10f}`",
        f"Mean paired fold delta: `{report['validation']['mean_fold_auc_delta']:+.10f}` "
        f"(SD `{report['validation']['std_fold_auc_delta']:.10f}`; "
        f"{report['validation']['positive_folds']}/5 positive folds).",
        "",
        "| Fold | Phase 2 AUC | Group-context AUC | Paired delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        md.append(
            f"| {row['fold']} | {row['phase2_reference_auc']:.8f} | "
            f"{row['group_context_auc']:.8f} | {row['paired_auc_delta']:+.8f} |"
        )
    md.extend(
        [
            "",
            "The historical champion was not involved in feature design or selection. "
            "This is a screening result on the already-used seed-42 folds; do not promote "
            "unless a predeclared independent seed confirms it.",
            "",
            f"Candidate SHA-256: `{report['integrity']['candidate_sha256']}`",
            "Official score: **NOT EVALUATED**.",
            "",
        ]
    )
    (run_dir / "group_context_report.md").write_text(
        "\n".join(md), encoding="utf-8", newline="\n"
    )

    source_paths = {
        "experiment_driver": Path(__file__).resolve(),
        "group_context_feature_code": Path(__file__).with_name(
            "phase2_group_context.py"
        ),
        "frozen_phase2_runner": PHASE2_SCRIPT,
        "frozen_baseline_source": FROZEN_ROOT
        / "SOURCE"
        / "project"
        / "source"
        / "run_independent_baseline.py",
        "frozen_phase1_source": FROZEN_ROOT
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py",
    }
    input_records = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in {**input_paths, **source_paths}.items()
    }
    input_records["frozen_phase2_candidate"] = {
        "path": str(frozen_candidate.resolve()),
        "size_bytes": frozen_candidate.stat().st_size,
        "sha256": sha256_file(frozen_candidate),
    }
    input_records["frozen_phase2_oof"] = {
        "path": str((PHASE2_REFERENCE / "oof_50_50_blend.npy").resolve()),
        "size_bytes": (PHASE2_REFERENCE / "oof_50_50_blend.npy").stat().st_size,
        "sha256": sha256_file(PHASE2_REFERENCE / "oof_50_50_blend.npy"),
    }
    output_records = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            output_records[path.relative_to(run_dir).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest = {
        "run_id": report["experiment_id"],
        "input_records": input_records,
        "output_records": output_records,
        "gpu_monitor": {
            "sample_count": monitor.sample_count,
            "interval_seconds": monitor.interval_seconds,
            "errors": monitor.errors,
        },
        "report_sha256": sha256_file(run_dir / "group_context_report.json"),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen fold-safe category-group peer features on frozen Phase 2 folds."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
