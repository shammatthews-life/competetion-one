from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
FROZEN_ROOT = PROJECT_ROOT / "EXP85_PHASE2_FROZEN_RELEASE"
PHASE2_REFERENCE_ROOT = FROZEN_ROOT / "EVIDENCE"
EXPECTED_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
    "phase2_candidate": "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa",
}
POLY_COLUMNS = (
    "P_EXTMEAN_PAY",
    "P_EXTMEAN_CTI",
    "P_EXTMEAN_PAY_AGE",
    "P_CREDIT_GOODS_INC",
    "P_EXTMEAN_ANN_INC",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_group_context_experiment as runtime_helpers
import run_raw_xgb_diversity as xgb_helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0.0)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def build_epoly_features(raw: pd.DataFrame) -> pd.DataFrame:
    if "TARGET" in raw.columns:
        raise ValueError("Interaction builder must not receive TARGET")
    required = {
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_ANNUITY",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing interaction sources: {sorted(missing)}")
    ext_mean = raw[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].mean(axis=1).to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    income = raw["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64)
    goods = raw["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    age_years = -raw["DAYS_BIRTH"].to_numpy(dtype=np.float64) / 365.25
    annuity_credit = safe_ratio(annuity, credit)
    credit_income = safe_ratio(credit, income)
    annuity_income = safe_ratio(annuity, income)
    features = pd.DataFrame(
        {
            "P_EXTMEAN_PAY": ext_mean * annuity_credit,
            "P_EXTMEAN_CTI": ext_mean * credit_income,
            "P_EXTMEAN_PAY_AGE": ext_mean * annuity_credit * age_years,
            "P_CREDIT_GOODS_INC": safe_ratio(credit - goods, income),
            "P_EXTMEAN_ANN_INC": ext_mean * annuity_income,
        }
    )
    if list(features.columns) != list(POLY_COLUMNS):
        raise RuntimeError("Polynomial interaction schema changed")
    if np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Interaction features contain infinity")
    return features


def add_epoly_features(
    raw: pd.DataFrame, rowwise_features: pd.DataFrame
) -> pd.DataFrame:
    if "TARGET" in raw.columns:
        raise ValueError("Feature builder must not receive TARGET")
    if len(raw) != len(rowwise_features):
        raise ValueError("Raw and rowwise feature rows do not align")
    duplicates = set(POLY_COLUMNS) & set(rowwise_features.columns)
    if duplicates:
        raise ValueError(f"Interaction columns already exist: {sorted(duplicates)}")
    return pd.concat(
        [
            rowwise_features.reset_index(drop=True),
            build_epoly_features(raw).reset_index(drop=True),
        ],
        axis=1,
    )


def reference_paths(seed: int) -> tuple[Path, Path, str]:
    if seed == 42:
        directory = PHASE2_REFERENCE_ROOT / "original_runs" / "phase_02"
        oof_name = "oof_50_50_blend.npy"
    elif seed == 20261101:
        directory = PHASE2_REFERENCE_ROOT / "second_seed" / "main"
        oof_name = "oof_phase2_fold_local_blend.npy"
    else:
        raise ValueError("Only prevalidated seeds 42 and 20261101 are supported")
    return directory, directory / oof_name, oof_name


def load_phase2_module():
    script = (
        FROZEN_ROOT
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Frozen Phase 2 runner is missing: {script}")
    spec = importlib.util.spec_from_file_location(
        "frozen_phase2_epoly_runner", script
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen Phase 2 runner: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(data_dir: Path, run_dir: Path, seed: int) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be new or empty: {run_dir}")
    raw_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for name, path in raw_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} CSV is missing: {path}")
        if sha256_file(path) != EXPECTED_SHA256[name]:
            raise ValueError(f"{name} raw input hash changed")
    frozen_candidate = (
        FROZEN_ROOT / "CANDIDATE" / "submission_key_target_encoding.csv"
    )
    if sha256_file(frozen_candidate) != EXPECTED_SHA256["phase2_candidate"]:
        raise ValueError("Frozen Phase 2 candidate hash changed")
    fold_reference, reference_oof_path, reference_oof_name = reference_paths(seed)
    reference_oof = np.load(reference_oof_path, allow_pickle=False).astype(
        np.float64
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    module = load_phase2_module()
    original_builder = module.build_rowwise_features

    def build_with_epoly(raw: pd.DataFrame) -> pd.DataFrame:
        base = original_builder(raw)
        return add_epoly_features(raw, base)

    module.build_rowwise_features = build_with_epoly
    module.FOLD_SEED = seed
    module.train_lightgbm_fold.__globals__["FOLD_SEED"] = seed
    module.train_catboost_fold.__globals__["FOLD_SEED"] = seed
    monitor_path = run_dir.parent / f".{run_dir.name}.gpu_samples.tmp"
    log_path = run_dir.parent / f".{run_dir.name}.training.log.tmp"
    monitor = runtime_helpers.GpuMonitor(monitor_path)
    tee = runtime_helpers.Tee(sys.stdout, log_path)
    original_stdout = sys.stdout
    memory_before = runtime_helpers_memory_snapshot()
    sys.stdout = tee
    monitor.start()
    try:
        model_summary = module.run(
            argparse.Namespace(data_dir=data_dir, run_dir=run_dir)
        )
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
    train_raw = pd.read_csv(raw_paths["train"])
    test_raw = pd.read_csv(raw_paths["test"])
    sample = pd.read_csv(raw_paths["sample"])
    labels = train_raw["TARGET"].to_numpy(dtype=np.int8)
    oof = np.load(run_dir / "oof_50_50_blend.npy", allow_pickle=False)
    if len(reference_oof) != len(labels) or len(oof) != len(labels):
        raise ValueError("OOF arrays do not match raw training rows")
    per_fold = []
    for fold_number in range(1, 6):
        expected_path = fold_reference / f"fold_{fold_number}_indices.npz"
        actual_path = run_dir / f"fold_{fold_number}_indices.npz"
        with np.load(expected_path, allow_pickle=False) as expected, np.load(
            actual_path, allow_pickle=False
        ) as actual:
            for name in ("fit_indices", "valid_indices"):
                if not np.array_equal(expected[name], actual[name]):
                    raise ValueError(
                        f"Run fold {fold_number} differs from Phase 2 seed {seed}"
                    )
            valid_indices = actual["valid_indices"]
            fit_rows = int(len(actual["fit_indices"]))
        phase2_fold_auc = float(
            roc_auc_score(labels[valid_indices], reference_oof[valid_indices])
        )
        epoly_fold_auc = float(
            roc_auc_score(labels[valid_indices], oof[valid_indices])
        )
        per_fold.append(
            {
                "fold": fold_number,
                "phase2_auc": phase2_fold_auc,
                "epoly_auc": epoly_fold_auc,
                "paired_auc_delta": epoly_fold_auc - phase2_fold_auc,
                "fit_rows": fit_rows,
                "valid_rows": int(len(valid_indices)),
            }
        )

    phase2_auc = float(roc_auc_score(labels, reference_oof))
    epoly_auc = float(roc_auc_score(labels, oof))
    deltas = np.asarray(
        [row["paired_auc_delta"] for row in per_fold], dtype=np.float64
    )
    mean_delta = float(deltas.mean())
    delta_se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    positive_folds = int((deltas > 0).sum())
    screen_pass = (
        mean_delta >= 0.0002
        and positive_folds >= 4
        and mean_delta > 2.0 * delta_se
    )

    candidate_path = run_dir / "submission_key_target_encoding.csv"
    candidate = pd.read_csv(candidate_path)
    test_ids = test_raw["SK_ID_CURR"]
    sample_ids = sample["SK_ID_CURR"]
    predictions = candidate["TARGET"].to_numpy(dtype=np.float64)
    if list(candidate.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Submission schema changed")
    if (
        len(candidate) != 61500
        or candidate["SK_ID_CURR"].duplicated().any()
        or not candidate["SK_ID_CURR"].equals(test_ids)
        or not candidate["SK_ID_CURR"].equals(sample_ids)
        or not np.isfinite(predictions).all()
        or np.any((predictions < 0.0) | (predictions > 1.0))
    ):
        raise ValueError("Submission ID/schema/probability integrity failed")
    gpu_rows = pd.read_csv(run_dir / "gpu_samples.csv")
    gpu_util = pd.to_numeric(gpu_rows["utilization.gpu"], errors="raise")
    gpu_memory = pd.to_numeric(gpu_rows["memory.used"], errors="raise")
    if int(gpu_util.max()) <= 0:
        raise RuntimeError("No GPU utilization was observed during model fitting")

    report = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_SCREEN",
        "experiment_family": "E_POLY_FINANCIAL_AND_EXT_SOURCE_INTERACTIONS",
        "outer_seed": seed,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "hypothesis": (
            "Five pre-specified rowwise interactions between EXT_SOURCE mean, "
            "repayment ratios, credit/income, and age may expose monotone "
            "risk interactions more efficiently than tree splits alone."
        ),
        "feature_definition": {
            "columns": list(POLY_COLUMNS),
            "source": "raw application covariates only",
            "rowwise_and_label_free": True,
            "target_or_id_used": False,
            "outer_fit_or_validation_statistics": False,
        },
        "model": {
            "models": ["LightGBM CPU", "CatBoost GPU"],
            "configuration_source": str(
                FROZEN_ROOT
                / "SOURCE"
                / "project"
                / "source"
                / "run_independent_baseline.py"
            ),
            "nested_target_encoding": "unchanged Phase 2 implementation",
            "outer_seed": seed,
        },
        "validation": {
            "reference_oof": str(reference_oof_path),
            "same_exact_folds_as_phase2": True,
            "phase2_oof_auc": phase2_auc,
            "epoly_oof_auc": epoly_auc,
            "full_oof_auc_delta": epoly_auc - phase2_auc,
            "mean_paired_fold_delta": mean_delta,
            "paired_fold_delta_standard_error": delta_se,
            "positive_folds": positive_folds,
            "predeclared_screen_gate": {
                "mean_delta_at_least": 0.0002,
                "positive_folds_at_least": 4,
                "mean_delta_more_than_two_standard_errors": True,
            },
            "screen_gate_passed": screen_pass,
            "per_fold": per_fold,
        },
        "runtime": {
            "device_by_model_fold": model_summary["model_folds"],
            "fold_seconds": model_summary["fold_execution"],
            "process_memory_before": memory_before,
            "process_memory_after": runtime_helpers_memory_snapshot(),
            "gpu_samples": int(len(gpu_rows)),
            "gpu_max_utilization_percent": int(gpu_util.max()),
            "gpu_max_vram_used_mib": int(gpu_memory.max()),
        },
        "integrity": {
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_rows": int(len(candidate)),
            "candidate_schema": list(candidate.columns),
            "unique_ids": True,
            "matches_raw_test_order": True,
            "matches_sample_order": True,
            "probability_min": float(predictions.min()),
            "probability_max": float(predictions.max()),
            "official_competition_score": "NOT_EVALUATED",
        },
        "decision": (
            "PROMISING_SCREEN_REQUIRES_INDEPENDENT_SEED"
            if screen_pass
            else "REJECT_ON_SCREENING"
        ),
    }
    (run_dir / "epoly_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# E_POLY Interaction Screen",
        "",
        f"- Run: `{report['experiment_id']}`",
        f"- Seed: `{seed}`",
        f"- Decision: `{report['decision']}`",
        "- Same exact saved outer folds as the Phase 2 reference.",
        "- Interaction definitions are rowwise and label-free; no target or applicant ID is used.",
        "",
        f"Frozen Phase 2 OOF AUC: `{phase2_auc:.10f}`",
        f"E_POLY OOF AUC: `{epoly_auc:.10f}`",
        f"Pooled OOF delta: `{epoly_auc - phase2_auc:+.10f}`",
        f"Mean paired fold delta: `{mean_delta:+.10f}`",
        f"Paired-fold standard error: `{delta_se:.10f}`",
        f"Positive folds: `{positive_folds}/5`",
        "",
        "| Fold | Phase 2 AUC | E_POLY AUC | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        lines.append(
            f"| {row['fold']} | {row['phase2_auc']:.8f} | "
            f"{row['epoly_auc']:.8f} | {row['paired_auc_delta']:+.8f} |"
        )
    lines.extend(
        [
            "",
            f"- Runtime: `{report['elapsed_seconds']:.1f}` seconds.",
            f"- GPU samples: `{len(gpu_rows)}`; max utilization `{int(gpu_util.max())}%`; peak VRAM `{int(gpu_memory.max())}` MiB.",
            f"- Candidate SHA-256: `{report['integrity']['candidate_sha256']}`",
            "- Official score: **NOT EVALUATED**.",
            "",
        ]
    )
    (run_dir / "epoly_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    phase2_script = (
        FROZEN_ROOT
        / "SOURCE"
        / "project"
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py"
    )
    feature_source = (
        FROZEN_ROOT
        / "SOURCE"
        / "project"
        / "source"
        / "run_independent_baseline.py"
    )
    source_paths = {
        "experiment_driver": Path(__file__).resolve(),
        "runtime_helpers": Path(__file__).with_name(
            "run_group_context_experiment.py"
        ),
        "phase2_runner": phase2_script,
        "phase2_feature_source": feature_source,
        "phase2_reference_oof": reference_oof_path,
        "phase2_candidate": frozen_candidate,
    }
    for fold_number in range(1, 6):
        source_paths[f"phase2_reference_fold_{fold_number}"] = (
            fold_reference / f"fold_{fold_number}_indices.npz"
        )
    inputs = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in {**raw_paths, **source_paths}.items()
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


def runtime_helpers_memory_snapshot() -> dict[str, Any]:
    return {
        "process_memory": xgb_helpers.windows_memory_snapshot()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Screen five fixed E_POLY interactions using the frozen Phase 2 models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 20261101], default=42)
    args = parser.parse_args()
    run(args.data_dir, args.run_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
