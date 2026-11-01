from __future__ import annotations

import argparse
import hashlib
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

from phase2_category_keys import CATEGORY_SPECS, build_category_keys


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
FROZEN_ROOT = PROJECT_ROOT / "EXP85_PHASE2_FROZEN_RELEASE"
PHASE2_REFERENCE = FROZEN_ROOT / "EVIDENCE" / "original_runs" / "phase_02"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_group_context_experiment as runtime_helpers

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
    if not (PHASE2_REFERENCE / "oof_50_50_blend.npy").is_file():
        raise FileNotFoundError("Frozen Phase 2 OOF reference is missing")

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    module = runtime_helpers.load_phase2_module()
    original_builder = module.build_rowwise_features
    original_specs = tuple(module.KEY_SPECS)
    module.KEY_SPECS = original_specs + CATEGORY_SPECS
    captured: dict[str, pd.DataFrame] = {}
    builder_calls = 0

    def capture_builder(raw: pd.DataFrame) -> pd.DataFrame:
        nonlocal builder_calls
        if "TARGET" in raw.columns:
            raise ValueError("Feature builder received TARGET")
        result = original_builder(raw)
        label = "train" if builder_calls == 0 else "test" if builder_calls == 1 else None
        if label is None:
            raise RuntimeError("Unexpected additional rowwise feature-builder call")
        columns = [column for _, column, _ in CATEGORY_SPECS]
        captured[label] = raw.loc[:, columns].reset_index(drop=True).copy()
        builder_calls += 1
        return result

    module.build_rowwise_features = capture_builder

    def category_aware_key_arrays(fee_train, fee_test):
        if set(captured) != {"train", "test"}:
            raise RuntimeError("Category covariates were not captured")
        combined_fee = pd.concat([fee_train, fee_test], ignore_index=True)
        keys = {}
        for name, column, _ in original_specs:
            values = combined_fee[column].round(
                0 if name == "PRODUCT_K" else int(column[-1])
            )
            keys[name] = (
                values.astype("string")
                .fillna(module.MISSING_KEY)
                .to_numpy(dtype=str)
            )
        keys.update(build_category_keys(captured["train"], captured["test"]))
        return keys

    module.key_arrays = category_aware_key_arrays
    monitor_temp_path = run_dir.parent / f".{run_dir.name}.gpu_samples.tmp"
    log_temp_path = run_dir.parent / f".{run_dir.name}.training.log.tmp"
    monitor = runtime_helpers.GpuMonitor(monitor_temp_path)
    tee = runtime_helpers.Tee(sys.stdout, log_temp_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    monitor.start()
    try:
        summary = module.run(
            argparse.Namespace(data_dir=data_dir, run_dir=run_dir)
        )
        if builder_calls != 2 or set(captured) != {"train", "test"}:
            raise RuntimeError("Category raw covariate capture failed")
    except Exception as error:
        (run_dir / "failure_report.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
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
        sys.stdout = old_stdout
        tee.close()
        if monitor_temp_path.is_file():
            monitor_temp_path.replace(run_dir / "gpu_samples.csv")
        if log_temp_path.is_file():
            log_temp_path.replace(run_dir / "training.log")

    if monitor.errors or monitor.sample_count < 2:
        raise RuntimeError(
            f"GPU evidence incomplete: samples={monitor.sample_count}, "
            f"errors={monitor.errors}"
        )

    labels = pd.read_csv(input_paths["train"], usecols=["TARGET"])[
        "TARGET"
    ].to_numpy(dtype=np.int8)
    reference_blend = np.load(
        PHASE2_REFERENCE / "oof_50_50_blend.npy", allow_pickle=False
    ).astype(np.float64)
    candidate_blend = np.load(
        run_dir / "oof_50_50_blend.npy", allow_pickle=False
    ).astype(np.float64)
    if len(labels) != len(reference_blend) or len(labels) != len(candidate_blend):
        raise ValueError("OOF vector lengths do not match training labels")

    per_fold = []
    for fold_number in range(1, 6):
        reference_fit, reference_valid = load_reference_fold(fold_number)
        with np.load(run_dir / f"fold_{fold_number}_indices.npz") as fold:
            fit_indices = fold["fit_indices"]
            valid_indices = fold["valid_indices"]
        if not np.array_equal(fit_indices, reference_fit) or not np.array_equal(
            valid_indices, reference_valid
        ):
            raise ValueError(f"Fold {fold_number} differs from frozen Phase 2 folds")
        ref_auc = float(
            roc_auc_score(labels[valid_indices], reference_blend[valid_indices])
        )
        candidate_auc = float(
            roc_auc_score(labels[valid_indices], candidate_blend[valid_indices])
        )
        per_fold.append(
            {
                "fold": fold_number,
                "phase2_reference_auc": ref_auc,
                "category_te_auc": candidate_auc,
                "paired_auc_delta": candidate_auc - ref_auc,
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
            }
        )

    candidate_path = run_dir / "submission_key_target_encoding.csv"
    candidate = pd.read_csv(candidate_path)
    test_ids = pd.read_csv(input_paths["test"], usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    sample = pd.read_csv(input_paths["sample"])
    predictions = candidate["TARGET"].to_numpy(dtype=np.float64)
    if list(candidate.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Candidate submission schema is invalid")
    if not np.array_equal(candidate["SK_ID_CURR"].to_numpy(), test_ids) or not np.array_equal(
        candidate["SK_ID_CURR"].to_numpy(), sample["SK_ID_CURR"].to_numpy()
    ):
        raise ValueError("Candidate IDs/order do not match raw test and sample")
    if (
        len(candidate) != 61500
        or candidate["SK_ID_CURR"].duplicated().any()
        or not np.isfinite(predictions).all()
        or np.any((predictions < 0.0) | (predictions > 1.0))
    ):
        raise ValueError("Candidate ID, row-count, or probability check failed")

    fold_deltas = np.asarray(
        [row["paired_auc_delta"] for row in per_fold], dtype=np.float64
    )
    ref_auc = float(roc_auc_score(labels, reference_blend))
    candidate_auc = float(roc_auc_score(labels, candidate_blend))
    gpu_rows = pd.read_csv(run_dir / "gpu_samples.csv")
    gpu_util = pd.to_numeric(gpu_rows["utilization.gpu"], errors="raise")
    gpu_memory = pd.to_numeric(gpu_rows["memory.used"], errors="raise")
    report = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_DIAGNOSTIC_VALIDATION",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_definition": {
            "new_target_encoding_keys": [
                {"name": name, "source_column": column, "smoothing": smoothing}
                for name, column, smoothing in CATEGORY_SPECS
            ],
            "nested_training_encodings": True,
            "own_target_excluded_from_training_encoding": True,
            "validation_and_test_labels_used": False,
            "training_and_validation_folds": 5,
            "inner_seed": 7,
            "outer_seed": 42,
        },
        "validation": {
            "same_exact_folds_as_frozen_phase2": True,
            "phase2_reference_full_oof_auc": ref_auc,
            "category_te_full_oof_auc": candidate_auc,
            "full_oof_auc_delta": candidate_auc - ref_auc,
            "mean_fold_auc_delta": float(fold_deltas.mean()),
            "std_fold_auc_delta": float(fold_deltas.std(ddof=1)),
            "positive_folds": int((fold_deltas > 0).sum()),
            "per_fold": per_fold,
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
            "gpu_samples": int(len(gpu_rows)),
            "gpu_max_utilization_percent": int(gpu_util.max()),
            "gpu_max_vram_used_mib": int(gpu_memory.max()),
            "model_devices": summary["model_folds"],
            "official_competition_score": "NOT_EVALUATED",
        },
        "decision": (
            "REJECT_ON_SCREENING"
            if candidate_auc <= ref_auc
            else "PROMISING_SCREEN_ONLY_SECOND_SEED_REQUIRED"
        ),
    }
    (run_dir / "category_te_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# Core-Category Nested Target-Encoding Experiment",
        "",
        f"- Run: `{report['experiment_id']}`",
        f"- Decision: `{report['decision']}`",
        "- Same exact five seed-42 folds as frozen Phase 2.",
        "- The target encoder uses nested OOF maps for outer-training rows; validation/test encodings use only outer-training labels.",
        "- No historical champion or test labels were read.",
        "",
        f"Frozen Phase 2 OOF AUC: `{ref_auc:.10f}`",
        f"Core-category TE OOF AUC: `{candidate_auc:.10f}`",
        f"Paired delta: `{candidate_auc - ref_auc:+.10f}`",
        f"Positive folds: `{report['validation']['positive_folds']}/5`",
        "",
        "| Fold | Phase 2 AUC | Category-TE AUC | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_fold:
        markdown.append(
            f"| {row['fold']} | {row['phase2_reference_auc']:.8f} | "
            f"{row['category_te_auc']:.8f} | {row['paired_auc_delta']:+.8f} |"
        )
    markdown.extend(
        [
            "",
            "This is a screening run only. A positive result requires a predeclared independent-seed replication before promotion.",
            f"Candidate SHA-256: `{report['integrity']['candidate_sha256']}`",
            "Official score: **NOT EVALUATED**.",
            "",
        ]
    )
    (run_dir / "category_te_report.md").write_text(
        "\n".join(markdown), encoding="utf-8", newline="\n"
    )

    source_paths = {
        "experiment_driver": Path(__file__).resolve(),
        "category_key_builder": Path(__file__).with_name("phase2_category_keys.py"),
        "runtime_helpers": Path(__file__).with_name("run_group_context_experiment.py"),
        "frozen_phase2_runner": runtime_helpers.PHASE2_SCRIPT,
        "frozen_phase2_oof": PHASE2_REFERENCE / "oof_50_50_blend.npy",
        "frozen_phase2_candidate": frozen_candidate,
    }
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
        description="Screen nested target encodings for core categorical features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
