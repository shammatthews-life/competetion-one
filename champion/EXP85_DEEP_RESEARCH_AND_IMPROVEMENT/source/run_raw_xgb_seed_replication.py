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
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import run_raw_xgb_diversity as screen


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
FROZEN_ROOT = PROJECT_ROOT / "EXP85_PHASE2_FROZEN_RELEASE"
SECOND_SEED_REFERENCE = (
    FROZEN_ROOT / "EVIDENCE" / "second_seed" / "main"
)
PHASE2_OOF_NAME = "oof_phase2_fold_local_blend.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_manifest() -> dict[str, Any]:
    manifest_path = SECOND_SEED_REFERENCE / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["outputs"]
    for relative, entry in expected.items():
        path = SECOND_SEED_REFERENCE / relative
        if not path.is_file():
            raise FileNotFoundError(f"Second-seed reference is missing: {path}")
        if (
            path.stat().st_size != entry["size_bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Second-seed reference hash mismatch: {relative}")
    return manifest


def run(data_dir: Path, run_dir: Path) -> dict[str, Any]:
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
        if sha256_file(path) != screen.EXPECTED_SHA256[name]:
            raise ValueError(f"{name} raw input hash changed")

    reference_manifest = verify_reference_manifest()
    summary_path = SECOND_SEED_REFERENCE / "experiment_summary.json"
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if reference_summary["outer_seed"] != 20261101:
        raise ValueError("Unexpected second-seed Phase 2 fold seed")
    if reference_summary["status"] != "COMPLETED" or reference_summary["decision"] != "PASS":
        raise ValueError("Frozen independent-seed Phase 2 reference is not validated")
    phase2_oof_path = SECOND_SEED_REFERENCE / PHASE2_OOF_NAME
    phase2_oof = np.load(phase2_oof_path, allow_pickle=False).astype(np.float64)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    feature_module = screen.load_feature_builder()
    train_raw = pd.read_csv(raw_paths["train"])
    test_raw = pd.read_csv(raw_paths["test"])
    sample = pd.read_csv(raw_paths["sample"])
    if not sample["SK_ID_CURR"].equals(test_raw["SK_ID_CURR"]):
        raise ValueError("Raw test/sample IDs or ordering differ")
    labels = train_raw["TARGET"].to_numpy(dtype=np.int8)
    train_ids = train_raw["SK_ID_CURR"].to_numpy()
    test_ids = test_raw["SK_ID_CURR"].to_numpy()
    train_features = feature_module.build_rowwise_features(
        train_raw.drop(columns=["TARGET"])
    )
    test_features = feature_module.build_rowwise_features(test_raw)
    if len(phase2_oof) != len(labels):
        raise ValueError("Independent Phase 2 OOF length does not match raw train")
    phase2_auc = float(roc_auc_score(labels, phase2_oof))
    expected_phase2_auc = reference_summary["metrics"]["phase2_fold_local"][
        "oof_blend_auc"
    ]
    if abs(phase2_auc - expected_phase2_auc) > 1e-12:
        raise ValueError("Independent Phase 2 OOF AUC does not match its report")

    monitor_path = run_dir.parent / f".{run_dir.name}.gpu_samples.tmp"
    log_path = run_dir.parent / f".{run_dir.name}.training.log.tmp"
    monitor = screen.runtime_helpers.GpuMonitor(monitor_path)
    tee = screen.runtime_helpers.Tee(sys.stdout, log_path)
    original_stdout = sys.stdout
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    test_fold_predictions = []
    fold_records = []
    process_memory_before = screen.windows_memory_snapshot()
    sys.stdout = tee
    monitor.start()
    try:
        for fold_number in range(1, 6):
            fold_started = time.perf_counter()
            fold_path = SECOND_SEED_REFERENCE / f"fold_{fold_number}_indices.npz"
            with np.load(fold_path, allow_pickle=False) as fold:
                fit_indices = fold["fit_indices"].astype(np.int64)
                valid_indices = fold["valid_indices"].astype(np.int64)
            fit_matrix, valid_matrix, test_matrix, matrix_info = (
                screen.build_fold_matrices(
                    train_features,
                    test_features,
                    fit_indices,
                    valid_indices,
                )
            )
            model = xgb.XGBClassifier(
                **screen.MODEL_CONFIG,
                random_state=20261101 + fold_number,
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
                raise RuntimeError(f"XGBoost silently left CUDA: {actual_device}")
            prediction_range = (0, int(model.best_iteration) + 1)
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
                raise ValueError(f"Fold {fold_number} predictions are invalid")
            oof[valid_indices] = valid_prediction
            test_fold_predictions.append(test_prediction)
            record = {
                "fold": fold_number,
                "fit_rows": int(len(fit_indices)),
                "valid_rows": int(len(valid_indices)),
                "best_iteration": int(model.best_iteration),
                "device": actual_device,
                "auc": float(roc_auc_score(labels[valid_indices], valid_prediction)),
                "seconds": time.perf_counter() - fold_started,
                **matrix_info,
            }
            fold_records.append(record)
            np.savez_compressed(
                run_dir / f"fold_{fold_number}_indices.npz",
                fit_indices=fit_indices.astype(np.int32),
                valid_indices=valid_indices.astype(np.int32),
            )
            print(
                f"[xgboost] independent seed 20261101 fold "
                f"{fold_number}/5 device={actual_device} "
                f"auc={record['auc']:.8f} best_iteration={record['best_iteration']} "
                f"seconds={record['seconds']:.1f}",
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
        raise ValueError("Independent-seed OOF coverage is incomplete")

    xgb_test = np.mean(
        np.stack(test_fold_predictions), axis=0, dtype=np.float64
    ).astype(np.float32)
    xgb_auc = float(roc_auc_score(labels, oof))
    rank_blend = (
        0.80 * screen.rank_percentile(phase2_oof)
        + 0.20 * screen.rank_percentile(oof)
    )
    blend_auc = float(roc_auc_score(labels, rank_blend))
    fold_metrics = []
    for fold_number in range(1, 6):
        reference_fold = SECOND_SEED_REFERENCE / f"fold_{fold_number}_indices.npz"
        with np.load(reference_fold, allow_pickle=False) as expected, np.load(
            run_dir / f"fold_{fold_number}_indices.npz", allow_pickle=False
        ) as actual:
            for name in ("fit_indices", "valid_indices"):
                if not np.array_equal(expected[name], actual[name]):
                    raise ValueError(f"Independent fold {fold_number} changed")
            valid_indices = expected["valid_indices"]
        p2_fold = float(
            roc_auc_score(labels[valid_indices], phase2_oof[valid_indices])
        )
        xgb_fold = float(roc_auc_score(labels[valid_indices], oof[valid_indices]))
        blend_fold = float(
            roc_auc_score(labels[valid_indices], rank_blend[valid_indices])
        )
        fold_metrics.append(
            {
                "fold": fold_number,
                "phase2_auc": p2_fold,
                "xgboost_auc": xgb_fold,
                "xgboost_delta_vs_phase2": xgb_fold - p2_fold,
                "fixed_rank_blend_auc": blend_fold,
                "blend_delta_vs_phase2": blend_fold - p2_fold,
            }
        )

    candidate_path = run_dir / "submission_raw_xgboost_seed20261101.csv"
    candidate = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": xgb_test})
    if (
        len(candidate) != 61500
        or candidate["SK_ID_CURR"].duplicated().any()
        or not candidate["SK_ID_CURR"].equals(test_raw["SK_ID_CURR"])
        or not candidate["SK_ID_CURR"].equals(sample["SK_ID_CURR"])
    ):
        raise ValueError("Independent-seed XGBoost submission integrity failed")
    candidate.to_csv(candidate_path, index=False, float_format="%.10f")

    residual_xgb = labels.astype(np.float64) - oof.astype(np.float64)
    residual_phase2 = labels.astype(np.float64) - phase2_oof
    gpu_rows = pd.read_csv(run_dir / "gpu_samples.csv")
    gpu_util = pd.to_numeric(gpu_rows["utilization.gpu"], errors="raise")
    gpu_memory = pd.to_numeric(gpu_rows["memory.used"], errors="raise")
    if int(gpu_util.max()) <= 0:
        raise RuntimeError("Independent-seed training recorded no GPU activity")
    report = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED_INDEPENDENT_SEED_REPLICATION",
        "lineage": "INDEPENDENT_RAW_ONLY_XGBOOST",
        "outer_seed": 20261101,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "hypothesis": (
            "Replicate the identical raw-only XGBoost configuration and fixed "
            "80/20 rank-blend diagnostic on the frozen independent Phase 2 folds."
        ),
        "feature_generation": {
            "rowwise_and_label_free": True,
            "target_or_id_in_features": False,
            "categorical_encoding": (
                "one-hot fitted on each outer-fit fold; unseen categories ignored"
            ),
            "historical_prediction_inputs": [],
            "auxiliary_tables_used": [],
        },
        "model_config": screen.MODEL_CONFIG,
        "runtime": {
            "xgboost": xgb.__version__,
            "device_by_fold": [row["device"] for row in fold_records],
            "fold_seconds": [row["seconds"] for row in fold_records],
            "process_memory_before": process_memory_before,
            "process_memory_after": screen.windows_memory_snapshot(),
        },
        "validation": {
            "same_exact_folds_as_frozen_independent_phase2_seed": True,
            "phase2_oof_auc": phase2_auc,
            "xgboost_oof_auc": xgb_auc,
            "xgboost_delta_vs_phase2": xgb_auc - phase2_auc,
            "fixed_diagnostic_rank_blend": {
                "weights": {"phase2": 0.80, "xgboost": 0.20},
                "oof_auc": blend_auc,
                "delta_vs_phase2": blend_auc - phase2_auc,
                "weight_unchanged_from_seed42_screen": True,
                "weight_tuned_on_this_oof": False,
            },
            "prediction_pearson_correlation": float(
                np.corrcoef(oof, phase2_oof)[0, 1]
            ),
            "residual_pearson_correlation": float(
                np.corrcoef(residual_xgb, residual_phase2)[0, 1]
            ),
            "squared_error_pearson_correlation": float(
                np.corrcoef(residual_xgb**2, residual_phase2**2)[0, 1]
            ),
            "per_fold": fold_metrics,
            "positive_blend_folds": int(
                sum(row["blend_delta_vs_phase2"] > 0 for row in fold_metrics)
            ),
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
            "REPLICATED_POSITIVE_DIAGNOSTIC_ONLY"
            if blend_auc > phase2_auc
            else "REPLICATION_FAILED"
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
        "# Independent XGBoost Seed Replication",
        "",
        f"- Run: `{report['experiment_id']}`",
        f"- Decision: `{report['decision']}`",
        "- New five-fold seed 20261101; exact frozen Phase 2 fold indices and fold-local OOF reference.",
        "- Same XGBoost configuration, feature builder, categorical handling, and fixed 80/20 diagnostic as the seed-42 screen.",
        "",
        f"Frozen Phase 2 OOF AUC: `{phase2_auc:.10f}`",
        f"Independent XGBoost OOF AUC: `{xgb_auc:.10f}`",
        f"XGBoost delta: `{xgb_auc - phase2_auc:+.10f}`",
        f"Fixed 80/20 rank-blend OOF AUC: `{blend_auc:.10f}`",
        f"Blend delta: `{blend_auc - phase2_auc:+.10f}`",
        f"Positive blend folds: `{report['validation']['positive_blend_folds']}/5`",
        f"Residual correlation: `{report['validation']['residual_pearson_correlation']:.6f}`",
        "",
        "| Fold | Phase 2 AUC | XGBoost AUC | XGB delta | Fixed blend AUC | Blend delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_metrics:
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
            "- Official score: **NOT EVALUATED**.",
            "",
        ]
    )
    (run_dir / "xgboost_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    source_paths = {
        "replication_driver": Path(__file__).resolve(),
        "xgboost_screen_driver": Path(screen.__file__).resolve(),
        "feature_builder": screen.BASELINE_SOURCE,
        "phase2_seed_reference_summary": summary_path,
        "phase2_seed_reference_manifest": (
            SECOND_SEED_REFERENCE / "sha256_manifest.json"
        ),
        "phase2_seed_reference_oof": phase2_oof_path,
    }
    for fold_number in range(1, 6):
        source_paths[f"phase2_seed_reference_fold_{fold_number}"] = (
            SECOND_SEED_REFERENCE / f"fold_{fold_number}_indices.npz"
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
                "phase2_reference_manifest_verified": True,
                "phase2_reference_manifest_files": len(
                    reference_manifest["outputs"]
                ),
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
        description="Replicate the fixed raw XGBoost screen on Phase 2 seed 20261101."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
