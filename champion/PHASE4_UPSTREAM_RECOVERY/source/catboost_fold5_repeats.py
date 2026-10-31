from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


WORKSPACE = Path(r"D:\gci")
RAW_DIR = WORKSPACE / "inpuy"
GEMINI_DIR = WORKSPACE / "gemini"
CHAMPION_DIR = WORKSPACE / "champion"
PHASE3_DIR = CHAMPION_DIR / "EXP85_MASTER_RECOVERY" / "PHASE3_RAW_LINEAGE_RECOVERY"
HISTORICAL_ZOO = GEMINI_DIR / "cache" / "hasan_expanded_zoo.npz"
CANDIDATE_ZOO = (
    CHAMPION_DIR
    / "EXP85_MASTER_RECOVERY"
    / "runs"
    / "anchor74_historical_zoo_20261031_215500_retry1"
    / "historical_anchor_zoo.npz"
)
HISTORICAL_TRAIN_MATRIX = GEMINI_DIR / "cache" / "hasan_selected_550_tr.feather"
HISTORICAL_TEST_MATRIX = GEMINI_DIR / "cache" / "hasan_selected_550_te.feather"
CURRENT_TRAIN_MATRIX = (
    CHAMPION_DIR
    / "EXP85_MASTER_RECOVERY"
    / "runs"
    / "selector_replay_20261031_202500"
    / "hasan_selected_550_tr.feather"
)
CURRENT_TEST_MATRIX = (
    CHAMPION_DIR
    / "EXP85_MASTER_RECOVERY"
    / "runs"
    / "selector_replay_20261031_202500"
    / "hasan_selected_550_te.feather"
)
ROW_MAP_DIR = (
    CHAMPION_DIR
    / "EXP85_MASTER_RECOVERY"
    / "runs"
    / "feature_parity_20261031_200506"
)
REPOSITORY_DIR = WORKSPACE / "_gci_push_tmp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def id_sha256(values: np.ndarray) -> str:
    ids = np.asarray(values, dtype=np.int64).ravel()
    return hashlib.sha256(ids.tobytes()).hexdigest()


def comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.shape != right.shape:
        raise ValueError(f"Prediction shapes differ: {left.shape} != {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Prediction comparisons require finite values.")
    difference = left - right
    rank_displacement = np.abs(
        rankdata(left, method="average") - rankdata(right, method="average")
    )
    return {
        "count": len(left),
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_rank_displacement": float(rank_displacement.mean()),
        "maximum_rank_displacement": float(rank_displacement.max()),
        "exact_value_match": bool(np.array_equal(left, right)),
    }


def restore_permuted_predictions(
    predictions: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    predictions = np.asarray(predictions)
    permutation = np.asarray(permutation, dtype=np.int64)
    expected = np.arange(len(permutation), dtype=np.int64)
    if predictions.ndim != 1 or len(predictions) != len(permutation):
        raise ValueError("Permutation and prediction lengths must match.")
    if not np.array_equal(np.sort(permutation), expected):
        raise ValueError("Row-order check requires a complete permutation.")
    restored = np.empty_like(predictions)
    restored[permutation] = predictions
    return restored


def gpu_snapshot() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def record_id_map(
    train_ids: np.ndarray, validation_ids: np.ndarray, test_ids: np.ndarray, run_dir: Path
) -> dict[str, Any]:
    path = run_dir / "fold5_row_ids.npz"
    np.savez_compressed(
        path,
        training_ids=np.asarray(train_ids, dtype=np.int64),
        validation_ids=np.asarray(validation_ids, dtype=np.int64),
        test_ids=np.asarray(test_ids, dtype=np.int64),
    )
    return {
        "file": file_record(path),
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "test_rows": len(test_ids),
        "training_id_sha256": id_sha256(train_ids),
        "validation_id_sha256": id_sha256(validation_ids),
        "test_id_sha256": id_sha256(test_ids),
    }


def create_model(
    task_type: str, seed: int, train_dir: Path
) -> cb.CatBoostClassifier:
    parameters: dict[str, Any] = {
        "iterations": 750,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "task_type": task_type,
        "random_seed": seed,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "verbose": 0,
        "allow_writing_files": True,
        "train_dir": str(train_dir),
    }
    if task_type == "GPU":
        parameters["devices"] = "0"
    return cb.CatBoostClassifier(**parameters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated same-seed CatBoost fold-5 GPU repeats."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fold-number", type=int, default=5)
    parser.add_argument("--include-cpu", action="store_true")
    args = parser.parse_args()

    if args.repeats < 5:
        raise ValueError("At least five GPU repeats are required.")
    if args.fold_number != 5:
        raise ValueError("This controlled audit is scoped to the known-divergent fold 5.")
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    raw_train_path = RAW_DIR / "train.csv"
    raw_test_path = RAW_DIR / "test.csv"
    train_frame = pd.read_csv(raw_train_path, usecols=["SK_ID_CURR", "TARGET"])
    test_frame = pd.read_csv(raw_test_path, usecols=["SK_ID_CURR"])
    train_ids = train_frame["SK_ID_CURR"].to_numpy(dtype=np.int64)
    test_ids = test_frame["SK_ID_CURR"].to_numpy(dtype=np.int64)
    y = train_frame["TARGET"].to_numpy(dtype=np.int64)
    train_row_map = pd.read_csv(ROW_MAP_DIR / "train_row_id_map.csv")[
        "SK_ID_CURR"
    ].to_numpy(dtype=np.int64)
    test_row_map = pd.read_csv(ROW_MAP_DIR / "test_row_id_map.csv")[
        "SK_ID_CURR"
    ].to_numpy(dtype=np.int64)
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Raw train/test IDs are not unique.")
    if not np.array_equal(train_ids, train_row_map):
        raise ValueError("Raw train IDs do not match the feature row map.")
    if not np.array_equal(test_ids, test_row_map):
        raise ValueError("Raw test IDs do not match the feature row map.")

    x_train = pd.read_feather(CURRENT_TRAIN_MATRIX)
    x_test = pd.read_feather(CURRENT_TEST_MATRIX)
    historical_train = pd.read_feather(HISTORICAL_TRAIN_MATRIX)
    historical_test = pd.read_feather(HISTORICAL_TEST_MATRIX)
    if x_train.shape != (len(train_ids), 550) or x_test.shape != (len(test_ids), 550):
        raise ValueError(f"Unexpected selected matrix sizes: {x_train.shape}, {x_test.shape}")
    if not x_train.columns.equals(x_test.columns):
        raise ValueError("Current train/test feature order differs.")
    if not x_train.columns.equals(historical_train.columns):
        raise ValueError("Historical/current train feature order differs.")
    if not x_test.columns.equals(historical_test.columns):
        raise ValueError("Historical/current test feature order differs.")
    pd.testing.assert_frame_equal(x_train, historical_train, check_exact=True)
    pd.testing.assert_frame_equal(x_test, historical_test, check_exact=True)

    with np.load(HISTORICAL_ZOO, allow_pickle=True) as archive:
        historical_oof = archive["oof_cat"]
        historical_y = archive["y_true"]
    with np.load(CANDIDATE_ZOO, allow_pickle=True) as archive:
        candidate_oof = archive["oof_cat"]
        candidate_y = archive["y_true"]
    if not np.array_equal(historical_y, y) or not np.array_equal(candidate_y, y):
        raise ValueError("Historical/candidate labels differ from raw train TARGET.")
    if not np.isfinite(historical_oof).all() or not np.isfinite(candidate_oof).all():
        raise ValueError("Historical/candidate OOF predictions contain non-finite values.")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(x_train, y))
    fit_indices, valid_indices = folds[args.fold_number - 1]
    fit_ids = train_ids[fit_indices]
    valid_ids = train_ids[valid_indices]
    row_id_record = record_id_map(fit_ids, valid_ids, test_ids, args.run_dir)
    x_fit = x_train.iloc[fit_indices]
    y_fit = y[fit_indices]
    x_valid = x_train.iloc[valid_indices]
    y_valid = y[valid_indices]

    input_records = {
        "raw_train": file_record(raw_train_path),
        "raw_test": file_record(raw_test_path),
        "train_row_id_map": file_record(ROW_MAP_DIR / "train_row_id_map.csv"),
        "test_row_id_map": file_record(ROW_MAP_DIR / "test_row_id_map.csv"),
        "historical_train_matrix": file_record(HISTORICAL_TRAIN_MATRIX),
        "historical_test_matrix": file_record(HISTORICAL_TEST_MATRIX),
        "current_train_matrix": file_record(CURRENT_TRAIN_MATRIX),
        "current_test_matrix": file_record(CURRENT_TEST_MATRIX),
        "historical_zoo": file_record(HISTORICAL_ZOO),
        "candidate_zoo": file_record(CANDIDATE_ZOO),
        "experiment_source": file_record(Path(__file__).resolve()),
    }
    input_records["fold5_row_ids.npz"] = row_id_record["file"]

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(args.run_dir.resolve()),
        "--repeats",
        str(args.repeats),
        "--fold-number",
        str(args.fold_number),
    ]
    if args.include_cpu:
        command.append("--include-cpu")
    commit_result = subprocess.run(
        ["git", "-C", str(REPOSITORY_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "classification": "single-fold CatBoost diagnostic, not raw-rebuild certification",
        "raw_rebuild_certified": False,
        "command": command,
        "working_directory": str(Path.cwd()),
        "repository_commit": commit_result.stdout.strip(),
        "fold_number": args.fold_number,
        "fold_definition": {
            "class": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "model_seed": 46,
        },
        "row_alignment": row_id_record,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "catboost": cb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu_before": gpu_snapshot(),
        },
        "inputs": input_records,
        "feature_matrix": {
            "train_shape": list(x_train.shape),
            "test_shape": list(x_test.shape),
            "feature_count": len(x_train.columns),
            "feature_order_sha256": hashlib.sha256(
                "\n".join(map(str, x_train.columns)).encode("utf-8")
            ).hexdigest(),
            "historical_and_current_train_exact": True,
            "historical_and_current_test_exact": True,
            "raw_id_order_matches_feature_maps": True,
            "labels_match_raw_train": True,
        },
        "gpu_repeats": [],
        "cpu_control": None,
        "elapsed_seconds": None,
    }
    evidence_path = args.run_dir / "execution.json"
    write_json(evidence_path, evidence)

    gpu_predictions: list[dict[str, np.ndarray]] = []
    for repeat in range(1, args.repeats + 1):
        run_name = f"gpu_repeat_{repeat:02d}"
        repeat_dir = args.run_dir / run_name
        repeat_dir.mkdir(parents=False, exist_ok=False)
        telemetry_dir = repeat_dir / "catboost_info"
        model = create_model("GPU", 46, telemetry_dir)
        fit_started = time.monotonic()
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_valid, y_valid),
            early_stopping_rounds=60,
            verbose=0,
        )
        fit_seconds = time.monotonic() - fit_started
        effective = model.get_all_params()
        if effective.get("task_type") != "GPU":
            raise RuntimeError(
                f"Requested GPU but CatBoost reports {effective.get('task_type')!r}."
            )
        validation_prediction = model.predict_proba(x_valid)[:, 1]
        test_prediction = model.predict_proba(x_test)[:, 1]
        permutation = np.random.default_rng(20261101).permutation(len(test_ids))
        shuffled_prediction = model.predict_proba(x_test.iloc[permutation])[:, 1]
        restored_prediction = restore_permuted_predictions(
            shuffled_prediction, permutation
        )

        model_path = repeat_dir / "model.cbm"
        model.save_model(str(model_path), format="cbm")
        prediction_path = repeat_dir / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            validation_prediction=validation_prediction,
            test_prediction=test_prediction,
        )
        run_record = {
            "name": run_name,
            "requested_parameters": {
                "iterations": 750,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "task_type": "GPU",
                "devices": "0",
                "random_seed": 46,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
                "random_strength": 1.0,
                "early_stopping_rounds": 60,
            },
            "effective_parameters": effective,
            "validation_auc": float(roc_auc_score(y_valid, validation_prediction)),
            "historical_oof_comparison": comparison_metrics(
                historical_oof[valid_indices], validation_prediction
            ),
            "candidate_oof_comparison": comparison_metrics(
                candidate_oof[valid_indices], validation_prediction
            ),
            "row_order_invariance": comparison_metrics(
                test_prediction, restored_prediction
            ),
            "best_iteration_zero_based": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            "device_reported_by_model": effective.get("task_type"),
            "fit_elapsed_seconds": fit_seconds,
            "model_file": file_record(model_path),
            "prediction_file": file_record(prediction_path),
            "catboost_info_directory": str(telemetry_dir),
            "gpu_after_fit": gpu_snapshot(),
        }
        evidence["gpu_repeats"].append(run_record)
        gpu_predictions.append(
            {
                "validation": validation_prediction,
                "test": test_prediction,
            }
        )
        evidence["elapsed_seconds"] = time.monotonic() - started
        write_json(evidence_path, evidence)
        print(
            f"{run_name}: AUC={run_record['validation_auc']:.9f}; "
            f"fit={fit_seconds:.2f}s; model_sha256={run_record['model_file']['sha256']}"
        )
        del model

    evidence["pairwise_gpu_comparisons"] = []
    for left_index in range(args.repeats):
        for right_index in range(left_index + 1, args.repeats):
            evidence["pairwise_gpu_comparisons"].append(
                {
                    "left": f"gpu_repeat_{left_index + 1:02d}",
                    "right": f"gpu_repeat_{right_index + 1:02d}",
                    "validation": comparison_metrics(
                        gpu_predictions[left_index]["validation"],
                        gpu_predictions[right_index]["validation"],
                    ),
                    "test": comparison_metrics(
                        gpu_predictions[left_index]["test"],
                        gpu_predictions[right_index]["test"],
                    ),
                }
            )

    evidence["status"] = "GPU_REPEATS_COMPLETE"
    evidence["elapsed_seconds"] = time.monotonic() - started
    write_json(evidence_path, evidence)

    if args.include_cpu:
        cpu_dir = args.run_dir / "cpu_control"
        cpu_dir.mkdir(parents=False, exist_ok=False)
        model = create_model("CPU", 46, cpu_dir / "catboost_info")
        fit_started = time.monotonic()
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_valid, y_valid),
            early_stopping_rounds=60,
            verbose=0,
        )
        fit_seconds = time.monotonic() - fit_started
        effective = model.get_all_params()
        if effective.get("task_type") != "CPU":
            raise RuntimeError(
                f"Requested CPU but CatBoost reports {effective.get('task_type')!r}."
            )
        validation_prediction = model.predict_proba(x_valid)[:, 1]
        test_prediction = model.predict_proba(x_test)[:, 1]
        permutation = np.random.default_rng(20261101).permutation(len(test_ids))
        shuffled_prediction = model.predict_proba(x_test.iloc[permutation])[:, 1]
        restored_prediction = restore_permuted_predictions(
            shuffled_prediction, permutation
        )
        model_path = cpu_dir / "model.cbm"
        model.save_model(str(model_path), format="cbm")
        prediction_path = cpu_dir / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            validation_prediction=validation_prediction,
            test_prediction=test_prediction,
        )
        evidence["cpu_control"] = {
            "requested_parameters": {
                "iterations": 750,
                "learning_rate": 0.04,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "task_type": "CPU",
                "random_seed": 46,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
                "random_strength": 1.0,
                "early_stopping_rounds": 60,
            },
            "effective_parameters": effective,
            "validation_auc": float(roc_auc_score(y_valid, validation_prediction)),
            "historical_oof_comparison": comparison_metrics(
                historical_oof[valid_indices], validation_prediction
            ),
            "candidate_oof_comparison": comparison_metrics(
                candidate_oof[valid_indices], validation_prediction
            ),
            "gpu_repeat_01_comparison": {
                "validation": comparison_metrics(
                    gpu_predictions[0]["validation"], validation_prediction
                ),
                "test": comparison_metrics(
                    gpu_predictions[0]["test"], test_prediction
                ),
            },
            "row_order_invariance": comparison_metrics(
                test_prediction, restored_prediction
            ),
            "best_iteration_zero_based": int(model.get_best_iteration()),
            "tree_count": int(model.tree_count_),
            "device_reported_by_model": effective.get("task_type"),
            "fit_elapsed_seconds": fit_seconds,
            "model_file": file_record(model_path),
            "prediction_file": file_record(prediction_path),
            "catboost_info_directory": str(cpu_dir / "catboost_info"),
        }
        evidence["status"] = "GPU_REPEATS_AND_CPU_CONTROL_COMPLETE"
        evidence["elapsed_seconds"] = time.monotonic() - started
        write_json(evidence_path, evidence)
        del model

    print(f"Evidence: {evidence_path}")


if __name__ == "__main__":
    main()
