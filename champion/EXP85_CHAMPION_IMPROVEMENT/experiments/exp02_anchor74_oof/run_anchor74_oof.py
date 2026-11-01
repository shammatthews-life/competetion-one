from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
import traceback
from csv import DictReader
from datetime import datetime, timezone
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


WORKSPACE = Path(__file__).resolve().parents[2]
REBUILT_ROOT = WORKSPACE.parent / "02_EXP85_REBUILT"
ANCHOR_BUILDER = REBUILT_ROOT / "source" / "generate_anchor74_from_scratch.py"
RAW_DIR = Path(r"D:\gci\inpuy")
BASELINE_ARTIFACTS = WORKSPACE / "baseline" / "artifacts"
RUNS_DIR = WORKSPACE / "experiments" / "runs"
SEED = 20261101
N_FOLDS = 5
N_FEATURES = 695
THREADS = 4
FAMILY_WEIGHTS = np.asarray([0.22, 0.18, 0.24, 0.20, 0.16], dtype=np.float64)
FAMILY_NAMES = ("lgb_cons", "lgb_et", "xgb", "cat", "lgb_deep")
SUITE_MEMBERS = (
    "v17_e1Aet",
    "xgb10",
    "v17_final",
    "v55_cbprod",
    "v17_e1et",
    "v55_deep_et",
    "v55_allfam",
    "v55_v2_mono",
)
SUITE_WEIGHTS = np.asarray(
    [0.2932, 0.2596, 0.2051, 0.0874, 0.0489, 0.0442, 0.0383, 0.0233],
    dtype=np.float64,
)
BLEND_ANCHOR_WEIGHTS = (0.80, 0.85, 0.90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_probit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Rank-probit input must be nonempty and finite")
    uniform = rankdata(values, method="average") / (len(values) + 1.0)
    return norm.ppf(np.clip(uniform, 1e-6, 1.0 - 1e-6))


def tail_sharpen(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    high = result > 1.28
    low = result < -1.28
    result[high] = 1.28 + np.power(result[high] - 1.28, 1.025)
    result[low] = -1.28 - np.power(np.abs(result[low]) - 1.28, 1.010)
    return result


def compute_regime_delta(z_models: np.ndarray, missing_count: np.ndarray) -> np.ndarray:
    if z_models.ndim != 2 or z_models.shape[1] != 5:
        raise ValueError("Expected five model streams in z_models")
    if len(missing_count) != len(z_models):
        raise ValueError("Missingness vector and model streams are misaligned")
    z_cons, z_et, z_xgb, z_cat, z_deep = z_models.T
    result = np.zeros(len(z_models), dtype=np.float64)
    regime0 = missing_count == 0
    regime1 = missing_count == 1
    regime2 = missing_count >= 2
    result[regime0] = (
        0.60 * (z_cat[regime0] - z_cons[regime0])
        + 0.40 * (z_deep[regime0] - z_cons[regime0])
    )
    result[regime1] = (
        0.50 * (z_cons[regime1] - z_et[regime1])
        + 0.50 * (z_cons[regime1] - z_deep[regime1])
    )
    result[regime2] = (
        0.70 * (z_cat[regime2] - z_xgb[regime2])
        + 0.30 * (z_cat[regime2] - z_cons[regime2])
    )
    return result


def _entity_graph_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "DAYS_EMPLOYED",
        "DAYS_LAST_PHONE_CHANGE",
    ]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw graph inputs are missing columns: {sorted(missing)}")
    result = frame[columns].copy()
    result["entity_key"] = (
        (result["DAYS_BIRTH"] - result["DAYS_REGISTRATION"])
        .astype(int)
        .astype(str)
        + "|"
        + (result["DAYS_BIRTH"] - result["DAYS_ID_PUBLISH"])
        .astype(int)
        .astype(str)
        + "|"
        + result["CODE_GENDER"].astype(str)
        + "|"
        + result["NAME_EDUCATION_TYPE"].astype(str)
    )
    return result


def entity_group_ids(frame: pd.DataFrame) -> np.ndarray:
    return _entity_graph_inputs(frame)["entity_key"].to_numpy()


def make_grouped_splits(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("Rows, labels, and group IDs must have equal lengths")
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    splits = list(splitter.split(x, y, groups))
    for fold_number, (fit_idx, valid_idx) in enumerate(splits, start=1):
        overlap = set(groups[fit_idx]) & set(groups[valid_idx])
        if overlap:
            raise ValueError(
                f"Borrower/entity group leakage detected in fold {fold_number}"
            )
    return splits


def calibrate_fold(
    z_base: np.ndarray,
    z_models: np.ndarray,
    missing_count: np.ndarray,
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    fit_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the historical-style gates with fold-local label access only."""
    if len(z_base) != len(valid_frame) or len(z_models) != len(valid_frame):
        raise ValueError("Fold prediction and validation rows are misaligned")
    if len(fit_frame) != len(fit_labels):
        raise ValueError("Fold training rows and labels are misaligned")

    z_disp = np.std(z_models, axis=1)
    spread = float(np.std(z_disp))
    if spread == 0.0:
        gate = np.zeros(len(z_disp), dtype=np.float64)
    else:
        threshold = float(np.percentile(z_disp, 80.0))
        gate = expit((z_disp - threshold) / (spread * 0.35))
    delta = compute_regime_delta(z_models, missing_count)

    z_step = np.asarray(z_base, dtype=np.float64).copy()
    z_step += (0.045 + 0.080) * gate * delta

    graph_frame = pd.concat([fit_frame, valid_frame], ignore_index=True)
    test_to_train, test_to_test = _load_anchor_builder().build_entity_graph(
        graph_frame,
        len(fit_frame),
        len(valid_frame),
        np.asarray(fit_labels, dtype=np.int64),
    )

    z_cal = z_step.copy()
    if not test_to_train.empty:
        means = test_to_train.groupby("test_idx")["train_target"].mean()
        for row_idx, mean_target in means.items():
            idx = int(row_idx)
            if mean_target >= 0.5:
                z_cal[idx] = 0.88 * z_cal[idx] + 0.12 * 2.50
            elif mean_target == 0.0:
                z_cal[idx] = 0.88 * z_cal[idx] + 0.12 * -1.30

    z_diffused = z_cal.copy()
    for row_idx, group in test_to_test.groupby("test_idx"):
        weights = group["weight"].to_numpy(dtype=np.float64)
        partners = group["partner_test_idx"].to_numpy(dtype=np.int64)
        partner_mean = np.average(z_cal[partners], weights=weights)
        z_diffused[int(row_idx)] = 0.85 * z_cal[int(row_idx)] + 0.15 * partner_mean

    z_final = z_diffused.copy()
    for row_idx, group in test_to_test.groupby("test_idx"):
        weights = group["weight"].to_numpy(dtype=np.float64)
        partners = group["partner_test_idx"].to_numpy(dtype=np.int64)
        partner_mean = np.average(z_diffused[partners], weights=weights)
        dz = np.clip(0.12 * (partner_mean - z_diffused[int(row_idx)]), -0.08, 0.08)
        z_final[int(row_idx)] = z_diffused[int(row_idx)] + dz

    z_final = z_base + np.clip(z_final - z_base, -0.35, 0.35)
    return z_final, {
        "train_to_validation_pairs": int(len(test_to_train)),
        "validation_to_validation_pairs": int(len(test_to_test)),
    }


_ANCHOR_MODULE = None


def _load_anchor_builder():
    global _ANCHOR_MODULE
    if _ANCHOR_MODULE is None:
        if not ANCHOR_BUILDER.is_file():
            raise FileNotFoundError(f"Required feature builder is missing: {ANCHOR_BUILDER}")
        spec = importlib.util.spec_from_file_location("exp85_anchor_builder", ANCHOR_BUILDER)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load feature builder: {ANCHOR_BUILDER}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ANCHOR_MODULE = module
    return _ANCHOR_MODULE


def _gpu_preflight(run_dir: Path) -> dict[str, object]:
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if gpu_info.returncode != 0:
        raise RuntimeError(f"nvidia-smi preflight failed: {gpu_info.stderr}")

    sample_x = np.asarray(
        [[0.0, 0.1], [1.0, 0.9], [0.2, 0.8], [0.9, 0.1]] * 20,
        dtype=np.float32,
    )
    sample_y = np.asarray([0, 1, 0, 1] * 20, dtype=np.int32)
    xgb_model = xgb.XGBClassifier(
        n_estimators=2,
        max_depth=2,
        tree_method="hist",
        device="cuda",
        n_jobs=THREADS,
        eval_metric="logloss",
        verbosity=1,
    )
    xgb_model.fit(sample_x, sample_y, verbose=True)
    xgb_config = json.loads(xgb_model.get_booster().save_config())
    xgb_device = xgb_config["learner"]["generic_param"]["device"]
    if not xgb_device.startswith("cuda"):
        raise RuntimeError(f"XGBoost did not confirm CUDA execution: {xgb_device}")

    cat_model = CatBoostClassifier(
        iterations=2,
        depth=2,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
        verbose=1,
    )
    cat_model.fit(sample_x, sample_y)
    cat_task = cat_model.get_all_params().get("task_type")
    if cat_task != "GPU":
        raise RuntimeError(f"CatBoost did not confirm GPU execution: {cat_task}")
    return {
        "nvidia_smi": gpu_info.stdout.strip(),
        "xgboost_device": xgb_device,
        "xgboost_version": xgb.__version__,
        "catboost_task_type": cat_task,
        "catboost_version": catboost.__version__,
        "lightgbm_version": lgb.__version__,
        "lightgbm_backend": "CPU; installed LightGBM build is CPU-based",
    }


class GpuTelemetry:
    def __init__(self, path: Path):
        self.path = path
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "--loop=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def drain() -> None:
            assert self.process is not None and self.process.stdout is not None
            with self.path.open("w", encoding="utf-8", newline="") as output:
                output.write("timestamp,utilization_gpu_pct,memory_used_mib,memory_free_mib\n")
                for line in self.process.stdout:
                    output.write(line)
                    output.flush()

        self.thread = threading.Thread(target=drain, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        if self.thread is not None:
            self.thread.join(timeout=10)
        if self.process.stderr is not None:
            error_text = self.process.stderr.read().strip()
            if error_text and self.process.returncode not in (0, 1):
                self.error = error_text


def _load_suite_oof(labels: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    artifacts = {
        "v17": BASELINE_ARTIFACTS / "v_17_full_suite_bundle.npz",
        "v55": BASELINE_ARTIFACTS / "v55_members.npz",
        "xgb": BASELINE_ARTIFACTS / "oof_10f_xgb.npy",
        "labels": BASELINE_ARTIFACTS / "y_train.npy",
    }
    if any(not path.is_file() for path in artifacts.values()):
        raise FileNotFoundError("Required frozen suite OOF artifacts are missing")
    manifest_path = WORKSPACE / "baseline" / "BASELINE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen baseline manifest is missing: {manifest_path}")
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash_keys = {
        "v17": "v17",
        "v55": "v55",
        "xgb": "xgb10_oof",
        "labels": "labels",
    }
    for artifact_key, manifest_key in expected_hash_keys.items():
        expected = baseline_manifest["direct_synthesis_inputs"][manifest_key]["sha256"]
        actual = sha256_file(artifacts[artifact_key])
        if actual != expected:
            raise ValueError(f"Frozen suite artifact hash mismatch: {artifact_key}")
    if not np.array_equal(
        np.load(artifacts["labels"], allow_pickle=False).astype(np.int32), labels
    ):
        raise ValueError("Frozen suite labels differ from raw train.csv")
    with np.load(artifacts["v17"], allow_pickle=False) as v17:
        members = {
            "v17_e1Aet": v17["oof_e1Aet"],
            "v17_final": v17["final_oof"],
            "v17_e1et": v17["oof_e1et"],
        }
    with np.load(artifacts["v55"], allow_pickle=False) as v55:
        members.update(
            {
                "v55_cbprod": v55["oof_cb_product"],
                "v55_deep_et": v55["oof_deep_et"],
                "v55_allfam": v55["oof_v2_allfam_oh"],
                "v55_v2_mono": v55["oof_v2_mono"],
            }
        )
    members["xgb10"] = np.load(artifacts["xgb"], allow_pickle=False)
    matrix = np.column_stack([rank_probit(members[name]) for name in SUITE_MEMBERS])
    if matrix.shape != (len(labels), len(SUITE_MEMBERS)):
        raise ValueError(f"Unexpected suite OOF shape: {matrix.shape}")
    normalized_weights = SUITE_WEIGHTS / SUITE_WEIGHTS.sum()
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    return matrix @ normalized_weights, hashes


def _fold_auc_deltas(
    fold_auc_by_blend: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    baseline = fold_auc_by_blend["0.85"]
    deltas = {
        weight: [float(candidate - base) for candidate, base in zip(values, baseline)]
        for weight, values in fold_auc_by_blend.items()
    }
    positive = {
        weight: int(sum(delta > 0.0 for delta in values))
        for weight, values in deltas.items()
    }
    return deltas, positive


def _make_model_predictions(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    fold_number: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    predictions: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.60,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "min_child_samples": 80,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "verbosity": -1,
        "num_threads": THREADS,
        "seed": 42,
        "feature_fraction_seed": 42 + fold_number,
        "bagging_seed": 42 + fold_number,
        "data_random_seed": 42 + fold_number,
        "force_col_wise": True,
    }
    started = time.perf_counter()
    dtrain = lgb.Dataset(x_train, label=y_train)
    cons = lgb.train(lgb_params, dtrain, num_boost_round=700)
    predictions["lgb_cons"] = cons.predict(x_valid)
    runtimes["lgb_cons"] = time.perf_counter() - started
    del cons, dtrain

    et_params = {
        **lgb_params,
        "extra_trees": True,
        "learning_rate": 0.035,
        "num_leaves": 45,
        "max_depth": 7,
        "feature_fraction": 0.45,
        "min_child_samples": 60,
        "reg_alpha": 3.0,
        "reg_lambda": 8.0,
        "seed": 43,
        "feature_fraction_seed": 43 + fold_number,
        "bagging_seed": 43 + fold_number,
        "data_random_seed": 43 + fold_number,
    }
    started = time.perf_counter()
    dtrain = lgb.Dataset(x_train, label=y_train)
    et = lgb.train(et_params, dtrain, num_boost_round=750)
    predictions["lgb_et"] = et.predict(x_valid)
    runtimes["lgb_et"] = time.perf_counter() - started
    del et, dtrain

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=5.0,
        tree_method="hist",
        device="cuda",
        eval_metric="auc",
        random_state=44,
        n_jobs=THREADS,
        verbosity=1,
    )
    started = time.perf_counter()
    xgb_model.fit(x_train, y_train, verbose=True)
    predictions["xgb"] = xgb_model.predict_proba(x_valid)[:, 1]
    runtimes["xgb"] = time.perf_counter() - started

    cat_model = CatBoostClassifier(
        iterations=450,
        depth=6,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        task_type="GPU",
        devices="0",
        random_seed=45,
        allow_writing_files=False,
        verbose=100,
    )
    started = time.perf_counter()
    cat_model.fit(x_train, y_train, verbose=100)
    predictions["cat"] = cat_model.predict_proba(x_valid)[:, 1]
    runtimes["cat"] = time.perf_counter() - started

    deep_params = {
        **lgb_params,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.50,
        "min_child_samples": 50,
        "reg_alpha": 2.5,
        "reg_lambda": 6.0,
        "seed": 46,
        "feature_fraction_seed": 46 + fold_number,
        "bagging_seed": 46 + fold_number,
        "data_random_seed": 46 + fold_number,
    }
    started = time.perf_counter()
    dtrain = lgb.Dataset(x_train, label=y_train)
    deep = lgb.train(deep_params, dtrain, num_boost_round=750)
    predictions["lgb_deep"] = deep.predict(x_valid)
    runtimes["lgb_deep"] = time.perf_counter() - started
    del deep, dtrain

    for name in FAMILY_NAMES:
        pred = predictions[name]
        if pred.shape != (len(x_valid),) or not np.isfinite(pred).all():
            raise ValueError(f"{name} produced invalid fold-{fold_number} predictions")
    return predictions, runtimes


def main() -> None:
    started = time.perf_counter()
    if not ANCHOR_BUILDER.is_file():
        raise FileNotFoundError(f"Raw feature builder not found: {ANCHOR_BUILDER}")
    raw_train_path = RAW_DIR / "train.csv"
    raw_test_path = RAW_DIR / "test.csv"
    if not all(path.is_file() for path in (raw_train_path, raw_test_path)):
        raise FileNotFoundError("Raw train/test CSV inputs are missing")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"{run_id}_exp02_anchor74_oof_surrogate"
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "training.log"
    log_stream = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = log_stream
    sys.stderr = log_stream
    telemetry = GpuTelemetry(run_dir / "gpu_telemetry.csv")
    exit_code = 1
    report: dict[str, object] = {}

    try:
        print(f"Experiment: {run_id}")
        print(f"Classification: INDEPENDENT_RAW_DATA_SURROGATE_OOF")
        print(f"Python: {sys.executable} | {sys.version}")
        gpu_info = _gpu_preflight(run_dir)
        telemetry.start()

        builder = _load_anchor_builder()
        train_df, test_df = builder.load_raw_data(data_dir=RAW_DIR)
        y = train_df["TARGET"].to_numpy(dtype=np.int32)
        x_train, _, builder_y, test_ids, full_raw = builder.build_full_hasan_features(
            train_df, test_df, random_state=SEED
        )
        if not np.array_equal(y, builder_y):
            raise ValueError("Feature builder changed training label ordering")
        if len(x_train) != 171_202 or len(test_ids) != 61_500:
            raise ValueError("Unexpected raw competition dimensions")
        if not np.array_equal(x_train.index.to_numpy(), np.arange(len(train_df))):
            raise ValueError("Feature builder changed training row ordering")
        if not np.array_equal(test_ids, test_df["SK_ID_CURR"].to_numpy()):
            raise ValueError("Feature builder changed test ID ordering")
        if "TARGET" in x_train.columns:
            raise ValueError("Target column leaked into the feature matrix")
        if x_train.shape[1] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {x_train.shape[1]}")
        if not train_df["SK_ID_CURR"].is_unique or not test_df["SK_ID_CURR"].is_unique:
            raise ValueError("Raw train/test IDs are not unique")
        suite_oof, suite_hashes = _load_suite_oof(y)
        baseline_manifest = json.loads(
            (WORKSPACE / "baseline" / "BASELINE_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        for name, path in (("train.csv", raw_train_path), ("test.csv", raw_test_path)):
            expected_hash = baseline_manifest["raw_inputs_hashed_not_copied"][name]["sha256"]
            if sha256_file(path) != expected_hash:
                raise ValueError(f"Raw input hash differs from frozen baseline: {name}")

        groups = entity_group_ids(train_df)
        splits = make_grouped_splits(x_train, y, groups)
        fold_membership = np.full(len(y), -1, dtype=np.int8)
        family_oof = {
            name: np.full(len(y), np.nan, dtype=np.float64)
            for name in FAMILY_NAMES
        }
        fold_reports = []
        feature_names = list(x_train.columns)
        print(
            f"Raw rows={len(y)} test_rows={len(test_df)} features={len(feature_names)} "
            f"outer_folds={N_FOLDS} seed={SEED} groups={len(np.unique(groups))}"
        )
        print("Outer validation labels are not passed to any fit/early-stopping call.")
        print("StratifiedGroupKFold isolates all exact applicant entity keys.")

        for fold_number, (fit_idx, valid_idx) in enumerate(splits, start=1):
            fold_started = time.perf_counter()
            if np.intersect1d(fit_idx, valid_idx).size:
                raise RuntimeError(f"Fold {fold_number} fit/validation overlap")
            group_overlap = set(groups[fit_idx]) & set(groups[valid_idx])
            if group_overlap:
                raise RuntimeError(f"Fold {fold_number} contains borrower group leakage")
            fit_x = x_train.iloc[fit_idx].replace([np.inf, -np.inf], np.nan)
            valid_x = x_train.iloc[valid_idx].replace([np.inf, -np.inf], np.nan)
            fold_predictions, model_runtimes = _make_model_predictions(
                fit_x, y[fit_idx], valid_x, fold_number
            )
            for name, values in fold_predictions.items():
                family_oof[name][valid_idx] = values
            fold_membership[valid_idx] = fold_number - 1
            family_auc = {
                name: float(roc_auc_score(y[valid_idx], fold_predictions[name]))
                for name in FAMILY_NAMES
            }
            fold_reports.append(
                {
                    "fold": fold_number,
                    "fit_rows": int(len(fit_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "seed": SEED,
                    "model_auc": family_auc,
                    "model_runtime_seconds": model_runtimes,
                    "elapsed_seconds": time.perf_counter() - fold_started,
                }
            )
            print(
                f"Fold {fold_number}/{N_FOLDS}: "
                + ", ".join(f"{key}={value:.6f}" for key, value in family_auc.items())
            )
            print(f"Fold {fold_number} fit_rows={len(fit_idx)} valid_rows={len(valid_idx)}")

        if (fold_membership < 0).any():
            raise RuntimeError("Outer folds did not cover every training row")
        for name, values in family_oof.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"OOF coverage or finiteness failure: {name}")

        family_matrix_z = np.column_stack(
            [rank_probit(family_oof[name]) for name in FAMILY_NAMES]
        )
        z_zoo = family_matrix_z @ FAMILY_WEIGHTS
        z_base = tail_sharpen(z_zoo)
        missing_count = train_df[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].isna().sum(axis=1).to_numpy()
        anchor_oof = np.full(len(y), np.nan, dtype=np.float64)
        graph_counts = []
        graph_columns = [
            "DAYS_BIRTH",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "CODE_GENDER",
            "NAME_EDUCATION_TYPE",
            "DAYS_EMPLOYED",
            "DAYS_LAST_PHONE_CHANGE",
        ]
        fold_auc_by_blend = {str(w): [] for w in BLEND_ANCHOR_WEIGHTS}
        anchor_fold_auc = []
        z_anchor_global = np.full(len(y), np.nan, dtype=np.float64)
        for fold_number, (_, valid_idx) in enumerate(splits, start=1):
            z_fold, graph_info = calibrate_fold(
                z_base[valid_idx],
                family_matrix_z[valid_idx],
                missing_count[valid_idx],
                full_raw.iloc[splits[fold_number - 1][0]][graph_columns],
                full_raw.iloc[valid_idx][graph_columns],
                y[splits[fold_number - 1][0]],
            )
            z_anchor_global[valid_idx] = z_fold
            graph_counts.append({"fold": fold_number, **graph_info})
            anchor_fold_auc.append(
                float(roc_auc_score(y[valid_idx], z_fold))
            )

        if not np.isfinite(z_anchor_global).all():
            raise RuntimeError("Calibrated surrogate OOF did not cover all rows")
        anchor_oof[:] = rank_probit(z_anchor_global)
        historical_ratio_scores = (
            0.85 * anchor_oof + 0.15 * suite_oof
        )
        meta_oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold_number, (fit_idx, valid_idx) in enumerate(splits, start=1):
            stack = np.column_stack([anchor_oof, suite_oof])
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=SEED + fold_number,
                ),
            )
            model.fit(stack[fit_idx], y[fit_idx])
            meta_oof[valid_idx] = model.decision_function(stack[valid_idx])
        if not np.isfinite(meta_oof).all():
            raise RuntimeError("Cross-fit two-stream meta OOF incomplete")
        base_blend_by_fold = []
        for fold_number, (_, valid_idx) in enumerate(splits, start=1):
            metrics = {
                "fold": fold_number,
                "surrogate_anchor_auc": anchor_fold_auc[fold_number - 1],
                "suite_oof_auc": float(roc_auc_score(y[valid_idx], suite_oof[valid_idx])),
                "historical_ratio_surrogate_blend_auc": float(
                    roc_auc_score(y[valid_idx], historical_ratio_scores[valid_idx])
                ),
                "two_stream_meta_auc": float(
                    roc_auc_score(y[valid_idx], meta_oof[valid_idx])
                ),
            }
            for anchor_weight in BLEND_ANCHOR_WEIGHTS:
                blend = (
                    anchor_weight * anchor_oof[valid_idx]
                    + (1.0 - anchor_weight) * suite_oof[valid_idx]
                )
                fold_auc = float(roc_auc_score(y[valid_idx], blend))
                fold_auc_by_blend[str(anchor_weight)].append(fold_auc)
                metrics[f"anchor_{anchor_weight:.2f}_suite_{1-anchor_weight:.2f}_auc"] = fold_auc
            base_blend_by_fold.append(metrics)
        fold_deltas, positive_folds = _fold_auc_deltas(fold_auc_by_blend)

        report = {
            "experiment_id": run_id,
            "classification": "INDEPENDENT_RAW_DATA_SURROGATE_OOF",
            "historical_anchor74_claim": False,
            "status": "DIAGNOSTIC_ONLY_NO_CANDIDATE",
            "reason_no_candidate": (
                "This is a nonhistorical raw-data surrogate; a verified, leakage-safe "
                "historical Anchor74 OOF baseline is unavailable."
            ),
            "official_score": "NOT_EVALUATED",
            "configuration": {
                "outer_folds": N_FOLDS,
                "shuffle": True,
                "seed": SEED,
                "splitter": "StratifiedGroupKFold",
                "split_deviation": (
                    "Uses exact historical applicant entity key as groups; "
                    "row-wise seed-matched folds were rejected to prevent borrower leakage."
                ),
                "unique_entity_groups": int(len(np.unique(groups))),
                "multirow_entity_groups": int(
                    sum(count > 1 for count in pd.Series(groups).value_counts())
                ),
                "group_overlap_by_fold": [
                    int(len(set(groups[fit_idx]) & set(groups[valid_idx])))
                    for fit_idx, valid_idx in splits
                ],
                "feature_count": int(x_train.shape[1]),
                "features": (
                    "Tracked raw builder returns 695 columns despite its 143-feature "
                    "header; all transformations are covariate-only and transductive."
                ),
                "outer_validation_labels_used_for_fit": False,
                "early_stopping": "Disabled; fixed iteration counts prevent validation-label selection.",
                "feature_selection": "None; no global-label-selected feature list used.",
                "family_names": list(FAMILY_NAMES),
                "family_weights": FAMILY_WEIGHTS.tolist(),
                "model_parameters": {
                    "lgb_cons": {
                        "rounds": 700, "learning_rate": 0.03, "num_leaves": 31,
                        "max_depth": 6, "feature_fraction": 0.60,
                        "bagging_fraction": 0.80, "min_child_samples": 80,
                        "reg_alpha": 2.0, "reg_lambda": 5.0, "threads": THREADS,
                        "seed": 42, "fold_seed_offset": "feature/bagging/data=42+fold",
                    },
                    "lgb_et": {
                        "rounds": 750, "extra_trees": True, "learning_rate": 0.035,
                        "num_leaves": 45, "max_depth": 7, "feature_fraction": 0.45,
                        "bagging_fraction": 0.80, "min_child_samples": 60,
                        "reg_alpha": 3.0, "reg_lambda": 8.0, "threads": THREADS,
                        "seed": 43, "fold_seed_offset": "feature/bagging/data=43+fold",
                    },
                    "xgb": {
                        "rounds": 500, "learning_rate": 0.04, "max_depth": 6,
                        "subsample": 0.8, "colsample_bytree": 0.6,
                        "reg_alpha": 2.0, "reg_lambda": 5.0,
                        "tree_method": "hist", "device": "cuda",
                        "random_state": 44, "threads": THREADS,
                    },
                    "cat": {
                        "iterations": 450, "depth": 6, "learning_rate": 0.045,
                        "loss_function": "Logloss", "eval_metric": "AUC",
                        "l2_leaf_reg": 5.0, "task_type": "GPU", "devices": "0",
                        "random_seed": 45,
                    },
                    "lgb_deep": {
                        "rounds": 750, "learning_rate": 0.03, "num_leaves": 63,
                        "max_depth": 8, "feature_fraction": 0.50,
                        "bagging_fraction": 0.80, "min_child_samples": 50,
                        "reg_alpha": 2.5, "reg_lambda": 6.0, "threads": THREADS,
                        "seed": 46, "fold_seed_offset": "feature/bagging/data=46+fold",
                    },
                },
                "blend_anchor_weights_tested": list(BLEND_ANCHOR_WEIGHTS),
                "blend_suite_weights": {
                    f"{w:.2f}": round(1.0 - w, 2) for w in BLEND_ANCHOR_WEIGHTS
                },
                "primary_ratio": [0.85, 0.15],
                "gpu": gpu_info,
                "lgbm_cpu_reason": "Installed LightGBM build is CPU-based.",
            },
            "provenance": {
                "anchor_builder_path": str(ANCHOR_BUILDER),
                "anchor_builder_sha256": sha256_file(ANCHOR_BUILDER),
                "experiment_source_path": str(Path(__file__).resolve()),
                "experiment_source_sha256": sha256_file(Path(__file__).resolve()),
                "command": [sys.executable, str(Path(__file__).resolve())],
                "fold_indices_saved": True,
                "feature_cache_used": False,
                "historical_test_predictions_used": False,
                "raw_train_sha256": sha256_file(raw_train_path),
                "raw_test_sha256": sha256_file(raw_test_path),
                "suite_cache_hashes": suite_hashes,
                "suite_cache_limitation": (
                    "Historical suite OOF fold sidecars and producer metadata are unavailable."
                ),
                "source_notes": [
                    "This is not the historical Anchor74 architecture: its 143-feature "
                    "header yields 695 columns and differs from the original 669/550-feature zoo and "
                    "upstream optimal-transductive stack.",
                    "The historical Anchor74 source applies a cached optimal-apex "
                    "prediction; no verified OOF for that upstream stage was found.",
                    "This surrogate's OOF predictions are independently trained from raw "
                    "data; its 85:15 suite blend is hybrid-cache-assisted diagnostic only.",
                    "Unsupervised feature construction includes train/test covariates; "
                    "no target labels are used by the feature builder.",
                    "All graph label calibration is refit per fold using outer-fit labels only.",
                ],
            },
            "folds": fold_reports,
            "graph_calibration": graph_counts,
            "oof_metrics": {
                "family_pooled_auc": {
                    name: float(roc_auc_score(y, family_oof[name]))
                    for name in FAMILY_NAMES
                },
                "surrogate_anchor_fold_auc": anchor_fold_auc,
                "suite_pooled_auc": float(roc_auc_score(y, suite_oof)),
                "surrogate_anchor_pooled_auc": float(roc_auc_score(y, anchor_oof)),
                "historical_ratio_surrogate_blend_pooled_auc": float(
                    roc_auc_score(y, historical_ratio_scores)
                ),
                "two_stream_crossfit_logistic_pooled_auc": float(
                    roc_auc_score(y, meta_oof)
                ),
                "blend_fold_auc": base_blend_by_fold,
                "blend_mean_auc": {
                    weight: float(np.mean(values))
                    for weight, values in fold_auc_by_blend.items()
                },
                "fold_delta_vs_85_15_surrogate": {
                    weight: values for weight, values in fold_deltas.items()
                },
                "positive_folds_vs_85_15_surrogate": positive_folds,
            },
            "runtime_seconds_before_report": time.perf_counter() - started,
        }

        np.savez_compressed(
            run_dir / "raw_family_oof.npz",
            **{name: values for name, values in family_oof.items()},
            fold_membership=fold_membership,
            **{
                f"fold_{fold_number}_fit_idx": fit_idx
                for fold_number, (fit_idx, _) in enumerate(splits, start=1)
            },
            **{
                f"fold_{fold_number}_valid_idx": valid_idx
                for fold_number, (_, valid_idx) in enumerate(splits, start=1)
            },
        )
        np.save(run_dir / "anchor_surrogate_oof.npy", anchor_oof, allow_pickle=False)
        np.save(run_dir / "suite_oof_rank_probit.npy", suite_oof, allow_pickle=False)
        np.save(run_dir / "blend_85_15_oof.npy", historical_ratio_scores, allow_pickle=False)
        np.save(run_dir / "two_stream_meta_oof.npy", meta_oof, allow_pickle=False)
        report["outputs"] = {
            path.name: sha256_file(path)
            for path in run_dir.iterdir()
            if path.is_file() and path.name not in {"report.json", "training.log", "gpu_telemetry.csv"}
        }
        report["runtime_seconds"] = time.perf_counter() - started
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        exit_code = 0
        print(json.dumps(report, indent=2))
    except Exception as error:
        report = {
            "experiment_id": run_id,
            "classification": "INDEPENDENT_RAW_DATA_SURROGATE_OOF",
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - started,
            "provenance": {
                "anchor_builder_path": str(ANCHOR_BUILDER),
                "anchor_builder_sha256": (
                    sha256_file(ANCHOR_BUILDER) if ANCHOR_BUILDER.is_file() else None
                ),
                "raw_train_sha256": (
                    sha256_file(raw_train_path) if raw_train_path.is_file() else None
                ),
                "raw_test_sha256": (
                    sha256_file(raw_test_path) if raw_test_path.is_file() else None
                ),
                "outer_validation_labels_used_for_fit": False,
            },
        }
        (run_dir / "failure_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        raise
    finally:
        telemetry.stop()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_stream.close()
        report_path = run_dir / "report.json"
        if report_path.is_file():
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            telemetry_path = run_dir / "gpu_telemetry.csv"
            telemetry_rows = []
            if telemetry_path.is_file():
                with telemetry_path.open(encoding="utf-8", newline="") as stream:
                    telemetry_rows = list(DictReader(stream))
            util_values = [
                int(row["utilization_gpu_pct"])
                for row in telemetry_rows
                if row.get("utilization_gpu_pct", "").strip().isdigit()
            ]
            memory_values = [
                int(row["memory_used_mib"])
                for row in telemetry_rows
                if row.get("memory_used_mib", "").strip().isdigit()
            ]
            saved_report["gpu_telemetry"] = {
                "sample_count": len(telemetry_rows),
                "peak_utilization_percent": max(util_values) if util_values else None,
                "peak_memory_used_mib": max(memory_values) if memory_values else None,
                "sha256": sha256_file(telemetry_path) if telemetry_path.is_file() else None,
                "monitor_error": telemetry.error,
            }
            saved_report["runtime_seconds"] = time.perf_counter() - started
            saved_report["exit_code"] = exit_code
            saved_report["outputs"]["gpu_telemetry.csv"] = (
                sha256_file(telemetry_path) if telemetry_path.is_file() else None
            )
            saved_report["outputs"]["training.log"] = (
                sha256_file(run_dir / "training.log")
                if (run_dir / "training.log").is_file()
                else None
            )
            (run_dir / "report.json").write_text(
                json.dumps(saved_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif (run_dir / "failure_report.json").is_file():
            failure = json.loads(
                (run_dir / "failure_report.json").read_text(encoding="utf-8")
            )
            failure["gpu_telemetry_sha256"] = (
                sha256_file(run_dir / "gpu_telemetry.csv")
                if (run_dir / "gpu_telemetry.csv").is_file()
                else None
            )
            failure["gpu_telemetry_samples"] = sum(
                1
                for _ in (
                    run_dir / "gpu_telemetry.csv"
                ).open(encoding="utf-8")
            ) - 1 if (run_dir / "gpu_telemetry.csv").is_file() else 0
            (run_dir / "failure_report.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"Run directory: {run_dir}")
        if exit_code:
            print(f"Experiment failed; inspect {run_dir / 'failure_report.json'}")


if __name__ == "__main__":
    main()
