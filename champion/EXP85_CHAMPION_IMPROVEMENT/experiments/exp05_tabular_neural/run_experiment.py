#!/usr/bin/env python3
"""Train a raw-only residual tabular neural net on matched Phase 2 folds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import catboost
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_DIR = Path(__file__).resolve().parent
IMPROVEMENT_DIR = EXPERIMENT_DIR.parents[1]
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[3]
FROZEN_PROJECT_DIR = (
    REPOSITORY_ROOT
    / "champion"
    / "EXP85_PHASE2_FROZEN_RELEASE"
    / "SOURCE"
    / "project"
)
PHASE_04_DIR = FROZEN_PROJECT_DIR / "experiments" / "phase_04_final_validation_gate"
REFERENCE_DIR = (
    REPOSITORY_ROOT
    / "champion"
    / "EXP85_PHASE2_FROZEN_RELEASE"
    / "EVIDENCE"
    / "second_seed"
    / "main"
)
EXP04_RUN_DIR = (
    IMPROVEMENT_DIR
    / "experiments"
    / "exp04_catboost_lossguide"
    / "runs"
    / "exp04_lossguide_strict_seed20261101"
)
DATA_DIR = Path(r"D:\gci\inpuy")
CHAMPION_REFERENCE = (
    IMPROVEMENT_DIR
    / "baseline"
    / "reference"
    / "EXP85_HISTORICAL_CHAMPION.csv"
)
OUTER_SEED = 20261101
INNER_SEED_BASE = 100_000
N_SPLITS = 5
BATCH_SIZE = 2048
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
INNER_VALIDATION_FRACTION = 0.15
GPU_MEMORY_FRACTION = 0.76
EXPECTED_CHAMPION_SHA256 = (
    "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
)
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}

sys.path.insert(0, str(FROZEN_PROJECT_DIR / "source"))
sys.path.insert(0, str(PHASE_04_DIR))

from run_final_gate import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    GpuMonitor,
    build_rowwise_features,
    make_variant_fold,
)


MODEL_CONFIG = {
    "architecture": "categorical embeddings + 2-block residual MLP",
    "hidden_width": 256,
    "residual_blocks": 2,
    "dropout": 0.20,
    "embedding_min_dim": 4,
    "embedding_max_dim": 16,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "loss": "BCEWithLogitsLoss",
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "inner_validation_fraction": INNER_VALIDATION_FRACTION,
    "gpu_memory_fraction": GPU_MEMORY_FRACTION,
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
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def validate_probabilities(values: np.ndarray, expected_rows: int, name: str) -> None:
    require(values.ndim == 1, f"{name} must be a one-dimensional array")
    require(len(values) == expected_rows, f"{name} row count is incorrect")
    require(np.isfinite(values).all(), f"{name} contains non-finite values")
    require(
        ((values >= 0.0) & (values <= 1.0)).all(),
        f"{name} contains values outside [0, 1]",
    )


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class ResidualTabularNet(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        category_sizes: list[int],
        embedding_dimensions: list[int],
        hidden_width: int,
        residual_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(category_size, embedding_dimension)
                for category_size, embedding_dimension in zip(
                    category_sizes, embedding_dimensions
                )
            ]
        )
        input_width = numeric_count + sum(embedding_dimensions)
        self.stem = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_width, dropout)
                for _ in range(residual_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width // 2, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            layer(categorical[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        values = torch.cat([numeric, *embedded], dim=1)
        return self.head(self.blocks(self.stem(values))).squeeze(1)


class FoldPreprocessor:
    def __init__(self, categorical_columns: tuple[str, ...]) -> None:
        self.categorical_columns = categorical_columns
        self.numeric_columns: list[str] = []
        self.numeric_median: np.ndarray | None = None
        self.numeric_mean: np.ndarray | None = None
        self.numeric_scale: np.ndarray | None = None
        self.category_maps: list[dict[str, int]] = []
        self.category_sizes: list[int] = []
        self.embedding_dimensions: list[int] = []

    def fit(
        self, frame: pd.DataFrame, fit_indices: np.ndarray
    ) -> "FoldPreprocessor":
        missing_categoricals = set(self.categorical_columns) - set(frame.columns)
        require(
            not missing_categoricals,
            f"Missing categorical features: {sorted(missing_categoricals)}",
        )
        self.numeric_columns = [
            column
            for column in frame.columns
            if column not in self.categorical_columns
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        require(self.numeric_columns, "No numeric features were produced")
        fit_numeric = frame.iloc[fit_indices][self.numeric_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        fit_numeric[~np.isfinite(fit_numeric)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(fit_numeric, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        fit_numeric = np.where(np.isnan(fit_numeric), medians, fit_numeric)
        means = np.mean(fit_numeric, axis=0)
        scales = np.std(fit_numeric, axis=0)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
        self.numeric_median = medians
        self.numeric_mean = means
        self.numeric_scale = scales

        self.category_maps = []
        self.category_sizes = []
        self.embedding_dimensions = []
        for column in self.categorical_columns:
            values = (
                frame.iloc[fit_indices][column]
                .fillna("__MISSING__")
                .astype(str)
                .unique()
                .tolist()
            )
            mapping = {
                category: index + 1
                for index, category in enumerate(sorted(values))
            }
            size = len(mapping) + 1
            dimension = min(
                MODEL_CONFIG["embedding_max_dim"],
                max(
                    MODEL_CONFIG["embedding_min_dim"],
                    int(round(1.6 * np.sqrt(size))),
                ),
            )
            self.category_maps.append(mapping)
            self.category_sizes.append(size)
            self.embedding_dimensions.append(dimension)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        require(
            self.numeric_median is not None
            and self.numeric_mean is not None
            and self.numeric_scale is not None,
            "Preprocessor must be fit before transforming",
        )
        numeric = frame[self.numeric_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        numeric[~np.isfinite(numeric)] = np.nan
        numeric = np.where(np.isnan(numeric), self.numeric_median, numeric)
        numeric = (numeric - self.numeric_mean) / self.numeric_scale
        numeric = np.clip(numeric, -12.0, 12.0).astype(np.float32)
        categorical = np.empty(
            (len(frame), len(self.categorical_columns)), dtype=np.int64
        )
        for index, (column, mapping) in enumerate(
            zip(self.categorical_columns, self.category_maps)
        ):
            values = frame[column].fillna("__MISSING__").astype(str)
            categorical[:, index] = (
                values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
            )
        require(
            np.isfinite(numeric).all(),
            "Preprocessed numeric matrix contains non-finite values",
        )
        return numeric, categorical

    def state_dict(self) -> dict[str, Any]:
        require(
            self.numeric_median is not None
            and self.numeric_mean is not None
            and self.numeric_scale is not None,
            "Preprocessor has not been fit",
        )
        return {
            "numeric_columns": self.numeric_columns,
            "numeric_median": self.numeric_median.astype(np.float64),
            "numeric_mean": self.numeric_mean.astype(np.float64),
            "numeric_scale": self.numeric_scale.astype(np.float64),
            "categorical_columns": list(self.categorical_columns),
            "category_maps": self.category_maps,
            "category_sizes": self.category_sizes,
            "embedding_dimensions": self.embedding_dimensions,
        }


def make_model(preprocessor: FoldPreprocessor) -> ResidualTabularNet:
    return ResidualTabularNet(
        numeric_count=len(preprocessor.numeric_columns),
        category_sizes=preprocessor.category_sizes,
        embedding_dimensions=preprocessor.embedding_dimensions,
        hidden_width=MODEL_CONFIG["hidden_width"],
        residual_blocks=MODEL_CONFIG["residual_blocks"],
        dropout=MODEL_CONFIG["dropout"],
    ).to("cuda:0")


def predict_batches(
    model: ResidualTabularNet,
    numeric: np.ndarray,
    categorical: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            stop = min(start + batch_size, len(numeric))
            numeric_batch = torch.from_numpy(numeric[start:stop]).to(
                device="cuda:0", non_blocking=False
            )
            categorical_batch = torch.from_numpy(categorical[start:stop]).to(
                device="cuda:0", non_blocking=False
            )
            outputs.append(
                torch.sigmoid(model(numeric_batch, categorical_batch))
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs).astype(np.float32, copy=False)


def fit_neural_network(
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    preprocessor: FoldPreprocessor,
    seed: int,
    run_dir: Path,
    checkpoint_name: str,
    log_path: Path,
    monitor: GpuMonitor,
) -> dict[str, Any]:
    seed_everything(seed)
    model = make_model(preprocessor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"],
        weight_decay=MODEL_CONFIG["weight_decay"],
    )
    loss_function = nn.BCEWithLogitsLoss()
    fit_numeric = torch.from_numpy(numeric[fit_indices])
    fit_categorical = torch.from_numpy(categorical[fit_indices])
    fit_labels = torch.from_numpy(labels[fit_indices].astype(np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(fit_numeric, fit_categorical, fit_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_losses = []
        for numeric_batch, categorical_batch, label_batch in loader:
            numeric_batch = numeric_batch.to("cuda:0", non_blocking=False)
            categorical_batch = categorical_batch.to("cuda:0", non_blocking=False)
            label_batch = label_batch.to("cuda:0", non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric_batch, categorical_batch)
            loss = loss_function(logits, label_batch)
            require(
                torch.isfinite(loss).item(),
                f"Non-finite training loss at epoch {epoch + 1}",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        validation_prediction = predict_batches(
            model,
            numeric[validation_indices],
            categorical[validation_indices],
            BATCH_SIZE * 2,
        )
        validation_auc = float(
            roc_auc_score(labels[validation_indices], validation_prediction)
        )
        mean_loss = float(np.mean(epoch_losses))
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "fit_bce": mean_loss,
                "inner_validation_auc": validation_auc,
            }
        )
        if validation_auc > best_auc + 1e-6:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (epoch + 1) % 10 == 0:
            log_message(
                log_path,
                f"{checkpoint_name}: epoch={epoch + 1} "
                f"fit_bce={mean_loss:.6f} inner_auc={validation_auc:.6f} "
                f"best_epoch={best_epoch + 1}",
            )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break
    require(best_state is not None, f"{checkpoint_name} produced no best model")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "preprocessor": preprocessor.state_dict(),
            "model_config": MODEL_CONFIG,
            "seed": seed,
            "best_epoch": best_epoch + 1,
            "best_inner_auc": float(best_auc),
        },
        run_dir / checkpoint_name,
    )
    return {
        "model": model,
        "best_epoch": best_epoch + 1,
        "best_inner_auc": float(best_auc),
        "epochs_run": len(epoch_records),
        "epoch_records": epoch_records,
        "fit_bce_at_best": epoch_records[best_epoch]["fit_bce"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_inner_early_stopping_split(
    labels: np.ndarray, outer_fit_indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed,
    )
    fit_local, valid_local = next(
        split.split(
            np.zeros(len(outer_fit_indices)), labels[outer_fit_indices]
        )
    )
    fit_indices = outer_fit_indices[fit_local]
    valid_indices = outer_fit_indices[valid_local]
    require(
        not np.intersect1d(fit_indices, valid_indices).size,
        "Inner fit and validation indices overlap",
    )
    require(
        np.isin(fit_indices, outer_fit_indices).all()
        and np.isin(valid_indices, outer_fit_indices).all(),
        "Inner split contains rows outside the outer fit partition",
    )
    return fit_indices, valid_indices


def verified_prediction_reference(
    run_dir: Path,
    prediction_name: str,
    expected_status: str,
    labels: np.ndarray,
    data_hashes: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = run_dir / "experiment_summary.json"
    manifest_path = run_dir / "sha256_manifest.json"
    prediction_path = run_dir / prediction_name
    require(summary_path.is_file(), f"Reference summary missing: {summary_path}")
    require(manifest_path.is_file(), f"Reference manifest missing: {manifest_path}")
    require(prediction_path.is_file(), f"Reference predictions missing: {prediction_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        summary.get("status") == expected_status,
        f"Unexpected reference status for {run_dir}: {summary.get('status')}",
    )
    for key in ("train", "test", "sample"):
        expected = manifest["inputs"][key]
        require(
            expected["sha256"] == data_hashes[key]["sha256"],
            f"Reference {run_dir.name} used different raw {key} input",
        )
    expected_prediction = manifest["outputs"][prediction_name]
    require(
        sha256_file(prediction_path) == expected_prediction["sha256"],
        f"Reference prediction hash mismatch: {prediction_path}",
    )
    prediction = np.load(prediction_path, allow_pickle=False)
    validate_probabilities(prediction, len(labels), f"Reference {prediction_name}")
    return prediction, {
        "path": str(prediction_path.resolve()),
        "sha256": expected_prediction["sha256"],
        "status": summary["status"],
        "classification": "OOF validation comparator only; never a model input",
    }


def preflight_gpu() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA-enabled PyTorch is required; CPU fallback is disabled")
    require(torch.cuda.device_count() >= 1, "PyTorch found no CUDA devices")
    require(
        torch.version.cuda is not None,
        "Installed PyTorch does not include CUDA support",
    )
    device_name = torch.cuda.get_device_name(0)
    require("RTX 4050" in device_name, f"Unexpected CUDA device: {device_name}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total_mib = int(total_bytes // (1024 * 1024))
    free_mib = int(free_bytes // (1024 * 1024))
    require(total_mib <= 7000, f"GPU reports more memory than the requested 6 GB class: {total_mib} MiB")
    require(free_mib >= 4500, f"Insufficient free GPU memory: {free_mib} MiB")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
    seed_everything(0)
    test_numeric = torch.rand((16, 8), device="cuda:0")
    test_categorical = torch.zeros((16, 2), dtype=torch.long, device="cuda:0")
    model = ResidualTabularNet(
        numeric_count=8,
        category_sizes=[4, 5],
        embedding_dimensions=[4, 4],
        hidden_width=32,
        residual_blocks=1,
        dropout=0.1,
    ).to("cuda:0")
    result = model(test_numeric, test_categorical).mean()
    result.backward()
    torch.cuda.synchronize()
    require(torch.isfinite(result).item(), "CUDA neural smoke fit was non-finite")
    return {
        "torch_version": torch.__version__,
        "torch_module_path": str(Path(torch.__file__).resolve()),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device_name,
        "device_index": 0,
        "total_memory_mib": total_mib,
        "free_memory_mib_before_run": free_mib,
        "process_memory_fraction": GPU_MEMORY_FRACTION,
        "smoke_fit": "PASS",
        "cpu_fallback": False,
    }


def compare_predictions(
    labels: np.ndarray,
    candidate: np.ndarray,
    references: dict[str, np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    output = {}
    for name, reference in references.items():
        validate_probabilities(reference, len(labels), f"{name} OOF comparator")
        candidate_auc = float(roc_auc_score(labels, candidate))
        reference_auc = float(roc_auc_score(labels, reference))
        candidate_folds = [
            float(roc_auc_score(labels[valid], candidate[valid]))
            for _, valid in folds
        ]
        reference_folds = [
            float(roc_auc_score(labels[valid], reference[valid]))
            for _, valid in folds
        ]
        deltas = [
            current - baseline
            for current, baseline in zip(candidate_folds, reference_folds)
        ]
        output[name] = {
            "candidate_oof_auc": candidate_auc,
            "reference_oof_auc": reference_auc,
            "pooled_auc_delta": candidate_auc - reference_auc,
            "candidate_fold_auc": candidate_folds,
            "reference_fold_auc": reference_folds,
            "paired_fold_auc_delta": deltas,
            "positive_folds": sum(delta > 0 for delta in deltas),
            "oof_pearson": float(np.corrcoef(candidate, reference)[0, 1]),
            "oof_spearman": float(
                spearmanr(candidate, reference).statistic
            ),
        }
    return output


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def importance_from_state(
    state: dict[str, torch.Tensor],
    numeric_columns: list[str],
    categorical_columns: tuple[str, ...],
    embedding_dimensions: list[int],
) -> pd.DataFrame:
    input_weights = state["stem.0.weight"].detach().cpu().numpy()
    scores = []
    for index, column in enumerate(numeric_columns):
        scores.append((column, float(np.mean(np.abs(input_weights[:, index])))))
    start = len(numeric_columns)
    for column, dimension in zip(categorical_columns, embedding_dimensions):
        segment = input_weights[:, start : start + dimension]
        scores.append((column, float(np.mean(np.abs(segment)))))
        start += dimension
    return pd.DataFrame(scores, columns=["feature", "mean_abs_stem_weight"])


def run_experiment(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.perf_counter()
    source_paths = {
        "experiment_runner": Path(__file__).resolve(),
        "phase2_raw_feature_builder": FROZEN_PROJECT_DIR
        / "source"
        / "run_independent_baseline.py",
        "phase1_fee_product_features": FROZEN_PROJECT_DIR
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py",
        "phase2_target_encodings": FROZEN_PROJECT_DIR
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py",
        "phase2_fold_local_pipeline": PHASE_04_DIR / "run_final_gate.py",
        "exp04_runner": IMPROVEMENT_DIR
        / "experiments"
        / "exp04_catboost_lossguide"
        / "run_experiment.py",
    }
    for name, path in source_paths.items():
        require(path.is_file(), f"Required source is missing: {path} ({name})")
    source_records = {name: file_record(path) for name, path in source_paths.items()}
    input_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in input_paths.values():
        require(path.is_file(), f"Required raw input is missing: {path}")
    input_records = {name: file_record(path) for name, path in input_paths.items()}
    for name, record in input_records.items():
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[name],
            f"Unexpected raw {name} input hash: {record['sha256']}",
        )
    gpu_before = preflight_gpu()
    log_message(
        log_path,
        f"CUDA preflight: {gpu_before['device']} torch={gpu_before['torch_version']} "
        f"CUDA={gpu_before['torch_cuda_build']} "
        f"free={gpu_before['free_memory_mib_before_run']}MiB",
    )

    train_raw = pd.read_csv(input_paths["train"])
    test_raw = pd.read_csv(input_paths["test"])
    sample = pd.read_csv(input_paths["sample"])
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Unexpected sample-submission schema",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()),
        "Test/sample IDs or ordering differ",
    )
    require(
        train_raw[ID_COLUMN].is_unique and test_raw[ID_COLUMN].is_unique,
        "Duplicate applicant IDs found",
    )
    require(
        set(train_raw[ID_COLUMN]).isdisjoint(set(test_raw[ID_COLUMN])),
        "Training and test IDs overlap",
    )
    labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(set(np.unique(labels)).issubset({0, 1}), "Target is not binary")

    raw_train_features = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    raw_test_features = build_rowwise_features(test_raw)
    require(
        raw_train_features.columns.tolist() == raw_test_features.columns.tolist(),
        "Raw rowwise train/test schemas differ",
    )
    require(
        ID_COLUMN not in raw_train_features
        and TARGET_COLUMN not in raw_train_features,
        "ID or target reached raw rowwise features",
    )
    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED
        ).split(np.zeros(len(labels)), labels)
    )
    phase2_oof, phase2_record = verified_prediction_reference(
        REFERENCE_DIR,
        "oof_phase2_fold_local_blend.npy",
        "COMPLETED",
        labels,
        input_records,
    )
    exp04_oof, exp04_record = verified_prediction_reference(
        EXP04_RUN_DIR,
        "oof_catboost_lossguide.npy",
        "COMPLETED",
        labels,
        input_records,
    )
    phase2_manifest = json.loads(
        (REFERENCE_DIR / "sha256_manifest.json").read_text(encoding="utf-8")
    )
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        fold_path = REFERENCE_DIR / f"fold_{fold_number}_indices.npz"
        expected = phase2_manifest["outputs"][fold_path.name]
        require(
            sha256_file(fold_path) == expected["sha256"],
            f"Phase 2 reference fold hash mismatch: {fold_path}",
        )
        with np.load(fold_path, allow_pickle=False) as archive:
            require(
                np.array_equal(archive["fit_indices"], fit_indices)
                and np.array_equal(archive["valid_indices"], valid_indices),
                f"EXP05 split differs from matched Phase 2 fold {fold_number}",
            )
    exp04_manifest = json.loads(
        (EXP04_RUN_DIR / "sha256_manifest.json").read_text(encoding="utf-8")
    )
    exp04_fold_arrays = np.load(
        EXP04_RUN_DIR / "fold_indices.npz", allow_pickle=False
    )
    try:
        for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
            expected = exp04_manifest["outputs"][
                f"fold_{fold_number}_indices.npz"
            ]
            fold_path = EXP04_RUN_DIR / f"fold_{fold_number}_indices.npz"
            require(
                sha256_file(fold_path) == expected["sha256"],
                f"EXP04 reference fold hash mismatch: {fold_path}",
            )
            require(
                np.array_equal(
                    exp04_fold_arrays[f"fit_{fold_number}"], fit_indices
                )
                and np.array_equal(
                    exp04_fold_arrays[f"valid_{fold_number}"], valid_indices
                ),
                f"EXP04 split differs from matched fold {fold_number}",
            )
    finally:
        exp04_fold_arrays.close()

    oof = np.full(len(labels), np.nan, dtype=np.float32)
    test_fold_predictions = []
    fold_records = []
    fold_arrays: dict[str, np.ndarray] = {}
    importance_frames = []
    gpu_samples = []
    feature_names: list[str] | None = None
    log_message(
        log_path,
        f"Training {run_dir.name}: rows={len(labels)} test_rows={len(test_raw)} "
        f"folds={N_SPLITS} seed={OUTER_SEED}",
    )
    for fold_number, (outer_fit_indices, outer_valid_indices) in enumerate(
        folds, 1
    ):
        torch.cuda.reset_peak_memory_stats(0)
        require(
            not np.intersect1d(outer_fit_indices, outer_valid_indices).size,
            f"Outer fold {fold_number} fit/validation overlap",
        )
        (
            outer_frame,
            outer_test_frame,
            _,
            _,
            outer_feature_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            raw_train_features,
            raw_test_features,
            outer_fit_indices,
            outer_valid_indices,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if feature_names is None:
            feature_names = outer_frame.columns.tolist()
        require(
            outer_frame.columns.tolist() == feature_names
            and outer_test_frame.columns.tolist() == feature_names,
            f"Outer fold {fold_number} feature schema differs",
        )
        require(
            ID_COLUMN not in feature_names
            and TARGET_COLUMN not in feature_names,
            "ID or target leaked into outer feature matrix",
        )

        inner_seed = OUTER_SEED + fold_number
        inner_fit_indices, inner_valid_indices = build_inner_early_stopping_split(
            labels, outer_fit_indices, inner_seed
        )
        (
            inner_frame,
            _,
            _,
            _,
            inner_feature_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            raw_train_features,
            raw_test_features,
            inner_fit_indices,
            inner_valid_indices,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        require(
            inner_frame.columns.tolist() == feature_names,
            f"Inner fold {fold_number} feature schema differs",
        )
        inner_preprocessor = FoldPreprocessor(CATEGORICAL_COLUMNS).fit(
            inner_frame, inner_fit_indices
        )
        inner_numeric, inner_categorical = inner_preprocessor.transform(
            inner_frame
        )
        inner_model_seed = INNER_SEED_BASE + fold_number
        fold_started = time.perf_counter()
        with GpuMonitor() as monitor:
            inner_fit = fit_neural_network(
                inner_numeric,
                inner_categorical,
                labels,
                inner_fit_indices,
                inner_valid_indices,
                inner_preprocessor,
                inner_model_seed,
                run_dir,
                f"checkpoint_fold_{fold_number}_inner.pt",
                log_path,
                monitor,
            )
            selected_epochs = inner_fit["best_epoch"]
            del inner_fit["model"]

            outer_preprocessor = FoldPreprocessor(CATEGORICAL_COLUMNS).fit(
                outer_frame, outer_fit_indices
            )
            outer_numeric, outer_categorical = outer_preprocessor.transform(
                outer_frame
            )
            test_numeric, test_categorical = outer_preprocessor.transform(
                outer_test_frame
            )
            outer_seed = OUTER_SEED + 10_000 + fold_number
            seed_everything(outer_seed)
            outer_model = make_model(outer_preprocessor)
            optimizer = torch.optim.AdamW(
                outer_model.parameters(),
                lr=MODEL_CONFIG["learning_rate"],
                weight_decay=MODEL_CONFIG["weight_decay"],
            )
            criterion = nn.BCEWithLogitsLoss()
            outer_dataset = TensorDataset(
                torch.from_numpy(outer_numeric[outer_fit_indices]),
                torch.from_numpy(outer_categorical[outer_fit_indices]),
                torch.from_numpy(labels[outer_fit_indices].astype(np.float32)),
            )
            generator = torch.Generator()
            generator.manual_seed(outer_seed)
            outer_loader = DataLoader(
                outer_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=generator,
                num_workers=0,
                drop_last=False,
                pin_memory=False,
            )
            for _ in range(selected_epochs):
                outer_model.train()
                for numeric_batch, category_batch, label_batch in outer_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = outer_model(
                        numeric_batch.to("cuda:0"),
                        category_batch.to("cuda:0"),
                    )
                    loss = criterion(
                        logits,
                        label_batch.to("cuda:0"),
                    )
                    require(
                        torch.isfinite(loss).item(),
                        f"Non-finite outer fit loss in fold {fold_number}",
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        outer_model.parameters(), max_norm=5.0
                    )
                    optimizer.step()
            torch.cuda.synchronize()

            valid_prediction = predict_batches(
                outer_model,
                outer_numeric[outer_valid_indices],
                outer_categorical[outer_valid_indices],
                BATCH_SIZE * 2,
            )
            test_prediction = predict_batches(
                outer_model,
                test_numeric,
                test_categorical,
                BATCH_SIZE * 2,
            )
        require(
            not monitor.errors,
            f"GPU telemetry failed in fold {fold_number}: "
            + "; ".join(monitor.errors),
        )
        require(
            monitor.samples
            and all(
                "RTX 4050" in item["name"] for item in monitor.samples
            ),
            f"RTX 4050 training telemetry missing for fold {fold_number}",
        )
        validate_probabilities(
            valid_prediction, len(outer_valid_indices), f"Fold {fold_number} OOF"
        )
        validate_probabilities(
            test_prediction, len(test_raw), f"Fold {fold_number} test"
        )
        oof[outer_valid_indices] = valid_prediction
        test_fold_predictions.append(test_prediction)
        fold_arrays[f"fit_{fold_number}"] = outer_fit_indices.astype(np.int32)
        fold_arrays[f"valid_{fold_number}"] = outer_valid_indices.astype(
            np.int32
        )

        final_checkpoint = run_dir / f"checkpoint_fold_{fold_number}_final.pt"
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in outer_model.state_dict().items()
                },
                "preprocessor": outer_preprocessor.state_dict(),
                "model_config": MODEL_CONFIG,
                "seed": outer_seed,
                "selected_epochs": selected_epochs,
                "inner_best_auc": inner_fit.get("best_inner_auc"),
            },
            final_checkpoint,
        )
        importance_frames.append(
            importance_from_state(
                {
                    key: value.detach().cpu()
                    for key, value in outer_model.state_dict().items()
                },
                outer_preprocessor.numeric_columns,
                CATEGORICAL_COLUMNS,
                outer_preprocessor.embedding_dimensions,
            ).assign(fold=fold_number)
        )
        gpu_samples.extend(monitor.samples)
        fold_record = {
            "fold": fold_number,
            "fit_rows": int(len(outer_fit_indices)),
            "validation_rows": int(len(outer_valid_indices)),
            "fit_validation_disjoint": True,
            "inner_fit_rows": int(len(inner_fit_indices)),
            "inner_validation_rows": int(len(inner_valid_indices)),
            "inner_fit_outer_validation_overlap": int(
                np.intersect1d(inner_fit_indices, outer_valid_indices).size
            ),
            "inner_validation_outer_validation_overlap": int(
                np.intersect1d(inner_valid_indices, outer_valid_indices).size
            ),
            "outer_validation_used_for_early_stopping": False,
            "inner_split_seed": inner_seed,
            "inner_model_seed": inner_model_seed,
            "outer_refit_seed": outer_seed,
            "selected_epochs": selected_epochs,
            "inner_best_auc": float(
                torch.load(
                    run_dir / f"checkpoint_fold_{fold_number}_inner.pt",
                    map_location="cpu",
                    weights_only=False,
                )["best_inner_auc"]
            ),
            "outer_oof_auc": float(
                roc_auc_score(labels[outer_valid_indices], valid_prediction)
            ),
            "elapsed_seconds": time.perf_counter() - fold_started,
            "gpu": {
                "device": "cuda:0",
                "name": gpu_before["device"],
                "telemetry_samples": len(monitor.samples),
                "max_utilization_percent": max(
                    item["utilization_percent"] for item in monitor.samples
                ),
                "max_memory_used_mib": max(
                    item["memory_used_mib"] for item in monitor.samples
                ),
                "torch_peak_allocated_mib": int(
                    torch.cuda.max_memory_allocated(0) // (1024 * 1024)
                ),
                "torch_peak_reserved_mib": int(
                    torch.cuda.max_memory_reserved(0) // (1024 * 1024)
                ),
            },
            "feature_protocol": outer_feature_audit,
            "inner_feature_protocol": inner_feature_audit,
            "feature_count": len(feature_names or []),
            "numeric_feature_count": len(outer_preprocessor.numeric_columns),
            "categorical_embedding_dims": outer_preprocessor.embedding_dimensions,
            "checkpoint": str(final_checkpoint.resolve()),
        }
        fold_records.append(fold_record)
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=outer_fit_indices.astype(np.int32),
            valid_indices=outer_valid_indices.astype(np.int32),
        )
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_predictions.npz",
            valid_indices=outer_valid_indices.astype(np.int32),
            oof=valid_prediction,
            test=test_prediction,
        )
        log_message(
            log_path,
            f"Fold {fold_number}/{N_SPLITS}: OOF_AUC="
            f"{fold_record['outer_oof_auc']:.8f} "
            f"inner_AUC={fold_record['inner_best_auc']:.8f} "
            f"epochs={selected_epochs} "
            f"gpu={fold_record['gpu']['max_utilization_percent']}% "
            f"VRAM={fold_record['gpu']['max_memory_used_mib']}MiB",
        )
        del outer_model
        torch.cuda.empty_cache()

    validate_probabilities(oof, len(labels), "Complete candidate OOF")
    test_prediction = np.mean(
        np.stack(test_fold_predictions), axis=0, dtype=np.float64
    ).astype(np.float32)
    validate_probabilities(test_prediction, len(test_raw), "Candidate test")
    np.save(run_dir / "oof_tabular_neural.npy", oof)
    np.save(run_dir / "test_tabular_neural.npy", test_prediction)
    np.savez_compressed(run_dir / "fold_indices.npz", **fold_arrays)

    candidate_comparisons = compare_predictions(
        labels,
        oof,
        {"phase2_fold_local": phase2_oof, "exp04_lossguide": exp04_oof},
        folds,
    )
    importance = (
        pd.concat(importance_frames)
        .groupby("feature", sort=False)["mean_abs_stem_weight"]
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_stem_weight")
        .reset_index()
    )
    importance.to_csv(run_dir / "feature_diagnostics.csv", index=False)
    write_json(run_dir / "feature_columns.json", feature_names or [])

    candidate = pd.DataFrame(
        {
            ID_COLUMN: test_raw[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: test_prediction,
        }
    )
    require(
        candidate.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and candidate[ID_COLUMN].is_unique
        and np.array_equal(
            candidate[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()
        ),
        "Submission schema/ID alignment failed",
    )
    submission_path = run_dir / "submission_tabular_neural.csv"
    candidate.to_csv(submission_path, index=False, float_format="%.10f")
    parsed = pd.read_csv(submission_path)
    require(
        parsed.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and np.array_equal(
            parsed[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()
        ),
        "Serialized submission schema or ID order changed",
    )
    require(
        np.max(
            np.abs(
                parsed[TARGET_COLUMN].to_numpy(dtype=np.float64)
                - test_prediction
            )
        )
        < 1e-9,
        "Serialized candidate does not match test predictions",
    )

    champion_hash = sha256_file(CHAMPION_REFERENCE)
    require(
        champion_hash == EXPECTED_CHAMPION_SHA256,
        "Frozen historical champion hash changed",
    )
    champion = pd.read_csv(CHAMPION_REFERENCE)
    require(
        champion.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and np.array_equal(
            champion[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()
        ),
        "Historical champion schema/IDs differ from current test data",
    )
    champion_prediction = champion[TARGET_COLUMN].to_numpy(dtype=np.float64)
    exp04_test = np.load(
        EXP04_RUN_DIR / "test_catboost_lossguide.npy", allow_pickle=False
    )
    validate_probabilities(exp04_test, len(test_raw), "EXP04 test comparator")
    test_correlations = {
        "exp04_lossguide": {
            "pearson": float(np.corrcoef(test_prediction, exp04_test)[0, 1]),
            "spearman": float(spearmanr(test_prediction, exp04_test).statistic),
        },
        "historical_champion_diagnostic_only": {
            "pearson": float(np.corrcoef(test_prediction, champion_prediction)[0, 1]),
            "spearman": float(
                spearmanr(test_prediction, champion_prediction).statistic
            ),
            "classification": "Post-training diagnostic only; not a tuning target",
        },
    }
    rank_displacement = np.abs(
        rankdata(test_prediction, method="average")
        - rankdata(champion_prediction, method="average")
    )
    write_json(run_dir / "gpu_telemetry.json", gpu_samples)
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "lineage": "INDEPENDENT_RAW_ONLY_PYTORCH_RESIDUAL_TABULAR_NET",
        "official_competition_score": "NOT_EVALUATED",
        "historical_public_score_reference": 0.801340,
        "historical_public_score_verified_for_candidate": False,
        "raw_training_labels_used": True,
        "historical_predictions_used_as_training_labels": False,
        "historical_predictions_used_as_model_inputs": False,
        "champion_reference_read_after_training_for_diagnostic_only": True,
        "champion_reference_sha256": champion_hash,
        "inputs": input_records,
        "references": {
            "phase2_oof": phase2_record,
            "exp04_oof": exp04_record,
        },
        "folds": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "seed": OUTER_SEED,
            "matched_phase2_and_exp04_indices": True,
            "outer_validation_used_for_stopping": False,
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        },
        "feature_generation": {
            "base": "Frozen Phase 2 rowwise features",
            "additional_features": "Existing Phase 2 fold-local frequency/product features and nested target encodings",
            "frequency_reference": "Outer-fit rows for outer OOF/test features; inner-fit rows for inner early stopping",
            "target_encoding": "Nested inner OOF for model-fit rows; only fit-partition labels used for apply rows",
            "feature_names": feature_names,
            "feature_count": len(feature_names or []),
            "feature_name_sha256": hashlib.sha256(
                json.dumps(feature_names, separators=(",", ":")).encode()
            ).hexdigest(),
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_preprocessing": "Fold-fit medians, means, standard deviations; z-score clipping to [-12,12]",
            "unknown_categories": "Dedicated index zero; category maps fit on training partition only",
        },
        "model": {
            "parameters": MODEL_CONFIG,
            "seeds": {
                "inner_split": [
                    OUTER_SEED + fold
                    for fold in range(1, N_SPLITS + 1)
                ],
                "inner_model": [
                    INNER_SEED_BASE + fold for fold in range(1, N_SPLITS + 1)
                ],
                "outer_refit": [
                    OUTER_SEED + 10_000 + fold
                    for fold in range(1, N_SPLITS + 1)
                ],
            },
            "gpu_preflight": gpu_before,
            "gpu_training_confirmed": all(
                item["gpu"]["device"] == "cuda:0" for item in fold_records
            ),
            "gpu_telemetry": {
                "sample_count": len(gpu_samples),
                "max_utilization_percent": max(
                    item["utilization_percent"] for item in gpu_samples
                ),
                "max_memory_used_mib": max(
                    item["memory_used_mib"] for item in gpu_samples
                ),
                "max_torch_allocated_mib": max(
                    item["gpu"]["torch_peak_allocated_mib"]
                    for item in fold_records
                ),
            },
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "fold_seeds_and_checkpoints_saved": True,
            "independent_full_run_repeat": "NOT_PERFORMED",
        },
        "fold_results": fold_records,
        "comparisons": candidate_comparisons,
        "prediction_diagnostics": {
            "oof": summarize(oof),
            "test": summarize(test_prediction),
            "test_correlations": test_correlations,
            "historical_champion_rank_displacement": {
                "classification": "Diagnostic only",
                "mean": float(np.mean(rank_displacement)),
                "max": float(np.max(rank_displacement)),
            },
        },
        "candidate_submission": {
            "path": str(submission_path.resolve()),
            "sha256": sha256_file(submission_path),
            "rows": int(len(candidate)),
            "columns": candidate.columns.tolist(),
            "unique_ids": int(candidate[ID_COLUMN].nunique()),
            "prediction_min": float(test_prediction.min()),
            "prediction_max": float(test_prediction.max()),
        },
        "outputs": {},
        "source_hashes": source_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name in (
        "oof_tabular_neural.npy",
        "test_tabular_neural.npy",
        "fold_indices.npz",
        "feature_columns.json",
        "feature_diagnostics.csv",
        "gpu_telemetry.json",
        "submission_tabular_neural.csv",
    ):
        summary["outputs"][name] = file_record(run_dir / name)
    write_json(run_dir / "experiment_summary.json", summary)

    manifest = {
        "experiment_id": run_dir.name,
        "inputs": {
            **input_records,
            "phase2_oof_reference": file_record(
                REFERENCE_DIR / "oof_phase2_fold_local_blend.npy"
            ),
            "exp04_oof_reference": file_record(
                EXP04_RUN_DIR / "oof_catboost_lossguide.npy"
            ),
            "historical_champion_diagnostic_only": file_record(
                CHAMPION_REFERENCE
            ),
        },
        "sources": source_records,
        "outputs": {
            str(path.relative_to(run_dir)): file_record(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    }
    write_json(run_dir / "sha256_manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    default_run_dir = EXPERIMENT_DIR / "runs" / (
        "exp05_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_residual_mlp_seed20261101"
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()
