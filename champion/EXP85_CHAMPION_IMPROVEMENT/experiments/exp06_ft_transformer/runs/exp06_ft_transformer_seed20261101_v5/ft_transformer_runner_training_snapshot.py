#!/usr/bin/env python3
"""Strict five-fold FT-Transformer experiment with a protocol-matched baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import torch
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
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
SOURCE_DIR = FROZEN_PROJECT_DIR / "source"
PHASE_01_DIR = FROZEN_PROJECT_DIR / "experiments" / "phase_01_fee_product_features"
PHASE_02_DIR = FROZEN_PROJECT_DIR / "experiments" / "phase_02_fold_safe_key_encodings"
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
EXP05_RUN_DIR = (
    IMPROVEMENT_DIR
    / "experiments"
    / "exp05_tabular_neural"
    / "runs"
    / "exp05_cuda_seed20261101_v1"
)
DATA_DIR = Path(r"D:\gci\inpuy")

sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(PHASE_04_DIR))
from run_independent_baseline import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    MODEL_CONFIG as PHASE2_MODEL_CONFIG,
    TARGET_COLUMN,
    build_rowwise_features,
)
from run_final_gate import make_variant_fold  # noqa: E402


OUTER_SEED = 20261101
INNER_SEED_BASE = 20261101
INNER_MODEL_SEED_BASE = 310_000
N_SPLITS = 5
INNER_VALIDATION_FRACTION = 0.15
GPU_MEMORY_FRACTION = 0.76
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 2048
EXPECTED_INPUT_SHA256 = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
EXPECTED_CHAMPION_SHA256 = (
    "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
)
FT_CONFIG = {
    "architecture": "FT-Transformer: per-feature numeric tokenizers, categorical embeddings, CLS token, Transformer encoder",
    "token_dimension": 16,
    "attention_heads": 2,
    "transformer_layers": 2,
    "feedforward_dimension": 64,
    "dropout": 0.15,
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "loss": "BCEWithLogitsLoss",
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "inner_validation_fraction": INNER_VALIDATION_FRACTION,
    "mixed_precision": False,
    "gpu_memory_fraction": GPU_MEMORY_FRACTION,
}
ADVANCEMENT_GATE = {
    "minimum_pooled_auc_delta": 0.001,
    "minimum_positive_folds": 4,
    "number_of_folds": N_SPLITS,
}


class GpuTelemetryMonitor:
    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                timestamp = datetime.now(timezone.utc).isoformat()
                for line in result.stdout.splitlines():
                    fields = [part.strip() for part in line.split(",")]
                    if len(fields) != 4:
                        raise ValueError(f"Unexpected nvidia-smi sample: {line!r}")
                    self.samples.append(
                        {
                            "timestamp": timestamp,
                            "index": int(fields[0]),
                            "name": fields[1],
                            "memory_used_mib": int(fields[2]),
                            "utilization_percent": int(fields[3]),
                        }
                    )
            except (
                OSError,
                subprocess.SubprocessError,
                ValueError,
            ) as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "GpuTelemetryMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=32)
            if self._thread.is_alive():
                self.errors.append("Telemetry thread did not stop within 32 seconds")


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


def validate_fold_partition(
    folds: list[tuple[np.ndarray, np.ndarray]], row_count: int
) -> None:
    valid_parts = []
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        require(
            not np.intersect1d(fit_indices, valid_indices).size,
            f"Fold {fold_number} has overlapping fit/validation indices",
        )
        require(
            len(np.unique(fit_indices)) == len(fit_indices)
            and len(np.unique(valid_indices)) == len(valid_indices),
            f"Fold {fold_number} contains duplicate indices",
        )
        require(
            np.all((fit_indices >= 0) & (fit_indices < row_count))
            and np.all((valid_indices >= 0) & (valid_indices < row_count)),
            f"Fold {fold_number} contains an out-of-range index",
        )
        valid_parts.append(valid_indices)
    all_valid = np.concatenate(valid_parts)
    require(
        len(all_valid) == row_count
        and np.array_equal(np.sort(all_valid), np.arange(row_count)),
        "Outer validation folds do not partition every training row exactly once",
    )


def validate_feature_audit(audit: dict[str, Any], name: str) -> None:
    target_audit = audit["target_encoding"]
    require(
        target_audit["apply_row_targets_used_by_mapping"] is False,
        f"{name} target encoding used apply-row targets",
    )
    require(
        target_audit["validation_encoded_from_outer_fit_only"] is True
        and target_audit["test_encoded_from_outer_fit_only"] is True,
        f"{name} target encodings were not limited to fit-partition labels",
    )
    for partition, record in audit["frequency"].items():
        if not isinstance(record, dict) or "count_reference_rows" not in record:
            continue
        require(
            record["reference_contains_validation_or_test_rows"] is False,
            f"{name} frequency map included validation or test rows ({partition})",
        )
        require(
            record["target_used"] is False,
            f"{name} frequency map used labels ({partition})",
        )


def build_inner_split(
    labels: np.ndarray,
    outer_fit_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed,
    )
    fit_local, valid_local = next(
        split.split(np.zeros(len(outer_fit_indices)), labels[outer_fit_indices])
    )
    fit_indices = outer_fit_indices[fit_local]
    valid_indices = outer_fit_indices[valid_local]
    require(
        not np.intersect1d(fit_indices, valid_indices).size,
        "Inner fit/validation indices overlap",
    )
    require(
        np.isin(fit_indices, outer_fit_indices).all()
        and np.isin(valid_indices, outer_fit_indices).all(),
        "Inner partition contains rows outside outer fit",
    )
    return fit_indices, valid_indices


class FoldPreprocessor:
    def __init__(self, categorical_columns: tuple[str, ...]) -> None:
        self.categorical_columns = categorical_columns
        self.numeric_columns: list[str] = []
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.category_maps: list[dict[str, int]] = []
        self.category_sizes: list[int] = []

    def fit(
        self, frame: pd.DataFrame, fit_indices: np.ndarray
    ) -> "FoldPreprocessor":
        missing = set(self.categorical_columns) - set(frame.columns)
        require(not missing, f"Missing categorical features: {sorted(missing)}")
        self.numeric_columns = [
            column
            for column in frame.columns
            if column not in self.categorical_columns
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        require(self.numeric_columns, "No numeric FT-Transformer features")
        values = frame.iloc[fit_indices][self.numeric_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        values[~np.isfinite(values)] = np.nan
        with np.errstate(all="ignore"):
            median = np.nanmedian(values, axis=0)
        median[~np.isfinite(median)] = 0.0
        values = np.where(np.isnan(values), median, values)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        self.median, self.mean, self.scale = median, mean, scale
        self.category_maps = []
        self.category_sizes = []
        for column in self.categorical_columns:
            categories = (
                frame.iloc[fit_indices][column]
                .fillna("__MISSING__")
                .astype(str)
                .unique()
                .tolist()
            )
            mapping = {
                category: index + 1
                for index, category in enumerate(sorted(categories))
            }
            self.category_maps.append(mapping)
            self.category_sizes.append(len(mapping) + 1)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        require(
            self.median is not None
            and self.mean is not None
            and self.scale is not None,
            "Preprocessor must be fit before transform",
        )
        numeric = frame[self.numeric_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        numeric[~np.isfinite(numeric)] = np.nan
        numeric = np.where(np.isnan(numeric), self.median, numeric)
        numeric = ((numeric - self.mean) / self.scale).clip(-12.0, 12.0)
        numeric = numeric.astype(np.float32)
        categorical = np.empty(
            (len(frame), len(self.categorical_columns)), dtype=np.int64
        )
        for column_index, (column, mapping) in enumerate(
            zip(self.categorical_columns, self.category_maps)
        ):
            values = frame[column].fillna("__MISSING__").astype(str)
            categorical[:, column_index] = (
                values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
            )
        require(np.isfinite(numeric).all(), "Numeric features are non-finite")
        require(
            all(
                np.all((categorical[:, index] >= 0) & (categorical[:, index] < size))
                for index, size in enumerate(self.category_sizes)
            ),
            "Categorical index outside its embedding range",
        )
        return numeric, categorical

    def state_dict(self) -> dict[str, Any]:
        require(self.mean is not None, "Preprocessor has not been fit")
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": list(self.categorical_columns),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "category_maps": self.category_maps,
            "category_sizes": self.category_sizes,
        }


class FTTransformer(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        category_sizes: list[int],
        token_dimension: int = 32,
        attention_heads: int = 4,
        transformer_layers: int = 3,
        feedforward_dimension: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.numeric_weight = nn.Parameter(
            torch.empty(numeric_count, token_dimension)
        )
        self.numeric_bias = nn.Parameter(
            torch.zeros(numeric_count, token_dimension)
        )
        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(size, token_dimension)
                for size in category_sizes
            ]
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dimension))
        nn.init.normal_(self.numeric_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for embedding in self.category_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

        layer = nn.TransformerEncoderLayer(
            d_model=token_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(token_dimension)
        self.head = nn.Sequential(
            nn.Linear(token_dimension, token_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dimension, 1),
        )

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        numeric_tokens = (
            numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )
        categorical_tokens = [
            embedding(categorical[:, index]).unsqueeze(1)
            for index, embedding in enumerate(self.category_embeddings)
        ]
        tokens = torch.cat(
            [
                self.cls_token.expand(len(numeric), -1, -1),
                numeric_tokens,
                *categorical_tokens,
            ],
            dim=1,
        )
        encoded = self.transformer(tokens)
        return self.head(self.output_norm(encoded[:, 0])).squeeze(1)


def make_model(preprocessor: FoldPreprocessor) -> FTTransformer:
    return FTTransformer(
        numeric_count=len(preprocessor.numeric_columns),
        category_sizes=preprocessor.category_sizes,
        token_dimension=FT_CONFIG["token_dimension"],
        attention_heads=FT_CONFIG["attention_heads"],
        transformer_layers=FT_CONFIG["transformer_layers"],
        feedforward_dimension=FT_CONFIG["feedforward_dimension"],
        dropout=FT_CONFIG["dropout"],
    ).to("cuda:0")


def gpu_preflight() -> dict[str, Any]:
    require(
        torch.cuda.is_available(),
        "CUDA-enabled PyTorch is required; CPU fallback is disabled",
    )
    require(torch.cuda.device_count() > 0, "No CUDA device is visible")
    require(torch.version.cuda is not None, "PyTorch build has no CUDA support")
    device = torch.cuda.get_device_name(0)
    require("RTX 4050" in device, f"Unexpected GPU: {device}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mib = int(free_bytes // (1024 * 1024))
    total_mib = int(total_bytes // (1024 * 1024))
    require(total_mib <= 7000, f"Unexpected GPU memory class: {total_mib} MiB")
    require(free_mib >= 4500, f"Insufficient free GPU memory: {free_mib} MiB")
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
    seed_everything(0)
    model = FTTransformer(6, [4, 5], 16, 4, 1, 32, 0.1).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    numeric = torch.randn(16, 6, device="cuda:0")
    categorical = torch.zeros((16, 2), dtype=torch.long, device="cuda:0")
    target = torch.randint(0, 2, (16,), device="cuda:0").float()
    loss = nn.BCEWithLogitsLoss()(model(numeric, categorical), target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    require(torch.isfinite(loss).item(), "CUDA FT smoke training was non-finite")
    del model, optimizer, numeric, categorical, target, loss
    torch.cuda.empty_cache()
    return {
        "torch_version": torch.__version__,
        "torch_module_path": str(Path(torch.__file__).resolve()),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "index": 0,
        "total_memory_mib": total_mib,
        "free_memory_mib_before_run": free_mib,
        "allocator_fraction": GPU_MEMORY_FRACTION,
        "ft_transformer_forward_backward_step": "PASS",
        "cpu_fallback": False,
    }


def predict_batches(
    model: nn.Module,
    numeric: np.ndarray,
    categorical: np.ndarray,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(numeric), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(numeric))
            n = torch.from_numpy(numeric[start:stop]).to("cuda:0")
            c = torch.from_numpy(categorical[start:stop]).to("cuda:0")
            predictions.append(torch.sigmoid(model(n, c)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def train_ft(
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    preprocessor: FoldPreprocessor,
    seed: int,
    log_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    seed_everything(seed)
    model = make_model(preprocessor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FT_CONFIG["learning_rate"],
        weight_decay=FT_CONFIG["weight_decay"],
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(numeric[fit_indices]),
            torch.from_numpy(categorical[fit_indices]),
            torch.from_numpy(labels[fit_indices].astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=True,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    epoch_records = []
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        model.train()
        losses = []
        for numeric_batch, category_batch, target_batch in loader:
            numeric_batch = numeric_batch.to("cuda:0", non_blocking=True)
            category_batch = category_batch.to("cuda:0", non_blocking=True)
            target_batch = target_batch.to("cuda:0", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric_batch, category_batch)
            loss = criterion(logits, target_batch)
            require(torch.isfinite(loss).item(), "Non-finite FT training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_prediction = predict_batches(
            model, numeric[valid_indices], categorical[valid_indices]
        )
        valid_auc = float(
            roc_auc_score(labels[valid_indices], valid_prediction)
        )
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "fit_bce": float(np.mean(losses)),
                "inner_validation_auc": valid_auc,
            }
        )
        if valid_auc > best_auc + 1e-6:
            best_auc = valid_auc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if (epoch + 1) % 5 == 0:
            log_message(
                log_path,
                f"FT {checkpoint_path.name}: epoch={epoch + 1} "
                f"inner_auc={valid_auc:.6f} best_epoch={best_epoch}",
            )
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            break
    require(best_state is not None, "FT training never produced a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "preprocessor": preprocessor.state_dict(),
            "architecture": FT_CONFIG,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_inner_auc": float(best_auc),
        },
        checkpoint_path,
    )
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_inner_auc": float(best_auc),
        "epochs_run": len(epoch_records),
        "epoch_records": epoch_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def make_inner_lgbm_settings(fold_number: int) -> tuple[dict[str, Any], int, int]:
    settings = copy.deepcopy(PHASE2_MODEL_CONFIG["lightgbm"])
    rounds = settings.pop("num_boost_round")
    patience = settings.pop("early_stopping_rounds")
    settings.pop("device", None)
    seed = INNER_SEED_BASE + fold_number
    settings.update(
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
    )
    return settings, rounds, patience


def make_catboost_settings(fold_number: int, iterations: int) -> dict[str, Any]:
    settings = copy.deepcopy(PHASE2_MODEL_CONFIG["catboost"])
    settings.pop("device", None)
    settings.pop("early_stopping_rounds", None)
    settings.update(
        iterations=iterations,
        random_seed=INNER_SEED_BASE + fold_number,
        allow_writing_files=False,
        verbose=False,
    )
    return settings


def fit_matched_baseline_fold(
    inner_cat_frame: pd.DataFrame,
    inner_lgb_frame: pd.DataFrame,
    outer_cat_frame: pd.DataFrame,
    outer_lgb_frame: pd.DataFrame,
    labels: np.ndarray,
    inner_fit_indices: np.ndarray,
    inner_valid_indices: np.ndarray,
    outer_fit_indices: np.ndarray,
    outer_valid_indices: np.ndarray,
    outer_cat_test: pd.DataFrame,
    outer_lgb_test: pd.DataFrame,
    fold_number: int,
    log_path: Path,
) -> dict[str, Any]:
    lgb_settings, max_rounds, patience = make_inner_lgbm_settings(fold_number)
    inner_lgb_train = lgb.Dataset(
        inner_lgb_frame.iloc[inner_fit_indices],
        label=labels[inner_fit_indices],
        free_raw_data=True,
    )
    inner_lgb_valid = lgb.Dataset(
        inner_lgb_frame.iloc[inner_valid_indices],
        label=labels[inner_valid_indices],
        reference=inner_lgb_train,
        free_raw_data=True,
    )
    inner_lgb = lgb.train(
        lgb_settings,
        inner_lgb_train,
        num_boost_round=max_rounds,
        valid_sets=[inner_lgb_valid],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    lgb_best_iteration = max(1, int(inner_lgb.best_iteration))
    del inner_lgb, inner_lgb_train, inner_lgb_valid

    cat_settings = make_catboost_settings(
        fold_number, PHASE2_MODEL_CONFIG["catboost"]["iterations"]
    )
    cat_patience = PHASE2_MODEL_CONFIG["catboost"]["early_stopping_rounds"]
    inner_cat = CatBoostClassifier(**cat_settings)
    with GpuTelemetryMonitor() as cat_monitor:
        inner_cat.fit(
            inner_cat_frame.iloc[inner_fit_indices],
            labels[inner_fit_indices],
            cat_features=list(CATEGORICAL_COLUMNS),
            eval_set=(
                inner_cat_frame.iloc[inner_valid_indices],
                labels[inner_valid_indices],
            ),
            early_stopping_rounds=cat_patience,
            verbose=False,
        )
    require(
        cat_monitor.samples,
        "CatBoost inner baseline produced no GPU telemetry samples",
    )
    require(
        all("RTX 4050" in sample["name"] for sample in cat_monitor.samples),
        "CatBoost inner baseline telemetry reports a different GPU",
    )
    require(cat_monitor.samples, "CatBoost inner baseline produced no GPU samples")
    cat_best_iteration = max(1, int(inner_cat.best_iteration_) + 1)
    del inner_cat

    final_lgb_settings = copy.deepcopy(lgb_settings)
    final_lgb_train = lgb.Dataset(
        outer_lgb_frame.iloc[outer_fit_indices],
        label=labels[outer_fit_indices],
        free_raw_data=True,
    )
    final_lgb = lgb.train(
        final_lgb_settings,
        final_lgb_train,
        num_boost_round=lgb_best_iteration,
    )
    lgb_valid_prediction = final_lgb.predict(
        outer_lgb_frame.iloc[outer_valid_indices],
        num_iteration=lgb_best_iteration,
    ).astype(np.float32)
    lgb_test_prediction = final_lgb.predict(
        outer_lgb_test, num_iteration=lgb_best_iteration
    ).astype(np.float32)
    del final_lgb, final_lgb_train

    final_cat = CatBoostClassifier(
        **make_catboost_settings(fold_number, cat_best_iteration)
    )
    with GpuTelemetryMonitor() as final_cat_monitor:
        final_cat.fit(
            outer_cat_frame.iloc[outer_fit_indices],
            labels[outer_fit_indices],
            cat_features=list(CATEGORICAL_COLUMNS),
            verbose=False,
        )
    require(
        final_cat_monitor.samples,
        "CatBoost outer refit produced no GPU telemetry samples",
    )
    require(
        all(
            "RTX 4050" in sample["name"]
            for sample in final_cat_monitor.samples
        ),
        "CatBoost outer refit telemetry reports a different GPU",
    )
    cat_valid_prediction = final_cat.predict_proba(
        outer_cat_frame.iloc[outer_valid_indices]
    )[:, 1].astype(np.float32)
    cat_test_prediction = final_cat.predict_proba(outer_cat_test)[:, 1].astype(
        np.float32
    )
    del final_cat
    require(
        cat_monitor.samples and final_cat_monitor.samples,
        "Matched baseline GPU telemetry is incomplete",
    )
    log_message(
        log_path,
        f"Matched Phase 2 fold {fold_number}: "
        f"inner_iterations lgb={lgb_best_iteration} cat={cat_best_iteration}",
    )
    return {
        "lgb_valid": lgb_valid_prediction,
        "lgb_test": lgb_test_prediction,
        "cat_valid": cat_valid_prediction,
        "cat_test": cat_test_prediction,
        "lgb_best_iteration": lgb_best_iteration,
        "cat_best_iteration": cat_best_iteration,
        "cat_gpu_samples": cat_monitor.samples + final_cat_monitor.samples,
        "cat_gpu_monitor_errors": cat_monitor.errors + final_cat_monitor.errors,
        "cat_gpu_max_utilization": max(
            sample["utilization_percent"]
            for sample in cat_monitor.samples + final_cat_monitor.samples
        ),
        "cat_gpu_max_memory_mib": max(
            sample["memory_used_mib"]
            for sample in cat_monitor.samples + final_cat_monitor.samples
        ),
        "lightgbm_fold_auc": float(
            roc_auc_score(labels[outer_valid_indices], lgb_valid_prediction)
        ),
        "catboost_fold_auc": float(
            roc_auc_score(labels[outer_valid_indices], cat_valid_prediction)
        ),
    }


def verified_reference(
    run_dir: Path,
    prediction_name: str,
    expected_status: str,
    labels: np.ndarray,
    input_records: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = run_dir / "experiment_summary.json"
    manifest_path = run_dir / "sha256_manifest.json"
    prediction_path = run_dir / prediction_name
    require(summary_path.is_file(), f"Reference summary missing: {summary_path}")
    require(manifest_path.is_file(), f"Reference manifest missing: {manifest_path}")
    require(prediction_path.is_file(), f"Reference prediction missing: {prediction_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(summary.get("status") == expected_status, "Reference run status invalid")
    for key in ("train", "test", "sample"):
        require(
            manifest["inputs"][key]["sha256"] == input_records[key]["sha256"],
            f"{run_dir.name} raw {key} hash differs",
        )
    expected = manifest["outputs"][prediction_name]
    require(
        sha256_file(prediction_path) == expected["sha256"],
        f"Reference output hash mismatch: {prediction_name}",
    )
    predictions = np.load(prediction_path, allow_pickle=False)
    validate_probabilities(predictions, len(labels), prediction_name)
    return predictions, {
        "path": str(prediction_path.resolve()),
        "sha256": expected["sha256"],
        "status": summary["status"],
        "classification": "Post-training OOF comparator; never a model input",
    }


def verify_reference_folds(
    folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    records = {}
    for name, folder, expected_fold_prefix in (
        ("phase2", REFERENCE_DIR, "fold"),
        ("exp04", EXP04_RUN_DIR, "fold"),
    ):
        manifest_path = folder / "sha256_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
            fold_path = folder / f"{expected_fold_prefix}_{fold_number}_indices.npz"
            expected = manifest["outputs"][fold_path.name]
            require(
                sha256_file(fold_path) == expected["sha256"],
                f"{name} fold {fold_number} index hash mismatch",
            )
            with np.load(fold_path, allow_pickle=False) as archive:
                stored_fit = archive["fit_indices"]
                stored_valid = archive["valid_indices"]
                require(
                    np.array_equal(stored_fit, fit_indices)
                    and np.array_equal(stored_valid, valid_indices),
                    f"{name} fold {fold_number} indices differ",
                )
        records[name] = {
            "manifest": str(manifest_path.resolve()),
            "folds_verified": len(folds),
        }
    exp05_indices_path = EXP05_RUN_DIR / "fold_indices.npz"
    exp05_manifest = json.loads(
        (EXP05_RUN_DIR / "sha256_manifest.json").read_text(encoding="utf-8")
    )
    require(
        sha256_file(exp05_indices_path)
        == exp05_manifest["outputs"]["fold_indices.npz"]["sha256"],
        "EXP05 fold index archive hash mismatch",
    )
    exp05_indices = np.load(exp05_indices_path, allow_pickle=False)
    try:
        for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
            require(
                np.array_equal(exp05_indices[f"fit_{fold_number}"], fit_indices)
                and np.array_equal(exp05_indices[f"valid_{fold_number}"], valid_indices),
                f"EXP05 fold {fold_number} indices differ",
            )
    finally:
        exp05_indices.close()
    records["exp05"] = {
        "path": str(exp05_indices_path.resolve()),
        "folds_verified": len(folds),
    }
    return records


def auc_by_fold(
    labels: np.ndarray,
    predictions: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    return [
        float(roc_auc_score(labels[valid], predictions[valid]))
        for _, valid in folds
    ]


def compare_predictions(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    candidate_fold_auc = auc_by_fold(labels, candidate, folds)
    reference_fold_auc = auc_by_fold(labels, reference, folds)
    deltas = [
        current - base
        for current, base in zip(candidate_fold_auc, reference_fold_auc)
    ]
    candidate_auc = float(roc_auc_score(labels, candidate))
    reference_auc = float(roc_auc_score(labels, reference))
    return {
        "candidate_pooled_auc": candidate_auc,
        "reference_pooled_auc": reference_auc,
        "pooled_auc_delta": candidate_auc - reference_auc,
        "candidate_macro_fold_auc": float(np.mean(candidate_fold_auc)),
        "reference_macro_fold_auc": float(np.mean(reference_fold_auc)),
        "candidate_fold_auc": candidate_fold_auc,
        "reference_fold_auc": reference_fold_auc,
        "paired_fold_auc_delta": deltas,
        "positive_folds": sum(delta > 0 for delta in deltas),
        "oof_pearson": float(np.corrcoef(candidate, reference)[0, 1]),
        "oof_spearman": float(spearmanr(candidate, reference).statistic),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
    }


def run_experiment(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir, run_dir = data_dir.resolve(), run_dir.resolve()
    require(
        not run_dir.exists() or not any(run_dir.iterdir()),
        f"Run directory must be empty: {run_dir}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    started = time.perf_counter()

    source_paths = {
        "ft_transformer_runner": Path(__file__).resolve(),
        "phase2_raw_feature_builder": SOURCE_DIR / "run_independent_baseline.py",
        "phase1_fee_product_features": PHASE_01_DIR / "run_experiment.py",
        "phase2_target_encodings": PHASE_02_DIR / "run_experiment.py",
        "phase2_fold_local_pipeline": PHASE_04_DIR / "run_final_gate.py",
    }
    for name, path in source_paths.items():
        require(path.is_file(), f"Required source missing ({name}): {path}")
    source_records = {name: file_record(path) for name, path in source_paths.items()}
    exp05_manifest_path = EXP05_RUN_DIR / "sha256_manifest.json"
    require(exp05_manifest_path.is_file(), "EXP05 source manifest is missing")
    exp05_manifest = json.loads(exp05_manifest_path.read_text(encoding="utf-8"))
    for name, record in source_records.items():
        if name == "ft_transformer_runner":
            continue
        expected = exp05_manifest["sources"][name]
        require(
            record["sha256"] == expected["sha256"],
            f"Frozen feature source changed since EXP05: {name}",
        )
    input_paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in input_paths.values():
        require(path.is_file(), f"Raw input missing: {path}")
    input_records = {name: file_record(path) for name, path in input_paths.items()}
    for name, record in input_records.items():
        require(
            record["sha256"] == EXPECTED_INPUT_SHA256[name],
            f"Unexpected raw {name} SHA-256: {record['sha256']}",
        )

    gpu_before = gpu_preflight()
    log_message(
        log_path,
        f"CUDA preflight: {gpu_before['device']} torch={gpu_before['torch_version']} "
        f"CUDA={gpu_before['cuda_build']} free={gpu_before['free_memory_mib_before_run']}MiB",
    )
    train_raw = pd.read_csv(input_paths["train"])
    test_raw = pd.read_csv(input_paths["test"])
    sample = pd.read_csv(input_paths["sample"])
    require(
        sample.columns.tolist() == [ID_COLUMN, TARGET_COLUMN],
        "Unexpected sample submission columns",
    )
    require(
        np.array_equal(sample[ID_COLUMN].to_numpy(), test_raw[ID_COLUMN].to_numpy()),
        "Test IDs/order differ from sample submission",
    )
    require(
        train_raw[ID_COLUMN].is_unique
        and test_raw[ID_COLUMN].is_unique
        and set(train_raw[ID_COLUMN]).isdisjoint(test_raw[ID_COLUMN]),
        "Training/test applicant IDs are duplicated or overlapping",
    )
    labels = train_raw[TARGET_COLUMN].to_numpy(dtype=np.int8)
    require(set(np.unique(labels)).issubset({0, 1}), "Target must be binary")

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED
        ).split(np.zeros(len(labels)), labels)
    )
    validate_fold_partition(folds, len(labels))
    phase2_oof_path = REFERENCE_DIR / "oof_phase2_fold_local_blend.npy"
    phase2_historical_oof, phase2_oof_details = verified_reference(
        REFERENCE_DIR,
        "oof_phase2_fold_local_blend.npy",
        "COMPLETED",
        labels,
        input_records,
    )
    exp04_oof, exp04_oof_details = verified_reference(
        EXP04_RUN_DIR,
        "oof_catboost_lossguide.npy",
        "COMPLETED",
        labels,
        input_records,
    )
    exp05_oof, exp05_oof_details = verified_reference(
        EXP05_RUN_DIR,
        "oof_tabular_neural.npy",
        "COMPLETED",
        labels,
        input_records,
    )
    reference_fold_records = verify_reference_folds(folds)

    raw_train_features = build_rowwise_features(
        train_raw.drop(columns=[TARGET_COLUMN])
    )
    raw_test_features = build_rowwise_features(test_raw)
    require(
        raw_train_features.columns.tolist() == raw_test_features.columns.tolist(),
        "Rowwise train/test feature schemas differ",
    )

    candidate_oof = np.full(len(labels), np.nan, dtype=np.float32)
    matched_lgb_oof = np.full(len(labels), np.nan, dtype=np.float32)
    matched_cat_oof = np.full(len(labels), np.nan, dtype=np.float32)
    matched_blend_oof = np.full(len(labels), np.nan, dtype=np.float32)
    candidate_test_folds: list[np.ndarray] = []
    matched_test_folds: list[np.ndarray] = []
    fold_records: list[dict[str, Any]] = []
    fold_artifacts: dict[str, np.ndarray] = {}
    gpu_telemetry: list[dict[str, Any]] = []
    baseline_gpu_telemetry: list[dict[str, Any]] = []
    feature_columns: list[str] | None = None

    for fold_number, (outer_fit, outer_valid) in enumerate(folds, 1):
        fold_started = time.perf_counter()
        inner_fit, inner_valid = build_inner_split(
            labels, outer_fit, OUTER_SEED + fold_number
        )
        (
            outer_cat_frame,
            outer_cat_test,
            outer_lgb_frame,
            outer_lgb_test,
            outer_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            raw_train_features,
            raw_test_features,
            outer_fit,
            outer_valid,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        (
            inner_cat_frame,
            _,
            inner_lgb_frame,
            _,
            inner_audit,
        ) = make_variant_fold(
            "phase2_fold_local",
            train_raw,
            test_raw,
            raw_train_features,
            raw_test_features,
            inner_fit,
            inner_valid,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        validate_feature_audit(outer_audit, f"fold {fold_number} outer")
        validate_feature_audit(inner_audit, f"fold {fold_number} inner")
        require(
            outer_cat_frame.columns.tolist() == outer_cat_test.columns.tolist()
            and inner_cat_frame.columns.tolist() == outer_cat_frame.columns.tolist()
            and outer_lgb_frame.columns.tolist() == outer_lgb_test.columns.tolist(),
            f"Fold {fold_number} feature schemas differ",
        )
        if feature_columns is None:
            feature_columns = outer_cat_frame.columns.tolist()

        baseline = fit_matched_baseline_fold(
            inner_cat_frame,
            inner_lgb_frame,
            outer_cat_frame,
            outer_lgb_frame,
            labels,
            inner_fit,
            inner_valid,
            outer_fit,
            outer_valid,
            outer_cat_test,
            outer_lgb_test,
            fold_number,
            log_path,
        )
        matched_lgb_oof[outer_valid] = baseline["lgb_valid"]
        matched_cat_oof[outer_valid] = baseline["cat_valid"]
        matched_blend_oof[outer_valid] = (
            baseline["lgb_valid"].astype(np.float64)
            + baseline["cat_valid"].astype(np.float64)
        ).astype(np.float32) / 2.0
        matched_test_folds.append(
            (
                baseline["lgb_test"].astype(np.float64)
                + baseline["cat_test"].astype(np.float64)
            ).astype(np.float32)
            / 2.0
        )
        baseline_gpu_telemetry.extend(baseline["cat_gpu_samples"])

        inner_preprocessor = FoldPreprocessor(CATEGORICAL_COLUMNS).fit(
            inner_cat_frame, inner_fit
        )
        inner_numeric, inner_categorical = inner_preprocessor.transform(
            inner_cat_frame
        )
        torch.cuda.reset_peak_memory_stats(0)
        with GpuTelemetryMonitor() as monitor:
            inner_result = train_ft(
                inner_numeric,
                inner_categorical,
                labels,
                inner_fit,
                inner_valid,
                inner_preprocessor,
                INNER_MODEL_SEED_BASE + fold_number,
                log_path,
                run_dir / f"checkpoint_fold_{fold_number}_inner.pt",
            )
            selected_epochs = inner_result["best_epoch"]
            best_inner_auc = inner_result["best_inner_auc"]
            inner_ft_seconds = inner_result["elapsed_seconds"]
            del inner_result["model"]

            outer_preprocessor = FoldPreprocessor(CATEGORICAL_COLUMNS).fit(
                outer_cat_frame, outer_fit
            )
            outer_numeric, outer_categorical = outer_preprocessor.transform(
                outer_cat_frame
            )
            test_numeric, test_categorical = outer_preprocessor.transform(
                outer_cat_test
            )
            outer_seed = OUTER_SEED + 20_000 + fold_number
            seed_everything(outer_seed)
            outer_model = make_model(outer_preprocessor)
            outer_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(outer_numeric[outer_fit]),
                    torch.from_numpy(outer_categorical[outer_fit]),
                    torch.from_numpy(labels[outer_fit].astype(np.float32)),
                ),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(outer_seed),
                num_workers=0,
                pin_memory=True,
            )
            optimizer = torch.optim.AdamW(
                outer_model.parameters(),
                lr=FT_CONFIG["learning_rate"],
                weight_decay=FT_CONFIG["weight_decay"],
            )
            criterion = nn.BCEWithLogitsLoss()
            outer_refit_started = time.perf_counter()
            for _ in range(selected_epochs):
                outer_model.train()
                for numeric_batch, category_batch, target_batch in outer_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = outer_model(
                        numeric_batch.to("cuda:0", non_blocking=True),
                        category_batch.to("cuda:0", non_blocking=True),
                    )
                    loss = criterion(
                        logits, target_batch.to("cuda:0", non_blocking=True)
                    )
                    require(torch.isfinite(loss).item(), "Non-finite outer FT loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        outer_model.parameters(), max_norm=5.0
                    )
                    optimizer.step()
            torch.cuda.synchronize()
            outer_refit_seconds = time.perf_counter() - outer_refit_started
            candidate_valid = predict_batches(
                outer_model,
                outer_numeric[outer_valid],
                outer_categorical[outer_valid],
            )
            candidate_test = predict_batches(
                outer_model, test_numeric, test_categorical
            )
            final_checkpoint = run_dir / f"checkpoint_fold_{fold_number}_final.pt"
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in outer_model.state_dict().items()
                    },
                    "preprocessor": outer_preprocessor.state_dict(),
                    "architecture": FT_CONFIG,
                    "seed": outer_seed,
                    "selected_epochs": selected_epochs,
                    "inner_best_auc": best_inner_auc,
                },
                final_checkpoint,
            )
            torch.cuda.synchronize()
        require(
            monitor.samples,
            "FT GPU telemetry produced no samples",
        )
        require(
            all("RTX 4050" in item["name"] for item in monitor.samples),
            "FT telemetry did not identify the required RTX 4050",
        )
        validate_probabilities(
            candidate_valid, len(outer_valid), f"Fold {fold_number} OOF"
        )
        validate_probabilities(candidate_test, len(test_raw), "Fold test")
        candidate_oof[outer_valid] = candidate_valid
        candidate_test_folds.append(candidate_test)
        fold_artifacts[f"fit_{fold_number}"] = outer_fit.astype(np.int32)
        fold_artifacts[f"valid_{fold_number}"] = outer_valid.astype(np.int32)
        fold_auc = float(roc_auc_score(labels[outer_valid], candidate_valid))
        fold_telemetry = {
            "fold": fold_number,
            "device": gpu_before["device"],
            "sample_count": len(monitor.samples),
            "max_utilization_percent": max(
                sample["utilization_percent"] for sample in monitor.samples
            ),
            "max_memory_used_mib": max(
                sample["memory_used_mib"] for sample in monitor.samples
            ),
            "torch_peak_allocated_mib": int(
                torch.cuda.max_memory_allocated(0) // (1024 * 1024)
            ),
            "torch_peak_reserved_mib": int(
                torch.cuda.max_memory_reserved(0) // (1024 * 1024)
            ),
            "nvidia_smi_monitor_errors": monitor.errors,
        }
        gpu_telemetry.extend(monitor.samples)
        fold_record = {
            "fold": fold_number,
            "fit_rows": int(len(outer_fit)),
            "validation_rows": int(len(outer_valid)),
            "inner_fit_rows": int(len(inner_fit)),
            "inner_validation_rows": int(len(inner_valid)),
            "inner_split_seed": OUTER_SEED + fold_number,
            "inner_model_seed": INNER_MODEL_SEED_BASE + fold_number,
            "outer_refit_seed": outer_seed,
            "selected_epochs": selected_epochs,
            "inner_best_auc": best_inner_auc,
            "outer_validation_used_for_stopping": False,
            "outer_oof_auc": fold_auc,
            "oof_prediction": summarize(candidate_valid),
            "matched_phase2_lgbm_auc": baseline["lightgbm_fold_auc"],
            "matched_phase2_catboost_auc": baseline["catboost_fold_auc"],
            "matched_phase2_blend_auc": float(
                roc_auc_score(labels[outer_valid], matched_blend_oof[outer_valid])
            ),
            "matched_phase2_lgbm_best_iteration": baseline["lgb_best_iteration"],
            "matched_phase2_catboost_best_iteration": baseline["cat_best_iteration"],
            "matched_phase2_catboost_monitor_errors": baseline[
                "cat_gpu_monitor_errors"
            ],
            "ft_inner_selection_seconds": inner_ft_seconds,
            "ft_outer_refit_seconds": outer_refit_seconds,
            "elapsed_seconds": time.perf_counter() - fold_started,
            "gpu": fold_telemetry,
            "feature_audit": outer_audit,
            "inner_feature_audit": inner_audit,
            "numeric_feature_count": len(outer_preprocessor.numeric_columns),
            "categorical_feature_count": len(CATEGORICAL_COLUMNS),
            "final_checkpoint": str(final_checkpoint.resolve()),
        }
        fold_records.append(fold_record)
        baseline["cat_gpu_samples"] = []
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_indices.npz",
            fit_indices=outer_fit.astype(np.int32),
            valid_indices=outer_valid.astype(np.int32),
        )
        np.savez_compressed(
            run_dir / f"fold_{fold_number}_predictions.npz",
            valid_indices=outer_valid.astype(np.int32),
            ft_transformer=candidate_valid,
            matched_phase2_blend=matched_blend_oof[outer_valid],
            matched_phase2_lightgbm=baseline["lgb_valid"],
            matched_phase2_catboost=baseline["cat_valid"],
        )
        log_message(
            log_path,
            f"Fold {fold_number}/{N_SPLITS}: FT_AUC={fold_auc:.8f} "
            f"matched_AUC={fold_record['matched_phase2_blend_auc']:.8f} "
            f"epochs={selected_epochs} "
            f"gpu={fold_telemetry['max_utilization_percent']}% "
            f"VRAM={fold_telemetry['max_memory_used_mib']}MiB",
        )
        del outer_model, inner_numeric, inner_categorical
        del outer_numeric, outer_categorical, test_numeric, test_categorical
        torch.cuda.empty_cache()

    validate_probabilities(candidate_oof, len(labels), "FT OOF")
    validate_probabilities(matched_blend_oof, len(labels), "Matched Phase 2 OOF")
    candidate_test = np.mean(
        np.stack(candidate_test_folds), axis=0, dtype=np.float64
    ).astype(np.float32)
    matched_test = np.mean(
        np.stack(matched_test_folds), axis=0, dtype=np.float64
    ).astype(np.float32)
    validate_probabilities(candidate_test, len(test_raw), "FT test prediction")
    np.save(run_dir / "oof_ft_transformer.npy", candidate_oof)
    np.save(run_dir / "test_ft_transformer.npy", candidate_test)
    np.save(run_dir / "oof_phase2_strict_matched.npy", matched_blend_oof)
    np.save(run_dir / "oof_phase2_strict_matched_lightgbm.npy", matched_lgb_oof)
    np.save(run_dir / "oof_phase2_strict_matched_catboost.npy", matched_cat_oof)
    np.save(run_dir / "test_phase2_strict_matched.npy", matched_test)
    np.savez_compressed(run_dir / "fold_indices.npz", **fold_artifacts)

    comparisons = {
        "phase2_strict_matched": compare_predictions(
            labels, candidate_oof, matched_blend_oof, folds
        ),
        "exp04_catboost_lossguide": compare_predictions(
            labels,
            candidate_oof,
            exp04_oof,
            folds,
        ),
        "exp05_residual_mlp": compare_predictions(
            labels,
            candidate_oof,
            exp05_oof,
            folds,
        ),
        "phase2_archived_outer_stopping_diagnostic": compare_predictions(
            labels, candidate_oof, phase2_historical_oof, folds
        ),
    }
    matched_auc_delta = comparisons["phase2_strict_matched"]["pooled_auc_delta"]
    positive_folds = comparisons["phase2_strict_matched"]["positive_folds"]
    gate_passed = (
        matched_auc_delta >= ADVANCEMENT_GATE["minimum_pooled_auc_delta"]
        and positive_folds >= ADVANCEMENT_GATE["minimum_positive_folds"]
    )
    decision = "PROMOTION_CANDIDATE_PENDING_REVIEW" if gate_passed else "REJECTED"

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_raw[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: candidate_test,
        }
    )
    require(
        submission.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and submission[ID_COLUMN].is_unique
        and np.array_equal(submission[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
        "Submission columns/ID order invalid",
    )
    submission_path = run_dir / "submission_ft_transformer.csv"
    submission.to_csv(submission_path, index=False, float_format="%.10f")
    reread = pd.read_csv(submission_path)
    require(
        reread.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and np.array_equal(reread[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
        "Serialized submission schema or ID ordering is invalid",
    )
    validate_probabilities(
        reread[TARGET_COLUMN].to_numpy(dtype=np.float64),
        len(test_raw),
        "Serialized submission",
    )

    champion_path = IMPROVEMENT_DIR / "baseline" / "reference" / "EXP85_HISTORICAL_CHAMPION.csv"
    champion_record = file_record(champion_path)
    require(
        champion_record["sha256"] == EXPECTED_CHAMPION_SHA256,
        "Historical champion reference hash changed",
    )
    champion = pd.read_csv(champion_path)
    require(
        champion.columns.tolist() == [ID_COLUMN, TARGET_COLUMN]
        and np.array_equal(champion[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()),
        "Champion reference schema/IDs differ",
    )
    champion_test = champion[TARGET_COLUMN].to_numpy(dtype=np.float64)
    summary = {
        "experiment_id": run_dir.name,
        "status": "COMPLETED",
        "decision": decision,
        "lineage": "INDEPENDENT_RAW_ONLY_FT_TRANSFORMER_WITH_STRICT_MATCHED_PHASE2_BASELINE",
        "official_competition_score": "NOT_EVALUATED",
        "historical_public_score_reference": 0.801340,
        "historical_public_score_verified_for_candidate": False,
        "raw_training_labels_used": True,
        "historical_predictions_used_as_training_labels": False,
        "historical_predictions_used_as_model_inputs": False,
        "champion_reference_read_after_training_for_diagnostic_only": True,
        "champion_reference": champion_record,
        "inputs": input_records,
        "source_hashes": source_records,
        "references": {
            "phase2_archived_outer_stopping_oof": phase2_oof_details,
            "exp04_oof": exp04_oof_details,
            "exp05_oof": exp05_oof_details,
            "folds_verified": reference_fold_records,
        },
        "fold_protocol": {
            "n_splits": N_SPLITS,
            "outer_seed": OUTER_SEED,
            "matched_phase2_exp04_exp05_outer_indices": True,
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
            "outer_validation_used_for_stopping": False,
            "feature_pipeline": "Frozen Phase 2 fold-local fee/product frequency and nested target encodings",
            "inner_target_encoding": "Nested inner OOF; apply encodings use only current fit labels",
            "outer_refit": "All outer-fit rows, epoch count selected on inner validation only",
        },
        "model": {
            "parameters": FT_CONFIG,
            "gpu_preflight": gpu_before,
            "gpu_training_confirmed": all(
                fold["gpu"]["device"] == gpu_before["device"]
                for fold in fold_records
            ),
            "seeds": {
                "inner_split": [
                    OUTER_SEED + fold for fold in range(1, N_SPLITS + 1)
                ],
                "inner_training": [
                    INNER_MODEL_SEED_BASE + fold
                    for fold in range(1, N_SPLITS + 1)
                ],
                "outer_refit": [
                    OUTER_SEED + 20_000 + fold
                    for fold in range(1, N_SPLITS + 1)
                ],
            },
            "gpu_telemetry": {
                "sample_count": len(gpu_telemetry),
                "max_utilization_percent": max(
                    item["utilization_percent"] for item in gpu_telemetry
                ),
                "max_memory_used_mib": max(
                    item["memory_used_mib"] for item in gpu_telemetry
                ),
                "max_torch_allocated_mib": max(
                    fold["gpu"]["torch_peak_allocated_mib"]
                    for fold in fold_records
                ),
                "monitor_errors": [
                    error
                    for fold in fold_records
                    for error in fold["gpu"]["nvidia_smi_monitor_errors"]
                ],
            },
        },
        "matched_phase2_benchmark": {
            "definition": "Arithmetic mean of frozen Phase 2 LightGBM and CatBoost configurations, with each best iteration selected on inner validation and each model refit on full outer-fit rows",
            "outer_fold_auc": [
                fold["matched_phase2_blend_auc"] for fold in fold_records
            ],
            "lightgbm_fold_auc": [
                fold["matched_phase2_lgbm_auc"] for fold in fold_records
            ],
            "catboost_fold_auc": [
                fold["matched_phase2_catboost_auc"] for fold in fold_records
            ],
            "catboost_gpu_telemetry_samples": len(baseline_gpu_telemetry),
            "catboost_peak_utilization_percent": max(
                item["utilization_percent"] for item in baseline_gpu_telemetry
            ),
            "catboost_peak_memory_used_mib": max(
                item["memory_used_mib"] for item in baseline_gpu_telemetry
            ),
            "monitor_errors": [
                error
                for fold in fold_records
                for error in fold["matched_phase2_catboost_monitor_errors"]
            ],
        },
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns or []),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "catboost": catboost.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "fold_results": fold_records,
        "comparisons": comparisons,
        "exp05_oof_reference_sha256": exp05_oof_details["sha256"],
        "advancement_gate": {
            **ADVANCEMENT_GATE,
            "pooled_auc_delta": matched_auc_delta,
            "positive_folds": positive_folds,
            "passed": gate_passed,
            "decision": decision,
        },
        "prediction_diagnostics": {
            "oof": summarize(candidate_oof),
            "test": summarize(candidate_test),
            "test_pearson_historical_champion_diagnostic_only": float(
                np.corrcoef(candidate_test, champion_test)[0, 1]
            ),
            "test_spearman_historical_champion_diagnostic_only": float(
                spearmanr(candidate_test, champion_test).statistic
            ),
        },
        "candidate_submission": {
            "path": str(submission_path.resolve()),
            "sha256": sha256_file(submission_path),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "unique_ids": int(submission[ID_COLUMN].nunique()),
            "prediction_min": float(candidate_test.min()),
            "prediction_max": float(candidate_test.max()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name, values in (
        ("oof_ft_transformer.npy", candidate_oof),
        ("test_ft_transformer.npy", candidate_test),
        ("oof_phase2_strict_matched.npy", matched_blend_oof),
        ("oof_phase2_strict_matched_lightgbm.npy", matched_lgb_oof),
        ("oof_phase2_strict_matched_catboost.npy", matched_cat_oof),
        ("test_phase2_strict_matched.npy", matched_test),
    ):
        summary.setdefault("outputs", {})[name] = {
            "sha256": sha256_file(run_dir / name),
            "size_bytes": (run_dir / name).stat().st_size,
        }
    write_json(run_dir / "feature_columns.json", feature_columns or [])
    write_json(run_dir / "gpu_telemetry.json", gpu_telemetry)
    write_json(run_dir / "matched_phase2_gpu_telemetry.json", baseline_gpu_telemetry)
    write_json(run_dir / "fold_metrics.json", fold_records)
    write_json(run_dir / "experiment_summary.json", summary)

    manifest = {
        "experiment_id": run_dir.name,
        "inputs": {
            **input_records,
            "phase2_archived_oof": file_record(phase2_oof_path),
            "phase2_manifest": file_record(phase2_manifest_path),
            "exp04_oof": file_record(EXP04_RUN_DIR / "oof_catboost_lossguide.npy"),
            "exp05_oof": file_record(EXP05_RUN_DIR / "oof_tabular_neural.npy"),
            "champion_diagnostic_only": champion_record,
        },
        "sources": source_records,
        "outputs": {
            str(path.relative_to(run_dir)): file_record(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "sha256_manifest.json"
        },
    }
    write_json(run_dir / "sha256_manifest.json", manifest)
    log_message(
        log_path,
        f"Decision={decision} FT_OOF_AUC="
        f"{comparisons['phase2_strict_matched']['candidate_pooled_auc']:.8f} "
        f"Matched_Phase2_AUC="
        f"{comparisons['phase2_strict_matched']['reference_pooled_auc']:.8f} "
        f"delta={matched_auc_delta:+.8f}",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=EXPERIMENT_DIR
        / "runs"
        / (
            "exp06_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_ft_transformer_seed20261101"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir, args.run_dir)


if __name__ == "__main__":
    main()
