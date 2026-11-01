from __future__ import annotations

import subprocess
import threading
from typing import Any


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {"available": True, "query": result.stdout.strip()}


def require_gpu_preflight() -> dict[str, Any]:
    snapshot = gpu_snapshot()
    if "NVIDIA GeForce RTX 4050" not in snapshot["query"]:
        raise RuntimeError(f"Expected RTX 4050 GPU; detected: {snapshot['query']}")
    from catboost import CatBoostClassifier
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        {
            "numeric": np.arange(100, dtype=np.float32),
            "category": pd.Series(["a", "b"] * 50, dtype="object"),
        }
    )
    labels = np.asarray([0, 1] * 50, dtype=np.int8)
    model = CatBoostClassifier(
        iterations=2,
        depth=2,
        task_type="GPU",
        devices="0",
        allow_writing_files=False,
        verbose=False,
        thread_count=1,
    )
    model.fit(frame, labels, cat_features=["category"], verbose=False)
    return {
        "gpu_snapshot": snapshot,
        "catboost_gpu_smoke_test": "PASS",
        "catboost_gpu_task_type": model.get_params()["task_type"],
        "catboost_gpu_devices": model.get_params()["devices"],
        "smoke_test_tree_count": int(model.tree_count_),
    }


class GpuMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.samples.append({"query": result.stdout.strip()})
        except (OSError, subprocess.SubprocessError) as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> GpuMonitor:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                self.errors.append("GPU telemetry sampling thread did not stop.")
        self._sample()

    def report(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "sample_count": len(self.samples),
            "errors": self.errors,
        }
