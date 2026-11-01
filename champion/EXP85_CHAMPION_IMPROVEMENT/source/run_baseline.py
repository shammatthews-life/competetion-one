from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
BASELINE = WORKSPACE / "baseline"
ARTIFACTS = BASELINE / "artifacts"
ORIGINAL_SOURCE = (
    BASELINE / "source" / "build_exp85_optimal_macro_apex.py"
)
REFERENCE = (
    BASELINE / "reference" / "EXP85_HISTORICAL_CHAMPION.csv"
)
RAW_TEST = Path(r"D:\gci\inpuy\test.csv")
RAW_TRAIN = Path(r"D:\gci\inpuy\train.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not ORIGINAL_SOURCE.is_file() or not REFERENCE.is_file():
        raise FileNotFoundError(
            "Run source\\prepare_baseline.py before executing the baseline."
        )
    manifest_path = BASELINE / "BASELINE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["direct_synthesis_inputs"].values():
        snapshot = Path(record["path"])
        if not snapshot.is_file() or sha256_file(snapshot) != record["sha256"]:
            raise RuntimeError(f"Frozen baseline input hash mismatch: {snapshot}")
    if sha256_file(REFERENCE) != manifest["historical_champion_reference"]["sha256"]:
        raise RuntimeError("Frozen historical champion reference hash changed")
    if not RAW_TRAIN.is_file() or not RAW_TEST.is_file():
        raise FileNotFoundError("Raw train.csv and test.csv are required for checks.")
    for name, path in (("train.csv", RAW_TRAIN), ("test.csv", RAW_TEST)):
        expected = manifest["raw_inputs_hashed_not_copied"][name]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Raw input hash changed: {path}")
    cached_labels = np.load(
        ARTIFACTS / "y_train.npy", allow_pickle=False
    )
    raw_labels = pd.read_csv(RAW_TRAIN, usecols=["TARGET"])["TARGET"].to_numpy()
    if not np.array_equal(cached_labels, raw_labels):
        raise ValueError("Frozen label array differs from raw train.csv TARGET")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = WORKSPACE / "baseline" / "validation_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    replacements = {
        'ANCHOR74_PATH = "d:/gci/gemini/v_20261027_precision_transductive_apex.csv"':
            f'ANCHOR74_PATH = {json.dumps(str(ARTIFACTS / "v_20261027_precision_transductive_apex.csv"))}',
        'CHAMP79_PATH  = "d:/gci/gemini/v_20261027_grand_master_macro_stacker_apex.csv"':
            f'CHAMP79_PATH  = {json.dumps(str(ARTIFACTS / "v_20261027_grand_master_macro_stacker_apex.csv"))}',
        'V17_PATH      = "d:/gci/gemini/cache/v_17_full_suite_bundle.npz"':
            f'V17_PATH      = {json.dumps(str(ARTIFACTS / "v_17_full_suite_bundle.npz"))}',
        'V55_PATH      = "d:/gci/gemini/cache/v55_members.npz"':
            f'V55_PATH      = {json.dumps(str(ARTIFACTS / "v55_members.npz"))}',
        'XGB10_OOF     = "d:/gci/gemini/cache/oof_10f_xgb.npy"':
            f'XGB10_OOF     = {json.dumps(str(ARTIFACTS / "oof_10f_xgb.npy"))}',
        'XGB10_TEST    = "d:/gci/gemini/cache/test_10f_xgb.npy"':
            f'XGB10_TEST    = {json.dumps(str(ARTIFACTS / "test_10f_xgb.npy"))}',
        'Y_PATH        = "d:/gci/gemini/cache/y_train.npy"':
            f'Y_PATH        = {json.dumps(str(ARTIFACTS / "y_train.npy"))}',
        'OUT_PATH      = "d:/gci/gemini/v_20261027_optimal_macro_apex_v85.csv"':
            f'OUT_PATH      = {json.dumps(str(run_dir / "baseline_submission.csv"))}',
        'EXP_LOG       = "d:/gci/gemini/experminetbygeminioct22.csv"':
            f'EXP_LOG       = {json.dumps(str(run_dir / "isolated_experiment_log.csv"))}',
    }

    original_text = ORIGINAL_SOURCE.read_text(encoding="utf-8")
    rewritten_text = original_text
    for old, new in replacements.items():
        if original_text.count(old) != 1:
            raise RuntimeError(
                f"Expected exactly one occurrence of baseline path {old!r}"
            )
        rewritten_text = rewritten_text.replace(old, new)
    isolated_source = run_dir / "build_exp85_optimal_macro_apex_isolated.py"
    isolated_source.write_text(rewritten_text, encoding="utf-8")

    started = time.perf_counter()
    execution = subprocess.run(
        [sys.executable, str(isolated_source)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started
    (run_dir / "stdout.log").write_text(execution.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(execution.stderr, encoding="utf-8")
    if execution.returncode != 0:
        raise RuntimeError(
            f"Isolated baseline exited {execution.returncode}; see {run_dir}"
        )

    output = run_dir / "baseline_submission.csv"
    if not output.is_file():
        raise FileNotFoundError(f"Baseline source did not write {output}")
    if not RAW_TEST.is_file():
        raise FileNotFoundError(f"Raw test IDs are required for alignment: {RAW_TEST}")
    submission = pd.read_csv(output)
    reference = pd.read_csv(REFERENCE)
    test_ids = pd.read_csv(RAW_TEST, usecols=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy()
    if list(submission.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"Unexpected output schema: {list(submission.columns)}")
    if len(submission) != 61_500 or len(test_ids) != 61_500:
        raise ValueError("Unexpected submission/test row count")
    if not submission["SK_ID_CURR"].is_unique:
        raise ValueError("Isolated baseline contains duplicate IDs")
    if not np.array_equal(submission["SK_ID_CURR"].to_numpy(), test_ids):
        raise ValueError("Isolated baseline IDs do not match raw test order")
    values = submission["TARGET"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Isolated baseline predictions are invalid")

    ref_ids = reference["SK_ID_CURR"].to_numpy()
    ref_values = reference["TARGET"].to_numpy(dtype=np.float64)
    if not np.array_equal(ref_ids, test_ids):
        raise ValueError("Frozen champion IDs do not match raw test order")
    if not np.array_equal(values, ref_values):
        raise ValueError("Isolated output differs numerically from frozen champion")
    if sha256_file(output) != sha256_file(REFERENCE):
        raise ValueError("Isolated output is not byte-identical to frozen champion")

    manifest = {
        "run_id": run_id,
        "classification": "HISTORICAL_CACHE_ASSISTED_EXECUTION",
        "raw_data_model_retraining": False,
        "command": [sys.executable, str(isolated_source)],
        "baseline_manifest_sha256": sha256_file(manifest_path),
        "exit_code": execution.returncode,
        "elapsed_seconds": elapsed_seconds,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "gpu": "Not used; the historical synthesis performs prediction blending only.",
        "baseline_inputs": {
            name: {
                "path": str(Path(record["path"])),
                "sha256": record["sha256"],
            }
            for name, record in manifest["direct_synthesis_inputs"].items()
        },
        "raw_inputs": {
            name: {
                "path": str(RAW_TRAIN if name == "train.csv" else RAW_TEST),
                "sha256": manifest["raw_inputs_hashed_not_copied"][name]["sha256"],
            }
            for name in ("train.csv", "test.csv")
        },
        "source_original": {
            "path": str(ORIGINAL_SOURCE),
            "sha256": sha256_file(ORIGINAL_SOURCE),
        },
        "source_isolated": {
            "path": str(isolated_source),
            "sha256": sha256_file(isolated_source),
            "changes": "Redirected direct-input reads and output/log paths only.",
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
        },
        "historical_reference": {
            "path": str(REFERENCE),
            "sha256": sha256_file(REFERENCE),
        },
        "comparison": {
            "byte_identical": True,
            "value_identical": True,
            "rows": len(submission),
            "test_ids_unique_and_ordered": True,
            "finite_probabilities_in_unit_interval": True,
        },
        "official_public_score": (
            "NOT_EVALUATED_BY_THIS_LOCAL_REPLAY; historical source reports 0.801340"
        ),
    }
    manifest_path = run_dir / "execution_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
