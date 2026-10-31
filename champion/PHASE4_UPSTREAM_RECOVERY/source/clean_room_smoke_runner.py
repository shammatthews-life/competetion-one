from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GENERATOR_NAMES = (
    "generate_xgb10_from_scratch.py",
    "generate_v55_from_scratch.py",
    "generate_v17_from_scratch.py",
    "generate_anchor74_from_scratch.py",
)
EXPECTED_ARTIFACTS = (
    "oof_10f_xgb.npy",
    "test_10f_xgb.npy",
    "y_train.npy",
    "v55_members.npz",
    "v_17_full_suite_bundle.npz",
    "v_20261027_precision_transductive_apex.csv",
)


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test raw-only generators in an isolated staging directory."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite staging directory: {args.staging_dir}")
    for filename in ("train.csv", "test.csv"):
        if not (args.raw_dir / filename).is_file():
            raise FileNotFoundError(f"Missing raw input: {args.raw_dir / filename}")
    for filename in GENERATOR_NAMES:
        if not (args.source_dir / filename).is_file():
            raise FileNotFoundError(f"Missing generator source: {args.source_dir / filename}")

    staging_dir = args.staging_dir.resolve()
    source_stage = staging_dir / "source"
    raw_stage = staging_dir / "raw"
    output_stage = staging_dir / "artifacts"
    log_stage = staging_dir / "logs"
    for directory in (source_stage, raw_stage, output_stage, log_stage):
        directory.mkdir(parents=True, exist_ok=False)

    for filename in GENERATOR_NAMES:
        shutil.copy2(args.source_dir / filename, source_stage / filename)
    for filename in ("train.csv", "test.csv"):
        shutil.copy2(args.raw_dir / filename, raw_stage / filename)

    records: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "provenance_status": "SMOKE_ONLY_NOT_CERTIFIED",
        "raw_rebuild_certified": False,
        "scope": (
            "Isolated fast smoke test only: three-fold/short-iteration generator modes; "
            "not a full model rebuild and not historical parity evidence."
        ),
        "staging_directory": str(staging_dir),
        "raw_inputs": {
            filename: file_record(raw_stage / filename)
            for filename in ("train.csv", "test.csv")
        },
        "source_files": {
            filename: file_record(source_stage / filename)
            for filename in GENERATOR_NAMES
        },
        "executions": [],
        "outputs": {},
    }
    evidence_path = staging_dir.parent / "execution.json"
    write_json(evidence_path, records)

    for index, filename in enumerate(GENERATOR_NAMES, start=1):
        command = [
            sys.executable,
            str(source_stage / filename),
            "--fast",
            "--data_dir",
            str(raw_stage),
            "--output_dir",
            str(output_stage),
        ]
        log_path = log_stage / f"{index:02d}_{Path(filename).stem}.txt"
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=source_stage,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - started
        log_path.write_text(
            "COMMAND: " + subprocess.list2cmdline(command) + "\n"
            + f"EXIT_CODE: {result.returncode}\n"
            + f"ELAPSED_SECONDS: {elapsed:.6f}\n"
            + "\n--- STDOUT ---\n"
            + result.stdout
            + "\n--- STDERR ---\n"
            + result.stderr,
            encoding="utf-8",
        )
        execution = {
            "step": index,
            "script": filename,
            "command": command,
            "exit_code": result.returncode,
            "elapsed_seconds": elapsed,
            "log": file_record(log_path),
        }
        records["executions"].append(execution)
        write_json(evidence_path, records)
        if result.returncode != 0:
            records["status"] = "FAILED"
            write_json(evidence_path, records)
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout, stderr=result.stderr
            )

    raw_test_ids = pd.read_csv(raw_stage / "test.csv", usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    raw_train = pd.read_csv(raw_stage / "train.csv", usecols=["TARGET"])
    for filename in EXPECTED_ARTIFACTS:
        path = output_stage / filename
        if not path.is_file():
            raise FileNotFoundError(f"Generator output missing: {path}")
        record = file_record(path)
        if filename.endswith(".npy"):
            array = np.load(path, allow_pickle=False)
            record.update(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "finite": bool(np.isfinite(array).all()),
                }
            )
        elif filename == "v55_members.npz":
            with np.load(path, allow_pickle=False) as archive:
                record["keys"] = sorted(archive.files)
                record["all_arrays_finite"] = all(
                    np.isfinite(archive[key]).all()
                    for key in archive.files
                    if archive[key].ndim > 0
                )
        elif filename == "v_17_full_suite_bundle.npz":
            with np.load(path, allow_pickle=False) as archive:
                record["keys"] = sorted(archive.files)
                record["all_arrays_finite"] = all(
                    np.isfinite(archive[key]).all()
                    for key in archive.files
                    if archive[key].ndim > 0
                )
        else:
            submission = pd.read_csv(path)
            record.update(
                {
                    "rows": len(submission),
                    "columns": list(submission.columns),
                    "id_order_matches_raw_test": bool(
                        np.array_equal(
                            submission["SK_ID_CURR"].to_numpy(), raw_test_ids
                        )
                    ),
                    "target_finite": bool(
                        np.isfinite(submission["TARGET"].to_numpy()).all()
                    ),
                }
            )
        records["outputs"][filename] = record

    y_path = output_stage / "y_train.npy"
    y_generated = np.load(y_path, allow_pickle=False)
    raw_y = raw_train["TARGET"].to_numpy()
    if not np.array_equal(y_generated, raw_y):
        raise ValueError("Smoke-generated labels do not match raw train TARGET.")
    if len(np.load(output_stage / "oof_10f_xgb.npy", allow_pickle=False)) != len(raw_y):
        raise ValueError("Smoke-generated XGB OOF row count differs from raw train.")
    if len(np.load(output_stage / "test_10f_xgb.npy", allow_pickle=False)) != len(raw_test_ids):
        raise ValueError("Smoke-generated XGB test row count differs from raw test.")
    if not records["outputs"]["v_20261027_precision_transductive_apex.csv"][
        "id_order_matches_raw_test"
    ]:
        raise ValueError("Smoke-generated Anchor74 IDs do not match raw test order.")

    records["status"] = "SMOKE_COMPLETE_NOT_CERTIFIED"
    records["total_elapsed_seconds"] = sum(
        execution["elapsed_seconds"] for execution in records["executions"]
    )
    write_json(evidence_path, records)
    print(f"Smoke evidence: {evidence_path}")


if __name__ == "__main__":
    main()
