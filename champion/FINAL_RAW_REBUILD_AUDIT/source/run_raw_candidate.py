from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
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
FORBIDDEN_PREDICTION_READS = {
    "np.load",
    "numpy.load",
    "joblib.load",
    "torch.load",
    "pickle.load",
    "pd.read_feather",
    "pandas.read_feather",
    "pd.read_parquet",
    "pandas.read_parquet",
}
EXPECTED_V17_KEYS = {
    "oof_old", "test_old", "oof_e1", "test_e1", "oof_e1et", "test_e1et",
    "oof_e1Aet", "test_e1Aet", "oof_cb", "test_cb", "final_oof",
    "final_test", "final_auc",
}
EXPECTED_V55_KEYS = {
    "oof_qf_shallow", "oof_qf_reg", "oof_v1_allfam_oh", "oof_v2_allfam_oh",
    "oof_v2_mono", "oof_deep_et", "oof_cb_product", "test_qf_shallow",
    "test_qf_reg", "test_v1_allfam_oh", "test_v2_allfam_oh", "test_v2_mono",
    "test_deep_et", "test_cb_product",
}
EXPECTED_OUTPUTS = (
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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted_name(node.value)}.{node.attr}"
    return ""


def audit_generator_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden = []
    csv_reads = []
    local_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name in FORBIDDEN_PREDICTION_READS:
                forbidden.append({"line": node.lineno, "call": name})
            if name in {"pd.read_csv", "pandas.read_csv"}:
                csv_reads.append(
                    {
                        "line": node.lineno,
                        "argument": ast.get_source_segment(source, node.args[0])
                        if node.args
                        else None,
                    }
                )
        elif isinstance(node, ast.ImportFrom) and node.level:
            local_imports.append({"line": node.lineno, "module": node.module})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "forbidden_prediction_reads": forbidden,
        "pandas_csv_reads": csv_reads,
        "relative_local_imports": local_imports,
        "static_raw_only_check_passed": not forbidden and not local_imports,
    }


def run_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"command": command, "error": str(exc)}


def package_versions() -> dict[str, str | None]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "scipy": "scipy",
    }
    result = {}
    for display_name, distribution in packages.items():
        try:
            result[display_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[display_name] = None
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def validate_components(output_dir: Path, raw_dir: Path) -> dict[str, Any]:
    raw_train = pd.read_csv(raw_dir / "train.csv", usecols=["SK_ID_CURR", "TARGET"])
    raw_test = pd.read_csv(raw_dir / "test.csv", usecols=["SK_ID_CURR"])
    y = raw_train["TARGET"].to_numpy(dtype=np.int32)
    train_ids = raw_train["SK_ID_CURR"].to_numpy()
    test_ids = raw_test["SK_ID_CURR"].to_numpy()
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Raw train/test IDs are not unique.")

    y_output = np.load(output_dir / "y_train.npy", allow_pickle=False)
    if not np.array_equal(y_output, y):
        raise ValueError("Generated y_train.npy does not match raw train TARGET exactly.")
    xgb_oof = np.load(output_dir / "oof_10f_xgb.npy", allow_pickle=False)
    xgb_test = np.load(output_dir / "test_10f_xgb.npy", allow_pickle=False)
    if xgb_oof.shape != y.shape or xgb_test.shape != test_ids.shape:
        raise ValueError("XGB OOF/test row counts do not match raw CSVs.")
    if not np.isfinite(xgb_oof).all() or not np.isfinite(xgb_test).all():
        raise ValueError("XGB OOF/test arrays contain non-finite values.")
    if (
        np.any((xgb_oof < 0) | (xgb_oof > 1))
        or np.any((xgb_test < 0) | (xgb_test > 1))
    ):
        raise ValueError("XGB OOF/test predictions fall outside [0, 1].")

    archive_records = {}
    for filename, required_keys, train_prefix, test_prefix in (
        ("v_17_full_suite_bundle.npz", EXPECTED_V17_KEYS, "oof_", "test_"),
        ("v55_members.npz", EXPECTED_V55_KEYS, "oof_", "test_"),
    ):
        with np.load(output_dir / filename, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != required_keys:
                raise ValueError(
                    f"{filename} keys differ: missing={sorted(required_keys - keys)}, "
                    f"extra={sorted(keys - required_keys)}"
                )
            member_shapes = {}
            for key in archive.files:
                array = archive[key]
                member_shapes[key] = list(array.shape)
                if key.startswith(train_prefix) and array.shape != y.shape:
                    raise ValueError(f"{filename}:{key} has wrong train row count.")
                if key.startswith(test_prefix) and array.shape != test_ids.shape:
                    raise ValueError(f"{filename}:{key} has wrong test row count.")
                if not np.isfinite(array).all():
                    raise ValueError(f"{filename}:{key} contains non-finite values.")
                if array.ndim and np.any((array < 0) | (array > 1)):
                    raise ValueError(f"{filename}:{key} contains values outside [0, 1].")
                if key == "final_auc" and not 0 <= float(array) <= 1:
                    raise ValueError(f"{filename}:{key} is outside [0, 1].")
        archive_records[filename] = {
            "keys": sorted(keys),
            "shapes": member_shapes,
            "sha256": sha256_file(output_dir / filename),
        }

    anchor_path = output_dir / "v_20261027_precision_transductive_apex.csv"
    anchor = pd.read_csv(anchor_path)
    if list(anchor.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Anchor74 columns differ from the expected submission schema.")
    if len(anchor) != len(test_ids) or not np.array_equal(
        anchor["SK_ID_CURR"].to_numpy(), test_ids
    ):
        raise ValueError("Anchor74 ID order differs from raw test.csv.")
    anchor_values = anchor["TARGET"].to_numpy(dtype=np.float64)
    if not np.isfinite(anchor_values).all() or np.any((anchor_values < 0) | (anchor_values > 1)):
        raise ValueError("Anchor74 targets are non-finite or outside [0, 1].")

    return {
        "raw_train_rows": len(y),
        "raw_test_rows": len(test_ids),
        "raw_train_ids_unique": True,
        "raw_test_ids_unique": True,
        "y_train_exact_raw_target_match": True,
        "xgb_oof_shape": list(xgb_oof.shape),
        "xgb_test_shape": list(xgb_test.shape),
        "xgb_arrays_finite": True,
        "archives": archive_records,
        "anchor74": {
            "sha256": sha256_file(anchor_path),
            "rows": len(anchor),
            "ids_exact_raw_order": True,
            "finite_targets": True,
            "target_min": float(anchor_values.min()),
            "target_max": float(anchor_values.max()),
        },
    }


def run_command(
    command: list[str], cwd: Path, log_path: Path, source_hash: str | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.write(f"CWD: {cwd}\n")
        log.flush()
        result = subprocess.run(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    elapsed = time.monotonic() - started
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nEXIT_CODE: {result.returncode}\n")
        log.write(f"ELAPSED_SECONDS: {elapsed:.6f}\n")
    record = {
        "command": command,
        "working_directory": str(cwd),
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "log_size_bytes": log_path.stat().st_size,
    }
    if source_hash is not None:
        record["source_sha256"] = source_hash
    return record


def run_gpu_preflight(log_path: Path) -> dict[str, Any]:
    preflight = (
        "import numpy as np, xgboost as xgb; "
        "from catboost import CatBoostClassifier; "
        "X=np.arange(60,dtype=np.float32).reshape(30,2); "
        "y=np.tile([0,1],15); "
        "xgb.XGBClassifier(n_estimators=2,tree_method='hist',device='cuda',"
        "eval_metric='logloss').fit(X,y); "
        "CatBoostClassifier(iterations=2,task_type='GPU',verbose=False)"
        ".fit(X,y); "
        "print('XGBoost CUDA and CatBoost GPU preflight passed')"
    )
    return run_command(
        [sys.executable, "-c", preflight],
        log_path.parent.parent,
        log_path,
    )


def assert_gpu_execution(script_name: str, log_path: Path) -> None:
    log = log_path.read_text(encoding="utf-8")
    if script_name == "generate_xgb10_from_scratch.py":
        required = "XGBoost Compute Device: NVIDIA CUDA GPU"
    elif script_name in {
        "generate_v17_from_scratch.py",
        "generate_v55_from_scratch.py",
    }:
        required = "CatBoost Compute Device: NVIDIA CUDA GPU"
    else:
        required = None
    if required is not None and required not in log:
        raise RuntimeError(
            f"{script_name} did not confirm required GPU use; refusing CPU fallback."
        )
    if script_name == "generate_anchor74_from_scratch.py":
        if (
            "xgb Complete (cuda)" not in log
            or "cat Complete (GPU)" not in log
        ):
            raise RuntimeError(
                "Anchor74 did not confirm XGBoost CUDA and CatBoost GPU; "
                "refusing CPU fallback."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full raw-only component run and EXP85 candidate validation in an isolated directory."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-champion", type=Path, required=True)
    parser.add_argument("--previous-candidate", type=Path)
    args = parser.parse_args()

    repo = args.repository_root.resolve()
    raw_dir = args.raw_dir.resolve()
    run_dir = args.run_dir.resolve()
    audit_root = repo / "champion" / "FINAL_RAW_REBUILD_AUDIT"
    try:
        run_dir.relative_to(audit_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Run directory must be under {audit_root}.") from exc
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    frozen_champion = args.frozen_champion.resolve()
    if not frozen_champion.is_file():
        raise FileNotFoundError(frozen_champion)
    for name in ("train.csv", "test.csv"):
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(raw_dir / name)

    source_dir = repo / "champion" / "02_EXP85_REBUILT" / "source"
    audit_source = audit_root / "source"
    for name in GENERATOR_NAMES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)
    if not (audit_source / "raw_candidate_formula.py").is_file():
        raise FileNotFoundError(audit_source / "raw_candidate_formula.py")
    if not (audit_source / "validate_raw_candidate.py").is_file():
        raise FileNotFoundError(audit_source / "validate_raw_candidate.py")

    generator_audits = [audit_generator_source(source_dir / name) for name in GENERATOR_NAMES]
    if any(not entry["static_raw_only_check_passed"] for entry in generator_audits):
        raise RuntimeError("Static audit found a historical prediction/cache reader or local import.")

    run_dir.mkdir(parents=True, exist_ok=False)
    component_dir = run_dir / "components"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    for directory in (component_dir, logs_dir, reports_dir):
        directory.mkdir()

    raw_records = {
        filename: {
            "path": str(raw_dir / filename),
            "size_bytes": (raw_dir / filename).stat().st_size,
            "sha256": sha256_file(raw_dir / filename),
        }
        for filename in ("train.csv", "test.csv")
    }
    source_records = {
        entry["path"]: {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
        for entry in generator_audits
    }

    execution: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "classification": "RAW_COMPONENT_RUN; HISTORICAL_PARITY_NOT_ESTABLISHED",
        "full_mode": True,
        "fast_mode": False,
        "raw_rebuild_certified": False,
        "historical_parity_verified": False,
        "official_leaderboard_score": None,
        "note": (
            "This run regenerates the currently available raw-component candidate. "
            "It does not establish historical model parity or reproduce an official score."
        ),
        "repository_head": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "raw_inputs": raw_records,
        "generator_source_audit": generator_audits,
        "source_hashes": source_records,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
            "gpu_before": run_nvidia_smi(),
        },
        "generator_executions": [],
    }
    execution_path = run_dir / "execution.json"
    write_json(execution_path, execution)

    preflight = run_gpu_preflight(logs_dir / "00_gpu_preflight.txt")
    execution["gpu_preflight"] = preflight
    execution["environment"]["gpu_after_preflight"] = run_nvidia_smi()
    write_json(execution_path, execution)
    if preflight["exit_code"] != 0:
        execution["status"] = "FAILED_GPU_PREFLIGHT"
        write_json(execution_path, execution)
        raise RuntimeError(
            "GPU-only preflight failed; refusing to run generators that can fall back to CPU."
        )

    for index, name in enumerate(GENERATOR_NAMES, start=1):
        source_path = source_dir / name
        command = [
            sys.executable,
            str(source_path),
            "--data_dir",
            str(raw_dir),
            "--output_dir",
            str(component_dir),
        ]
        record = run_command(
            command,
            run_dir,
            logs_dir / f"{index:02d}_{Path(name).stem}.txt",
            sha256_file(source_path),
        )
        record["script"] = name
        record["mode"] = "full; --fast omitted"
        execution["generator_executions"].append(record)
        write_json(execution_path, execution)
        if record["exit_code"] != 0:
            execution["status"] = "FAILED"
            execution["failure_step"] = name
            write_json(execution_path, execution)
            raise subprocess.CalledProcessError(
                record["exit_code"], command
            )
        try:
            assert_gpu_execution(name, logs_dir / f"{index:02d}_{Path(name).stem}.txt")
        except RuntimeError as exc:
            execution["status"] = "FAILED"
            execution["failure_step"] = name
            execution["failure_reason"] = str(exc)
            write_json(execution_path, execution)
            raise

    missing = [name for name in EXPECTED_OUTPUTS if not (component_dir / name).is_file()]
    if missing:
        execution["status"] = "FAILED"
        execution["missing_outputs"] = missing
        write_json(execution_path, execution)
        raise FileNotFoundError(f"Raw generators did not produce: {missing}")

    execution["component_validation"] = validate_components(component_dir, raw_dir)

    candidate_path = run_dir / "candidate_submission.csv"
    blend_path = audit_source / "raw_candidate_formula.py"
    blend_command = [
        sys.executable, str(blend_path),
        "--raw-dir", str(raw_dir),
        "--components-dir", str(component_dir),
        "--output", str(candidate_path),
    ]
    blend_record = run_command(
        blend_command, run_dir, logs_dir / "05_synthesize_candidate.txt",
        sha256_file(blend_path),
    )
    execution["blend_execution"] = blend_record
    write_json(execution_path, execution)
    if blend_record["exit_code"] != 0:
        execution["status"] = "FAILED"
        write_json(execution_path, execution)
        raise subprocess.CalledProcessError(blend_record["exit_code"], blend_command)

    validation_path = reports_dir / "final_validation.json"
    validator_path = audit_source / "validate_raw_candidate.py"
    validator_command = [
        sys.executable, str(validator_path),
        "--candidate", str(candidate_path),
        "--components-dir", str(component_dir),
        "--raw-dir", str(raw_dir),
        "--frozen-champion", str(frozen_champion),
        "--output", str(validation_path),
    ]
    if args.previous_candidate is not None:
        validator_command.extend(
            ["--previous-candidate", str(args.previous_candidate.resolve())]
        )
    validator_record = run_command(
        validator_command, run_dir, logs_dir / "06_validate_candidate.txt",
        sha256_file(validator_path),
    )
    execution["validation_execution"] = validator_record
    if validator_record["exit_code"] != 0:
        execution["status"] = "FAILED"
        write_json(execution_path, execution)
        raise subprocess.CalledProcessError(
            validator_record["exit_code"], validator_command
        )

    execution["candidate"] = {
        "path": str(candidate_path),
        "sha256": sha256_file(candidate_path),
        "size_bytes": candidate_path.stat().st_size,
    }
    execution["validation_report"] = {
        "path": str(validation_path),
        "sha256": sha256_file(validation_path),
        "size_bytes": validation_path.stat().st_size,
    }
    execution["component_outputs"] = {
        name: {
            "path": str(component_dir / name),
            "sha256": sha256_file(component_dir / name),
            "size_bytes": (component_dir / name).stat().st_size,
        }
        for name in EXPECTED_OUTPUTS
    }
    execution["environment"]["gpu_after"] = run_nvidia_smi()
    execution["status"] = "FULL_RAW_CANDIDATE_COMPLETE; HISTORICAL_PARITY_NOT_ESTABLISHED"
    execution["elapsed_seconds"] = sum(
        item["elapsed_seconds"] for item in execution["generator_executions"]
    ) + blend_record["elapsed_seconds"] + validator_record["elapsed_seconds"]
    write_json(execution_path, execution)
    print(f"Execution evidence: {execution_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Status: {execution['status']}")


if __name__ == "__main__":
    main()
