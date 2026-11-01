from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from data_validation import sha256_file
from submission_builder import validate_submission


def verify_run(run_dir: Path, data_dir: Path) -> dict[str, object]:
    run_manifest_path = run_dir / "run_manifest.json"
    hash_manifest_path = run_dir / "sha256_manifest.json"
    if not run_manifest_path.is_file() or not hash_manifest_path.is_file():
        raise FileNotFoundError("Run or SHA-256 manifest is missing.")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    hash_manifest = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "PASS":
        raise ValueError(f"Pipeline status is not PASS: {run_manifest.get('status')}")
    expected = hash_manifest["files"]
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    if actual_paths != set(expected):
        raise ValueError("Run directory file inventory differs from its manifest.")
    for relative, record in expected.items():
        path = run_dir / Path(relative)
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Run artifact hash mismatch: {relative}")
    report = json.loads(
        (run_dir / "reports" / "implementation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if report["reproduction_validation"]["status"] != "PASS":
        raise ValueError("Frozen-reference reproduction gate did not pass.")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    candidate_path = run_dir / "submissions" / "submission_phase2_complete.csv"
    candidate = pd.read_csv(candidate_path)
    submission_checks = validate_submission(candidate, test, sample)
    for name in ("lightgbm", "catboost", "phase2_blend"):
        for kind, expected_rows in (("oof", len(train)), ("test", len(test))):
            prediction_path = run_dir / "predictions" / f"{kind}_{name}.npy"
            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Required {kind} prediction artifact is missing: {prediction_path}"
                )
            predictions = np.load(prediction_path, allow_pickle=False)
            if (
                predictions.shape != (expected_rows,)
                or not np.isfinite(predictions).all()
                or np.any((predictions < 0.0) | (predictions > 1.0))
            ):
                raise ValueError(f"Invalid {kind} predictions: {prediction_path.name}")
    promotion_path = report["submission"]["promoted_path"]
    if promotion_path:
        promoted = Path(promotion_path)
        if not promoted.is_file() or sha256_file(promoted) != sha256_file(candidate_path):
            raise ValueError("Promoted submission is missing or differs from verified run output.")
    return {
        "status": "PASS",
        "run_id": run_manifest["run_id"],
        "manifest_files": len(expected),
        "submission_sha256": sha256_file(candidate_path),
        "submission_checks": submission_checks,
        "promoted_path": promotion_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a completed pipeline run and submission.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    args = parser.parse_args()
    result = verify_run(args.run_dir.resolve(), args.data_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
