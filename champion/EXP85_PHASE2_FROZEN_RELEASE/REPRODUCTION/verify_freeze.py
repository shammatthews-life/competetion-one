from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "FREEZE_MANIFEST.json"
MANIFEST_GENERATOR = ROOT / "REPRODUCTION" / "update_freeze_manifest.py"
MANIFEST_MODULE_NAME = "freeze_manifest_generator"
EXPECTED_CANDIDATE_SHA256 = (
    "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_candidate_order(
    candidate_path: Path,
    test_path: Path,
    sample_path: Path,
    expected_rows: int = 61500,
) -> dict[str, Any]:
    issues: list[str] = []
    result: dict[str, Any] = {
        "candidate_path": str(candidate_path),
        "test_path": str(test_path),
        "sample_submission_path": str(sample_path),
        "expected_rows": expected_rows,
        "row_count": None,
        "schema": None,
        "unique_ids": False,
        "ids_match_test_order": False,
        "ids_match_sample_order": False,
        "missing_from_candidate_vs_test": None,
        "extra_in_candidate_vs_test": None,
        "prediction_min": None,
        "prediction_max": None,
        "finite_predictions": False,
        "issues": issues,
    }

    try:
        candidate_columns, candidate_rows = read_csv(candidate_path)
        test_columns, test_rows = read_csv(test_path)
        sample_columns, sample_rows = read_csv(sample_path)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        issues.append(f"Could not read required CSV: {error}")
        return result

    result["row_count"] = len(candidate_rows)
    result["schema"] = candidate_columns
    if candidate_columns != ["SK_ID_CURR", "TARGET"]:
        issues.append(
            "Candidate schema must be exactly ['SK_ID_CURR', 'TARGET']; "
            f"found {candidate_columns!r}"
        )
    if len(candidate_rows) != expected_rows:
        issues.append(
            f"Candidate row count {len(candidate_rows)} != expected {expected_rows}"
        )
    for label, rows, columns in (
        ("test", test_rows, test_columns),
        ("sample submission", sample_rows, sample_columns),
    ):
        if "SK_ID_CURR" not in columns:
            issues.append(f"{label} is missing SK_ID_CURR")
        if len(rows) != expected_rows:
            issues.append(
                f"{label} row count {len(rows)} != expected {expected_rows}"
            )

    candidate_ids = [row.get("SK_ID_CURR", "") for row in candidate_rows]
    test_ids = [row.get("SK_ID_CURR", "") for row in test_rows]
    sample_ids = [row.get("SK_ID_CURR", "") for row in sample_rows]
    result["unique_ids"] = (
        len(candidate_ids) == len(set(candidate_ids))
        and all(candidate_ids)
    )
    if not result["unique_ids"]:
        issues.append("Candidate IDs are empty or duplicated")
    if len(test_ids) == len(set(test_ids)) and candidate_ids:
        missing = sorted(set(test_ids) - set(candidate_ids))
        extra = sorted(set(candidate_ids) - set(test_ids))
        result["missing_from_candidate_vs_test"] = len(missing)
        result["extra_in_candidate_vs_test"] = len(extra)
        if missing:
            issues.append(f"Candidate is missing {len(missing)} test IDs")
        if extra:
            issues.append(f"Candidate has {len(extra)} IDs not in test")
    elif len(test_ids) != len(set(test_ids)):
        issues.append("Raw test IDs are duplicated")
    result["ids_match_test_order"] = candidate_ids == test_ids
    result["ids_match_sample_order"] = candidate_ids == sample_ids
    if not result["ids_match_test_order"]:
        issues.append("Candidate ID order does not exactly match test.csv")
    if not result["ids_match_sample_order"]:
        issues.append("Candidate ID order does not exactly match sample_submission.csv")

    values: list[float] = []
    for row_number, row in enumerate(candidate_rows, start=2):
        raw_value = row.get("TARGET")
        try:
            value = float(raw_value) if raw_value is not None else math.nan
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            issues.append(f"TARGET is nonnumeric or nonfinite at CSV line {row_number}")
            break
        if not 0.0 <= value <= 1.0:
            issues.append(f"TARGET is outside [0, 1] at CSV line {row_number}")
            break
        values.append(value)
    result["finite_predictions"] = len(values) == len(candidate_rows)
    if values:
        result["prediction_min"] = min(values)
        result["prediction_max"] = max(values)
    if len(values) == len(candidate_rows) and values:
        result["probability_range_valid"] = True
    else:
        result["probability_range_valid"] = False
    return result


def verify_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("content_files", {})
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path != root / "FREEZE_MANIFEST.json"
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    issues: list[str] = []
    missing = sorted(set(expected) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected))
    for relative in missing:
        issues.append(f"Manifest file missing: {relative}")
    for relative in extra:
        issues.append(f"Unmanifested release file: {relative}")
    hash_failures = []
    for relative in sorted(set(expected) & set(actual_paths)):
        path = actual_paths[relative]
        record = expected[relative]
        actual_hash = sha256_file(path)
        if path.stat().st_size != record.get("size_bytes"):
            hash_failures.append(relative)
            issues.append(f"Manifest size mismatch: {relative}")
        if actual_hash != record.get("sha256"):
            hash_failures.append(relative)
            issues.append(f"Manifest SHA-256 mismatch: {relative}")

    for relative in manifest.get("required_files", []):
        if not (root / Path(relative)).is_file():
            issues.append(f"Required freeze evidence missing: {relative}")
    return {
        "expected_file_count": len(expected),
        "actual_file_count": len(actual_paths),
        "missing_files": missing,
        "unmanifested_files": extra,
        "hash_failures": sorted(set(hash_failures)),
        "issues": issues,
    }


def run_verification(
    train_csv: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    source_candidate: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "release_label": "FROZEN_DEVELOPMENT_CANDIDATE",
        "status": "FAIL",
        "candidate_sha256": None,
        "source_candidate_sha256": None,
        "input_sha256": {},
        "candidate_checks": {},
        "manifest_checks": {},
        "errors": errors,
    }
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Cannot read freeze manifest {MANIFEST_PATH}: {error}")
        return result

    if manifest.get("release_label") != "FROZEN_DEVELOPMENT_CANDIDATE":
        errors.append("Freeze manifest has an unexpected release label")
    manifest_result = verify_manifest(ROOT, manifest)
    result["manifest_checks"] = manifest_result
    errors.extend(manifest_result["issues"])

    candidate_record = manifest.get("source_candidate", {})
    expected_sha = candidate_record.get("sha256")
    if expected_sha != EXPECTED_CANDIDATE_SHA256:
        errors.append("Freeze manifest candidate hash differs from the known hash")
    release_candidate = ROOT / candidate_record.get(
        "release_relative_path", "CANDIDATE/submission_key_target_encoding.csv"
    )
    try:
        result["candidate_sha256"] = sha256_file(release_candidate)
        if result["candidate_sha256"] != expected_sha:
            errors.append("Archived candidate SHA-256 does not match freeze manifest")
    except OSError as error:
        errors.append(f"Archived candidate is unavailable: {error}")

    try:
        result["source_candidate_sha256"] = sha256_file(source_candidate)
        if result["source_candidate_sha256"] != expected_sha:
            errors.append(
                "Original source candidate changed or differs from frozen SHA-256"
            )
        if result["candidate_sha256"] != result["source_candidate_sha256"]:
            errors.append("Archived and original candidate are not byte-identical")
    except OSError as error:
        errors.append(f"Original source candidate unavailable: {source_candidate} ({error})")

    raw_inputs = manifest.get("raw_inputs", {})
    for name, path in (
        ("train.csv", train_csv),
        ("test.csv", test_csv),
        ("sample_submission.csv", sample_submission_csv),
    ):
        expected = raw_inputs.get(name, {}).get("sha256")
        try:
            actual = sha256_file(path)
            result["input_sha256"][name] = actual
            if actual != expected:
                errors.append(f"{name} SHA-256 differs from frozen input evidence")
        except OSError as error:
            result["input_sha256"][name] = None
            errors.append(f"Required input unavailable: {path} ({error})")

    result["candidate_checks"] = validate_candidate_order(
        release_candidate,
        test_csv,
        sample_submission_csv,
        int(candidate_record.get("expected_rows", 61500)),
    )
    errors.extend(result["candidate_checks"]["issues"])
    result["errors"] = errors
    result["status"] = "PASS" if not errors else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Phase 2 candidate, inputs, and release manifest."
    )
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--sample-submission-csv", type=Path)
    parser.add_argument("--source-candidate", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inputs = manifest["raw_inputs"]
    source_candidate = manifest["source_candidate"]["path"]
    result = run_verification(
        args.train_csv or Path(inputs["train.csv"]["path"]),
        args.test_csv or Path(inputs["test.csv"]["path"]),
        args.sample_submission_csv or Path(inputs["sample_submission.csv"]["path"]),
        args.source_candidate or Path(source_candidate),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
