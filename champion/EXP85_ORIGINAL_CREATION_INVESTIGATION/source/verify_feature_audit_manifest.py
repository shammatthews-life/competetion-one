#!/usr/bin/env python3
"""Verify files recorded by the historical-lineage SHA-256 CSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest: Path) -> dict:
    started = time.monotonic()
    root = root.resolve()
    manifest = manifest.resolve()
    rows = []
    seen_paths = set()
    issues = []

    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "size_bytes", "sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest must contain columns: {', '.join(sorted(required))}"
            )
        for line_number, row in enumerate(reader, 2):
            relative_path = row["relative_path"]
            if not relative_path or relative_path in seen_paths:
                issues.append(
                    {
                        "line": line_number,
                        "path": relative_path,
                        "issue": "empty or duplicate manifest path",
                    }
                )
                continue
            seen_paths.add(relative_path)
            path = (root / Path(relative_path)).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                issues.append(
                    {
                        "line": line_number,
                        "path": relative_path,
                        "issue": "path escapes manifest root",
                    }
                )
                continue

            if not path.is_file():
                rows.append(
                    {
                        "path": relative_path,
                        "expected_size_bytes": row["size_bytes"],
                        "expected_sha256": row["sha256"],
                        "exists": False,
                        "size_matches": False,
                        "actual_size_bytes": None,
                        "actual_sha256": None,
                        "sha256_matches": False,
                    }
                )
                continue

            expected_size = int(row["size_bytes"])
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            rows.append(
                {
                    "path": relative_path,
                    "expected_size_bytes": expected_size,
                    "expected_sha256": row["sha256"].lower(),
                    "exists": True,
                    "size_matches": actual_size == expected_size,
                    "actual_size_bytes": actual_size,
                    "actual_sha256": actual_sha,
                    "sha256_matches": actual_sha == row["sha256"].lower(),
                }
            )

    passed = bool(rows) and not issues and all(
        row["exists"] and row["size_matches"] and row["sha256_matches"]
        for row in rows
    )
    return {
        "audit": "historical-lineage feature-audit manifest verification",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_entries": len(rows) + len(issues),
        "checked_files": len(rows),
        "sha256_matches": sum(row["sha256_matches"] for row in rows),
        "size_matches": sum(row["size_matches"] for row in rows),
        "issues": issues,
        "passed": passed,
        "elapsed_seconds": time.monotonic() - started,
        "files": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "FEATURE_AUDIT_SHA256_VERIFICATION.json",
    )
    args = parser.parse_args()

    result = verify(args.root, args.manifest)
    result["command"] = sys.argv
    result["exit_code"] = 0 if result["passed"] else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Evidence SHA-256: {sha256_file(args.output)}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
