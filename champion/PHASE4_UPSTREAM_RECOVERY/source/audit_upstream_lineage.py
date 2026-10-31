from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


V20_NAME = "v_20_apex_grand_champion_submission.csv"
V080041_NAME = "v_solo_080041_pure_champion.csv"
SUBMISSION_FINAL_NAME = "submission_final.csv"
FINAL_SUBMISSION_SOURCE = Path("final_submission") / "reproduce_final_submission.py"
V080041_SOURCE = "solo_grand_master_apex_080.py"


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


def resolve_path_references(source_path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    references: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_path"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            references.add((node.args[0].value, node.lineno))
    return sorted(references)


def source_mentions(root: Path, needle: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for pattern in ("*.py", "*.ipynb"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    matches.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
    return matches


def find_named_files(name: str, roots: list[Path]) -> list[Path]:
    results: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / name
        if direct.is_file():
            results.add(direct.resolve())
        for match in root.rglob(name):
            if match.is_file():
                results.add(match.resolve())
    return sorted(results)


def assert_submission_alignment(path: Path, raw_test_ids: np.ndarray) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required_columns = ["SK_ID_CURR", "TARGET"]
    if list(frame.columns) != required_columns:
        raise ValueError(f"{path} columns are {list(frame.columns)}, expected {required_columns}.")
    ids = frame["SK_ID_CURR"].to_numpy()
    if len(ids) != len(raw_test_ids):
        raise ValueError(f"{path} has {len(ids)} rows, expected {len(raw_test_ids)}.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate IDs.")
    if not np.array_equal(ids, raw_test_ids):
        raise ValueError(f"{path} IDs do not exactly match raw test order.")
    targets = frame["TARGET"].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError(f"{path} has non-finite TARGET values.")
    return {
        "rows": len(frame),
        "unique_ids": True,
        "id_order_matches_raw_test": True,
        "finite_targets": True,
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit V20/V080041 lineage and legacy submission dependencies."
    )
    parser.add_argument(
        "--gemini-root", type=Path, default=Path(r"D:\gci\gemini")
    )
    parser.add_argument(
        "--repository-root", type=Path, default=Path(r"D:\gci\_gci_push_tmp")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\gci\_gci_push_tmp\champion\PHASE4_UPSTREAM_RECOVERY\reports"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_test_path = args.raw_dir / "test.csv"
    raw_test_ids = pd.read_csv(raw_test_path, usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    v20_path = args.gemini_root / V20_NAME
    v080041_path = args.gemini_root / V080041_NAME
    submission_final_path = (
        args.gemini_root / "scratch" / "final_sub" / "final_submission" / SUBMISSION_FINAL_NAME
    )
    final_source_path = args.gemini_root / FINAL_SUBMISSION_SOURCE
    v080041_source_path = args.gemini_root / V080041_SOURCE

    search_roots = [
        args.gemini_root,
        args.repository_root,
        Path.cwd(),
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path(r"C:\Users\User\Documents\Personal Stuffs\GCI World\output"),
    ]
    if raw_test_path.is_file() is False:
        raise FileNotFoundError(raw_test_path)

    grouped_references: dict[str, list[int]] = {}
    for filename, line_number in resolve_path_references(final_source_path):
        grouped_references.setdefault(filename, []).append(line_number)

    dependencies: dict[str, Any] = {}
    missing_rows: list[dict[str, Any]] = []
    for filename, line_numbers in sorted(grouped_references.items()):
        found = find_named_files(filename, search_roots)
        dependencies[filename] = {
            "consumer": str(final_source_path),
            "consumer_lines": line_numbers,
            "status": "AVAILABLE" if found else "MISSING",
            "found_paths": [str(path) for path in found],
        }
        missing_rows.append(
            {
                "artifact": filename,
                "consumer": str(final_source_path),
                "consumer_lines": ";".join(str(line) for line in line_numbers),
                "status": "AVAILABLE" if found else "MISSING",
                "found_paths": ";".join(str(path) for path in found),
                "search_roots": ";".join(str(root) for root in search_roots),
            }
        )

    v20_record: dict[str, Any] = {
        "artifact": str(v20_path),
        "available": v20_path.is_file(),
        "source_mentions": source_mentions(args.gemini_root, V20_NAME),
    }
    if v20_path.is_file():
        v20_record.update(file_record(v20_path))
        v20_record["submission_validation"] = assert_submission_alignment(
            v20_path, raw_test_ids
        )

    v080041_record: dict[str, Any] = {
        "artifact": str(v080041_path),
        "available": v080041_path.is_file(),
        "source": str(v080041_source_path),
        "source_available": v080041_source_path.is_file(),
        "submission_final": str(submission_final_path),
        "submission_final_source": str(final_source_path),
        "submission_final_source_available": final_source_path.is_file(),
    }
    if v080041_path.is_file():
        v080041_record.update(file_record(v080041_path))
        v080041_record["submission_validation"] = assert_submission_alignment(
            v080041_path, raw_test_ids
        )
    if submission_final_path.is_file():
        v080041_record["submission_final_record"] = file_record(
            submission_final_path
        )
        v080041_record["submission_final_validation"] = assert_submission_alignment(
            submission_final_path, raw_test_ids
        )
    v080041_record["byte_identical_to_submission_final"] = (
        v080041_path.is_file()
        and submission_final_path.is_file()
        and sha256_file(v080041_path) == sha256_file(submission_final_path)
    )

    evidence = {
        "scope": "Static source and local artifact audit; no model training executed.",
        "raw_test": file_record(raw_test_path),
        "v20": v20_record,
        "v080041": v080041_record,
        "legacy_submission_inputs": dependencies,
        "search_roots": [str(root) for root in search_roots],
    }
    evidence_path = args.output_dir / "upstream_lineage_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    missing_path = args.output_dir / "MISSING_UPSTREAM_DEPENDENCIES.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "artifact",
                "consumer",
                "consumer_lines",
                "status",
                "found_paths",
                "search_roots",
            ),
        )
        writer.writeheader()
        writer.writerows(missing_rows)
    print(f"Evidence: {evidence_path}")
    print(f"Dependency inventory: {missing_path}")


if __name__ == "__main__":
    main()
