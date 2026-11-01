from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR / "source"))

from data_validation import sha256_file


MANIFEST_NAME = "PACKAGE_SHA256_MANIFEST.json"


def inventory(package_dir: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(package_dir.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files[path.relative_to(package_dir).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return files


def create_manifest(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / MANIFEST_NAME
    result = {
        "algorithm": "SHA-256",
        "manifest_excludes_itself": True,
        "files": inventory(package_dir, manifest_path),
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"status": "PASS", "manifest": str(manifest_path), "file_count": len(result["files"])}


def verify_manifest(package_dir: Path) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Package SHA-256 manifest is missing: {manifest_path}")
    expected_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError("Package SHA-256 manifest has no valid file inventory.")
    actual = inventory(package_dir, manifest_path)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = [
        path
        for path in sorted(set(expected) & set(actual))
        if expected[path] != actual[path]
    ]
    if missing or extra or mismatched:
        raise ValueError(
            json.dumps(
                {"missing": missing, "extra": extra, "mismatched": mismatched},
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "manifest": str(manifest_path),
        "file_count": len(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify the package SHA-256 inventory.")
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    args = parser.parse_args()
    result = (
        create_manifest(args.package_dir)
        if args.action == "create"
        else verify_manifest(args.package_dir)
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
