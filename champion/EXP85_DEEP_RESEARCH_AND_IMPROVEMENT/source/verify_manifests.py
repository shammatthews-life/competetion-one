from __future__ import annotations

import json
from pathlib import Path

from build_manifests import MANIFEST_PATH, ROOT, package_files, sha256_file


def verify() -> tuple[list[str], int]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("format") != "EXP85_DEEP_RESEARCH_SHA256_V1":
        raise ValueError("Unsupported manifest format")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    current_files = {
        path.relative_to(ROOT).as_posix(): path for path in package_files()
    }
    errors = []
    for relative in sorted(expected.keys() - current_files.keys()):
        errors.append(f"Missing file: {relative}")
    for relative in sorted(current_files.keys() - expected.keys()):
        errors.append(f"Unexpected file: {relative}")
    verified = 0
    for relative in sorted(expected.keys() & current_files.keys()):
        path = current_files[relative]
        entry = expected[relative]
        if path.stat().st_size != entry["size_bytes"]:
            errors.append(f"Size mismatch: {relative}")
            continue
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {relative}")
            continue
        verified += 1
    return errors, verified


if __name__ == "__main__":
    problems, count = verify()
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print(f"Verified {count} files against {MANIFEST_PATH}")
