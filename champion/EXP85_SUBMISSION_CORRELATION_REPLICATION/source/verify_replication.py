from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATION_MANIFEST = ROOT / "REPLICATION_MANIFEST.json"
SHA256_MANIFEST = ROOT / "hashes" / "SHA256_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not REPLICATION_MANIFEST.is_file() or not SHA256_MANIFEST.is_file():
        raise FileNotFoundError("Replication manifest or SHA-256 manifest is missing.")
    manifest = json.loads(REPLICATION_MANIFEST.read_text(encoding="utf-8"))
    hash_manifest = json.loads(SHA256_MANIFEST.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in hash_manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != SHA256_MANIFEST
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        unexpected = sorted(actual_paths - set(expected))
        raise ValueError(
            f"Manifest inventory differs: missing={missing}, unexpected={unexpected}"
        )
    for relative_path, entry in expected.items():
        path = ROOT / Path(relative_path)
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Size mismatch: {relative_path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative_path}")

    covered = {entry["path"]: entry for entry in manifest["files"]}
    for relative_path, entry in covered.items():
        if expected.get(relative_path) != entry:
            raise ValueError(f"Replication manifest mismatch: {relative_path}")
    if manifest["primary_reference_sha256"] != (
        "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
    ):
        raise ValueError("Unexpected primary reference identity.")
    if manifest["primary_candidate_sha256"] != (
        "91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482"
    ):
        raise ValueError("Unexpected primary candidate identity.")
    print(
        f"Replication integrity verified: {len(expected)} files; "
        f"{len(manifest['valid_correlation_runs'])} valid runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
