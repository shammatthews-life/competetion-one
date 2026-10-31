from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PHASE_ROOT / "manifests" / "PHASE4_SHA256_MANIFEST.json"
FIXED_FILES = (
    "champion/02_EXP85_REBUILT/README.md",
    "champion/02_EXP85_REBUILT/source/provenance_contract.py",
    "champion/02_EXP85_REBUILT/source/run_all_recreations.py",
    "champion/02_EXP85_REBUILT/rebuilt_artifacts/RECONSTRUCTION_PROVENANCE_MANIFEST.json",
    "champion/CLEAN_ROOM_VERIFICATION_REPORT.md",
    "champion/REPRODUCIBILITY_STATUS.md",
    "champion/create_exp85_submissions.py",
    "champion/create_submission_recovery_v5_rebuilt.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files() -> list[Path]:
    candidates: set[Path] = set()
    for dirname in ("reports", "source", "tests"):
        candidates.update((PHASE_ROOT / dirname).rglob("*"))
    candidates.add(PHASE_ROOT / ".gitignore")
    candidates.update((PHASE_ROOT / "runs").rglob("execution.json"))
    candidates.update(REPOSITORY_ROOT / name for name in FIXED_FILES)

    files = []
    for path in candidates:
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path == MANIFEST_PATH:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def main() -> None:
    files = collect_files()
    expected = [
        REPOSITORY_ROOT / name
        for name in FIXED_FILES
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required manifest inputs are missing: {missing}")

    records = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": (
                    "Phase 4 source, tests, reports, run execution records, "
                    "and directly modified provenance-gating files. The manifest "
                    "does not hash itself or bulky local training/staging outputs."
                ),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} SHA-256 records to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
