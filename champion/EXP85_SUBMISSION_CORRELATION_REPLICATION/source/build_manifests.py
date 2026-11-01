from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPLICATION_MANIFEST = ROOT / "REPLICATION_MANIFEST.json"
SHA256_MANIFEST = ROOT / "hashes" / "SHA256_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(excluded: set[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def valid_run_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "VALID" or report.get("exit_code") != 0:
        raise ValueError(f"Non-valid correlation run cannot be manifested: {path}")
    return {
        "run_id": report["run_id"],
        "method": report["configuration"]["method"],
        "reference_sha256": report["reference"]["sha256"],
        "candidate_sha256": report["candidate"]["sha256"],
        "script_sha256": report["script_sha256"],
        "configuration_sha256": report["configuration_sha256"],
        "pearson": report["metrics"]["pearson"],
        "spearman": report["metrics"]["spearman"],
        "official_competition_score": report["official_competition_score"],
    }


def main() -> None:
    SHA256_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    excluded = {REPLICATION_MANIFEST, SHA256_MANIFEST}
    files = list_files(excluded)
    reports = sorted((ROOT / "runs").glob("*/correlation_report.json"))
    run_summaries = [valid_run_summary(path) for path in reports]
    replication_manifest = {
        "manifest_version": 1,
        "project": "EXP85_SUBMISSION_CORRELATION_REPLICATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "NUMERIC RESULT REPRODUCED; high-correlation candidate is a "
            "HYBRID HISTORICAL-CACHE REPLAY, not a full raw-only rebuild."
        ),
        "primary_reference_sha256": (
            "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
        ),
        "primary_candidate_sha256": (
            "91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482"
        ),
        "valid_correlation_runs": run_summaries,
        "files": files,
        "manifest_excludes_itself": True,
    }
    REPLICATION_MANIFEST.write_text(
        json.dumps(replication_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    inventory = list_files({SHA256_MANIFEST})
    SHA256_MANIFEST.write_text(
        json.dumps(
            {
                "algorithm": "SHA-256",
                "files": inventory,
                "manifest_excludes_itself": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {REPLICATION_MANIFEST.relative_to(ROOT)} "
        f"and {SHA256_MANIFEST.relative_to(ROOT)}; "
        f"{len(run_summaries)} valid runs, {len(inventory)} hashed files."
    )


if __name__ == "__main__":
    main()
