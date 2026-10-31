from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(
    entries: dict[str, dict[str, Any]],
    path: Path,
    role: str,
    expected_hash: str | None = None,
) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
        )
    key = str(path)
    existing = entries.get(key)
    record = {
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
        "roles": [role],
    }
    if existing is not None and existing != record:
        if (
            existing["sha256"] != actual_hash
            or existing["size_bytes"] != path.stat().st_size
        ):
            raise ValueError(f"Conflicting manifest hashes for {path}")
        if role not in existing["roles"]:
            existing["roles"].append(role)
    else:
        entries[key] = record


def build_manifest(audit_root: Path, run_dir: Path, output_path: Path) -> dict[str, Any]:
    audit_root = audit_root.resolve()
    run_dir = run_dir.resolve()
    execution_path = run_dir / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != (
        "FULL_RAW_CANDIDATE_COMPLETE; HISTORICAL_PARITY_NOT_ESTABLISHED"
    ):
        raise ValueError(
            "Cannot create final manifest: full raw candidate execution is incomplete."
        )
    validation_path = run_dir / "reports" / "final_validation_strict.json"
    if not validation_path.is_file():
        validation_path = Path(execution["validation_report"]["path"])
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation["candidate"]["sha256"] != execution["candidate"]["sha256"]:
        raise ValueError("Strict validation candidate hash differs from execution record.")
    parity_path = run_dir / "reports" / "historical_output_comparison.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("exit_code") != 0:
        raise ValueError("Historical output comparison did not complete successfully.")

    entries: dict[str, dict[str, Any]] = {}
    for path in audit_root.rglob("*"):
        if not path.is_file() or path.resolve() == output_path.resolve():
            continue
        relative = path.relative_to(audit_root)
        parts = set(relative.parts)
        if "__pycache__" in parts:
            continue
        if "components" in parts or "catboost_info" in parts:
            continue
        if path.name == "candidate_submission.csv":
            continue
        add_file(entries, path, "audit source, report, test, execution record, or log")

    for filename, record in execution["raw_inputs"].items():
        add_file(
            entries,
            Path(record["path"]),
            f"raw input: {filename}",
            record["sha256"],
        )
    for entry in execution["generator_source_audit"]:
        add_file(
            entries,
            Path(entry["path"]),
            "executed raw generator source",
            entry["sha256"],
        )
    for record in execution["generator_executions"]:
        add_file(
            entries,
            Path(record["log_path"]),
            f"generator log: {record['script']}",
            record["log_sha256"],
        )
    add_file(
        entries,
        Path(execution["candidate"]["path"]),
        "candidate CSV",
        execution["candidate"]["sha256"],
    )
    add_file(
        entries,
        Path(execution["validation_report"]["path"]),
        "initial wrapper validation report",
        execution["validation_report"]["sha256"],
    )
    add_file(
        entries,
        validation_path,
        "strict post-run candidate validation report",
        sha256_file(validation_path),
    )
    add_file(
        entries,
        parity_path,
        "read-only historical prediction comparison report",
        sha256_file(parity_path),
    )
    for filename, record in execution["component_outputs"].items():
        add_file(
            entries,
            Path(record["path"]),
            f"fresh generated component (not committed)",
            record["sha256"],
        )

    add_file(
        entries,
        Path(validation["frozen_champion"]["path"]),
        "read-only frozen champion reference",
        validation["frozen_champion"]["sha256"],
    )
    previous_candidate = Path(
        r"D:\gci\_gci_push_tmp\champion\08_SUBMISSIONS\submission_exp85_full_rebuilt.csv"
    )
    add_file(
        entries,
        previous_candidate,
        "read-only prior raw-candidate comparison reference",
    )
    for name, digest in parity["historical_artifact_sha256"].items():
        add_file(
            entries,
            Path(parity["historical_directory"]) / name,
            "read-only historical prediction cache used for comparison only",
            digest,
        )
    if parity.get("anchor74_vs_historical"):
        anchor_record = parity["anchor74_vs_historical"]
        add_file(
            entries,
            Path(anchor_record["path"]),
            "read-only historical Anchor74 comparison reference",
            anchor_record["sha256"],
        )

    references = (
        (
            Path(r"D:\gci\gemini\build_exp85_optimal_macro_apex.py"),
            "historical EXP85 formula source",
        ),
        (
            Path(r"D:\gci\gemini\final_submission\reproduce_final_submission.py"),
            "V080041 stacker consumer source",
        ),
        (
            Path(r"D:\gci\gemini\v_20_apex_grand_champion_submission.csv"),
            "V20 lineage-unknown historical prediction artifact",
        ),
        (
            Path(r"D:\gci\gemini\v_solo_080041_pure_champion.csv"),
            "V080041 historical prediction artifact",
        ),
        (
            Path(r"D:\gci\gemini\v_20261027_optimal_macro_apex_v85.csv"),
            "read-only historical v85 formula output",
        ),
        (
            Path(r"D:\gci\pre info\oct22\meta.json"),
            "recovered V080041 metadata",
        ),
        (
            Path(r"D:\gci\_gci_push_tmp\champion\PHASE4_UPSTREAM_RECOVERY\reports\PHASE4_MASTER_REPORT.md"),
            "prior phase evidence",
        ),
        (
            Path(r"D:\gci\_gci_push_tmp\champion\PHASE4_UPSTREAM_RECOVERY\reports\V20_V080041_LINEAGE_REPORT.md"),
            "prior V20/V080041 lineage evidence",
        ),
        (
            Path(r"D:\gci\_gci_push_tmp\champion\PHASE4_UPSTREAM_RECOVERY\reports\CATBOOST_REPRODUCIBILITY_REPORT.md"),
            "prior CatBoost reproducibility evidence",
        ),
        (
            Path(r"D:\gci\_gci_push_tmp\champion\PHASE4_UPSTREAM_RECOVERY\reports\MISSING_UPSTREAM_DEPENDENCIES.csv"),
            "previous V080041 missing-dependency inventory",
        ),
        (
            Path(r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\reports\FIRST_DIVERGENCE_REPORT.md"),
            "separate 695-feature production lineage evidence",
        ),
        (
            Path(r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\reports\PRODUCTION_FEATURE_MATRIX_CAPTURE_REPORT.md"),
            "separate 695-feature production matrix capture evidence",
        ),
        (
            Path(r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\historical_lineage_bridge\reports\FIRST_TRANSFORMATION_DIVERGENCE.md"),
            "separate current production transformation divergence evidence",
        ),
    )
    for path, role in references:
        add_file(entries, path, role)

    manifest = {
        "schema_version": 1,
        "classification": execution["classification"],
        "raw_rebuild_certified": False,
        "official_leaderboard_score": None,
        "candidate_sha256": execution["candidate"]["sha256"],
        "run_execution_sha256": sha256_file(execution_path),
        "manifest_sha256_self_excluded": True,
        "files": [
            {"path": path, **record}
            for path, record in sorted(entries.items(), key=lambda item: item[0].lower())
        ],
        "file_count": len(entries),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SHA-256 manifest for audit evidence and generated run artifacts."
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.audit_root, args.run_dir, args.output)
    print(f"Manifest records: {manifest['file_count']}")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
