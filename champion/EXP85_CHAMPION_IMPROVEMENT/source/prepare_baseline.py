from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
GIT_ROOT = WORKSPACE.parents[1]
GEMINI_ROOT = Path(r"D:\gci\gemini")
REFERENCE_PATH = Path(
    r"D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"
)
RAW_DIR = Path(r"D:\gci\inpuy")
NOTEBOOK_PATH = GEMINI_ROOT / "GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb"
SCORE_REPORT_PATH = GEMINI_ROOT / "EXP85_DEPENDENCY_AUDIT.md"

DIRECT_INPUTS = {
    "anchor74": "v_20261027_precision_transductive_apex.csv",
    "exp79_comparison": "v_20261027_grand_master_macro_stacker_apex.csv",
    "v17": "cache/v_17_full_suite_bundle.npz",
    "v55": "cache/v55_members.npz",
    "xgb10_oof": "cache/oof_10f_xgb.npy",
    "xgb10_test": "cache/test_10f_xgb.npy",
    "labels": "cache/y_train.npy",
}

SOURCES = {
    "exp85_final_synthesis": "build_exp85_optimal_macro_apex.py",
    "anchor74_builder": "build_precision_transductive_apex_champion.py",
    "optimal_transductive_builder": "build_optimal_transductive_apex_champion.py",
    "asymmetric_apex_builder": "build_asymmetric_convex_gated_champion.py",
    "level2_builder": "build_level2_regime_meta_stacker.py",
    "v080200_builder": "build_v80200_apex.py",
    "v080150_builder": "solo_grand_master_apex_080.py",
    "feature_builder": "build_hasan_600_features.py",
    "feature_selector": "screen_and_select_features.py",
    "v17_trainer": "train_full_v16_suite.py",
    "v55_trainer": "train_clean_honors_champion.py",
    "zoo_trainer_a": "train_hasan_master_zoo.py",
    "zoo_trainer_b": "train_catboost_and_deep_lgb.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_immutable(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(f"Required baseline evidence is missing: {source}")
    expected_hash = sha256_file(source)
    if destination.exists():
        actual_hash = sha256_file(destination)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Refusing to overwrite changed baseline snapshot: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    record = file_record(destination)
    if record["sha256"] != expected_hash:
        raise RuntimeError(f"Snapshot verification failed: {destination}")
    record["original_path"] = str(source)
    return record


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(GIT_ROOT), *args],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def main() -> None:
    baseline = WORKSPACE / "baseline"
    artifact_dir = baseline / "artifacts"
    reference_dir = baseline / "reference"
    source_dir = baseline / "source"

    inputs: dict[str, dict[str, object]] = {}
    for key, relative in DIRECT_INPUTS.items():
        inputs[key] = copy_immutable(
            GEMINI_ROOT / relative, artifact_dir / Path(relative).name
        )
    reference = copy_immutable(
        REFERENCE_PATH, reference_dir / "EXP85_HISTORICAL_CHAMPION.csv"
    )

    sources: dict[str, dict[str, object]] = {}
    for key, relative in SOURCES.items():
        source = GEMINI_ROOT / relative
        destination = source_dir / source.name
        if source.is_file():
            sources[key] = copy_immutable(source, destination)
        else:
            sources[key] = {
                "original_path": str(source),
                "status": "NOT_FOUND",
            }

    raw_hashes = {}
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        path = RAW_DIR / name
        raw_hashes[name] = file_record(path) if path.is_file() else {
            "path": str(path),
            "status": "NOT_FOUND",
        }

    source_git_root = GEMINI_ROOT / ".git"
    manifest = {
        "baseline_id": "EXP85_HISTORICAL_0.801340",
        "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "HISTORICAL_CACHE_ASSISTED_BASELINE",
        "user_target_public_auc": 0.80314,
        "reported_historical_public_auc": 0.80134,
        "score_evidence_note": (
            "0.801340 is reported by the historical EXP85 source and audit; "
            "an official upload receipt bound to this CSV hash was not found."
        ),
        "workspace_git_root": str(GIT_ROOT),
        "workspace_git_commit_at_snapshot": git_value("rev-parse", "HEAD"),
        "historical_source_git_metadata": (
            "AVAILABLE" if source_git_root.exists() else "UNAVAILABLE"
        ),
        "historical_source_commit": None,
        "historical_source_commit_note": (
            "D:\\gci\\gemini is not a Git worktree; source SHA-256 values are "
            "the available identity evidence."
        ),
        "historical_champion_reference": reference,
        "historical_notebook_reference": (
            file_record(NOTEBOOK_PATH)
            if NOTEBOOK_PATH.is_file()
            else {"path": str(NOTEBOOK_PATH), "status": "NOT_FOUND"}
        ),
        "historical_score_report_reference": (
            file_record(SCORE_REPORT_PATH)
            if SCORE_REPORT_PATH.is_file()
            else {"path": str(SCORE_REPORT_PATH), "status": "NOT_FOUND"}
        ),
        "direct_synthesis_inputs": inputs,
        "relevant_upstream_sources": sources,
        "raw_inputs_hashed_not_copied": raw_hashes,
        "execution_scope": (
            "Final 85:15 synthesis only. Upstream models and Anchor74 remain "
            "historical cached predictions; this is not a raw rebuild."
        ),
        "v20_v080041_scope": (
            "Not direct inputs to the final EXP85 synthesis script; their "
            "lineage is upstream of the historical Anchor74 cache."
        ),
    }
    manifest_path = baseline / "BASELINE_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["snapshot_created_utc"] = existing["snapshot_created_utc"]
        manifest["workspace_git_commit_at_snapshot"] = existing[
            "workspace_git_commit_at_snapshot"
        ]
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(
            f"Baseline manifest already exists and differs: {manifest_path}"
        )
    manifest_path.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "baseline_manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "inputs_snapshotted": len(inputs),
        "sources_found": sum(v.get("status") != "NOT_FOUND" for v in sources.values()),
        "historical_champion_sha256": reference["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
