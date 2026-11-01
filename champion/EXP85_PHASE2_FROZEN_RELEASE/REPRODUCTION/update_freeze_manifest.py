from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "FREEZE_MANIFEST.json"
EXPECTED_CANDIDATE_SHA256 = (
    "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa"
)
REQUIRED_FILES = (
    "README.md",
    "MODEL_CARD.md",
    "VALIDATION_SUMMARY.md",
    "CHANGELOG.md",
    "CANDIDATE/submission_key_target_encoding.csv",
    "CANDIDATE/submission_independent_baseline.csv",
    "SOURCE/project/source/run_independent_baseline.py",
    "SOURCE/project/tests/test_independent_features.py",
    "SOURCE/project/experiments/phase_01_fee_product_features/run_experiment.py",
    "SOURCE/project/experiments/phase_02_fold_safe_key_encodings/run_experiment.py",
    "SOURCE/project/experiments/phase_04_final_validation_gate/verify_final_gate.py",
    "REPORTS/FINAL_PHASE2_GATE_REPORT.md",
    "EVIDENCE/FINAL_PHASE2_GATE_EVIDENCE.json",
    "EVIDENCE/MODEL_LINEAGE_MANIFEST.json",
    "EVIDENCE/VALIDATION_EVIDENCE.json",
    "HASHES/SOURCE_SHA256_MANIFEST.json",
    "HASHES/INPUT_HASHES.json",
    "REPRODUCTION/verify_freeze.py",
    "REPRODUCTION/test_verify_freeze.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_head() -> str | None:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def write_manifest() -> dict:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)

    content_files = {
        path.relative_to(ROOT).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    }
    inputs = json.loads((ROOT / "HASHES" / "INPUT_HASHES.json").read_text(
        encoding="utf-8"
    ))
    manifest = {
        "schema_version": 1,
        "release_label": "FROZEN_DEVELOPMENT_CANDIDATE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/shammatthews-life/competetion-one",
        "source_commit_before_freeze": "db867d015d522e49aa895eff2630a71e7a4525a7",
        "git_head_at_manifest_generation": current_git_head(),
        "source_candidate": {
            "path": (
                "D:\\gci\\_gci_push_tmp\\champion\\FINAL_RESEARCH_AND_OPTIMIZATION"
                "\\runs\\phase02_key_te_20261101_181500"
                "\\submission_key_target_encoding.csv"
            ),
            "release_relative_path": "CANDIDATE/submission_key_target_encoding.csv",
            "sha256": EXPECTED_CANDIDATE_SHA256,
            "expected_rows": 61500,
            "columns": ["SK_ID_CURR", "TARGET"],
        },
        "raw_inputs": inputs["inputs"],
        "validation_metrics": {
            "original_seed": 42,
            "original_baseline_oof_auc": 0.7625457907047022,
            "original_phase2_oof_auc": 0.7715614423106346,
            "original_oof_auc_delta": 0.0090156516059324,
            "second_seed": 20261101,
            "second_seed_baseline_oof_auc": 0.7620389005999739,
            "second_seed_pooled_phase2_oof_auc": 0.7711413866258005,
            "second_seed_pooled_delta": 0.0091024860258265,
            "second_seed_fold_local_phase2_oof_auc": 0.7709638203765881,
            "second_seed_fold_local_delta": 0.0089249197766141,
            "frequency_decoder_enabled_vs_disabled_delta": 0.0031900391533038,
            "positive_paired_folds": 5,
            "folds": 5,
            "tests_passed": 18,
            "official_competition_score": "NOT_EVALUATED",
        },
        "required_files": list(REQUIRED_FILES),
        "content_files": content_files,
        "manifest_excludes_itself": True,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = write_manifest()
    print(
        f"Wrote freeze manifest for {len(result['content_files'])} files; "
        f"source candidate SHA-256={result['source_candidate']['sha256']}"
    )
