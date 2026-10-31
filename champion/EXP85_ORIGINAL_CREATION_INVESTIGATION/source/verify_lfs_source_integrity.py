#!/usr/bin/env python3
"""Verify the checked-out Git LFS payloads used by the EXP85 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


LFS_TARGETS = (
    "gemini/EXP85_DEPENDENCY_MANIFEST.json",
    "gemini/EXP85_GENERATED_ARTIFACTS.json",
    "gemini/EXP85_SHA256_MANIFEST.json",
    "gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb",
    "gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb",
    "gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb",
    "gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb",
    "gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb",
    "gemini/build_exp85_optimal_macro_apex.py",
    "gemini/cache/oof_10f_xgb.npy",
    "gemini/cache/test_10f_xgb.npy",
    "gemini/cache/v55_members.npz",
    "gemini/cache/v_17_full_suite_bundle.npz",
    "gemini/cache/y_train.npy",
    "gemini/champion/cache/oof_10f_xgb.npy",
    "gemini/champion/cache/test_10f_xgb.npy",
    "gemini/champion/cache/v55_members.npz",
    "gemini/champion/cache/v_17_full_suite_bundle.npz",
    "gemini/champion/cache/v_20261027_precision_transductive_apex.csv",
    "gemini/champion/cache/y_train.npy",
    "gemini/champion/exp85_solo.ipynb",
    "gemini/champion/exp85_solo_runner.py",
    "gemini/champion/run_log.txt",
    "gemini/champion/submission_exp85_solo_champion.csv",
    "gemini/v_20261027_optimal_macro_apex_v85.csv",
    "gemini/v_20261027_precision_transductive_apex.csv",
)

NORMAL_GIT_SOURCE = "champion/exp85_solo.pre_independent_wiring_backup.ipynb"
NORMAL_GIT_SOURCE_SHA256 = (
    "1ccfab28ef8534fc176c757c79250547c03a041a922851d683c82d0df1241d23"
)


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lfs_listing(text: str) -> dict[str, str]:
    listing = {}
    for line in text.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        oid, _, path = fields
        if len(oid) == 64 and all(char in "0123456789abcdef" for char in oid.lower()):
            listing[path.replace("\\", "/")] = oid.lower()
    return listing


def verify(repo: Path, output_dir: Path) -> dict:
    started = time.monotonic()
    lfs_listing = parse_lfs_listing(run_git(repo, "lfs", "ls-files", "-l"))
    lfs_objects_dir = Path(run_git(repo, "rev-parse", "--git-path", "lfs/objects").strip())
    if not lfs_objects_dir.is_absolute():
        lfs_objects_dir = repo / lfs_objects_dir

    records = []
    for relative_path in LFS_TARGETS:
        worktree_path = repo / Path(relative_path)
        oid = lfs_listing.get(relative_path)
        exists = worktree_path.is_file()
        pointer_only = False
        actual_sha256 = None
        size_bytes = None
        object_cached = False
        if exists:
            size_bytes = worktree_path.stat().st_size
            with worktree_path.open("rb") as stream:
                pointer_only = stream.read(128).startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                )
            if not pointer_only:
                actual_sha256 = sha256_file(worktree_path)
        object_path = (
            lfs_objects_dir / oid[:2] / oid[2:4] / oid if oid is not None else None
        )
        if object_path is not None and object_path.is_file():
            object_cached = sha256_file(object_path) == oid
        records.append(
            {
                "path": relative_path,
                "tracked_by_lfs": oid is not None,
                "lfs_sha256": oid,
                "worktree_exists": exists,
                "worktree_is_pointer_only": pointer_only,
                "worktree_size_bytes": size_bytes,
                "worktree_sha256": actual_sha256,
                "worktree_matches_lfs_oid": actual_sha256 == oid if oid else False,
                "lfs_object_cached_and_valid": object_cached,
            }
        )

    normal_path = repo / Path(NORMAL_GIT_SOURCE)
    normal_tracked = True
    try:
        run_git(repo, "ls-files", "--error-unmatch", NORMAL_GIT_SOURCE)
    except RuntimeError:
        normal_tracked = False
    normal_source = {
        "path": NORMAL_GIT_SOURCE,
        "tracked_by_git": normal_tracked,
        "exists": normal_path.is_file(),
        "sha256": sha256_file(normal_path) if normal_path.is_file() else None,
    }
    normal_source["sha256_matches_baseline"] = (
        normal_source["sha256"] == NORMAL_GIT_SOURCE_SHA256
    )

    checks_passed = all(
        record["tracked_by_lfs"]
        and record["worktree_exists"]
        and not record["worktree_is_pointer_only"]
        and record["worktree_matches_lfs_oid"]
        for record in records
    ) and all(
        normal_source[key]
        for key in ("tracked_by_git", "exists", "sha256_matches_baseline")
    )
    return {
        "audit": "EXP85 historical source and cache LFS integrity",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "training_performed": False,
        "network_fetch_performed": False,
        "historical_files_modified": False,
        "raw_rebuild_certified": False,
        "repository": str(repo),
        "checked_lfs_files": len(records),
        "checked_out_lfs_payloads_matching_git_oids": sum(
            record["worktree_matches_lfs_oid"] for record in records
        ),
        "lfs_object_cache_valid_count": sum(
            record["lfs_object_cached_and_valid"] for record in records
        ),
        "checks_passed": checks_passed,
        "elapsed_seconds": time.monotonic() - started,
        "normal_git_source": normal_source,
        "files": records,
        "output_directory": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = verify(repo, output_dir)
    result["exit_code"] = 0 if result["checks_passed"] else 1
    result["script_sha256"] = sha256_file(Path(__file__).resolve())
    json_path = output_dir / "GEMINI_LFS_SOURCE_VERIFICATION.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    json_sha256 = sha256_file(json_path)
    md_path = output_dir / "GEMINI_LFS_SOURCE_VERIFICATION.md"
    lines = [
        "# EXP85 Git LFS source integrity",
        "",
        f"- Command: `{' '.join(result['command'])}`",
        f"- Exit code: `{result['exit_code']}`",
        f"- Elapsed seconds: `{result['elapsed_seconds']:.3f}`",
        f"- LFS objects checked: `{result['checked_lfs_files']}`",
        f"- Checked-out payloads matching LFS OIDs: "
        f"`{result['checked_out_lfs_payloads_matching_git_oids']}`",
        f"- Cached LFS objects independently hashed: "
        f"`{result['lfs_object_cache_valid_count']}`",
        f"- Full integrity result: `{'PASS' if result['checks_passed'] else 'FAIL'}`",
        f"- Checker source SHA-256: `{result['script_sha256']}`",
        f"- JSON evidence SHA-256: `{json_sha256}`",
        "- Training performed: `false`",
        "- Network fetch performed: `false`",
        "- Raw rebuild certified: `false`",
        "",
        "| Path | LFS OID | Worktree SHA-256 | Match | Local LFS object valid |",
        "|---|---|---|---|---|",
    ]
    for record in result["files"]:
        lines.append(
            "| `{path}` | `{lfs_sha256}` | `{worktree_sha256}` | "
            "{match} | {object_valid} |".format(
                path=record["path"],
                lfs_sha256=record["lfs_sha256"] or "missing",
                worktree_sha256=record["worktree_sha256"] or "missing",
                match="yes" if record["worktree_matches_lfs_oid"] else "no",
                object_valid=(
                    "yes" if record["lfs_object_cached_and_valid"] else "no"
                ),
            )
        )
    lines.extend(
        [
            "",
            f"The user-designated notebook is tracked by ordinary Git: "
            f"`{result['normal_git_source']['path']}` "
            f"(SHA-256 `{result['normal_git_source']['sha256']}`).",
            "",
            "This check validates the repository's LFS pointer OIDs against the "
            "checked-out payload bytes. It does not validate model lineage or "
            "certify a raw rebuild.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    markdown_sha256 = sha256_file(md_path)
    print(json.dumps(result, indent=2))
    print(f"JSON evidence SHA-256: {json_sha256}")
    print(f"Markdown evidence SHA-256: {markdown_sha256}")
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
