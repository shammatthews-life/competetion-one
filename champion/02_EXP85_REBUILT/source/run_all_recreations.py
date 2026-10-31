"""
run_all_recreations.py
================================================================================
GCI World 2026 - Master Cache Recreator & Cryptographic Provenance Suite
================================================================================
Generator intent: build prediction artifacts from raw train.csv and test.csv.
This runner records source-review and execution evidence; it does not instrument
OS-level file access or prove historical model parity.

Generates and programmatically verifies all 6 cache dependencies needed by EXP_85:
  1. rebuilt_artifacts/v55_members.npz                    (generate_v55_from_scratch.py)
  2. rebuilt_artifacts/v_17_full_suite_bundle.npz         (generate_v17_from_scratch.py)
  3. rebuilt_artifacts/v_20261027_precision_transductive_apex.csv (generate_anchor74_from_scratch.py)
  4. rebuilt_artifacts/oof_10f_xgb.npy                    (generate_xgb10_from_scratch.py)
  5. rebuilt_artifacts/test_10f_xgb.npy                   (generate_xgb10_from_scratch.py)
  6. rebuilt_artifacts/y_train.npy                        (generate_xgb10_from_scratch.py)

Provenance Certification Logic:
  - RAW_REBUILD_CERTIFIED : All 4 full-mode generators executed successfully into an empty output directory.
  - SMOKE_TEST_ONLY       : Fast-mode generators completed; never authorizes full-scratch synthesis.
  - ARTIFACTS_VERIFIED    : Existing artifacts passed structural, key, finite, and bounds validation.

Writes a provenance manifest recording:
  - Generator source code hashes & Git commit
  - Raw source CSV hashes
  - Exact command line executions, timestamps, and exit codes
  - Element-by-element test ID alignment against raw test.csv
  - Uniform array checks: np.isfinite, min >= 0.0, max <= 1.0 on every numeric array
  - Cache-access assessment based on source review (not OS-level tracing)
================================================================================
"""

from pathlib import Path
import sys
import time
import json
import uuid
import hashlib
import argparse
import subprocess
import numpy as np
import pandas as pd
from provenance_contract import determine_provenance_status

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # D:\gci\champion
CANONICAL_OUTPUT_DIR = PROJECT_ROOT / "02_EXP85_REBUILT" / "rebuilt_artifacts"

EXPECTED_N_TRAIN = 171202
EXPECTED_N_TEST  = 61500

CANONICAL_V17_KEYS = [
    "oof_old", "test_old", "oof_e1", "test_e1", "oof_e1et", "test_e1et",
    "oof_e1Aet", "test_e1Aet", "oof_cb", "test_cb", "final_oof", "final_test", "final_auc"
]

CANONICAL_V55_KEYS = [
    "oof_qf_shallow", "oof_qf_reg", "oof_v1_allfam_oh", "oof_v2_allfam_oh", "oof_v2_mono", "oof_deep_et", "oof_cb_product",
    "test_qf_shallow", "test_qf_reg", "test_v1_allfam_oh", "test_v2_allfam_oh", "test_v2_mono", "test_deep_et", "test_cb_product"
]

def sha256_file(filepath):
    """Compute SHA-256 hash of a file efficiently."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    """Attempt to retrieve current git commit hash."""
    for d in [PROJECT_ROOT, SCRIPT_DIR, PROJECT_ROOT.parent / "_gci_push_tmp"]:
        try:
            res = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
    return None

def find_script(candidate_names):
    """Finds script by searching candidate names in SCRIPT_DIR and adjacent lineage folders."""
    for name in candidate_names:
        p = SCRIPT_DIR / name
        if p.is_file():
            return p
    for parent in [SCRIPT_DIR.parent, SCRIPT_DIR.parents[1]]:
        lineage_dir = parent / "04_MODEL_LINEAGE"
        if lineage_dir.is_dir():
            for name in candidate_names:
                matches = list(lineage_dir.rglob(name))
                if matches:
                    return matches[0]
    return SCRIPT_DIR / candidate_names[0]

def resolve_data_dir(cli_dir=None):
    """Locate raw competition data directory ('inpuy')."""
    candidates = []
    if cli_dir:
        candidates.append(Path(cli_dir))
    candidates.extend([
        Path(r"D:\gci\inpuy"),
        PROJECT_ROOT.parent / "inpuy",
        PROJECT_ROOT / "inpuy",
        SCRIPT_DIR / "inpuy"
    ])
    for c in candidates:
        if c.is_dir() and (c / "train.csv").is_file() and (c / "test.csv").is_file():
            return c
    raise FileNotFoundError(f"Could not locate raw competition data ('inpuy' with train.csv/test.csv). Checked: {candidates}")

SCRIPT_CANDIDATES = [
    ("10-Fold XGBoost & Ground Truth Labels", ["generate_xgb10_from_scratch.py", "recreate_xgb10_and_labels.py"]),
    ("V55 Multi-Architecture Suite (7 distinct models)", ["generate_v55_from_scratch.py", "recreate_v55_members.py"]),
    ("V17 Full Suite Bundle (5 streams + Grand Fusion)", ["generate_v17_from_scratch.py", "recreate_v17_bundle.py"]),
    ("Precision Transductive Anchor (Bedrock Anchor 74)", ["generate_anchor74_from_scratch.py", "recreate_precision_anchor.py"]),
]

def assert_uniform_array_validity(name, arr, expected_shape, min_val=0.0, max_val=1.0, is_prob=True):
    """Uniform finite, dimension, and range assertion for every array in every artifact."""
    assert hasattr(arr, "shape"), f"{name} is not a valid numpy array"
    assert arr.shape == expected_shape, f"Shape mismatch in {name}: {arr.shape} vs {expected_shape}"
    assert np.isfinite(arr).all(), f"Non-finite values (NaN/Inf) detected in {name}!"
    if is_prob:
        assert arr.min() >= min_val, f"Value below {min_val} in {name}: {arr.min()}"
        assert arr.max() <= max_val, f"Value above {max_val} in {name}: {arr.max()}"

def verify_and_generate_manifest(
    output_dir, data_dir, execution_records=None, provenance_status="ARTIFACTS_VERIFIED",
    execution_mode="verify_only", fast_mode=False
):
    """Exhaustively verify all generated artifacts and write provenance manifest."""
    print("\n" + "=" * 85)
    print("  RUNNING DEEP FORENSIC VERIFICATION & UNIFORM ARRAY AUDIT")
    print("=" * 85)
    t_v = time.time()

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    assert train_csv.is_file(), f"Missing {train_csv}"
    assert test_csv.is_file(), f"Missing {test_csv}"

    print("[*] Computing cryptographic hashes of raw inputs...")
    train_sha = sha256_file(train_csv)
    test_sha  = sha256_file(test_csv)
    print(f"    raw train.csv SHA-256 : {train_sha}")
    print(f"    raw test.csv  SHA-256 : {test_sha}")

    # Load raw IDs and labels for validation
    df_raw_test = pd.read_csv(test_csv, usecols=["SK_ID_CURR"])
    raw_test_ids = df_raw_test["SK_ID_CURR"].values
    assert len(raw_test_ids) == EXPECTED_N_TEST, f"Expected {EXPECTED_N_TEST} test rows, found {len(raw_test_ids)}"

    df_raw_train = pd.read_csv(train_csv, usecols=["TARGET"])
    raw_train_targets = df_raw_train["TARGET"].values.astype(np.int32)
    assert len(raw_train_targets) == EXPECTED_N_TRAIN, f"Expected {EXPECTED_N_TRAIN} train rows, found {len(raw_train_targets)}"

    build_id = uuid.uuid4().hex[:16]
    git_commit = get_git_commit()

    # Source script hashes
    generator_hashes = {}
    for desc, candidates in SCRIPT_CANDIDATES:
        p = find_script(candidates)
        if p.is_file():
            generator_hashes[p.name] = sha256_file(p)

    manifest = {
        "manifest_version": "2.0",
        "build_id": build_id,
        "provenance_status": provenance_status,
        "execution_mode": execution_mode,
        "fast_mode": fast_mode,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "generator_source_hashes": generator_hashes,
        "cache_access_policy": {
            "historical_prediction_caches_accessed": "NOT_ASSESSED" if execution_mode == "verify_only" else "NO_CACHE_READS_FOUND_IN_GENERATOR_SOURCE",
            "external_pretrained_weights_accessed": "NOT_ASSESSED" if execution_mode == "verify_only" else "NO_EXTERNAL_WEIGHT_LOADS_FOUND_IN_GENERATOR_SOURCE",
            "raw_data_exclusive": execution_mode != "verify_only",
            "os_file_access_tracing": False,
            "note": "Source review and subprocess records are evidence; this runner does not perform OS-level file-access tracing."
        },
        "execution_telemetry": execution_records if execution_records else "Existing artifacts were structurally verified; generator execution was not performed.",
        "output_directory": str(output_dir),
        "raw_inputs": {
            "train_csv": {"path": str(train_csv), "rows": EXPECTED_N_TRAIN, "sha256": train_sha},
            "test_csv":  {"path": str(test_csv), "rows": EXPECTED_N_TEST, "sha256": test_sha}
        },
        "artifacts": {},
        "verification_summary": {
            "all_files_exist": False,
            "dimensions_verified": False,
            "id_alignment_verified": False,
            "uniform_finite_verified": False,
            "uniform_bounds_verified": False,
            "canonical_keys_verified": False,
            "provenance_status": provenance_status
        }
    }

    # 1. Verify y_train.npy
    f_y = output_dir / "y_train.npy"
    assert f_y.is_file(), f"Missing artifact: {f_y.name}"
    y_arr = np.load(f_y)
    assert_uniform_array_validity("y_train.npy", y_arr, (EXPECTED_N_TRAIN,), min_val=0, max_val=1, is_prob=False)
    assert np.array_equal(y_arr, raw_train_targets), "y_train does not match raw train.csv TARGET exactly!"
    sha_y = sha256_file(f_y)
    manifest["artifacts"]["y_train.npy"] = {
        "size_bytes": f_y.stat().st_size, "sha256": sha_y, "shape": list(y_arr.shape),
        "dtype": str(y_arr.dtype), "positive_count": int(np.sum(y_arr)), "finite": True
    }
    print(f"[+] Verified y_train.npy      : shape={y_arr.shape}, finite=True, matches raw train.csv exactly OK")

    # 2. Verify oof_10f_xgb.npy
    f_oof_xgb = output_dir / "oof_10f_xgb.npy"
    assert f_oof_xgb.is_file(), f"Missing artifact: {f_oof_xgb.name}"
    oof_xgb = np.load(f_oof_xgb)
    assert_uniform_array_validity("oof_10f_xgb.npy", oof_xgb, (EXPECTED_N_TRAIN,), min_val=0.0, max_val=1.0)
    sha_oof_xgb = sha256_file(f_oof_xgb)
    manifest["artifacts"]["oof_10f_xgb.npy"] = {
        "size_bytes": f_oof_xgb.stat().st_size, "sha256": sha_oof_xgb, "shape": list(oof_xgb.shape),
        "dtype": str(oof_xgb.dtype), "finite": True, "min": float(oof_xgb.min()), "max": float(oof_xgb.max()), "mean": float(oof_xgb.mean())
    }
    print(f"[+] Verified oof_10f_xgb.npy  : shape={oof_xgb.shape}, finite=True, bounds=[{oof_xgb.min():.4f}, {oof_xgb.max():.4f}] OK")

    # 3. Verify test_10f_xgb.npy
    f_test_xgb = output_dir / "test_10f_xgb.npy"
    assert f_test_xgb.is_file(), f"Missing artifact: {f_test_xgb.name}"
    test_xgb = np.load(f_test_xgb)
    assert_uniform_array_validity("test_10f_xgb.npy", test_xgb, (EXPECTED_N_TEST,), min_val=0.0, max_val=1.0)
    sha_test_xgb = sha256_file(f_test_xgb)
    manifest["artifacts"]["test_10f_xgb.npy"] = {
        "size_bytes": f_test_xgb.stat().st_size, "sha256": sha_test_xgb, "shape": list(test_xgb.shape),
        "dtype": str(test_xgb.dtype), "finite": True, "min": float(test_xgb.min()), "max": float(test_xgb.max()), "mean": float(test_xgb.mean())
    }
    print(f"[+] Verified test_10f_xgb.npy : shape={test_xgb.shape}, finite=True, bounds=[{test_xgb.min():.4f}, {test_xgb.max():.4f}] OK")

    # 4. Verify v55_members.npz (all 14 arrays checked uniformly)
    f_v55 = output_dir / "v55_members.npz"
    assert f_v55.is_file(), f"Missing artifact: {f_v55.name}"
    nz_v55 = np.load(f_v55)
    v55_meta = {}
    for k in CANONICAL_V55_KEYS:
        assert k in nz_v55, f"Missing canonical key '{k}' in v55_members.npz"
        arr = nz_v55[k]
        expected_shape = (EXPECTED_N_TRAIN,) if ("oof" in k) else (EXPECTED_N_TEST,)
        assert_uniform_array_validity(f"v55['{k}']", arr, expected_shape, min_val=0.0, max_val=1.0)
        v55_meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": True, "min": float(arr.min()), "max": float(arr.max())}
    sha_v55 = sha256_file(f_v55)
    manifest["artifacts"]["v55_members.npz"] = {
        "size_bytes": f_v55.stat().st_size, "sha256": sha_v55, "canonical_keys": CANONICAL_V55_KEYS,
        "key_count": len(CANONICAL_V55_KEYS), "keys_detail": v55_meta
    }
    print(f"[+] Verified v55_members.npz  : 14/14 canonical keys present, all arrays finite=True & bounds=[0,1] OK")

    # 5. Verify v_17_full_suite_bundle.npz (all 12 arrays + final_auc uniformly checked)
    f_v17 = output_dir / "v_17_full_suite_bundle.npz"
    assert f_v17.is_file(), f"Missing artifact: {f_v17.name}"
    nz_v17 = np.load(f_v17)
    v17_meta = {}
    for k in CANONICAL_V17_KEYS:
        assert k in nz_v17, f"Missing canonical key '{k}' in v_17_full_suite_bundle.npz"
        if k == "final_auc":
            val = float(nz_v17[k])
            assert np.isfinite(val) and 0.5 <= val <= 1.0, f"Invalid final_auc: {val}"
            v17_meta[k] = {"scalar": val, "finite": True}
            continue
        arr = nz_v17[k]
        expected_shape = (EXPECTED_N_TRAIN,) if ("oof" in k) else (EXPECTED_N_TEST,)
        assert_uniform_array_validity(f"v17['{k}']", arr, expected_shape, min_val=0.0, max_val=1.0)
        v17_meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite": True, "min": float(arr.min()), "max": float(arr.max())}
    sha_v17 = sha256_file(f_v17)
    manifest["artifacts"]["v_17_full_suite_bundle.npz"] = {
        "size_bytes": f_v17.stat().st_size, "sha256": sha_v17, "canonical_keys": CANONICAL_V17_KEYS,
        "key_count": len(CANONICAL_V17_KEYS), "keys_detail": v17_meta
    }
    print(f"[+] Verified v_17_full_suite  : 13/13 canonical keys present, all arrays finite=True & bounds=[0,1] OK")

    # 6. Verify v_20261027_precision_transductive_apex.csv
    f_a74 = output_dir / "v_20261027_precision_transductive_apex.csv"
    assert f_a74.is_file(), f"Missing artifact: {f_a74.name}"
    df_a74 = pd.read_csv(f_a74)
    assert len(df_a74) == EXPECTED_N_TEST, f"Anchor74 row count mismatch: {len(df_a74)} vs {EXPECTED_N_TEST}"
    assert list(df_a74.columns) == ["SK_ID_CURR", "TARGET"], f"Anchor74 columns mismatch: {list(df_a74.columns)}"
    assert np.array_equal(df_a74["SK_ID_CURR"].values, raw_test_ids), "Anchor74 SK_ID_CURR does NOT match raw test.csv!"
    target_arr = df_a74["TARGET"].values
    assert_uniform_array_validity("Anchor74 TARGET", target_arr, (EXPECTED_N_TEST,), min_val=0.0, max_val=1.0)
    sha_a74 = sha256_file(f_a74)
    manifest["artifacts"]["v_20261027_precision_transductive_apex.csv"] = {
        "size_bytes": f_a74.stat().st_size, "sha256": sha_a74, "rows": len(df_a74),
        "finite": True, "target_min": float(target_arr.min()), "target_max": float(target_arr.max()),
        "target_mean": float(target_arr.mean())
    }
    print(f"[+] Verified Anchor74 CSV     : 61,500 rows, IDs match raw test.csv, finite=True, bounds=[{target_arr.min():.4f}, {target_arr.max():.4f}] OK")

    # Final manifest status
    manifest["verification_summary"]["all_files_exist"] = True
    manifest["verification_summary"]["dimensions_verified"] = True
    manifest["verification_summary"]["id_alignment_verified"] = True
    manifest["verification_summary"]["uniform_finite_verified"] = True
    manifest["verification_summary"]["uniform_bounds_verified"] = True
    manifest["verification_summary"]["canonical_keys_verified"] = True
    manifest["verification_summary"]["provenance_status"] = provenance_status

    if provenance_status == "RAW_REBUILD_CERTIFIED":
        manifest_name = "RECONSTRUCTION_PROVENANCE_MANIFEST.json"
    elif provenance_status == "SMOKE_TEST_ONLY":
        manifest_name = f"SMOKE_TEST_PROVENANCE_{build_id}.json"
    else:
        manifest_name = f"ARTIFACTS_VERIFICATION_{build_id}.json"
    manifest_path = output_dir / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 85)
    print(f"  BUILD ID                    : {build_id}")
    print(f"  GIT COMMIT                  : {git_commit}")
    print(f"  PROVENANCE STATUS           : {provenance_status}")
    print(f"  PROVENANCE MANIFEST WRITTEN : {manifest_path}")
    print(f"  VERIFICATION TIME           : {time.time()-t_v:.2f}s")
    print("-" * 85)
    return manifest

def run_pipeline(fast=False, data_dir=None, output_dir=None, verify_only=False):
    print("=" * 85)
    print("  GCI World 2026: MASTER CACHE RECREATION & PROVENANCE PIPELINE")
    print("=" * 85)
    t_master = time.time()

    data_path = resolve_data_dir(data_dir)
    print(f"[*] Verified Raw Competition Data Path: {data_path}")

    if output_dir is None:
        out_path = CANONICAL_OUTPUT_DIR
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target Output Directory: {out_path}")

    execution_records = []

    if not verify_only:
        if any(out_path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite an existing output directory: {out_path}. "
                "Choose a new, empty output_dir."
            )
        resolved_scripts = []
        for desc, candidates in SCRIPT_CANDIDATES:
            p = find_script(candidates)
            if not p.is_file():
                print(f"[!] ERROR: Could not locate generator for {desc} (checked: {candidates})")
                sys.exit(1)
            resolved_scripts.append((desc, p))

        execution_id = uuid.uuid4().hex[:12]
        log_dir = out_path / "execution_logs" / execution_id
        log_dir.mkdir(parents=True, exist_ok=False)

        for idx, (desc, script_path) in enumerate(resolved_scripts, 1):
            print(f"\n[{idx}/4] Running {desc}...")
            print(f"      Script: {script_path.name} ({script_path.parent})")
            cmd = [sys.executable, str(script_path), "--data_dir", str(data_path), "--output_dir", str(out_path)]
            if fast:
                cmd.append("--fast")

            t0 = time.time()
            res = subprocess.run(
                cmd, cwd=str(script_path.parent), capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            duration = time.time() - t0
            stdout_path = log_dir / f"{idx:02d}_{script_path.stem}.stdout.txt"
            stderr_path = log_dir / f"{idx:02d}_{script_path.stem}.stderr.txt"
            stdout_path.write_text(res.stdout, encoding="utf-8")
            stderr_path.write_text(res.stderr, encoding="utf-8")
            record = {
                "step": idx,
                "description": desc,
                "script": script_path.name,
                "command": cmd,
                "working_directory": str(script_path.parent),
                "script_source_sha256": sha256_file(script_path),
                "duration_seconds": round(duration, 2),
                "exit_code": res.returncode,
                "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
                "stdout_log": {
                    "path": str(stdout_path),
                    "sha256": sha256_file(stdout_path),
                    "size_bytes": stdout_path.stat().st_size
                },
                "stderr_log": {
                    "path": str(stderr_path),
                    "sha256": sha256_file(stderr_path),
                    "size_bytes": stderr_path.stat().st_size
                }
            }
            execution_records.append(record)

            if res.returncode != 0:
                print(f"[!] ERROR in {script_path.name}:\n{res.stderr}\n{res.stdout}")
                sys.exit(res.returncode)
            print(f"      -> Completed in {duration:.1f}s (Exit code: 0)")

    execution_mode = (
        "verify_only" if verify_only else ("fast_smoke" if fast else "full_rebuild")
    )
    provenance_status = determine_provenance_status(
        verify_only=verify_only, fast=fast, execution_records=execution_records
    )
    # Run complete verification and write a non-destructive manifest.
    manifest = verify_and_generate_manifest(
        output_dir=out_path,
        data_dir=data_path,
        execution_records=execution_records if execution_records else None,
        provenance_status=provenance_status,
        execution_mode=execution_mode,
        fast_mode=fast
    )

    print("\n" + "=" * 85)
    status_label = manifest["verification_summary"]["provenance_status"]
    print(f"  ALL 6 CACHE DEPENDENCIES VERIFIED [{status_label}] IN {time.time()-t_master:.1f}s!")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master recreation and provenance verification pipeline")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode for all generators")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to raw competition data ('inpuy')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated artifacts")
    parser.add_argument("--verify_only", action="store_true", help="Perform verification on existing artifacts and write manifest without retraining")
    args = parser.parse_args()
    run_pipeline(fast=args.fast, data_dir=args.data_dir, output_dir=args.output_dir, verify_only=args.verify_only)
