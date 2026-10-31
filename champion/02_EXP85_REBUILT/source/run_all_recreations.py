"""
run_all_recreations.py
================================================================================
GCI World 2026 - Master Cache Recreator (All 6 Artifacts from Scratch)
================================================================================
Rule Compliance: Strictly Independent (train.csv and test.csv ONLY)
Zero Upstream Caches | Zero Pre-computed Feathers | Zero Serialized Pickles

Generates all 6 cache dependencies needed by EXP_85:
  1. cache/v55_members.npz                    (generate_v55_from_scratch.py)
  2. cache/v_17_full_suite_bundle.npz         (generate_v17_from_scratch.py)
  3. cache/v_20261027_precision_transductive_apex.csv (generate_anchor74_from_scratch.py)
  4. cache/oof_10f_xgb.npy                    (generate_xgb10_from_scratch.py)
  5. cache/test_10f_xgb.npy                   (generate_xgb10_from_scratch.py)
  6. cache/y_train.npy                        (generate_xgb10_from_scratch.py)
================================================================================
"""

from pathlib import Path
import sys
import time
import argparse
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent

def find_script(candidate_names):
    """Finds script by searching candidate names in SCRIPT_DIR and adjacent lineage folders."""
    for name in candidate_names:
        p = SCRIPT_DIR / name
        if p.is_file():
            return p
    # Search in 04_MODEL_LINEAGE
    for parent in [SCRIPT_DIR.parent, SCRIPT_DIR.parents[1]]:
        lineage_dir = parent / "04_MODEL_LINEAGE"
        if lineage_dir.is_dir():
            for name in candidate_names:
                matches = list(lineage_dir.rglob(name))
                if matches:
                    return matches[0]
    return SCRIPT_DIR / candidate_names[0]

SCRIPT_CANDIDATES = [
    ("10-Fold XGBoost & Ground Truth Labels", ["generate_xgb10_from_scratch.py", "recreate_xgb10_and_labels.py"]),
    ("V55 Multi-Architecture Suite (7 distinct models)", ["generate_v55_from_scratch.py", "recreate_v55_members.py"]),
    ("V17 Full Suite Bundle (5 streams + Grand Fusion)", ["generate_v17_from_scratch.py", "recreate_v17_bundle.py"]),
    ("Precision Transductive Anchor (Bedrock Anchor 74)", ["generate_anchor74_from_scratch.py", "recreate_precision_anchor.py"]),
]

def run_pipeline(fast=False, data_dir=None, output_dir=None):
    print("=" * 85)
    print("  GCI World 2026: MASTER CACHE RECREATION PIPELINE (ALL FROM RAW DATA)")
    print("=" * 85)
    t_master = time.time()

    resolved_scripts = []
    for desc, candidates in SCRIPT_CANDIDATES:
        p = find_script(candidates)
        if not p.is_file():
            print(f"[!] ERROR: Could not locate generator for {desc} (checked: {candidates})")
            sys.exit(1)
        resolved_scripts.append((desc, p))

    for idx, (desc, script_path) in enumerate(resolved_scripts, 1):
        print(f"\n[{idx}/4] Running {desc}...")
        print(f"      Script: {script_path.name} ({script_path.parent})")
        cmd = [sys.executable, str(script_path)]
        if fast:
            cmd.append("--fast")
        if data_dir:
            cmd.extend(["--data_dir", str(data_dir)])
        if output_dir:
            cmd.extend(["--output_dir", str(output_dir)])

        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(script_path.parent), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] ERROR in {script_path.name}:\n{res.stderr}\n{res.stdout}")
            sys.exit(res.returncode)
        print(f"      -> Completed in {time.time()-t0:.1f}s")

    print("\n" + "=" * 85)
    print(f"  ALL 6 CACHE DEPENDENCIES RECREATED SUCCESSFULLY IN {time.time()-t_master:.1f}s!")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master recreation pipeline for all 6 cache dependencies")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode for all generators")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to raw competition data ('inpuy')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated artifacts")
    args = parser.parse_args()
    run_pipeline(fast=args.fast, data_dir=args.data_dir, output_dir=args.output_dir)
