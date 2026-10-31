"""
run_all_recreations.py
================================================================================
GCI World 2026 - Master Cache Recreator (All 6 Artifacts from Scratch)
================================================================================
Rule Compliance: Strictly Independent (train.csv and test.csv ONLY)
Zero Upstream Caches | Zero Pre-computed Feathers | Zero Serialized Pickles

Generates all 6 cache dependencies needed by EXP_85:
  1. cache/v55_members.npz                    (recreate_v55_members.py)
  2. cache/v_17_full_suite_bundle.npz         (recreate_v17_bundle.py)
  3. cache/v_20261027_precision_transductive_apex.csv (recreate_precision_anchor.py)
  4. cache/oof_10f_xgb.npy                    (recreate_xgb10_and_labels.py)
  5. cache/test_10f_xgb.npy                   (recreate_xgb10_and_labels.py)
  6. cache/y_train.npy                        (recreate_xgb10_and_labels.py)
================================================================================
"""

from pathlib import Path
import sys
import time
import argparse
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("10-Fold XGBoost & Ground Truth Labels", SCRIPT_DIR / "recreate_xgb10_and_labels.py"),
    ("V55 Multi-Architecture Suite (7 distinct models)", SCRIPT_DIR / "recreate_v55_members.py"),
    ("V17 Full Suite Bundle (5 streams + Grand Fusion)", SCRIPT_DIR / "recreate_v17_bundle.py"),
    ("Precision Transductive Anchor (Bedrock Anchor 74)", SCRIPT_DIR / "recreate_precision_anchor.py"),
]


def run_pipeline(fast=False):
    print("=" * 85)
    print("  GCI World 2026: MASTER CACHE RECREATION PIPELINE (ALL FROM RAW DATA)")
    print("=" * 85)
    t_master = time.time()

    for idx, (desc, script_path) in enumerate(SCRIPTS, 1):
        print(f"\n[{idx}/4] Running {desc}...")
        print(f"      Script: {script_path.name}")
        cmd = [sys.executable, str(script_path)]
        if fast:
            cmd.append("--fast")

        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[!] ERROR in {script_path.name}:\n{res.stderr}")
            sys.exit(res.returncode)
        print(f"      -> Completed in {time.time()-t0:.1f}s")

    print("\n" + "=" * 85)
    print(f"  ALL 6 CACHE DEPENDENCIES RECREATED SUCCESSFULLY IN {time.time()-t_master:.1f}s!")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master recreation pipeline for all 6 cache dependencies")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode for all generators")
    args = parser.parse_args()
    run_pipeline(fast=args.fast)

