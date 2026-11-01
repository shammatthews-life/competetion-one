from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from pipeline import DEFAULT_DATA_DIR, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain seed-42 Phase 2 and compare all folds/predictions with the frozen reference."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-id", help="Unique run name; never overwritten.")
    args = parser.parse_args()
    report = run_pipeline(args.data_dir, "reproduce", args.run_id)
    return 0 if report["reproduction_validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
