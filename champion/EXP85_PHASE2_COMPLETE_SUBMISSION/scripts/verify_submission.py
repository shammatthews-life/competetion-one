from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from submission_builder import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently verify a Phase 2 submission CSV.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    args = parser.parse_args()
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    candidate = pd.read_csv(args.submission)
    result = validate_submission(candidate, test, sample)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
