"""Backfill MPII PCKh curves/AUC from an existing prediction archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.evaluation.pckh_curve import backfill_prediction_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.archive.with_name("pckh_curve_auc.json")
    print(json.dumps(backfill_prediction_archive(args.archive, output), indent=2))


if __name__ == "__main__":
    main()
