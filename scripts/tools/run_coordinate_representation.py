from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES = (
    ("coordinate_classification", ("cc1", "cc2", "cc3")),
    ("coordinate_regression", ("rg1", "rg2")),
)


def command(group: str, devices: list[int], args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/tools/run_ablation.py"),
        "--dataset", "mpii",
        "--group", group,
        "--batch-name", args.batch_name,
        "--seeds", "42",
        "--devices", *(str(device) for device in devices),
        "--epochs", "20",
        "--output-root", str(args.output_root),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the E9-based MPII coordinate-classification and regression series",
    )
    parser.add_argument("--cc-devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--rg-devices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--batch-name", default="profile_b_ep020_seed42")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    overlap = set(args.cc_devices) & set(args.rg_devices)
    if overlap:
        parser.error(f"CC and RG device lists overlap: {sorted(overlap)}")
    commands = (
        command("coordinate_classification", args.cc_devices, args),
        command("coordinate_regression", args.rg_devices, args),
    )
    if args.dry_run:
        for item in commands:
            print(" ".join(map(str, item)))
        return
    processes = [subprocess.Popen(item, cwd=PROJECT_ROOT) for item in commands]
    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError(f"Coordinate series failed with exit codes {failures}")
    print("MPII coordinate-classification and regression series completed", flush=True)


if __name__ == "__main__":
    main()
