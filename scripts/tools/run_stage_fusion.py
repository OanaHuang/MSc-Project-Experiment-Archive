from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.artifacts import output_run_dir, verify_completed_runs


EXPERIMENTS = ("s0", "s1", "s2", "s3", "s4", "s5", "s6")


def phase_command(args: argparse.Namespace, epochs: int, resume: bool) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tools" / "run_ablation.py"),
        "--dataset", "mpii",
        "--group", "stage_fusion",
        "--batch-name", args.batch_name,
        "--seeds", "42",
        "--devices", *(str(device) for device in args.devices),
        "--epochs", str(epochs),
        "--output-root", str(args.output_root),
    ]
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.num_workers is not None:
        command.extend(("--num-workers", str(args.num_workers)))
    if resume:
        command.append("--resume")
    return command


def verify_twenty_epoch_outputs(args: argparse.Namespace) -> None:
    verify_completed_runs(
        (
            (
                experiment,
                output_run_dir(
                    args.output_root, "mpii", "stage_fusion", args.batch_name,
                    experiment, 42,
                ),
            )
            for experiment in EXPERIMENTS
        ),
        context="20-epoch verification",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train S0-S6 for 20 epochs, wait for every run to finish, then "
            "resume the same runs through 120 total epochs."
        ),
    )
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-name", default="profile_b_seed42")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print both phase commands without launching training",
    )
    args = parser.parse_args()

    commands = (
        phase_command(args, epochs=20, resume=False),
        phase_command(args, epochs=120, resume=True),
    )
    if args.dry_run:
        for command in commands:
            print(" ".join(map(str, command)))
        return

    print("Stage-fusion phase 1/2: training every S0-S6 run to epoch 20", flush=True)
    subprocess.run(commands[0], cwd=PROJECT_ROOT, check=True)
    verify_twenty_epoch_outputs(args)
    print(
        "Stage-fusion phase 1/2 status and checkpoints verified for all runs; "
        "starting phase 2/2 to epoch 120",
        flush=True,
    )
    subprocess.run(commands[1], cwd=PROJECT_ROOT, check=True)
    print("Stage-fusion S0-S6 training completed through epoch 120", flush=True)


if __name__ == "__main__":
    main()
