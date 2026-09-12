from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
LOG_ROOT = PROJECT_ROOT / "launch_logs"


@dataclass(frozen=True)
class CFRun:
    display_id: str
    experiment: str
    gpu: int
    purpose: str


RUNS = (
    CFRun("CF5-Motion", "t5", 0, "motion-adaptive temporal weighting"),
    CFRun("CF6-Repeat4", "cf6", 1, "four-step repeated-current control"),
    CFRun("CF7-Reverse", "cf7", 2, "temporal-order control"),
    CFRun("CF8-Shuffle", "cf8", 3, "history-identity control"),
)


def build_command(run: CFRun, *, epochs: int = 20, seed: int = 42,
                  batch_size: int | None = None,
                  num_workers: int | None = None,
                  resume: bool = False) -> list[str]:
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "NTU_RGBD" / "train.py"),
        "--experiment", run.experiment,
        "--category", "cross_frame",
        "--batch-name", f"clip4_256_{epochs}ep",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--device", f"cuda:{run.gpu}",
        "--physical-gpu", str(run.gpu),
        "--extracted-frames-dir", "Datasets/NTU_RGBD/frames/S010/clip4_256",
        "--frame-stride", "1",
        "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3",
        "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--skip-visualization",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    ]
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    if num_workers is not None:
        command.extend(("--num-workers", str(num_workers)))
    if resume:
        command.append("--resume")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the first four-GPU NTU cross-frame batch",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if len(RUNS) != 4 or {run.gpu for run in RUNS} != {0, 1, 2, 3}:
        raise RuntimeError("CF batch must map exactly one run to each of GPUs 0-3")
    if not PYTHON.is_file() and args.launch:
        raise FileNotFoundError(f"Project virtualenv Python not found: {PYTHON}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    print("display_id\texperiment\tgpu\tepochs\tpurpose")
    for run in RUNS:
        print(f"{run.display_id}\t{run.experiment}\t{run.gpu}\t{args.epochs}\t{run.purpose}")
    if args.list:
        return
    commands = [
        build_command(
            run, epochs=args.epochs, seed=args.seed,
            batch_size=args.batch_size, num_workers=args.num_workers,
            resume=args.resume,
        )
        for run in RUNS
    ]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for run, command in zip(RUNS, commands, strict=True):
        log_path = LOG_ROOT / f"ntu_{run.experiment}_cf_20ep.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(f"launched {run.display_id} pid={process.pid} log={log_path}")


if __name__ == "__main__":
    main()
