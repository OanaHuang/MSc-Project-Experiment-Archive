from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAME_SOURCE = "Datasets/NTU_RGBD/frames/S010/clip4_256"
METADATA_ROOT = "Datasets/NTU_RGBD/metadata/s010"


@dataclass(frozen=True)
class MotionExperiment:
    experiment: str
    history: str
    fusion: str
    purpose: str


EXPERIMENTS = (
    MotionExperiment("m0", "unused", "none", "current-frame T0 reference"),
    MotionExperiment("m1", "four-frame evidence", "mean", "ordinary video baseline"),
    MotionExperiment("m2", "three-frame forecast", "fixed residual", "test motion trend"),
    MotionExperiment("m3", "three-frame forecast", "joint gate", "adaptive fusion"),
)
EXPERIMENT_BY_ID = {item.experiment: item for item in EXPERIMENTS}


def build_command(
    experiment: str, *, python_executable: str = sys.executable,
    device: str = "cuda:0", physical_gpu: int | None = None,
    seed: int = 42, epochs: int = 20, batch_size: int | None = None,
    num_workers: int | None = None, resume: bool = False,
    skip_visualization: bool = True,
) -> list[str]:
    if experiment not in EXPERIMENT_BY_ID:
        raise ValueError(f"Unknown M-series experiment: {experiment}")
    command = [
        python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", experiment, "--category", "motion_trend",
        "--batch-name", f"clip4_256_{epochs}ep", "--seed", str(seed),
        "--epochs", str(epochs), "--device", device,
        "--extracted-frames-dir", FRAME_SOURCE,
        "--frame-stride", "1", "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--train-metadata", f"{METADATA_ROOT}/train_split.csv",
        "--validation-metadata", f"{METADATA_ROOT}/val_split.csv",
    ]
    if physical_gpu is not None:
        command.extend(("--physical-gpu", str(physical_gpu)))
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    if num_workers is not None:
        command.extend(("--num-workers", str(num_workers)))
    if resume:
        command.append("--resume")
    if skip_visualization:
        command.append("--skip-visualization")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NTU M motion-trend series")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_ID))
    action.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--with-visualization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("id\thistory\tfusion\tpurpose")
        for item in EXPERIMENTS:
            print(f"{item.experiment}\t{item.history}\t{item.fusion}\t{item.purpose}")
        return
    command = build_command(
        args.experiment, device=args.device, physical_gpu=args.physical_gpu,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, resume=args.resume,
        skip_visualization=not args.with_visualization,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
