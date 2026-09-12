"""List or train the NTU external human-pose baseline (NHB) series."""

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
class NHBExperiment:
    experiment: str
    model: str
    initialization: str


EXPERIMENTS = (
    NHBExperiment("nhb1", "Pose-ResNet-50", "scratch"),
    NHBExperiment("nhb2", "Pose-ResNet-50", "MPII HB0"),
    NHBExperiment("nhb3", "HRNet-W32", "scratch"),
    NHBExperiment("nhb4", "HRNet-W32", "MPII HB1"),
    NHBExperiment("nhb5", "HRNet-W48", "scratch"),
    NHBExperiment("nhb6", "HRNet-W48", "MPII HB2"),
)
EXPERIMENT_BY_ID = {item.experiment: item for item in EXPERIMENTS}


def build_command(
    experiment: str, *, python_executable: str = sys.executable,
    device: str = "cuda:0", physical_gpu: int | None = None,
    seed: int = 42, epochs: int = 20, batch_size: int | None = None,
    num_workers: int | None = None, max_train_samples: int | None = None,
    max_validation_samples: int | None = None, resume: bool = False,
) -> list[str]:
    if experiment not in EXPERIMENT_BY_ID:
        raise ValueError(f"Unknown NHB experiment: {experiment}")
    command = [
        python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", experiment,
        "--category", "external_baselines",
        "--batch-name", f"clip4_256_{epochs}ep",
        "--seed", str(seed), "--epochs", str(epochs), "--device", device,
        "--extracted-frames-dir", FRAME_SOURCE,
        "--frame-stride", "1", "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--train-metadata", f"{METADATA_ROOT}/train_split.csv",
        "--validation-metadata", f"{METADATA_ROOT}/val_split.csv",
        "--skip-visualization",
    ]
    if physical_gpu is not None:
        command.extend(("--physical-gpu", str(physical_gpu)))
    for flag, value in {
        "batch-size": batch_size, "num-workers": num_workers,
        "max-train-samples": max_train_samples,
        "max-validation-samples": max_validation_samples,
    }.items():
        if value is not None:
            command.extend((f"--{flag}", str(value)))
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_ID))
    action.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("id\tmodel\tinitialization")
        for item in EXPERIMENTS:
            print(f"{item.experiment}\t{item.model}\t{item.initialization}")
        return
    command = build_command(
        args.experiment, device=args.device, physical_gpu=args.physical_gpu,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples, resume=args.resume,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    manifest = PROJECT_ROOT / FRAME_SOURCE / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepared clip dataset manifest not found: {manifest}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
