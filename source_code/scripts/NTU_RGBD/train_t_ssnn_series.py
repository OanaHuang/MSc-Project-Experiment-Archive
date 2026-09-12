from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TSSNNExperiment:
    experiment: str
    stage_steps: str
    early_classifier: bool
    purpose: str


EXPERIMENTS = (
    TSSNNExperiment("t_ssnn1", "4-4-4-4", False, "full-timestep temporal baseline"),
    TSSNNExperiment("t_ssnn2", "4-3-2-1", False, "timestep shrinkage"),
    TSSNNExperiment("t_ssnn3", "4-3-2-1", True, "shrinkage plus early classifiers"),
)
EXPERIMENT_BY_ID = {item.experiment: item for item in EXPERIMENTS}


def build_command(experiment: str, *, python_executable: str = sys.executable,
                  device: str = "cuda:0", physical_gpu: int | None = None,
                  seed: int = 42, epochs: int = 20,
                  batch_size: int | None = None, num_workers: int | None = None,
                  max_train_samples: int | None = None,
                  max_validation_samples: int | None = None,
                  skip_visualization: bool = False,
                  resume: bool = False) -> list[str]:
    if experiment not in EXPERIMENT_BY_ID:
        raise ValueError(f"Unknown T-SSNN experiment: {experiment}")
    command = [
        python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", experiment, "--category", "t_ssnn",
        "--batch-name", f"clip4_256_{epochs}ep", "--seed", str(seed),
        "--epochs", str(epochs), "--device", device,
        "--extracted-frames-dir", "Datasets/NTU_RGBD/frames/S010/clip4_256",
        "--frame-stride", "1", "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    ]
    if physical_gpu is not None:
        command.extend(("--physical-gpu", str(physical_gpu)))
    for flag, value in (("batch-size", batch_size), ("num-workers", num_workers),
                        ("max-train-samples", max_train_samples),
                        ("max-validation-samples", max_validation_samples)):
        if value is not None:
            command.extend((f"--{flag}", str(value)))
    if skip_visualization:
        command.append("--skip-visualization")
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="List or train the NTU T-SSNN series")
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
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("id\tstage_steps\tearly_classifier\tpurpose")
        for item in EXPERIMENTS:
            print(f"{item.experiment}\t{item.stage_steps}\t{item.early_classifier}\t{item.purpose}")
        return
    command = build_command(
        args.experiment, device=args.device, physical_gpu=args.physical_gpu,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        skip_visualization=args.skip_visualization, resume=args.resume,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
