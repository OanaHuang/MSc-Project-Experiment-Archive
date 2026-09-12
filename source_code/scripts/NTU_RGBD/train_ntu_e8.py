"""Train the E8 ResFormer-S4 architecture under the NTU NHB protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_command(
    *, python_executable: str = sys.executable, device: str = "cuda:0",
    physical_gpu: int | None = None, seed: int = 42, epochs: int = 20,
    batch_size: int | None = None, num_workers: int | None = None,
    resume: bool = False,
) -> list[str]:
    command = [
        python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", "ntu_e8", "--category", "external_baselines",
        "--batch-name", f"clip4_256_{epochs}ep", "--seed", str(seed),
        "--epochs", str(epochs), "--device", device,
        "--extracted-frames-dir", "Datasets/NTU_RGBD/frames/S010/clip4_256",
        "--frame-stride", "1", "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
        "--skip-visualization",
    ]
    if physical_gpu is not None:
        command.extend(("--physical-gpu", str(physical_gpu)))
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    if num_workers is not None:
        command.extend(("--num-workers", str(num_workers)))
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_command(
        device=args.device, physical_gpu=args.physical_gpu, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, resume=args.resume,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
