from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FRAME_SOURCES = {
    "clip4_256": "Datasets/NTU_RGBD/frames/S010/clip4_256",
}


@dataclass(frozen=True)
class CrossFrameExperiment:
    experiment: str
    frames: int
    offsets: tuple[int | str, ...]
    comparison: str
    purpose: str


EXPERIMENTS = (
    CrossFrameExperiment("t0", 1, (0,), "spatial baseline", "Current frame only"),
    CrossFrameExperiment(
        "t1", 2, (0, 0), "temporal compute control",
        "Repeated current frame without historical information",
    ),
    CrossFrameExperiment(
        "t2", 4, (-3, -2, -1, 0), "joint-wise temporal baseline",
        "Learned causal temporal weights for each joint",
    ),
    CrossFrameExperiment(
        "t3", 4, (-3, -2, -1, 0), "T2 vs T3 spatial contribution",
        "Decoupled temporal aggregation plus NTU skeleton refinement",
    ),
    CrossFrameExperiment(
        "t4", 4, (-3, -2, -1, 0), "robust temporal weighting",
        "Per-joint heatmap-confidence temporal gate",
    ),
    CrossFrameExperiment(
        "t5", 4, (-3, -2, -1, 0), "motion-adaptive temporal weighting",
        "Per-joint soft-coordinate motion gate",
    ),
)
EXPERIMENT_BY_ID = {item.experiment: item for item in EXPERIMENTS}


def build_command(
    experiment: str,
    *,
    frame_source: str = "clip4_256",
    python_executable: str = sys.executable,
    device: str = "cuda:0",
    physical_gpu: int | None = None,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    skip_visualization: bool = False,
    resume: bool = False,
) -> list[str]:
    if experiment not in EXPERIMENT_BY_ID:
        raise ValueError(f"Unknown cross-frame experiment: {experiment}")
    if frame_source not in FRAME_SOURCES:
        raise ValueError(f"Unknown frame source: {frame_source}")
    metadata = "Datasets/NTU_RGBD/metadata/s010"
    command = [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "NTU_RGBD" / "train.py"),
        "--experiment", experiment,
        "--category", "t_series",
        "--batch-name", f"clip4_256_{epochs}ep",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--device", device,
        "--extracted-frames-dir", FRAME_SOURCES[frame_source],
        "--frame-stride", "1",
        "--temporal-frame-gap", "1",
        # Every experiment predicts the exact same current-frame population.
        "--minimum-temporal-history", "3",
        "--preprocessed-pose-cache",
        "--disable-augmentation",
        "--train-metadata", f"{metadata}/train_split.csv",
        "--validation-metadata", f"{metadata}/val_split.csv",
    ]
    if physical_gpu is not None:
        command.extend(("--physical-gpu", str(physical_gpu)))
    for flag, value in {
        "batch-size": batch_size,
        "num-workers": num_workers,
        "max-train-samples": max_train_samples,
        "max-validation-samples": max_validation_samples,
    }.items():
        if value is not None:
            command.extend((f"--{flag}", str(value)))
    if skip_visualization:
        command.append("--skip-visualization")
    if resume:
        command.append("--resume")
    return command


def print_manifest() -> None:
    print("id\tframes\toffsets\tcomparison\tpurpose")
    for item in EXPERIMENTS:
        offsets = "[" + ", ".join(map(str, item.offsets)) + "]"
        print(
            f"{item.experiment}\t{item.frames}\t{offsets}\t"
            f"{item.comparison}\t{item.purpose}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List or train the causal NTU cross-frame state T-series",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_ID))
    action.add_argument("--list", action="store_true")
    parser.add_argument("--frame-source", choices=tuple(FRAME_SOURCES),
                        default="clip4_256")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print_manifest()
        return
    command = build_command(
        args.experiment, frame_source=args.frame_source, device=args.device,
        physical_gpu=args.physical_gpu, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        skip_visualization=args.skip_visualization, resume=args.resume,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    manifest = PROJECT_ROOT / FRAME_SOURCES[args.frame_source] / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Prepared clip dataset manifest not found: {manifest}"
        )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
