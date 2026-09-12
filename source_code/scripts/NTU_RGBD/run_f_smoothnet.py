from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"


def run(command: list[str]) -> None:
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and train one F-SmoothNet experiment")
    parser.add_argument("--f-id", required=True, choices=("f1", "f2", "f3", "f4"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New")
    args = parser.parse_args()
    source = args.output_root / "ntu_rgbd/frame_count/clip4_256_20ep" / args.f_id / "seed_42"
    checkpoint = source / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    fs_id = "fs" + args.f_id[1:]
    run_dir = args.output_root / "ntu_rgbd/f_smoothnet/window32_50ep" / fs_id / "seed_42"
    sequence_root = source / "smoothnet_sequences"
    for split, metadata in (
        ("train", "Datasets/NTU_RGBD/metadata/s010/train_split.csv"),
        ("validation", "Datasets/NTU_RGBD/metadata/s010/val_split.csv"),
    ):
        matrices = sequence_root / split / "prediction_matrices.npz"
        if not matrices.is_file():
            run([
                str(PYTHON), "scripts/NTU_RGBD/export_smoothnet_sequences.py",
                "--checkpoint", str(checkpoint), "--metadata", metadata,
                "--output-dir", str(sequence_root / split), "--device", args.device,
            ])
    run([
        str(PYTHON), "scripts/NTU_RGBD/train_smoothnet.py", "--id", fs_id,
        "--train-matrices", str(sequence_root / "train/prediction_matrices.npz"),
        "--validation-matrices", str(sequence_root / "validation/prediction_matrices.npz"),
        "--output-dir", str(run_dir), "--device", args.device, "--epochs", str(args.epochs),
    ])


if __name__ == "__main__":
    main()
