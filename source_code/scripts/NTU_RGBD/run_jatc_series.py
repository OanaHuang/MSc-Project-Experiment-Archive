from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = PROJECT_ROOT / ".venv/bin/python"


@dataclass(frozen=True)
class JATCExperiment:
    run_id: str
    experiment: str
    seed: int
    stage: str
    purpose: str


EXPERIMENTS = (
    JATCExperiment("jp0", "jp0", 42, "offline", "T0 reference on identical sequences"),
    JATCExperiment("jp1", "jp1", 42, "offline", "global fixed decay sweep"),
    JATCExperiment("jp2", "jp2", 42, "offline", "per-joint fixed decay sweep"),
    JATCExperiment("jp3", "jp3", 42, "offline", "confidence-motion rule decay"),
    JATCExperiment("jp4", "jp4", 42, "training", "learned adaptive decay seed 42"),
    JATCExperiment("jp4_s2", "jp4", 3407, "replication", "learned adaptive decay seed 3407"),
    JATCExperiment("jp4_s3", "jp4", 2026, "replication", "learned adaptive decay seed 2026"),
)
EXPERIMENT_BY_ID = {item.run_id: item for item in EXPERIMENTS}


def source_run(root: Path, source_category: str, source_batch: str) -> Path:
    return root / "ntu_rgbd" / source_category / source_batch / "t0/seed_42"


def run_dir(root: Path, item: JATCExperiment, epochs: int) -> Path:
    return root / "ntu_rgbd/jatc" / f"window16_{epochs}ep" / item.experiment / f"seed_{item.seed}"


def build_command(run_id: str, *, python_executable: str = sys.executable,
                  output_root: Path = PROJECT_ROOT / "Outputs_New", epochs: int = 30,
                  device: str = "cpu", source_category: str = "t_ssnn",
                  source_batch: str = "clip4_256_20ep") -> list[str]:
    item = EXPERIMENT_BY_ID[run_id]
    source = source_run(output_root, source_category, source_batch)
    sequences = source / "jatc_sequences"
    command = [
        python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/run_jatc_pilot.py"),
        "--experiment", item.experiment,
        "--train-matrices", str(sequences / "train/prediction_matrices.npz"),
        "--validation-matrices", str(sequences / "validation/prediction_matrices.npz"),
        "--output-dir", str(run_dir(output_root, item, epochs)),
        "--seed", str(item.seed), "--epochs", str(epochs), "--device", device,
    ]
    if item.experiment in {"jp3", "jp4"}:
        jp2 = EXPERIMENT_BY_ID["jp2"]
        command.extend(("--jp2-beta", str(
            run_dir(output_root, jp2, epochs) / "selected_beta_per_joint.json"
        )))
    return command


def export_commands(*, python_executable: str, output_root: Path,
                    source_category: str, source_batch: str, device: str) -> tuple[list[str], ...]:
    source = source_run(output_root, source_category, source_batch)
    checkpoint = source / "checkpoints/best.pt"
    base = [python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/export_smoothnet_sequences.py"),
            "--checkpoint", str(checkpoint), "--device", device]
    return (
        base + ["--metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
                "--output-dir", str(source / "jatc_sequences/train")],
        base + ["--metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
                "--output-dir", str(source / "jatc_sequences/validation")],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List, export, or run one NTU JATC pilot experiment")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--export", action="store_true")
    action.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_ID))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New")
    parser.add_argument("--source-category", default="t_ssnn")
    parser.add_argument("--source-batch", default="clip4_256_20ep")
    args = parser.parse_args()
    if args.list:
        print("id\texperiment\tseed\tstage\tpurpose")
        for item in EXPERIMENTS:
            print(f"{item.run_id}\t{item.experiment}\t{item.seed}\t{item.stage}\t{item.purpose}")
        return
    python = str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable))
    commands = export_commands(
        python_executable=python, output_root=args.output_root,
        source_category=args.source_category, source_batch=args.source_batch,
        device=args.device,
    ) if args.export else (build_command(
        args.experiment, python_executable=python, output_root=args.output_root,
        epochs=args.epochs, device=args.device, source_category=args.source_category,
        source_batch=args.source_batch,
    ),)
    for command in commands:
        if args.export:
            destination = Path(command[command.index("--output-dir") + 1])
            if (destination / "prediction_matrices.npz").is_file():
                print(f"skip existing {destination / 'prediction_matrices.npz'}", flush=True)
                continue
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
