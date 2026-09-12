from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New" / "ntu_rgbd"
LOG_ROOT = PROJECT_ROOT / "launch_logs"


@dataclass(frozen=True)
class Run:
    display_id: str
    experiment: str
    category: str
    gpu: int
    gap: int = 1

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / self.category / "clip4_256_20ep" / self.experiment / "seed_42"


CF_BATCH = (
    Run("CF5-Motion", "t5", "cross_frame", 0),
    Run("CF6-Repeat4", "cf6", "cross_frame", 1),
    Run("CF7-Reverse", "cf7", "cross_frame", 2),
    Run("CF8-Shuffle", "cf8", "cross_frame", 3),
)
F_BATCH = tuple(
    Run(f"F{index}", f"f{index}", "frame_count", index - 1)
    for index in range(1, 5)
)
G_BATCH = (
    Run("G2", "g2", "frame_gap", 0, 2),
    Run("G3", "g3", "frame_gap", 1, 3),
    Run("G5", "g5", "frame_gap", 2, 5),
    Run("R2", "e0", "frame_gap_control", 3, 1),
)
QUEUE = (CF_BATCH, F_BATCH, G_BATCH)


def status(run: Run) -> str | None:
    path = run.run_dir / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def build_command(run: Run) -> list[str]:
    command = [
        str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", run.experiment,
        "--category", run.category,
        "--batch-name", "clip4_256_20ep",
        "--seed", "42", "--epochs", "20",
        "--device", f"cuda:{run.gpu}", "--physical-gpu", str(run.gpu),
        "--extracted-frames-dir", "Datasets/NTU_RGBD/frames/S010/clip4_256",
        "--frame-stride", "1", "--temporal-frame-gap", str(run.gap),
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation", "--skip-visualization",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    ]
    return command


def launch(batch: tuple[Run, ...]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for run in batch:
        if run.run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {run.run_dir}")
    for run in batch:
        log_path = LOG_ROOT / f"ntu_{run.experiment}_20ep.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                build_command(run), cwd=PROJECT_ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(
            f"launched {run.display_id} gpu={run.gpu} pid={process.pid} log={log_path}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue fixed 20-epoch NTU four-GPU batches")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    for number, batch in enumerate(QUEUE, 1):
        for run in batch:
            print(
                f"batch={number}\t{run.display_id}\tgpu={run.gpu}\t"
                f"status={status(run) or 'pending'}\toutput={run.run_dir.relative_to(PROJECT_ROOT)}"
            )
            if args.dry_run and number > 1:
                print(shlex.join(build_command(run)))
    if args.list or args.dry_run:
        return
    # Batch 1 is launched separately. This watcher only appends later batches.
    for previous, pending in zip(QUEUE, QUEUE[1:], strict=False):
        while not all(status(run) == "completed" for run in previous):
            failed = [run.display_id for run in previous if status(run) == "failed"]
            if failed:
                raise RuntimeError(f"Previous batch failed: {failed}")
            time.sleep(args.poll_seconds)
        if all(status(run) == "completed" for run in pending):
            continue
        launch(pending)
    print("all queued NTU batches completed", flush=True)


if __name__ == "__main__":
    main()
