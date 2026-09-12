from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
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
    batch_name: str
    gpu: int
    frame_source: str
    gap: int
    minimum_history: int

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / self.category / self.batch_name / self.experiment / "seed_42"


G_FIXED_BATCH = tuple(
    Run(f"G{gap}", f"g{gap}", "frame_gap", "clip6_256_20ep", gpu,
        "Datasets/NTU_RGBD/frames/S010/clip6_256", gap, 5)
    for gpu, gap in enumerate((1, 2, 3, 5))
)
CF_ALIGN_BATCH = tuple(
    Run(f"CF{number}", f"cf{number}", "cross_frame", "clip4_256_20ep", gpu,
        "Datasets/NTU_RGBD/frames/S010/clip4_256", 1, 3)
    for gpu, number in enumerate(range(9, 13))
)
CF_FINAL_BATCH = (
    Run("CF13", "cf13", "cross_frame", "clip4_256_20ep", 0,
        "Datasets/NTU_RGBD/frames/S010/clip4_256", 1, 3),
)
QUEUE = (G_FIXED_BATCH, CF_ALIGN_BATCH, CF_FINAL_BATCH)


def status(run: Run) -> str | None:
    path = run.run_dir / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def build_command(run: Run) -> list[str]:
    return [
        str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", run.experiment,
        "--category", run.category,
        "--batch-name", run.batch_name,
        "--seed", "42", "--epochs", "20",
        "--device", f"cuda:{run.gpu}", "--physical-gpu", str(run.gpu),
        "--extracted-frames-dir", run.frame_source,
        "--frame-stride", "1", "--temporal-frame-gap", str(run.gap),
        "--minimum-temporal-history", str(run.minimum_history),
        "--preprocessed-pose-cache", "--disable-augmentation", "--skip-visualization",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    ]


def launch_batch(batch: tuple[Run, ...]) -> dict[str, subprocess.Popen]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    collisions = [str(run.run_dir) for run in batch if run.run_dir.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite runs: " + ", ".join(collisions))
    processes = {}
    for run in batch:
        log_path = LOG_ROOT / f"ntu_{run.experiment}_{run.batch_name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                build_command(run), cwd=PROJECT_ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(
            f"launched {run.display_id} gpu={run.gpu} pid={process.pid} log={log_path}",
            flush=True,
        )
        processes[run.display_id] = process
    return processes


def wait_for_batch(batch: tuple[Run, ...], processes: dict[str, subprocess.Popen],
                   poll_seconds: int) -> None:
    while True:
        # Popen.poll() calls waitpid(WNOHANG), so completed children are reaped
        # instead of accumulating as zombies while this queue remains alive.
        exit_codes = {name: process.poll() for name, process in processes.items()}
        crashed = {name: code for name, code in exit_codes.items()
                   if code is not None and code != 0}
        if crashed:
            raise RuntimeError(f"Training processes exited unsuccessfully: {crashed}")
        states = {run.display_id: status(run) for run in batch}
        if all(value == "completed" for value in states.values()):
            for process in processes.values():
                process.wait(timeout=30)
            return
        failed = [name for name, value in states.items() if value == "failed"]
        if failed:
            raise RuntimeError(f"Training batch failed: {failed}")
        print("waiting " + " ".join(f"{key}={value or 'starting'}" for key, value in states.items()),
              flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run corrected G1/G2/G3/G5 and CF9-CF13 as a four-GPU queue",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    for batch_number, batch in enumerate(QUEUE, 1):
        for run in batch:
            print(
                f"batch={batch_number}\t{run.display_id}\tgpu={run.gpu}\t"
                f"status={status(run) or 'pending'}\toutput={run.run_dir.relative_to(PROJECT_ROOT)}"
            )
            if args.dry_run:
                print(shlex.join(build_command(run)))
    if args.list or args.dry_run:
        return
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Project virtualenv Python not found: {PYTHON}")
    clip6_manifest = PROJECT_ROOT / "Datasets/NTU_RGBD/frames/S010/clip6_256/manifest.json"
    if not clip6_manifest.is_file():
        raise FileNotFoundError(f"Prepare the corrected G frame source first: {clip6_manifest}")
    for batch in QUEUE:
        pending = tuple(run for run in batch if status(run) != "completed")
        if not pending:
            print(
                "skipping completed batch "
                + " ".join(run.display_id for run in batch),
                flush=True,
            )
            continue
        processes = launch_batch(pending)
        wait_for_batch(pending, processes, args.poll_seconds)
    print("corrected G series and CF9-CF13 completed", flush=True)


if __name__ == "__main__":
    main()
