from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New/ntu_rgbd"
LOG_ROOT = PROJECT_ROOT / "launch_logs"
DEPENDENCY = (
    OUTPUT_ROOT / "cross_frame/clip4_256_20ep/cf13/seed_42/status.json"
)


@dataclass(frozen=True)
class Run:
    display_id: str
    experiment: str
    gpu: int
    purpose: str

    @property
    def run_dir(self) -> Path:
        return (
            OUTPUT_ROOT / "video_hpe_solution/clip4_256_20ep"
            / self.experiment / "seed_42"
        )


RUNS = (
    Run("V1", "v1", 0, "DCPose-inspired residual correction"),
    Run("V2", "v2", 1, "TDMI-inspired temporal difference gate"),
    Run("V3", "v3", 2, "DSTA-inspired decoupled joint space-time fusion"),
    Run("V4", "v4", 3, "SmoothNet-inspired acceleration supervision"),
)


def read_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def build_command(run: Run) -> list[str]:
    return [
        str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/train.py"),
        "--experiment", run.experiment,
        "--category", "video_hpe_solution",
        "--batch-name", "clip4_256_20ep",
        "--seed", "42", "--epochs", "20",
        "--device", f"cuda:{run.gpu}", "--physical-gpu", str(run.gpu),
        "--extracted-frames-dir", "Datasets/NTU_RGBD/frames/S010/clip4_256",
        "--frame-stride", "1", "--temporal-frame-gap", "1",
        "--minimum-temporal-history", "3", "--preprocessed-pose-cache",
        "--disable-augmentation", "--skip-visualization",
        "--train-metadata", "Datasets/NTU_RGBD/metadata/s010/train_split.csv",
        "--validation-metadata", "Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    ]


def wait_for_dependency(poll_seconds: int) -> None:
    while read_status(DEPENDENCY) != "completed":
        print(
            f"waiting CF13={read_status(DEPENDENCY) or 'pending'} before V-series",
            flush=True,
        )
        time.sleep(poll_seconds)


def launch() -> None:
    collisions = [str(run.run_dir) for run in RUNS if run.run_dir.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite runs: " + ", ".join(collisions))
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        log_path = LOG_ROOT / f"ntu_{run.experiment}_video_hpe_20ep.log"
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
    parser = argparse.ArgumentParser(
        description="Queue four literature-guided NTU video-HPE solutions after CF13",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    for run in RUNS:
        print(
            f"{run.display_id}\t{run.experiment}\tgpu={run.gpu}\t"
            f"status={read_status(run.run_dir / 'status.json') or 'pending'}\t{run.purpose}"
        )
        if args.dry_run:
            print(shlex.join(build_command(run)))
    if args.list or args.dry_run:
        return
    wait_for_dependency(args.poll_seconds)
    launch()


if __name__ == "__main__":
    main()
