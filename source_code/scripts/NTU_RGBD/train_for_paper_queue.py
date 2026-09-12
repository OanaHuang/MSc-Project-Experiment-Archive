from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.train_for_paper_series import (
    EXPERIMENTS, build_command,
)


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New/ntu_rgbd/for_paper"
PAPER_SEEDS = (42, 2071461405, 365198782)


@dataclass(frozen=True)
class Job:
    phase: str
    experiment: str
    seed: int
    epochs: int

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / f"clip4_256_{self.epochs}ep" / self.experiment / f"seed_{self.seed}"


def jobs_for_phase(phase: str) -> tuple[Job, ...]:
    experiment_ids = tuple(item.experiment for item in EXPERIMENTS)
    if phase == "screen":
        return tuple(Job(phase, experiment, 42, 20) for experiment in experiment_ids)
    if phase == "main":
        return tuple(
            Job(phase, experiment, seed, 120)
            for experiment in experiment_ids for seed in PAPER_SEEDS
        )
    if phase == "extension":
        return tuple(Job(phase, experiment, 42, 140) for experiment in experiment_ids)
    raise ValueError(f"Unknown phase: {phase}")


def status(job: Job) -> str:
    path = job.run_dir / "status.json"
    if not path.is_file():
        return "pending"
    return json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")


def command_for_job(job: Job, gpu: int, python_executable: str) -> list[str]:
    return build_command(
        job.experiment, python_executable=python_executable,
        device=f"cuda:{gpu}", physical_gpu=gpu, seed=job.seed,
        epochs=job.epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the publication FP-series queue")
    parser.add_argument("--phase", required=True, choices=("screen", "main", "extension"))
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--experiments", nargs="+", choices=tuple(
        item.experiment for item in EXPERIMENTS
    ))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("--gpus must contain distinct non-negative GPU indices")

    jobs = jobs_for_phase(args.phase)
    if args.experiments:
        selected = set(args.experiments)
        jobs = tuple(job for job in jobs if job.experiment in selected)
    runnable = [job for job in jobs if status(job) == "pending"]
    for index, job in enumerate(jobs):
        gpu = args.gpus[index % len(args.gpus)]
        print(
            f"{job.phase}\t{job.experiment}\tseed={job.seed}\tepochs={job.epochs}"
            f"\tgpu={gpu}\tstatus={status(job)}"
        )
        if args.dry_run:
            print(shlex.join(command_for_job(job, gpu, args.python)))
    if args.list or args.dry_run:
        return

    collisions = [str(job.run_dir) for job in jobs if job.run_dir.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite phase runs: " + ", ".join(collisions))
    processes: dict[int, subprocess.Popen] = {}
    for index, job in enumerate(runnable):
        gpu = args.gpus[index % len(args.gpus)]
        previous = processes.get(gpu)
        if previous is not None and previous.wait() != 0:
            raise subprocess.CalledProcessError(previous.returncode, previous.args)
        command = command_for_job(job, gpu, args.python)
        print(f"launching {job.experiment} seed={job.seed} epochs={job.epochs} gpu={gpu}")
        processes[gpu] = subprocess.Popen(command, cwd=PROJECT_ROOT)
    failures = []
    for gpu, process in processes.items():
        returncode = process.wait()
        if returncode:
            failures.append((gpu, returncode))
    if failures:
        raise RuntimeError(f"FP queue failures: {failures}")


if __name__ == "__main__":
    main()
