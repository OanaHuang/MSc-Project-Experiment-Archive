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

from scripts.NTU_RGBD.train_motion_series import EXPERIMENTS, build_command


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New/ntu_rgbd/motion_trend"
MAIN_SEEDS = (42, 2071461405, 365198782)


@dataclass(frozen=True)
class Job:
    experiment: str
    seed: int
    epochs: int

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / f"clip4_256_{self.epochs}ep" / self.experiment / f"seed_{self.seed}"


def jobs_for_phase(phase: str) -> tuple[Job, ...]:
    ids = tuple(item.experiment for item in EXPERIMENTS)
    if phase == "screen":
        return tuple(Job(experiment, 42, 20) for experiment in ids)
    if phase == "main":
        return tuple(Job(experiment, seed, 120) for experiment in ids for seed in MAIN_SEEDS)
    raise ValueError(f"Unknown phase: {phase}")


def status(job: Job) -> str:
    path = job.run_dir / "status.json"
    if not path.is_file():
        return "pending"
    return json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NTU M-series GPU queue")
    parser.add_argument("--phase", required=True, choices=("screen", "main"))
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
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
    for index, job in enumerate(jobs):
        gpu = args.gpus[index % len(args.gpus)]
        print(
            f"{job.experiment}\tseed={job.seed}\tepochs={job.epochs}"
            f"\tgpu={gpu}\tstatus={status(job)}"
        )
        if args.dry_run:
            print(shlex.join(build_command(
                job.experiment, python_executable=args.python,
                device=f"cuda:{gpu}", physical_gpu=gpu,
                seed=job.seed, epochs=job.epochs,
            )))
    if args.list or args.dry_run:
        return
    runnable = [job for job in jobs if status(job) == "pending"]
    collisions = [str(job.run_dir) for job in runnable if job.run_dir.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite runs: " + ", ".join(collisions))
    processes: dict[int, subprocess.Popen] = {}
    for index, job in enumerate(runnable):
        gpu = args.gpus[index % len(args.gpus)]
        previous = processes.get(gpu)
        if previous is not None and previous.wait() != 0:
            raise subprocess.CalledProcessError(previous.returncode, previous.args)
        command = build_command(
            job.experiment, python_executable=args.python,
            device=f"cuda:{gpu}", physical_gpu=gpu,
            seed=job.seed, epochs=job.epochs,
        )
        print(f"launching {job.experiment} seed={job.seed} gpu={gpu}", flush=True)
        processes[gpu] = subprocess.Popen(command, cwd=PROJECT_ROOT)
    failures = []
    for gpu, process in processes.items():
        returncode = process.wait()
        if returncode:
            failures.append((gpu, returncode))
    if failures:
        raise RuntimeError(f"M-series queue failures: {failures}")


if __name__ == "__main__":
    main()
