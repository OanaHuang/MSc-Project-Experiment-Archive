from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.artifacts import output_run_dir


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
MILESTONES = (140, 160, 180, 210)
RUNS = (
    ("E7", "resformer_s3", "main_route", "mpii_main_route_20ep", 0),
    ("E9", "resformer_fpn", "backbone_fpn", "mpii_bf_20ep", 1),
    ("E8", "resformer_s4", "main_route", "mpii_main_route_20ep", 2),
    ("E5", "mem_spikefpn_annhead", "main_route", "mpii_main_route_20ep", 3),
)


def run_dir(category: str, batch: str, experiment: str) -> Path:
    return output_run_dir(
        OUTPUT_ROOT, "mpii", category, batch, experiment, 42,
    )


def verify_inputs() -> None:
    problems = []
    for label, experiment, category, batch, _ in RUNS:
        path = run_dir(category, batch, experiment)
        if not (path / "checkpoints" / "last.pt").is_file():
            problems.append(f"{label}: missing {path / 'checkpoints' / 'last.pt'}")
    if problems:
        raise RuntimeError("; ".join(problems))


def training_command(experiment: str, category: str, batch: str, gpu: int) -> tuple[list[str], dict]:
    path = run_dir(category, batch, experiment)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "MPII" / "train.py"),
        "--experiment", experiment,
        "--category", category,
        "--batch-name", batch,
        "--seed", "42",
        "--device", "cuda:0",
        "--physical-gpu", str(gpu),
        "--output-root", str(OUTPUT_ROOT),
        "--manifest", str(path.parents[1] / "sample_manifest.json"),
        "--resume",
        "--epochs", "210",
        "--checkpoint-epochs", *(str(epoch) for epoch in MILESTONES),
        "--skip-visualization",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, environment


def evaluate_milestones(label: str, experiment: str, category: str, batch: str, gpu: int) -> None:
    path = run_dir(category, batch, experiment)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for epoch in MILESTONES:
        for flip in (False, True):
            output = path / "milestones" / f"epoch_{epoch}" / ("flip_test" if flip else "no_flip")
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "MPII" / "evaluate.py"),
                "--run", str(path),
                "--device", "cuda:0",
                "--checkpoint", f"checkpoints/epoch_{epoch}.pt",
                "--output-dir", str(output),
            ]
            if flip:
                command.append("--flip-test")
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        print(f"{label}: evaluated epoch {epoch} with and without flip", flush=True)


def main() -> None:
    verify_inputs()
    running = []
    for label, experiment, category, batch, gpu in RUNS:
        command, environment = training_command(experiment, category, batch, gpu)
        log_path = run_dir(category, batch, experiment) / "logs" / "continue_to_210.log"
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        running.append((label, experiment, category, batch, gpu, process, log_handle))
        print(f"{label}: training on GPU {gpu}, pid={process.pid}", flush=True)
    failures = []
    for label, experiment, category, batch, gpu, process, log_handle in running:
        result = process.wait()
        log_handle.close()
        if result:
            failures.append(f"{label} exit={result}")
            continue
        evaluate_milestones(label, experiment, category, batch, gpu)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
