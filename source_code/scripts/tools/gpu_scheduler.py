from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class Task:
    dataset: str
    experiment: str
    seed: int


def run_tasks(tasks: list[Task], devices: list[int], project_root: Path,
              category: str, batch_name: str, output_root: Path,
              manifest: Path, extra_args: list[str] | None = None) -> None:
    if not devices:
        raise ValueError("At least one GPU is required")
    pending = deque(tasks)
    running: dict[int, tuple[subprocess.Popen, Task]] = {}
    failures = []
    extra_args = extra_args or []
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            task = pending.popleft()
            script_dir = "MPII" if task.dataset == "mpii" else "NTU_RGBD"
            command = [
                sys.executable, str(project_root / "scripts" / script_dir / "train.py"),
                "--experiment", task.experiment,
                "--category", category,
                "--batch-name", batch_name,
                "--seed", str(task.seed),
                "--device", "cuda:0",
                "--physical-gpu", str(device),
                "--output-root", str(output_root),
                "--manifest", str(manifest),
                *extra_args,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            print(f"GPU {device}: {task.dataset} {task.experiment} seed={task.seed}", flush=True)
            running[device] = (subprocess.Popen(command, env=environment), task)
        time.sleep(1)
        for device, (process, task) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            del running[device]
            if result != 0:
                failures.append((device, task, result))
    if failures:
        details = ", ".join(
            f"GPU {device} {task.experiment} seed={task.seed} exit={result}"
            for device, task, result in failures
        )
        raise RuntimeError(f"Ablation tasks failed: {details}")
