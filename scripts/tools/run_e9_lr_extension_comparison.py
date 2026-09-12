from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.artifacts import output_run_dir


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
CATEGORY = "training_extension"
EXPERIMENT = "resformer_fpn"
SEED = 42
MILESTONES = (140, 160, 180, 210)
SOURCE_RUN = output_run_dir(
    OUTPUT_ROOT, "mpii", "backbone_fpn", "mpii_bf_20ep", EXPERIMENT, SEED,
)
COMPARISONS = (
    {
        "label": "constant_1e6",
        "gpu": 0,
        "resume": True,
        "extra": (
            "--learning-rate", "0.000001",
            "--scheduler-kind", "constant",
        ),
    },
    {
        "label": "cosine_restart_1e5",
        "gpu": 1,
        "resume": True,
        "extra": (
            "--learning-rate", "0.00001",
            "--scheduler-kind", "cosine_restart",
            "--scheduler-start-epoch", "121",
            "--scheduler-base-lr", "0.00001",
            "--scheduler-min-lr", "0.000001",
        ),
    },
    {
        "label": "scratch_cosine_210",
        "gpu": 2,
        "resume": False,
        "extra": (),
    },
)


def target_run(label: str) -> Path:
    return output_run_dir(
        OUTPUT_ROOT, "mpii", CATEGORY, label, EXPERIMENT, SEED,
    )


def prepare_resume_target(label: str) -> None:
    target = target_run(label)
    if target.exists():
        raise FileExistsError(
            f"Refusing to reuse existing comparison directory: {target}")
    (target / "checkpoints").mkdir(parents=True)
    for checkpoint in ("last.pt", "best.pt"):
        shutil.copy2(SOURCE_RUN / "checkpoints" / checkpoint,
                     target / "checkpoints" / checkpoint)


def command_for(item: dict) -> tuple[list[str], dict]:
    label = item["label"]
    gpu = item["gpu"]
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "MPII" / "train.py"),
        "--experiment", EXPERIMENT,
        "--category", CATEGORY,
        "--batch-name", label,
        "--seed", str(SEED),
        "--device", "cuda:0",
        "--physical-gpu", str(gpu),
        "--output-root", str(OUTPUT_ROOT),
        "--manifest", str(SOURCE_RUN.parents[1] / "sample_manifest.json"),
        "--epochs", "210",
        "--checkpoint-epochs", *(str(epoch) for epoch in MILESTONES),
        "--skip-visualization",
        *item["extra"],
    ]
    if item["resume"]:
        command.append("--resume")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, environment


def evaluate_milestones(item: dict) -> None:
    run = target_run(item["label"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
    for epoch in MILESTONES:
        for flip in (False, True):
            output = run / "milestones" / f"epoch_{epoch}" / (
                "flip_test" if flip else "no_flip")
            command = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "MPII" / "evaluate.py"),
                "--run", str(run),
                "--device", "cuda:0",
                "--checkpoint", f"checkpoints/epoch_{epoch}.pt",
                "--output-dir", str(output),
            ]
            if flip:
                command.append("--flip-test")
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    if not (SOURCE_RUN / "checkpoints" / "last.pt").is_file():
        raise FileNotFoundError(f"Missing source E9 checkpoint: {SOURCE_RUN}")
    for item in COMPARISONS:
        if item["resume"]:
            prepare_resume_target(item["label"])
        elif target_run(item["label"]).exists():
            raise FileExistsError(
                f"Refusing to reuse existing comparison directory: {target_run(item['label'])}")

    processes = []
    for item in COMPARISONS:
        command, environment = command_for(item)
        run = target_run(item["label"])
        (run / "logs").mkdir(parents=True, exist_ok=True)
        log_handle = (run / "logs" / "extension_to_210.log").open(
            "a", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        processes.append((item, process, log_handle))
        print(f"{item['label']}: GPU {item['gpu']}, pid={process.pid}", flush=True)

    failures = []
    for item, process, log_handle in processes:
        result = process.wait()
        log_handle.close()
        if result:
            failures.append(f"{item['label']} exit={result}")
            continue
        evaluate_milestones(item)
    status = {
        "status": "failed" if failures else "completed",
        "source": str(SOURCE_RUN),
        "milestones": list(MILESTONES),
        "experiments": [item["label"] for item in COMPARISONS],
        "failures": failures,
    }
    status_path = target_run(COMPARISONS[0]["label"]).parents[2] / "status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
