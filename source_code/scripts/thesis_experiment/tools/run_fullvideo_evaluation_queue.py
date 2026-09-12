#!/usr/bin/env python3
"""Run the remaining frozen NTU paper evaluations across several GPUs.

Every job scores the same exhaustive test videos with one fixed tube crop per
video and DARK decoding. Frame-wise and frame/update-control models use the
general evaluator. The ANN+MAM checkpoint uses the reset-policy evaluator and
the paper result is its ``video_reset`` output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = Path(__file__).resolve().with_name("evaluate_fullvideo_subset.py")
MAM_EVALUATOR = Path(__file__).resolve().with_name("validate_mam_reset_policy.py")
RUN_ROOT = PROJECT_ROOT / "Outputs_Thesis_Pilot20" / "runs" / "ntu60_cs"


@dataclass(frozen=True)
class JobSpec:
    label: str
    experiment: str
    relative_run: str
    evaluator: str = "standard"


MODELS = (
    # Table 1 baselines not covered by the completed core-MAM validation.
    JobSpec("simplebaseline_r50", "pilot20_ntu_simplebaseline_r50", "paper_baseline/pilot20_ntu_simplebaseline_r50/seed_42"),
    JobSpec("hrnet_w32", "pilot20_ntu_hrnet_w32", "paper_baseline/pilot20_ntu_hrnet_w32/seed_42"),
    JobSpec("spikepose_ann", "pilot20_ntu_spikepose_ann", "paper_baseline/pilot20_ntu_spikepose_ann/seed_42"),
    JobSpec("spikepose_ann_mam", "pilot20_ntu_spikepose_ann_mam", "spikepose_ann_mam/pilot20_ntu_spikepose_ann_mam/seed_42", "mam_reset"),
    JobSpec("spikeyolo", "pilot20_ntu_spikeyolo", "paper_baseline/pilot20_ntu_spikeyolo/seed_42"),
    # Table 3 frame/update controls, evaluated on the same complete videos.
    JobSpec("v1u1", "pilot20_factor_v1u1", "paper_factor_control/pilot20_factor_v1u1/seed_42"),
    JobSpec("v1u2", "pilot20_factor_v1u2", "paper_factor_control/pilot20_factor_v1u2/seed_42"),
    JobSpec("v1u4", "pilot20_factor_v1u4", "paper_factor_control/pilot20_factor_v1u4/seed_42"),
    JobSpec("v2u2", "pilot20_factor_v2u2", "paper_factor_control/pilot20_factor_v2u2/seed_42"),
    JobSpec("v2u4", "pilot20_factor_v2u4", "paper_factor_control/pilot20_factor_v2u4/seed_42"),
    JobSpec("v4u4", "pilot20_factor_v4u4", "paper_factor_control/pilot20_factor_v4u4/seed_42"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-job", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--labels",
        help="Comma-separated model labels to run; defaults to the full queue",
    )
    return parser


def _select_models(labels: str | None) -> tuple[JobSpec, ...]:
    if not labels:
        return MODELS
    requested = [item.strip() for item in labels.split(",") if item.strip()]
    available = {spec.label: spec for spec in MODELS}
    unknown = [label for label in requested if label not in available]
    if unknown:
        raise ValueError(f"Unknown model labels: {', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("Model labels must not be repeated")
    return tuple(available[label] for label in requested)


def _write_status(
    path: Path, jobs: list[dict], started_at: str, models_total: int,
) -> None:
    payload = {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models_total": models_total,
        "jobs": [
            {key: value for key, value in job.items() if key not in {"process", "handle"}}
            for job in jobs
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_inputs(models: tuple[JobSpec, ...]) -> None:
    missing = []
    for spec in models:
        run_dir = RUN_ROOT / spec.relative_run
        for path in (run_dir / "resolved_config.yaml", run_dir / "checkpoints" / "best.pt"):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing frozen evaluation inputs:\n" + "\n".join(missing))


def main() -> None:
    args = _parser().parse_args()
    models = _select_models(args.labels)
    _validate_inputs(models)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    # Keep the virtual-environment launcher path intact. Resolving its symlink
    # would execute the base interpreter without the venv packages.
    python = Path(sys.executable)
    started_at = datetime.now(timezone.utc).isoformat()
    jobs: list[dict] = []
    pending = list(models)
    running: dict[str, dict] = {}
    status_path = output_root / "queue_status.json"

    while pending or running:
        free_gpus = [gpu for gpu in gpus if gpu not in running]
        while pending and free_gpus:
            spec = pending.pop(0)
            gpu = free_gpus.pop(0)
            run_dir = RUN_ROOT / spec.relative_run
            config = run_dir / "resolved_config.yaml"
            checkpoint = run_dir / "checkpoints" / "best.pt"
            model_output = output_root / spec.label
            model_output.mkdir(parents=True, exist_ok=True)
            log_path = output_root / f"{spec.label}.log"
            evaluator = MAM_EVALUATOR if spec.evaluator == "mam_reset" else EVALUATOR
            command = [
                str(python), "-u", str(evaluator),
                "--experiment", spec.experiment,
                "--resolved-config", str(config),
                "--seed", "42",
                "--checkpoint", str(checkpoint),
                "--output", str(model_output),
                "--device", f"cuda:{gpu}",
                "--workers", str(args.workers_per_job),
            ]
            if args.max_videos is not None:
                command.extend(["--max-videos", str(args.max_videos)])
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=handle,
                stderr=subprocess.STDOUT, env=os.environ.copy(),
            )
            job = {
                "label": spec.label,
                "experiment": spec.experiment,
                "evaluator": spec.evaluator,
                "gpu": gpu,
                "pid": process.pid,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output": str(model_output),
                "log": str(log_path),
                "command": command,
                "process": process,
                "handle": handle,
            }
            jobs.append(job)
            running[gpu] = job
            print(
                f"started label={spec.label} experiment={spec.experiment} "
                f"gpu={gpu} pid={process.pid}",
                flush=True,
            )
        _write_status(status_path, jobs, started_at, len(models))
        if running:
            time.sleep(max(args.poll_seconds, 1))
        for gpu, job in list(running.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            job["returncode"] = int(returncode)
            job["status"] = "complete" if returncode == 0 else "failed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            del running[gpu]
            print(
                f"finished label={job['label']} gpu={gpu} returncode={returncode}",
                flush=True,
            )

    _write_status(status_path, jobs, started_at, len(models))
    failed = [job for job in jobs if job["status"] != "complete"]
    final = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "models_total": len(jobs),
        "models_complete": len(jobs) - len(failed),
        "models_failed": len(failed),
        "failed_labels": [job["label"] for job in failed],
        "paper_result_paths": {
            job["label"]: (
                str(Path(job["output"]) / "video_reset" / "summary.json")
                if job["evaluator"] == "mam_reset"
                else str(Path(job["output"]) / "summary.json")
            )
            for job in jobs if job["status"] == "complete"
        },
    }
    (output_root / "queue_complete.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8",
    )
    print(json.dumps(final, indent=2), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
