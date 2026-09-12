#!/usr/bin/env python3
"""Run the two native NTU-25 warm-start preview jobs in parallel."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from spikepose_thesis.data import audit_data
from spikepose_thesis.experiments import experiment_readiness, plan_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("launch_logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _command(job: dict, output_root: Path | None) -> list[str]:
    command = [
        sys.executable, "-m", "spikepose_thesis", "train",
        "--experiment", job["experiment"],
        "--seed", str(job["seed"]),
        "--device", "cuda:0",
    ]
    if output_root is not None:
        command.extend(("--output-root", str(output_root)))
    if job["readiness"] == "RESUMABLE":
        command.append("--resume")
    return command


def main() -> None:
    args = _parser().parse_args()
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must contain unique physical GPU indices")
    report = audit_data()
    if not report["ready"]:
        raise RuntimeError("Dataset audit is not ready")
    jobs = []
    completed = []
    for config in plan_study("ntu25_preview8"):
        seed = int(config["training"]["seeds"][0])
        readiness = experiment_readiness(
            config, seed, output_root=args.output_root, report=report,
        )
        record = {
            "experiment": config["id"],
            "seed": seed,
            "source": config["initialization"]["source"],
            "epochs": int(config["training"]["epochs"]),
            "readiness": readiness["status"],
            "blockers": readiness["blockers"],
        }
        if readiness["status"] == "COMPLETE":
            completed.append(record)
        elif readiness["status"] in {"READY", "RESUMABLE"}:
            jobs.append(record)
        else:
            raise RuntimeError(
                f"{config['id']} is {readiness['status']}: {readiness['blockers']}"
            )
    plan = {"jobs": jobs, "completed": completed, "gpus": args.gpus}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run or not jobs:
        return
    if len(args.gpus) < len(jobs):
        raise ValueError(f"Need {len(jobs)} GPUs to run both previews in parallel")

    args.log_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.log_root / "ntu25_preview8.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another NTU-25 preview queue is active") from error
        run_root = args.log_root / time.strftime("ntu25_preview8_%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True)
        status_path = run_root / "queue_status.json"
        status = {
            "state": "running", "started_at": time.time(), "plan": plan,
            "active": [], "history": [],
        }
        processes = []
        for gpu, job in zip(args.gpus, jobs):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = run_root / f"{job['experiment']}_seed_{job['seed']}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                _command(job, args.output_root), stdout=handle,
                stderr=subprocess.STDOUT, env=environment,
            )
            record = {
                **job, "gpu": gpu, "pid": process.pid,
                "log": str(log_path), "started_at": time.time(),
            }
            status["active"].append(record)
            status["history"].append({**record, "state": "started"})
            processes.append((process, handle, record))
        _write_json(status_path, status)

        failures = []
        for process, handle, record in processes:
            return_code = process.wait()
            handle.close()
            status["active"] = [
                item for item in status["active"] if item["pid"] != process.pid
            ]
            finished = {
                **record, "return_code": return_code,
                "finished_at": time.time(),
                "state": "completed" if return_code == 0 else "failed",
            }
            status["history"].append(finished)
            if return_code:
                failures.append(finished)
            _write_json(status_path, status)
        status["state"] = "failed" if failures else "completed"
        status["finished_at"] = time.time()
        status["failures"] = failures
        _write_json(status_path, status)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
