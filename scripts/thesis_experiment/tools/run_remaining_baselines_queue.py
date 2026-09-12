#!/usr/bin/env python3
"""Two cooperating GPU workers; one job per GPU, no duplicate launches.

Use the same --queue-dir for workers on physical GPUs 2 and 3. A failure stops
that worker without starting another job. A killed worker remains claimed and
must be inspected before its queue state is manually reset. Existing experiment
checkpoints are resumed, never overwritten by a new scratch launch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from spikepose_thesis.artifacts import run_dir
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.data import audit_data
from spikepose_thesis.experiments import experiment_readiness


EXPERIMENTS = [
    "mpii_official_spikepose_u2",
    "pilot20_mamv2_nomotiontoken",
    "pilot20_ntu_spikepose_ann",
    "mpii_official_spikeyolo",
    "mpii_official_spikformer",
    "pilot20_mamv2_nooffsethead",
    "pilot20_mamv2_nomemoryupdate",
    "pilot20_ntu_spikeyolo",
    "pilot20_ntu_spikformer",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--first", choices=EXPERIMENTS)
    args = parser.parse_args()
    directory = args.queue_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "queue.json"
    lock = (directory / "queue.lock").open("a")
    report = audit_data()
    configs = {name: load_experiment(name) for name in EXPERIMENTS}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH="src")
    cwd = PROJECT_ROOT / "scripts/thesis_experiment"
    first = args.first
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()

    def read_state():
        return json.loads(state_path.read_text()) if state_path.exists() else {}

    def write_state(state):
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    while True:
        selected = None
        with_lock_order = ([first] if first else []) + [n for n in EXPERIMENTS if n != first]
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = read_state()
            for name in with_lock_order:
                if name in state:
                    continue
                readiness = experiment_readiness(configs[name], 42, report=report)
                if readiness["status"] in {"READY", "RESUMABLE", "COMPLETE"}:
                    selected = (name, readiness["status"])
                    state[name] = {"state": "running", "gpu": args.gpu,
                                   "pid": os.getpid(), "code_commit": code_commit,
                                   "started": datetime.now(timezone.utc).isoformat()}
                    write_state(state)
                    break
            finished = len(state) == len(EXPERIMENTS) and all(
                job["state"] in {"completed", "failed"} for job in state.values()
            )
            stranded = (selected is None and not finished
                        and not any(job["state"] == "running" for job in state.values()))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if selected is None:
            if finished:
                return
            if stranded:
                raise RuntimeError("Pending experiments are blocked with no active source job; inspect readiness/queue.json")
            # Pending source dependencies are expected. Do not force readiness.
            time.sleep(30)
            continue
        first = None
        name, status = selected
        config = configs[name]
        run = run_dir(config, 42)
        base = [sys.executable, "-u", "-m", "spikepose_thesis.cli"]
        common = ["--experiment", name, "--seed", "42", "--device", "cuda:0"]
        commands = []
        if status != "COMPLETE":
            commands.append(base + ["train"] + common + (["--resume"] if status == "RESUMABLE" else []))
        if config["dataset"] == "mpii":
            commands.append(base + ["evaluate"] + common + ["--split", "validation"])
        # NTU full-video test evaluation is a separate explicit step after the
        # best checkpoint is fixed; training validation stays in its usual path.
        try:
            for command in commands:
                action = command[4]
                logfile = directory / f"{name}_{action}.log"
                print(f"GPU{args.gpu} START {name}: {' '.join(command)}", flush=True)
                with logfile.open("a") as output:
                    subprocess.run(command, cwd=cwd, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT, check=True)
            result = json.loads((run / "status.json").read_text())
            if result.get("state") != "completed":
                raise RuntimeError(f"{name} exited without completed status")
            final = "completed"
        except BaseException:
            final = "failed"
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = read_state()
                state[name].update(state=final, finished=datetime.now(timezone.utc).isoformat())
                write_state(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        print(f"GPU{args.gpu} DONE {name}", flush=True)


if __name__ == "__main__":
    main()
