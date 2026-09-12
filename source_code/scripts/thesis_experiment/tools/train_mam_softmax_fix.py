#!/usr/bin/env python3
"""Run one frozen-spatial repair experiment and its matched validation diagnostics."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from spikepose_thesis.artifacts.manager import locate_source_run
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.evaluation.softmax_fix import sha256
from spikepose_thesis.training.runner import train_experiment

from diagnose_mam_softmax_fix import SOURCE, SOURCE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    if config["stage"] != "mam_softmax_fix":
        raise ValueError("This worker only runs mam_softmax_fix experiments")
    if sha256(locate_source_run(SOURCE, 42) / "checkpoints/best.pt") != SOURCE_SHA256:
        raise RuntimeError("Frame source checkpoint fingerprint mismatch")
    for phase in config["training"]["phases"]:
        if not phase["freeze_batch_norm"] or any(not name.startswith("mam.") for name in phase["train_modules"]):
            raise ValueError("Frozen study must not train spatial parameters or BN statistics")
    if not (args.diagnosis / "complete.json").is_file():
        raise RuntimeError("Fixed-heatmap screening must complete before training")
    run = locate_source_run(args.experiment, 42)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "softmax_fix_worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        if status.get("state") != "completed":
            train_experiment(args.experiment, 42, "cuda:0", resume=(run / "checkpoints/last.pt").exists())
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("diagnose_mam_softmax_fix.py")), "evaluate",
            "--experiment", args.experiment, "--diagnosis", str(args.diagnosis), "--output", str(args.output),
        ], check=True)


if __name__ == "__main__":
    main()
