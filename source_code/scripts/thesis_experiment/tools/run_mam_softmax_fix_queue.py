#!/usr/bin/env python3
"""Validation screening -> four frozen MAM jobs -> winner's two alignment controls.

No test evaluation, backbone fine-tuning, or new architectures are launched.
GPU occupancy is checked before launch, with per-device cooperative locks.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.evaluation.softmax_fix import write_json

from run_icassp2027_queue import _gpu_occupancy

TOOLS = Path(__file__).resolve().parent


def run_jobs(jobs, args, state):
    pending = list(jobs)
    active = []
    failures = []
    try:
        while pending or active:
            remaining = []
            for process, log, lock, job in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, log, lock, job))
                    continue
                log.close()
                lock.close()
                state["history"].append({**job, "returncode": code, "finished_at": time.time()})
                if code:
                    failures.append(job)
            active = remaining
            if pending and not failures:
                discovered, busy = _gpu_occupancy()
                if set(args.gpus) - discovered:
                    raise ValueError("Requested GPU does not exist")
                reserved = {job["gpu"] for _, _, _, job in active}
                for gpu in args.gpus:
                    if not pending or gpu in busy or gpu in reserved:
                        continue
                    lock = (args.output.parent / f"mam_softmax_fix_gpu{gpu}.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    job = pending.pop(0)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
                        OPENCV_FOR_THREADS_NUM="1", PYTHONUNBUFFERED="1")
                    log_path = args.output / f"{job['name']}.log"
                    log = log_path.open("a")
                    process = subprocess.Popen(job["command"], cwd=PROJECT_ROOT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    record = {**job, "gpu": gpu, "pid": process.pid, "log": str(log_path), "started_at": time.time()}
                    active.append((process, log, lock, record))
                    print(f"START {job['name']} gpu={gpu} pid={process.pid}", flush=True)
            state.update(active=[job for _, _, _, job in active], pending=[job["name"] for job in pending],
                         state="draining_after_failure" if failures else "running", updated_at=time.time())
            write_json(args.output / "queue_status.json", state)
            if failures and not active:
                raise RuntimeError("Queue stopped after failed jobs: " + ", ".join(job["name"] for job in failures))
            if pending or active:
                time.sleep(args.poll_seconds)
    finally:
        # Do not terminate unrelated jobs or silently launch more after failure.
        for _process, log, lock, _job in active:
            log.close()
            lock.close()


def training_job(label, args):
    name = "mam_softmax_fix_" + label
    return {"name": name, "command": [sys.executable, "-u", str(TOOLS / "train_mam_softmax_fix.py"),
            "--experiment", name, "--diagnosis", str(args.output / "diagnosis"),
            "--output", str(args.output / "evaluation" / name)]}


def select_ablation_candidate(output, candidates):
    summaries = {name: json.loads((output / "evaluation" / ("mam_softmax_fix_" + name) / "summary.json").read_text())["scores"]["trained"]
                 for name in ["softmax", *candidates]}
    # At most 0.2 percentage-point PCKHB loss against the retrained control.
    # Eligible candidates are ordered by acceleration, velocity, then PCKHB.
    eligible = [name for name in candidates if summaries[name]["pckhb"] >= summaries["softmax"]["pckhb"] - 0.002]
    if eligible:
        winner = min(eligible, key=lambda name: (summaries[name]["mpjacce"], summaries[name]["mpjve"], -summaries[name]["pckhb"], name))
    else:
        winner = min(candidates, key=lambda name: (-summaries[name]["pckhb"], summaries[name]["mpjacce"], name))
    return {"winner": winner, "eligible": eligible, "scores": summaries,
            "pck_tolerance": 0.002, "no_eligible_candidate": not bool(eligible),
            "purpose": "choose mechanism ablations, not declare repair success"}


def compare_results(output, labels, winner):
    """Paired video bootstrap; sampled clips remain separate in temporal metrics."""
    videos = {}
    for label in labels:
        summary = json.loads((output / "evaluation" / ("mam_softmax_fix_" + label) / "summary.json").read_text())
        grouped = {}
        for row in summary["per_clip"]:
            if row["mode"] != "trained":
                continue
            values = grouped.setdefault(row["video"], np.zeros(6))
            values += [row["pck_correct"], row["pck_valid"], row["velocity_sum"], row["velocity_n"], row["acceleration_sum"], row["acceleration_n"]]
        videos[label] = grouped
    order = sorted(videos["softmax"])
    if any(sorted(group) != order for group in videos.values()):
        raise RuntimeError("Paired comparisons require identical video identities")
    arrays = {label: np.stack([group[name] for name in order]) for label, group in videos.items()}
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(order), size=(2000, len(order)))
    comparisons = {}
    for reference in ("softmax", winner + "_noalign", winner + "_coarse"):
        for label in labels:
            if label == reference:
                continue
            result = {}
            for column, metric in ((0, "pckhb"), (2, "mpjve"), (4, "mpjacce")):
                a, b = arrays[label], arrays[reference]
                if np.any(a[:, column+1] <= 0) or np.any(b[:, column+1] <= 0):
                    raise RuntimeError("A video has no valid observations for paired bootstrap")
                differences = a[:, column] / a[:, column+1] - b[:, column] / b[:, column+1]
                low, high = np.percentile(differences[indices].mean(1), [2.5, 97.5])
                result[metric] = {"video_macro_difference": float(differences.mean()), "ci95": [float(low), float(high)]}
            comparisons[label + "_minus_" + reference] = result
    write_json(output / "paired_comparisons.json", {"split": "validation", "videos": len(order),
        "bootstrap_samples": 2000, "seed": 42, "comparisons": comparisons,
        "limitation": "single training seed; same validation subset also used for screening, not independent test evidence"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--videos-per-action", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not 0 < args.poll_seconds <= 60:
        parser.error("Unique GPUs and poll-seconds in (0,60] required")
    for label in ("softmax", "relu", "power2", "expm1", "dark"):
        for suffix in ("", "_noalign", "_coarse"):
            load_experiment("mam_softmax_fix_" + label + suffix)
    if args.dry_run:
        print(json.dumps({"diagnosis": "canonical validation 2x16; two videos/action",
            "training": "softmax + DARK + two screened differentiable decoders; seed42, 20 epochs, spatial frozen",
            "ablations": "validation-selected candidate: noalign and coarse-only, 20 epochs each",
            "test": False, "joint_finetune": False}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / "mam_softmax_fix_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "starting", "pid": os.getpid(), "gpus": args.gpus, "history": [], "started_at": time.time()}
        try:
            diagnosis = args.output / "diagnosis"
            if not (diagnosis / "complete.json").exists():
                state["phase"] = "fixed_heatmap_diagnosis"
                run_jobs([{"name": "diagnosis", "command": [sys.executable, "-u", str(TOOLS / "diagnose_mam_softmax_fix.py"),
                    "diagnose", "--output", str(diagnosis), "--videos-per-action", str(args.videos_per_action)]}], args, state)
            selection = json.loads((diagnosis / "selection.json").read_text())
            candidates = ["dark", *selection["selected"]]
            state.update(phase="frozen_spatial_training", selected=candidates)
            run_jobs([training_job(label, args) for label in ["softmax", *candidates]], args, state)
            chosen = select_ablation_candidate(args.output, candidates)
            write_json(args.output / "ablation_selection.json", chosen)
            state.update(phase="alignment_ablations", ablation_selection=chosen)
            run_jobs([training_job(chosen["winner"] + suffix, args) for suffix in ("_noalign", "_coarse")], args, state)
            compare_results(args.output, ["softmax", *candidates, chosen["winner"] + "_noalign", chosen["winner"] + "_coarse"], chosen["winner"])
            state.update(state="completed", phase="awaiting_result_review", finished_at=time.time(),
                         next="Review validation evidence before joint fine-tuning or any test evaluation")
        except Exception as error:
            state.update(state="failed", error=repr(error), finished_at=time.time())
            raise
        finally:
            write_json(args.output / "queue_status.json", state)


if __name__ == "__main__":
    main()
