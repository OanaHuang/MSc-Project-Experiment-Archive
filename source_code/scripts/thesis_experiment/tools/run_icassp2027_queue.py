#!/usr/bin/env python3
"""Run the final ICASSP Pilot20/Confirm140 workflow in dependency waves.

The queue never promotes a Pilot checkpoint into Confirm140.  Existing runs
are skipped, interrupted training runs are resumed, and each wave must finish
before jobs that consume its checkpoints are reconsidered.
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

from spikepose_thesis.core.config import (
    apply_pilot_runtime_overrides,
    load_experiment,
)
from spikepose_thesis.data import audit_data
from spikepose_thesis.experiments import experiment_readiness


PAPER_SEEDS = (42, 365198782, 2071461405)


def _job(experiment: str, *, profile: str | None = None,
         full_ntu_runtime: bool = False,
         source_override: str | None = None,
         seeds: tuple[int, ...] | None = None) -> dict:
    return {
        "experiment": experiment,
        "profile": profile,
        "full_ntu_runtime": full_ntu_runtime,
        "source_override": source_override,
        "seeds": seeds,
    }


WORKFLOWS = {
    "spikepose-ann-mam-pilot20": (
        # Keep the stages in separate waves: every target consumes the exact
        # best checkpoint produced by the preceding stage for the same seed.
        ("mpii_ann", (
            _job("mpii_official_spikepose_ann", seeds=(42,)),
        )),
        ("ntu_ann", (
            _job("pilot20_ntu_spikepose_ann"),
        )),
        ("ann_mam", (
            _job("pilot20_ntu_spikepose_ann_mam"),
        )),
    ),
    "pilot20": (
        ("mpii", (
            _job("pilot20_m_ann_s3", profile="pilot20"),
            _job("pilot20_m_s3_u1", profile="pilot20"),
            _job("pilot20_m_s3_u2", profile="pilot20"),
        )),
        ("frame_and_baselines", (
            _job("mamv2_fullcs_p00_source", profile="pilot20"),
            _job("pilot20_ntu_simplebaseline_r50", profile="pilot20"),
            _job("pilot20_ntu_hrnet_w32", profile="pilot20"),
        )),
        ("factor_controls", (
            _job("pilot20_factor_v1u1", profile="pilot20"),
            _job("pilot20_factor_v1u2", profile="pilot20"),
            _job("pilot20_factor_v1u4", profile="pilot20"),
            _job("pilot20_factor_v2u2", profile="pilot20"),
            _job("pilot20_factor_v2u4", profile="pilot20"),
            _job("pilot20_factor_v4u4", profile="pilot20"),
        )),
        ("mam_and_ablations", (
            _job("mamv2_fullcs20", profile="pilot20"),
            # These two completed Pilot runs already live under the historical
            # setups_full variant.  Preserve that identity and only redirect
            # their spatial source to the matching Full-CS source.
            _job(
                "mamv2_p06_noalign", profile="pilot20",
                full_ntu_runtime=True,
                source_override="mamv2_fullcs_p00_source",
            ),
            _job(
                "mamv2_p07_fixed_decay", profile="pilot20",
                full_ntu_runtime=True,
                source_override="mamv2_fullcs_p00_source",
            ),
            _job("pilot20_mamv2_noresidual", profile="pilot20"),
        )),
    ),
    "confirm140": (
        ("mpii", (
            _job("confirm140_m_ann_s3"),
            _job("confirm140_m_s3_u1"),
            _job("confirm140_m_s3_u2"),
        )),
        ("frame_and_baselines", (
            _job("confirm140_ntu_spikepose_frame"),
            _job("confirm140_ntu_simplebaseline_r50"),
            _job("confirm140_ntu_hrnet_w32"),
        )),
        ("factor_controls", (
            _job("confirm140_factor_v1u1"),
            _job("confirm140_factor_v1u2"),
            _job("confirm140_factor_v1u4"),
            _job("confirm140_factor_v2u2"),
            _job("confirm140_factor_v2u4"),
            _job("confirm140_factor_v4u4"),
        )),
        ("mam_and_ablations", (
            _job("confirm140_ntu_mamv2"),
            _job("confirm140_ntu_mamv2_noalign"),
            _job("confirm140_ntu_mamv2_fixedleak"),
            _job("confirm140_ntu_mamv2_noresidual"),
        )),
    ),
    "pilot-eval": (
        ("controls_and_filters", (
            _job("pilot20_control_v1u1", profile="pilot20"),
            _job("pilot20_control_v1u2", profile="pilot20"),
            _job("pilot20_control_v2u2", profile="pilot20"),
            _job("pilot20_eval_ema_shared", profile="pilot20"),
            _job("pilot20_refine_jointwise", profile="pilot20"),
            _job("pilot20_eval_sg_causal", profile="pilot20"),
            _job("pilot20_eval_one_euro", profile="pilot20"),
        )),
    ),
    "confirm-eval": (
        ("controls_and_filters", (
            _job("confirm140_control_v1u1"),
            _job("confirm140_control_v1u2"),
            _job("confirm140_control_v2u2"),
            _job("confirm140_eval_ema_shared"),
            _job("confirm140_eval_sg_causal"),
            _job("confirm140_eval_one_euro"),
        )),
    ),
    "pilot20-factors": (
        ("factor_controls", (
            _job("pilot20_factor_v1u1", profile="pilot20"),
            _job("pilot20_factor_v1u2", profile="pilot20"),
            _job("pilot20_factor_v1u4", profile="pilot20"),
            _job("pilot20_factor_v2u2", profile="pilot20"),
            _job("pilot20_factor_v2u4", profile="pilot20"),
            _job("pilot20_factor_v4u4", profile="pilot20"),
        )),
    ),
    "confirm140-factors": (
        ("factor_controls", (
            _job("confirm140_factor_v1u1"),
            _job("confirm140_factor_v1u2"),
            _job("confirm140_factor_v1u4"),
            _job("confirm140_factor_v2u2"),
            _job("confirm140_factor_v2u4"),
            _job("confirm140_factor_v4u4"),
        )),
    ),
    "mam-components-pilot20": (
        ("mam_components", (
            _job("mamv2_fullcs20", profile="pilot20"),
            _job(
                "mamv2_p06_noalign", profile="pilot20",
                full_ntu_runtime=True,
                source_override="mamv2_fullcs_p00_source",
            ),
            _job(
                "mamv2_p07_fixed_decay", profile="pilot20",
                full_ntu_runtime=True,
                source_override="mamv2_fullcs_p00_source",
            ),
            _job("pilot20_mamv2_noresidual", profile="pilot20"),
            _job("pilot20_mamv2_nomotiontoken", profile="pilot20"),
            _job("pilot20_mamv2_nooffsethead", profile="pilot20"),
            _job("pilot20_mamv2_nomemoryupdate", profile="pilot20"),
        )),
    ),
    "mam-ktp-pilot20": (
        ("mam_ktp", (
            # The first cell is the already-trained KPA-off/TPA-off control.
            _job("mamv2_fullcs20", profile="pilot20"),
            _job("pilot20_mamv2_kpa", profile="pilot20"),
            _job("pilot20_mamv2_tpa", profile="pilot20"),
            _job("pilot20_mamv2_ktp", profile="pilot20"),
        )),
    ),
    "jointwise-ref-pilot20": (
        ("jointwise_refinement", (
            _job("pilot20_refine_jointwise", profile="pilot20"),
        )),
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final-paper Pilot20/Confirm140 experiment queue",
    )
    parser.add_argument("--phase", choices=tuple(WORKFLOWS), required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-root", type=Path, default=Path("launch_logs"))
    parser.add_argument(
        "--wait-for-free-gpus", action="store_true",
        help=(
            "Dynamically start jobs as requested GPUs become free instead of "
            "assuming every requested GPU is idle at queue startup"
        ),
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=30.0,
        help="GPU polling interval used with --wait-for-free-gpus",
    )
    return parser


def _write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _resolved(spec: dict) -> tuple[dict, dict]:
    config = load_experiment(spec["experiment"], profile=spec["profile"])
    if spec["full_ntu_runtime"]:
        config = apply_pilot_runtime_overrides(
            config, epochs=20, ntu_setups="full",
        )
    readiness_config = config
    if spec["source_override"]:
        readiness_config = {
            **config,
            "initialization": {
                **config.get("initialization", {}),
                "source": spec["source_override"],
            },
        }
    return config, readiness_config


def _command(job: dict) -> list[str]:
    action = "refine" if job["action"] == "refinement" else "train"
    command = [
        sys.executable, "-m", "spikepose_thesis", action,
        "--experiment", job["experiment"],
        "--seed", str(job["seed"]),
        "--device", "cuda:0",
    ]
    if job["profile"]:
        command.extend(("--profile", job["profile"]))
    if job["full_ntu_runtime"]:
        command.extend(("--epochs", "20", "--ntu-setups", "full"))
    if job["source_override"]:
        command.extend(("--source-experiment", job["source_override"]))
    if action == "train" and job["readiness"] == "RESUMABLE":
        command.append("--resume")
    return command


def _plan_wave(specs: tuple[dict, ...], report: dict) -> dict:
    result = {"runnable": [], "completed": [], "waiting": []}
    for spec in specs:
        config, readiness_config = _resolved(spec)
        selected_seeds = (
            spec["seeds"]
            if spec.get("seeds") is not None
            else config["training"]["seeds"]
        )
        unknown_seeds = set(selected_seeds) - set(config["training"]["seeds"])
        if unknown_seeds:
            raise ValueError(
                f"{spec['experiment']} queue selects unregistered seeds: "
                f"{sorted(unknown_seeds)}"
            )
        for seed in selected_seeds:
            readiness = experiment_readiness(
                readiness_config, seed, report=report,
            )
            record = {
                **spec,
                "seed": seed,
                "paper_id": config["paper_id"],
                "dataset": config["dataset"],
                "epochs": int(config["training"]["epochs"]),
                "action": config.get("action", "train"),
                "run_type": config.get("run_type", "formal"),
                "run_variant": config.get("run_variant"),
                "readiness": readiness["status"],
                "blockers": readiness["blockers"],
            }
            if readiness["status"] == "COMPLETE":
                result["completed"].append(record)
            elif readiness["status"] in {"READY", "RESUMABLE"}:
                result["runnable"].append(record)
            else:
                result["waiting"].append(record)
    return result


def build_plan(phase: str, report: dict) -> list[dict]:
    """Return a serializable plan without starting a process."""
    return [
        {"wave": wave, **_plan_wave(specs, report)}
        for wave, specs in WORKFLOWS[phase]
    ]


def _parse_gpu_occupancy(
    inventory_output: str, compute_output: str,
) -> tuple[set[int], set[int]]:
    """Return the discovered and compute-busy GPU indices from CSV output."""
    uuid_to_index = {}
    for line in inventory_output.splitlines():
        if not line.strip():
            continue
        index_text, uuid = (part.strip() for part in line.split(",", 1))
        uuid_to_index[uuid] = int(index_text)

    busy = set()
    unknown = set()
    for line in compute_output.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
        else:
            unknown.add(uuid)
    if unknown:
        raise RuntimeError(
            "nvidia-smi reported compute processes on unknown GPU UUIDs: "
            + ", ".join(sorted(unknown))
        )
    return set(uuid_to_index.values()), busy


def _gpu_occupancy() -> tuple[set[int], set[int]]:
    inventory = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    compute = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    return _parse_gpu_occupancy(inventory.stdout, compute.stdout)


def _start_job(
    wave: str, job: dict, gpu: int, run_dir: Path, state: dict,
) -> tuple[subprocess.Popen, object, dict]:
    label = f"{wave}_{job['experiment']}_seed_{job['seed']}_gpu{gpu}"
    log_path = run_dir / f"{label}.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["OPENCV_FOR_THREADS_NUM"] = "1"
    command = _command(job)
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT,
            env=environment,
        )
    except Exception:
        handle.close()
        raise
    record = {
        **job, "gpu": gpu, "pid": process.pid,
        "command": command, "log": str(log_path),
        "started_at": time.time(),
    }
    state["active"].append(record)
    state["history"].append({**record, "state": "started"})
    return process, handle, record


def _finish_job(
    process: subprocess.Popen, handle: object, record: dict, state: dict,
    return_code: int,
) -> dict:
    handle.close()
    state["active"] = [
        item for item in state["active"] if item["pid"] != process.pid
    ]
    final = {
        **record, "return_code": return_code,
        "finished_at": time.time(),
        "state": "completed" if return_code == 0 else "failed",
    }
    state["history"].append(final)
    return final


def _run_wave(wave: str, jobs: list[dict], gpus: list[int], run_dir: Path,
              status_path: Path, state: dict) -> None:
    state["wave"] = wave
    state["state"] = "running"
    _write_status(status_path, state)
    for offset in range(0, len(jobs), len(gpus)):
        batch = jobs[offset:offset + len(gpus)]
        processes = []
        for gpu, job in zip(gpus, batch):
            processes.append(_start_job(wave, job, gpu, run_dir, state))
        _write_status(status_path, state)

        failures = []
        for process, handle, record in processes:
            return_code = process.wait()
            final = _finish_job(
                process, handle, record, state, return_code,
            )
            if return_code:
                failures.append(final)
            _write_status(status_path, state)
        if failures:
            raise RuntimeError(
                f"Wave {wave} stopped after failures: "
                + ", ".join(
                    f"{item['experiment']} seed={item['seed']} "
                    f"rc={item['return_code']}"
                    for item in failures
                )
            )


def _run_wave_dynamic(
    wave: str, jobs: list[dict], gpus: list[int], run_dir: Path,
    status_path: Path, state: dict, poll_seconds: float,
) -> None:
    """Backfill jobs onto allowed GPUs whenever each GPU becomes free."""
    state["wave"] = wave
    state["state"] = "running"
    state["scheduler"] = "wait_for_free_gpus"
    state["poll_seconds"] = poll_seconds
    pending = list(jobs)
    processes = []
    failures = []

    while pending or processes:
        still_running = []
        for process, handle, record in processes:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, handle, record))
                continue
            final = _finish_job(
                process, handle, record, state, return_code,
            )
            if return_code:
                failures.append(final)
        processes = still_running

        if not failures and pending:
            discovered, busy = _gpu_occupancy()
            missing = set(gpus) - discovered
            if missing:
                raise RuntimeError(
                    "Requested GPU indices do not exist: "
                    + ", ".join(str(gpu) for gpu in sorted(missing))
                )
            reserved = {record["gpu"] for _, _, record in processes}
            available = [
                gpu for gpu in gpus if gpu not in busy and gpu not in reserved
            ]
            state["observed_busy_gpus"] = sorted(busy)
            state["available_gpus"] = available
            state["last_gpu_poll_at"] = time.time()
            while pending and available:
                gpu = available.pop(0)
                job = pending.pop(0)
                processes.append(
                    _start_job(wave, job, gpu, run_dir, state)
                )

        state["pending"] = [
            {
                "experiment": item["experiment"],
                "seed": item["seed"],
                "readiness": item["readiness"],
            }
            for item in pending
        ]
        if failures:
            state["state"] = "draining_after_failure"
        _write_status(status_path, state)

        if failures and not processes:
            raise RuntimeError(
                f"Wave {wave} stopped after failures: "
                + ", ".join(
                    f"{item['experiment']} seed={item['seed']} "
                    f"rc={item['return_code']}"
                    for item in failures
                )
            )
        if pending or processes:
            time.sleep(poll_seconds)


def main() -> None:
    args = _parser().parse_args()
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        raise ValueError("--gpus must contain unique GPU indices")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    args.log_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.log_root / f"icassp2027_{args.phase}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another {args.phase} queue already holds the lock"
            ) from error

        report = audit_data()
        if not report["ready"]:
            raise RuntimeError("Dataset audit is not ready")
        plan = build_plan(args.phase, report)
        print(json.dumps({"phase": args.phase, "waves": plan}, indent=2), flush=True)
        if args.dry_run:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_root / f"icassp2027_{args.phase}_{timestamp}"
        run_dir.mkdir(parents=True)
        status_path = run_dir / "queue_status.json"
        state = {
            "state": "starting", "phase": args.phase, "wave": None,
            "gpus": args.gpus, "started_at": time.time(),
            "active": [], "history": [], "initial_plan": plan,
        }
        _write_status(status_path, state)
        try:
            for wave, specs in WORKFLOWS[args.phase]:
                refreshed = audit_data()
                current = _plan_wave(specs, refreshed)
                if current["waiting"]:
                    reasons = "; ".join(
                        f"{item['experiment']} seed={item['seed']}: "
                        + ", ".join(item["blockers"])
                        for item in current["waiting"]
                    )
                    raise RuntimeError(f"Wave {wave} is blocked: {reasons}")
                if args.wait_for_free_gpus:
                    _run_wave_dynamic(
                        wave, current["runnable"], args.gpus, run_dir,
                        status_path, state, args.poll_seconds,
                    )
                else:
                    _run_wave(
                        wave, current["runnable"], args.gpus, run_dir,
                        status_path, state,
                    )
        except Exception as error:
            state["state"] = "failed"
            state["error"] = str(error)
            state["finished_at"] = time.time()
            _write_status(status_path, state)
            raise
        state["state"] = "completed"
        state["wave"] = "done"
        state["finished_at"] = time.time()
        _write_status(status_path, state)


if __name__ == "__main__":
    main()
