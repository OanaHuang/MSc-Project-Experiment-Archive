from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New/ntu_rgbd/joint_timestep"
LOG_ROOT = PROJECT_ROOT / "launch_logs"
HAND_JOINTS = (
    "wrist_left", "wrist_right", "hand_left", "hand_right",
    "hand_tip_left", "hand_tip_right", "thumb_left", "thumb_right",
)


@dataclass(frozen=True)
class Run:
    display_id: str
    experiment: str
    gpu: int
    stage_steps: str

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / "clip4_256_20ep" / self.experiment / "seed_42"


RUNS = tuple(
    Run(f"JTS{index}", f"jts{index}", index - 1, schedule)
    for index, schedule in enumerate(("4-4-4-4", "4-3-2-1", "4-4-3-1", "4-4-4-1"), 1)
)


def build_command(run: Run) -> list[str]:
    return [
        str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/train_jts_series.py"),
        "--experiment", run.experiment, "--device", f"cuda:{run.gpu}",
        "--physical-gpu", str(run.gpu), "--epochs", "20",
        "--seed", "42", "--skip-visualization",
    ]


def status(run: Run) -> dict | None:
    path = run.run_dir / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def launch_runs() -> dict[str, subprocess.Popen]:
    collisions = [str(run.run_dir) for run in RUNS if run.run_dir.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite runs: " + ", ".join(collisions))
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    processes = {}
    for run in RUNS:
        log_path = LOG_ROOT / f"ntu_{run.experiment}_joint_timestep_20ep.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                build_command(run), cwd=PROJECT_ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        processes[run.experiment] = process
        print(f"launched {run.display_id} gpu={run.gpu} pid={process.pid} "
              f"log={log_path}", flush=True)
    return processes


def wait_for_runs(processes: dict[str, subprocess.Popen], poll_seconds: int) -> None:
    while True:
        exit_codes = {name: process.poll() for name, process in processes.items()}
        crashed = {name: code for name, code in exit_codes.items()
                   if code is not None and code != 0}
        if crashed:
            raise RuntimeError(f"JTS training processes failed: {crashed}")
        states = {run.experiment: (status(run) or {}).get("status") for run in RUNS}
        if all(value == "completed" for value in states.values()):
            for process in processes.values():
                process.wait(timeout=30)
            return
        failed = [name for name, value in states.items() if value == "failed"]
        if failed:
            raise RuntimeError(f"JTS runs reported failure: {failed}")
        print("waiting " + " ".join(
            f"{name}={value or 'starting'}" for name, value in states.items()
        ), flush=True)
        time.sleep(poll_seconds)


def hand_group_score(run: Run) -> float:
    payload = status(run) or {}
    per_joint = payload.get("metrics", {}).get("per_joint_pckhn", {})
    missing = [name for name in HAND_JOINTS if per_joint.get(name) is None]
    if missing:
        raise ValueError(f"{run.display_id} is missing hand metrics: {missing}")
    return sum(float(per_joint[name]) for name in HAND_JOINTS) / len(HAND_JOINTS)


def best_run() -> tuple[Run, dict[str, float]]:
    scores = {run.experiment: hand_group_score(run) for run in RUNS}
    selected = max(RUNS, key=lambda run: scores[run.experiment])
    return selected, scores


def probe_command(run: Run) -> list[str]:
    return [
        str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/probe_jts_stages.py"),
        "--run", str(run.run_dir),
        "--output-dir", str(OUTPUT_ROOT / "stage_probes_20ep/jts6/seed_42"),
        "--device", "cuda:0", "--epochs", "20", "--seed", "42",
    ]


def run_probes(run: Run) -> None:
    output = OUTPUT_ROOT / "stage_probes_20ep/jts6/seed_42"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage probes: {output}")
    log_path = LOG_ROOT / "ntu_jts6_stage_probes_20ep.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            probe_command(run), cwd=PROJECT_ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )
    print(f"completed JTS6 source={run.display_id} log={log_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run JTS1-JTS4 on four GPUs, then probe the best hand-joint model",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    for run in RUNS:
        print(f"{run.display_id}\tgpu={run.gpu}\tsteps={run.stage_steps}\t"
              f"status={(status(run) or {}).get('status', 'pending')}")
        if args.dry_run:
            print(shlex.join(build_command(run)))
    if args.list or args.dry_run:
        return
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    processes = launch_runs()
    wait_for_runs(processes, args.poll_seconds)
    selected, scores = best_run()
    print("hand-group " + " ".join(f"{key}={value:.6f}" for key, value in scores.items()),
          flush=True)
    print(f"selected {selected.display_id} for JTS6 Stage probes", flush=True)
    if not args.skip_probes:
        run_probes(selected)
    print("JTS queue completed", flush=True)


if __name__ == "__main__":
    main()
