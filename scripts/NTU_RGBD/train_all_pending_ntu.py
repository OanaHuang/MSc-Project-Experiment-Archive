from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
OUTPUT = ROOT / "Outputs_New/ntu_rgbd"
LOGS = ROOT / "launch_logs"
POLL_SECONDS = 60


def completed(path: Path) -> bool:
    status = path / "status.json"
    if not status.is_file():
        return False
    return json.loads(status.read_text(encoding="utf-8")).get("status") == "completed"


def wait_for(paths: list[Path], label: str) -> None:
    while not all(completed(path) for path in paths):
        states = []
        for path in paths:
            status_path = path / "status.json"
            state = "pending"
            if status_path.is_file():
                state = json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown")
            states.append(f"{path.parent.name}={state}")
        print(f"waiting {label}: " + " ".join(states), flush=True)
        time.sleep(POLL_SECONDS)


def run_batch(items: list[tuple[str, Path, int, list[str]]]) -> None:
    processes: dict[str, subprocess.Popen] = {}
    for name, run_dir, gpu, command in items:
        if completed(run_dir):
            print(f"skip completed {name}", flush=True)
            continue
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete run: {run_dir}")
        log_path = LOGS / f"ntu_all_pending_{name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes[name] = process
        print(f"launched {name} gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    while processes:
        failed = {name: process.poll() for name, process in processes.items()
                  if process.poll() not in (None, 0)}
        if failed:
            raise RuntimeError(f"pending batch failed: {failed}")
        if all(process.poll() == 0 for process in processes.values()):
            break
        print("running " + " ".join(
            f"{name}=pid{process.pid}" for name, process in processes.items()
        ), flush=True)
        time.sleep(POLL_SECONDS)


def jts_items() -> list[tuple[str, Path, int, list[str]]]:
    result = []
    for experiment, gpu in (("jts0", 0), ("jts5", 1)):
        run_dir = OUTPUT / f"joint_timestep/clip4_256_20ep/{experiment}/seed_42"
        command = [
            str(PYTHON), "scripts/NTU_RGBD/train_jts_series.py",
            "--experiment", experiment, "--device", f"cuda:{gpu}",
            "--physical-gpu", str(gpu), "--epochs", "20", "--seed", "42",
            "--skip-visualization",
        ]
        result.append((experiment, run_dir, gpu, command))
    return result


def fs_items() -> list[tuple[str, Path, int, list[str]]]:
    result = []
    for index in range(1, 5):
        name = f"fs{index}"
        gpu = index - 1
        run_dir = OUTPUT / f"f_smoothnet/window32_50ep/{name}/seed_42"
        command = [
            str(PYTHON), "scripts/NTU_RGBD/run_f_smoothnet.py",
            "--f-id", f"f{index}", "--device", f"cuda:{gpu}", "--epochs", "50",
        ]
        result.append((name, run_dir, gpu, command))
    return result


def pv_items(start: int, stop: int) -> list[tuple[str, Path, int, list[str]]]:
    sequence_root = OUTPUT / "t_ssnn/clip4_256_20ep/t0/seed_42/jatc_sequences"
    result = []
    for index in range(start, stop):
        name = f"pv{index}"
        gpu = index - start
        run_dir = OUTPUT / f"preliminary_occlusion/window16_20ep/{name}/seed_42"
        command = [
            str(PYTHON), "scripts/NTU_RGBD/train_occlusion_preliminary.py",
            "--experiment", name,
            "--train-matrices", str(sequence_root / "train/prediction_matrices.npz"),
            "--validation-matrices", str(sequence_root / "validation/prediction_matrices.npz"),
            "--output-dir", str(run_dir), "--device", f"cuda:{gpu}",
            "--epochs", "20", "--seed", "42",
        ]
        result.append((name, run_dir, gpu, command))
    return result


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    upstream = [
        OUTPUT / "cross_frame/clip4_256_20ep/cf13/seed_42",
        *(OUTPUT / f"video_hpe_solution/clip4_256_20ep/v{i}/seed_42" for i in range(1, 5)),
    ]
    wait_for(upstream, "CF13 and V-series")
    run_batch(jts_items())
    run_batch(fs_items())
    run_batch(pv_items(0, 4))
    run_batch(pv_items(4, 8))
    print("all unconditional pending NTU experiments completed", flush=True)


if __name__ == "__main__":
    main()
