from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.run_jatc_series import (
    EXPERIMENT_BY_ID, build_command, run_dir,
)


PYTHON = PROJECT_ROOT / ".venv/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
LOG_ROOT = PROJECT_ROOT / "launch_logs"
OFFLINE = ("jp0", "jp1", "jp2", "jp3")
REPLICATIONS = ("jp4_s2", "jp4_s3")


def directory(run_id: str, epochs: int) -> Path:
    return run_dir(OUTPUT_ROOT, EXPERIMENT_BY_ID[run_id], epochs)


def status(run_id: str, epochs: int) -> dict | None:
    path = directory(run_id, epochs) / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def metrics(run_id: str, epochs: int) -> dict:
    value = status(run_id, epochs)
    if not value or value.get("status") != "completed":
        raise RuntimeError(f"{run_id} is not complete")
    return value["metrics"]


def jp2_passes(epochs: int) -> bool:
    jp0, jp1, jp2 = (metrics(item, epochs) for item in ("jp0", "jp1", "jp2"))
    return (jp2["pckhn"] >= jp0["pckhn"] - 0.005
            and jp2["nacce"] <= 0.95 * jp1["nacce"])


def jp3_passes(epochs: int) -> bool:
    jp0, jp2, jp3 = (metrics(item, epochs) for item in ("jp0", "jp2", "jp3"))
    return jp3["pckhn"] >= jp0["pckhn"] - 0.005 and jp3["nacce"] < jp2["nacce"]


def jp4_passes(epochs: int) -> bool:
    jp0, jp3, jp4 = (metrics(item, epochs) for item in ("jp0", "jp3", "jp4"))
    return (jp4["pckhn"] >= jp0["pckhn"] - 0.005
            and jp4["nacce"] <= 0.85 * jp0["nacce"]
            and jp4["nacce"] < jp3["nacce"])


def command(run_id: str, epochs: int, device: str) -> list[str]:
    return build_command(run_id, python_executable=str(PYTHON), output_root=OUTPUT_ROOT,
                         epochs=epochs, device=device)


def run_foreground(run_id: str, epochs: int, device: str) -> None:
    if directory(run_id, epochs).exists():
        raise FileExistsError(f"Refusing to overwrite {directory(run_id, epochs)}")
    subprocess.run(command(run_id, epochs, device), cwd=PROJECT_ROOT, check=True)


def launch(run_id: str, epochs: int, device: str) -> subprocess.Popen:
    if directory(run_id, epochs).exists():
        raise FileExistsError(f"Refusing to overwrite {directory(run_id, epochs)}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"ntu_jatc_{run_id}.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command(run_id, epochs, device), cwd=PROJECT_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print(f"launched {run_id} pid={process.pid} log={log_path}", flush=True)
    return process


def wait(run_ids: tuple[str, ...], epochs: int, poll_seconds: int) -> None:
    while True:
        states = {item: (status(item, epochs) or {}).get("status") for item in run_ids}
        if all(value == "completed" for value in states.values()): return
        if any(value == "failed" for value in states.values()):
            raise RuntimeError(f"JATC run failed: {states}")
        print("waiting " + " ".join(f"{key}={value or 'running'}" for key, value in states.items()),
              flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gated seven-run NTU JATC pilot queue")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--replication-gpus", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    runs = (*OFFLINE, "jp4", *REPLICATIONS)
    for run_id in runs:
        item = EXPERIMENT_BY_ID[run_id]
        device = "cpu" if item.stage == "offline" else f"cuda:{args.gpu}"
        print(f"{run_id}\tseed={item.seed}\tstage={item.stage}\t"
              f"status={(status(run_id, args.epochs) or {}).get('status', 'pending')}\t"
              f"output={directory(run_id, args.epochs).relative_to(PROJECT_ROOT)}")
        if args.dry_run: print(shlex.join(command(run_id, args.epochs, device)))
    if args.list or args.dry_run: return
    if not PYTHON.is_file(): raise FileNotFoundError(PYTHON)
    for run_id in OFFLINE[:3]: run_foreground(run_id, args.epochs, "cpu")
    if not jp2_passes(args.epochs):
        print("STOP: JP2 did not pass the pre-registered JP1/PCKh gate", flush=True); return
    run_foreground("jp3", args.epochs, "cpu")
    if not jp3_passes(args.epochs):
        print("STOP: JP3 did not improve on JP2 within the PCKh gate", flush=True); return
    launch("jp4", args.epochs, f"cuda:{args.gpu}"); wait(("jp4",), args.epochs, args.poll_seconds)
    if not jp4_passes(args.epochs):
        print("STOP: JP4 did not pass the replication gate", flush=True); return
    for run_id, gpu in zip(REPLICATIONS, args.replication_gpus):
        launch(run_id, args.epochs, f"cuda:{gpu}")
    wait(REPLICATIONS, args.epochs, args.poll_seconds)
    print("seven-run JATC pilot completed", flush=True)


if __name__ == "__main__":
    main()
