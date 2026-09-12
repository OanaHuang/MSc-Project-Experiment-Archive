from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv/bin/python"
LOG_ROOT = PROJECT_ROOT / "launch_logs"


def command(index: int, epochs: int) -> list[str]:
    return [str(PYTHON), str(PROJECT_ROOT / "scripts/NTU_RGBD/run_f_smoothnet.py"),
            "--f-id", f"f{index}", "--device", f"cuda:{index - 1}",
            "--epochs", str(epochs)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FS1-FS4 on four GPUs")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    for index in range(1, 5):
        cmd = command(index, args.epochs)
        print(shlex.join(cmd), flush=True)
        if args.launch:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = LOG_ROOT / f"ntu_fs{index}_smoothnet.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            print(f"launched FS{index} pid={process.pid} log={log_path}", flush=True)


if __name__ == "__main__":
    main()
