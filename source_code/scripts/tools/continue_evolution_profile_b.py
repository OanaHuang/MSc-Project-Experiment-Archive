from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.artifacts import (
    output_batch_dir, output_run_dir, verify_completed_runs,
)


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
BATCH_ROOT = output_batch_dir(
    OUTPUT_ROOT, "mpii", "main_route", "evolution_profile_b",
)
EXPERIMENTS = ("e0", "e1", "e2")
TRAINING_SESSION = "evolution_e0_e2"


def session_exists() -> bool:
    result = subprocess.run(
        ("tmux", "list-sessions", "-F", "#{session_name}"),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return TRAINING_SESSION in result.stdout.splitlines()


def verify_twenty_epoch_outputs() -> None:
    verify_completed_runs(
        (
            (
                experiment,
                output_run_dir(
                    OUTPUT_ROOT, "mpii", "main_route", "evolution_profile_b",
                    experiment, 42,
                ),
            )
            for experiment in EXPERIMENTS
        ),
        context="20-epoch verification",
    )


def main() -> None:
    while session_exists():
        time.sleep(30)
    verify_twenty_epoch_outputs()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tools" / "run_ablation.py"),
        "--dataset", "mpii",
        "--experiments", *EXPERIMENTS,
        "--category", "main_route",
        "--batch-name", "evolution_profile_b",
        "--seeds", "42",
        "--devices", "0", "2", "3",
        "--resume",
        "--epochs", "120",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
