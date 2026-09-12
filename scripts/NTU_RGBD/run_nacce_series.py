from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NAccEExperiment:
    experiment: str
    source_experiment: str
    category: str
    purpose: str


EXPERIMENTS = (
    NAccEExperiment("nacce_t0", "t0", "t_ssnn", "single-frame continuity baseline"),
)
EXPERIMENT_BY_ID = {item.experiment: item for item in EXPERIMENTS}


def build_command(experiment: str, *, python_executable: str = sys.executable,
                  seed: int = 42, epochs: int = 20) -> list[str]:
    item = EXPERIMENT_BY_ID[experiment]
    run = PROJECT_ROOT / "Outputs_New/ntu_rgbd" / item.category / f"clip4_256_{epochs}ep" / item.source_experiment / f"seed_{seed}"
    return [python_executable, str(PROJECT_ROOT / "scripts/NTU_RGBD/evaluate_nacce.py"), "--run", str(run)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed NTU NAccE series")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_ID))
    action.add_argument("--list", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("id\tsource\tpurpose")
        for item in EXPERIMENTS:
            print(f"{item.experiment}\t{item.source_experiment}\t{item.purpose}")
        return
    command = build_command(args.experiment, seed=args.seed, epochs=args.epochs)
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
