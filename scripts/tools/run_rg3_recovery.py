from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
BATCH_NAME = "integral_fix_ep020_seed42"
RUN_DIR = (
    OUTPUT_ROOT / "mpii" / "coordinate_regression" / BATCH_NAME
    / "rg3" / "seed_42"
)
RG2_PCKH = 0.462056482738428
MINIMUM_IMPROVEMENT = 0.02


def train_command(epochs: int, resume: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "MPII" / "train.py"),
        "--experiment", "rg3",
        "--category", "coordinate_regression",
        "--batch-name", BATCH_NAME,
        "--seed", "42",
        "--device", "cuda:0",
        "--physical-gpu", "3",
        "--output-root", str(OUTPUT_ROOT),
        "--epochs", str(epochs),
        "--checkpoint-epochs", "20", "120",
        "--skip-visualization",
    ]
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "3"
    subprocess.run(train_command(20), cwd=PROJECT_ROOT, env=environment, check=True)
    status = json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8"))
    pckh = float(status["metrics"]["official_pckh"])
    threshold = RG2_PCKH + MINIMUM_IMPROVEMENT
    decision = {
        "rg2_reference_pckh": RG2_PCKH,
        "minimum_improvement": MINIMUM_IMPROVEMENT,
        "continuation_threshold": threshold,
        "rg3_epoch20_pckh": pckh,
        "continue_to_epoch120": pckh >= threshold,
    }
    (RUN_DIR / "continuation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(decision, indent=2), flush=True)
    if pckh >= threshold:
        subprocess.run(
            train_command(120, resume=True),
            cwd=PROJECT_ROOT, env=environment, check=True,
        )


if __name__ == "__main__":
    main()
