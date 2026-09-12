from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def verify_completed_runs(
    runs: Iterable[tuple[str, Path]], *, checkpoint_names=("last.pt", "best.pt"),
    context: str = "run completion",
) -> None:
    """Apply the shared E/S continuation gate to a collection of run dirs."""
    problems = []
    for label, run_dir in runs:
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            problems.append(f"{label}: missing status.json")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            problems.append(f"{label}: status={status.get('status')}")
        for checkpoint in checkpoint_names:
            if not (run_dir / "checkpoints" / checkpoint).is_file():
                problems.append(f"{label}: missing checkpoints/{checkpoint}")
    if problems:
        raise RuntimeError(f"{context} failed: " + "; ".join(problems))
