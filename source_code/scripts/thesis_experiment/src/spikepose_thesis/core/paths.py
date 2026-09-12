from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get(
    "SPIKEPOSE_PROJECT_ROOT", PACKAGE_ROOT.parents[1],
)).resolve()


def default_output_root(run_type: str = "formal") -> Path:
    if run_type == "pilot":
        return Path(os.environ.get(
            "SPIKEPOSE_PILOT_OUTPUT_ROOT",
            PROJECT_ROOT / "Outputs_Thesis_Pilot20",
        )).resolve()
    if run_type != "formal":
        raise ValueError(f"Unsupported run type: {run_type}")
    return Path(os.environ.get(
        "SPIKEPOSE_OUTPUT_ROOT", PROJECT_ROOT / "Outputs_Thesis",
    )).resolve()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
