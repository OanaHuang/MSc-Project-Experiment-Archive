from __future__ import annotations

import json
from pathlib import Path

import yaml

from .paths import output_run_dir


class OutputManager:
    def __init__(self, root: Path, dataset: str, category: str,
                 batch_name: str, experiment: str, model_name: str,
                 seed: int) -> None:
        self.run_dir = output_run_dir(
            root, dataset, category, batch_name, experiment, seed,
        )
        self.batch_dir = self.run_dir.parents[1]
        self.experiment = experiment
        self.model_name = model_name
        for name in ("config", "checkpoints", "logs", "metrics",
                     "visualizations", "analysis"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict) -> None:
        path = self.run_dir / "config" / "resolved.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def status(self, state: str, **details) -> None:
        payload = {
            "status": state,
            "experiment_id": self.experiment,
            "model_name": self.model_name,
            **details,
        }
        (self.run_dir / "status.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
