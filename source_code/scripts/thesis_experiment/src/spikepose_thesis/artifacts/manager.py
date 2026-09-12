from __future__ import annotations

import json
import hashlib
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch
import yaml

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import default_output_root
from spikepose_thesis.core.paths import resolve_project_path


def file_fingerprint(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path), "exists": True, "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_dir(config: dict, seed: int, output_root: Path | None = None) -> Path:
    root = output_root or default_output_root(str(config.get("run_type", "formal")))
    base = root / "runs" / config["dataset"] / config["stage"] / config["id"]
    variant = config.get("run_variant")
    if variant is not None:
        variant = str(variant)
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:__[A-Za-z0-9_]+)*", variant):
            raise ValueError(f"Unsafe run variant: {variant!r}")
        base = base / "variants" / variant
    return base / f"seed_{seed}"


def locate_source_run(experiment: str, seed: int,
                      output_root: Path | None = None) -> Path:
    return run_dir(load_experiment(experiment), seed, output_root)


class RunArtifacts:
    def __init__(self, config: dict, seed: int, output_root: Path | None = None):
        self.config = config
        self.seed = int(seed)
        self.output_root = output_root or default_output_root(
            str(config.get("run_type", "formal")),
        )
        self.path = run_dir(config, seed, self.output_root)
        for child in ("checkpoints", "training", "predictions", "metrics", "analysis", "logs"):
            (self.path / child).mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        resolved_config = self.path / "resolved_config.yaml"
        resolved_config.write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8",
        )
        data = self.config.get("data", {})
        metadata = {
            key: file_fingerprint(resolve_project_path(value))
            for key, value in data.items()
            if key.endswith("_metadata") and isinstance(value, str)
        }
        self.write_json("dataset_fingerprint.json", {
            "dataset": self.config.get("dataset"), "metadata": metadata,
            "frames_dir": data.get("frames_dir"),
            "frame_layout": data.get("frame_layout"),
            "frame_clip_subdir": data.get("frame_clip_subdir"),
            "setup_filter": data.get("setup_filter"),
            "runtime_overrides": self.config.get("runtime_overrides"),
        })
        self.write_json("run_identity.json", {
            "resolved_config": file_fingerprint(resolved_config),
            "git_commit": _git_commit(),
        })
        self.write_json("environment.json", {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "git_commit": _git_commit(),
        })
        self.status("initialized")

    def status(self, state: str, **details: Any) -> None:
        payload = {"state": state, "experiment": self.config["id"],
                   "paper_id": self.config["paper_id"], "seed": self.seed,
                   "run_type": self.config.get("run_type", "formal"),
                   "profile": self.config.get("profile"), "pid": os.getpid(),
                   **details}
        temporary = self.path / "status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path / "status.json")

    def write_json(self, relative: str, value: Any) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path
