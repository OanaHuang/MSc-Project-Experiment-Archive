from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_path(config: dict, path: str, value: Any) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _roots(root: Path | Iterable[Path]) -> tuple[Path, ...]:
    if isinstance(root, Path):
        return (root,)
    return tuple(Path(item) for item in root)


def experiment_roots(project_root: Path, dataset: str) -> tuple[Path, ...]:
    """Return dataset-specific experiments before shared model definitions."""
    if dataset not in {"mpii", "ntu_rgbd"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    dataset_dir = {"mpii": "MPII", "ntu_rgbd": "NTU_RGBD"}[dataset]
    return (
        project_root / "scripts" / dataset_dir / "experiments",
        project_root / "scripts" / "experiments",
    )


def ablation_roots(project_root: Path, dataset: str) -> tuple[Path, ...]:
    """Return dataset-specific ablations before shared study definitions."""
    if dataset not in {"mpii", "ntu_rgbd"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    dataset_dir = {"mpii": "MPII", "ntu_rgbd": "NTU_RGBD"}[dataset]
    return (
        project_root / "scripts" / dataset_dir / "ablations",
        project_root / "scripts" / "ablations",
    )


def _find_yaml(root: Path | Iterable[Path], name: str) -> Path:
    candidates = [item / f"{name}.yaml" for item in _roots(root)]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Experiment or ablation {name!r} not found in: "
            + ", ".join(str(path.parent) for path in candidates)
        )
    return matches[0]


def load_experiment(root: Path | Iterable[Path], name: str) -> dict:
    path = _find_yaml(root, name)
    config = read_yaml(path)
    parent = config.pop("parent", None)
    changes = config.pop("changes", {})
    if parent:
        config = deep_merge(load_experiment(root, parent), config)
    for key, value in changes.items():
        set_path(config, key, value)
    config["experiment"] = name
    return config


def load_ablation(root: Path | Iterable[Path], name: str) -> dict:
    config = read_yaml(_find_yaml(root, name))
    if config.get("id") != name:
        raise ValueError(f"Ablation id must match its filename: {name}")
    experiments = config.get("experiments", [])
    if not experiments or len(experiments) != len(set(experiments)):
        raise ValueError(f"Ablation {name} must contain unique experiments")
    return config


def resolve_config(root: Path | Iterable[Path], experiment: str, task_path: Path,
                   training_path: Path, overrides: dict | None = None) -> dict:
    config = load_experiment(root, experiment)
    config = deep_merge(config, read_yaml(task_path))
    profile = config.get("training_profile")
    profile_path = training_path.with_name(f"{profile}.yaml") if profile else training_path
    if profile and not profile_path.exists():
        raise FileNotFoundError(f"Training profile not found: {profile_path}")
    config = deep_merge(config, read_yaml(profile_path))
    if overrides:
        config = deep_merge(config, overrides)
    return config
