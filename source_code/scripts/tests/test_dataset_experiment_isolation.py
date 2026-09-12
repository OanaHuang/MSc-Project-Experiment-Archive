from __future__ import annotations

from pathlib import Path

import pytest

from scripts.spikepose.experiments import (
    ablation_roots, experiment_roots, load_ablation, resolve_config,
    validate_config,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(dataset: str, experiment: str) -> dict:
    directory = "MPII" if dataset == "mpii" else "NTU_RGBD"
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, dataset), experiment,
        PROJECT_ROOT / "scripts" / directory / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / directory / "configs" / "training.yaml",
    )
    validate_config(config)
    return config


def test_same_short_t_id_resolves_with_dataset_specific_semantics():
    mpii = resolve("mpii", "t0")
    ntu = resolve("ntu_rgbd", "t0")
    assert mpii["id"] == ntu["id"] == "t0"
    assert mpii["model"]["temporal"]["input_strategy"] == "repeat"
    assert ntu["model"]["temporal"]["input_strategy"] == "frames"
    assert mpii["dataset"] == "mpii"
    assert ntu["dataset"] == "ntu_rgbd"


def test_dataset_scopes_reject_cross_dataset_configs():
    mpii_only = resolve_config(
        experiment_roots(PROJECT_ROOT, "mpii"), "t0",
        PROJECT_ROOT / "scripts" / "NTU_RGBD" / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / "NTU_RGBD" / "configs" / "training.yaml",
    )
    with pytest.raises(ValueError, match="scoped to mpii"):
        validate_config(mpii_only)

    ntu_only = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), "t0",
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml",
    )
    with pytest.raises(ValueError, match="scoped to ntu_rgbd"):
        validate_config(ntu_only)


def test_temporal_ablation_lists_are_isolated():
    mpii = load_ablation(ablation_roots(PROJECT_ROOT, "mpii"), "temporal")
    ntu = load_ablation(ablation_roots(PROJECT_ROOT, "ntu_rgbd"), "temporal")
    assert mpii["experiments"] == [f"t{index}" for index in range(20)]
    assert ntu["experiments"] == [f"t{index}" for index in range(6)]
