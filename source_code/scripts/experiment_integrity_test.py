from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ROOT = PROJECT_ROOT / "scripts" / "experiments"
ABLATION_ROOT = PROJECT_ROOT / "scripts" / "ablations"
MAIN_ROUTE = [
    "baseline", "e0", "e1", "e2", "four_stage", "mem_ann",
    "mem_spikefpn_annhead", "mem_fpn", "resformer_s3", "resformer_s4",
    "resformer_fpn",
]
PROFILE_ROUTE = ["p0", "p1", "p2", "p3", "p4", "p5"]


def read_yaml(path: Path) -> dict:
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


def set_path(config: dict, path: str, value: object) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_experiment(root: Path, name: str) -> dict:
    config = read_yaml(root / f"{name}.yaml")
    parent = config.pop("parent", None)
    changes = config.pop("changes", {})
    if parent:
        config = deep_merge(load_experiment(root, parent), config)
    for key, value in changes.items():
        set_path(config, key, value)
    config["experiment"] = name
    return config


def load_ablation(root: Path, name: str) -> dict:
    config = read_yaml(root / f"{name}.yaml")
    assert config["id"] == name
    assert len(config["experiments"]) == len(set(config["experiments"]))
    return config


def validate_config(config: dict) -> None:
    assert config["id"] == config["experiment"]
    assert config.get("name")
    model = config["model"]
    backbone = model["backbone"]
    assert len(backbone["channels"]) == len(backbone["depths"]) == len(backbone["blocks"])
    stages = set(range(1, len(backbone["channels"]) + 1))
    assert set(backbone["output_stages"]).issubset(stages)
    assert set(backbone.get("attention_stages", [])).issubset(stages)
    assert model["neck"]["kind"] in {"concat", "add", "spike_fpn"}
    assert model["head"]["kind"] in {
        "ann_heatmap", "spiking_heatmap", "linear_heatmap",
        "coordinate_classification", "coordinate_regression",
    }


def model_config(name: str) -> dict:
    config = load_experiment(EXPERIMENT_ROOT, name)
    validate_config(config)
    return config["model"]


def flatten(value: dict, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def changed_keys(parent: str, child: str) -> set[str]:
    left = flatten(model_config(parent))
    right = flatten(model_config(child))
    return {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def main() -> None:
    route = load_ablation(ABLATION_ROOT, "main_route")
    assert route["experiments"] == MAIN_ROUTE
    assert load_experiment(EXPERIMENT_ROOT, "baseline")["training_profile"] == "pose_ablation"
    for name in ("e0", "e1", "e2"):
        assert load_experiment(EXPERIMENT_ROOT, name)["training_profile"] == "residual_bf"
    assert load_experiment(EXPERIMENT_ROOT, "four_stage")["training_profile"] == "residual_bf"

    profile_route = load_ablation(ABLATION_ROOT, "training_profile")
    assert profile_route["experiments"] == PROFILE_ROUTE
    expected_profiles = {
        "p0": "pose_ablation",
        "p1": "pose_ablation",
        "p2": "profile_p2_schedule",
        "p3": "profile_p3_clipping",
        "p4": "profile_p4_scale",
        "p5": "residual_bf",
    }
    for name, profile in expected_profiles.items():
        assert load_experiment(EXPERIMENT_ROOT, name)["training_profile"] == profile
    assert changed_keys("p0", "p1") == {"head.output_init"}
    for left, right in zip(PROFILE_ROUTE[1:], PROFILE_ROUTE[2:]):
        assert changed_keys(left, right) == set(), (left, right)

    # Strict new comparisons. The two Spike-FPN settings form one fusion-method
    # intervention and reproduce the already-trained mem_fpn implementation.
    expected_diffs = {
        # B0 -> E0 changes the training recipe and aligns initialization with
        # the completed Profile B subchain.
        ("baseline", "e0"): {"head.output_init"},
        ("e0", "e1"): {"head.upsample_factor"},
        ("e1", "e2"): {"backbone.output_stages"},
        ("e2", "four_stage"): {
            "backbone.channels", "backbone.depths", "backbone.blocks",
            "backbone.output_stages",
        },
        ("four_stage", "mem_ann"): {"backbone.blocks"},
        ("mem_ann", "mem_spikefpn_annhead"): {
            "neck.kind", "neck.interpolation",
        },
        ("mem_spikefpn_annhead", "mem_fpn"): {"head.kind"},
        ("mem_fpn", "resformer_s3"): {"backbone.attention_stages"},
        ("mem_fpn", "resformer_s4"): {"backbone.attention_stages"},
        ("mem_fpn", "resformer_fpn"): {"backbone.attention_stages"},
    }
    for pair, expected in expected_diffs.items():
        actual = changed_keys(*pair)
        assert actual == expected, (pair, actual, expected)

    # Compatibility snapshots protect canonical architectures and the completed
    # downstream models while the inheritance graph is cleaned up.
    snapshots = {
        "baseline": {
            "backbone.channels": [64, 128, 256],
            "backbone.output_stages": [2, 3],
            "neck.kind": "concat",
            "head.kind": "ann_heatmap",
            "head.upsample_factor": 2,
        },
        "four_stage": {
            "backbone.channels": [64, 128, 256, 256],
            "backbone.blocks": ["all_conv", "all_conv", "conv", "conv"],
            "backbone.output_stages": [1, 2, 3, 4],
            "neck.kind": "concat",
            "head.kind": "ann_heatmap",
        },
        "mem_ann": {
            "backbone.blocks": ["all_conv", "membrane", "membrane", "membrane"],
            "neck.kind": "concat",
            "head.kind": "ann_heatmap",
        },
        "mem_fpn": {
            "backbone.blocks": ["all_conv", "membrane", "membrane", "membrane"],
            "neck.kind": "spike_fpn",
            "neck.interpolation": "bilinear",
            "head.kind": "linear_heatmap",
        },
        "resformer_fpn": {
            "backbone.attention_stages": [3, 4],
            "neck.kind": "spike_fpn",
            "head.kind": "linear_heatmap",
        },
    }
    for name, expected in snapshots.items():
        actual = flatten(model_config(name))
        for key, value in expected.items():
            assert actual[key] == value, (name, key, actual[key], value)

    print("Experiment integrity checks passed for B0, E0-E9, P0-P5, and trained-model snapshots.")


if __name__ == "__main__":
    main()
