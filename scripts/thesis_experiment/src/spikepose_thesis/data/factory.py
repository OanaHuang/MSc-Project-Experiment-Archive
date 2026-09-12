from __future__ import annotations

from typing import Any

from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path

from .mpii.datasets import MPIIPoseDataset, OfficialMPIIDataset
from .ntu.datasets import build_dataset as build_ntu_dataset


def build_dataset(config: dict[str, Any], split: str, max_samples: int | None = None):
    if split not in {"train", "train_evaluation", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    training_split = split == "train"
    if config["dataset"] == "mpii":
        data = config["data"]
        key = "train_metadata" if split in {"train", "train_evaluation"} else "validation_metadata"
        augmentation = config["training"].get("augmentation", {})
        if data.get("split_protocol") == "official_hrnet_mpii":
            return OfficialMPIIDataset(
                resolve_project_path(data[key]),
                resolve_project_path(data["images_dir"]),
                data["image_size"], data["heatmap_size"], data["sigma"],
                training_split, max_samples,
                float(augmentation.get("scale_factor", 0.25)),
                float(augmentation.get("rotation_degrees", 30.0)),
                bool(augmentation.get("flip_probability", 0.5) > 0),
                bool(data.get("color_rgb", True)),
            )
        return MPIIPoseDataset(
            resolve_project_path(data[key]), resolve_project_path(data["images_dir"]),
            data["image_size"], data["heatmap_size"], data["sigma"],
            data["crop_expansion"], training_split, max_samples,
            augmentation.get("scale_range") if training_split else None,
            augmentation.get("rotation_degrees", 0.0) if training_split else 0.0,
        )
    return build_ntu_dataset(PROJECT_ROOT, config, split, max_samples)
