from __future__ import annotations

from pathlib import Path

from .ntu_frame_dataset import NTUFrameDataset
from .transforms import (
    build_eval_transform, build_train_transform,
)


def build_dataset(project_root: Path, config: dict, split: str,
                  max_samples: int | None = None) -> NTUFrameDataset:
    data = config["data"]
    frame_backend = data.get("frame_backend", "images")
    if frame_backend != "images":
        raise ValueError(
            f"Stage A supports frame_backend='images'; got {frame_backend!r}"
        )
    training = split == "train"
    key = "train_metadata" if training else "validation_metadata"
    augmentation = config.get("training", {}).get("augmentation", {})
    if training:
        transform = build_train_transform(
            image_size=data["image_size"],
            scale_range=augmentation.get("scale_range"),
            rotation_degrees=augmentation.get("rotation_degrees", 0.0),
            flip_probability=augmentation.get("flip_probability", 0.0),
        )
    else:
        transform = build_eval_transform(image_size=data["image_size"])
    return NTUFrameDataset(
        metadata_csv=project_root / data[key],
        transform=transform,
        image_size=data["image_size"], heatmap_size=data["heatmap_size"],
        sigma=data["sigma"], frame_stride=data["frame_stride"],
        temporal_steps=(
            int(config["model"]["num_steps"])
            if config["model"].get("temporal", {}).get("input_strategy") == "frames"
            else 1
        ),
        temporal_frame_gap=int(data.get("temporal_frame_gap", 1)),
        minimum_temporal_history=int(data.get("minimum_temporal_history", 0)),
        preprocessed_pose_cache=bool(data.get("preprocessed_pose_cache", False)),
        return_temporal_sequence=(
            config["model"].get("temporal", {}).get("input_strategy") == "frames"
        ),
        return_temporal_targets=(
            float(config["model"].get("temporal", {}).get(
                "kinematic_loss_weight", 0.0,
            )) > 0.0
        ),
        single_person_only=True, max_samples=max_samples,
        skeleton_cache_size=int(data.get("skeleton_cache_size", 8)),
        person_crop=data["person_crop"], bbox_expansion=data["bbox_expansion"],
        extracted_frames_dir=project_root / data["extracted_frames_dir"],
        validate_pose_sequences=data.get("validate_pose_sequences", True),
        path_root=project_root,
        contiguous_clip_subdir=data.get("contiguous_clip_subdir"),
        minimum_visible_joints=int(data.get("minimum_visible_joints", 1)),
    )


__all__ = ["NTUFrameDataset", "build_dataset"]
