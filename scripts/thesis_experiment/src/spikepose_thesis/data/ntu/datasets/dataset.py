from __future__ import annotations

from pathlib import Path

from .ntu_frame_dataset import NTUFrameDataset
from .transforms import (
    build_eval_transform, build_train_transform,
)
from ..core.joint_mapping import joint_layout
from ..preflight import load_validated_sample_ids
from ..runtime_cache import default_runtime_cache_root, require_runtime_cache


def build_dataset(project_root: Path, config: dict, split: str,
                  max_samples: int | None = None) -> NTUFrameDataset:
    data = config["data"]
    layout = joint_layout(data.get("joint_mapping", "ntu25_to_mpii16_v1"))
    configured_joints = int(config["model"].get("num_joints", 16))
    if configured_joints != int(layout["num_joints"]):
        raise ValueError(
            "Model/data joint-count mismatch: "
            f"model={configured_joints} layout={layout['num_joints']}"
        )
    frames_dir = data.get("frames_dir", data.get("extracted_frames_dir"))
    if frames_dir is None:
        raise ValueError("NTU data config must define frames_dir")
    frame_backend = data.get("frame_backend", "images")
    if frame_backend != "images":
        raise ValueError(
            f"Stage A supports frame_backend='images'; got {frame_backend!r}"
        )
    metadata_keys = {
        "train": "train_metadata",
        "train_evaluation": "train_metadata",
        "validation": "validation_metadata",
        "test": "test_metadata",
    }
    if split not in metadata_keys:
        raise ValueError(f"Unknown NTU split: {split}")
    training = split == "train"
    key = metadata_keys[split]
    metadata_path = project_root / data[key]
    validation_mode = data.get("pose_validation_mode")
    if validation_mode is None:
        validation_mode = (
            "strict" if data.get("validate_pose_sequences", True) else "skip"
        )
    if validation_mode not in {"strict", "manifest", "skip"}:
        raise ValueError(f"Unknown NTU pose_validation_mode: {validation_mode!r}")
    validated_sample_ids = (
        load_validated_sample_ids(data, metadata_path)
        if validation_mode == "manifest"
        else None
    )
    runtime_cache_mode = str(data.get("runtime_cache_mode", "disabled"))
    if runtime_cache_mode not in {"disabled", "optional", "required"}:
        raise ValueError(f"Unknown NTU runtime_cache_mode: {runtime_cache_mode!r}")
    if runtime_cache_mode == "required":
        require_runtime_cache(data)
    runtime_root = (
        default_runtime_cache_root(data)
        if runtime_cache_mode != "disabled" else None
    )
    augmentation = config.get("training", {}).get("augmentation", {})
    if training:
        transform = build_train_transform(
            image_size=data["image_size"],
            scale_range=augmentation.get("scale_range"),
            rotation_degrees=augmentation.get("rotation_degrees", 0.0),
            flip_probability=augmentation.get("flip_probability", 0.0),
            flip_pairs=layout["flip_pairs"],
        )
    else:
        transform = build_eval_transform(image_size=data["image_size"])
    return NTUFrameDataset(
        metadata_csv=metadata_path,
        transform=transform,
        image_size=data["image_size"], heatmap_size=data["heatmap_size"],
        sigma=data["sigma"], frame_stride=data["frame_stride"],
        temporal_steps=(
            int(config["model"].get("temporal", {}).get("video_frames", 1))
            if config["model"].get("temporal", {}).get("input_strategy")
            in {"frames", "scheduled_frames"}
            else 1
        ),
        temporal_frame_gap=int(data.get("temporal_frame_gap", 1)),
        minimum_temporal_history=int(data.get("minimum_temporal_history", 0)),
        preprocessed_pose_cache=bool(data.get("preprocessed_pose_cache", False)),
        return_temporal_sequence=(
            config["model"].get("temporal", {}).get("input_strategy")
            in {"frames", "scheduled_frames"}
        ),
        return_temporal_targets=(
            bool(config.get("training", {}).get("loss_all_frames", False)) or
            float(config["model"].get("temporal", {}).get(
                "kinematic_loss_weight", 0.0,
            )) > 0.0
        ),
        single_person_only=False, max_samples=max_samples,
        skeleton_cache_size=int(data.get("skeleton_cache_size", 8)),
        person_crop=data["person_crop"], bbox_expansion=data["bbox_expansion"],
        extracted_frames_dir=project_root / frames_dir,
        frame_layout=data.get("frame_layout", "flat"),
        frame_clip_subdir=data.get("frame_clip_subdir", "contiguous_2x16"),
        exclude_overlapping_clips=bool(data.get("exclude_overlapping_clips", False)),
        validate_pose_sequences=validation_mode == "strict",
        head_index=int(layout["head_index"]),
        neck_index=int(layout["neck_index"]),
        map_to_mpii16=bool(layout["map_to_mpii16"]),
        tube_crop=bool(data.get("tube_crop", True)),
        tube_crop_scope=str(data.get("tube_crop_scope", "window")),
        validated_sample_ids=validated_sample_ids,
        generate_heatmaps=(
            str(config.get("training", {}).get("target_heatmap_backend", "cpu"))
            != "gpu"
        ),
        runtime_cache_root=runtime_root,
        runtime_spatial_crops=bool(data.get("runtime_spatial_crops", False)),
        allowed_setups=data.get("setup_filter"),
    )


__all__ = ["NTUFrameDataset", "build_dataset"]
