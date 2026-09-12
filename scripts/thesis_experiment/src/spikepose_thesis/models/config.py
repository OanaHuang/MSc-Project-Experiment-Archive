from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeuronConfig:
    kind: str = "ilif"
    decay: float = 0.90
    threshold: float = 1.0
    max_spikes: int = 4
    membrane_readout: bool = False
    membrane_readout_init: float = 0.01
    learnable_decay: bool = False
    learnable_initial_membrane: bool = False
    initial_membrane_scale: float = 0.25


@dataclass(frozen=True)
class BackboneConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    depths: tuple[int, ...] = (1, 2, 2)
    blocks: tuple[str, ...] = ("all_conv", "all_conv", "conv")
    output_stages: tuple[int, ...] = (2, 3)
    attention_stages: tuple[int, ...] = ()


@dataclass(frozen=True)
class NeckConfig:
    kind: str = "concat"
    out_channels: int = 128
    interpolation: str = "nearest"
    align_corners: bool = False
    stage_mask: tuple[int, ...] | None = None


@dataclass(frozen=True)
class HeadConfig:
    kind: str = "ann_heatmap"
    hidden_channels: int = 128
    upsample_factor: int = 2
    output_init: str = "small_normal"
    output_init_std: float = 0.001
    split_ratio: int = 1
    label_sigma: float = 2.0
    coordinate_size: int = 256
    regression_variant: str = "gap"
    regression_channels: int = 32
    spatial_pool_size: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class TemporalConfig:
    video_frames: int = 1
    input_strategy: str = "repeat"
    aggregation: str = "mean"
    translation_pixels: int = 0
    randomize_training: bool = True
    eval_translation: tuple[int, int] | None = None
    intermediate_loss_weight: float = 0.0
    kinematic_loss_weight: float = 0.0
    feedback_edges: tuple[tuple[int, int], ...] = ()
    history_mode: str = "real"
    state_mode: str = "continuous"
    stage_steps: tuple[int, ...] = ()
    transform_kind: str = "none"
    output_steps: int | None = None
    early_classifier_stages: tuple[int, ...] = ()
    early_classifier_weights: tuple[float, ...] = ()
    snn_steps_per_frame: int = 1
    decouple_video_time: bool = False
    # Maps each continuous SNN update to one position in the physical-frame
    # input.  For example (0, 0, 1, 1) implements V2U4 as
    # [I_(t-1), I_(t-1), I_t, I_t] without resetting neuron state.
    update_schedule: tuple[int, ...] = ()
    memory_kind: str = "none"
    memory_decay: float = 0.90
    memory_decay_min: float = 0.50
    memory_decay_max: float = 0.99
    memory_shared_decay: bool = False
    memory_alignment: bool = False
    memory_adaptive: bool = False
    memory_predictive: bool = False
    memory_use_current_residual: bool = True
    # MAM V2 keeps these values explicit so YAML fully determines temporal
    # behavior. The legacy MAM implementation ignores them.
    memory_residual_gamma_init: float = 0.0
    memory_max_residual_offset_px: float = 2.0
    memory_align_corners: bool = False
    memory_padding_mode: str = "border"
    memory_use_current_direct_path: bool = True
    memory_use_motion_token: bool = True
    memory_use_residual_offset: bool = True
    memory_use_dynamic_gate: bool = True
    memory_use_memory_update: bool = True
    memory_use_predictive_head: bool = False
    memory_bptt_mode: str = "full16"
    memory_detach_interval: int = 0
    memory_offset_hidden_features: int = 64
    memory_joint_embedding_features: int = 8
    # KTPFormer-inspired graph priors operate on MAM's 17-D motion tokens.
    # TPA remains causal by default so online inference never sees the future.
    memory_use_kpa: bool = False
    memory_use_tpa: bool = False
    memory_ktp_causal: bool = True
    memory_ktp_global_init: float = 1e-6


@dataclass(frozen=True)
class SpikePoseConfig:
    in_channels: int = 3
    num_joints: int = 16
    num_steps: int = 2
    heatmap_size: tuple[int, int] | None = (64, 64)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpikePoseConfig":
        model = value.get("model", value)
        heatmap = model.get("heatmap_size", (64, 64))
        return cls(
            in_channels=int(model.get("in_channels", 3)),
            num_joints=int(model.get("num_joints", 16)),
            num_steps=int(model.get("num_steps", 2)),
            heatmap_size=None if heatmap is None else tuple(heatmap),
            temporal=TemporalConfig(**{
                key: (
                    tuple(tuple(edge) for edge in item)
                    if key == "feedback_edges" and isinstance(item, list)
                    else tuple(item)
                    if key in {
                        "eval_translation", "stage_steps", "early_classifier_stages",
                        "early_classifier_weights", "update_schedule",
                    } and isinstance(item, list)
                    else item
                )
                for key, item in model.get("temporal", {}).items()
            }),
            neuron=NeuronConfig(**model.get("neuron", {})),
            backbone=BackboneConfig(**{
                key: tuple(item) if isinstance(item, list) else item
                for key, item in model.get("backbone", {}).items()
            }),
            neck=NeckConfig(**{
                key: tuple(item) if key == "stage_mask" and isinstance(item, list) else item
                for key, item in model.get("neck", {}).items()
            }),
            head=HeadConfig(**{
                key: tuple(item) if key == "spatial_pool_size" and isinstance(item, list)
                else item
                for key, item in model.get("head", {}).items()
            }),
        )
