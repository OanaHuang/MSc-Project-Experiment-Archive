from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SpikePoseMSBackbone
from .config import SpikePoseConfig
from .heads import build_head
from .necks import build_neck
from .baselines import build_hrnet, build_pose_resnet
from .mam import MotionAlignedMembrane
from .mam_v2 import MAMV2Config, MAMV2Output, MotionAlignedMembraneV2


MAM_V2_KINDS = {
    "carry_fixed", "plif_low", "gt_align_oracle", "pred_align_static",
    "mam_v2", "mam_v2_noalign", "mam_v2_predictive", "mam_v2_temporal",
}


class SpikePose(nn.Module):
    """Dataset-independent pose model assembled from backbone, neck and head."""

    def __init__(self, config: SpikePoseConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SpikePoseMSBackbone(
            config.in_channels, config.backbone, config.neuron,
            config.temporal, config.num_steps,
        )
        channels = {index: value for index, value in
                    enumerate(config.backbone.channels, 1)}
        self.neck = build_neck(
            config.neck.kind, config.backbone.output_stages,
            channels, config.neck.out_channels,
            config.neuron, config.neck.interpolation, config.neck.align_corners,
            config.neck.stage_mask,
        )
        self.head = build_head(
            config.head.kind, self.neck.out_channels,
            config.head.hidden_channels, config.num_joints,
            config.heatmap_size, config.neuron, config.num_steps,
            config.head.upsample_factor, config.temporal.aggregation,
            config.temporal.feedback_edges, config.head,
        )
        memory_kind = config.temporal.memory_kind.lower()
        if memory_kind in MAM_V2_KINDS:
            self.mam = MotionAlignedMembraneV2(
                config.num_joints,
                MAMV2Config(
                    mode=memory_kind,
                    decay_min=config.temporal.memory_decay_min,
                    decay_max=config.temporal.memory_decay_max,
                    decay_init=config.temporal.memory_decay,
                    residual_gamma_init=(
                        config.temporal.memory_residual_gamma_init
                    ),
                    max_residual_offset_px=(
                        config.temporal.memory_max_residual_offset_px
                    ),
                    align_corners=config.temporal.memory_align_corners,
                    padding_mode=config.temporal.memory_padding_mode,
                    use_current_direct_path=(
                        config.temporal.memory_use_current_direct_path
                    ),
                    use_motion_token=(
                        config.temporal.memory_use_motion_token
                    ),
                    use_residual_offset=(
                        config.temporal.memory_use_residual_offset
                    ),
                    use_dynamic_gate=config.temporal.memory_use_dynamic_gate,
                    use_memory_update=(
                        config.temporal.memory_use_memory_update
                    ),
                    use_predictive_head=(
                        config.temporal.memory_use_predictive_head
                    ),
                    shared_decay=config.temporal.memory_shared_decay,
                    bptt_mode=config.temporal.memory_bptt_mode,
                    detach_interval=config.temporal.memory_detach_interval,
                    offset_hidden_features=(
                        config.temporal.memory_offset_hidden_features
                    ),
                    joint_embedding_features=(
                        config.temporal.memory_joint_embedding_features
                    ),
                    use_kpa=config.temporal.memory_use_kpa,
                    use_tpa=config.temporal.memory_use_tpa,
                    ktp_frames=config.temporal.video_frames,
                    ktp_causal=config.temporal.memory_ktp_causal,
                    ktp_global_init=config.temporal.memory_ktp_global_init,
                ),
            )
        else:
            self.mam = (
                None if memory_kind in {"none", ""} else MotionAlignedMembrane(
                config.num_joints, memory_kind,
                decay=config.temporal.memory_decay,
                decay_min=config.temporal.memory_decay_min,
                decay_max=config.temporal.memory_decay_max,
                shared_decay=config.temporal.memory_shared_decay,
                use_current_residual=config.temporal.memory_use_current_residual,
                )
            )
        self.last_mam_aux: dict[str, torch.Tensor] = {}
        self.early_classifiers = nn.ModuleDict({
            str(stage): nn.Conv2d(
                channels[stage], config.num_joints, kernel_size=1, bias=True,
            )
            for stage in config.temporal.early_classifier_stages
        })
        self._initialize()
        self._initialize_output()
        # The generic Conv2d initializer above intentionally serves the legacy
        # network. Re-apply V2 zero-init invariants afterwards so gamma=0 and
        # residual_offset=0 remain numerically baseline-preserving.
        if isinstance(self.mam, MotionAlignedMembraneV2):
            self.mam.reset_parameters()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _initialize_output(self) -> None:
        for classifier in self.early_classifiers.values():
            nn.init.normal_(classifier.weight, std=self.config.head.output_init_std)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)
        if self.config.head.output_init != "small_normal":
            return
        output_modules = getattr(self.head, "output_modules", None)
        if output_modules is None:
            output = getattr(self.head, "final", getattr(self.head, "output", None))
            output_modules = () if output is None else (output,)
        if not output_modules:
            raise ValueError("Configured small output initialization for a head without output")
        for output in output_modules:
            nn.init.normal_(output.weight, std=self.config.head.output_init_std)
            if output.bias is not None:
                nn.init.zeros_(output.bias)

    def _translate_previous(self, image: torch.Tensor) -> torch.Tensor:
        radius = self.config.temporal.translation_pixels
        batch, _, height, width = image.shape
        if self.training and self.config.temporal.randomize_training:
            shifts = torch.randint(
                -radius, radius + 1, (batch, 2), device=image.device,
            ).to(image.dtype)
        else:
            fixed = self.config.temporal.eval_translation or (radius, 0)
            shifts = image.new_tensor(fixed).reshape(1, 2).repeat(batch, 1)
        theta = image.new_zeros((batch, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        # affine_grid specifies sampling offsets, hence the negative sign makes
        # positive configured shifts move visible content right/down.
        theta[:, 0, 2] = -2.0 * shifts[:, 0] / max(width - 1, 1)
        theta[:, 1, 2] = -2.0 * shifts[:, 1] / max(height - 1, 1)
        grid = F.affine_grid(theta, image.shape, align_corners=True)
        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="reflection",
            align_corners=True,
        )

    def _make_sequence(self, image: torch.Tensor) -> torch.Tensor:
        temporal = self.config.temporal
        if temporal.input_strategy == "repeat":
            return image.unsqueeze(0).repeat(self.config.num_steps, 1, 1, 1, 1)
        if temporal.input_strategy == "previous_translate_current":
            if self.config.num_steps != 2:
                raise ValueError("previous_translate_current requires model.num_steps=2")
            return torch.stack((self._translate_previous(image), image), dim=0)
        raise ValueError(f"Unknown temporal input strategy: {temporal.input_strategy}")

    def _apply_history_control(self, image: torch.Tensor) -> torch.Tensor:
        mode = self.config.temporal.history_mode
        if mode == "real":
            return image
        if mode == "repeat_current":
            return image[:, -1:].expand_as(image)
        if mode == "reverse":
            return image.flip(1)
        if mode == "batch_shuffle_history":
            value = image.clone()
            if value.shape[0] > 1:
                value[:, :-1] = value[:, :-1].roll(1, dims=0)
            else:
                value[:, :-1] = 0
            return value
        if mode == "history_only":
            if image.shape[1] < 2:
                raise ValueError("history_only requires at least two input frames")
            # Remove the supervised current-frame pixels while retaining the same
            # four-step compute budget as the other cross-frame experiments.
            return torch.cat((image[:, :-1], image[:, -2:-1]), dim=1)
        raise ValueError(f"Unknown temporal history mode: {mode}")

    def _scheduled_sequence(self, image: torch.Tensor) -> torch.Tensor:
        """Expand physical frames into one continuous, causal SNN trajectory."""
        temporal = self.config.temporal
        if image.ndim != 5:
            raise ValueError("scheduled frame input requires B x V x C x H x W")
        if image.shape[1] != temporal.video_frames:
            raise ValueError(
                f"Expected {temporal.video_frames} physical frames, got {image.shape[1]}"
            )
        schedule = temporal.update_schedule
        if not schedule:
            raise ValueError("scheduled frame input has no update_schedule")
        indices = torch.tensor(schedule, device=image.device, dtype=torch.long)
        return image.index_select(1, indices).transpose(0, 1).contiguous()

    def _early_predictions(self, features) -> list[torch.Tensor]:
        predictions = []
        for stage in self.config.temporal.early_classifier_stages:
            value = features.features[stage]
            steps, batch, channels, height, width = value.shape
            value = self.early_classifiers[str(stage)](
                value.reshape(steps * batch, channels, height, width)
            )
            value = value.reshape(steps, batch, self.config.num_joints, height, width)
            value = value.mean(dim=0)
            if self.config.heatmap_size is not None and value.shape[-2:] != self.config.heatmap_size:
                value = F.interpolate(
                    value, size=self.config.heatmap_size, mode="bilinear",
                    align_corners=False,
                )
            predictions.append(value)
        return predictions

    def _forward_sequence(self, sequence: torch.Tensor,
                          return_intermediates: bool = False,
                          return_early_classifiers: bool = False):
        features = self.backbone(sequence)
        value = self.neck(features.features)
        prediction = self.head(value)
        if return_early_classifiers:
            return prediction, self._early_predictions(features)
        if return_intermediates and hasattr(self.head, "forward_with_intermediates"):
            return self.head.forward_with_intermediates(value)
        return prediction

    def _forward_decoupled_frames(self, image: torch.Tensor,
                                  return_intermediates: bool = False):
        """Run an independent SNN trajectory for every video frame.

        Video time and neuron time deliberately remain separate here.  Each
        frame is repeated for ``snn_steps_per_frame`` neuron updates, reduced
        to one frame-level feature, and only then passed to the causal video
        temporal head.
        """
        snn_steps = self.config.temporal.snn_steps_per_frame
        frame_features = []
        for frame in image.unbind(dim=1):
            sequence = frame.unsqueeze(0).repeat(snn_steps, 1, 1, 1, 1)
            features = self.backbone(sequence)
            value = self.neck(features.features)
            # Temporal-preserving necks return Ts x B x C x H x W; ordinary
            # necks already reduce neuron time and return B x C x H x W.
            if value.ndim == 5:
                value = value.mean(dim=0)
            frame_features.append(value)
        video_features = torch.stack(frame_features, dim=0)
        if return_intermediates and hasattr(self.head, "forward_with_intermediates"):
            return self.head.forward_with_intermediates(video_features)
        return self.head(video_features)

    def forward(self, image: torch.Tensor,
                return_intermediates: bool = False,
                return_early_classifiers: bool = False):
        if image.ndim == 5 and self.mam is not None:
            return self.forward_per_step(image)[-1]
        if image.ndim == 5:
            input_strategy = self.config.temporal.input_strategy
            if input_strategy not in {"frames", "scheduled_frames"}:
                raise ValueError(
                    "B x T x C x H x W input requires a frame input strategy"
                )
            expected_frames = (
                self.config.temporal.video_frames
                if input_strategy == "scheduled_frames" else self.config.num_steps
            )
            if image.shape[1] != expected_frames:
                raise ValueError(
                    f"Expected {expected_frames} input frames, got {image.shape[1]}"
                )
            image = self._apply_history_control(image)
            if input_strategy == "scheduled_frames":
                return self._forward_sequence(
                    self._scheduled_sequence(image), return_intermediates,
                    return_early_classifiers,
                )
            if (self.config.temporal.decouple_video_time or
                    self.config.temporal.snn_steps_per_frame > 1):
                if return_early_classifiers:
                    raise ValueError(
                        "decoupled frame/SNN time does not support early classifiers"
                    )
                return self._forward_decoupled_frames(image, return_intermediates)
            if self.config.temporal.state_mode == "reset":
                if self.config.temporal.stage_steps:
                    raise ValueError("state_mode=reset cannot use stage_steps")
                predictions = [
                    self._forward_sequence(
                        image[:, step:step + 1].transpose(0, 1).contiguous(),
                        return_intermediates=False,
                    )
                    for step in range(image.shape[1])
                ]
                return torch.stack(predictions, dim=0).mean(0)
            sequence = image.transpose(0, 1).contiguous()
        elif image.ndim == 4:
            if self.config.temporal.input_strategy in {"frames", "scheduled_frames"}:
                raise ValueError("frame input strategies require B x T x C x H x W input")
            sequence = self._make_sequence(image)
        else:
            raise ValueError("Expected B x C x H x W or B x T x C x H x W input")
        return self._forward_sequence(
            sequence, return_intermediates, return_early_classifiers,
        )

    def forward_per_step(self, image: torch.Tensor) -> torch.Tensor:
        """Return one heatmap per physical frame for state-carry/reset training."""
        if image.ndim != 5:
            raise ValueError("forward_per_step expects B x V x C x H x W")
        input_strategy = self.config.temporal.input_strategy
        if input_strategy not in {"frames", "scheduled_frames"}:
            raise ValueError(
                "forward_per_step requires a frame input strategy"
            )
        if self.mam is not None:
            if input_strategy != "frames":
                raise ValueError("MAM requires input_strategy=frames")
            spatial = self.forward_spatial_per_step(image)
            return self.forward_mam_per_step(spatial)
        if input_strategy == "scheduled_frames":
            image = self._apply_history_control(image)
            sequence = self._scheduled_sequence(image)
            features = self.backbone(sequence)
            value = self.neck(features.features)
            if value.ndim != 5:
                raise ValueError(
                    "Scheduled per-frame output requires a temporal-preserving neck"
                )
            schedule = self.config.temporal.update_schedule
            last_update_for_frame = {
                frame_index: update_index
                for update_index, frame_index in enumerate(schedule)
            }
            expected_frames = list(range(image.shape[1]))
            if sorted(last_update_for_frame) != expected_frames:
                raise ValueError(
                    "update_schedule must include every physical frame exactly in "
                    "causal order"
                )
            return torch.stack([
                self.head(value[last_update_for_frame[frame_index]])
                for frame_index in expected_frames
            ], dim=0)
        if self.config.temporal.state_mode == "reset":
            predictions = [
                self._forward_sequence(frame.unsqueeze(0))
                for frame in image.unbind(dim=1)
            ]
            return torch.stack(predictions, dim=0)
        sequence = image.transpose(0, 1).contiguous()
        features = self.backbone(sequence)
        value = self.neck(features.features)
        if value.ndim != 5:
            raise ValueError(
                "Per-step supervision requires a temporal-preserving neck"
            )
        return torch.stack([self.head(step) for step in value.unbind(dim=0)], dim=0)

    def forward_spatial_per_step(self, image: torch.Tensor) -> torch.Tensor:
        """Compute raw spatial heatmaps once for MAM diagnostic grids."""
        if image.ndim != 5:
            raise ValueError("forward_spatial_per_step expects B x V x C x H x W")
        if self.config.temporal.input_strategy != "frames":
            raise ValueError("forward_spatial_per_step requires input_strategy=frames")
        # Flatten video time into the batch whenever BatchNorm is frozen/eval.
        # This replaces 16 small spatial launches with one high-throughput
        # launch while preserving independent one-step SNN trajectories.
        batch_norm_training = any(
            isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
            for module in self.modules()
        )
        if not batch_norm_training:
            batch, steps, channels, height, width = image.shape
            flattened = image.reshape(batch * steps, channels, height, width)
            prediction = self._forward_sequence(flattened.unsqueeze(0))
            return prediction.reshape(
                batch, steps, *prediction.shape[1:],
            ).transpose(0, 1).contiguous()
        return torch.stack([
            self._forward_sequence(frame.unsqueeze(0))
            for frame in image.unbind(dim=1)
        ], dim=0)

    def forward_mam_per_step(
        self, spatial: torch.Tensor, *,
        gt_keypoints_heatmap: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        diagnostic: bool = False,
    ) -> torch.Tensor:
        """Apply only temporal memory to cached raw spatial heatmaps."""
        if self.mam is None:
            return spatial
        if isinstance(self.mam, MotionAlignedMembraneV2):
            result = self.mam.forward_sequence(
                spatial,
                gt_keypoints_heatmap=gt_keypoints_heatmap,
                visibility=visibility,
                diagnostic=diagnostic,
            )
        else:
            result = self.mam.forward_sequence(spatial)
        if isinstance(result, MAMV2Output):
            prediction = result.heatmap
            self.last_mam_aux = result.auxiliary()
        else:
            prediction, self.last_mam_aux = result
        self.last_mam_aux["spatial_heatmaps"] = spatial
        return prediction


def build_model(config: SpikePoseConfig | dict) -> SpikePose:
    if isinstance(config, dict):
        experiment_id = config.get("id", "spikepose")
        model_name = config.get("name", "SpikePose")
        model_config = config.get("model", {})
        family = model_config.get("family")
        if family == "pose_resnet":
            model = build_pose_resnet(
                int(model_config["depth"]), int(model_config["num_joints"]),
                model_config.get("input_color", "rgb"),
            )
            model.experiment_id = experiment_id
            model.model_name = model_name
            return model
        if family == "hrnet":
            model = build_hrnet(
                int(model_config["width"]), int(model_config["num_joints"]),
            )
            model.experiment_id = experiment_id
            model.model_name = model_name
            return model
        if family == "spikeyolo":
            from .baselines.spikeyolo import build_spikeyolo

            model = build_spikeyolo(model_config)
            model.experiment_id = experiment_id
            model.model_name = model_name
            return model
        if family == "spikformer":
            from .baselines.spikformer import build_spikformer

            model = build_spikformer(model_config)
            model.experiment_id = experiment_id
            model.model_name = model_name
            return model
        if family is not None and family != "spikepose":
            raise ValueError(f"Unknown model family: {family}")
        config = SpikePoseConfig.from_dict(config)
    else:
        experiment_id = "spikepose"
        model_name = "SpikePose"
    model = SpikePose(config)
    model.experiment_id = experiment_id
    model.model_name = model_name
    return model
