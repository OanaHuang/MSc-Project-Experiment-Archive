from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SpikePoseMSBackbone
from .config import SpikePoseConfig
from .heads import build_head
from .necks import build_neck
from .baselines import build_hrnet, build_pose_resnet


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
        self.early_classifiers = nn.ModuleDict({
            str(stage): nn.Conv2d(
                channels[stage], config.num_joints, kernel_size=1, bias=True,
            )
            for stage in config.temporal.early_classifier_stages
        })
        self._initialize()
        self._initialize_output()

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
        if image.ndim == 5:
            if self.config.temporal.input_strategy != "frames":
                raise ValueError("B x T x C x H x W input requires input_strategy=frames")
            if image.shape[1] != self.config.num_steps:
                raise ValueError(
                    f"Expected {self.config.num_steps} input frames, got {image.shape[1]}"
                )
            image = self._apply_history_control(image)
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
            if self.config.temporal.input_strategy == "frames":
                raise ValueError("input_strategy=frames requires B x T x C x H x W input")
            sequence = self._make_sequence(image)
        else:
            raise ValueError("Expected B x C x H x W or B x T x C x H x W input")
        return self._forward_sequence(
            sequence, return_intermediates, return_early_classifiers,
        )


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
