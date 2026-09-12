from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior import KTPPriorEncoder


_MODES = {
    "reset",
    "carry_fixed",
    "plif_low",
    "gt_align_oracle",
    "pred_align_static",
    "mam_v2",
    "mam_v2_noalign",
    "mam_v2_predictive",
    "mam_v2_temporal",
}


@dataclass(frozen=True)
class MAMV2Config:
    """Explicit MAM V2 configuration with no hidden temporal constants."""

    mode: str = "mam_v2"
    decay_min: float = 0.02
    decay_max: float = 0.95
    decay_init: float = 0.08
    residual_gamma_init: float = 0.0
    max_residual_offset_px: float = 2.0
    align_corners: bool = False
    padding_mode: str = "border"
    use_current_direct_path: bool = True
    use_motion_token: bool = True
    use_residual_offset: bool = True
    use_dynamic_gate: bool = True
    use_memory_update: bool = True
    use_predictive_head: bool = False
    shared_decay: bool = False
    bptt_mode: str = "full16"
    detach_interval: int = 0
    offset_hidden_features: int = 64
    joint_embedding_features: int = 8
    use_kpa: bool = False
    use_tpa: bool = False
    ktp_frames: int = 16
    ktp_causal: bool = True
    ktp_global_init: float = 1e-6

    def __post_init__(self) -> None:
        mode = self.mode.lower()
        if mode not in _MODES:
            raise ValueError(f"Unknown MAM V2 mode: {self.mode}")
        if not 0.0 <= self.decay_min < self.decay_max <= 1.0:
            raise ValueError("MAM V2 decay bounds must satisfy 0 <= min < max <= 1")
        if not self.decay_min <= self.decay_init <= self.decay_max:
            raise ValueError("MAM V2 decay_init must lie inside the configured range")
        if not 0.0 <= self.residual_gamma_init <= 1.0:
            raise ValueError("MAM V2 residual_gamma_init must lie in [0, 1]")
        if self.max_residual_offset_px < 0.0:
            raise ValueError("max_residual_offset_px must be non-negative")
        if self.padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("Unsupported grid_sample padding mode")
        if self.bptt_mode not in {"full16", "tbptt4", "detach1"}:
            raise ValueError("bptt_mode must be full16, tbptt4, or detach1")
        if self.detach_interval < 0:
            raise ValueError("detach_interval must be non-negative")
        if self.ktp_frames < 1:
            raise ValueError("ktp_frames must be positive")
        if self.ktp_global_init < 0.0:
            raise ValueError("ktp_global_init must be non-negative")
        if not self.use_motion_token and (self.use_kpa or self.use_tpa):
            raise ValueError("KPA/TPA require MAM motion tokens")
        if not self.use_motion_token and self.use_residual_offset:
            raise ValueError("The residual offset head requires MAM motion tokens")


@dataclass
class MAMV2Output:
    heatmap: torch.Tensor
    raw_heatmap: torch.Tensor
    state: torch.Tensor
    aligned_state: torch.Tensor
    coarse_offset: torch.Tensor
    residual_offset: torch.Tensor
    final_offset: torch.Tensor
    decay: torch.Tensor
    gamma: torch.Tensor
    confidence: torch.Tensor
    predictive_heatmap: torch.Tensor | None = None

    def auxiliary(self) -> dict[str, torch.Tensor]:
        """Expose both V2 diagnostics and legacy loss aliases."""
        result = {
            "raw_heatmap": self.raw_heatmap,
            "state": self.state,
            "aligned_state": self.aligned_state,
            "coarse_offset": self.coarse_offset,
            "residual_offset": self.residual_offset,
            "final_offset": self.final_offset,
            "offset": self.final_offset,
            "decay": self.decay,
            "gamma": self.gamma,
            "confidence": self.confidence,
        }
        if self.predictive_heatmap is not None:
            result["prediction_next"] = self.predictive_heatmap
        return result


class MotionAlignedMembraneV2(nn.Module):
    """Baseline-preserving, causal heatmap-level temporal memory.

    The current heatmap always has a direct output path. History enters only
    through a per-joint residual coefficient ``gamma`` which is exactly zero
    at initialization by default. State is local to ``forward_sequence``.
    """

    def __init__(self, joints: int, config: MAMV2Config) -> None:
        super().__init__()
        self.joints = int(joints)
        self.config = config
        self.mode = config.mode.lower()
        decay_channels = 1 if config.shared_decay else self.joints

        self.decay_logit = nn.Parameter(torch.empty(1, decay_channels, 1, 1))
        self.gamma_parameter = nn.Parameter(torch.empty(1, self.joints, 1, 1))
        self.joint_embedding = nn.Embedding(
            self.joints, config.joint_embedding_features,
        )
        offset_features = 9 + config.joint_embedding_features
        self.motion_feature_dim = offset_features
        self.ktp_prior = (
            KTPPriorEncoder(
                self.joints,
                offset_features,
                config.ktp_frames,
                use_kpa=config.use_kpa,
                use_tpa=config.use_tpa,
                causal=config.ktp_causal,
                global_init=config.ktp_global_init,
            )
            if config.use_kpa or config.use_tpa else None
        )
        self.offset_head = nn.Sequential(
            nn.Linear(offset_features, config.offset_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(config.offset_hidden_features, 2),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(offset_features, config.offset_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(config.offset_hidden_features, 1),
        )
        self.predictor = nn.Conv2d(self.joints, self.joints, 3, padding=1)
        self.reset_parameters()

    @property
    def predictive(self) -> bool:
        return self.config.use_predictive_head or self.mode in {
            "mam_v2_predictive", "mam_v2_temporal",
        }

    @property
    def aligned(self) -> bool:
        return self.mode in {
            "gt_align_oracle", "pred_align_static", "mam_v2",
            "mam_v2_predictive", "mam_v2_temporal",
        }

    @property
    def adaptive(self) -> bool:
        return self.config.use_dynamic_gate and self.mode in {
            "mam_v2", "mam_v2_noalign", "mam_v2_predictive",
            "mam_v2_temporal",
        }

    def reset_parameters(self) -> None:
        """Restore the numerical invariants required by the V2 design."""
        ratio = (
            (self.config.decay_init - self.config.decay_min)
            / (self.config.decay_max - self.config.decay_min)
        )
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        decay_logit = math.log(ratio / (1.0 - ratio))
        nn.init.constant_(self.decay_logit, decay_logit)
        nn.init.constant_(self.gamma_parameter, self.config.residual_gamma_init)
        nn.init.normal_(self.joint_embedding.weight, std=0.02)
        for module in (self.offset_head[0], self.gate_head[0]):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Zero final offset layer makes the initial predicted warp coarse-only.
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        # A constant gate starts exactly at decay_init.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, decay_logit)
        nn.init.zeros_(self.predictor.bias)
        nn.init.kaiming_normal_(self.predictor.weight, mode="fan_out")
        if self.ktp_prior is not None:
            self.ktp_prior.reset_parameters()

    @staticmethod
    def _soft_coordinates_and_confidence(
        heatmap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, joints, height, width = heatmap.shape
        probability = heatmap.flatten(2).softmax(-1)
        y, x = torch.meshgrid(
            torch.arange(height, device=heatmap.device, dtype=heatmap.dtype),
            torch.arange(width, device=heatmap.device, dtype=heatmap.dtype),
            indexing="ij",
        )
        grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
        coordinate = probability @ grid
        confidence = probability.amax(-1, keepdim=True)
        return coordinate, confidence

    def _motion_features(
        self,
        current_xy: torch.Tensor,
        previous_xy: torch.Tensor,
        current_confidence: torch.Tensor,
        previous_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = current_xy - previous_xy
        magnitude = coarse.norm(dim=-1, keepdim=True)
        joint_ids = torch.arange(self.joints, device=current_xy.device)
        embedding = self.joint_embedding(joint_ids).unsqueeze(0).expand(
            current_xy.shape[0], -1, -1,
        )
        features = torch.cat((
            current_xy, previous_xy,
            current_confidence, previous_confidence,
            coarse, magnitude, embedding,
        ), dim=-1)
        return coarse, features

    def _sequence_motion_features(
        self, heatmaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build causal 17-D motion tokens and apply optional KTP priors."""
        coordinates = []
        confidences = []
        for current in heatmaps.unbind(0):
            coordinate, confidence = self._soft_coordinates_and_confidence(current)
            coordinates.append(coordinate)
            confidences.append(confidence)
        coordinate_sequence = torch.stack(coordinates)
        confidence_sequence = torch.stack(confidences)
        coarse_offsets = []
        motion_features = []
        for step in range(len(coordinates)):
            previous_step = max(step - 1, 0)
            coarse, features = self._motion_features(
                coordinate_sequence[step], coordinate_sequence[previous_step],
                confidence_sequence[step], confidence_sequence[previous_step],
            )
            coarse_offsets.append(coarse)
            motion_features.append(features)
        coarse_sequence = torch.stack(coarse_offsets)
        feature_sequence = torch.stack(motion_features)
        if self.ktp_prior is not None:
            feature_sequence = self.ktp_prior(feature_sequence)
        if not self.config.use_motion_token:
            # Keep the coarse coordinate displacement available to the
            # alignment path, but remove the learned 17-D motion-token input
            # from every head.  Ablation configs also disable offset_head;
            # gate_head remains present and learns only its neutral-input bias.
            feature_sequence = torch.zeros_like(feature_sequence)
        return (
            coordinate_sequence,
            confidence_sequence,
            coarse_sequence,
            feature_sequence,
        )

    def _static_decay(self, reference: torch.Tensor) -> torch.Tensor:
        value = self.config.decay_min + (
            self.config.decay_max - self.config.decay_min
        ) * self.decay_logit.sigmoid()
        return value.expand(reference.shape[0], self.joints, 1, 1)

    def _decay(self, features: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self.mode == "reset":
            return reference.new_zeros((reference.shape[0], self.joints, 1, 1))
        if self.mode in {"carry_fixed", "gt_align_oracle"}:
            return reference.new_full(
                (reference.shape[0], self.joints, 1, 1),
                self.config.decay_init,
            )
        if not self.adaptive:
            return self._static_decay(reference)
        raw = self.gate_head(features).unsqueeze(-1)
        if self.config.shared_decay:
            raw = raw.mean(dim=1, keepdim=True).expand(-1, self.joints, -1, -1)
        return self.config.decay_min + (
            self.config.decay_max - self.config.decay_min
        ) * raw.sigmoid()

    def _warp(self, value: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        batch, joints, height, width = value.shape
        if self.config.align_corners:
            ys = torch.linspace(-1, 1, height, device=value.device, dtype=value.dtype)
            xs = torch.linspace(-1, 1, width, device=value.device, dtype=value.dtype)
        else:
            ys = (
                2.0 * (torch.arange(height, device=value.device, dtype=value.dtype) + 0.5)
                / height - 1.0
            )
            xs = (
                2.0 * (torch.arange(width, device=value.device, dtype=value.dtype) + 0.5)
                / width - 1.0
            )
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((x, y), dim=-1).reshape(1, 1, height, width, 2)
        shift = displacement.clone()
        if self.config.align_corners:
            shift[..., 0] *= 2.0 / max(width - 1, 1)
            shift[..., 1] *= 2.0 / max(height - 1, 1)
        else:
            shift[..., 0] *= 2.0 / max(width, 1)
            shift[..., 1] *= 2.0 / max(height, 1)
        grid = grid - shift.reshape(batch, joints, 1, 1, 2)
        warped = F.grid_sample(
            value.reshape(batch * joints, 1, height, width),
            grid.reshape(batch * joints, height, width, 2),
            mode="bilinear",
            padding_mode=self.config.padding_mode,
            align_corners=self.config.align_corners,
        )
        return warped.reshape(batch, joints, height, width)

    def _detach_due(self, step: int) -> bool:
        if self.config.bptt_mode == "detach1":
            return True
        interval = self.config.detach_interval
        if self.config.bptt_mode == "tbptt4" and interval == 0:
            interval = 4
        return interval > 0 and (step + 1) % interval == 0

    def forward_sequence(
        self,
        heatmaps: torch.Tensor,
        *,
        gt_keypoints_heatmap: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        diagnostic: bool = False,
    ) -> MAMV2Output:
        if heatmaps.ndim != 5:
            raise ValueError("MAM V2 expects T x B x J x H x W heatmaps")
        if heatmaps.shape[2] != self.joints:
            raise ValueError("MAM V2 heatmap joint dimension does not match config")
        if self.mode == "gt_align_oracle":
            if self.training or not diagnostic:
                raise RuntimeError(
                    "GT alignment oracle is eval-only and requires diagnostic=True"
                )
            if gt_keypoints_heatmap is None:
                raise ValueError("GT alignment oracle requires heatmap-space keypoints")
            if gt_keypoints_heatmap.shape[:3] != heatmaps.shape[:3] or (
                gt_keypoints_heatmap.shape[-1] != 2
            ):
                raise ValueError("GT keypoints must have shape T x B x J x 2")
            if visibility is not None and visibility.shape != heatmaps.shape[:3]:
                raise ValueError("Oracle visibility must have shape T x B x J")

        (
            coordinate_sequence,
            confidence_sequence,
            coarse_sequence,
            feature_sequence,
        ) = self._sequence_motion_features(heatmaps)
        memory = None
        outputs: dict[str, list[torch.Tensor]] = {
            "heatmap": [], "state": [], "aligned_state": [],
            "coarse_offset": [], "residual_offset": [], "final_offset": [],
            "decay": [], "gamma": [], "confidence": [], "predictive": [],
        }
        for step, current in enumerate(heatmaps.unbind(0)):
            current_xy = coordinate_sequence[step]
            confidence = confidence_sequence[step]
            empty_offset = current.new_zeros((current.shape[0], self.joints, 2))
            empty_decay = current.new_zeros((current.shape[0], self.joints, 1, 1))
            if memory is None or self.mode == "reset":
                aligned = current
                coarse = residual = final_offset = empty_offset
                decay = empty_decay
                state = current
                memory = current
            else:
                coarse = coarse_sequence[step]
                features = feature_sequence[step]
                if self.mode == "gt_align_oracle":
                    oracle = gt_keypoints_heatmap[step] - gt_keypoints_heatmap[step - 1]
                    if visibility is not None:
                        valid = (visibility[step] > 0) & (visibility[step - 1] > 0)
                        oracle = torch.where(valid.unsqueeze(-1), oracle, coarse)
                    coarse = oracle
                residual = empty_offset
                if self.config.use_residual_offset and self.mode in {
                    "pred_align_static", "mam_v2", "mam_v2_predictive",
                    "mam_v2_temporal",
                }:
                    residual = self.config.max_residual_offset_px * torch.tanh(
                        self.offset_head(features)
                    )
                final_offset = coarse + residual if self.aligned else empty_offset
                aligned = self._warp(memory, final_offset) if self.aligned else memory
                decay = self._decay(features, current)
                state = (1.0 - decay) * current + decay * aligned
                # The no-memory-update ablation retains the one-step aligned
                # candidate used by residual fusion, but does not write that
                # candidate back into recurrent memory.  The next step starts
                # from the preceding raw heatmap instead.
                memory = state if self.config.use_memory_update else current

            gamma = self.gamma_parameter.clamp(0.0, 1.0).expand(
                current.shape[0], self.joints, 1, 1,
            )
            output = (
                current + gamma * (state - current)
                if self.config.use_current_direct_path else state
            )
            outputs["heatmap"].append(output)
            outputs["state"].append(state)
            outputs["aligned_state"].append(aligned)
            outputs["coarse_offset"].append(coarse)
            outputs["residual_offset"].append(residual)
            outputs["final_offset"].append(final_offset)
            outputs["decay"].append(decay)
            outputs["gamma"].append(gamma)
            outputs["confidence"].append(confidence)
            # Predictive supervision is training-only; it is excluded from
            # inference latency and never feeds future GT into the state.
            if self.predictive and self.training:
                outputs["predictive"].append(self.predictor(state))
            if self._detach_due(step):
                memory = memory.detach()

        stack = lambda name: torch.stack(outputs[name])
        predictive = stack("predictive") if outputs["predictive"] else None
        return MAMV2Output(
            heatmap=stack("heatmap"),
            raw_heatmap=heatmaps,
            state=stack("state"),
            aligned_state=stack("aligned_state"),
            coarse_offset=stack("coarse_offset"),
            residual_offset=stack("residual_offset"),
            final_offset=stack("final_offset"),
            decay=stack("decay"),
            gamma=stack("gamma"),
            confidence=stack("confidence"),
            predictive_heatmap=predictive,
        )
