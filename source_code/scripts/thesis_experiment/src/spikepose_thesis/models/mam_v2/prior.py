from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# The retained NTU protocol predicts the MPII-16 layout.  Keeping the topology
# next to the prior module avoids coupling the model package to a dataset
# implementation while making the joint order explicit.
MPII16_SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15),
)

NTU25_SKELETON_EDGES = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)


def _skeleton_edges(nodes: int) -> tuple[tuple[int, int], ...]:
    if nodes == 16:
        return MPII16_SKELETON_EDGES
    if nodes == 25:
        return NTU25_SKELETON_EDGES
    # Tiny synthetic layouts used by unit tests retain a valid chain prior.
    return tuple((index - 1, index) for index in range(1, nodes))


def _local_affinity(
    nodes: int, edges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    value = torch.eye(nodes, dtype=torch.float32)
    for left, right in edges:
        if not 0 <= left < nodes or not 0 <= right < nodes:
            raise ValueError(f"Topology edge {(left, right)} exceeds {nodes} nodes")
        value[left, right] = 1.0
        value[right, left] = 1.0
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1.0)


class PriorGraphLayer(nn.Module):
    """KTPFormer-style local-plus-global graph prior.

    The two feature projections distinguish self and non-self messages, as in
    the official KTPFormer implementation.  LayerNorm replaces its BatchNorm:
    temporal BatchNorm would aggregate future frames during training and would
    therefore violate the causal MAM contract.
    """

    def __init__(
        self,
        nodes: int,
        features: int,
        local_affinity: torch.Tensor,
        *,
        causal: bool = False,
        global_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if tuple(local_affinity.shape) != (nodes, nodes):
            raise ValueError("Prior local affinity has the wrong shape")
        self.nodes = int(nodes)
        self.features = int(features)
        self.causal = bool(causal)
        self.global_init = float(global_init)
        self.register_buffer("local_affinity", local_affinity.clone())
        self.global_affinity = nn.Parameter(torch.empty(nodes, nodes))
        self.weight = nn.Parameter(torch.empty(2, features, features))
        self.node_weight = nn.Parameter(torch.empty(nodes, features))
        self.bias = nn.Parameter(torch.empty(features))
        self.norm = nn.LayerNorm(features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.global_affinity, self.global_init)
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        nn.init.ones_(self.node_weight)
        bound = 1.0 / math.sqrt(max(self.features, 1))
        nn.init.uniform_(self.bias, -bound, bound)
        self.norm.reset_parameters()

    def affinity(self, length: int | None = None) -> torch.Tensor:
        length = self.nodes if length is None else int(length)
        if not 1 <= length <= self.nodes:
            raise ValueError(
                f"Prior sequence length {length} exceeds configured {self.nodes}"
            )
        local = self.local_affinity[:length, :length]
        learned = self.global_affinity[:length, :length]
        value = local + learned
        value = 0.5 * (value + value.transpose(0, 1))
        if self.causal:
            value = value * torch.ones_like(value).tril()
        # Learned global affinities are unconstrained, matching KTPFormer.  An
        # absolute row scale prevents a near-zero signed denominator.
        return value / value.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.features:
            raise ValueError("PriorGraphLayer expects B x N x C tokens")
        length = int(value.shape[1])
        affinity = self.affinity(length).to(dtype=value.dtype)
        identity = torch.eye(
            length, device=value.device, dtype=value.dtype,
        )
        self_features = value @ self.weight[0]
        other_features = value @ self.weight[1]
        node_weight = self.node_weight[:length].to(dtype=value.dtype)
        output = torch.matmul(
            affinity * identity, node_weight * self_features,
        ) + torch.matmul(
            affinity * (1.0 - identity), node_weight * other_features,
        )
        return self.norm(output + self.bias.to(dtype=value.dtype))


class KTPPriorEncoder(nn.Module):
    """Adapt KPA/TPA to MAM's per-frame, per-joint motion tokens.

    Input and output use ``T x B x J x C``. KPA mixes joints independently in
    each frame. TPA mixes each joint along time with two graph layers and a
    residual connection. When both are enabled the order is KPA then TPA.
    """

    def __init__(
        self,
        joints: int,
        features: int,
        frames: int,
        *,
        use_kpa: bool,
        use_tpa: bool,
        causal: bool = True,
        global_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if not use_kpa and not use_tpa:
            raise ValueError("KTPPriorEncoder requires KPA or TPA")
        if frames < 1:
            raise ValueError("KTP prior frame count must be positive")
        self.joints = int(joints)
        self.features = int(features)
        self.frames = int(frames)
        self.use_kpa = bool(use_kpa)
        self.use_tpa = bool(use_tpa)
        self.causal = bool(causal)
        spatial = _local_affinity(joints, _skeleton_edges(joints))
        temporal = _local_affinity(
            frames, tuple((index - 1, index) for index in range(1, frames)),
        )
        self.kpa = (
            PriorGraphLayer(
                joints, features, spatial, causal=False,
                global_init=global_init,
            )
            if use_kpa else None
        )
        self.tpa = (
            nn.ModuleList((
                PriorGraphLayer(
                    frames, features, temporal, causal=causal,
                    global_init=global_init,
                ),
                PriorGraphLayer(
                    frames, features, temporal, causal=causal,
                    global_init=global_init,
                ),
            ))
            if use_tpa else None
        )

    def reset_parameters(self) -> None:
        if self.kpa is not None:
            self.kpa.reset_parameters()
        if self.tpa is not None:
            for layer in self.tpa:
                layer.reset_parameters()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("KTPPriorEncoder expects T x B x J x C tokens")
        steps, batch, joints, features = value.shape
        if joints != self.joints or features != self.features:
            raise ValueError("KTP prior token shape does not match its configuration")
        if steps > self.frames:
            raise ValueError(
                f"KTP prior received {steps} frames, configured for {self.frames}"
            )
        output = value
        if self.kpa is not None:
            spatial = output.reshape(steps * batch, joints, features)
            spatial = spatial + F.relu(self.kpa(spatial))
            output = spatial.reshape(steps, batch, joints, features)
        if self.tpa is not None:
            temporal = output.permute(1, 2, 0, 3).reshape(
                batch * joints, steps, features,
            )
            residual = temporal
            temporal = F.relu(self.tpa[0](temporal))
            temporal = F.relu(self.tpa[1](temporal))
            temporal = temporal + residual
            output = temporal.reshape(
                batch, joints, steps, features,
            ).permute(2, 0, 1, 3).contiguous()
        return output
