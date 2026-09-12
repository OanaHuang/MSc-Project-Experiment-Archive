from __future__ import annotations

import torch
from torch import nn

from scripts.NTU_RGBD.core.config import NTU_SKELETON_EDGES
from scripts.spikepose.models.neurons.ilif import IntegerSpike


def pose_features(pose: torch.Tensor, confidence: torch.Tensor,
                  observed: torch.Tensor) -> torch.Tensor:
    masked = pose * observed[..., None]
    velocity = torch.zeros_like(masked)
    velocity[:, 1:] = masked[:, 1:] - masked[:, :-1]
    return torch.cat((masked, velocity, confidence[..., None], observed[..., None]), dim=-1)


class IdentityRefiner(nn.Module):
    def forward(self, pose, confidence, observed):
        return pose


class LastObservationRefiner(nn.Module):
    def forward(self, pose, confidence, observed):
        output = pose.clone()
        for step in range(1, pose.shape[1]):
            output[:, step] = torch.where(
                observed[:, step, :, None].bool(), pose[:, step], output[:, step - 1],
            )
        return output


class ConfidenceKalmanRefiner(nn.Module):
    """Differentiation-free causal alpha filter with confidence-weighted updates."""
    def forward(self, pose, confidence, observed):
        output = pose.clone()
        state = pose[:, 0]
        for step in range(1, pose.shape[1]):
            gain = (confidence[:, step] * observed[:, step]).clamp(0.0, 1.0)[..., None]
            state = state + gain * (pose[:, step] - state)
            output[:, step] = state
        return output


class GRURefiner(nn.Module):
    def __init__(self, joints: int = 25, hidden_size: int = 128) -> None:
        super().__init__()
        self.joints = joints
        self.input = nn.Linear(6, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, pose, confidence, observed):
        batch, steps, joints, _ = pose.shape
        value = self.input(pose_features(pose, confidence, observed))
        value = value.permute(0, 2, 1, 3).reshape(batch * joints, steps, -1)
        value, _ = self.gru(value)
        residual = self.output(value).reshape(batch, joints, steps, 2).permute(0, 2, 1, 3)
        gate = observed[..., None]
        return gate * pose + (1.0 - gate) * (pose + residual)


class SpikingRefiner(nn.Module):
    def __init__(self, joints: int = 25, hidden_size: int = 128,
                 dynamic_decay: bool = False, graph: bool = False) -> None:
        super().__init__()
        self.joints = joints
        self.hidden_size = hidden_size
        self.dynamic_decay = dynamic_decay
        self.graph = graph
        self.input = nn.Linear(6, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, 2)
        self.decay_logit = nn.Parameter(torch.full((joints, hidden_size), 2.2))
        self.decay_gate = nn.Linear(3, hidden_size) if dynamic_decay else None
        if graph:
            adjacency = torch.eye(joints)
            for left, right in NTU_SKELETON_EDGES:
                adjacency[left, right] = adjacency[right, left] = 1.0
            adjacency /= adjacency.sum(-1, keepdim=True)
            self.register_buffer("adjacency", adjacency)
            self.graph_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, pose, confidence, observed):
        features = pose_features(pose, confidence, observed)
        batch, steps, joints, _ = features.shape
        membrane = features.new_zeros((batch, joints, self.hidden_size))
        previous = torch.zeros_like(membrane)
        outputs = []
        for step in range(steps):
            current = self.input(features[:, step]) + self.recurrent(previous)
            if self.graph:
                neighbours = torch.einsum("ij,bjh->bih", self.adjacency, previous)
                current = current + self.graph_projection(neighbours)
            decay = 0.5 + 0.49 * self.decay_logit.sigmoid()[None]
            if self.dynamic_decay:
                speed = features[:, step, :, 2:4].square().sum(-1, keepdim=True).sqrt()
                controls = torch.cat((confidence[:, step, :, None], observed[:, step, :, None], speed), -1)
                decay = 0.5 + 0.49 * self.decay_gate(controls).sigmoid()
            membrane = membrane * decay + current - previous.detach()
            previous = IntegerSpike.apply(membrane, 4)
            residual = self.output(membrane)
            gate = observed[:, step, :, None]
            outputs.append(gate * pose[:, step] + (1.0 - gate) * (pose[:, step] + residual))
        return torch.stack(outputs, dim=1)


def build_refiner(kind: str, joints: int = 25, hidden_size: int = 128) -> nn.Module:
    options = {
        "identity": lambda: IdentityRefiner(),
        "last_observation": lambda: LastObservationRefiner(),
        "kalman": lambda: ConfidenceKalmanRefiner(),
        "gru": lambda: GRURefiner(joints, hidden_size),
        "snn": lambda: SpikingRefiner(joints, hidden_size),
        "confidence_snn": lambda: SpikingRefiner(joints, hidden_size, dynamic_decay=True),
        "confidence_graph_snn": lambda: SpikingRefiner(
            joints, hidden_size, dynamic_decay=True, graph=True,
        ),
    }
    if kind not in options:
        raise KeyError(f"unknown occlusion refiner: {kind}")
    return options[kind]()
