from __future__ import annotations

import math

import torch
import torch.nn as nn

from scripts.spikepose.analysis.theoretical import MAC_ENERGY_PJ, profile_theoretical
from scripts.spikepose.models.layers import TimeConvBN


class _TemporalDenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = TimeConvBN(1, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.temporal(value)


def test_ann_architecture_energy_uses_complete_temporal_input(tmp_path) -> None:
    model = _TemporalDenseModel()
    image = torch.ones(4, 1, 1, 2, 2)

    report = profile_theoretical(model, image, tmp_path / "energy.json")

    expected_macs = 4 * 1 * 1 * 2 * 2
    expected_energy_mj = expected_macs * MAC_ENERGY_PJ / 1e9
    assert report["dense_macs"] == expected_macs
    assert report["ann_architecture_macs"] == expected_macs
    assert math.isclose(report["ann_architecture_energy_mj"], expected_energy_mj)
    assert math.isclose(report["ann_equivalent_energy_mj"], expected_energy_mj)
    assert math.isclose(report["theoretical_energy_efficiency_ratio"], 1.0)
