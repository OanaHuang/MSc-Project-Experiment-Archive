from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spikepose_thesis.analysis import profile_theoretical_dataset
from spikepose_thesis.models.baselines import _spikeyolo_layers as spikeyolo
from spikepose_thesis.models.config import NeuronConfig
from spikepose_thesis.models.layers import SpikeConv
from tools.profile_fullvideo_theoretical_energy import _scale_cost_scope


class _SpikePoseToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = SpikeConv(
            1, 1, 1,
            NeuronConfig(decay=0.9, threshold=1.0, max_spikes=4),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image.transpose(0, 1))


class _SpikeYOLOToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = spikeyolo.MS_StandardConv(1, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image.transpose(0, 1))


class _ANNToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layer(image)


def test_dataset_energy_uses_weighted_real_integer_activity(tmp_path) -> None:
    model = _SpikePoseToy().eval()
    batches = [
        torch.ones(2, 1, 1, 2, 2),
        torch.full((1, 1, 1, 2, 2), 2.0),
    ]
    report = profile_theoretical_dataset(
        model, batches, tmp_path / "energy.json",
        video_steps=1,
    )

    assert report["profiled_batches"] == 2
    assert report["profiled_windows"] == 3
    assert report["analog_macs"] == 0
    assert report["spiking_dense_macs"] == pytest.approx(4.0)
    assert report["effective_sops"] == pytest.approx(16.0 / 3.0)
    assert report["operation_weighted_mean_integer_spikes"] == pytest.approx(
        4.0 / 3.0
    )
    assert report["operation_weighted_firing_rate"] == pytest.approx(1.0 / 3.0)
    assert report["spiking_ac_energy_mj"] == pytest.approx(
        (16.0 / 3.0) * 0.9 / 1e9
    )
    assert report["normalization"] == "per 1-frame window"


def test_spikeyolo_direct_spike_convolution_is_counted_as_ac(tmp_path) -> None:
    model = _SpikeYOLOToy().eval()
    report = profile_theoretical_dataset(
        model, [torch.ones(1, 1, 1, 4, 4)],
        tmp_path / "spikeyolo_energy.json", video_steps=1,
    )

    assert report["spiking_dense_macs"] > 0
    assert report["effective_sops"] > 0
    assert report["analog_macs"] == 0
    assert report["layers"][0]["operation"] == "AC"


def test_ann_energy_is_normalized_per_window(tmp_path) -> None:
    model = _ANNToy().eval()
    report = profile_theoretical_dataset(
        model, [torch.ones(3, 1, 2, 2)],
        tmp_path / "ann_energy.json", video_steps=1,
    )

    assert report["profiled_windows"] == 3
    assert report["analog_macs"] == pytest.approx(4.0)
    assert report["spiking_dense_macs"] == 0
    assert report["effective_sops"] == 0
    assert report["ann_architecture_macs"] == pytest.approx(4.0)
    assert report["total_theoretical_energy_mj"] == pytest.approx(4.6 * 4 / 1e9)


def test_framewise_cost_can_be_scaled_to_sixteen_frames() -> None:
    report = {
        "analog_macs": 2.0,
        "end_to_end_flops": 6.0,
        "end_to_end_gflops": 6e-9,
        "end_to_end_operator_flops": {"convolution": 6.0},
        "layers": [{
            "invocations_per_window": 1.0,
            "dense_macs": 2.0,
            "ann_architecture_macs": 2.0,
            "ann_equivalent_macs": 2.0,
            "legacy_single_step_ann_equivalent_macs": 2.0,
            "effective_operations": 2.0,
            "energy_mj": 9.2e-9,
        }],
    }

    scaled = _scale_cost_scope(report, 16)

    assert scaled["analog_macs"] == 32.0
    assert scaled["end_to_end_flops"] == 96.0
    assert scaled["end_to_end_operator_flops"]["convolution"] == 96.0
    assert scaled["layers"][0]["invocations_per_window"] == 16.0


@pytest.mark.parametrize("groups", [1, 2])
def test_transposed_convolution_energy_counts_input_scatter(tmp_path, groups) -> None:
    model = nn.ConvTranspose2d(4, 6, 4, stride=2, padding=1,
                               groups=groups, bias=False).eval()
    report = profile_theoretical_dataset(
        model, [torch.ones(2, 4, 8, 8)], tmp_path / "deconv_energy.json",
        video_steps=1,
    )
    # Each input scalar contributes to (out_channels / groups) * kernel_area
    # products. Output resolution is 16x16 but must not quadruple this count.
    expected_macs = 4 * 8 * 8 * (6 // groups) * 4 * 4
    assert report["analog_macs"] == expected_macs
    assert report["total_theoretical_energy_mj"] == pytest.approx(
        expected_macs * 4.6 / 1e9
    )
    assert report["end_to_end_flops"] == 2 * expected_macs
