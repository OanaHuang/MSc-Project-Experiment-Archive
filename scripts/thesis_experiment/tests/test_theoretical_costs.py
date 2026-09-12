from __future__ import annotations

import torch
import torch.nn as nn

from spikepose_thesis.analysis import profile_end_to_end_flops
from spikepose_thesis.models.mam_v2 import MAMV2Config, MotionAlignedMembraneV2
from tools.profile_table3_theoretical_costs import _refiner_flops


class _MAM(nn.Module):
    def __init__(self, *, use_kpa: bool = False, use_tpa: bool = False) -> None:
        super().__init__()
        self.memory = MotionAlignedMembraneV2(
            4,
            MAMV2Config(
                mode="mam_v2", offset_hidden_features=8,
                joint_embedding_features=2, ktp_frames=4,
                use_kpa=use_kpa, use_tpa=use_tpa,
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.memory.forward_sequence(value).heatmap


def test_conv_transpose_counts_input_contributions_exactly() -> None:
    model = nn.ConvTranspose2d(2, 3, 3, bias=True)
    report = profile_end_to_end_flops(
        model, torch.zeros(1, 2, 4, 4), strict=True,
    )
    products = 1 * 2 * 4 * 4 * 3 * 3 * 3
    bias_adds = 1 * 3 * 6 * 6
    assert report["flops"] == 2 * products + bias_adds


def test_graph_priors_increase_complete_mam_cost() -> None:
    value = torch.zeros(4, 1, 4, 8, 8)
    reports = {
        key: profile_end_to_end_flops(_MAM(**options), value, strict=True)
        for key, options in {
            "core": {},
            "kpa": {"use_kpa": True},
            "tpa": {"use_tpa": True},
            "ktp": {"use_kpa": True, "use_tpa": True},
        }.items()
    }
    assert reports["kpa"]["flops"] > reports["core"]["flops"]
    assert reports["tpa"]["flops"] > reports["core"]["flops"]
    assert reports["ktp"]["flops"] > reports["kpa"]["flops"]
    assert reports["ktp"]["flops"] > reports["tpa"]["flops"]
    assert all(report["complete"] for report in reports.values())


def test_matmul_and_elementwise_are_both_counted() -> None:
    class Product(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value @ value.transpose(-1, -2) + 1.0

    value = torch.zeros(2, 3, 5)
    report = profile_end_to_end_flops(Product(), value, strict=True)
    product = 2 * 3 * 3 * (2 * 5)
    addition = 2 * 3 * 3
    assert report["flops"] == product + addition


def test_refiner_counts_respect_sequence_resets() -> None:
    lengths = (16, 16)
    transitions = 2 * 15
    assert _refiner_flops("shared_ema", lengths) == transitions * 97
    assert _refiner_flops("jointwise", lengths) == 2 * 48 + transitions * 112
    assert _refiner_flops("one_euro", lengths) == transitions * 552
    assert _refiner_flops("causal_sg", lengths) == 2 * 3_904
