from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ..models.layers import SpikeConv, SpikePoseDownsample, TimeConvBN
from ..models.neurons import MultiStepILIF


MAC_ENERGY_PJ = 4.6
AC_ENERGY_PJ = 0.9


def _operation_maps(model: nn.Module) -> tuple[dict[int, MultiStepILIF], set[int]]:
    """Map physical convolutions to their spike source and temporal execution."""
    spike_sources: dict[int, MultiStepILIF] = {}
    temporal_convolutions: set[int] = set()
    for module in model.modules():
        if isinstance(module, TimeConvBN):
            temporal_convolutions.add(id(module.conv))
        elif isinstance(module, SpikeConv):
            temporal_convolutions.add(id(module.conv.conv))
            if isinstance(module.neuron, MultiStepILIF):
                spike_sources[id(module.conv.conv)] = module.neuron
        elif isinstance(module, SpikePoseDownsample):
            temporal_convolutions.add(id(module.conv.conv))
            if isinstance(module.neuron, MultiStepILIF):
                spike_sources[id(module.conv.conv)] = module.neuron
    return spike_sources, temporal_convolutions


@torch.no_grad()
def profile_theoretical(model: nn.Module, image: torch.Tensor,
                        output_path: Path) -> dict:
    """Estimate compute energy with the layer-wise SpikeYOLO convention."""
    spike_sources, temporal_convolutions = _operation_maps(model)
    config = getattr(model, "config", None)
    video_steps = int(getattr(config, "num_steps", 1))
    snn_steps = int(getattr(getattr(config, "temporal", None),
                            "snn_steps_per_frame", 1))
    steps = video_steps * snn_steps
    layers: list[dict] = []
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue

        def count_hook(layer, inputs, output, layer_name=name):
            if isinstance(layer, nn.Linear):
                macs = int(output.numel() * layer.in_features)
            else:
                kernel_area = 1
                for size in layer.kernel_size:
                    kernel_area *= int(size)
                macs = int(output.numel() * (layer.in_channels // layer.groups)
                           * kernel_area)
            source = spike_sources.get(id(layer))
            temporal = id(layer) in temporal_convolutions
            # Full-input ANN architecture baseline: use every dense MAC executed
            # for the actual image/clip.  The previous temporal ``/ steps``
            # value is retained separately for audit only; it is not comparable
            # with the full-input SNN energy in the efficiency ratio.
            ann_architecture_macs = float(macs)
            legacy_single_step_ann_equivalent_macs = (
                macs / steps if temporal else float(macs)
            )

            if source is None:
                operation = "MAC"
                mean_integer_spikes = None
                normalized_firing_rate = None
                effective_operations = float(macs)
                energy_pj = effective_operations * MAC_ENERGY_PJ
            else:
                operation = "AC"
                values = inputs[0].detach()
                mean_integer_spikes = float(values.sum() / values.numel())
                normalized_firing_rate = mean_integer_spikes / source.max_spikes
                effective_operations = macs * mean_integer_spikes
                energy_pj = effective_operations * AC_ENERGY_PJ

            layers.append({
                "name": layer_name,
                "operation": operation,
                "temporal": temporal,
                "dense_macs": macs,
                "ann_architecture_macs": ann_architecture_macs,
                "ann_equivalent_macs": ann_architecture_macs,
                "legacy_single_step_ann_equivalent_macs": (
                    legacy_single_step_ann_equivalent_macs
                ),
                "max_spikes": source.max_spikes if source is not None else None,
                "mean_integer_spikes": mean_integer_spikes,
                "normalized_firing_rate": normalized_firing_rate,
                "effective_operations": effective_operations,
                "energy_mj": energy_pj / 1e9,
            })

        handles.append(module.register_forward_hook(count_hook))

    model.eval()
    try:
        model(image)
    finally:
        for handle in handles:
            handle.remove()

    ann_macs = sum(item["dense_macs"] for item in layers
                   if item["operation"] == "MAC")
    spiking_dense_macs = sum(item["dense_macs"] for item in layers
                             if item["operation"] == "AC")
    effective_sops = sum(item["effective_operations"] for item in layers
                         if item["operation"] == "AC")
    analog_energy_pj = ann_macs * MAC_ENERGY_PJ
    spike_energy_pj = effective_sops * AC_ENERGY_PJ
    total_energy_pj = analog_energy_pj + spike_energy_pj
    ann_architecture_macs = sum(item["ann_architecture_macs"] for item in layers)
    ann_architecture_energy_pj = ann_architecture_macs * MAC_ENERGY_PJ
    legacy_single_step_ann_equivalent_macs = sum(
        item["legacy_single_step_ann_equivalent_macs"] for item in layers
    )
    weighted_firing_rate = (
        effective_sops / sum(
            item["dense_macs"] * item["max_spikes"] for item in layers
            if item["operation"] == "AC"
        ) if spiking_dense_macs else 0.0
    )

    report = {
        "experiment_id": getattr(model, "experiment_id", "spikepose"),
        "model_name": getattr(model, "model_name", "SpikePose"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "time_steps": steps,
        "video_time_steps": video_steps,
        "snn_steps_per_frame": snn_steps,
        "analog_macs": ann_macs,
        "spiking_dense_macs": spiking_dense_macs,
        "dense_macs": ann_macs + spiking_dense_macs,
        "dense_flops": 2 * (ann_macs + spiking_dense_macs),
        "operation_weighted_firing_rate": weighted_firing_rate,
        "effective_sops": effective_sops,
        "analog_mac_energy_mj": analog_energy_pj / 1e9,
        "spiking_ac_energy_mj": spike_energy_pj / 1e9,
        "total_theoretical_energy_mj": total_energy_pj / 1e9,
        "ann_architecture_macs": ann_architecture_macs,
        "ann_architecture_energy_mj": ann_architecture_energy_pj / 1e9,
        # Backward-compatible aliases now use the same full-input scope.
        "ann_equivalent_macs": ann_architecture_macs,
        "ann_equivalent_energy_mj": ann_architecture_energy_pj / 1e9,
        "legacy_single_step_ann_equivalent_macs": (
            legacy_single_step_ann_equivalent_macs
        ),
        "legacy_single_step_ann_equivalent_energy_mj": (
            legacy_single_step_ann_equivalent_macs * MAC_ENERGY_PJ / 1e9
        ),
        "architecture_energy_efficiency_ratio": (
            ann_architecture_energy_pj / total_energy_pj
            if total_energy_pj else None
        ),
        "theoretical_energy_efficiency_ratio": (
            ann_architecture_energy_pj / total_energy_pj
            if total_energy_pj else None
        ),
        "input_shape": list(image.shape),
        "layers": layers,
        "assumptions": {
            "method": "layer-wise SpikeYOLO analytical compute energy",
            "technology": "45 nm",
            "precision": "FP32",
            "fp32_mac_pj": MAC_ENERGY_PJ,
            "accumulate_pj": AC_ENERGY_PJ,
            "flops_per_mac": 2,
            "scope": "arithmetic operations only",
            "input_scope": "complete supplied image or video clip",
            "ann_baseline": "all dense MACs over the complete supplied input",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
