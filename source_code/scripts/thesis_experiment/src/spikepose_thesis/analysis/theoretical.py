from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from ..models.layers import SpikeConv, SpikePoseDownsample, TimeConvBN
from ..models.neurons import MultiStepILIF


MAC_ENERGY_PJ = 4.6
AC_ENERGY_PJ = 0.9


END_TO_END_FLOP_ASSUMPTIONS = {
    "scope": "complete inference arithmetic over the supplied tensor",
    "flops_per_mac": 2,
    "elementwise_arithmetic": "one FLOP per output scalar",
    "elementary_functions": (
        "exp, sqrt, reciprocal, sigmoid and tanh each count as one scalar FLOP"
    ),
    "softmax": "max, subtraction, exponential, sum and division are counted",
    "normalization": "inference arithmetic, including affine scale and bias",
    "bilinear_sampling": (
        "coordinate/weight arithmetic plus four weighted samples; addressing, "
        "flooring, bounds checks and memory traffic are excluded"
    ),
    "excluded": "indexing, shape/view operations and memory movement",
}


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    return None


def _numel(value: Any) -> int:
    tensor = _first_tensor(value)
    return int(tensor.numel()) if tensor is not None else 0


def _matmul_flops(left: torch.Tensor, right: torch.Tensor,
                  output: Any) -> int:
    """Count dense vector/matrix products, including broadcast batches."""
    if left.ndim == 0 or right.ndim == 0:
        return _numel(output)
    reduction = int(left.shape[-1])
    return 2 * reduction * _numel(output)


class EndToEndFlopCounter(TorchDispatchMode):
    """Auditable ATen-level inference FLOP counter.

    PyTorch's built-in FLOP counter intentionally focuses on convolutions and
    matrix products.  The thesis models also perform causal state arithmetic,
    graph products, normalization, softmax coordinate extraction and bilinear
    warping.  Dispatch-level accounting sees all of those operations without
    modifying the model and keeps an explicit list of any unsupported operator.
    """

    _ELEMENTWISE_ONE = frozenset({
        "abs", "acos", "acosh", "add", "bitwise_and", "bitwise_or",
        "bitwise_xor", "ceil", "clamp", "clamp_max", "clamp_min", "cos",
        "cosh", "div", "eq", "exp", "expm1", "floor", "floor_divide",
        "fmod", "ge", "gelu", "gt", "hardshrink", "hardtanh", "le",
        "leaky_relu", "log", "log10", "log1p", "log2", "logical_and",
        "logical_not", "logical_or", "lt", "maximum", "minimum", "mul",
        "mul_", "ne", "neg", "pow", "reciprocal", "relu", "relu_",
        "remainder", "round",
        "rsqrt", "rsub", "sigmoid", "sign", "sin", "sinh", "sqrt", "sub",
        "tan", "tanh", "threshold",
    })
    _ELEMENTWISE_TWO = frozenset({"addcmul", "addcdiv"})
    _ELEMENTWISE_THREE = frozenset({"lerp"})
    _REDUCTION_SUM = frozenset({"sum", "nansum"})
    _REDUCTION_MEAN = frozenset({"mean", "nanmean"})
    _REDUCTION_EXTREME = frozenset({"amax", "amin", "max", "min"})
    _ZERO_ARITHMETIC = frozenset({
        "_to_copy", "_unsafe_view", "alias", "arange", "as_strided", "cat", "clone",
        "constant_pad_nd", "contiguous", "copy_", "detach", "embedding",
        "empty", "empty_like", "empty_strided", "expand", "eye", "fill_",
        "flatten", "flip", "full", "full_like", "gather", "index",
        "index_select", "lift_fresh", "lift_fresh_copy", "masked_fill",
        "meshgrid", "new_empty", "new_full", "new_ones", "new_zeros",
        "ones", "ones_like", "pad", "permute", "reflection_pad1d",
        "reflection_pad2d", "repeat", "reshape", "roll", "scalar_tensor",
        "select", "slice", "split", "split_with_sizes", "squeeze", "stack",
        "t", "to", "transpose", "tril", "unbind", "unfold", "unsqueeze",
        "view", "where", "zero_", "zeros", "zeros_like",
    })

    def __init__(self) -> None:
        super().__init__()
        self.flops_by_operator: Counter[str] = Counter()
        self.calls_by_operator: Counter[str] = Counter()
        self.zero_flop_calls: Counter[str] = Counter()
        self.unsupported_calls: Counter[str] = Counter()

    @staticmethod
    def _base_name(func: Any) -> tuple[str, str]:
        operator = str(func)
        parts = operator.split(".")
        base = parts[1] if len(parts) > 1 else operator
        return operator, base

    @staticmethod
    def _convolution(args: tuple[Any, ...], output: Any,
                     *, transposed: bool = False) -> int:
        value, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        if transposed:
            kernel = math.prod(int(item) for item in weight.shape[2:])
            groups = int(args[6]) if len(args) > 6 else 1
            products = (
                int(value.numel()) * int(weight.shape[1]) * kernel
            )
            # weight.shape[1] is already output_channels / groups.
            if groups < 1:
                raise ValueError("Convolution groups must be positive")
        else:
            products = _numel(output) * math.prod(
                int(item) for item in weight.shape[1:]
            )
        return 2 * products + (_numel(output) if bias is not None else 0)

    @staticmethod
    def _softmax(args: tuple[Any, ...], output: Any) -> int:
        value = args[0]
        dimension = int(args[1])
        if dimension < 0:
            dimension += value.ndim
        length = int(value.shape[dimension])
        vectors = int(value.numel()) // max(length, 1)
        # Per vector: max (L-1), subtract L, exp L, sum (L-1), divide L.
        return vectors * max(5 * length - 2, 0)

    @staticmethod
    def _layer_norm(args: tuple[Any, ...]) -> int:
        value = args[0]
        normalized_shape = tuple(int(item) for item in args[1])
        normalized = math.prod(normalized_shape)
        groups = int(value.numel()) // max(normalized, 1)
        weight = args[2] if len(args) > 2 else None
        bias = args[3] if len(args) > 3 else None
        # Mean, centering, variance, eps/rsqrt and normalization = 5N+2;
        # optional affine scale and bias contribute N each.
        per_group = 5 * normalized + 2
        if weight is not None:
            per_group += normalized
        if bias is not None:
            per_group += normalized
        return groups * per_group

    @staticmethod
    def _batch_norm(args: tuple[Any, ...]) -> int:
        value = args[0]
        weight = args[1] if len(args) > 1 else None
        bias = args[2] if len(args) > 2 else None
        # Frozen inference: center and scale by reciprocal std, then affine.
        per_element = 2 + int(weight is not None) + int(bias is not None)
        return per_element * int(value.numel())

    @staticmethod
    def _grid_sampler(args: tuple[Any, ...], output: Any) -> int:
        value = args[0]
        result = _first_tensor(output)
        if result is None or value.ndim != 4:
            return 0
        channels = int(result.shape[1])
        points = int(result.numel()) // max(channels, 1)
        # Per output point: normalized-to-source x/y (4), complements (2),
        # four bilinear weights (4). Per channel: four multiplies + three adds.
        return 10 * points + 7 * int(result.numel())

    @staticmethod
    def _upsample_bilinear(output: Any) -> int:
        result = _first_tensor(output)
        if result is None or result.ndim != 4:
            return 0
        points = int(result.shape[0] * result.shape[2] * result.shape[3])
        # Source-coordinate setup is shared across channels.
        return 4 * points + 7 * int(result.numel())

    @staticmethod
    def _pool(args: tuple[Any, ...], output: Any, *, average: bool) -> int:
        kernel = args[1]
        if isinstance(kernel, int):
            area = int(kernel) ** (args[0].ndim - 2)
        else:
            area = math.prod(int(item) for item in kernel)
        per_output = area if average else max(area - 1, 0)
        return per_output * _numel(output)

    def _count(self, base: str, args: tuple[Any, ...],
               kwargs: dict[str, Any], output: Any) -> int | None:
        del kwargs
        if base in self._ZERO_ARITHMETIC:
            return 0
        if base in self._ELEMENTWISE_ONE:
            return _numel(output)
        if base in self._ELEMENTWISE_TWO:
            return 2 * _numel(output)
        if base in self._ELEMENTWISE_THREE:
            return 3 * _numel(output)
        if base in self._REDUCTION_SUM:
            return max(int(args[0].numel()) - _numel(output), 0)
        if base in self._REDUCTION_MEAN:
            return max(int(args[0].numel()) - _numel(output), 0) + _numel(output)
        if base in self._REDUCTION_EXTREME:
            return max(int(args[0].numel()) - _numel(output), 0)
        if base in {"conv1d", "conv2d", "conv3d"}:
            return self._convolution(args, output)
        if base in {"conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}:
            return self._convolution(args, output, transposed=True)
        if base in {"convolution", "_convolution"}:
            transposed = bool(args[6])
            return self._convolution(args, output, transposed=transposed)
        if base == "linear":
            value, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else None
            return (
                2 * _numel(output) * int(value.shape[-1])
                + (_numel(output) if bias is not None else 0)
            )
        if base in {"matmul", "mm", "bmm"}:
            return _matmul_flops(args[0], args[1], output)
        if base == "addmm":
            return _matmul_flops(args[1], args[2], output) + _numel(output)
        if base == "baddbmm":
            return _matmul_flops(args[1], args[2], output) + _numel(output)
        if base in {"softmax", "_softmax", "log_softmax", "_log_softmax"}:
            return self._softmax(args, output)
        if base in {"layer_norm", "native_layer_norm"}:
            return self._layer_norm(args)
        if base in {"batch_norm", "cudnn_batch_norm", "native_batch_norm",
                    "_native_batch_norm_legit"}:
            return self._batch_norm(args)
        if base in {"grid_sampler", "grid_sampler_2d"}:
            return self._grid_sampler(args, output)
        if base in {"upsample_bilinear2d", "_upsample_bilinear2d_aa"}:
            return self._upsample_bilinear(output)
        if base in {"upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d"}:
            return 0
        if base in {"max_pool1d", "max_pool2d", "max_pool3d",
                    "max_pool2d_with_indices"}:
            return self._pool(args, output, average=False)
        if base in {"avg_pool1d", "avg_pool2d", "avg_pool3d"}:
            return self._pool(args, output, average=True)
        if base == "linalg_vector_norm":
            value = args[0]
            result_elements = _numel(output)
            return int(value.numel()) + max(
                int(value.numel()) - result_elements, 0,
            ) + result_elements
        if base in {"argmax", "argmin"}:
            return max(int(args[0].numel()) - _numel(output), 0)
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        output = func(*args, **kwargs)
        operator, base = self._base_name(func)
        self.calls_by_operator[operator] += 1
        flops = self._count(base, args, kwargs, output)
        if flops is None:
            self.unsupported_calls[operator] += 1
        elif flops == 0:
            self.zero_flop_calls[operator] += 1
        else:
            self.flops_by_operator[operator] += int(flops)
        return output

    @property
    def total_flops(self) -> int:
        return int(sum(self.flops_by_operator.values()))

    def report(self) -> dict[str, Any]:
        return {
            "flops": self.total_flops,
            "gflops": self.total_flops / 1e9,
            "operator_flops": dict(sorted(self.flops_by_operator.items())),
            "operator_calls": dict(sorted(self.calls_by_operator.items())),
            "zero_arithmetic_calls": dict(sorted(self.zero_flop_calls.items())),
            "unsupported_calls": dict(sorted(self.unsupported_calls.items())),
            "complete": not self.unsupported_calls,
            "assumptions": dict(END_TO_END_FLOP_ASSUMPTIONS),
        }


@torch.no_grad()
def profile_end_to_end_flops(model: nn.Module, image: torch.Tensor,
                             *, strict: bool = True) -> dict[str, Any]:
    """Profile complete inference arithmetic for one supplied model input."""
    model.eval()
    counter = EndToEndFlopCounter()
    with counter:
        model(image)
    report = counter.report()
    report.update({
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "input_shape": list(image.shape),
    })
    if strict and not report["complete"]:
        names = ", ".join(report["unsupported_calls"])
        raise RuntimeError(f"Unsupported arithmetic operators: {names}")
    return report


def _source_max_spikes(source: nn.Module) -> int:
    """Return the event-count ceiling used by a supported spike source."""
    value = getattr(source, "max_spikes", None)
    if value is not None:
        return int(value)
    # The vendored SpikeYOLO ``mem_update`` quantises with MultiSpike4.
    if source.__class__.__name__ == "mem_update" and hasattr(source, "qtrick"):
        return 4
    raise TypeError(
        f"Unsupported spike source for energy profiling: {type(source).__name__}"
    )


def _operation_maps(model: nn.Module) -> tuple[dict[int, nn.Module], set[int]]:
    """Map physical convolutions to their spike source and temporal execution."""
    spike_sources: dict[int, nn.Module] = {}
    temporal_convolutions: set[int] = set()
    for module in model.modules():
        if isinstance(module, TimeConvBN):
            temporal_convolutions.add(id(module.conv))
        elif isinstance(module, SpikeConv):
            temporal_convolutions.add(id(module.conv.conv))
            if (
                isinstance(module.neuron, MultiStepILIF)
                and not module.neuron.membrane_readout
            ):
                spike_sources[id(module.conv.conv)] = module.neuron
        elif isinstance(module, SpikePoseDownsample):
            temporal_convolutions.add(id(module.conv.conv))
            if (
                isinstance(module.neuron, MultiStepILIF)
                and not module.neuron.membrane_readout
            ):
                spike_sources[id(module.conv.conv)] = module.neuron

    # SpikeYOLO uses a separate vendored implementation. Only convolutions
    # receiving the direct integer output of ``mem_update`` are eligible for
    # AC accounting; intermediate dense convolutions remain MACs.
    try:
        from ..models.baselines import _spikeyolo_layers as spikeyolo
    except ImportError:  # pragma: no cover - optional baseline installation
        spikeyolo = None
    if spikeyolo is not None:
        for module in model.modules():
            mappings: list[tuple[nn.Module, nn.Module]] = []
            if isinstance(module, spikeyolo.MS_DownSampling):
                temporal_convolutions.add(id(module.encode_conv))
                source = getattr(module, "encode_lif", None)
                if source is not None:
                    mappings.append((module.encode_conv, source))
            elif isinstance(module, spikeyolo.MS_StandardConv):
                mappings.append((module.conv, module.lif))
            elif isinstance(module, spikeyolo.SpikeConv):
                mappings.append((module.conv, module.lif))
            elif isinstance(module, spikeyolo.SepConv):
                mappings.extend([
                    (module.pwconv1, module.lif1),
                    (module.dwconv2, module.lif2),
                ])
            elif isinstance(module, spikeyolo.MS_ConvBlock):
                mappings.extend([
                    (module.conv1.body[0], module.lif1),
                    (module.conv2.body[0], module.lif2),
                ])
            for convolution, source in mappings:
                temporal_convolutions.add(id(convolution))
                spike_sources[id(convolution)] = source
    return spike_sources, temporal_convolutions


@torch.no_grad()
def profile_theoretical(model: nn.Module, image: torch.Tensor,
                        output_path: Path | None = None, *,
                        video_steps: int | None = None,
                        snn_steps_per_frame: int | None = None,
                        include_end_to_end: bool = True,
                        validate_spikes: bool = True) -> dict:
    """Estimate arithmetic energy for one supplied batch.

    The calculation follows the common 45-nm MAC/AC convention. Inputs to
    spike-driven convolutions are interpreted as integer event multiplicities,
    so an activation value of four contributes four accumulations.
    """
    spike_sources, temporal_convolutions = _operation_maps(model)
    config = getattr(model, "config", None)
    temporal_config = getattr(config, "temporal", None)
    inferred_video_steps = int(getattr(temporal_config, "video_frames", 1))
    inferred_snn_steps = int(getattr(temporal_config, "snn_steps_per_frame", 1))
    video_steps = inferred_video_steps if video_steps is None else int(video_steps)
    snn_steps = (
        inferred_snn_steps if snn_steps_per_frame is None
        else int(snn_steps_per_frame)
    )
    if video_steps < 1 or snn_steps < 1:
        raise ValueError("video_steps and snn_steps_per_frame must be positive")
    steps = video_steps * snn_steps
    layers: list[dict] = []
    handles = []

    for name, module in model.named_modules():
        if not isinstance(
            module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        ):
            continue

        def count_hook(layer, inputs, output, layer_name=name):
            if isinstance(layer, nn.Linear):
                macs = int(output.numel() * layer.in_features)
            else:
                kernel_area = 1
                for size in layer.kernel_size:
                    kernel_area *= int(size)
                if isinstance(layer, nn.ConvTranspose2d):
                    # A transposed convolution scatters each input value over
                    # output channels and kernel positions. Using output.numel()
                    # here incorrectly multiplies cost by the spatial upsampling.
                    macs = int(inputs[0].numel()
                               * (layer.out_channels // layer.groups) * kernel_area)
                else:
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
                max_spikes = _source_max_spikes(source)
                if validate_spikes:
                    tolerance = 1e-5
                    minimum = float(values.min())
                    maximum = float(values.max())
                    integer_error = float((values - values.round()).abs().max())
                    if (minimum < -tolerance or maximum > max_spikes + tolerance
                            or integer_error > tolerance):
                        raise ValueError(
                            f"{layer_name} was classified as spike-driven but "
                            "received values outside the integer event range "
                            f"[0, {max_spikes}]"
                        )
                mean_integer_spikes = float(values.sum()) / values.numel()
                normalized_firing_rate = mean_integer_spikes / max_spikes
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
                "max_spikes": (
                    _source_max_spikes(source) if source is not None else None
                ),
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
    # Run the operator-exact pass after removing energy hooks. Otherwise the
    # hook reductions used to measure spike activity would contaminate FLOPs.
    end_to_end_report = (
        profile_end_to_end_flops(model, image, strict=False)
        if include_end_to_end else None
    )

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
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "model_size_fp32_mb": (
            sum(parameter.numel() for parameter in model.parameters()) * 4
            / (1024 ** 2)
        ),
        "time_steps": steps,
        "video_time_steps": video_steps,
        "snn_steps_per_frame": snn_steps,
        "analog_macs": ann_macs,
        "spiking_dense_macs": spiking_dense_macs,
        "dense_macs": ann_macs + spiking_dense_macs,
        "dense_flops": 2 * (ann_macs + spiking_dense_macs),
        "legacy_conv_linear_flops": 2 * (ann_macs + spiking_dense_macs),
        "end_to_end_flops": (
            end_to_end_report["flops"] if end_to_end_report else None
        ),
        "end_to_end_gflops": (
            end_to_end_report["gflops"] if end_to_end_report else None
        ),
        "end_to_end_complete": (
            end_to_end_report["complete"] if end_to_end_report else None
        ),
        "end_to_end_operator_flops": (
            end_to_end_report["operator_flops"] if end_to_end_report else {}
        ),
        "end_to_end_unsupported_calls": (
            end_to_end_report["unsupported_calls"] if end_to_end_report else {}
        ),
        "operation_weighted_firing_rate": weighted_firing_rate,
        "operation_weighted_mean_integer_spikes": (
            effective_sops / spiking_dense_macs
            if spiking_dense_macs else 0.0
        ),
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
            "method": "layer-wise activity-weighted analytical arithmetic energy",
            "technology": "45 nm",
            "precision": "FP32",
            "fp32_mac_pj": MAC_ENERGY_PJ,
            "accumulate_pj": AC_ENERGY_PJ,
            "flops_per_mac": 2,
            "scope": "arithmetic operations only",
            "input_scope": "complete supplied image or video clip",
            "ann_baseline": "all dense MACs over the complete supplied input",
            "integer_spike_encoding": (
                "integer activation k is treated as k spike-triggered AC events"
            ),
            "excluded_energy": (
                "memory access, data movement, nonlinear functions, neuron state "
                "updates and non-MAC/AC arithmetic"
            ),
            "end_to_end_flops": dict(END_TO_END_FLOP_ASSUMPTIONS),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def profile_theoretical_dataset(
    model: nn.Module,
    images: Iterable[torch.Tensor],
    output_path: Path,
    *,
    video_steps: int,
    snn_steps_per_frame: int = 1,
    max_batches: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Average data-dependent theoretical energy over real input windows.

    Counts returned by this function are normalised per item in the leading
    batch dimension. For the NTU protocol, one such item is one physical
    ``video_steps``-frame window. End-to-end FLOPs are captured from the first
    batch because fixed-shape dense arithmetic is independent of spike activity;
    layer-wise AC counts are accumulated over every supplied batch.
    """
    if video_steps < 1 or snn_steps_per_frame < 1:
        raise ValueError("video_steps and snn_steps_per_frame must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")

    scalar_fields = (
        "analog_macs", "spiking_dense_macs", "dense_macs",
        "effective_sops", "ann_architecture_macs",
        "legacy_single_step_ann_equivalent_macs",
    )
    totals = {key: 0.0 for key in scalar_fields}
    layer_totals: dict[tuple[str, str], dict[str, Any]] = {}
    first_report: dict[str, Any] | None = None
    first_input_shape: list[int] | None = None
    profiled_batches = 0
    profiled_windows = 0

    for image in images:
        if max_batches is not None and profiled_batches >= max_batches:
            break
        if not isinstance(image, torch.Tensor):
            raise TypeError("profile_theoretical_dataset expects tensors")
        if image.ndim < 1 or int(image.shape[0]) < 1:
            raise ValueError(
                "Each profiling batch must have a non-empty leading dimension"
            )
        shape = list(image.shape)
        if first_input_shape is None:
            first_input_shape = shape
        elif shape[1:] != first_input_shape[1:]:
            raise ValueError("All profiling batches must share the same non-batch shape")

        report = profile_theoretical(
            model, image, None,
            video_steps=video_steps,
            snn_steps_per_frame=snn_steps_per_frame,
            include_end_to_end=first_report is None,
            validate_spikes=first_report is None,
        )
        if first_report is None:
            first_report = report

        windows = int(image.shape[0])
        profiled_windows += windows
        profiled_batches += 1
        for key in scalar_fields:
            totals[key] += float(report[key])
        for item in report["layers"]:
            key = (str(item["name"]), str(item["operation"]))
            aggregate = layer_totals.setdefault(key, {
                "name": item["name"],
                "operation": item["operation"],
                "temporal": bool(item["temporal"]),
                "max_spikes": item["max_spikes"],
                "invocations": 0,
                "dense_macs": 0.0,
                "ann_architecture_macs": 0.0,
                "legacy_single_step_ann_equivalent_macs": 0.0,
                "effective_operations": 0.0,
            })
            if aggregate["max_spikes"] != item["max_spikes"]:
                raise RuntimeError(f"Inconsistent spike ceiling for layer {item['name']}")
            # One module call processes every item in this batch. Weighting by
            # the leading batch size makes the final value interpretable as
            # logical invocations per profiled window.
            aggregate["invocations"] += windows
            for field in (
                "dense_macs", "ann_architecture_macs",
                "legacy_single_step_ann_equivalent_macs",
                "effective_operations",
            ):
                aggregate[field] += float(item[field])

    if first_report is None or first_input_shape is None or profiled_windows == 0:
        raise ValueError("No profiling batches were supplied")

    per_window = {
        key: value / profiled_windows for key, value in totals.items()
    }
    analog_energy_pj = per_window["analog_macs"] * MAC_ENERGY_PJ
    spike_energy_pj = per_window["effective_sops"] * AC_ENERGY_PJ
    total_energy_pj = analog_energy_pj + spike_energy_pj
    ann_energy_pj = per_window["ann_architecture_macs"] * MAC_ENERGY_PJ
    max_weighted_dense = sum(
        item["dense_macs"] * item["max_spikes"]
        for item in layer_totals.values()
        if item["operation"] == "AC"
    ) / profiled_windows

    layers = []
    for aggregate in sorted(layer_totals.values(), key=lambda item: (
        str(item["name"]), str(item["operation"]),
    )):
        dense_macs = aggregate["dense_macs"] / profiled_windows
        effective_operations = (
            aggregate["effective_operations"] / profiled_windows
        )
        max_spikes = aggregate["max_spikes"]
        mean_integer_spikes = (
            effective_operations / dense_macs
            if aggregate["operation"] == "AC" and dense_macs else None
        )
        layers.append({
            "name": aggregate["name"],
            "operation": aggregate["operation"],
            "temporal": aggregate["temporal"],
            "invocations_per_window": aggregate["invocations"] / profiled_windows,
            "dense_macs": dense_macs,
            "ann_architecture_macs": (
                aggregate["ann_architecture_macs"] / profiled_windows
            ),
            "ann_equivalent_macs": (
                aggregate["ann_architecture_macs"] / profiled_windows
            ),
            "legacy_single_step_ann_equivalent_macs": (
                aggregate["legacy_single_step_ann_equivalent_macs"]
                / profiled_windows
            ),
            "max_spikes": max_spikes,
            "mean_integer_spikes": mean_integer_spikes,
            "normalized_firing_rate": (
                mean_integer_spikes / max_spikes
                if mean_integer_spikes is not None and max_spikes else None
            ),
            "effective_operations": effective_operations,
            "energy_mj": (
                effective_operations * (
                    AC_ENERGY_PJ if aggregate["operation"] == "AC"
                    else MAC_ENERGY_PJ
                ) / 1e9
            ),
        })

    first_windows = int(first_input_shape[0])
    end_to_end_flops = (
        float(first_report["end_to_end_flops"]) / first_windows
        if first_report["end_to_end_flops"] is not None else None
    )
    end_to_end_operator_flops = {
        key: float(value) / first_windows
        for key, value in first_report["end_to_end_operator_flops"].items()
    }
    parameters = int(first_report["parameters"])
    report = {
        "experiment_id": first_report["experiment_id"],
        "model_name": first_report["model_name"],
        "parameters": parameters,
        "trainable_parameters": int(first_report["trainable_parameters"]),
        "model_size_fp32_mb": parameters * 4 / (1024 ** 2),
        "profiled_batches": profiled_batches,
        "profiled_windows": profiled_windows,
        "normalization": f"per {video_steps}-frame window",
        "time_steps": video_steps * snn_steps_per_frame,
        "video_time_steps": video_steps,
        "snn_steps_per_frame": snn_steps_per_frame,
        "analog_macs": per_window["analog_macs"],
        "spiking_dense_macs": per_window["spiking_dense_macs"],
        "dense_macs": per_window["dense_macs"],
        "dense_flops": 2 * per_window["dense_macs"],
        "legacy_conv_linear_flops": 2 * per_window["dense_macs"],
        "end_to_end_flops": end_to_end_flops,
        "end_to_end_gflops": (
            end_to_end_flops / 1e9 if end_to_end_flops is not None else None
        ),
        "end_to_end_complete": first_report["end_to_end_complete"],
        "end_to_end_operator_flops": end_to_end_operator_flops,
        "end_to_end_unsupported_calls": (
            first_report["end_to_end_unsupported_calls"]
        ),
        "operation_weighted_firing_rate": (
            per_window["effective_sops"] / max_weighted_dense
            if max_weighted_dense else 0.0
        ),
        "operation_weighted_mean_integer_spikes": (
            per_window["effective_sops"] / per_window["spiking_dense_macs"]
            if per_window["spiking_dense_macs"] else 0.0
        ),
        "effective_sops": per_window["effective_sops"],
        "analog_mac_energy_mj": analog_energy_pj / 1e9,
        "spiking_ac_energy_mj": spike_energy_pj / 1e9,
        "total_theoretical_energy_mj": total_energy_pj / 1e9,
        "ann_architecture_macs": per_window["ann_architecture_macs"],
        "ann_architecture_energy_mj": ann_energy_pj / 1e9,
        "ann_equivalent_macs": per_window["ann_architecture_macs"],
        "ann_equivalent_energy_mj": ann_energy_pj / 1e9,
        "legacy_single_step_ann_equivalent_macs": (
            per_window["legacy_single_step_ann_equivalent_macs"]
        ),
        "legacy_single_step_ann_equivalent_energy_mj": (
            per_window["legacy_single_step_ann_equivalent_macs"]
            * MAC_ENERGY_PJ / 1e9
        ),
        "architecture_energy_efficiency_ratio": (
            ann_energy_pj / total_energy_pj if total_energy_pj else None
        ),
        "theoretical_energy_efficiency_ratio": (
            ann_energy_pj / total_energy_pj if total_energy_pj else None
        ),
        "input_shape_per_window": [1, *first_input_shape[1:]],
        "layers": layers,
        "profile_metadata": dict(metadata or {}),
        "assumptions": {
            **first_report["assumptions"],
            "input_scope": (
                f"mean per {video_steps}-frame window over real supplied data"
            ),
            "activity_aggregation": (
                "effective operations and firing activity are averaged over all "
                "profiled windows"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
