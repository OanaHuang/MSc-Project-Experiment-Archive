from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from scripts.MPII.core.geometry import heatmaps_to_keypoints
from scripts.MPII.datasets import MPIIPoseDataset
from scripts.MPII.visualization.render import tensor_to_bgr
from scripts.MPII.visualization.skeleton import MPII_EDGES
from scripts.spikepose.models import build_model
from scripts.spikepose.models.neurons import (
    MultiStepILIF, MultiStepLIF, TimeDistributedReLU,
)
from scripts.spikepose.training import load_model
from scripts.spikepose.visualization import draw_skeleton, load_manifest


def _feature_view(value: torch.Tensor) -> torch.Tensor:
    """Return one sample as T x C x H x W, adding T for ANN tensors."""
    value = value.detach().float().cpu()
    if value.ndim == 5:
        return value[:, 0]
    if value.ndim == 4:
        return value[0].unsqueeze(0)
    raise ValueError(f"Expected a 4D or 5D feature tensor, got {tuple(value.shape)}")


def feature_statistics(name: str, value: torch.Tensor) -> dict:
    feature = _feature_view(value)
    channel_strength = feature.abs().mean(dim=(0, 2, 3))
    top = torch.topk(channel_strength, min(8, feature.shape[1])).indices.tolist()
    return {
        "name": name,
        "shape": list(value.shape),
        "steps": int(feature.shape[0]),
        "channels": int(feature.shape[1]),
        "height": int(feature.shape[2]),
        "width": int(feature.shape[3]),
        "min": float(feature.min()),
        "max": float(feature.max()),
        "mean": float(feature.mean()),
        "std": float(feature.std()),
        "mean_abs": float(feature.abs().mean()),
        "nonzero_rate": float((feature != 0).float().mean()),
        "positive_rate": float((feature > 0).float().mean()),
        "top_channels": top,
    }


def _colorize(value: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    value = np.asarray(value, np.float32)
    finite = value[np.isfinite(value)]
    if not finite.size:
        normalized = np.zeros(value.shape, np.uint8)
    else:
        low, high = np.percentile(finite, (1, 99))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        normalized = np.clip((value - low) / max(high - low, 1e-12), 0, 1)
        normalized = np.round(normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 27), (18, 27, 38), -1)
    cv2.putText(output, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _fixed_colorize(value: np.ndarray, maximum: float,
                    size: tuple[int, int] = (256, 256)) -> np.ndarray:
    normalized = np.clip(np.asarray(value, np.float32) / max(maximum, 1e-12), 0, 1)
    colored = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)


def capture_backbone_neuron_outputs(model, sequence: torch.Tensor):
    """Run the backbone while observing neuron outputs without changing it."""
    traces: dict[int, list[torch.Tensor]] = {
        index: [] for index in range(1, len(model.backbone.stages) + 1)
    }
    handles = []
    for stage_index, stage in enumerate(model.backbone.stages, 1):
        for module in stage.modules():
            if isinstance(module, (MultiStepLIF, MultiStepILIF, TimeDistributedReLU)):
                handles.append(module.register_forward_hook(
                    lambda _module, _inputs, output, index=stage_index:
                    traces[index].append(output.detach().float().cpu())
                ))
    try:
        backbone = model.backbone(sequence)
    finally:
        for handle in handles:
            handle.remove()
    return backbone, traces


def _aggregate_firing(outputs: list[torch.Tensor]) -> tuple[np.ndarray, float, float] | None:
    if not outputs:
        return None
    target_h = max(int(value.shape[-2]) for value in outputs)
    target_w = max(int(value.shape[-1]) for value in outputs)
    maps = []
    total_values = 0
    active_values = 0
    spike_sum = 0.0
    for value in outputs:
        spatial = value.mean(dim=(0, 1, 2)).numpy()
        if spatial.shape != (target_h, target_w):
            spatial = cv2.resize(spatial, (target_w, target_h),
                                 interpolation=cv2.INTER_NEAREST)
        maps.append(spatial)
        total_values += value.numel()
        active_values += int((value > 0).sum())
        spike_sum += float(value.sum())
    return (np.mean(maps, axis=0), spike_sum / total_values,
            active_values / total_values)


def _save_neuron_firing_overview(input_bgr: np.ndarray,
                                 traces: dict[int, list[torch.Tensor]],
                                 max_spikes: int, neuron_kind: str,
                                 path: Path) -> None:
    panels = [_label(cv2.resize(input_bgr, (256, 256)), "Input")]
    for stage_index in range(1, 5):
        aggregate = _aggregate_firing(traces.get(stage_index, []))
        if aggregate is None:
            panels.append(_label(np.full((256, 256, 3), 38, np.uint8),
                                 f"Stage {stage_index} | no neuron"))
            continue
        firing_map, mean_spikes, active_rate = aggregate
        state = "firing" if neuron_kind in {"lif", "ilif"} else "ReLU activation"
        text = (f"Stage {stage_index} {state} | mean={mean_spikes:.3f} "
                f"active={active_rate:.1%}")
        image = (_fixed_colorize(firing_map, float(max_spikes))
                 if neuron_kind in {"lif", "ilif"} else _colorize(firing_map))
        panels.append(_label(image, text))
    cv2.imwrite(str(path), np.hstack(panels))


def _save_single_channel_firing(stage_index: int, outputs: list[torch.Tensor],
                                max_spikes: int, neuron_kind: str, output_dir: Path,
                                top_channels: int = 8) -> None:
    """Render raw T/B/channel slices from the last neuron executed in a stage."""
    if not outputs:
        return
    value = outputs[-1]
    if value.ndim != 5:
        return
    sample = value[:, 0]
    strength = sample.sum(dim=(0, 2, 3))
    selected = torch.topk(strength, min(top_channels, sample.shape[1])).indices.tolist()
    rows = []
    for step in range(sample.shape[0]):
        panels = []
        for channel in selected:
            raw = sample[step, channel].numpy()
            label = f"T{step + 1} C{channel} sum={raw.sum():.0f}"
            image = (_fixed_colorize(raw, float(max_spikes), (192, 192))
                     if neuron_kind in {"lif", "ilif"} else _colorize(raw, (192, 192)))
            panels.append(_label(image, label))
        rows.append(np.hstack(panels))
    cv2.imwrite(str(output_dir / f"stage_{stage_index}_neuron_single_channels.png"),
                np.vstack(rows))


def _mean_map(value: torch.Tensor) -> np.ndarray:
    return _feature_view(value).abs().mean(dim=(0, 1)).numpy()


def _max_map(value: torch.Tensor) -> np.ndarray:
    return _feature_view(value).abs().amax(dim=(0, 1)).numpy()


def _activity_map(value: torch.Tensor) -> np.ndarray:
    feature = _feature_view(value)
    active = feature != 0 if float(feature.min()) >= 0 else feature > 0
    return active.float().mean(dim=(0, 1)).numpy()


def _representative_firing(outputs: list[torch.Tensor]) -> tuple[np.ndarray, int, int] | None:
    """Return the strongest raw channel accumulated across time."""
    if not outputs or outputs[-1].ndim != 5:
        return None
    sample = outputs[-1][:, 0]
    strength = sample.sum(dim=(0, 2, 3))
    channel = int(torch.argmax(strength))
    return sample[:, channel].sum(0).numpy(), channel, int(sample.shape[0])


def _save_overview(input_bgr: np.ndarray, features: dict[str, torch.Tensor],
                   neuron_traces: dict[int, list[torch.Tensor]], max_spikes: int,
                   neuron_kind: str, heatmaps: np.ndarray,
                   prediction: np.ndarray, path: Path) -> None:
    panels = [_label(cv2.resize(input_bgr, (256, 256)), "Input")]
    for stage_index, name in enumerate(("Stage 1", "Stage 2", "Stage 3", "Stage 4"), 1):
        value = features.get(name)
        if value is None:
            panels.append(_label(np.full((256, 256, 3), 38, np.uint8),
                                 f"{name} | not present"))
        else:
            panels.append(_label(_colorize(_mean_map(value)), f"{name} | mean |x|"))
        firing = _representative_firing(neuron_traces.get(stage_index, []))
        if firing is None:
            state = "firing" if neuron_kind in {"lif", "ilif"} else "activation"
            panels.append(_label(np.full((256, 256, 3), 38, np.uint8),
                                 f"{name} {state} | not present"))
        else:
            firing_map, channel, steps = firing
            state = "firing" if neuron_kind in {"lif", "ilif"} else "activation"
            image = (_fixed_colorize(firing_map, float(max_spikes * steps))
                     if neuron_kind in {"lif", "ilif"} else _colorize(firing_map))
            time_label = "T1" if steps == 1 else "+".join(
                f"T{step}" for step in range(1, steps + 1))
            panels.append(_label(
                image, f"{name} {state} C{channel} | {time_label}",
            ))
    panels.append(_label(_colorize(_mean_map(features["Neck"])), "Neck | mean |x|"))
    panels.append(_label(_colorize(np.max(heatmaps, axis=0)), "Head | max joint heatmap"))
    panels.append(_label(cv2.resize(prediction, (256, 256)), "Final pose"))
    cv2.imwrite(str(path), np.hstack(panels))


def _save_stage_diagnostics(name: str, value: torch.Tensor, output_dir: Path) -> None:
    feature = _feature_view(value)
    stats = feature_statistics(name, value)
    maps = [
        _label(_colorize(_mean_map(value)), f"{name} mean |x|"),
        _label(_colorize(_max_map(value)), f"{name} max |x|"),
        _label(_colorize(_activity_map(value)), f"{name} activity density"),
    ]
    file_stem = name.lower().replace(" ", "_")
    cv2.imwrite(str(output_dir / f"{file_stem}_summary.png"), np.hstack(maps))

    channel_panels = []
    averaged = feature.abs().mean(0)
    for channel in stats["top_channels"]:
        channel_panels.append(_label(
            _colorize(averaged[channel].numpy(), (192, 192)),
            f"{name} channel {channel}",
        ))
    if channel_panels:
        columns = 4
        blank = np.zeros_like(channel_panels[0])
        while len(channel_panels) % columns:
            channel_panels.append(blank)
        rows = [np.hstack(channel_panels[i:i + columns])
                for i in range(0, len(channel_panels), columns)]
        cv2.imwrite(str(output_dir / f"{file_stem}_top_channels.png"), np.vstack(rows))

    if feature.shape[0] > 1:
        temporal = []
        for step in range(feature.shape[0]):
            temporal.append(_label(
                _colorize(feature[step].abs().mean(0).numpy()),
                f"{name} T{step + 1}",
            ))
        difference = (feature[-1] - feature[0]).abs().mean(0).numpy()
        temporal.append(_label(_colorize(difference), f"{name} |T{feature.shape[0]}-T1|"))
        cv2.imwrite(str(output_dir / f"{file_stem}_temporal.png"), np.hstack(temporal))


def _dataset(project_root: Path, config: dict) -> MPIIPoseDataset:
    data = config["data"]
    return MPIIPoseDataset(
        project_root / data["visualization_metadata"],
        project_root / data["images_dir"], data["image_size"],
        data["heatmap_size"], data["sigma"], data["crop_expansion"], False,
    )


@torch.no_grad()
def visualize_run(project_root: Path, run_dir: Path, manifest_path: Path,
                  output_dir: Path, device: str = "cpu",
                  groups: tuple[str, ...] = ("first", "random"),
                  max_per_group: int | None = None, save_raw: bool = False) -> dict:
    config = yaml.safe_load((run_dir / "config" / "resolved.yaml").read_text())
    torch_device = torch.device(device)
    model = build_model(config).to(torch_device)
    checkpoint = load_model(run_dir / "checkpoints" / "best.pt", model, torch_device)
    model.eval()
    dataset = _dataset(project_root, config)
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    processed = 0
    for group in groups:
        selected_items = manifest[group]
        if max_per_group is not None:
            selected_items = selected_items[:max_per_group]
        for order, selected in enumerate(selected_items, 1):
            sample = dataset[int(selected["index"])]
            image = sample["image"].unsqueeze(0).to(torch_device)
            sequence = image.unsqueeze(0).repeat(config["model"]["num_steps"], 1, 1, 1, 1)
            backbone, neuron_traces = capture_backbone_neuron_outputs(model, sequence)
            neck = model.neck(backbone.features)
            heatmap_tensor = model.head(neck)
            heatmaps = heatmap_tensor[0].detach().cpu().numpy()

            features = {f"Stage {index}": value for index, value in backbone.features.items()}
            features["Neck"] = neck
            input_bgr = tensor_to_bgr(sample["image"])
            keypoints, confidence = heatmaps_to_keypoints(heatmaps, dataset.image_size)
            prediction = draw_skeleton(input_bgr, keypoints, confidence > 0,
                                       MPII_EDGES, (40, 200, 255))
            sample_dir = output_dir / "samples" / group / f"{order:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            max_spikes = int(config["model"]["neuron"]["max_spikes"])
            neuron_kind = str(config["model"]["neuron"]["kind"])
            _save_overview(input_bgr, features, neuron_traces, max_spikes,
                           neuron_kind, heatmaps, prediction,
                           sample_dir / "stage_overview.png")
            _save_neuron_firing_overview(
                input_bgr, neuron_traces, max_spikes, neuron_kind,
                sample_dir / ("neuron_firing_overview.png"
                              if neuron_kind in {"lif", "ilif"}
                              else "neuron_activation_overview.png"),
            )
            for stage_index, outputs in neuron_traces.items():
                _save_single_channel_firing(
                    stage_index, outputs,
                    max_spikes, neuron_kind, sample_dir,
                )
            for name, value in features.items():
                _save_stage_diagnostics(name, value, sample_dir)
                row = feature_statistics(name, value)
                row.update({"group": group, "order": order, "sample_id": selected["id"]})
                all_stats.append(row)
            if save_raw:
                arrays = {name.lower().replace(" ", "_"): value.detach().cpu().numpy()
                          for name, value in features.items()}
                arrays["heatmaps"] = heatmaps
                np.savez_compressed(sample_dir / "activations.npz", **arrays)
            processed += 1

    fields = ["group", "order", "sample_id", "name", "shape", "steps", "channels",
              "height", "width", "min", "max", "mean", "std", "mean_abs",
              "nonzero_rate", "positive_rate", "top_channels"]
    with (output_dir / "activation_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_stats:
            writer.writerow({**row, "shape": json.dumps(row["shape"]),
                             "top_channels": json.dumps(row["top_channels"])})
    summary = {
        "status": "completed", "run": str(run_dir), "manifest": str(manifest_path),
        "checkpoint_epoch": checkpoint.get("epoch"), "samples": processed,
        "groups": list(groups), "save_raw": save_raw,
        "normalization": "Feature panels use per-panel 1st-99th percentile; raw statistics are retained",
        "neuron_firing": {
            "neuron_kind": config["model"]["neuron"]["kind"],
            "included_in_standard_output": True,
            "overview_scale": (f"fixed 0-{config['model']['neuron']['max_spikes']} spikes"
                               if config["model"]["neuron"]["kind"] in {"lif", "ilif"}
                               else "per-panel 1st-99th percentile ReLU activation"),
            "single_channel_scale": (f"fixed 0-{config['model']['neuron']['max_spikes']} spikes"
                                     if config["model"]["neuron"]["kind"] in {"lif", "ilif"}
                                     else "per-panel 1st-99th percentile ReLU activation"),
            "single_channel_selection": "top 8 channels from the last neuron executed in each stage",
            "stage_overview_firing": "strongest channel from the last neuron in each stage, accumulated across time",
            "spatial_smoothing": False,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
