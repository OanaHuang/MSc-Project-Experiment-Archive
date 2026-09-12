from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikepose_thesis.artifacts import RunArtifacts, locate_source_run
from spikepose_thesis.artifacts.manager import file_fingerprint
from spikepose_thesis.core.config import apply_pilot_runtime_overrides, load_experiment
from spikepose_thesis.core.paths import resolve_project_path
from spikepose_thesis.data import build_dataset
from spikepose_thesis.data.heatmaps import generate_gaussian_heatmaps_torch
from spikepose_thesis.evaluation import evaluate_checkpoint
from spikepose_thesis.evaluation.runner import collect_predictions, summarize_predictions
from spikepose_thesis.evaluation.temporal import temporal_summary
from spikepose_thesis.models import build_model
from spikepose_thesis.models.baselines import load_official_checkpoint
from spikepose_thesis.models.neurons import MultiStepILIF

from .checkpoint import (
    load_checkpoint, load_model, load_mpii16_to_ntu25_model,
    load_spatial_model, restore_training_state, save_checkpoint,
)
from .losses import VisibleHeatmapMSE
from .schedulers import build_scheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # Runtime crops are stored as lossless WebP.  OpenCV otherwise creates a
    # machine-sized thread pool in every DataLoader process, so four concurrent
    # jobs with four workers each can expand into hundreds of runnable threads
    # and starve the GPUs.  Decoding remains bit-identical with one OpenCV
    # thread while each worker itself still runs in parallel.
    import cv2

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _StageFiringMonitor:
    """Accumulate normalized integer-spike rates without retaining activations."""

    def __init__(self, model: nn.Module):
        self.totals: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.handles = []
        self.closed = False
        for name, module in model.named_modules():
            if not isinstance(module, MultiStepILIF):
                continue
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "backbone" and parts[1] in {
                "downsamples", "stages",
            }:
                label = f"stage_{int(parts[2]) + 1}"
            elif parts and parts[0] in {"neck", "head"}:
                label = parts[0]
            else:
                label = "other"
            maximum = float(module.max_spikes)

            def hook(_module, _inputs, output, *, stage=label, scale=maximum):
                value = output.detach().float().clamp_(0.0, scale)
                normalized_sum = value.sum() / scale
                self.totals[stage] = self.totals.get(stage, normalized_sum.new_zeros(())) + normalized_sum
                self.counts[stage] = self.counts.get(stage, 0) + value.numel()

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True

    def summary(self) -> dict[str, float]:
        return {
            stage: float(self.totals[stage].cpu()) / max(self.counts[stage], 1)
            for stage in sorted(self.totals)
        }


def _loader(dataset, training: dict, train: bool, seed: int) -> DataLoader:
    workers = int(training["num_workers"])
    batch_size = int(
        training.get("clip_batch_size", training["batch_size"])
        if getattr(dataset, "temporal_steps", 1) > 1 else training["batch_size"]
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if training.get("deterministic_workers") else None,
        generator=torch.Generator().manual_seed(seed),
    )


def _phase_for_epoch(training: dict, epoch: int) -> dict | None:
    for phase in training.get("phases", []):
        if int(phase["start_epoch"]) <= epoch <= int(phase["end_epoch"]):
            return phase
    return None


def _configure_trainable(model: nn.Module, training: dict,
                         phase: dict | None) -> torch.optim.Optimizer:
    for parameter in model.parameters():
        parameter.requires_grad = phase is None
    groups = []
    if phase is None:
        learning_rate = float(training["learning_rate"])
        groups.append({"params": list(model.parameters()), "lr": learning_rate,
                       "base_lr": learning_rate, "name": "model"})
    else:
        rates = phase.get("learning_rates", {})
        seen_parameters: set[int] = set()
        for name in phase["train_modules"]:
            try:
                component = model.get_submodule(name)
                parameters = list(component.parameters())
            except AttributeError:
                try:
                    parameters = [model.get_parameter(name)]
                except AttributeError as error:
                    raise ValueError(
                        f"Unknown trainable module/parameter selector: {name}"
                    ) from error
            duplicate = [
                parameter for parameter in parameters
                if id(parameter) in seen_parameters
            ]
            if duplicate:
                raise ValueError(
                    f"Overlapping trainable selector {name!r}; list parent or child, not both"
                )
            seen_parameters.update(id(parameter) for parameter in parameters)
            for parameter in parameters:
                parameter.requires_grad = True
            learning_rate = float(rates.get(name, training["learning_rate"]))
            groups.append({"params": parameters, "lr": learning_rate,
                           "base_lr": learning_rate, "name": name})
    optimizer_type = str(training.get("optimizer", "adamw")).lower()
    constructor = torch.optim.Adam if optimizer_type == "adam" else torch.optim.AdamW
    return constructor(groups, weight_decay=float(training.get("weight_decay", 0.0)))


def _freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def _mam_batch_diagnostics(model: nn.Module) -> dict[str, float | str]:
    """Cheap sampled V2 diagnostics; callers control sampling frequency."""
    auxiliary = getattr(model, "last_mam_aux", {})
    required = {"gamma", "decay", "state", "raw_heatmap", "residual_offset"}
    if not required.issubset(auxiliary):
        return {}
    gamma = auxiliary["gamma"][1:].float()
    decay = auxiliary["decay"][1:].float()
    state = auxiliary["state"][1:].float()
    raw = auxiliary["raw_heatmap"][1:].float()
    residual = auxiliary["residual_offset"][1:].float()
    if gamma.numel() == 0:
        return {}
    temporal_config = getattr(getattr(model, "mam", None), "config", None)
    decay_min = float(getattr(temporal_config, "decay_min", 0.0))
    decay_max = float(getattr(temporal_config, "decay_max", 1.0))
    max_offset = float(getattr(temporal_config, "max_residual_offset_px", 0.0))
    span = max(decay_max - decay_min, 1e-8)
    saturation = (
        (decay <= decay_min + 0.01 * span)
        | (decay >= decay_max - 0.01 * span)
    ).float().mean()
    contribution = (gamma * (state - raw)).abs().mean()
    raw_scale = raw.abs().mean().clamp_min(1e-8)
    gamma_joint = gamma.mean(dim=(0, 1, 3, 4)).cpu().tolist()
    decay_joint = decay.mean(dim=(0, 1, 3, 4)).cpu().tolist()
    return {
        "mam_gamma_mean": float(gamma.mean().cpu()),
        "mam_gamma_std": float(gamma.std(unbiased=False).cpu()),
        "mam_decay_mean": float(decay.mean().cpu()),
        "mam_decay_std": float(decay.std(unbiased=False).cpu()),
        "mam_gate_saturation_ratio": float(saturation.cpu()),
        "mam_state_contribution_ratio": float((contribution / raw_scale).cpu()),
        "mam_residual_offset_mean": float(residual.norm(dim=-1).mean().cpu()),
        "mam_residual_offset_saturation_ratio": (
            float((residual.abs() >= 0.99 * max_offset).float().mean().cpu())
            if max_offset > 0.0 else 0.0
        ),
        "mam_gamma_per_joint": json.dumps(gamma_joint),
        "mam_decay_per_joint": json.dumps(decay_joint),
    }


def _heatmap_targets(
    batch: dict, device: torch.device, data: dict, temporal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    heatmap_key = "temporal_heatmaps" if temporal else "heatmaps"
    visibility_key = "temporal_visibility" if temporal else "visibility"
    visibility = batch[visibility_key].to(device, non_blocking=True)
    if heatmap_key in batch:
        target = batch[heatmap_key].to(device, non_blocking=True)
    else:
        keypoint_key = "temporal_keypoints" if temporal else "keypoints"
        keypoints = batch[keypoint_key].to(device, non_blocking=True)
        target = generate_gaussian_heatmaps_torch(
            keypoints, visibility,
            int(data["image_size"]), int(data["heatmap_size"]), float(data["sigma"]),
        )
    return target, visibility


def _prediction_loss(
    prediction: torch.Tensor, batch: dict, criterion,
    device: torch.device, data: dict,
) -> torch.Tensor:
    target, visibility = _heatmap_targets(batch, device, data)
    return criterion(prediction, target, visibility)


def _soft_heatmap_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable heatmap coordinates, preserving leading dimensions."""
    height, width = heatmaps.shape[-2:]
    probability = heatmaps.flatten(-2).softmax(-1)
    y, x = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)
    return probability @ coordinates


def _masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                      visibility: torch.Tensor) -> torch.Tensor:
    valid = (
        visibility.bool().unsqueeze(-1)
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(
        prediction[valid], target[valid], reduction="mean",
    )


_MAM_LOSS_NAMES = ("offset", "prediction", "velocity", "acceleration", "magnitude")


def _scheduled_mam_weights(training: dict, epoch: int | None) -> dict[str, float]:
    schedule = training.get("loss_schedule")
    if not schedule:
        legacy = training.get("mam_loss_weights", {})
        return {name: float(legacy.get(name, 0.0)) for name in _MAM_LOSS_NAMES}
    current_epoch = int(epoch or 1)
    result = {}
    for name in _MAM_LOSS_NAMES:
        spec = schedule.get(name, {})
        if isinstance(spec, (int, float)):
            result[name] = float(spec)
            continue
        start = int(spec.get("start_epoch", spec.get("start", 1)))
        end = spec.get("end_epoch", spec.get("end"))
        active = current_epoch >= start and (
            end is None or current_epoch <= int(end)
        )
        result[name] = float(spec.get("weight", 0.0)) if active else 0.0
    return result


def _mam_auxiliary_losses(
    model, prediction: torch.Tensor, batch: dict,
    device: torch.device, data: dict, training: dict, *,
    epoch: int | None = None,
    target_hm: torch.Tensor | None = None,
    visibility: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    zero = prediction.new_zeros(())
    losses = {f"loss_{name}": zero for name in _MAM_LOSS_NAMES}
    losses["loss_aux_total"] = zero
    auxiliary = getattr(model, "last_mam_aux", {})
    if not auxiliary:
        return losses
    weights = _scheduled_mam_weights(training, epoch)
    if max(weights.values(), default=0.0) <= 0.0:
        return losses
    if target_hm is None or visibility is None:
        target_hm, visibility = _heatmap_targets(
            batch, device, data, temporal=True,
        )
        target_hm = target_hm.transpose(0, 1)
        visibility = visibility.transpose(0, 1)
    total = zero

    predicted_next = auxiliary.get("prediction_next")
    prediction_weight = weights["prediction"]
    if prediction_weight > 0.0 and predicted_next is not None and predicted_next.shape[0] > 1:
        losses["loss_prediction"] = VisibleHeatmapMSE()(
            predicted_next[:-1].flatten(0, 1), target_hm[1:].flatten(0, 1),
            visibility[1:].flatten(0, 1),
        )
        total = total + prediction_weight * losses["loss_prediction"]

    offset_weight = weights["offset"]
    velocity_weight = weights["velocity"]
    acceleration_weight = weights["acceleration"]
    magnitude_weight = weights["magnitude"]
    if max(offset_weight, velocity_weight, acceleration_weight, magnitude_weight) <= 0.0:
        losses["loss_aux_total"] = total
        return losses

    predicted_xy = _soft_heatmap_coordinates(prediction)
    target_xy = batch["temporal_keypoints"].to(device, non_blocking=True).transpose(0, 1)
    scale = float(data["heatmap_size"]) / float(data["image_size"])
    target_xy = target_xy * scale
    visible = visibility > 0
    pair_visible = visible[1:] & visible[:-1]
    if prediction.shape[0] > 1:
        predicted_velocity = predicted_xy[1:] - predicted_xy[:-1]
        target_velocity = target_xy[1:] - target_xy[:-1]
        if velocity_weight > 0.0:
            losses["loss_velocity"] = _masked_smooth_l1(
                predicted_velocity, target_velocity, pair_visible,
            )
            total = total + velocity_weight * losses["loss_velocity"]
        offsets = auxiliary.get("offset")
        if offset_weight > 0.0 and offsets is not None:
            losses["loss_offset"] = _masked_smooth_l1(
                offsets[1:], target_velocity, pair_visible,
            )
            total = total + offset_weight * losses["loss_offset"]
    if prediction.shape[0] > 2:
        predicted_acceleration = predicted_xy[2:] - 2 * predicted_xy[1:-1] + predicted_xy[:-2]
        target_acceleration = target_xy[2:] - 2 * target_xy[1:-1] + target_xy[:-2]
        triple_visible = pair_visible[1:] & pair_visible[:-1]
        if acceleration_weight > 0.0:
            losses["loss_acceleration"] = _masked_smooth_l1(
                predicted_acceleration, target_acceleration, triple_visible,
            )
            total = total + acceleration_weight * losses["loss_acceleration"]
        if magnitude_weight > 0.0:
            finite = (
                triple_visible
                & torch.isfinite(predicted_acceleration).all(-1)
                & torch.isfinite(target_acceleration).all(-1)
            )
            if bool(finite.any()):
                losses["loss_magnitude"] = (
                    predicted_acceleration[finite].norm(dim=-1)
                    - target_acceleration[finite].norm(dim=-1)
                ).abs().mean()
            else:
                losses["loss_magnitude"] = predicted_acceleration.sum() * 0.0
            total = total + magnitude_weight * losses["loss_magnitude"]
    losses["loss_aux_total"] = total
    return losses


def _mam_auxiliary_loss(model, prediction: torch.Tensor, batch: dict,
                        device: torch.device, data: dict,
                        training: dict) -> torch.Tensor:
    """Backward-compatible scalar wrapper used by existing tests/tools."""
    return _mam_auxiliary_losses(
        model, prediction, batch, device, data, training,
    )["loss_aux_total"]


def _loss(model, batch: dict, criterion, device: torch.device,
          all_frames: bool, data: dict, training: dict | None = None,
          epoch: int | None = None) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    if all_frames:
        prediction = model.forward_per_step(image)
        target, visibility = _heatmap_targets(batch, device, data, temporal=True)
        target = target.transpose(0, 1)
        visibility = visibility.transpose(0, 1)
        heatmap_loss = criterion(
            prediction.flatten(0, 1), target.flatten(0, 1),
            visibility.flatten(0, 1),
        )
        auxiliary = _mam_auxiliary_losses(
            model, prediction, batch, device, data, training or {},
            epoch=epoch, target_hm=target, visibility=visibility,
        )
        total = heatmap_loss + auxiliary["loss_aux_total"]
        model.last_loss_breakdown = {
            "loss_pose": heatmap_loss.detach(),
            **{name: value.detach() for name, value in auxiliary.items()},
            "loss_total": total.detach(),
        }
        return total
    prediction = model(image)
    total = _prediction_loss(prediction, batch, criterion, device, data)
    model.last_loss_breakdown = {
        "loss_pose": total.detach(), "loss_total": total.detach(),
    }
    return total


def _epoch(model, loader, criterion, device, optimizer=None,
           gradient_clip: float | None = None, freeze_bn: bool = False,
           all_frames: bool = False,
           gradient_accumulation: int = 1,
           mixed_precision: bool = False,
           scaler=None, data: dict | None = None,
           firing_rate_batches: int = 0,
           progress_interval: int = 0,
           progress_label: str = "",
           epoch: int | None = None) -> tuple[float, float, dict[str, float]]:
    training = optimizer is not None
    gradient_accumulation = int(gradient_accumulation)
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    model.train(training)
    if training and freeze_bn:
        _freeze_batch_norm(model)
    total = count = 0
    loss_totals: dict[str, float] = {}
    gradient_norms: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    if data is None:
        raise ValueError("data heatmap configuration is required")
    firing_rate_batches = max(int(firing_rate_batches), 0)
    diagnostic_batches = max(int(
        getattr(model, "training_config", {}).get("mam_diagnostic_batches", 1)
    ), 0)
    diagnostic_rows: list[dict[str, float | str]] = []
    amp_skipped_steps = 0
    amp_overflow_rows: list[dict[str, object]] = []
    firing_monitor = (
        _StageFiringMonitor(model) if training and firing_rate_batches else None
    )
    if training:
        optimizer.zero_grad(set_to_none=True)
    try:
        with context:
            for batch_index, batch in enumerate(loader):
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(mixed_precision and device.type == "cuda"),
                ):
                    loss = _loss(
                        model, batch, criterion, device, all_frames, data,
                        getattr(model, "training_config", None), epoch=epoch,
                    )
                if batch_index < diagnostic_batches:
                    diagnostic_rows.append(_mam_batch_diagnostics(model))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at {progress_label or 'epoch'} "
                        f"batch={batch_index + 1}; aborting before backward"
                    )
                if (
                    firing_monitor is not None
                    and batch_index + 1 >= firing_rate_batches
                ):
                    firing_monitor.close()
                if training:
                    group_start = batch_index - (batch_index % gradient_accumulation)
                    group_size = min(gradient_accumulation, len(loader) - group_start)
                    scaled_loss = loss / group_size
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    group_end = (
                        (batch_index + 1) % gradient_accumulation == 0
                        or batch_index + 1 == len(loader)
                    )
                    if group_end:
                        amp_enabled = bool(
                            scaler is not None and scaler.is_enabled()
                        )
                        if amp_enabled:
                            scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(gradient_clip) if gradient_clip is not None else float("inf"),
                        )
                        if not bool(torch.isfinite(norm)):
                            nonfinite_parameters = []
                            for name, parameter in model.named_parameters():
                                if parameter.grad is None:
                                    continue
                                finite = torch.isfinite(parameter.grad)
                                if not bool(finite.all()):
                                    nonfinite_parameters.append({
                                        "name": name,
                                        "nan": int(torch.isnan(parameter.grad).sum()),
                                        "inf": int(torch.isinf(parameter.grad).sum()),
                                    })
                            if not amp_enabled:
                                names = [
                                    str(row["name"])
                                    for row in nonfinite_parameters[:8]
                                ]
                                raise FloatingPointError(
                                    f"Non-finite gradient norm at "
                                    f"{progress_label or 'epoch'} "
                                    f"batch={batch_index + 1}; parameters={names}"
                                )
                            old_scale = float(scaler.get_scale())
                            # unscale_ records genuine Inf/NaN gradients for
                            # GradScaler.update(). If only the aggregate norm
                            # overflowed, force the same conservative backoff.
                            if nonfinite_parameters:
                                scaler.update()
                            else:
                                scaler.update(new_scale=max(old_scale * 0.5, 1.0))
                            new_scale = float(scaler.get_scale())
                            overflow = {
                                "batch": batch_index + 1,
                                "old_scale": old_scale,
                                "new_scale": new_scale,
                                "parameters": nonfinite_parameters,
                            }
                            amp_overflow_rows.append(overflow)
                            amp_skipped_steps += 1
                            names = [
                                str(row["name"])
                                for row in nonfinite_parameters[:8]
                            ]
                            print(
                                f"{progress_label or 'epoch'} "
                                f"AMP overflow batch={batch_index + 1}; "
                                f"optimizer step skipped; scale "
                                f"{old_scale:g}->{new_scale:g}; "
                                f"parameters={names or ['aggregate_norm']}",
                                flush=True,
                            )
                        else:
                            gradient_norms.append(float(norm.detach()))
                            if amp_enabled:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                batch_size = len(batch["image"])
                for name, value in getattr(model, "last_loss_breakdown", {}).items():
                    loss_totals[name] = (
                        loss_totals.get(name, 0.0) + float(value) * batch_size
                    )
                total += float(loss.detach()) * batch_size
                count += batch_size
                if progress_interval > 0 and (
                    (batch_index + 1) % progress_interval == 0
                    or batch_index + 1 == len(loader)
                ):
                    print(
                        f"{progress_label} batch={batch_index + 1}/{len(loader)} "
                        f"loss={total / max(count, 1):.6f}",
                        flush=True,
                    )
    finally:
        if firing_monitor is not None:
            firing_monitor.close()
    mean_gradient_norm = (
        float(np.mean(gradient_norms)) if gradient_norms else float("nan")
    )
    firing_rates = firing_monitor.summary() if firing_monitor is not None else {}
    model.last_epoch_loss_breakdown = {
        name: value / max(count, 1) for name, value in loss_totals.items()
    }
    model.last_amp_skipped_steps = amp_skipped_steps
    model.last_amp_overflow_rows = amp_overflow_rows
    numeric_diagnostics = {
        key: float(np.mean([
            float(row[key]) for row in diagnostic_rows if key in row
        ]))
        for key in {
            key for row in diagnostic_rows for key, value in row.items()
            if not isinstance(value, str)
        }
    }
    string_diagnostics = next((
        {key: value for key, value in row.items() if isinstance(value, str)}
        for row in reversed(diagnostic_rows) if row
    ), {})
    model.last_epoch_mam_diagnostics = {
        **numeric_diagnostics, **string_diagnostics,
    }
    return total / max(count, 1), mean_gradient_norm, firing_rates


def _write_history(path: Path, history: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(
        key for row in history for key in row
    ))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_experiment(name: str, seed: int, device: str, output_root: Path | None = None,
                     max_train_samples: int | None = None,
                     max_validation_samples: int | None = None,
                     source_experiment: str | None = None,
                     allow_tbd: bool = False,
                     profile: str | None = None,
                     resume: bool = False,
                     epochs_override: int | None = None,
                     ntu_setups: str | None = None) -> Path:
    config = load_experiment(name, profile=profile)
    config = apply_pilot_runtime_overrides(
        config, epochs=epochs_override, ntu_setups=ntu_setups,
    )
    from spikepose_thesis.experiments import experiment_readiness
    readiness_config = config
    if source_experiment:
        readiness_config = {
            **config,
            "initialization": {
                **config.get("initialization", {}), "source": source_experiment,
            },
        }
    readiness = experiment_readiness(readiness_config, seed, output_root)
    allowed_statuses = {"READY", "RESUMABLE"} if resume else {"READY"}
    if readiness["status"] not in allowed_statuses and not allow_tbd:
        raise RuntimeError(
            f"{name} seed={seed} is {readiness['status']}: "
            + "; ".join(readiness["blockers"])
        )
    if seed not in config["training"]["seeds"]:
        raise ValueError(f"Seed {seed} is not registered for {name}")
    if config.get("action") == "refinement":
        raise ValueError("Use the refinement command for R-series experiments")
    artifacts = RunArtifacts(config, seed, output_root)
    resume_path = artifacts.path / "checkpoints" / "last.pt"
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Cannot resume without checkpoint: {resume_path}")
    else:
        artifacts.initialize()
    seed_everything(seed)
    device_value = torch.device(device)
    model = build_model(config).to(device_value)
    model.training_config = config["training"]
    model.experiment_id = config["id"]
    model.model_name = config["paper_id"]
    initialization = config.get("initialization", {"mode": "scratch"})
    if initialization.get("mode") == "checkpoint" and not resume:
        source = source_experiment or initialization["source"]
        source_root = (
            resolve_project_path(initialization["source_output_root"])
            if initialization.get("source_output_root") else artifacts.output_root
        )
        source_path = locate_source_run(source, seed, source_root)
        checkpoint = source_path / "checkpoints" / "best.pt"
        load_kind = initialization.get("load", "weights_only")
        initialization_load_report = None
        if load_kind == "spatial_weights_only":
            _, initialization_load_report = load_spatial_model(
                checkpoint, model, device_value,
            )
        elif load_kind == "mpii16_to_ntu25":
            _, initialization_load_report = load_mpii16_to_ntu25_model(
                checkpoint, model, device_value,
            )
        else:
            load_model(checkpoint, model, device_value)
        artifacts.write_json("lineage.json", {
            "source_experiment": source, "source_seed": seed,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_fingerprint": file_fingerprint(checkpoint),
            "load": load_kind,
            "load_report": initialization_load_report,
            "spatial_load_report": (
                initialization_load_report
                if load_kind == "spatial_weights_only" else None
            ),
        })
    elif initialization.get("mode") == "official_checkpoint" and not resume:
        checkpoint = resolve_project_path(initialization["path"])
        load_report = load_official_checkpoint(
            checkpoint, model, device_value,
        )
        artifacts.write_json("lineage.json", {
            "source_kind": "official_checkpoint",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_fingerprint": file_fingerprint(checkpoint),
            "format": initialization.get("format", "official_hrnet"),
            "load_report": load_report,
        })
    train_set = build_dataset(config, "train", max_train_samples)
    validation_set = build_dataset(config, "validation", max_validation_samples)
    train_loader = _loader(train_set, config["training"], True, seed)
    validation_loader = _loader(validation_set, config["training"], False, seed)
    if config.get("action") == "evaluate_only":
        summary = evaluate_checkpoint(
            model, validation_loader, config, device_value, artifacts.path / "predictions",
        )
        artifacts.status("completed", validation=summary)
        return artifacts.path
    criterion = VisibleHeatmapMSE()
    training = config["training"]
    epochs = int(training["epochs"])
    best = -float("inf")
    best_acceleration = float("inf")
    history: list[dict] = []
    optimizer = scheduler = None
    current_phase = object()
    mixed_precision = bool(training.get("mixed_precision", False))
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(mixed_precision and device_value.type == "cuda"),
        )
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(mixed_precision and device_value.type == "cuda"),
        )
    start_epoch = 1
    if resume:
        metadata = load_checkpoint(resume_path, "cpu")
        saved_epoch = int(metadata["epoch"])
        if saved_epoch >= epochs:
            raise ValueError(
                f"Checkpoint epoch {saved_epoch} already reaches target {epochs}"
            )
        saved_config = metadata.get("config", {})
        if saved_config.get("id") not in {None, config["id"]}:
            raise ValueError("Resume checkpoint belongs to a different experiment")
        phase = _phase_for_epoch(training, saved_epoch)
        current_phase = phase.get("name") if phase else "full"
        optimizer = _configure_trainable(model, training, phase)
        scheduler = build_scheduler(optimizer, training)
        checkpoint = restore_training_state(
            resume_path, model, optimizer, scheduler, device_value,
            scaler=scaler, data_loader_generator=train_loader.generator,
        )
        start_epoch = saved_epoch + 1
        artifacts.write_json(
            f"training/resume_target_{saved_epoch:03d}_to_{epochs:03d}.json",
            config,
        )
        best = float(checkpoint.get("best_metric", -float("inf")))
        history = list(checkpoint.get("history", []))
        acceleration_values = [
            float(row["validation_acceleration_error"])
            for row in history
            if row.get("validation_acceleration_error") is not None
            and np.isfinite(row["validation_acceleration_error"])
        ]
        best_acceleration = min(acceleration_values, default=float("inf"))
    artifacts.status(
        "running", stage="training", resumed=resume,
        start_epoch=start_epoch, target_epoch=epochs,
    )
    for epoch in range(start_epoch, epochs + 1):
        phase = _phase_for_epoch(training, epoch)
        phase_name = phase.get("name") if phase else "full"
        if phase_name != current_phase:
            optimizer = _configure_trainable(model, training, phase)
            scheduler = build_scheduler(optimizer, training)
            current_phase = phase_name
        scheduler.set_epoch(epoch)
        started = time.time()
        if device_value.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_value)
        train_loss, gradient_norm, firing_rates = _epoch(
            model, train_loader, criterion, device_value, optimizer,
            training.get("gradient_clip"), bool(phase and phase.get("freeze_batch_norm")),
            bool(training.get("loss_all_frames", False)),
            int(training.get("gradient_accumulation", 1)),
            mixed_precision, scaler, config["data"],
            int(training.get("firing_rate_batches", 0)),
            int(training.get("progress_interval", 0)),
            f"{name} seed={seed} epoch={epoch}/{epochs} train",
            epoch=epoch,
        )
        train_loss_breakdown = dict(
            getattr(model, "last_epoch_loss_breakdown", {})
        )
        train_mam_diagnostics = dict(
            getattr(model, "last_epoch_mam_diagnostics", {})
        )
        train_amp_skipped_steps = int(
            getattr(model, "last_amp_skipped_steps", 0)
        )
        train_amp_overflows = list(
            getattr(model, "last_amp_overflow_rows", [])
        )
        collect_temporal_frames = bool(
            config["dataset"] != "mpii"
            and training.get("loss_all_frames", False)
        )
        if training.get("single_pass_validation", False):
            collected = collect_predictions(
                model, validation_loader, config, device_value, decoder="argmax",
                loss_callback=lambda prediction, batch: _prediction_loss(
                    prediction, batch, criterion, device_value, config["data"],
                ),
                collect_temporal_frames=collect_temporal_frames,
            )
            if collect_temporal_frames:
                validation_values, temporal_values, val_loss = collected
            else:
                validation_values, val_loss = collected
                temporal_values = validation_values
        else:
            val_loss, _, _ = _epoch(
                model, validation_loader, criterion, device_value,
                mixed_precision=mixed_precision, data=config["data"],
                progress_interval=int(training.get("progress_interval", 0)),
                progress_label=f"{name} seed={seed} epoch={epoch}/{epochs} validation",
                epoch=epoch,
            )
            collected = collect_predictions(
                model, validation_loader, config, device_value, decoder="argmax",
                collect_temporal_frames=collect_temporal_frames,
            )
            if collect_temporal_frames:
                validation_values, temporal_values = collected
            else:
                validation_values = collected
                temporal_values = validation_values
        validation_metrics = summarize_predictions(validation_values)
        if config["dataset"] != "mpii":
            validation_metrics["temporal"] = temporal_summary(
                temporal_values,
                strict=collect_temporal_frames,
                expected_length=(
                    int(config["temporal"]["video_frames"])
                    if collect_temporal_frames else None
                ),
            )
        score = float(validation_metrics["pck_0.5"])
        pck_hb = score
        if config["dataset"] == "mpii":
            pck_hb = float(summarize_predictions({
                **validation_values, "scale": validation_values["scale_hb"],
            })["pck_0.5"])
        invalid_prediction_ratio = float(
            1.0 - np.isfinite(validation_values["prediction"]).all(axis=-1).mean()
        )
        temporal_metrics = validation_metrics.get("temporal", {})
        target_frames_per_batch = int(train_loader.batch_size) * (
            int(config["temporal"]["video_frames"])
            if training.get("loss_all_frames") else 1
        )
        accumulation = int(training.get("gradient_accumulation", 1))
        peak_memory_mib = (
            float(torch.cuda.max_memory_allocated(device_value) / (1024 ** 2))
            if device_value.type == "cuda" else 0.0
        )
        scheduler.step(val_loss)
        row = {
            "epoch": epoch, "phase": phase_name, "train_loss": train_loss,
            "val_loss": val_loss, "validation_pck": score,
            "validation_pck_hb": pck_hb,
            "validation_pck_auc": validation_metrics["pck_auc_0_0_5"],
            "validation_nme_hb": validation_metrics["nme_hb"],
            "validation_velocity_error": temporal_metrics.get("velocity_error"),
            "validation_acceleration_error": temporal_metrics.get("acceleration_error"),
            "validation_acceleration_error_hb": temporal_metrics.get(
                "acceleration_error_hb"
            ),
            "validation_relative_acceleration_error": temporal_metrics.get(
                "relative_acceleration_error"
            ),
            "validation_acceleration_magnitude_ratio": temporal_metrics.get(
                "acceleration_magnitude_ratio"
            ),
            "validation_velocity_magnitude_ratio": temporal_metrics.get(
                "velocity_magnitude_ratio"
            ),
            "validation_jerk_error": temporal_metrics.get("jerk_error"),
            "validation_high_motion_pck": temporal_metrics.get(
                "high_motion_pck_0_5"
            ),
            "validation_high_motion_acceleration_error": temporal_metrics.get(
                "high_motion_acceleration_error"
            ),
            "validation_peak_lag_frames": temporal_metrics.get("peak_lag_frames"),
            "validation_peak_lag_absolute_frames": temporal_metrics.get(
                "peak_lag_absolute_frames"
            ),
            "validation_temporal_segments": temporal_metrics.get(
                "contiguous_segments"
            ),
            "gradient_norm": gradient_norm,
            "amp_skipped_steps": train_amp_skipped_steps,
            "amp_overflows": json.dumps(train_amp_overflows, sort_keys=True),
            "amp_scale": (
                float(scaler.get_scale()) if scaler.is_enabled() else None
            ),
            "gpu_peak_memory_mib": peak_memory_mib,
            "invalid_prediction_ratio": invalid_prediction_ratio,
            "firing_rates": json.dumps(firing_rates, sort_keys=True),
            "clips_per_batch": int(train_loader.batch_size),
            "target_frames_per_batch": target_frames_per_batch,
            "gradient_accumulation": accumulation,
            "effective_target_frames": target_frames_per_batch * accumulation,
            "optimizer_steps_per_epoch": (
                (len(train_loader) + accumulation - 1) // accumulation
            ),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            **train_loss_breakdown,
            **train_mam_diagnostics,
        }
        history.append(row)
        save_checkpoint(
            artifacts.path / "checkpoints" / "last.pt", model, optimizer, scheduler,
            epoch, max(best, score), config, history,
            scaler=scaler, data_loader_generator=train_loader.generator,
        )
        if score > best:
            best = score
            save_checkpoint(
                artifacts.path / "checkpoints" / "best.pt", model, optimizer, scheduler,
                epoch, best, config, history,
                scaler=scaler, data_loader_generator=train_loader.generator,
            )
        acceleration = temporal_metrics.get("acceleration_error")
        if acceleration is not None and np.isfinite(acceleration) and acceleration < best_acceleration:
            best_acceleration = float(acceleration)
            save_checkpoint(
                artifacts.path / "checkpoints" / "best_acceleration.pt",
                model, optimizer, scheduler, epoch, -best_acceleration, config, history,
                scaler=scaler, data_loader_generator=train_loader.generator,
            )
        checkpoint_interval = int(training.get("checkpoint_interval", 0))
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(
                artifacts.path / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, max(best, score), config, history,
                scaler=scaler, data_loader_generator=train_loader.generator,
            )
        if epoch in {int(value) for value in training.get("milestone_epochs", [])}:
            save_checkpoint(
                artifacts.path / "checkpoints" / f"milestone_{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, max(best, score), config, history,
                scaler=scaler, data_loader_generator=train_loader.generator,
            )
        _write_history(artifacts.path / "training" / "history.csv", history)
        print(
            f"{name} seed={seed} epoch={epoch}/{epochs} train={train_loss:.6f} "
            f"val={val_loss:.6f} pck_hb={pck_hb:.6f} "
            f"acce={temporal_metrics.get('acceleration_error', float('nan')):.6f}",
            flush=True,
        )
    load_model(artifacts.path / "checkpoints" / "best.pt", model, device_value)
    train_evaluation_set = build_dataset(config, "train_evaluation", max_train_samples)
    train_evaluation_loader = _loader(
        train_evaluation_set, config["training"], False, seed,
    )
    evaluate_checkpoint(
        model, train_evaluation_loader, config, device_value,
        artifacts.path / "predictions" / "train",
    )
    summary = evaluate_checkpoint(
        model, validation_loader, config, device_value,
        artifacts.path / "predictions" / "validation",
    )
    artifacts.status(
        "completed", completed_epoch=epochs,
        best_validation_pck=best, validation=summary,
    )
    return artifacts.path
