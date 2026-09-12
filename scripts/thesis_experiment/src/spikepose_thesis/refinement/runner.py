from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spikepose_thesis.artifacts import RunArtifacts, locate_source_run
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import default_output_root
from spikepose_thesis.evaluation.runner import _ntu_group_summaries, summarize_predictions
from spikepose_thesis.evaluation.temporal import (
    contiguous_sequence_groups, sequence_groups, temporal_summary,
)

from .filters import OneEuroFilter, ema_shared, savgol_causal, savgol_offline
from .jtr import (
    CausalTCNLite, DynamicTemporalRefinement, JointwiseTemporalRefinement,
    jtr_loss,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def _source_prediction_path(source_run: Path, split: str, method: str) -> Path:
    # A learned temporal refiner needs all 16 observations in each source clip.
    # The default archive contains only the model readout (one row per clip).
    filename = (
        "predictions_per_frame.npz"
        if method == "jtr_jointwise" else "predictions.npz"
    )
    return source_run / "predictions" / split / filename


def _coordinate_scale(value: float) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("refinement coordinate_scale must be finite and positive")
    return scale


def _clips(values: dict[str, np.ndarray], length: int, stride: int,
           coordinate_scale: float):
    normalization = _coordinate_scale(coordinate_scale)
    inputs, targets, masks = [], [], []
    for contiguous in contiguous_sequence_groups(values):
        if len(contiguous) < length:
            continue
        for start in range(0, len(contiguous) - length + 1, stride):
            selected = contiguous[start:start + length]
            prediction = values["prediction"][selected]
            target = values["target"][selected]
            visibility = values["visibility"][selected]
            prediction_finite = np.isfinite(prediction).all(axis=-1)
            if not prediction_finite.all():
                # A non-finite observation would enter and persist in the
                # causal state, so the complete temporal window is unusable.
                continue
            target_finite = np.isfinite(target).all(axis=-1)
            valid = (
                (visibility > 0) & np.isfinite(visibility)
                & prediction_finite & target_finite
            )
            safe_target = np.where(
                target_finite[..., None], target, prediction,
            )
            inputs.append(prediction / normalization)
            targets.append(safe_target / normalization)
            masks.append(valid.astype(np.float32))
    if not inputs:
        raise RuntimeError(f"No complete refinement clips of length {length}")
    return tuple(torch.from_numpy(np.asarray(item)).float() for item in (
        inputs, targets, masks,
    ))


def _train_module(module: torch.nn.Module, values: dict[str, np.ndarray], epochs: int,
                  batch_size: int, learning_rate: float, weight_decay: float,
                  gradient_clip: float, device: torch.device, clip_length: int,
                  clip_stride: int, coordinate_scale: float) -> torch.nn.Module:
    tensors = _clips(values, clip_length, clip_stride, coordinate_scale)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    module.to(device).train()
    optimizer = torch.optim.Adam(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    for _ in range(epochs):
        for prediction, target, visibility in loader:
            prediction, target = prediction.to(device), target.to(device)
            visibility = visibility.to(device)
            optimizer.zero_grad(set_to_none=True)
            refined = module(prediction)
            if not torch.isfinite(refined).all():
                raise FloatingPointError("non-finite JTR prediction during training")
            loss = jtr_loss(refined, target, visibility)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite JTR loss during training")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip)
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all()
                for parameter in module.parameters()
            ):
                raise FloatingPointError("non-finite JTR parameter after optimizer step")
    return module.eval()


@torch.no_grad()
def _apply_module(module: torch.nn.Module, values: dict[str, np.ndarray],
                  device: torch.device, coordinate_scale: float) -> dict[str, np.ndarray]:
    normalization = _coordinate_scale(coordinate_scale)
    result = {key: item.copy() for key, item in values.items()}
    for indices in contiguous_sequence_groups(values):
        prediction = values["prediction"][indices]
        if not np.isfinite(prediction).all():
            count = int((~np.isfinite(prediction)).sum())
            raise FloatingPointError(
                f"source refinement prediction contains {count} non-finite values"
            )
        normalized = torch.from_numpy(
            prediction[None] / normalization
        ).float().to(device)
        refined = module(normalized)[0].cpu().numpy() * normalization
        if not np.isfinite(refined).all():
            raise FloatingPointError("non-finite JTR prediction during inference")
        result["prediction"][indices] = refined
    return result


def _apply_filter(values: dict[str, np.ndarray], function) -> dict[str, np.ndarray]:
    result = {key: item.copy() for key, item in values.items()}
    for indices in contiguous_sequence_groups(values):
        result["prediction"][indices] = function(values["prediction"][indices])
    return result


def _summary(values: dict[str, np.ndarray]) -> dict:
    summary = summarize_predictions(values)
    summary["temporal"] = temporal_summary(values)
    summary["groups"] = _ntu_group_summaries(values)
    if "scale_hb" in values:
        summary["pck_hb"] = summarize_predictions({
            **values, "scale": values["scale_hb"],
        })
    return summary


def _selection_score(summary: dict) -> float:
    # Preserve spatial accuracy first and use normalized acceleration as a
    # deterministic validation-only tie breaker.
    return float(
        summary["pck_0.5"]
        - 1e-3 * summary.get("temporal", {}).get("acceleration_error_hb", 0.0)
    )


def _best_classical(config: dict, validation: dict[str, np.ndarray]):
    method = config["refinement"]["method"]
    grid = config["refinement"]["validation_grid"]
    best = None
    if method == "ema_shared":
        for alpha in grid["alpha"]:
            function = lambda value, a=alpha: ema_shared(value, a)
            refined = _apply_filter(validation, function)
            candidate = (
                _selection_score(_summary(refined)),
                {"alpha": float(alpha)},
                function,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    elif method.startswith("savgol"):
        candidates = itertools.product(grid["window"], grid["polynomial_order"])
        for window, order in candidates:
            if method == "savgol_offline":
                function = lambda value, w=window, o=order: savgol_offline(value, w, o)
            else:
                function = lambda value, w=window, o=order: savgol_causal(value, w, o)
            refined = _apply_filter(validation, function)
            candidate = (_selection_score(_summary(refined)),
                         {"window": window, "order": order}, function)
            if best is None or candidate[0] > best[0]:
                best = candidate
    elif method == "one_euro":
        candidates = itertools.product(
            grid["min_cutoff"], grid["beta"], grid["derivative_cutoff"],
        )
        for minimum, beta, derivative in candidates:
            function = lambda value, a=minimum, b=beta, d=derivative: OneEuroFilter(a, b, d)(value)
            refined = _apply_filter(validation, function)
            candidate = (_selection_score(_summary(refined)), {
                "min_cutoff": minimum, "beta": beta, "derivative_cutoff": derivative,
            }, function)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError(f"No classical filter candidates for {method}")
    return best[1], best[2]


def run_refinement(name: str, seed: int, device: str,
                   output_root: Path | None = None,
                   allow_tbd: bool = False,
                   profile: str | None = None) -> Path:
    config = load_experiment(name, profile=profile)
    from spikepose_thesis.experiments import experiment_readiness
    readiness = experiment_readiness(config, seed, output_root)
    if readiness["status"] != "READY" and not allow_tbd:
        raise RuntimeError(
            f"{name} seed={seed} is {readiness['status']}: "
            + "; ".join(readiness["blockers"])
        )
    if config.get("action") != "refinement":
        raise ValueError(f"{name} is not a refinement experiment")
    if seed not in config["training"]["seeds"]:
        raise ValueError(f"Seed {seed} is not registered for {name}")
    source = config["initialization"]["source"]
    effective_root = output_root or default_output_root(
        str(config.get("run_type", "formal")),
    )
    source_run = locate_source_run(source, seed, effective_root)
    method = config["refinement"]["method"]
    allowed_splits = (
        ("train", "validation")
        if config.get("run_type") == "pilot"
        else ("train", "validation", "test")
    )
    source_paths = {
        split: _source_prediction_path(source_run, split, method)
        for split in allowed_splits
    }
    archives = {
        split: _load(path)
        for split, path in source_paths.items() if path.is_file()
    }
    if "train" not in archives or "validation" not in archives:
        raise FileNotFoundError("Refinement requires source train and validation archives")
    artifacts = RunArtifacts(config, seed, output_root)
    artifacts.initialize()
    artifacts.write_json("lineage.json", {
        "source_experiment": source, "source_seed": seed,
        "source_run": str(source_run), "spatial_model_frozen": True,
        "source_prediction_archives": {
            split: str(path) for split, path in source_paths.items()
            if path.is_file()
        },
    })
    device_value = torch.device(device)
    module = None
    parameters = {}
    if method == "raw":
        refined = archives
    elif method in {
        "ema_shared", "savgol_offline", "savgol_causal", "one_euro",
    }:
        parameters, function = _best_classical(config, archives["validation"])
        refined = {key: _apply_filter(value, function) for key, value in archives.items()}
    else:
        protocol = config["refinement_protocol"]
        coordinate_scale = _coordinate_scale(protocol["coordinate_scale"])
        parameters["coordinate_scale"] = coordinate_scale
        if method == "ema_global":
            module = JointwiseTemporalRefinement(16, jointwise=False)
            epochs = int(config["training"]["epochs"])
        elif method == "jtr_jointwise":
            shared = _train_module(
                JointwiseTemporalRefinement(16, jointwise=False), archives["train"],
                int(protocol["shared_epochs"]), int(config["training"]["batch_size"]),
                float(config["training"]["learning_rate"]), 0.0,
                float(config["training"]["gradient_clip"]), device_value,
                int(protocol["clip_length"]), int(protocol["clip_stride"]),
                coordinate_scale,
            )
            module = JointwiseTemporalRefinement(16, jointwise=True)
            module.logits.data.fill_(float(shared.logits.detach().cpu()[0]))
            epochs = int(protocol["jointwise_epochs"])
        elif method == "ema_dynamic":
            module = DynamicTemporalRefinement(16)
            epochs = int(config["training"]["epochs"])
        elif method == "tcn_lite":
            module = CausalTCNLite(
                16, int(config["refinement"]["hidden_channels"]),
                int(config["refinement"]["layers"]), int(config["refinement"]["window"]),
            )
            epochs = int(config["training"]["epochs"])
        else:
            raise ValueError(f"Unknown refinement method: {method}")
        module = _train_module(
            module, archives["train"], epochs, int(config["training"]["batch_size"]),
            float(config["training"]["learning_rate"]),
            float(config["training"].get("weight_decay", 0.0)),
            float(config["training"]["gradient_clip"]), device_value,
            int(protocol["clip_length"]), int(protocol["clip_stride"]),
            coordinate_scale,
        )
        if any(
            not torch.isfinite(parameter).all()
            for parameter in module.parameters()
        ):
            raise FloatingPointError("refinement produced non-finite parameters")
        torch.save({"model_state_dict": module.state_dict(), "config": config},
                   artifacts.path / "checkpoints" / "best.pt")
        refined = {
            key: _apply_module(module, value, device_value, coordinate_scale)
            for key, value in archives.items()
        }
        if isinstance(module, JointwiseTemporalRefinement):
            parameters["alpha"] = module.alpha.detach().cpu().tolist()
    summaries = {}
    for split, values in refined.items():
        path = artifacts.path / "predictions" / split
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "predictions.npz", **values)
        summaries[split] = _summary(values)
        (path / "summary.json").write_text(
            json.dumps(summaries[split], indent=2), encoding="utf-8",
        )
        (artifacts.path / "metrics" / f"{split}.json").write_text(
            json.dumps(summaries[split], indent=2), encoding="utf-8",
        )
    artifacts.write_json("parameters.json", parameters)
    artifacts.status("completed", validation=summaries["validation"])
    return artifacts.path
