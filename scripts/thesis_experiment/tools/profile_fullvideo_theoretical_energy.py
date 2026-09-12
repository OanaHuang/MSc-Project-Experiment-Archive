#!/usr/bin/env python3
"""Profile one frozen Table 1 model on the NTU full-video subset.

Temporal models are averaged over the same exhaustive 16-frame invocations
used by ``evaluate_fullvideo_subset.py``. Frame-wise models are profiled per
real image and their input-independent arithmetic cost is reported for 16
independent frame invocations so every Table 1 row has the same scope.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader
import yaml

from spikepose_thesis.analysis import profile_theoretical_dataset
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path
from spikepose_thesis.data import build_dataset
from spikepose_thesis.models import build_model
from spikepose_thesis.training.checkpoint import load_model


DEFAULT_METADATA = "Datasets/NTU_RGBD/metadata/fullvideo_xsub35/test_split.csv"
DEFAULT_FRAMES = "Datasets/NTU_RGBD/extracted_frames_full"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile theoretical energy on the frozen NTU subset.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--report-frames", type=int, default=16)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def _load_config(args: argparse.Namespace) -> tuple[dict, str]:
    if args.resolved_config is None:
        return load_experiment(args.experiment), "experiment_registry"
    source = args.resolved_config.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("id") != args.experiment:
        raise ValueError(
            f"Resolved config id {config.get('id')!r} does not match "
            f"--experiment {args.experiment!r}"
        )
    return config, str(source)


def _select_exhaustive_chunks(dataset) -> tuple[int, int]:
    """Match the non-duplicative full-video evaluation invocation schedule."""
    steps = int(getattr(dataset, "temporal_steps", 1))
    if steps <= 1:
        return len(dataset.frame_index), len(dataset.frame_index)
    if int(getattr(dataset, "temporal_frame_gap", 1)) != 1:
        raise ValueError("Full-video profiling requires temporal_frame_gap=1")

    last_position: dict[int, int] = {}
    for sample_index, _frame, _clip, position in dataset.frame_index:
        last_position[sample_index] = max(
            int(position), last_position.get(sample_index, -1),
        )
    if len(last_position) != len(dataset.samples):
        raise RuntimeError("Some dataset videos have no complete temporal window")
    dataset.frame_index = [
        item for item in dataset.frame_index
        if (
            int(item[3]) % steps == steps - 1
            or int(item[3]) == last_position[int(item[0])]
        )
    ]
    expected_frames = sum(position + 1 for position in last_position.values())
    return len(dataset.frame_index), expected_frames


def _scale_cost_scope(report: dict, factor: int) -> dict:
    """Scale one model invocation to a shared number of physical frames."""
    if factor < 1:
        raise ValueError("Cost scale factor must be positive")
    scalar_fields = (
        "analog_macs", "spiking_dense_macs", "dense_macs", "dense_flops",
        "legacy_conv_linear_flops", "end_to_end_flops", "end_to_end_gflops",
        "effective_sops", "analog_mac_energy_mj", "spiking_ac_energy_mj",
        "total_theoretical_energy_mj", "ann_architecture_macs",
        "ann_architecture_energy_mj", "ann_equivalent_macs",
        "ann_equivalent_energy_mj", "legacy_single_step_ann_equivalent_macs",
        "legacy_single_step_ann_equivalent_energy_mj",
    )
    if factor != 1:
        for key in scalar_fields:
            if report.get(key) is not None:
                report[key] = float(report[key]) * factor
        report["end_to_end_operator_flops"] = {
            key: float(value) * factor
            for key, value in report["end_to_end_operator_flops"].items()
        }
        for layer in report["layers"]:
            for key in (
                "invocations_per_window", "dense_macs",
                "ann_architecture_macs", "ann_equivalent_macs",
                "legacy_single_step_ann_equivalent_macs",
                "effective_operations", "energy_mj",
            ):
                layer[key] = float(layer[key]) * factor
    return report


def main() -> None:
    args = _parser().parse_args()
    if args.report_frames < 1:
        raise ValueError("report-frames must be positive")
    source_config, config_source = _load_config(args)
    config = deepcopy(source_config)
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    metadata = resolve_project_path(args.metadata).resolve()
    frames_dir = resolve_project_path(args.frames_dir).resolve()
    for path in (checkpoint, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)

    config["data"].update({
        "frames_dir": str(frames_dir.relative_to(PROJECT_ROOT)),
        "frame_layout": "flat",
        "test_metadata": str(metadata.relative_to(PROJECT_ROOT)),
        "exclude_overlapping_clips": False,
        "frame_stride": 1,
        "temporal_frame_gap": 1,
        "runtime_spatial_crops": False,
        "runtime_cache_mode": "optional",
        "pose_validation_mode": "skip",
    })
    temporal = config["model"].get("temporal", {})
    model_frames = int(temporal.get("video_frames", 1))
    temporal_input = temporal.get("input_strategy") in {
        "frames", "scheduled_frames",
    }
    if temporal_input:
        config["data"]["minimum_temporal_history"] = model_frames - 1
        config.setdefault("training", {})["loss_all_frames"] = True
    else:
        config["data"]["minimum_temporal_history"] = 0
    if args.report_frames % model_frames:
        raise ValueError(
            "report-frames must be divisible by the model input frame count"
        )

    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint_metadata = load_model(checkpoint, model, device)
    dataset = build_dataset(config, "test", args.max_videos)
    if temporal_input:
        model_inputs, physical_frames = _select_exhaustive_chunks(dataset)
    else:
        model_inputs = len(dataset)
        physical_frames = len(dataset)
    batch_size = args.batch_size or (
        int(config["training"].get("clip_batch_size", 8))
        if temporal_input else int(config["training"].get("batch_size", 32))
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
    )
    progress_interval = max(int(args.progress_interval), 1)

    def real_images():
        started = time.monotonic()
        total = min(
            len(loader),
            args.max_batches if args.max_batches is not None else len(loader),
        )
        for index, batch in enumerate(loader, start=1):
            if args.max_batches is not None and index > args.max_batches:
                break
            if index == 1 or index % progress_interval == 0 or index == total:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = index / elapsed
                remaining = (total - index) / rate if rate > 0 else float("nan")
                print(
                    f"progress batches={index}/{total} "
                    f"elapsed_min={elapsed / 60:.1f} "
                    f"eta_min={remaining / 60:.1f}",
                    flush=True,
                )
            yield batch["image"].to(
                device, non_blocking=torch.cuda.is_available(),
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    scale_factor = args.report_frames // model_frames
    report = profile_theoretical_dataset(
        model, real_images(), output,
        video_steps=model_frames,
        snn_steps_per_frame=int(temporal.get("snn_steps_per_frame", 1)),
        metadata={
            "experiment": args.experiment,
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "checkpoint_sha256": _sha256(checkpoint),
            "config_source": config_source,
            "split": "fullvideo_xsub35_test",
            "split_metadata": str(metadata),
            "split_metadata_sha256": _sha256(metadata),
            "frames_dir": str(frames_dir),
            "git_commit": _git_head(),
            "available_model_inputs": model_inputs,
            "available_physical_frames": physical_frames,
            "max_videos": args.max_videos,
            "max_batches": args.max_batches,
            "batch_size": batch_size,
            "workers": args.workers,
            "model_input_frames": model_frames,
            "report_frames": args.report_frames,
            "cost_scale_factor": scale_factor,
            "temporal_input": temporal_input,
        },
    )
    profiled_model_inputs = int(report["profiled_windows"])
    report = _scale_cost_scope(report, scale_factor)
    report["profiled_model_inputs"] = profiled_model_inputs
    report["profiled_physical_frames"] = profiled_model_inputs * model_frames
    report["normalization"] = f"per {args.report_frames}-frame window"
    report["assumptions"]["input_scope"] = (
        f"mean arithmetic cost per {args.report_frames} physical frames on the "
        "frozen fullvideo_xsub35 test subset"
    )
    report["assumptions"]["framewise_scaling"] = (
        "frame-wise models use identical independent arithmetic on each frame; "
        f"their per-image cost is multiplied by {scale_factor}"
        if scale_factor != 1 else "not applied"
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "experiment": args.experiment,
        "profiled_model_inputs": profiled_model_inputs,
        "profiled_physical_frames": report["profiled_physical_frames"],
        "normalization": report["normalization"],
        "total_theoretical_energy_mj": report["total_theoretical_energy_mj"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
