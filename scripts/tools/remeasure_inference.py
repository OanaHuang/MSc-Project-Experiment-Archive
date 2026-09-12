from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.datasets import MPIIPoseDataset
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.spikepose.analysis import (
    GpuPowerSampler,
    InferenceMeasurementProtocol,
    benchmark_inference_for_duration,
    gpu_compute_process_count,
    measure_idle_power,
    summarize_repeats,
)
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remeasure inference using the isolated measurement_v2 protocol")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="New-framework run containing config/resolved.yaml and checkpoints")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", type=Path,
                        default=PROJECT_ROOT / "scripts/configs/measurement.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_protocol(path: Path) -> InferenceMeasurementProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    protocol = InferenceMeasurementProtocol(**payload)
    protocol.validate()
    return protocol


def build_validation_dataset(config: dict):
    dataset = config["dataset"]
    if dataset == "mpii":
        data = config["data"]
        return MPIIPoseDataset(
            PROJECT_ROOT / data["validation_metadata"],
            PROJECT_ROOT / data["images_dir"],
            data["image_size"], data["heatmap_size"], data["sigma"],
            data["crop_expansion"], False,
        )
    if dataset == "ntu_rgbd":
        return build_dataset(PROJECT_ROOT, config, "validation")
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def write_flat_summary(path: Path, reports: list[dict]) -> None:
    keys = sorted({key for report in reports for key in report})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repeat", *keys])
        writer.writeheader()
        for index, report in enumerate(reports, 1):
            writer.writerow({"repeat": index, **report})


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU energy measurement")
    config_path = args.run_dir / "config" / "resolved.yaml"
    checkpoint_path = args.run_dir / "checkpoints" / "best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "run-dir must contain config/resolved.yaml and checkpoints/best.pt")

    protocol = load_protocol(args.protocol)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = build_model(config).to(device)
    load_model(checkpoint_path, model, device)
    validation = build_validation_dataset(config)
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    image = next(iter(loader))["image"].to(device)

    process_count = gpu_compute_process_count(args.physical_gpu)
    if process_count > 1 and not args.allow_shared_gpu:
        raise RuntimeError(
            f"GPU {args.physical_gpu} has {process_count} compute processes; "
            "stop other GPU workloads before measuring")

    output_dir = args.output_dir or (
        args.run_dir / "analysis" / "hardware" / "inference_remeasure" /
        protocol.version)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing measurement: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps({**protocol.to_dict(), "batch_size": args.batch_size,
                    "physical_gpu": args.physical_gpu,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "experiment_id": config.get("id"),
                    "dataset": config.get("dataset")}, indent=2),
        encoding="utf-8",
    )

    idle_sampler = GpuPowerSampler(args.physical_gpu,
                                   protocol.sample_interval_seconds)
    idle_report = measure_idle_power(idle_sampler, protocol.idle_seconds)
    idle_report = idle_sampler.save(
        output_dir / "idle_power.csv", output_dir / "idle_energy.json",
        {**idle_report, "protocol_version": protocol.version,
         "compute_processes_at_start": process_count},
    )
    idle_power_w = idle_report.get("average_power_w")
    if idle_power_w is None:
        raise RuntimeError("Idle power measurement produced no usable samples")

    reports = []
    for repeat in range(1, protocol.repeats + 1):
        sampler = GpuPowerSampler(args.physical_gpu,
                                  protocol.sample_interval_seconds)
        benchmark = benchmark_inference_for_duration(
            model, image, sampler,
            warmup_iterations=protocol.warmup_iterations,
            measured_seconds=protocol.measured_seconds,
            idle_power_w=idle_power_w,
        )
        repeat_dir = output_dir / f"repeat_{repeat:02d}"
        report = sampler.save(
            repeat_dir / "power.csv", repeat_dir / "energy.json",
            {**benchmark, "protocol_version": protocol.version,
             "repeat": repeat,
             "measurement_scope": "whole physical GPU",
             "compute_processes_at_start": gpu_compute_process_count(
                 args.physical_gpu)},
        )
        reports.append(report)

    aggregate = summarize_repeats(reports, protocol.maximum_cv)
    aggregate.update({
        "protocol": protocol.to_dict(),
        "idle_power_w": idle_power_w,
        "experiment_id": config.get("id"),
        "dataset": config.get("dataset"),
    })
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8")
    write_flat_summary(output_dir / "repeats.csv", reports)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
