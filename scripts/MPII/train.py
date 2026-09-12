from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.datasets import MPIIPoseDataset
from scripts.MPII.evaluation import evaluate
from scripts.MPII.visualization import generate_visualizations
from scripts.spikepose.analysis import (
    GpuPowerSampler, benchmark_inference, gpu_compute_process_count,
    profile_theoretical,
)
from scripts.spikepose.experiments import (
    experiment_roots, resolve_config, validate_config,
)
from scripts.spikepose.models import build_model
from scripts.spikepose.artifacts import OutputManager
from scripts.spikepose.training import (
    build_pose_loss, build_scheduler, load_model, resume_training, train,
)
from scripts.spikepose.visualization import create_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SpikePose experiment on MPII")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--category")
    parser.add_argument("--batch-name", default="standalone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int,
                        help="Target total epoch count, including completed epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from this run's checkpoints/last.pt")
    parser.add_argument(
        "--checkpoint-epochs", type=int, nargs="+", default=[],
        help="Additionally preserve checkpoints at these completed epochs",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--scheduler-kind", choices=(
            "plateau", "warmup_cosine", "constant", "cosine_restart"),
    )
    parser.add_argument("--scheduler-start-epoch", type=int)
    parser.add_argument("--scheduler-base-lr", type=float)
    parser.add_argument("--scheduler-min-lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: dict, split: str, max_samples=None):
    data = config["data"]
    augmentation = config.get("training", {}).get("augmentation", {})
    key = "train_metadata" if split == "train" else "validation_metadata"
    return MPIIPoseDataset(
        PROJECT_ROOT / data[key], PROJECT_ROOT / data["images_dir"],
        data["image_size"], data["heatmap_size"],
        data["sigma"], data["crop_expansion"], split == "train", max_samples,
        augmentation.get("scale_range") if split == "train" else None,
        augmentation.get("rotation_degrees", 0.0) if split == "train" else 0.0,
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    args = parse_args()
    overrides = {"training": {"seed": args.seed}}
    for key in ("epochs", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            overrides["training"][key] = value
    if args.learning_rate is not None:
        overrides["training"]["learning_rate"] = args.learning_rate
    scheduler_override = {}
    if args.scheduler_kind is not None:
        scheduler_override["kind"] = args.scheduler_kind
    if args.scheduler_start_epoch is not None:
        scheduler_override["start_epoch"] = args.scheduler_start_epoch
    if args.scheduler_base_lr is not None:
        scheduler_override["base_lr"] = args.scheduler_base_lr
    if args.scheduler_min_lr is not None:
        scheduler_override["min_lr"] = args.scheduler_min_lr
    if scheduler_override:
        overrides["training"]["scheduler"] = scheduler_override
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, "mpii"), args.experiment,
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml",
        overrides,
    )
    validate_config(config)
    category = args.category or config.get("category", "uncategorized")
    output = OutputManager(args.output_root, "mpii", category, args.batch_name,
                           config["id"], config["name"], args.seed)
    output.save_config(config)
    output.status("running", stage="training")
    seed_all(args.seed)

    training = config["training"]
    train_set = make_dataset(config, "train", args.max_train_samples)
    val_set = make_dataset(config, "validation", args.max_validation_samples)
    common = dict(
        batch_size=training["batch_size"], num_workers=training["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=training["num_workers"] > 0,
    )
    loader_seed = torch.Generator().manual_seed(args.seed)
    worker_init = seed_worker if training.get("deterministic_workers", False) else None
    train_loader = DataLoader(
        train_set, shuffle=True, generator=loader_seed, worker_init_fn=worker_init, **common)
    val_loader = DataLoader(val_set, shuffle=False, worker_init_fn=worker_init, **common)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, training)
    continuation = {}
    if args.resume:
        continuation = resume_training(
            output.run_dir / "checkpoints" / "last.pt", model, optimizer,
            scheduler, device, config["id"],
        )
        if training["epochs"] < continuation["start_epoch"]:
            raise ValueError(
                f"Checkpoint already completed epoch {continuation['completed_epoch']}; "
                f"--epochs must be at least {continuation['start_epoch']}"
            )
        output.status(
            "running", stage="training", resumed_from_epoch=continuation["completed_epoch"],
            target_total_epochs=training["epochs"],
        )
    training_sampler = (GpuPowerSampler(args.physical_gpu)
                        if args.physical_gpu is not None else None)
    process_count = (gpu_compute_process_count(args.physical_gpu)
                     if args.physical_gpu is not None else None)
    if training_sampler:
        training_sampler.start()
    try:
        history = train(
            model, train_loader, val_loader, build_pose_loss(config), optimizer,
            scheduler, device, training["epochs"], output.run_dir, config,
            start_epoch=continuation.get("start_epoch", 1),
            history=continuation.get("history"),
            best_metric=continuation.get("best_metric", float("inf")),
            checkpoint_epochs=set(args.checkpoint_epochs),
        )
    finally:
        if training_sampler:
            training_sampler.stop()
    if training_sampler:
        hardware_dir = output.run_dir / "analysis" / "hardware"
        start_epoch = int(continuation.get("start_epoch", 1))
        completed_epochs = int(history[-1]["epoch"])
        training_sampler.save(
            hardware_dir / "training_power.csv",
            hardware_dir / "training_energy.json",
            {"compute_processes_at_start": process_count,
             "gpu_shared_at_start": process_count is not None and process_count > 1,
             "requested_epochs": training["epochs"],
             "start_epoch": start_epoch,
             "completed_epochs": completed_epochs,
             "measured_epochs": completed_epochs - start_epoch + 1,
             "coverage": "full_training" if start_epoch == 1 else "resume_segment",
             "validity_status": "valid" if start_epoch == 1 else "partial"},
        )
    load_model(output.run_dir / "checkpoints" / "best.pt", model, device)
    summary = evaluate(model, val_loader, val_set, device, output.run_dir / "metrics")
    analysis_image = next(iter(val_loader))["image"][:1].to(device)
    profile_theoretical(
        model, analysis_image,
        output.run_dir / "analysis" / "theoretical_energy.json",
    )
    if args.physical_gpu is not None:
        inference_sampler = GpuPowerSampler(args.physical_gpu, interval=0.2)
        benchmark = benchmark_inference(model, analysis_image, inference_sampler)
        inference_sampler.save(
            output.run_dir / "analysis" / "hardware" / "inference_power.csv",
            output.run_dir / "analysis" / "hardware" / "inference_energy.json",
            {**benchmark, "compute_processes_at_start": gpu_compute_process_count(
                args.physical_gpu)},
        )

    if not args.skip_visualization:
        output.status("running", stage="visualizing")
        manifest_path = args.manifest or output.batch_dir / "sample_manifest.json"
        visual_set = make_dataset(config, "validation")
        if not manifest_path.exists():
            visual = config["visualization"]
            create_manifest(
                visual_set, manifest_path, visual["selection_seed"],
                visual["first_count"], visual["random_count"],
            )
        generate_visualizations(
            model, visual_set, device, manifest_path,
            output.run_dir / "visualizations",
        )
    output.status("completed", best_val_loss=min(row["val_loss"] for row in history),
                  metrics=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
