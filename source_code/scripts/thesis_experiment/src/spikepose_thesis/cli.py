from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spikepose_thesis.analysis import (
    GpuPowerSampler, benchmark_inference, gpu_compute_process_count,
    measure_idle_power, profile_theoretical_dataset, summarize_repeats,
)
from spikepose_thesis.artifacts import RunArtifacts
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.data import (
    audit_data, build_dataset, prepare_official_mpii_protocol,
    prepare_runtime_cache, run_ntu_preflight,
)
from spikepose_thesis.evaluation import (
    bootstrap_archives, evaluate_checkpoint, evaluate_official_mpii_baseline,
    fit_mpii_head_bone_scale,
)
from spikepose_thesis.evaluation.mpii.official_metrics import evaluate_official
from spikepose_thesis.experiments import (
    experiment_readiness, plan_study, validate_repository,
)
from spikepose_thesis.models import build_model
from spikepose_thesis.reporting import generate_tables
from spikepose_thesis.refinement import run_refinement
from spikepose_thesis.training.checkpoint import load_model
from spikepose_thesis.training.runner import train_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent SpikePose thesis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--forward", action="store_true")
    subparsers.add_parser("audit-data")
    preflight = subparsers.add_parser("preflight-data")
    preflight.add_argument("--workers", type=int, default=4)
    preflight.add_argument("--output", type=Path)
    preflight.add_argument("--force", action="store_true")
    runtime_cache = subparsers.add_parser("prepare-runtime-cache")
    runtime_cache.add_argument("--workers", type=int, default=4)
    runtime_cache.add_argument("--force", action="store_true")
    runtime_cache.add_argument(
        "--setups", default="full",
        help="Cache scope: full, S010-S015, or S001,S005,S010",
    )
    prepare_mpii = subparsers.add_parser("prepare-mpii-official")
    prepare_mpii.add_argument(
        "--official-root", type=Path,
        default=Path("Datasets/MPII/official_simplebaseline/annot"),
    )
    prepare_mpii.add_argument(
        "--legacy-root", type=Path, default=Path("Datasets/MPII/metadata"),
    )
    prepare_mpii.add_argument(
        "--output-root", type=Path,
        default=Path("Datasets/MPII/metadata/official_hrnet"),
    )
    prepare_mpii.add_argument(
        "--images-root", type=Path, default=Path("Datasets/MPII/images"),
    )
    plan = subparsers.add_parser("plan")
    plan.add_argument("--study", default="icassp2027_pilot20")
    commands = subparsers.add_parser("commands")
    commands.add_argument("--study", default="icassp2027_pilot20")
    commands.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    commands.add_argument(
        "--mpii-source",
        choices=(
            "pilot20_m_s3_u1", "pilot20_m_s3_u2",
            "confirm140_m_s3_u1", "confirm140_m_s3_u2",
        ),
    )
    commands.add_argument(
        "--include-tbd", action="store_true",
        help="print blocked/conditional commands as commented TBD lines",
    )
    train = subparsers.add_parser("train")
    train.add_argument("--experiment", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    train.add_argument("--output-root", type=Path)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-validation-samples", type=int)
    train.add_argument("--source-experiment")
    train.add_argument("--profile", choices=("pilot20",))
    train.add_argument("--allow-tbd", action="store_true")
    train.add_argument(
        "--resume", action="store_true",
        help="resume from this run's checkpoints/last.pt at the next epoch",
    )
    train.add_argument(
        "--epochs", type=int,
        help="Pilot-only runtime epoch target; formal configs remain immutable",
    )
    train.add_argument(
        "--ntu-setups",
        help="Pilot-only NTU scope: full, S010-S015, or S001,S005,S010",
    )
    refine = subparsers.add_parser("refine")
    refine.add_argument("--experiment", required=True)
    refine.add_argument("--seed", type=int, required=True)
    refine.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    refine.add_argument("--output-root", type=Path)
    refine.add_argument("--profile", choices=("pilot20",))
    refine.add_argument("--allow-tbd", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate.add_argument("--output-root", type=Path)
    evaluate.add_argument("--profile", choices=("pilot20",))
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--force", action="store_true")
    official = subparsers.add_parser("evaluate-official-mpii")
    official.add_argument("--experiment", required=True)
    official.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    official.add_argument("--output-root", type=Path)
    official.add_argument("--batch-size", type=int)
    official.add_argument("--num-workers", type=int)
    official.add_argument("--max-samples", type=int)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--experiment", required=True)
    profile.add_argument("--seed", type=int, required=True)
    profile.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    profile.add_argument("--output-root", type=Path)
    profile.add_argument("--profile", choices=("pilot20",))
    profile.add_argument("--split", choices=("validation", "test"))
    profile.add_argument("--max-samples", type=int)
    profile.add_argument("--max-batches", type=int)
    profile.add_argument("--batch-size", type=int)
    profile.add_argument("--num-workers", type=int)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--experiment", required=True)
    benchmark.add_argument("--seed", type=int, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--physical-gpu", type=int, required=True)
    benchmark.add_argument("--warmup", type=int, default=100)
    benchmark.add_argument("--iterations", type=int, default=1000)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--idle-seconds", type=float, default=60.0)
    benchmark.add_argument("--output-root", type=Path)
    benchmark.add_argument("--profile", choices=("pilot20",))
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--baseline", type=Path, required=True)
    bootstrap.add_argument("--candidate", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--repeats", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=42)
    report = subparsers.add_parser("report")
    report.add_argument("--study", default="icassp2027_confirm140")
    report.add_argument("--split", choices=("train", "validation", "test"), default="test")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--output-root", type=Path)
    calibrate = subparsers.add_parser("calibrate-hb")
    calibrate.add_argument(
        "--metadata", type=Path,
        default=Path("Datasets/MPII/metadata/train.jsonl"),
    )
    calibrate.add_argument(
        "--output", type=Path,
        default=Path("Datasets/MPII/metadata/head_bone_calibration.json"),
    )
    return parser


def _evaluate(args) -> None:
    config = load_experiment(args.experiment, profile=args.profile)
    official_mpii = (
        config.get("dataset") == "mpii"
        and config.get("data", {}).get("split_protocol") == "official_hrnet_mpii"
    )
    if official_mpii and args.split != "validation":
        raise ValueError(
            "The official local MPII protocol exposes only the labelled validation split"
        )
    if config.get("run_type") == "pilot" and args.split == "test":
        raise RuntimeError(
            "Pilot runs are validation-only; test evaluation is locked to prevent leakage"
        )
    artifacts = RunArtifacts(config, args.seed, args.output_root)
    model = build_model(config).to(args.device)
    load_model(artifacts.path / "checkpoints" / "best.pt", model, torch.device(args.device))
    prediction_dir = artifacts.path / "predictions" / args.split
    if args.split == "test" and (prediction_dir / "summary.json").is_file() and not args.force:
        raise FileExistsError(
            f"Test was already evaluated for this run: {prediction_dir}. "
            "Use --force only for an explicitly documented rerun."
        )
    dataset = build_dataset(config, args.split, args.max_samples)
    workers = int(config["training"]["num_workers"])
    batch_size = int(
        config["training"].get("clip_batch_size", config["training"]["batch_size"])
        if getattr(dataset, "temporal_steps", 1) > 1
        else config["training"]["batch_size"]
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )
    if official_mpii:
        model.experiment_id = config["id"]
        model.model_name = config["paper_id"]
        summary = evaluate_official(
            model, loader, dataset, torch.device(args.device), prediction_dir,
            flip_test=bool(config["evaluation"].get("flip_test", True)),
            flip_shift=bool(config["evaluation"].get("flip_shift", True)),
        )
    else:
        summary = evaluate_checkpoint(
            model, loader, config, torch.device(args.device),
            prediction_dir,
        )
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_repository(args.forward), indent=2))
    elif args.command == "audit-data":
        report = audit_data()
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
    elif args.command == "preflight-data":
        report = run_ntu_preflight(args.workers, args.output, args.force)
        summary = {
            key: report[key]
            for key in (
                "generated_at", "ready", "unique_samples", "valid_samples",
                "invalid_samples", "workers",
            )
        }
        summary["reused"] = bool(report.get("reused", False))
        print(json.dumps(summary, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
    elif args.command == "prepare-runtime-cache":
        report = prepare_runtime_cache(args.workers, args.force, args.setups)
        print(json.dumps({
            key: report[key]
            for key in ("generated_at", "ready", "unique_samples", "poses",
                        "spatial_frames", "bytes", "workers")
        }, indent=2))
    elif args.command == "prepare-mpii-official":
        from spikepose_thesis.core.paths import resolve_project_path

        result = prepare_official_mpii_protocol(
            resolve_project_path(args.official_root),
            resolve_project_path(args.legacy_root),
            resolve_project_path(args.output_root),
            resolve_project_path(args.images_root),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "plan":
        readiness_report = audit_data()
        for index, config in enumerate(plan_study(args.study), 1):
            seeds = ",".join(str(item) for item in config["training"]["seeds"])
            source = config.get("initialization", {}).get("source", "scratch")
            states = {
                experiment_readiness(config, seed, report=readiness_report)["status"]
                for seed in config["training"]["seeds"]
            }
            status = states.pop() if len(states) == 1 else "/".join(sorted(states))
            print(
                f"{index:02d} {config['id']:<24} {config['paper_id']:<16} "
                f"epochs={config['training']['epochs']:<3} seeds={seeds} "
                f"source={source} status={status}"
            )
    elif args.command == "commands":
        job_index = 0
        readiness_report = audit_data()
        for config in plan_study(args.study):
            if config.get("action") == "evaluate_official_checkpoint":
                readiness = experiment_readiness(
                    config, report=readiness_report,
                )
                gpu = args.gpus[job_index % len(args.gpus)]
                command = (
                    f"CUDA_VISIBLE_DEVICES={gpu} spikepose-thesis "
                    f"evaluate-official-mpii --experiment {config['id']} "
                    "--device cuda:0"
                )
                if readiness["status"] == "READY":
                    print(command)
                elif args.include_tbd:
                    reasons = "; ".join(readiness["blockers"])
                    print(f"# {readiness['status']}: {reasons}\n# {command}")
                job_index += 1
                continue
            for seed in config["training"]["seeds"]:
                readiness_config = config
                if args.mpii_source and args.mpii_source != config.get(
                    "initialization", {},
                ).get("source") and config["id"] in {
                    "confirm140_ntu_spikepose_frame",
                }:
                    readiness_config = {
                        **config,
                        "initialization": {
                            **config["initialization"], "source": args.mpii_source,
                        },
                    }
                readiness = experiment_readiness(
                    readiness_config, seed, report=readiness_report,
                )
                gpu = args.gpus[job_index % len(args.gpus)]
                action = "refine" if config.get("action") == "refinement" else "train"
                command = (
                    f"CUDA_VISIBLE_DEVICES={gpu} spikepose-thesis {action} "
                    f"--experiment {config['id']} --seed {seed} --device cuda:0"
                )
                if config.get("profile"):
                    command += f" --profile {config['profile']}"
                if action == "train" and readiness["status"] == "RESUMABLE":
                    command += " --resume"
                if (action == "train" and config["id"] in {
                    "confirm140_ntu_spikepose_frame",
                } and args.mpii_source
                    and args.mpii_source != config["initialization"]["source"]):
                    command += f" --source-experiment {args.mpii_source}"
                if readiness["status"] in {"READY", "RESUMABLE"}:
                    print(command)
                elif args.include_tbd:
                    reasons = "; ".join(readiness["blockers"])
                    print(f"# {readiness['status']}: {reasons}\n# {command}")
                job_index += 1
    elif args.command == "train":
        print(train_experiment(
            args.experiment, args.seed, args.device, args.output_root,
            args.max_train_samples, args.max_validation_samples,
            args.source_experiment, args.allow_tbd, args.profile,
            args.resume,
            args.epochs, args.ntu_setups,
        ))
    elif args.command == "refine":
        print(run_refinement(
            args.experiment, args.seed, args.device, args.output_root,
            args.allow_tbd, args.profile,
        ))
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "evaluate-official-mpii":
        result = evaluate_official_mpii_baseline(
            load_experiment(args.experiment), args.device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_samples=args.max_samples, output_root=args.output_root,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "profile":
        config = load_experiment(args.experiment, profile=args.profile)
        artifacts = RunArtifacts(config, args.seed, args.output_root)
        device = torch.device(args.device)
        model = build_model(config).to(device)
        load_model(artifacts.path / "checkpoints" / "best.pt", model, device)
        split = args.split or (
            "test" if config["dataset"].startswith("ntu") else "validation"
        )
        dataset = build_dataset(config, split, args.max_samples)
        workers = (
            int(config["training"]["num_workers"])
            if args.num_workers is None else int(args.num_workers)
        )
        default_batch_size = int(
            config["training"].get(
                "clip_batch_size", config["training"]["batch_size"],
            )
            if getattr(dataset, "temporal_steps", 1) > 1
            else config["training"]["batch_size"]
        )
        batch_size = (
            default_batch_size
            if args.batch_size is None else int(args.batch_size)
        )
        if batch_size < 1 or workers < 0:
            raise ValueError("batch-size must be positive and num-workers non-negative")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        )

        def real_images():
            for batch in loader:
                yield batch["image"].to(
                    device, non_blocking=torch.cuda.is_available(),
                )

        temporal = config["model"].get("temporal", {})
        video_steps = int(temporal.get("video_frames", 1))
        snn_steps = int(temporal.get("snn_steps_per_frame", 1))
        output_path = (
            artifacts.path / "analysis" / f"theoretical_energy_{split}.json"
        )
        report = profile_theoretical_dataset(
            model, real_images(), output_path,
            video_steps=video_steps,
            snn_steps_per_frame=snn_steps,
            max_batches=args.max_batches,
            metadata={
                "dataset": config["dataset"],
                "split": split,
                "max_samples": args.max_samples,
                "max_batches": args.max_batches,
                "batch_size": batch_size,
                "num_workers": workers,
                "checkpoint": str(artifacts.path / "checkpoints" / "best.pt"),
            },
        )
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark":
        if not str(args.device).startswith("cuda"):
            raise ValueError("Hardware benchmark requires a CUDA device")
        config = load_experiment(args.experiment, profile=args.profile)
        artifacts = RunArtifacts(config, args.seed, args.output_root)
        device = torch.device(args.device)
        model = build_model(config).to(device)
        load_model(artifacts.path / "checkpoints" / "best.pt", model, device)
        temporal = config["model"]["temporal"]
        shape = (
            (1, config["model"]["num_steps"], 3, config["data"]["image_size"], config["data"]["image_size"])
            if temporal["input_strategy"] == "frames"
            else (1, 3, config["data"]["image_size"], config["data"]["image_size"])
        )
        image = torch.zeros(*shape, device=device)
        idle_sampler = GpuPowerSampler(args.physical_gpu)
        idle = measure_idle_power(idle_sampler, args.idle_seconds)
        idle_power = float(idle.get("average_power_w") or 0.0)
        repetitions = []
        for _ in range(args.repeats):
            sampler = GpuPowerSampler(args.physical_gpu)
            processes = gpu_compute_process_count(args.physical_gpu)
            timing = benchmark_inference(
                model, image, sampler, warmup=args.warmup,
                iterations=args.iterations,
            )
            repetitions.append(sampler.report({
                **timing, "idle_power_w": idle_power,
                "compute_processes_at_start": processes,
            }))
        summary = summarize_repeats(repetitions)
        artifacts.write_json("analysis/hardware_idle.json", idle)
        artifacts.write_json("analysis/hardware_repetitions.json", repetitions)
        artifacts.write_json("analysis/hardware_summary.json", summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "bootstrap":
        result = bootstrap_archives(
            args.baseline, args.candidate, args.output, args.repeats, args.seed,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        report_configs = plan_study(args.study)
        if (args.split == "test"
                and report_configs
                and all(item.get("run_type") == "pilot" for item in report_configs)):
            raise RuntimeError(
                "Pilot reports are validation-only; test reporting is locked"
            )
        result = generate_tables(
            args.study, args.output, args.output_root, args.split,
        )
        print(json.dumps({
            "study": result["study"], "experiments": len(result["experiments"]),
            "output": str(args.output),
        }, indent=2))
    elif args.command == "calibrate-hb":
        result = fit_mpii_head_bone_scale(args.metadata, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
