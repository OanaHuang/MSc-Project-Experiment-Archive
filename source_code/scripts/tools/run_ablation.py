from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.experiments import ablation_roots, load_ablation
from scripts.spikepose.artifacts import (
    generate_random_seeds, output_batch_dir, validate_seed_targets,
)
from scripts.tools.create_sample_manifest import create_batch_manifest
from scripts.tools.gpu_scheduler import Task, run_tasks


def write_random_seed_plan(
    batch_dir: Path, args: argparse.Namespace, experiments: list[str],
    seeds: list[int], targets: list[Path],
) -> Path:
    created_at = datetime.now(timezone.utc)
    plan_dir = batch_dir / "seed_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "_".join(str(seed) for seed in seeds)
    path = plan_dir / f"random_{created_at:%Y%m%dT%H%M%SZ}_{seed_label}.json"
    if path.exists():
        raise FileExistsError(f"Seed plan already exists: {path}")
    payload = {
        "status": "scheduled",
        "source": "system_secure_random",
        "created_at_utc": created_at.isoformat(),
        "dataset": args.dataset,
        "category": args.category,
        "batch_name": args.batch_name,
        "seeds": seeds,
        "experiments": list(experiments),
        "targets": [str(target) for target in targets],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpikePose ablations across GPUs")
    parser.add_argument("--dataset", choices=("mpii", "ntu_rgbd"), required=True)
    parser.add_argument("--group")
    parser.add_argument("--category")
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--experiments", nargs="+")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument(
        "--random-seeds", type=int, metavar="COUNT",
        help="Generate COUNT unique seeds not already present in the output root",
    )
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--visualization-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--checkpoint-epochs", type=int, nargs="+", default=[],
        help="Additionally preserve checkpoints at these completed epochs",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Continue every task from its checkpoints/last.pt")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "Outputs_New")
    args = parser.parse_args()
    if bool(args.group) == bool(args.experiments):
        parser.error("Specify exactly one of --group or --experiments")
    if args.group:
        group = load_ablation(ablation_roots(PROJECT_ROOT, args.dataset), args.group)
        experiments = group["experiments"]
        category = args.category or group["id"]
    else:
        experiments = args.experiments
        category = args.category or "custom"
    if args.random_seeds is not None and args.resume:
        parser.error("--random-seeds cannot be combined with --resume")
    seeds = (
        generate_random_seeds(args.output_root, args.random_seeds)
        if args.random_seeds is not None else (args.seeds or [42])
    )
    targets = validate_seed_targets(
        args.output_root, args.dataset, category, args.batch_name,
        experiments, seeds, args.resume,
    )
    batch_dir = output_batch_dir(
        args.output_root, args.dataset, category, args.batch_name,
    )
    manifest = batch_dir / "sample_manifest.json"
    if not manifest.exists():
        create_batch_manifest(PROJECT_ROOT, args.dataset, manifest,
                              args.visualization_seed)
    if args.random_seeds is not None:
        args.category = category
        plan = write_random_seed_plan(
            batch_dir, args, experiments, seeds, targets,
        )
        print(f"Random seeds: {' '.join(str(seed) for seed in seeds)}", flush=True)
        print(f"Seed plan: {plan}", flush=True)
    extra = []
    for name in ("epochs", "batch_size", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            extra.extend((f"--{name.replace('_', '-')}", str(value)))
    if args.checkpoint_epochs:
        extra.append("--checkpoint-epochs")
        extra.extend(str(epoch) for epoch in args.checkpoint_epochs)
    if args.resume:
        extra.append("--resume")
    tasks = [Task(args.dataset, experiment, seed)
             for experiment in experiments for seed in seeds]
    run_tasks(tasks, args.devices, PROJECT_ROOT, category, args.batch_name,
              args.output_root, manifest, extra)


if __name__ == "__main__":
    main()
