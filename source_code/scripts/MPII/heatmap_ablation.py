from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
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
from scripts.MPII.core import MPII_JOINT_NAMES
from scripts.MPII.evaluation.heatmap_ablation import (
    HeatmapVariant, evaluate_cached_variant,
)
from scripts.MPII.evaluation.heatmap_cache import (
    generate_heatmap_cache, load_heatmap_cache,
)
from scripts.spikepose.artifacts import output_run_dir
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def default_device() -> str:
    """Select the fastest locally available PyTorch inference device."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an independent MPII heatmap-decoding ablation",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=PROJECT_ROOT / "scripts/MPII/configs/heatmap_ablation.yaml",
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/best.pt"))
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "Outputs_New")
    parser.add_argument("--batch-name")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--variants", nargs="+")
    return parser.parse_args()


def _status(path: Path, state: str, **details) -> None:
    payload = {"status": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               **details}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_per_joint_summary(summary: dict, variant_dir: Path) -> dict:
    if "official_per_joint_pckh" in summary:
        return summary
    with (variant_dir / "dual_pckh_per_joint.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary["official_per_joint_pckh"] = {
        row["joint_name"]: float(row["official_pckh"]) for row in rows
    }
    summary["custom_per_joint_pckh"] = {
        row["joint_name"]: float(row["custom_pckh"]) for row in rows
    }
    summary["official_per_joint_valid"] = {
        row["joint_name"]: int(row["official_valid"]) for row in rows
    }
    summary["custom_per_joint_valid"] = {
        row["joint_name"]: int(row["custom_valid"]) for row in rows
    }
    serialized = json.dumps(summary, indent=2)
    (variant_dir / "dual_pckh_summary.json").write_text(
        serialized, encoding="utf-8",
    )
    (variant_dir / "best.json").write_text(serialized, encoding="utf-8")
    return summary


def _copy_protocol_once(source: Path, destination: Path) -> None:
    content = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(
                f"Output run already uses a different protocol: {destination}"
            )
        return
    destination.write_bytes(content)


def _save_comparison(run_dir: Path, rows: list[dict], reference: str,
                     official_variant: str) -> None:
    by_id = {row["variant_id"]: row for row in rows}
    if reference not in by_id:
        raise ValueError(f"Reference variant was not evaluated: {reference}")
    if official_variant not in by_id:
        raise ValueError(
            f"Official-compatible variant was not evaluated: {official_variant}"
        )
    reference_pckh = float(by_id[reference]["official_pckh"])
    compact = []
    for row in rows:
        item = {
            "variant_id": row["variant_id"], "decoder": row["decoder"],
            "udp": row["udp"], "dark_kernel": row["dark_kernel"],
            "flip_test": row["flip_test"], "flip_shift": row["flip_shift"],
            "official_pckh": row["official_pckh"],
            "custom_pckh": row["custom_pckh"],
            "pckh_gain_pp": 100.0 * (float(row["official_pckh"]) - reference_pckh),
            "samples": row["samples"],
            "official_compatible": row["variant_id"] == official_variant,
        }
        for joint_name in MPII_JOINT_NAMES:
            item[f"{joint_name}_pckh"] = row["official_per_joint_pckh"][joint_name]
        compact.append(item)
    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    (run_dir / "comparison.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    source_config_path = source_run / "config/resolved.yaml"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {source_config_path}")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = source_run / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    seed = int(source_config.get("training", {}).get("seed", 42))
    batch_name = args.batch_name or protocol.get("batch_name", "heatmap_v1")
    run_dir = output_run_dir(
        args.output_root.resolve(), "mpii", "heatmap_decoding",
        batch_name, args.model_id, seed,
    )
    for name in ("config", "cache", "metrics", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    _copy_protocol_once(args.protocol, run_dir / "config/protocol.yaml")
    resolved = {
        "id": f"heatmap_decoding_{args.model_id}",
        "name": f"{source_config.get('name', args.model_id)} Heatmap Decoding",
        "category": "heatmap_decoding_ablation", "batch_name": batch_name,
        "seed": seed,
        "source": {
            "experiment_id": source_config.get("id"),
            "model_name": source_config.get("name"),
            "checkpoint": str(args.checkpoint),
            "runtime_source_run": str(source_run),
        },
        "protocol_id": protocol["id"],
        "max_validation_samples": args.max_validation_samples,
    }
    resolved_path = run_dir / "config/resolved.yaml"
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        identity_keys = ("id", "category", "batch_name", "seed", "protocol_id",
                         "max_validation_samples")
        if any(existing.get(key) != resolved.get(key) for key in identity_keys):
            raise ValueError(f"Output run identity does not match: {resolved_path}")
        if existing.get("source", {}).get("experiment_id") != source_config.get("id"):
            raise ValueError(f"Output run uses a different source model: {resolved_path}")
    else:
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8",
        )
    _status(status_path, "running", stage="preparing", model_id=args.model_id,
            source_experiment_id=source_config.get("id"))
    data = source_config["data"]
    dataset = MPIIPoseDataset(
        PROJECT_ROOT / data["validation_metadata"], PROJECT_ROOT / data["images_dir"],
        data["image_size"], data["heatmap_size"], data["sigma"],
        data["crop_expansion"], False, args.max_validation_samples,
    )
    training = source_config.get("training", {})
    loader = DataLoader(
        dataset, batch_size=args.batch_size or training.get("batch_size", 32),
        shuffle=False, num_workers=(args.num_workers if args.num_workers is not None
                                    else training.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    cache_dir = run_dir / "cache"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        # Remove the empty directory created above so cache creation is atomic about ownership.
        cache_dir.rmdir()
        device = torch.device(args.device)
        model = build_model(source_config).to(device)
        load_model(checkpoint, model, device)
        _status(status_path, "running", stage="caching", model_id=args.model_id)
        generate_heatmap_cache(
            model, loader, device, cache_dir, checkpoint,
            dtype=protocol.get("cache_dtype", "float16"), include_flipped=True,
        )
        del model
    cache = load_heatmap_cache(cache_dir, checkpoint, expected_samples=len(dataset))
    requested = set(args.variants or [])
    if requested:
        requested.add(protocol["reference"])
    variants = [HeatmapVariant.from_dict(item) for item in protocol["variants"]
                if not requested or item["id"] in requested]
    unknown = requested - {item.id for item in variants}
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    _status(status_path, "running", stage="decoding", model_id=args.model_id,
            variants=[item.id for item in variants])
    rows = []
    for variant in variants:
        variant_dir = run_dir / "metrics" / variant.id
        summary_path = variant_dir / "dual_pckh_summary.json"
        if summary_path.is_file():
            summary = _read_summary(summary_path)
            expected = {
                "decoder": variant.decoder, "udp": variant.udp,
                "dark_kernel": variant.dark_kernel,
                "flip_test": variant.flip_test, "flip_shift": variant.flip_shift,
            }
            if any(summary.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    f"Existing result does not match protocol variant {variant.id}"
                )
            if int(summary.get("samples", 0)) != len(dataset):
                raise ValueError(
                    f"Existing result sample count does not match {variant.id}"
                )
            summary = _ensure_per_joint_summary(summary, variant_dir)
        else:
            summary = evaluate_cached_variant(
                cache, dataset.image_size, variant, variant_dir,
                source_config.get("name", args.model_id), source_config.get("id", args.model_id),
            )
        rows.append(summary)
    _save_comparison(
        run_dir, rows, protocol["reference"],
        protocol.get("official_variant", "flip_quarter"),
    )
    _status(status_path, "completed", model_id=args.model_id,
            variants=[item.id for item in variants], samples=len(dataset),
            comparison="comparison.csv")
    print(run_dir)


if __name__ == "__main__":
    main()
