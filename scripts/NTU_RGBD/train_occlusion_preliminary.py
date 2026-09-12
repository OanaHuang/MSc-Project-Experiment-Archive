from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.occlusion import (
    MaskConfig, OcclusionWindowDataset, build_refiner, evaluate_occlusion,
    load_sequences,
)


EXPERIMENT_DIR = PROJECT_ROOT / "scripts/NTU_RGBD/experiments"


def load_experiment(experiment: str) -> dict:
    if experiment not in {f"pv{index}" for index in range(8)}:
        raise ValueError("experiment must be PV0-PV7")
    path = EXPERIMENT_DIR / f"{experiment}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    valid = mask[..., None].bool()
    valid = valid & torch.isfinite(prediction) & torch.isfinite(target)
    error = torch.where(valid, (prediction - target).abs(), 0.0)
    return error.sum() / valid.sum().clamp_min(1)


def objective(model, batch, device: torch.device, scale: float) -> torch.Tensor:
    pose = batch["pred"].to(device) / scale
    target = batch["gt"].to(device) / scale
    confidence = batch["confidence"].to(device)
    visibility = batch["visibility"].to(device)
    observed = batch["observed"].to(device)
    prediction = model(pose, confidence, observed)
    occluded = visibility * (1.0 - observed)
    visible = visibility * observed
    position = 2.0 * masked_l1(prediction, target, occluded)
    preservation = masked_l1(prediction, target, visible)
    valid_acceleration = visibility[:, :-2] * visibility[:, 1:-1] * visibility[:, 2:]
    acceleration = masked_l1(
        prediction.diff(n=2, dim=1), target.diff(n=2, dim=1), valid_acceleration,
    )
    return position + preservation + 0.1 * acceleration


@torch.no_grad()
def evaluate(model, loader, device: torch.device, scale: float) -> dict:
    model.eval()
    collected = {key: [] for key in ("pred", "gt", "visibility", "observed", "head_length")}
    for batch in loader:
        prediction = model(
            batch["pred"].to(device) / scale,
            batch["confidence"].to(device), batch["observed"].to(device),
        ) * scale
        # Count each causal window target once through its final frame.
        collected["pred"].append(prediction[:, -1].cpu().numpy())
        for key in ("gt", "visibility", "observed"):
            collected[key].append(batch[key][:, -1].numpy())
        collected["head_length"].append(batch["head_length"][:, -1].numpy())
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    return evaluate_occlusion(
        prediction=arrays["pred"], target=arrays["gt"],
        visibility=arrays["visibility"], observed=arrays["observed"],
        head_length=arrays["head_length"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an NTU T0 occlusion-validation refiner")
    parser.add_argument("--experiment", required=True, choices=tuple(f"pv{i}" for i in range(8)))
    parser.add_argument("--train-matrices", type=Path, required=True)
    parser.add_argument("--validation-matrices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--coordinate-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # PV0 is the clean T0 reference. PV1-PV7 share the masked protocol.
    mask_config = MaskConfig(probability=0.0 if args.experiment == "pv0" else 0.65)
    train_data = OcclusionWindowDataset(
        load_sequences(args.train_matrices), args.window_size, seed=args.seed,
        mask_config=mask_config,
    )
    validation_data = OcclusionWindowDataset(
        load_sequences(args.validation_matrices), args.window_size,
        stride=args.window_size, seed=args.seed + 100_000, mask_config=mask_config,
    )
    train_loader = DataLoader(
        train_data, args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data, args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_refiner(config["method"], hidden_size=args.hidden_size).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate) if trainable else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    resolved = {**vars(args), **config, "mask": vars(mask_config)}
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in resolved.items()}
    (args.output_dir / "resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    history = []
    if optimizer is not None:
        for epoch in range(1, args.epochs + 1):
            train_data.set_epoch(epoch)
            model.train(); total = 0.0; count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch, device, args.coordinate_scale)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch["pred"]); count += len(batch["pred"])
            row = {"epoch": epoch, "train_loss": total / max(count, 1)}
            history.append(row)
            print(f"{args.experiment} epoch={epoch}/{args.epochs} loss={row['train_loss']:.6f}", flush=True)
        torch.save({"model": model.state_dict(), "config": resolved}, args.output_dir / "last.pt")
    metrics = evaluate(model, validation_loader, device, args.coordinate_scale)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "completed", "experiment": args.experiment, "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
