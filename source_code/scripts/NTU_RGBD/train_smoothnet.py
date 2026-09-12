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

from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.NTU_RGBD.evaluation.metrics import compute_pckhn
from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce, save_nacce
from scripts.NTU_RGBD.smoothnet import SmoothNet, PoseWindowDataset, load_sequences, refine_sequences


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              visibility: torch.Tensor) -> torch.Tensor:
    mask = visibility[..., None].bool()
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    error = torch.where(mask, (prediction - target).abs(), 0.0)
    return error.sum() / mask.sum().clamp_min(1)


def objective(model: SmoothNet, batch: dict[str, torch.Tensor], device: torch.device,
              coordinate_scale: float, acceleration_weight: float) -> torch.Tensor:
    source = batch["pred"].to(device) / coordinate_scale
    target = batch["gt"].to(device) / coordinate_scale
    visibility = batch["visibility"].to(device)
    refined = model(source)
    position = masked_l1(refined, target, visibility)
    accel_visibility = visibility[:, :-2] * visibility[:, 1:-1] * visibility[:, 2:]
    acceleration = masked_l1(
        refined.diff(n=2, dim=1), target.diff(n=2, dim=1), accel_visibility,
    )
    return position + acceleration_weight * acceleration


def evaluate_and_save(model: SmoothNet, sequences, device: torch.device,
                      output_dir: Path, coordinate_scale: float) -> dict:
    prediction = refine_sequences(model, sequences, device, coordinate_scale)
    pck = compute_pckhn(prediction, sequences.gt, sequences.visibility,
                        sequences.head_length, 0.5)
    valid, correct = pck["valid_mask"], pck["correct_matrix"]
    temporal = compute_nacce(
        prediction, sequences.gt, sequences.visibility, sequences.head_length,
        sequences.sample_ids, sequences.frame_indices, NTU_JOINT_NAMES,
    )
    summary = {
        "metric": "pckhn", "threshold_label": "PCKhn@0.5",
        "pckhn": float(correct.sum() / max(int(valid.sum()), 1)),
        "valid_joints": int(valid.sum()), "samples": int(len(prediction)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pckhn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_nacce(temporal, output_dir / "temporal_metrics")
    np.savez_compressed(
        output_dir / "prediction_matrices.npz", pred=prediction, gt=sequences.gt,
        visibility=sequences.visibility, head_length=sequences.head_length,
        sample_ids=sequences.sample_ids, frame_indices=sequences.frame_indices,
    )
    return {**summary, **temporal}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one NTU F-SmoothNet post-processor")
    parser.add_argument("--id", required=True, choices=("fs1", "fs2", "fs3", "fs4"))
    parser.add_argument("--train-matrices", type=Path, required=True)
    parser.add_argument("--validation-matrices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--coordinate-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.window_size < 3 or args.epochs < 1 or args.coordinate_scale <= 0:
        raise ValueError("invalid SmoothNet training configuration")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_sequences = load_sequences(args.train_matrices)
    validation_sequences = load_sequences(args.validation_matrices)
    train_data = PoseWindowDataset(train_sequences, args.window_size, stride=1)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=device.type == "cuda")
    model = SmoothNet(args.window_size, args.hidden_size, args.num_blocks, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({"method": "SmoothNet coordinate post-processing", "primary_metric": "PCKhn@0.5"})
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.output_dir / "resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, batch, device, args.coordinate_scale, args.acceleration_weight)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch["pred"]); count += len(batch["pred"])
        scheduler.step()
        epoch_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       args.output_dir / "best.pt")
        print(f"{args.id} epoch={epoch}/{args.epochs} loss={epoch_loss:.6f}", flush=True)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    saved = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    metrics = evaluate_and_save(
        model, validation_sequences, device, args.output_dir / "metrics", args.coordinate_scale,
    )
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "completed", "id": args.id, "best_epoch": saved["epoch"], "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
