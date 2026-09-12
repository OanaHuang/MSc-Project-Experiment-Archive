from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.core.config import NTU_JOINT_INDEX
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.evaluation import evaluate
from scripts.spikepose.models import build_model
from scripts.spikepose.training import VisibleHeatmapMSE, load_model


HAND_GROUPS = {
    "wrist": ("wrist_left", "wrist_right"),
    "hand": ("hand_left", "hand_right"),
    "hand_tip": ("hand_tip_left", "hand_tip_right"),
    "thumb": ("thumb_left", "thumb_right"),
}


class StageProbe(nn.Module):
    def __init__(self, source, stage: int, channels: int, joints: int,
                 heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        self.source = source
        self.stage = int(stage)
        self.readout = nn.Conv2d(channels, joints, 1)
        nn.init.normal_(self.readout.weight, std=0.001)
        nn.init.zeros_(self.readout.bias)
        self.heatmap_size = heatmap_size
        self.experiment_id = f"jts6_stage{stage}_probe"
        self.model_name = f"SpikePose-NTU-JTS6-Stage{stage}Probe"

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        sequence = self.source._make_sequence(image)
        value = self.source.backbone(sequence).features[self.stage]
        return value.mean(0) if value.ndim == 5 else value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self._features(image)
        heatmaps = self.readout(features)
        return F.interpolate(
            heatmaps, size=self.heatmap_size, mode="bilinear", align_corners=False,
        )


def hand_group_summary(per_joint: dict[str, float | None]) -> dict[str, float | None]:
    result = {}
    for group, names in HAND_GROUPS.items():
        values = [per_joint[name] for name in names if per_joint.get(name) is not None]
        result[group] = float(np.mean(values)) if values else None
    values = [value for value in result.values() if value is not None]
    result["hand_group"] = float(np.mean(values)) if values else None
    return result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_probe(probe: StageProbe, train_loader, val_loader, val_set, device,
                epochs: int, learning_rate: float, output_dir: Path) -> dict:
    criterion = VisibleHeatmapMSE()
    optimizer = torch.optim.AdamW(probe.readout.parameters(), lr=learning_rate)
    probe.to(device)
    for epoch in range(1, epochs + 1):
        probe.train()
        # Probe training must not alter frozen-source normalization or spiking
        # state behavior even though the source is registered as a submodule.
        probe.source.eval()
        total = 0.0
        batches = 0
        for batch in train_loader:
            prediction = probe(batch["image"].to(device))
            loss = criterion(
                prediction, batch["heatmap"].to(device),
                batch["visibility"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(json.dumps({"stage": probe.stage, "epoch": epoch,
                          "train_loss": total / max(batches, 1)}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.readout.state_dict(), output_dir / "probe.pt")
    summary = evaluate(probe, val_loader, val_set, device, output_dir / "metrics")
    summary["hand_groups"] = hand_group_summary(summary["per_joint_pckhn"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train frozen equal-capacity Stage probes for NTU JTS checkpoints",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    seed_all(args.seed)
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    checkpoint_path = run / "checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"]["temporal"].get("input_strategy") != "repeat":
        raise ValueError("JTS6 probes require a repeated-current JTS checkpoint")
    source = build_model(config)
    load_model(checkpoint_path, source, torch.device("cpu"))
    source.eval()
    source.requires_grad_(False)
    train_set = build_dataset(PROJECT_ROOT, config, "train", args.max_train_samples)
    val_set = build_dataset(PROJECT_ROOT, config, "validation", args.max_validation_samples)
    common = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    device = torch.device(args.device)
    source.to(device)
    heatmap_size = tuple(int(item) for item in config["model"]["heatmap_size"])
    channels = config["model"]["backbone"]["channels"]
    results = {}
    for stage, stage_channels in enumerate(channels, 1):
        probe = StageProbe(source, stage, int(stage_channels),
                           int(config["model"]["num_joints"]), heatmap_size)
        results[f"stage_{stage}"] = train_probe(
            probe, train_loader, val_loader, val_set, device, args.epochs,
            args.learning_rate, args.output_dir / f"stage_{stage}",
        )
    comparison = {
        "id": "jts6", "name": "SpikePose-NTU-JTS6-StageTimestepProbe",
        "source_run": str(run), "seed": args.seed, "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
