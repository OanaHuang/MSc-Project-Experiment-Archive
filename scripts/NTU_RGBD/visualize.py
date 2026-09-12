from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.visualization import generate_visualizations
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate NTU RGB+D report images")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "config" / "resolved.yaml").read_text())
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    device = torch.device(args.device)
    model = build_model(config).to(device)
    load_model(args.run / "checkpoints" / "best.pt", model, device)
    generate_visualizations(model, dataset, device, args.manifest,
                            args.run / "visualizations")


if __name__ == "__main__":
    main()
