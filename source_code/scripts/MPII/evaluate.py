from __future__ import annotations

import argparse
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
from scripts.MPII.evaluation import evaluate
from scripts.spikepose.analysis import profile_theoretical
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MPII run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flip-test", action="store_true")
    parser.add_argument("--no-flip-shift", action="store_true")
    parser.add_argument(
        "--decoder", choices=("argmax", "quarter", "dark"),
        default="argmax",
    )
    parser.add_argument("--udp", action="store_true")
    parser.add_argument("--dark-kernel", type=int, default=11)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Checkpoint path relative to the run directory or an absolute path; defaults to checkpoints/best.pt",
    )
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "config" / "resolved.yaml").read_text())
    data = config["data"]
    dataset = MPIIPoseDataset(
        PROJECT_ROOT / data["validation_metadata"], PROJECT_ROOT / data["images_dir"],
        data["image_size"],
        data["heatmap_size"], data["sigma"], data["crop_expansion"], False,
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint = args.checkpoint or Path("checkpoints/best.pt")
    if not checkpoint.is_absolute():
        checkpoint = args.run / checkpoint
    load_model(checkpoint, model, device)
    analysis_image = next(iter(loader))["image"][:1].to(device)
    profile_theoretical(
        model, analysis_image,
        args.run / "analysis" / "theoretical_energy.json",
    )
    output_dir = args.output_dir or args.run / "metrics"
    summary = evaluate(
        model, loader, dataset, device, output_dir,
        flip_test=args.flip_test, flip_shift=not args.no_flip_shift,
        decoder=args.decoder, udp=args.udp, dark_kernel=args.dark_kernel,
    )
    if args.output_dir is not None:
        print(json.dumps(summary, indent=2))
        return
    status_path = args.run / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["metrics"] = summary
        status["metric_protocol"] = "MPII PCKh@0.5 calibrated 2026"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
