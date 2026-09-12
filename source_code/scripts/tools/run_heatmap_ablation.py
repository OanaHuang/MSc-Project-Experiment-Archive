from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must use MODEL_ID=SOURCE_RUN")
    model_id, source = value.split("=", 1)
    if not model_id or not source:
        raise argparse.ArgumentTypeError("models must use MODEL_ID=SOURCE_RUN")
    return model_id, Path(source)


def _run(model: tuple[str, Path], args: argparse.Namespace,
         gpu: int | None = None) -> None:
    model_id, source_run = model
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/MPII/heatmap_ablation.py"),
        "--source-run", str(source_run), "--model-id", model_id,
        "--protocol", str(args.protocol), "--output-root", str(args.output_root),
        "--batch-name", args.batch_name,
    ]
    if args.device != "auto":
        command.extend(("--device", "cuda:0" if args.device == "cuda" else args.device))
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.num_workers is not None:
        command.extend(("--num-workers", str(args.num_workers)))
    if args.max_validation_samples is not None:
        command.extend(("--max-validation-samples",
                        str(args.max_validation_samples)))
    if args.variants:
        command.append("--variants")
        command.extend(args.variants)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _run_device(models: list[tuple[str, Path]], gpu: int,
                args: argparse.Namespace) -> None:
    for model in models:
        _run(model, args, gpu)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent heatmap-decoding ablations across GPUs",
    )
    parser.add_argument("--models", nargs="+", type=_parse_model, required=True,
                        metavar="MODEL_ID=SOURCE_RUN")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--devices", nargs="+", type=int,
                        help="CUDA device indices; required only with --device cuda")
    parser.add_argument(
        "--protocol", type=Path,
        default=PROJECT_ROOT / "scripts/MPII/configs/heatmap_ablation.yaml",
    )
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "Outputs_New")
    parser.add_argument("--batch-name", default="heatmap_v1")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    if args.device != "cuda":
        # A single MPS device uses unified memory; sequential execution avoids
        # model workers competing for the same accelerator and memory pool.
        for model in args.models:
            _run(model, args)
        return
    if not args.devices:
        parser.error("--devices is required with --device cuda")
    queues = {gpu: [] for gpu in args.devices}
    for index, model in enumerate(args.models):
        queues[args.devices[index % len(args.devices)]].append(model)
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [executor.submit(_run_device, queues[gpu], gpu, args)
                   for gpu in args.devices if queues[gpu]]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
