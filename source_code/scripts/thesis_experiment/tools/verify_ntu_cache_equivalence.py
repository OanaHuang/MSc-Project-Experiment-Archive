#!/usr/bin/env python3
"""Verify that an NTU tube cache is identical to the source-frame path."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.data import build_dataset
from spikepose_thesis.models import build_model
from spikepose_thesis.training.checkpoint import load_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="confirm140_ntu_spikepose_frame")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def _assert_item_equal(source: dict, cached: dict, index: int) -> list[str]:
    if source.keys() != cached.keys():
        raise AssertionError(f"item {index} keys differ")
    tensor_keys: list[str] = []
    for key in source:
        left, right = source[key], cached[key]
        if isinstance(left, torch.Tensor):
            tensor_keys.append(key)
            try:
                torch.testing.assert_close(
                    left, right, rtol=0.0, atol=0.0, equal_nan=True,
                )
            except AssertionError as error:
                maximum = (
                    float((left - right).abs().nan_to_num().max())
                    if left.is_floating_point() else None
                )
                raise AssertionError(
                    f"item {index} tensor {key!r} differs; max_abs={maximum}"
                ) from error
        elif left != right:
            raise AssertionError(
                f"item {index} value {key!r} differs: {left!r} != {right!r}"
            )
    return tensor_keys


def main() -> None:
    args = _parser().parse_args()
    cached_config = load_experiment(args.experiment)
    source_config = deepcopy(cached_config)
    source_config["data"]["runtime_cache_mode"] = "disabled"
    source_config["data"]["runtime_spatial_crops"] = False
    source_dataset = build_dataset(
        source_config, args.split, max_samples=args.max_samples,
    )
    cached_dataset = build_dataset(
        cached_config, args.split, max_samples=args.max_samples,
    )
    if len(source_dataset) != len(cached_dataset):
        raise AssertionError(
            f"dataset lengths differ: {len(source_dataset)} != {len(cached_dataset)}"
        )
    checked = min(len(source_dataset), int(args.max_items))
    tensor_keys: list[str] = []
    started = time.perf_counter()
    source_items = [source_dataset[index] for index in range(checked)]
    source_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cached_items = [cached_dataset[index] for index in range(checked)]
    cached_seconds = time.perf_counter() - started
    for index, (source, cached) in enumerate(zip(source_items, cached_items)):
        tensor_keys = _assert_item_equal(source, cached, index)

    prediction_equal = None
    prediction_max_abs = None
    if args.checkpoint is not None:
        device = torch.device(args.device)
        source_model = build_model(source_config).to(device).eval()
        cached_model = build_model(cached_config).to(device).eval()
        load_model(args.checkpoint, source_model, device)
        load_model(args.checkpoint, cached_model, device)
        source_images = torch.stack(
            [item["image"] for item in source_items], dim=0,
        ).to(device)
        cached_images = torch.stack(
            [item["image"] for item in cached_items], dim=0,
        ).to(device)
        with torch.inference_mode():
            source_prediction = source_model(source_images)
            cached_prediction = cached_model(cached_images)
        prediction_equal = bool(torch.equal(source_prediction, cached_prediction))
        prediction_max_abs = float(
            (source_prediction - cached_prediction).abs().max().cpu()
        )
        if not prediction_equal:
            raise AssertionError(
                f"checkpoint predictions differ; max_abs={prediction_max_abs}"
            )

    print(json.dumps({
        "experiment": args.experiment,
        "split": args.split,
        "items_checked": checked,
        "tensor_keys": tensor_keys,
        "inputs_equal": True,
        "source_seconds": source_seconds,
        "cached_seconds": cached_seconds,
        "data_speedup": source_seconds / max(cached_seconds, 1e-12),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "predictions_equal": prediction_equal,
        "prediction_max_abs": prediction_max_abs,
    }, indent=2))


if __name__ == "__main__":
    main()
