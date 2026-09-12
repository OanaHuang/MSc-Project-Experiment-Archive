from __future__ import annotations

import time

import torch

from .hardware import GpuPowerSampler


@torch.no_grad()
def benchmark_inference(model, image: torch.Tensor, sampler: GpuPowerSampler,
                        warmup: int = 10, iterations: int = 100) -> dict:
    model.eval()
    for _ in range(warmup):
        model(image)
    torch.cuda.synchronize(image.device)
    sampler.start()
    started = time.perf_counter()
    for _ in range(iterations):
        model(image)
    torch.cuda.synchronize(image.device)
    seconds = time.perf_counter() - started
    sampler.stop()
    images = iterations * len(image)
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "batch_size": len(image),
        "wall_seconds": seconds,
        "milliseconds_per_batch": seconds * 1000.0 / iterations,
        "milliseconds_per_image": seconds * 1000.0 / images,
        "images_per_second": images / seconds,
    }
