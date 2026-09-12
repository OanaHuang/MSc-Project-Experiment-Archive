from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time
from typing import Callable

import torch

from .hardware import GpuPowerSampler


@dataclass(frozen=True)
class InferenceMeasurementProtocol:
    """Versioned, duration-based protocol for stable GPU measurements."""

    version: str = "measurement_v2"
    warmup_iterations: int = 100
    measured_seconds: float = 60.0
    repeats: int = 5
    sample_interval_seconds: float = 0.2
    idle_seconds: float = 60.0
    maximum_cv: float = 0.05

    def validate(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measured_seconds <= 0 or self.idle_seconds <= 0:
            raise ValueError("measurement durations must be positive")
        if self.repeats < 2:
            raise ValueError("repeats must be at least 2")
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        if not 0 < self.maximum_cv < 1:
            raise ValueError("maximum_cv must be between 0 and 1")

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_repeats(reports: list[dict], maximum_cv: float = 0.05) -> dict:
    """Aggregate valid repetitions without silently discarding outliers."""
    if len(reports) < 2:
        raise ValueError("At least two reports are required")
    fields = (
        "energy_joules_per_image",
        "dynamic_energy_joules_per_image",
        "milliseconds_per_image",
        "images_per_second",
        "average_power_w",
        "peak_memory_mib",
    )
    metrics = {}
    for field in fields:
        values = [float(report[field]) for report in reports
                  if report.get(field) is not None]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        metrics[field] = {
            "count": len(values),
            "mean": mean,
            "standard_deviation": std,
            "coefficient_of_variation": std / mean if mean else None,
            "minimum": min(values),
            "maximum": max(values),
        }
    energy = metrics.get("dynamic_energy_joules_per_image",
                         metrics.get("energy_joules_per_image"))
    cv = energy.get("coefficient_of_variation") if energy else None
    issues = []
    if any(report.get("errors") for report in reports):
        issues.append("one or more power samples failed")
    if any(int(report.get("compute_processes_at_start") or 0) > 1
           for report in reports):
        issues.append("GPU was shared with another compute process")
    if any(report.get("dynamic_energy_nonnegative") is False for report in reports):
        issues.append("idle-subtracted energy was negative")
    if cv is None or cv > maximum_cv:
        issues.append("energy coefficient of variation exceeds the limit")
    return {
        "repetitions": len(reports),
        "metrics": metrics,
        "quality": {
            "maximum_cv": maximum_cv,
            "observed_energy_cv": cv,
            "status": "valid" if not issues else "review",
            "issues": issues,
            "additional_repeats_recommended": cv is None or cv > maximum_cv,
        },
    }


def measure_idle_power(sampler: GpuPowerSampler, seconds: float) -> dict:
    """Measure whole-GPU idle power before model benchmarking."""
    sampler.start()
    try:
        time.sleep(seconds)
    finally:
        sampler.stop()
    return sampler.report({"measurement_kind": "idle", "requested_seconds": seconds})


@torch.no_grad()
def benchmark_inference_for_duration(
    model,
    image: torch.Tensor,
    sampler: GpuPowerSampler,
    *,
    warmup_iterations: int,
    measured_seconds: float,
    idle_power_w: float,
    synchronize: Callable | None = None,
) -> dict:
    """Benchmark until a minimum wall duration is reached."""
    if measured_seconds <= 0:
        raise ValueError("measured_seconds must be positive")
    synchronize = synchronize or torch.cuda.synchronize
    model.eval()
    for _ in range(warmup_iterations):
        model(image)
    synchronize(image.device)

    iterations = 0
    sampler.start()
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < measured_seconds:
            model(image)
            iterations += 1
        synchronize(image.device)
    finally:
        sampler.stop()
    seconds = time.perf_counter() - started
    images = iterations * len(image)
    if images == 0:
        raise RuntimeError("No inference iterations completed")
    return {
        "measurement_kind": "inference",
        "warmup_iterations": warmup_iterations,
        "requested_measurement_seconds": measured_seconds,
        "measured_iterations": iterations,
        "batch_size": len(image),
        "measured_images": images,
        "wall_seconds": seconds,
        "milliseconds_per_batch": seconds * 1000.0 / iterations,
        "milliseconds_per_image": seconds * 1000.0 / images,
        "images_per_second": images / seconds,
        "idle_power_w": idle_power_w,
    }
