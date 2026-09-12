from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time


def gpu_compute_process_count(physical_gpu: int) -> int:
    command = [
        "nvidia-smi", f"--id={int(physical_gpu)}",
        "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True,
                            text=True).stdout
    return len([line for line in output.splitlines() if line.strip()])


class GpuPowerSampler:
    """Sample whole-GPU power and integrate it over time."""

    def __init__(self, physical_gpu: int, interval: float = 0.5) -> None:
        self.physical_gpu = int(physical_gpu)
        self.interval = float(interval)
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> dict:
        command = [
            "nvidia-smi", f"--id={self.physical_gpu}",
            "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.run(command, check=True, capture_output=True,
                                text=True).stdout.strip()
        power, utilization, memory, temperature = [float(item.strip())
                                                    for item in output.split(",")]
        return {
            "monotonic_seconds": time.monotonic(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "power_w": power,
            "utilization_percent": utilization,
            "memory_mib": memory,
            "temperature_c": temperature,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._query())
            except Exception as error:
                self.errors.append(str(error))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples.clear()
        self.errors.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 4))

    def report(self, extra: dict | None = None) -> dict:
        """Build a summary without writing files, useful for repeated runs."""
        energy_j = 0.0
        for previous, current in zip(self.samples, self.samples[1:]):
            seconds = current["monotonic_seconds"] - previous["monotonic_seconds"]
            energy_j += 0.5 * (previous["power_w"] + current["power_w"]) * seconds
        duration = (self.samples[-1]["monotonic_seconds"] -
                    self.samples[0]["monotonic_seconds"]) if len(self.samples) > 1 else 0.0
        report = {
            "physical_gpu": self.physical_gpu,
            "samples": len(self.samples),
            "sample_interval_seconds": self.interval,
            "measured_duration_seconds": duration,
            "energy_joules": energy_j,
            "energy_wh": energy_j / 3600.0,
            "average_power_w": energy_j / duration if duration > 0 else None,
            "peak_memory_mib": max((item["memory_mib"] for item in self.samples), default=None),
            "errors": self.errors,
            "measurement_scope": "whole physical GPU",
            **(extra or {}),
        }
        if extra and extra.get("measured_iterations") and extra.get("batch_size"):
            images = int(extra["measured_iterations"]) * int(extra["batch_size"])
            report["energy_joules_per_image"] = energy_j / images
            idle_power = extra.get("idle_power_w")
            if idle_power is not None:
                dynamic_energy = energy_j - float(idle_power) * duration
                report["dynamic_energy_joules"] = dynamic_energy
                report["dynamic_energy_joules_per_image"] = dynamic_energy / images
                report["dynamic_energy_nonnegative"] = dynamic_energy >= 0
        return report

    def save(self, csv_path: Path, summary_path: Path,
             extra: dict | None = None) -> dict:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.samples:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.samples[0].keys())
                writer.writeheader()
                writer.writerows(self.samples)
        report = self.report(extra)
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
