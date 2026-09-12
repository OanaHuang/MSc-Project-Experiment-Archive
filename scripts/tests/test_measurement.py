from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.analysis.hardware import GpuPowerSampler
from scripts.spikepose.analysis.measurement import (
    InferenceMeasurementProtocol,
    summarize_repeats,
)
from scripts.tools.audit_energy_results import classify_training


class MeasurementTests(unittest.TestCase):
    def test_sampler_integrates_and_subtracts_idle_power(self) -> None:
        sampler = GpuPowerSampler(0)
        sampler.samples = [
            {"monotonic_seconds": 0.0, "timestamp_utc": "a", "power_w": 100.0,
             "utilization_percent": 50.0, "memory_mib": 1000.0,
             "temperature_c": 50.0},
            {"monotonic_seconds": 2.0, "timestamp_utc": "b", "power_w": 100.0,
             "utilization_percent": 50.0, "memory_mib": 1200.0,
             "temperature_c": 51.0},
        ]
        report = sampler.report({"measured_iterations": 10, "batch_size": 2,
                                 "idle_power_w": 40.0})
        self.assertAlmostEqual(report["energy_joules"], 200.0)
        self.assertAlmostEqual(report["energy_joules_per_image"], 10.0)
        self.assertAlmostEqual(report["dynamic_energy_joules_per_image"], 6.0)
        self.assertTrue(report["dynamic_energy_nonnegative"])

    def test_repeat_summary_flags_unstable_energy(self) -> None:
        reports = [
            {"energy_joules_per_image": value,
             "dynamic_energy_joules_per_image": value,
             "milliseconds_per_image": 10.0,
             "images_per_second": 100.0,
             "average_power_w": 100.0,
             "peak_memory_mib": 1000.0}
            for value in (1.0, 1.2, 0.8, 1.1, 0.9)
        ]
        summary = summarize_repeats(reports, maximum_cv=0.05)
        self.assertEqual(summary["quality"]["status"], "review")
        self.assertTrue(summary["quality"]["additional_repeats_recommended"])

    def test_training_audit_detects_partial_and_valid(self) -> None:
        valid = {"samples": 10, "measured_duration_seconds": 100,
                 "start_epoch": 1, "completed_epochs": 120,
                 "requested_epochs": 120, "measured_epochs": 120}
        partial = {**valid, "start_epoch": 81, "measured_epochs": 40}
        self.assertEqual(classify_training(valid)[0], "valid")
        self.assertEqual(classify_training(partial)[0], "partial")

    def test_protocol_validation(self) -> None:
        InferenceMeasurementProtocol().validate()
        with self.assertRaises(ValueError):
            InferenceMeasurementProtocol(repeats=1).validate()


if __name__ == "__main__":
    unittest.main()
