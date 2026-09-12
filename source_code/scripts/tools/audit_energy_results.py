from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def classify_training(report: dict) -> tuple[str, str]:
    duration = float(report.get("measured_duration_seconds") or 0)
    completed = report.get("completed_epochs")
    requested = report.get("requested_epochs") or report.get("epochs")
    start_epoch = int(report.get("start_epoch") or 1)
    if duration <= 0 or int(report.get("samples") or 0) < 2:
        return "invalid", "no usable power interval"
    if completed is None:
        return "pending_review", "completed epoch count was not recorded"
    expected_measured = int(completed) - start_epoch + 1
    measured = report.get("measured_epochs", expected_measured)
    if int(measured) != expected_measured:
        return "invalid", "epoch coverage metadata is inconsistent"
    if requested is not None and int(completed) < int(requested):
        return "partial", "measurement ended before the requested final epoch"
    if start_epoch != 1:
        return "partial", "measurement covers only a resumed training segment"
    return "valid", "complete training measurement"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit New-framework energy JSON files")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.output_root.rglob("training_energy.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        status, reason = classify_training(payload)
        rows.append({"path": str(path), "status": status, "reason": reason,
                     "duration_seconds": payload.get("measured_duration_seconds"),
                     "energy_wh": payload.get("energy_wh"),
                     "start_epoch": payload.get("start_epoch"),
                     "completed_epochs": payload.get("completed_epochs"),
                     "requested_epochs": payload.get("requested_epochs",
                                                     payload.get("epochs"))})
    durations = [float(row["duration_seconds"]) for row in rows
                 if row["duration_seconds"] is not None and
                 float(row["duration_seconds"]) > 0]
    median_duration = statistics.median(durations) if durations else None
    if median_duration:
        for row in rows:
            duration = float(row["duration_seconds"] or 0)
            if duration < median_duration * 0.25:
                row["status"] = "invalid"
                row["reason"] = (
                    "duration is below 25% of the batch median; likely partial capture")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        fields = ["path", "status", "reason", "duration_seconds", "energy_wh",
                  "start_epoch", "completed_epochs", "requested_epochs"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.report)


if __name__ == "__main__":
    main()
