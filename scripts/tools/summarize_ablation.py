from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed ablation runs")
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for experiment_dir in sorted(path for path in args.batch_dir.iterdir() if path.is_dir()):
        metrics = []
        for path in experiment_dir.glob("seed_*/metrics/best.json"):
            payload = json.loads(path.read_text())
            key = "official_pckh" if "official_pckh" in payload else "pckh"
            metrics.append(float(payload[key]))
        if metrics:
            config_path = next(experiment_dir.glob("seed_*/config/resolved.yaml"), None)
            config = (yaml.safe_load(config_path.read_text()) if config_path else {})
            rows.append({
                "experiment_id": config.get("id", experiment_dir.name),
                "model_name": config.get("name", experiment_dir.name),
                "runs": len(metrics),
                "mean": statistics.mean(metrics),
                "std": statistics.stdev(metrics) if len(metrics) > 1 else 0.0,
                "min": min(metrics),
                "max": max(metrics),
            })
    path = args.batch_dir / "ablation_summary.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
