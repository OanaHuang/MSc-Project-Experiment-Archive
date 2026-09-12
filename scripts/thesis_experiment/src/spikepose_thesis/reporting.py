from __future__ import annotations

import json
from pathlib import Path
import statistics

from spikepose_thesis.artifacts import run_dir
from spikepose_thesis.experiments import plan_study


def _flatten(value: dict, prefix: str = "") -> dict[str, float]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def aggregate_experiment(config: dict, output_root: Path | None, split: str) -> dict:
    rows = []
    for seed in config["training"]["seeds"]:
        root = run_dir(config, seed, output_root)
        path = root / "predictions" / split / "summary.json"
        if path.is_file():
            values = _flatten(json.loads(path.read_text(encoding="utf-8")))
            theoretical = root / "analysis" / "theoretical.json"
            if theoretical.is_file():
                values.update(_flatten(
                    json.loads(theoretical.read_text(encoding="utf-8")),
                    "efficiency",
                ))
            rows.append((seed, values))
    metrics = {}
    names = sorted(set.intersection(*(set(row) for _, row in rows))) if rows else []
    for name in names:
        values = [row[name] for _, row in rows]
        metrics[name] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return {
        "experiment": config["id"], "paper_id": config["paper_id"],
        "expected_seeds": config["training"]["seeds"],
        "available_seeds": [seed for seed, _ in rows], "metrics": metrics,
        "temporal": config.get("temporal", {}),
        "refinement": config.get("refinement", {}),
    }


def _metric(row: dict, name: str) -> str:
    value = row["metrics"].get(name)
    if not value:
        return "TBD"
    mean, std = value["mean"], value["sample_std"]
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def generate_tables(study: str, output_dir: Path,
                    output_root: Path | None = None, split: str = "test") -> dict:
    configs = plan_study(study)
    rows = [aggregate_experiment(config, output_root, split) for config in configs]
    sections = {
        "Table A — MPII spatial": [
            row for row in rows if "_m_" in row["experiment"]
        ],
        "Table B — NTU frame/factor": [
            row for row in rows
            if "factor_" in row["experiment"] or "_ntu_" in row["experiment"]
        ],
        "Table C — MAM and ablations": [
            row for row in rows if "mamv2" in row["experiment"]
        ],
        "Table D — evaluation-only controls": [
            row for row in rows
            if "control_" in row["experiment"] or "_eval_" in row["experiment"]
        ],
    }
    lines = [f"# {study} aggregated results", ""]
    for title, items in sections.items():
        lines.extend((f"## {title}", ""))
        if title.startswith(("Table A", "Table B")):
            lines.extend((
                "| Model | Seeds | PCK@0.5 | PCK@0.1 | Params | MACs | SynOps | Firing rate | Est. energy (mJ) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ))
            for row in items:
                lines.append(
                    f"| {row['paper_id']} | {len(row['available_seeds'])}/{len(row['expected_seeds'])} "
                    f"| {_metric(row, 'pck_0.5')} | {_metric(row, 'pck_0.1')} "
                    f"| {_metric(row, 'efficiency.parameters')} | {_metric(row, 'efficiency.dense_macs')} "
                    f"| {_metric(row, 'efficiency.effective_sops')} "
                    f"| {_metric(row, 'efficiency.operation_weighted_firing_rate')} "
                    f"| {_metric(row, 'efficiency.total_theoretical_energy_mj')} |"
                )
        else:
            lines.extend((
                "| Model | Seeds | PCK@0.5 | AccE | RAccE | AMR | HM-PCK | HM-AccE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ))
            for row in items:
                lines.append(
                    f"| {row['paper_id']} | {len(row['available_seeds'])}/{len(row['expected_seeds'])} "
                    f"| {_metric(row, 'pck_0.5')} "
                    f"| {_metric(row, 'temporal.acceleration_error')} "
                    f"| {_metric(row, 'temporal.relative_acceleration_error')} "
                    f"| {_metric(row, 'temporal.acceleration_magnitude_ratio')} "
                    f"| {_metric(row, 'temporal.high_motion_pck_0_5')} "
                    f"| {_metric(row, 'temporal.high_motion_acceleration_error')} |"
                )
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"study": study, "split": split, "experiments": rows}
    (output_dir / "tables.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
