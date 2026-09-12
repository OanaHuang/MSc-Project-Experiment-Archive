from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize heatmap-decoding model runs")
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    joint_rows = []
    for path in sorted(args.batch_dir.glob("*/seed_*/comparison.json")):
        model_id = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"model_id": model_id, "seed": seed, **item,
                         "result_path": str(path.parent)})
            joint_path = path.parent / "metrics" / item["variant_id"] / "dual_pckh_per_joint.csv"
            with joint_path.open(newline="", encoding="utf-8") as handle:
                for joint in csv.DictReader(handle):
                    joint_rows.append({
                        "model_id": model_id, "seed": seed,
                        "variant_id": item["variant_id"],
                        "joint_id": int(joint["joint_id"]),
                        "joint_name": joint["joint_name"],
                        "official_valid": int(joint["official_valid"]),
                        "official_pckh": float(joint["official_pckh"]),
                        "custom_valid": int(joint["custom_valid"]),
                        "custom_pckh": float(joint["custom_pckh"]),
                    })
    if not rows:
        raise RuntimeError(f"No completed heatmap comparisons under {args.batch_dir}")
    csv_path = args.batch_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.batch_dir / "model_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    variant_order = list(dict.fromkeys(row["variant_id"] for row in rows))
    wide_rows = []
    identities = list(dict.fromkeys(
        (row["model_id"], row["seed"]) for row in rows
    ))
    for model_id, seed in identities:
        selected = {
            row["variant_id"]: row for row in rows
            if row["model_id"] == model_id and row["seed"] == seed
        }
        wide = {"model_id": model_id, "seed": seed}
        for variant_id in variant_order:
            item = selected.get(variant_id, {})
            wide[f"{variant_id}_pckh"] = item.get("official_pckh")
            wide[f"{variant_id}_gain_pp"] = item.get("pckh_gain_pp")
        wide_rows.append(wide)
    wide_path = args.batch_dir / "model_comparison_wide.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0]))
        writer.writeheader()
        writer.writerows(wide_rows)
    reference = "argmax"
    protocol_paths = list(args.batch_dir.glob("*/seed_*/config/protocol.yaml"))
    if protocol_paths:
        protocol = yaml.safe_load(protocol_paths[0].read_text(encoding="utf-8"))
        reference = protocol.get("reference", reference)
    reference_scores = {
        (row["model_id"], row["seed"], row["joint_name"]): row["official_pckh"]
        for row in joint_rows if row["variant_id"] == reference
    }
    for row in joint_rows:
        baseline = reference_scores[
            (row["model_id"], row["seed"], row["joint_name"])
        ]
        row["pckh_gain_pp"] = 100.0 * (row["official_pckh"] - baseline)
    joint_long_path = args.batch_dir / "per_joint_comparison.csv"
    with joint_long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joint_rows[0]))
        writer.writeheader()
        writer.writerows(joint_rows)
    joint_names = list(dict.fromkeys(row["joint_name"] for row in joint_rows))
    joint_wide_rows = []
    for model_id, seed in identities:
        for variant_id in variant_order:
            selected = {
                row["joint_name"]: row for row in joint_rows
                if row["model_id"] == model_id and row["seed"] == seed
                and row["variant_id"] == variant_id
            }
            wide = {"model_id": model_id, "seed": seed,
                    "variant_id": variant_id}
            for joint_name in joint_names:
                wide[f"{joint_name}_pckh"] = selected[joint_name]["official_pckh"]
                wide[f"{joint_name}_gain_pp"] = selected[joint_name]["pckh_gain_pp"]
            joint_wide_rows.append(wide)
    joint_wide_path = args.batch_dir / "per_joint_comparison_wide.csv"
    with joint_wide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joint_wide_rows[0]))
        writer.writeheader()
        writer.writerows(joint_wide_rows)
    print(csv_path)
    print(wide_path)
    print(joint_long_path)
    print(joint_wide_path)


if __name__ == "__main__":
    main()
