from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.intermediate_visualization import visualize_run


def discover_runs(profile_dir: Path) -> list[Path]:
    return sorted(path.parent.parent for path in profile_dir.glob("*/seed_*/checkpoints/best.pt"))


def build_comparisons(output_root: Path, run_outputs: list[tuple[str, Path]],
                      groups: tuple[str, ...], max_per_group: int | None) -> None:
    comparison_root = output_root / "comparisons"
    for group in groups:
        available = []
        limit = max_per_group or 10
        for order in range(1, limit + 1):
            panels = []
            for label, directory in run_outputs:
                image = cv2.imread(str(directory / "samples" / group / f"{order:02d}" /
                                       "stage_overview.png"))
                if image is None:
                    continue
                cv2.rectangle(image, (0, 0), (230, 32), (10, 18, 30), -1)
                cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(image)
            if not panels:
                continue
            width = min(panel.shape[1] for panel in panels)
            panels = [cv2.resize(panel, (width, round(panel.shape[0] * width / panel.shape[1])))
                      for panel in panels]
            target = comparison_root / group
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{order:02d}_e_series.png"), cv2.vconcat(panels))
            available.append(order)
        (comparison_root / f"{group}_samples.json").write_text(
            json.dumps(available, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize B0-E9 intermediate features")
    parser.add_argument("--profile-dir", type=Path, default=(
        PROJECT_ROOT / "Server_outputs_New" / "mpii" / "architecture_evolution" /
        "profile_b"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=(
        PROJECT_ROOT / "Server_outputs_New" / "mpii" / "architecture_evolution" /
        "intermediate_visualization" / "profile_b"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--groups", nargs="+", choices=("first", "random"),
                        default=("first", "random"))
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()
    profile_dir = args.profile_dir.resolve()
    manifest = (args.manifest or profile_dir / "sample_manifest.json").resolve()
    output_root = args.output_root.resolve()
    runs = discover_runs(profile_dir)
    if not runs:
        raise FileNotFoundError(f"No B0-E9 checkpoints found below {profile_dir}")
    run_outputs = []
    summaries = []
    for run in runs:
        label = run.parent.name
        destination = output_root / label / run.name
        print(f"Visualizing {label}/{run.name}", flush=True)
        summary = visualize_run(
            PROJECT_ROOT, run, manifest, destination, args.device,
            tuple(args.groups), args.max_per_group, args.save_raw,
        )
        summaries.append({"model": label, "seed": run.name, **summary})
        run_outputs.append((label, destination))
    build_comparisons(output_root, run_outputs, tuple(args.groups), args.max_per_group)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "series_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "seed", "status", "checkpoint_epoch", "samples", "run"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    print(f"Generated {len(runs)} model visualizations in {output_root}")


if __name__ == "__main__":
    main()
