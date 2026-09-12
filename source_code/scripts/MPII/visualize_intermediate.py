from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.intermediate_visualization import visualize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SpikePose stage features")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--groups", nargs="+", choices=("first", "random"),
                        default=("first", "random"))
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.run / "intermediate_visualizations"
    summary = visualize_run(
        PROJECT_ROOT, args.run.resolve(), args.manifest.resolve(), output.resolve(),
        args.device, tuple(args.groups), args.max_per_group, args.save_raw,
    )
    print(f"Generated {summary['samples']} samples in {output.resolve()}")


if __name__ == "__main__":
    main()

