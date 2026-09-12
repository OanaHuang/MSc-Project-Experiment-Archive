from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (PROJECT_ROOT / "Server_outputs_New", PROJECT_ROOT / "Outputs_New")


def _is_completed(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _official_result_complete(run_dir: Path) -> bool:
    path = run_dir / "metrics" / "dual_pckh_summary.json"
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("flip_test") is True
        and summary.get("flip_shift") is True
        and summary.get("decoder") == "quarter"
        and summary.get("udp") is False
        and int(summary.get("samples", 0)) == 2925
    )


def discover_completed_runs(roots: list[Path]) -> list[Path]:
    runs: dict[Path, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for checkpoint in root.rglob("checkpoints/best.pt"):
            run_dir = checkpoint.parents[1]
            if not (run_dir / "config/resolved.yaml").is_file() or not _is_completed(run_dir):
                continue
            # A resolved run path is unique; keep the first root to avoid evaluating
            # a local mirror twice when Outputs_New and Server_outputs_New overlap.
            relative = run_dir.relative_to(root)
            runs.setdefault(relative, run_dir)
    return [runs[key] for key in sorted(runs)]


def evaluate_run(run_dir: Path, python: Path) -> None:
    command = [
        str(python),
        str(PROJECT_ROOT / "scripts/MPII/evaluate.py"),
        "--run", str(run_dir),
        "--device", "mps",
        "--flip-test",
        "--decoder", "quarter",
        "--output-dir", str(run_dir / "metrics"),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=True,
        capture_output=True, text=True,
    )
    if not _official_result_complete(run_dir):
        raise RuntimeError(f"Official-compatible evaluation did not validate: {run_dir}")
    summary = json.loads(
        (run_dir / "metrics" / "dual_pckh_summary.json").read_text(encoding="utf-8")
    )
    print(
        f"completed: PCKh={float(summary['official_pckh']):.4f}, "
        f"PCKHN={float(summary['custom_pckh']):.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite metrics for every completed MPII run with the official-compatible "
            "Flip + Shift + Quarter-pixel protocol on Apple MPS."
        ),
    )
    parser.add_argument("--roots", nargs="+", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    runs = discover_completed_runs([path.resolve() for path in args.roots])
    if not runs:
        raise RuntimeError("No completed runs with checkpoints were found")
    for index, run_dir in enumerate(runs, start=1):
        state = "ready" if args.force or not _official_result_complete(run_dir) else "complete"
        print(f"[{index}/{len(runs)}] {state}: {run_dir}", flush=True)
        if args.list or state == "complete":
            continue
        evaluate_run(run_dir, args.python)


if __name__ == "__main__":
    main()
