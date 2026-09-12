#!/usr/bin/env python3
"""Print a compact Pilot20 or Confirm140 training checklist."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spikepose_thesis.artifacts import run_dir
from spikepose_thesis.data import audit_data
from spikepose_thesis.experiments import experiment_readiness

from run_icassp2027_queue import WORKFLOWS, _resolved


PHASES = ("pilot20", "confirm140")


def _completed_epoch(path: Path, status: dict) -> int:
    if status.get("completed_epoch") is not None:
        return int(status["completed_epoch"])
    history = path / "training" / "history.csv"
    try:
        with history.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return int(rows[-1]["epoch"]) if rows else 0
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return 0


def collect_status(phase: str) -> list[dict]:
    report = audit_data()
    rows = []
    for wave, specs in WORKFLOWS[phase]:
        for spec in specs:
            config, readiness_config = _resolved(spec)
            for seed in config["training"]["seeds"]:
                path = run_dir(config, seed)
                try:
                    status = json.loads(
                        (path / "status.json").read_text(encoding="utf-8")
                    )
                except (FileNotFoundError, json.JSONDecodeError):
                    status = {}
                readiness = experiment_readiness(
                    readiness_config, seed, report=report,
                )
                state = readiness["status"]
                if state == "READY":
                    state = "NOT_STARTED"
                rows.append({
                    "wave": wave,
                    "experiment": config["id"],
                    "seed": int(seed),
                    "state": state,
                    "completed_epoch": _completed_epoch(path, status),
                    "target_epoch": int(config["training"]["epochs"]),
                    "blockers": readiness["blockers"],
                    "run_dir": str(path),
                })
    return rows


def _markdown(rows: list[dict]) -> str:
    lines = [
        "| Wave | Experiment | Seed | Status | Progress | Blocker |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        blocker = "; ".join(row["blockers"])
        progress = f'{row["completed_epoch"]}/{row["target_epoch"]}'
        lines.append(
            f'| {row["wave"]} | {row["experiment"]} | {row["seed"]} '
            f'| {row["state"]} | {progress} | {blocker} |'
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="pilot20")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = collect_status(args.phase)
    print(json.dumps(rows, indent=2) if args.json else _markdown(rows))


if __name__ == "__main__":
    main()
