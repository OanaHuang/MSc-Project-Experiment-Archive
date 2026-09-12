#!/usr/bin/env python3
"""Reconcile stale training ``running`` states after a safe server pause.

Only active runs below ``Outputs_Thesis_Pilot20/runs`` are considered.  A run
is changed only when no training process is present, its recorded PID is dead,
and a resumable ``last.pt`` plus at least one completed history epoch exist.
Every original status file is preserved beside it before the atomic update.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from spikepose_thesis.core.paths import PROJECT_ROOT


DEFAULT_ROOT = PROJECT_ROOT / "Outputs_Thesis_Pilot20" / "runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true")
    return parser


def _process_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _training_processes() -> list[str]:
    output = subprocess.run(
        ["ps", "-eo", "pid=,stat=,cmd="], check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout
    return [
        line.strip() for line in output.splitlines()
        if "spikepose_thesis" in line
        and (" train " in line or " refine " in line)
        and "reconcile_paused_training_status.py" not in line
    ]


def _last_epoch(run: Path) -> int:
    history = run / "training/history.csv"
    if not history.is_file():
        return 0
    with history.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["epoch"]) if rows else 0


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    live = _training_processes()
    if live:
        raise RuntimeError(
            "Refusing to reconcile while training processes exist:\n"
            + "\n".join(live)
        )
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    updates = []
    for status_path in sorted(root.rglob("status.json")):
        run = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") not in {"running", "initialized"}:
            continue
        previous_pid = status.get("pid")
        if _process_alive(previous_pid):
            raise RuntimeError(f"Recorded process is still alive: {status_path}")
        config_path = run / "resolved_config.yaml"
        checkpoint = run / "checkpoints/last.pt"
        if not config_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"Stale run is not safely resumable: {run}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        completed_epoch = _last_epoch(run)
        target_epoch = int(config.get("training", {}).get("epochs", 0))
        if not 0 < completed_epoch < target_epoch:
            raise RuntimeError(
                f"Unexpected epoch state for {run}: "
                f"{completed_epoch}/{target_epoch}"
            )
        updated = {
            **status,
            "state": "paused",
            "previous_state": status.get("state"),
            "pid": None,
            "previous_pid": previous_pid,
            "completed_epoch": completed_epoch,
            "target_epoch": target_epoch,
            "resumable": True,
            "resume_checkpoint": str(checkpoint),
            "pause_reason": "safe_pause_no_live_training_process",
            "status_reconciled_at": timestamp.isoformat(),
        }
        record = {
            "experiment": updated.get("experiment", config.get("id")),
            "status": str(status_path),
            "previous_state": status.get("state"),
            "new_state": "paused",
            "completed_epoch": completed_epoch,
            "target_epoch": target_epoch,
            "resumable": True,
        }
        if args.apply:
            backup = run / f"status.pre_pause_reconcile_{stamp}.json"
            if backup.exists():
                raise FileExistsError(backup)
            shutil.copy2(status_path, backup)
            temporary = run / "status.json.tmp"
            temporary.write_text(json.dumps(updated, indent=2), encoding="utf-8")
            temporary.replace(status_path)
            record["backup"] = str(backup)
        updates.append(record)
    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "root": str(root),
        "training_processes": live,
        "updated_runs": len(updates),
        "updates": updates,
    }, indent=2))


if __name__ == "__main__":
    main()
