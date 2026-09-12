from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


SUMMARY_NAMES = ("pckhn_summary.json", "best.json")
CANONICAL_FILES = (*SUMMARY_NAMES, "prediction_matrices.npz")
LEGACY_DIRECTORY = "metrics_original_video"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_complete_frame(summary: dict | None) -> bool:
    if not summary:
        return False
    return summary.get("coordinate_space") in {
        "original_image_pixels", "complete_frame_pixels",
    }


def canonicalize(summary: dict) -> dict:
    result = dict(summary)
    result["coordinate_space"] = "complete_frame_pixels"
    result["normalization"] = "head_to_neck_distance_in_original_image_pixels"
    result.pop("pckh", None)
    result.pop("per_joint_pckh", None)
    return result


def write_json(path: Path, value: dict, apply: bool) -> None:
    print(f"WRITE {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def migrate_run(run_dir: Path, apply: bool) -> tuple[int, int]:
    metrics_dir = run_dir / "metrics"
    legacy_dir = run_dir / LEGACY_DIRECTORY
    legacy_summary = read_json(legacy_dir / "pckhn_summary.json")
    current_summary = read_json(metrics_dir / "pckhn_summary.json")
    coordinate_series = {"jatc", "sn_series", "f_smoothnet", "occlusion"}
    intrinsically_complete_frame = any(part in coordinate_series for part in run_dir.parts)
    source_summary = legacy_summary if is_complete_frame(legacy_summary) else current_summary
    source_is_canonical = is_complete_frame(source_summary) or (
        intrinsically_complete_frame and source_summary is not None
    )
    writes = removals = 0

    if source_is_canonical:
        canonical = canonicalize(source_summary)
        for name in SUMMARY_NAMES:
            write_json(metrics_dir / name, canonical, apply)
            writes += 1
        legacy_matrices = legacy_dir / "prediction_matrices.npz"
        if legacy_matrices.is_file():
            print(f"COPY {legacy_matrices} -> {metrics_dir / 'prediction_matrices.npz'}")
            if apply:
                metrics_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_matrices, metrics_dir / "prediction_matrices.npz")
            writes += 1
        status_path = run_dir / "status.json"
        status = read_json(status_path)
        if status is not None:
            status["metrics"] = canonical
            if "pckhn" in status:
                status["pckhn"] = canonical["pckhn"]
            write_json(status_path, status, apply)
            writes += 1
    elif current_summary is not None and not is_complete_frame(current_summary):
        for name in SUMMARY_NAMES:
            target = metrics_dir / name
            if target.is_file():
                print(f"REMOVE non-canonical summary {target}")
                if apply:
                    target.unlink()
                removals += 1
        status_path = run_dir / "status.json"
        status = read_json(status_path)
        if status is not None and isinstance(status.get("metrics"), dict):
            if not is_complete_frame(status["metrics"]):
                status.pop("metrics", None)
                status.pop("pckhn", None)
                write_json(status_path, status, apply)
                writes += 1

    for name in SUMMARY_NAMES:
        target = legacy_dir / name
        if target.is_file():
            print(f"REMOVE legacy duplicate {target}")
            if apply:
                target.unlink()
            removals += 1
    return writes, removals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep complete-frame PCKhn as the sole NTU PCKhn result",
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    run_dirs = sorted({path.parent.parent for path in root.rglob("metrics/pckhn_summary.json")})
    run_dirs.extend(sorted({path.parent.parent for path in root.rglob(
        f"{LEGACY_DIRECTORY}/pckhn_summary.json"
    )} - set(run_dirs)))
    writes = removals = 0
    for run_dir in run_dirs:
        run_writes, run_removals = migrate_run(run_dir, args.apply)
        writes += run_writes
        removals += run_removals
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "root": str(root), "runs": len(run_dirs),
        "writes_or_copies": writes, "removals": removals,
    }, indent=2))


if __name__ == "__main__":
    main()
