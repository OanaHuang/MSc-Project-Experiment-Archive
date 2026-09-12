#!/usr/bin/env python3
"""Evaluate pose methods on GT-defined low/medium/high-motion videos."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.evaluation.speed_strata import (
    STRATA,
    assign_speed_tertiles,
    load_prediction_archive,
    speed_stratum_metrics,
    validate_shared_ground_truth,
    video_speed_scores,
)


LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Prediction archive supplying the canonical test rows and GT.",
    )
    parser.add_argument(
        "--prediction", action="append", required=True, metavar="LABEL=PATH",
        help="Method prediction archive; repeat once per method/reset policy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization", choices=("pose_bbox_diagonal", "head_bone"),
        default="pose_bbox_diagonal",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _prediction_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction must use LABEL=PATH syntax: {value!r}")
        label, path_value = value.split("=", 1)
        if not LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"Invalid prediction label: {label!r}")
        if label in labels:
            raise ValueError(f"Duplicate prediction label: {label}")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        result.append((label, path))
    return result


def _write_manifest(path: Path, assignments: list[dict[str, object]]) -> None:
    fields = (
        "video_id", "person_id", "clip_id", "frames",
        "valid_joint_transitions", "gt_movement_per_frame",
        "gt_movement_per_second", "gt_displacement_px_per_frame",
        "stratum", "speed_rank",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)


def _speed_statistics(assignments: list[dict[str, object]]) -> dict:
    result = {}
    for label in STRATA:
        scores = np.asarray([
            float(row["gt_movement_per_frame"])
            for row in assignments if row["stratum"] == label
        ])
        result[label] = {
            "videos": int(len(scores)),
            "minimum": float(scores.min()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "maximum": float(scores.max()),
        }
    return result


def _write_metrics_csv(path: Path, methods: dict[str, dict]) -> None:
    fields = (
        "method", "stratum", "videos", "frames",
        "pck_hb_0_5_all_frames", "pck_hb_0_5_video_macro",
        "vel_e_px_video_macro", "acc_e_px_video_macro", "amr_video_macro",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, payload in methods.items():
            for stratum in STRATA:
                writer.writerow({
                    "method": method,
                    "stratum": stratum,
                    **payload["metrics"][stratum],
                })


def main() -> None:
    args = _parser().parse_args()
    reference_path = args.reference.expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    specifications = _prediction_specs(args.prediction)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = load_prediction_archive(reference_path)
    scores = video_speed_scores(
        reference, normalization=args.normalization, fps=args.fps,
    )
    assignments, tertile_protocol = assign_speed_tertiles(scores)
    _write_manifest(output / "video_speed_strata.csv", assignments)

    methods = {}
    for label, path in specifications:
        values = (
            reference if path == reference_path else load_prediction_archive(path)
        )
        validate_shared_ground_truth(reference, values)
        methods[label] = {
            "prediction_archive": str(path),
            "prediction_sha256": _sha256(path),
            "metrics": speed_stratum_metrics(values, assignments),
        }
        print(f"speed_strata_complete method={label}", flush=True)
        if values is not reference:
            del values

    protocol = {
        "name": "gt_video_movement_tertiles_v1",
        "stratification_source": "ground_truth_only",
        "normalization": args.normalization,
        "fps": float(args.fps),
        "movement_score": (
            "Mean visible-joint Euclidean displacement between consecutive "
            "frames, divided by the mean adjacent-frame person-size scale."
        ),
        "person_size": (
            "Diagonal of the visible GT-joint bounding box per frame."
            if args.normalization == "pose_bbox_diagonal"
            else "GT 2D head-to-neck distance per frame."
        ),
        "relationship_to_mtpose": (
            "MTPose-inspired extension: MTPose averages bbox-normalized joint "
            "movement to compare joints; this protocol averages it per video "
            "and forms equal-count video tertiles. It is not an exact "
            "reproduction of the MTPose joint-ranking experiment."
        ),
        "mtpose_reference": (
            "Motion-Aware Heatmap Regression for Human Pose Estimation in "
            "Videos, IJCAI 2024, https://www.ijcai.org/proceedings/2024/0138.pdf"
        ),
        **tertile_protocol,
    }
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "speed_statistics": _speed_statistics(assignments),
        "reference": {
            "prediction_archive": str(reference_path),
            "prediction_sha256": _sha256(reference_path),
        },
        "methods": methods,
        "artifacts": {
            "video_manifest": str(output / "video_speed_strata.csv"),
            "metrics_csv": str(output / "speed_strata_metrics.csv"),
            "metrics_json": str(output / "speed_strata_metrics.json"),
        },
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
    }
    _write_metrics_csv(output / "speed_strata_metrics.csv", methods)
    (output / "speed_strata_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "videos": len(assignments),
        "counts": protocol["counts"],
        "methods": list(methods),
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
