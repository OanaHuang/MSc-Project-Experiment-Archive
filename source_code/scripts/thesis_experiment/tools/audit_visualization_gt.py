#!/usr/bin/env python3
"""Independently audit visualization GT arrays against raw NTU skeleton files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "scripts" / "thesis_experiment" / "src"
sys.path.insert(0, str(SRC))

from spikepose_thesis.data.ntu.core import (  # noqa: E402
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)
from spikepose_thesis.data.ntu.core.joint_mapping import (  # noqa: E402
    map_ntu25_to_mpii16,
)


def main() -> None:
    root = PROJECT_ROOT / "Video_Visualization"
    with (root / "selected_test_metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    model_ids = ("mamv2_fullcs20", "mamv2_fullcs_p00_source")
    summary = {"samples": len(rows), "models": {}, "cross_model_gt": {}}
    gt_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_id in model_ids:
        model_stats = {
            "frames": 0,
            "visible_joints": 0,
            "max_coordinate_error_px": 0.0,
            "mean_coordinate_error_px": 0.0,
            "visibility_mismatches": 0,
            "frame_index_mismatches": 0,
            "video_frame_mismatches": 0,
        }
        errors = []
        gt_by_model[model_id] = {}
        for row in rows:
            sample_id = row["sample_id"]
            sequence = read_skeleton_file(Path(row["skeleton_path"]))
            pose = extract_primary_pose_sequence(sequence)
            expected_gt = map_ntu25_to_mpii16(pose["color_xy"])
            expected_vis = map_ntu25_to_mpii16(
                coordinate_visibility(
                    pose["color_xy"],
                    tracking_state=pose["tracking_state"],
                    image_size=(int(row["width"]), int(row["height"])),
                    include_inferred=False,
                )[..., None]
            )[..., 0].astype(bool)
            archive_path = root / "models" / model_id / "predictions" / f"{sample_id}.npz"
            with np.load(archive_path) as archive:
                frames = archive["frame_index"].astype(int)
                actual_gt = archive["ground_truth"]
                actual_vis = archive["visibility"].astype(bool)
            gt_by_model[model_id][sample_id] = actual_gt
            expected_frames = np.arange(sequence.num_frames)
            model_stats["frame_index_mismatches"] += int(
                not np.array_equal(frames, expected_frames)
            )
            selected_expected = expected_gt[frames]
            selected_vis = expected_vis[frames]
            joint_mask = selected_vis & actual_vis
            distances = np.linalg.norm(actual_gt - selected_expected, axis=-1)
            errors.extend(distances[joint_mask].tolist())
            model_stats["visibility_mismatches"] += int(
                np.count_nonzero(selected_vis != actual_vis)
            )
            model_stats["frames"] += len(frames)
            model_stats["visible_joints"] += int(np.count_nonzero(joint_mask))
            video_path = root / "models" / model_id / "videos" / f"{sample_id}_pred_vs_gt.mp4"
            capture = cv2.VideoCapture(str(video_path))
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            model_stats["video_frame_mismatches"] += int(video_frames != len(frames))
        model_stats["max_coordinate_error_px"] = float(max(errors, default=0.0))
        model_stats["mean_coordinate_error_px"] = float(np.mean(errors)) if errors else 0.0
        summary["models"][model_id] = model_stats
    for row in rows:
        sample_id = row["sample_id"]
        left = gt_by_model[model_ids[0]][sample_id]
        right = gt_by_model[model_ids[1]][sample_id]
        summary["cross_model_gt"][sample_id] = float(np.max(np.abs(left - right)))
    summary["cross_model_max_abs_error_px"] = max(summary["cross_model_gt"].values())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
