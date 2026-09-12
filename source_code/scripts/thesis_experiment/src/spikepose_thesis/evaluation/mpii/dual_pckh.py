from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from spikepose_thesis.data.mpii.core.constants import MPII_JOINT_NAMES


def compute_dual_pckh(pred, gt, visibility, official_head_length,
                      custom_head_length, threshold=0.5):
    """Evaluate identical predictions with official and project PCKh scales."""
    pred, gt = np.asarray(pred, np.float32), np.asarray(gt, np.float32)
    visibility = np.asarray(visibility) > 0
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError("pred and gt must have matching shape [N, J, 2]")
    finite = np.isfinite(pred).all(-1) & np.isfinite(gt).all(-1)
    base_valid = visibility & finite
    distance = np.linalg.norm(pred - gt, axis=-1).astype(np.float32)
    result = {"distance_matrix": distance}
    for label, scale in (("official", official_head_length), ("custom", custom_head_length)):
        scale = np.asarray(scale, np.float32).reshape(-1)
        if len(scale) != len(pred):
            raise ValueError(f"{label} head scale must have shape [N]")
        valid = base_valid & (np.isfinite(scale) & (scale > 0))[:, None]
        normalized = np.full(distance.shape, np.nan, np.float32)
        np.divide(distance, scale[:, None], out=normalized, where=valid)
        correct = valid & (normalized <= threshold)
        result[f"{label}_normalized_distance_matrix"] = normalized
        result[f"{label}_valid_matrix"] = valid
        result[f"{label}_pckh_correct_matrix"] = correct
    result["threshold"] = float(threshold)
    return result


def _score_rows(matrices, label):
    valid = matrices[f"{label}_valid_matrix"]
    correct = matrices[f"{label}_pckh_correct_matrix"]
    rows = []
    for joint_id, joint_name in enumerate(MPII_JOINT_NAMES):
        count = int(valid[:, joint_id].sum())
        score = float(correct[:, joint_id].sum() / count) if count else float("nan")
        rows.append({"joint_id": joint_id, "joint_name": joint_name,
                     "valid": count, f"{label}_pckh": score})
    all_count = int(valid.sum())
    all_score = float(correct.sum() / all_count) if all_count else float("nan")
    # Official MPII reporting masks pelvis (6) and thorax (7), then weights
    # every remaining joint by its number of valid annotations.
    report_joints = np.ones(valid.shape[1], dtype=bool)
    report_joints[6:8] = False
    report_valid = valid[:, report_joints]
    report_correct = correct[:, report_joints]
    report_count = int(report_valid.sum())
    report_score = float(report_correct.sum() / report_count) if report_count else float("nan")
    return rows, report_score, report_count, all_score, all_count


def _grouped_scores(rows, key):
    by_id = {row["joint_id"]: row[key] for row in rows}
    return {
        "head": by_id[9], "shoulder": 0.5 * (by_id[12] + by_id[13]),
        "elbow": 0.5 * (by_id[11] + by_id[14]),
        "wrist": 0.5 * (by_id[10] + by_id[15]),
        "hip": 0.5 * (by_id[2] + by_id[3]),
        "knee": 0.5 * (by_id[1] + by_id[4]),
        "ankle": 0.5 * (by_id[0] + by_id[5]),
    }


def save_dual_pckh_reports(output_dir, matrices, model_name, samples):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_rows, official, official_count, official_all, official_all_count = _score_rows(matrices, "official")
    custom_rows, custom, custom_count, custom_all, custom_all_count = _score_rows(matrices, "custom")
    combined = []
    for official_row, custom_row in zip(official_rows, custom_rows):
        combined.append({
            "joint_id": official_row["joint_id"], "joint_name": official_row["joint_name"],
            "official_valid": official_row["valid"], "official_pckh": official_row["official_pckh"],
            "custom_valid": custom_row["valid"], "custom_pckh": custom_row["custom_pckh"],
            "custom_minus_official": custom_row["custom_pckh"] - official_row["official_pckh"],
        })
    with (output_dir / "dual_pckh_per_joint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined[0].keys())
        writer.writeheader(); writer.writerows(combined)
    summary = {
        "model": model_name, "samples": int(samples), "threshold": matrices["threshold"],
        "official_valid_joints": official_count, "official_pckh": official,
        "custom_valid_joints": custom_count, "custom_pckh": custom,
        "custom_minus_official": custom - official,
        "official_all_16_joints_pckh": official_all,
        "custom_all_16_joints_pckh": custom_all,
        "official_all_16_valid_joints": official_all_count,
        "custom_all_16_valid_joints": custom_all_count,
        "official_grouped_pckh": _grouped_scores(official_rows, "official_pckh"),
        "custom_grouped_pckh": _grouped_scores(custom_rows, "custom_pckh"),
        "official_per_joint_pckh": {
            row["joint_name"]: row["official_pckh"] for row in combined
        },
        "custom_per_joint_pckh": {
            row["joint_name"]: row["custom_pckh"] for row in combined
        },
        "official_per_joint_valid": {
            row["joint_name"]: row["official_valid"] for row in combined
        },
        "custom_per_joint_valid": {
            row["joint_name"]: row["custom_valid"] for row in combined
        },
        "reported_mean_excludes_joint_ids": [6, 7],
        "custom_definition": (
            "0.75 * distance(head_top[9], upper_neck[8]); equivalent to "
            "distance(HeadCenter, NeckCenter) at r=0.25"
        ),
        "official_definition": "0.6 * diagonal(head_rectangle)",
    }
    (output_dir / "dual_pckh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
