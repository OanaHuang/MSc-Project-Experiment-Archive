from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_mpii_head_bone_scale(metadata_path: Path, output_path: Path) -> dict:
    """Fit the checklist's fixed official-head / head-bone scale on train only."""
    ratios = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keypoints = np.asarray(row["keypoints"], dtype=np.float64)
            visibility = np.asarray(row["visibility"], dtype=np.float64)
            official = float(row["head_length"])
            bone = float(np.linalg.norm(keypoints[9] - keypoints[8]))
            if visibility[8] > 0 and visibility[9] > 0 and official > 0 and bone > 0:
                ratios.append(official / bone)
    if not ratios:
        raise RuntimeError("No valid MPII head-bone calibration samples")
    values = np.asarray(ratios, dtype=np.float64)
    report = {
        "definition": "median(official_head_scale / head_top_upper_neck_distance)",
        "source_split": "training_only",
        "samples": len(values),
        "scale_factor": float(np.median(values)),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_head_bone_scale(path: Path) -> float | None:
    if not path.is_file():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["scale_factor"])
