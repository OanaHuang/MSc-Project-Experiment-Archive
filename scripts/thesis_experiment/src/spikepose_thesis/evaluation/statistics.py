from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .runner import _basic_summary
from .temporal import temporal_summary


BOOTSTRAP_METRICS = (
    "pck_0.5", "nme_hb", "acceleration_error",
    "relative_acceleration_error", "acceleration_magnitude_ratio",
    "high_motion_pck_0.5",
)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key].copy() for key in handle.files}


def _sequence_rows(values: dict[str, np.ndarray]) -> dict[tuple[str, str], dict]:
    result = {}
    video_groups: dict[tuple[str, str], list[int]] = {}
    for index, (sample_id, person_id) in enumerate(zip(
        values["sample_id"], values["person_id"],
    )):
        video_id = str(sample_id).split("::", 1)[0]
        video_groups.setdefault((video_id, str(person_id)), []).append(index)
    for key, members in video_groups.items():
        indices = np.asarray(members, dtype=np.int64)
        spatial = _basic_summary(
            values["prediction"][indices], values["target"][indices],
            values["visibility"][indices], values["scale"][indices],
        )
        subset = {name: item[indices] for name, item in values.items()}
        temporal = temporal_summary(subset)
        result[key] = {**spatial, **{
            name: value for name, value in temporal.items()
            if isinstance(value, (int, float))
        }}
    return result


def paired_sequence_bootstrap(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    repeats: int = 1000,
    seed: int = 42,
) -> dict:
    if repeats < 1000:
        raise ValueError("ICASSP protocol requires at least 1000 bootstrap repeats")
    left, right = _sequence_rows(baseline), _sequence_rows(candidate)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("Prediction archives have no matched sequence-person rows")
    rng = np.random.default_rng(seed)
    result = {"sequences": len(keys), "repeats": repeats, "seed": seed, "metrics": {}}
    for metric in BOOTSTRAP_METRICS:
        pairs = np.asarray([
            (left[key].get(metric, np.nan), right[key].get(metric, np.nan))
            for key in keys
        ], dtype=np.float64)
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        if not len(pairs):
            continue
        delta = pairs[:, 1] - pairs[:, 0]
        sampled = np.empty(repeats, dtype=np.float64)
        for index in range(repeats):
            sampled[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
        result["metrics"][metric] = {
            "matched_sequences": len(pairs),
            "baseline_mean": float(pairs[:, 0].mean()),
            "candidate_mean": float(pairs[:, 1].mean()),
            "delta_candidate_minus_baseline": float(delta.mean()),
            "ci95_low": float(np.percentile(sampled, 2.5)),
            "ci95_high": float(np.percentile(sampled, 97.5)),
        }
    return result


def bootstrap_archives(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    repeats: int = 1000,
    seed: int = 42,
) -> dict:
    report = paired_sequence_bootstrap(
        load_archive(baseline_path), load_archive(candidate_path), repeats, seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
