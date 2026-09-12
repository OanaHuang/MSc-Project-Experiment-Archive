#!/usr/bin/env python3
"""Rank paired NTU videos by temporal improvement without rendering media."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=16)
    return parser


def _clip_arrays(path: Path, clip_length: int) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    rows = len(archive["prediction"])
    if rows % clip_length:
        raise ValueError(f"Rows are not divisible by clip length: {rows}")
    clips = rows // clip_length
    result = {
        "prediction": archive["prediction"].reshape(clips, clip_length, 16, 2),
        "target": archive["target"].reshape(clips, clip_length, 16, 2),
        "visibility": archive["visibility"].reshape(clips, clip_length, 16) > 0,
        "scale": archive["scale"].reshape(clips, clip_length),
        "video_id": archive["video_id"].reshape(clips, clip_length)[:, 0],
        "clip_id": archive["clip_id"].reshape(clips, clip_length)[:, 0],
        "frame_index": archive["frame_index"].reshape(clips, clip_length),
        "frame_position": archive["frame_position_in_clip"].reshape(
            clips, clip_length,
        ),
    }
    if not np.all(
        archive["video_id"].reshape(clips, clip_length)
        == result["video_id"][:, None]
    ):
        raise ValueError("A clip contains multiple video ids")
    if not np.all(
        archive["clip_id"].reshape(clips, clip_length)
        == result["clip_id"][:, None]
    ):
        raise ValueError("A clip contains multiple clip ids")
    return result


def _aggregate(values: np.ndarray, inverse: np.ndarray, videos: int) -> np.ndarray:
    finite = np.isfinite(values)
    total = np.bincount(inverse[finite], weights=values[finite], minlength=videos)
    count = np.bincount(inverse[finite], minlength=videos)
    return np.divide(
        total, count, out=np.full(videos, np.nan), where=count > 0,
    )


def _metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prediction = arrays["prediction"].astype(np.float64)
    target = arrays["target"].astype(np.float64)
    visible = arrays["visibility"]
    scale = arrays["scale"].astype(np.float64)
    video_ids, inverse = np.unique(arrays["video_id"], return_inverse=True)
    video_count = len(video_ids)

    distance = np.linalg.norm(prediction - target, axis=-1)
    pck_valid = visible & np.isfinite(scale[:, :, None]) & (scale[:, :, None] > 0)
    pck_correct = (distance <= 0.5 * scale[:, :, None]) & pck_valid
    clip_pck_num = pck_correct.sum(axis=(1, 2)).astype(np.float64)
    clip_pck_den = pck_valid.sum(axis=(1, 2)).astype(np.float64)
    video_pck_num = np.bincount(inverse, weights=clip_pck_num, minlength=video_count)
    video_pck_den = np.bincount(inverse, weights=clip_pck_den, minlength=video_count)
    pck = np.divide(
        video_pck_num, video_pck_den,
        out=np.full(video_count, np.nan), where=video_pck_den > 0,
    )

    pred_velocity = np.diff(prediction, axis=1)
    target_velocity = np.diff(target, axis=1)
    velocity_valid = visible[:, 1:] & visible[:, :-1]
    velocity_error = np.linalg.norm(pred_velocity - target_velocity, axis=-1)
    clip_vele = np.divide(
        (velocity_error * velocity_valid).sum(axis=(1, 2)),
        velocity_valid.sum(axis=(1, 2)),
        out=np.full(len(prediction), np.nan),
        where=velocity_valid.sum(axis=(1, 2)) > 0,
    )

    pred_acceleration = (
        prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    )
    target_acceleration = (
        target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    )
    acceleration_valid = visible[:, 2:] & visible[:, 1:-1] & visible[:, :-2]
    acceleration_error = np.linalg.norm(
        pred_acceleration - target_acceleration, axis=-1,
    )
    valid_count = acceleration_valid.sum(axis=(1, 2))
    clip_acce = np.divide(
        (acceleration_error * acceleration_valid).sum(axis=(1, 2)),
        valid_count,
        out=np.full(len(prediction), np.nan), where=valid_count > 0,
    )
    pred_acceleration_sum = (
        np.linalg.norm(pred_acceleration, axis=-1) * acceleration_valid
    ).sum(axis=(1, 2))
    target_acceleration_sum = (
        np.linalg.norm(target_acceleration, axis=-1) * acceleration_valid
    ).sum(axis=(1, 2))
    clip_amr = np.divide(
        pred_acceleration_sum, target_acceleration_sum,
        out=np.full(len(prediction), np.nan), where=target_acceleration_sum > 0,
    )

    first_frame = np.full(video_count, np.iinfo(np.int64).max, dtype=np.int64)
    last_frame = np.full(video_count, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(first_frame, inverse, arrays["frame_index"].min(axis=1))
    np.maximum.at(last_frame, inverse, arrays["frame_index"].max(axis=1))
    clips_per_video = np.bincount(inverse, minlength=video_count)
    return {
        "video_id": video_ids,
        "clips": clips_per_video,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "pck": pck,
        "vele": _aggregate(clip_vele, inverse, video_count),
        "acce": _aggregate(clip_acce, inverse, video_count),
        "amr": _aggregate(clip_amr, inverse, video_count),
        "valid_acceleration_triplets": np.bincount(
            inverse, weights=valid_count, minlength=video_count,
        ).astype(np.int64),
    }


def main() -> None:
    args = _parser().parse_args()
    baseline_arrays = _clip_arrays(args.baseline, args.clip_length)
    candidate_arrays = _clip_arrays(args.candidate, args.clip_length)
    for key in ("target", "visibility", "scale", "video_id", "clip_id",
                "frame_index", "frame_position"):
        left, right = baseline_arrays[key], candidate_arrays[key]
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if np.issubdtype(left.dtype, np.inexact)
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"Baseline/candidate identity mismatch: {key}")

    baseline = _metrics(baseline_arrays)
    candidate = _metrics(candidate_arrays)
    if not np.array_equal(baseline["video_id"], candidate["video_id"]):
        raise ValueError("Video ordering mismatch")

    rows = []
    for index, video_id in enumerate(baseline["video_id"]):
        base_acce = float(baseline["acce"][index])
        mam_acce = float(candidate["acce"][index])
        base_vele = float(baseline["vele"][index])
        mam_vele = float(candidate["vele"][index])
        base_amr = float(baseline["amr"][index])
        mam_amr = float(candidate["amr"][index])
        row = {
            "video_id": str(video_id),
            "clips": int(baseline["clips"][index]),
            "first_frame": int(baseline["first_frame"][index]),
            "last_frame": int(baseline["last_frame"][index]),
            "valid_acceleration_triplets": int(
                baseline["valid_acceleration_triplets"][index]
            ),
            "baseline_pck": float(baseline["pck"][index]),
            "candidate_pck": float(candidate["pck"][index]),
            "pck_delta_points": 100.0 * (
                float(candidate["pck"][index]) - float(baseline["pck"][index])
            ),
            "baseline_vele": base_vele,
            "candidate_vele": mam_vele,
            "vele_reduction": base_vele - mam_vele,
            "vele_reduction_pct": 100.0 * (base_vele - mam_vele) / base_vele,
            "baseline_acce": base_acce,
            "candidate_acce": mam_acce,
            "acce_reduction": base_acce - mam_acce,
            "acce_reduction_pct": 100.0 * (base_acce - mam_acce) / base_acce,
            "baseline_amr": base_amr,
            "candidate_amr": mam_amr,
            "amr_distance_to_one_gain": abs(base_amr - 1.0) - abs(mam_amr - 1.0),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["pck_delta_points"] >= -0.5,
            row["amr_distance_to_one_gain"] > 0,
            row["acce_reduction"],
            row["acce_reduction_pct"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    print(json.dumps(rows[:20], indent=2))


if __name__ == "__main__":
    main()
