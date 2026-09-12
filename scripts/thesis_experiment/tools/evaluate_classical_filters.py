#!/usr/bin/env python3
"""Tune causal coordinate filters on validation and evaluate them on test.

The source archives must be the paired ``predictions.npz`` (one decoded row per
clip) and ``predictions_per_frame.npz`` (all 16 decoded frames).  Hyperparameters
are selected only from validation.  The selected filter is then applied without
change to test, resetting its state at every clip boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from spikepose_thesis.data.ntu.core.joint_mapping import MPII16_JOINT_NAMES
from spikepose_thesis.evaluation.runner import (
    _ntu_group_summaries,
    summarize_predictions,
)
from spikepose_thesis.evaluation.temporal import temporal_summary


CLIP_LENGTH = 16


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-base", type=Path, required=True)
    parser.add_argument("--validation-frames", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--test-frames", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _validate_pair(
    base: dict[str, np.ndarray], frames: dict[str, np.ndarray], label: str,
) -> None:
    clips = len(base["prediction"])
    if len(frames["prediction"]) != clips * CLIP_LENGTH:
        raise ValueError(f"{label}: frame archive does not contain 16 rows per clip")
    positions = frames["frame_position_in_clip"].reshape(clips, CLIP_LENGTH)
    if not np.array_equal(positions, np.broadcast_to(np.arange(CLIP_LENGTH), positions.shape)):
        raise ValueError(f"{label}: rows are not ordered as complete 16-frame clips")
    sample_ids = frames["sample_id"].reshape(clips, CLIP_LENGTH)
    person_ids = frames["person_id"].reshape(clips, CLIP_LENGTH)
    if not np.all(sample_ids == sample_ids[:, :1]):
        raise ValueError(f"{label}: sample identity changes inside a clip")
    if not np.all(person_ids == person_ids[:, :1]):
        raise ValueError(f"{label}: person identity changes inside a clip")
    if not np.array_equal(sample_ids[:, 0], base["sample_id"]):
        raise ValueError(f"{label}: base/per-frame sample order differs")
    if not np.array_equal(person_ids[:, 0], base["person_id"]):
        raise ValueError(f"{label}: base/per-frame person order differs")
    if not np.array_equal(
        frames["frame_index"].reshape(clips, CLIP_LENGTH)[:, -1],
        base["frame_index"],
    ):
        raise ValueError(f"{label}: base rows are not the final frame of each clip")


def _ema(value: np.ndarray, alpha: float) -> np.ndarray:
    result = value.copy()
    for index in range(1, value.shape[1]):
        result[:, index] = (
            float(alpha) * value[:, index]
            + (1.0 - float(alpha)) * result[:, index - 1]
        )
    return result.astype(np.float32)


def _one_euro(
    value: np.ndarray, min_cutoff: float, beta: float,
    derivative_cutoff: float = 1.0, frequency: float = 30.0,
) -> np.ndarray:
    def alpha(cutoff: np.ndarray | float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff))
        return 1.0 / (1.0 + tau * frequency)

    result = value.copy()
    derivative = np.zeros_like(value[:, 0])
    derivative_alpha = alpha(derivative_cutoff)
    for index in range(1, value.shape[1]):
        raw_derivative = (value[:, index] - value[:, index - 1]) * frequency
        derivative = (
            derivative_alpha * raw_derivative
            + (1.0 - derivative_alpha) * derivative
        )
        cutoff = min_cutoff + beta * np.abs(derivative)
        value_alpha = alpha(cutoff)
        result[:, index] = (
            value_alpha * value[:, index]
            + (1.0 - value_alpha) * result[:, index - 1]
        )
    return result.astype(np.float32)


def _causal_savgol(value: np.ndarray, window: int, order: int) -> np.ndarray:
    result = value.copy()
    for index in range(1, value.shape[1]):
        start = max(0, index - int(window) + 1)
        history = value[:, start:index + 1]
        length = history.shape[1]
        if length <= order:
            result[:, index] = history.mean(axis=1)
            continue
        x = np.arange(length, dtype=np.float64)
        design = np.vander(x, N=order + 1, increasing=True)
        evaluate = np.power(float(x[-1]), np.arange(order + 1))
        weights = evaluate @ np.linalg.pinv(design)
        result[:, index] = np.tensordot(
            history, weights.astype(np.float32), axes=(1, 0),
        )
    return result.astype(np.float32)


def _apply(value: np.ndarray, method: str, parameters: dict) -> np.ndarray:
    if method == "shared_ema":
        return _ema(value, parameters["alpha"])
    if method == "one_euro":
        return _one_euro(value, **parameters)
    if method == "causal_sg":
        return _causal_savgol(value, parameters["window"], parameters["order"])
    raise ValueError(method)


def _candidate_metrics(
    refined: np.ndarray, base: dict[str, np.ndarray], frames: dict[str, np.ndarray],
) -> dict[str, float]:
    clips = len(base["prediction"])
    target = frames["target"].reshape(clips, CLIP_LENGTH, 16, 2)
    visible = frames["visibility"].reshape(clips, CLIP_LENGTH, 16) > 0
    scale = frames["scale"].reshape(clips, CLIP_LENGTH)
    distance = np.linalg.norm(refined - target, axis=-1)
    valid_pck = visible & np.isfinite(distance) & np.isfinite(scale[:, :, None])
    correct = distance <= 0.5 * scale[:, :, None]
    pck = float(correct[valid_pck].mean())

    pred_acc = np.diff(refined, n=2, axis=1)
    target_acc = np.diff(target, n=2, axis=1)
    acc_error = np.linalg.norm(pred_acc - target_acc, axis=-1)
    acc_mask = visible[:, 2:] & visible[:, 1:-1] & visible[:, :-2]
    normalized = acc_error / scale[:, 2:, None]
    valid = acc_mask & np.isfinite(normalized)
    numerator = np.where(valid, normalized, 0.0).sum(axis=(1, 2))
    denominator = valid.sum(axis=(1, 2))
    clip_values = np.divide(
        numerator, denominator, out=np.full(clips, np.nan), where=denominator > 0,
    )
    acc_e_hb = float(np.nanmean(clip_values))
    return {
        "pck_hb_0.5": pck,
        "acceleration_error_hb": acc_e_hb,
        "selection_score": pck - 1e-3 * acc_e_hb,
    }


def _full_frame_pck(
    refined: np.ndarray, frames: dict[str, np.ndarray],
) -> dict:
    clips = refined.shape[0]
    target = frames["target"].reshape(clips, CLIP_LENGTH, 16, 2)
    visible = frames["visibility"].reshape(clips, CLIP_LENGTH, 16) > 0
    scale = frames["scale_hb"].reshape(clips, CLIP_LENGTH)
    distance = np.linalg.norm(refined - target, axis=-1)
    valid = visible & np.isfinite(distance) & np.isfinite(scale[:, :, None])
    correct = distance <= 0.5 * scale[:, :, None]
    return {
        "pck_0.5": float(correct[valid].mean()),
        "samples": int(clips * CLIP_LENGTH),
        "per_joint": {
            name: {"pck_0.5": float(correct[:, :, joint][valid[:, :, joint]].mean())}
            for joint, name in enumerate(MPII16_JOINT_NAMES)
        },
    }


def _grids() -> dict[str, list[dict]]:
    return {
        "shared_ema": [{"alpha": value} for value in np.arange(0.1, 1.0, 0.1)],
        "one_euro": [
            {"min_cutoff": minimum, "beta": beta, "derivative_cutoff": 1.0}
            for minimum in (0.1, 0.5, 1.0, 2.0, 3.0)
            for beta in (0.0, 0.1, 0.5, 1.0)
        ],
        "causal_sg": [
            {"window": window, "order": 2} for window in (5, 7, 9)
        ],
    }


def _summary(
    base: dict[str, np.ndarray], frames: dict[str, np.ndarray], refined: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_values = {key: value.copy() for key, value in frames.items()}
    frame_values["prediction"] = refined.reshape(-1, 16, 2)
    base_values = {key: value.copy() for key, value in base.items()}
    base_values["prediction"] = refined[:, -1]
    summary = summarize_predictions(base_values)
    summary["pck_hb"] = summarize_predictions({
        **base_values, "scale": base_values["scale_hb"],
    })
    summary["pck_hb_all_frames"] = _full_frame_pck(refined, frames)
    summary["head_bone_calibration_ready"] = True
    summary["temporal"] = temporal_summary(
        frame_values, strict=True, expected_length=CLIP_LENGTH,
    )
    summary["groups"] = _ntu_group_summaries(base_values)
    return summary, base_values, frame_values


def main() -> None:
    args = _parser().parse_args()
    paths = {
        "validation_base": args.validation_base.resolve(),
        "validation_frames": args.validation_frames.resolve(),
        "test_base": args.test_base.resolve(),
        "test_frames": args.test_frames.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    validation_base = _load(paths["validation_base"])
    validation_frames = _load(paths["validation_frames"])
    test_base = _load(paths["test_base"])
    test_frames = _load(paths["test_frames"])
    _validate_pair(validation_base, validation_frames, "validation")
    _validate_pair(test_base, test_frames, "test")
    validation_prediction = validation_frames["prediction"].reshape(
        -1, CLIP_LENGTH, 16, 2,
    )
    test_prediction = test_frames["prediction"].reshape(-1, CLIP_LENGTH, 16, 2)

    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "evaluation_split": "test",
        "clip_length": CLIP_LENGTH,
        "state_reset": "every clip",
        "source_paths": {key: str(value) for key, value in paths.items()},
        "source_sha256": {key: _sha256(value) for key, value in paths.items()},
        "selection_rule": "PCK_HB@0.5 - 0.001 * AccE_HB",
    }
    (output_root / "filter_evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8",
    )

    results = {}
    for method, grid in _grids().items():
        grid_results = []
        for parameters in grid:
            refined_validation = _apply(validation_prediction, method, parameters)
            metrics = _candidate_metrics(
                refined_validation, validation_base, validation_frames,
            )
            grid_results.append({"parameters": parameters, **metrics})
        best = max(grid_results, key=lambda item: item["selection_score"])
        refined_test = _apply(test_prediction, method, best["parameters"])
        summary, base_values, frame_values = _summary(
            test_base, test_frames, refined_test,
        )
        output = output_root / method
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", **base_values)
        np.savez_compressed(output / "predictions_per_frame.npz", **frame_values)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
        (output / "parameters.json").write_text(
            json.dumps(best["parameters"], indent=2), encoding="utf-8",
        )
        (output / "validation_grid.json").write_text(
            json.dumps(grid_results, indent=2), encoding="utf-8",
        )
        results[method] = {
            "parameters": best["parameters"],
            "validation": {key: best[key] for key in (
                "pck_hb_0.5", "acceleration_error_hb", "selection_score",
            )},
            "test": {
                "pck_hb_0.5": summary["pck_hb_all_frames"]["pck_0.5"],
                "hm_pck_hb_0.5": summary["temporal"]["hm_pck_hb_05"],
                "vel_e": summary["temporal"]["vel_e"],
                "acc_e": summary["temporal"]["acc_e"],
                "amr": summary["temporal"]["amr"],
            },
        }
        print(json.dumps({method: results[method]}, indent=2), flush=True)
    (output_root / "filter_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
