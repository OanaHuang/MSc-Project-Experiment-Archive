from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.evaluate_original_videos import original_video_config
from scripts.NTU_RGBD.evaluation.metrics import compute_pckhn
from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce, save_nacce


def sequence_groups(sample_ids: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    groups = []
    for sample_id in dict.fromkeys(sample_ids.astype(str).tolist()):
        indices = np.flatnonzero(sample_ids == sample_id)
        indices = indices[np.argsort(frame_indices[indices], kind="stable")]
        if len(indices) and np.any(np.diff(frame_indices[indices]) != 1):
            raise ValueError(f"non-contiguous frames in {sample_id}")
        groups.append(indices)
    return tuple(groups)


def savgol_coefficients(window: int, polynomial: int) -> np.ndarray:
    if window < 3 or window % 2 == 0 or polynomial < 0 or polynomial >= window:
        raise ValueError("Savitzky-Golay requires odd window >= 3 and polynomial < window")
    half = window // 2
    design = np.vander(np.arange(-half, half + 1, dtype=np.float64), polynomial + 1,
                       increasing=True)
    return np.linalg.pinv(design)[0].astype(np.float32)


def savgol_filter(values: np.ndarray, groups: tuple[np.ndarray, ...],
                  window: int = 7, polynomial: int = 2) -> np.ndarray:
    output = values.copy()
    coefficients = savgol_coefficients(window, polynomial)
    half = window // 2
    for group in groups:
        if len(group) < window:
            continue
        padded = np.pad(values[group], ((half, half), (0, 0), (0, 0)), mode="edge")
        output[group] = sum(
            coefficients[offset] * padded[offset:offset + len(group)]
            for offset in range(window)
        )
    return output


def smoothing_factor(cutoff: np.ndarray | float, frequency: float) -> np.ndarray | float:
    return 1.0 / (1.0 + frequency / (2.0 * np.pi * cutoff))


def one_euro_filter(values: np.ndarray, groups: tuple[np.ndarray, ...], fps: float = 30.0,
                    min_cutoff: float = 1.0, beta: float = 0.007,
                    derivative_cutoff: float = 1.0) -> np.ndarray:
    if fps <= 0 or min_cutoff <= 0 or derivative_cutoff <= 0 or beta < 0:
        raise ValueError("invalid One Euro parameters")
    output = values.copy()
    derivative_alpha = float(smoothing_factor(derivative_cutoff, fps))
    for group in groups:
        if not len(group):
            continue
        filtered = values[group[0]].copy()
        filtered_derivative = np.zeros_like(filtered)
        output[group[0]] = filtered
        previous = values[group[0]]
        for row in group[1:]:
            current = values[row]
            derivative = (current - previous) * fps
            filtered_derivative += derivative_alpha * (derivative - filtered_derivative)
            cutoff = min_cutoff + beta * np.abs(filtered_derivative)
            alpha = smoothing_factor(cutoff, fps)
            filtered += alpha * (current - filtered)
            output[row] = filtered
            previous = current
    return output


def metrics(prediction: np.ndarray, arrays: dict[str, np.ndarray], sample_ids: np.ndarray,
            frame_indices: np.ndarray) -> dict:
    pck = compute_pckhn(prediction, arrays["gt"], arrays["visibility"],
                        arrays["head_length"], 0.5)
    temporal = compute_nacce(
        prediction, arrays["gt"], arrays["visibility"], arrays["head_length"],
        sample_ids, frame_indices, NTU_JOINT_NAMES,
    )
    per_joint = {}
    for index, name in enumerate(NTU_JOINT_NAMES):
        valid = int(pck["valid_mask"][:, index].sum())
        correct = int(pck["correct_matrix"][:, index].sum())
        per_joint[name] = {
            "valid_frames": valid, "correct_frames": correct,
            "pckhn": float(correct / valid) if valid else None,
            **temporal["per_joint"][name],
        }
    return {"pckhn": float(pck["correct_matrix"].sum() / max(int(pck["valid_mask"].sum()), 1)),
            "valid_joints": int(pck["valid_mask"].sum()), **temporal,
            "per_joint": per_joint}


def save_per_joint(result: dict, metrics_dir: Path) -> None:
    per_joint = result["per_joint"]
    (metrics_dir / "per_joint_summary.json").write_text(
        json.dumps(per_joint, indent=2), encoding="utf-8",
    )
    fields = ("joint", "valid_frames", "correct_frames", "pckhn", "valid_triplets",
              "nacce", "accel", "gt_accel", "naccel", "gt_naccel",
              "naccel_to_gt_ratio")
    with (metrics_dir / "per_joint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for joint, values in per_joint.items():
            writer.writerow({key: joint if key == "joint" else values[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training-free NTU SN series from T0 coordinates")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "Outputs_New/ntu_rgbd/sn_series/training_free_v1",
    )
    parser.add_argument("--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full")
    parser.add_argument("--validation-metadata", default="Datasets/NTU_RGBD/metadata/s010/val_split.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    checkpoint = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    config = original_video_config(checkpoint["config"], args.extracted_frames_dir,
                                   args.validation_metadata)
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    loaded = np.load(run / "metrics/prediction_matrices.npz")
    arrays = {key: loaded[key] for key in ("pred", "gt", "visibility", "head_length")}
    if len(arrays["pred"]) != len(dataset.frame_index):
        raise ValueError("stored predictions do not match the complete validation dataset")
    sample_ids = np.asarray([str(dataset.samples[sample]["sample_id"])
                             for sample, _ in dataset.frame_index])
    frame_indices = np.asarray([frame for _, frame in dataset.frame_index], dtype=np.int64)
    groups = sequence_groups(sample_ids, frame_indices)
    experiments = {
        "sn0_raw": {
            "id": "sn0", "name": "SN0-T0-Raw", "method": "identity",
            "prediction": arrays["pred"],
        },
        "sn1_savgol_w7_p2": {
            "id": "sn1", "name": "SN1-T0-SavitzkyGolay-W7-P2",
            "method": "savitzky_golay", "window": 7, "polynomial": 2,
            "prediction": savgol_filter(arrays["pred"], groups, 7, 2),
        },
        "sn2_one_euro": {
            "id": "sn2", "name": "SN2-T0-OneEuro",
            "method": "one_euro", "fps": args.fps, "min_cutoff": 1.0,
            "beta": 0.007, "derivative_cutoff": 1.0,
            "prediction": one_euro_filter(arrays["pred"], groups, args.fps, 1.0, 0.007, 1.0),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    comparison = {}
    for directory, experiment in experiments.items():
        prediction = experiment.pop("prediction")
        result = metrics(prediction, arrays, sample_ids, frame_indices)
        comparison[experiment["id"]] = {"config": experiment, "metrics": result}
        run_dir = output_root / directory / "seed_42"
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        config = {**experiment, "series": "SN", "seed": 42,
                  "source_experiment": "t0", "source_run": str(run)}
        (run_dir / "resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        save_nacce(result, metrics_dir)
        save_per_joint(result, metrics_dir)
        np.savez_compressed(metrics_dir / "prediction_matrices.npz", pred=prediction,
                            gt=arrays["gt"], visibility=arrays["visibility"],
                            head_length=arrays["head_length"], sample_ids=sample_ids,
                            frame_indices=frame_indices)
        (run_dir / "status.json").write_text(json.dumps({
            "status": "completed", "experiment_id": experiment["id"],
            "pckhn": result["pckhn"], "nacce": result["nacce"],
            "accel": result["accel"], "naccel": result["naccel"],
        }, indent=2), encoding="utf-8")
    (output_root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps({key: value["metrics"] for key, value in comparison.items()}, indent=2))


if __name__ == "__main__":
    main()
