from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from spikepose_thesis.data.mpii.core import MPII_FLIP_PAIRS
from spikepose_thesis.data.mpii.core.geometry import heatmaps_to_keypoints
from spikepose_thesis.evaluation.mpii.dual_pckh import compute_dual_pckh, save_dual_pckh_reports


@dataclass(frozen=True)
class HeatmapVariant:
    id: str
    decoder: str
    udp: bool = False
    dark_kernel: int = 11
    flip_test: bool = False
    flip_shift: bool = False

    @classmethod
    def from_dict(cls, value: dict) -> "HeatmapVariant":
        variant = cls(**value)
        if variant.decoder not in {"argmax", "quarter", "dark"}:
            raise ValueError(f"{variant.id}: unsupported decoder {variant.decoder}")
        if variant.dark_kernel < 3 or variant.dark_kernel % 2 == 0:
            raise ValueError(f"{variant.id}: dark_kernel must be odd and >= 3")
        if variant.flip_shift and not variant.flip_test:
            raise ValueError(f"{variant.id}: flip_shift requires flip_test")
        return variant


def flip_back_heatmaps(heatmaps: np.ndarray, shift: bool) -> np.ndarray:
    restored = heatmaps[..., ::-1].copy()
    for left, right in MPII_FLIP_PAIRS:
        restored[:, [left, right]] = restored[:, [right, left]].copy()
    if shift:
        shifted = restored.copy()
        shifted[..., 1:] = restored[..., :-1]
        restored = shifted
    return restored


def _decode(heatmaps: np.ndarray, image_size: int,
            variant: HeatmapVariant) -> tuple[np.ndarray, np.ndarray]:
    predictions, confidence = [], []
    for heatmap in heatmaps:
        xy, score = heatmaps_to_keypoints(
            np.asarray(heatmap, np.float32), image_size,
            method=variant.decoder, udp=variant.udp,
            dark_kernel=variant.dark_kernel,
        )
        predictions.append(xy)
        confidence.append(score)
    return np.asarray(predictions), np.asarray(confidence)


def evaluate_cached_variant(cache: dict, image_size: int,
                            variant: HeatmapVariant, output_dir: Path,
                            model_name: str, experiment_id: str,
                            chunk_size: int = 128) -> dict:
    original = cache["original"]
    flipped = cache.get("flipped_input")
    if variant.flip_test and flipped is None:
        raise ValueError(f"{variant.id} requires flipped-input heatmaps")
    prediction_chunks, confidence_chunks = [], []
    for start in range(0, len(original), chunk_size):
        stop = min(start + chunk_size, len(original))
        # Work in float32 so float16 is only a storage format. Chunking avoids
        # materialising both complete validation heatmap tensors in RAM.
        heatmaps = np.asarray(original[start:stop], np.float32)
        if variant.flip_test:
            heatmaps = 0.5 * (
                heatmaps + flip_back_heatmaps(
                    np.asarray(flipped[start:stop], np.float32),
                    variant.flip_shift,
                )
            )
        pred_chunk, confidence_chunk = _decode(heatmaps, image_size, variant)
        prediction_chunks.append(pred_chunk)
        confidence_chunks.append(confidence_chunk)
    pred = np.concatenate(prediction_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0)
    targets = cache["targets"]
    inverse = targets["inverse"]
    pred[..., 0] = pred[..., 0] * inverse[:, None, 0] + inverse[:, None, 2]
    pred[..., 1] = pred[..., 1] * inverse[:, None, 1] + inverse[:, None, 3]
    gt = targets["keypoints_original"]
    visibility = targets["visibility"]
    pckhn_scale = 0.75 * np.linalg.norm(gt[:, 9] - gt[:, 8], axis=1)
    valid_head_pair = (visibility[:, 8] > 0) & (visibility[:, 9] > 0)
    pckhn_scale[~valid_head_pair] = np.nan
    matrices = compute_dual_pckh(
        pred, gt, visibility, targets["head_length"], pckhn_scale, 0.5,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = save_dual_pckh_reports(
        output_dir, matrices, model_name, len(pred),
    )
    summary.update({
        "experiment_id": experiment_id,
        "variant_id": variant.id,
        **asdict(variant),
        "heatmap_source": "cache",
    })
    serialized = json.dumps(summary, indent=2)
    (output_dir / "best.json").write_text(serialized, encoding="utf-8")
    (output_dir / "dual_pckh_summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(variant), indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction_matrices.npz", pred=pred, confidence=confidence,
        gt=gt, visibility=visibility, head_length=targets["head_length"],
        pckhn_head_scale=pckhn_scale, **matrices,
    )
    return summary
