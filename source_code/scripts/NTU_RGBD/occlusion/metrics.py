from __future__ import annotations

import numpy as np


def evaluate_occlusion(prediction: np.ndarray, target: np.ndarray,
                       visibility: np.ndarray, observed: np.ndarray,
                       head_length: np.ndarray, threshold: float = 0.5) -> dict:
    distance = np.linalg.norm(prediction - target, axis=-1)
    scale = np.maximum(head_length[..., None], 1e-6)
    normalized = distance / scale
    valid = (visibility > 0) & np.isfinite(scale)
    occluded = valid & (observed <= 0)
    visible = valid & (observed > 0)

    def summary(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        return {
            "count": count,
            "pckhn": float((normalized[mask] <= threshold).mean()) if count else None,
            "nme": float(normalized[mask].mean()) if count else None,
        }
    return {"visible": summary(visible), "occluded": summary(occluded), "overall": summary(valid)}
