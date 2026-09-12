from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.MPII.core.geometry import heatmaps_to_keypoints
from scripts.spikepose.visualization import draw_skeleton, load_manifest, save_grid

from .skeleton import NTU_EDGES


def tensor_to_bgr(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().permute(1, 2, 0).numpy()
    if array.min() < 0 or array.max() > 1:
        array = array - array.min()
        array = array / max(float(array.max()), 1e-6)
    return cv2.cvtColor(np.clip(array * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


@torch.no_grad()
def generate_visualizations(model, dataset, device, manifest_path: Path,
                            output_dir: Path) -> None:
    manifest = load_manifest(manifest_path)
    model.eval()
    for group in ("first", "random"):
        panels = []
        group_dir = output_dir / f"{group}_10"
        group_dir.mkdir(parents=True, exist_ok=True)
        for order, selected in enumerate(manifest[group], 1):
            sample = dataset[int(selected["index"])]
            display_image = sample["image"][-1] if sample["image"].ndim == 4 else sample["image"]
            image = tensor_to_bgr(display_image)
            heatmap = model(sample["image"].unsqueeze(0).to(device))[0].cpu().numpy()
            prediction, confidence = heatmaps_to_keypoints(heatmap, dataset.image_size)
            predicted = draw_skeleton(image, prediction, confidence > 0, NTU_EDGES)
            ground_truth = draw_skeleton(
                image, sample["keypoints"].numpy(), sample["visibility"].numpy(),
                NTU_EDGES, (70, 220, 70),
            )
            panel = np.hstack((image, ground_truth, predicted))
            cv2.imwrite(str(group_dir / f"{order:02d}.png"), panel)
            panels.append(panel)
        save_grid(panels, output_dir / f"grid_{group}_10.png", columns=2)
