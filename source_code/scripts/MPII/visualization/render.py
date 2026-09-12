from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.MPII.evaluation.metrics import prediction_to_keypoints
from scripts.spikepose.visualization import draw_skeleton, load_manifest, save_grid

from .skeleton import MPII_EDGES

MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def tensor_to_bgr(value: torch.Tensor) -> np.ndarray:
    rgb = value.permute(1, 2, 0).numpy() * STD + MEAN
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
            image = tensor_to_bgr(sample["image"])
            prediction = model(sample["image"].unsqueeze(0).to(device))
            keypoints, confidence = prediction_to_keypoints(
                prediction, dataset.image_size,
            )
            keypoints, confidence = keypoints[0], confidence[0]
            predicted = draw_skeleton(image, keypoints, confidence > 0,
                                      MPII_EDGES, (40, 200, 255))
            ground_truth = draw_skeleton(
                image, sample["keypoints"].numpy(), sample["visibility"].numpy(),
                MPII_EDGES, (70, 220, 70),
            )
            panel = np.hstack((image, ground_truth, predicted))
            cv2.putText(panel, f"Input | Ground truth | Prediction ({selected['id']})",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imwrite(str(group_dir / f"{order:02d}.png"), panel)
            panels.append(panel)
        save_grid(panels, output_dir / f"grid_{group}_10.png", columns=2)
