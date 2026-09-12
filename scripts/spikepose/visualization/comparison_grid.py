from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_grid(images: list[np.ndarray], path: Path, columns: int = 5) -> None:
    if not images:
        return
    height, width = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    blank = np.full_like(images[0], 255)
    padded = images + [blank] * (rows * columns - len(images))
    grid = np.vstack([
        np.hstack(padded[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)
