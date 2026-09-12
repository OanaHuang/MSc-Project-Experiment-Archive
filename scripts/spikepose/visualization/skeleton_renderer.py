from __future__ import annotations

import cv2
import numpy as np


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray,
                  visibility: np.ndarray, edges: tuple[tuple[int, int], ...],
                  color: tuple[int, int, int] = (40, 200, 255)) -> np.ndarray:
    canvas = image.copy()
    for start, end in edges:
        if visibility[start] > 0 and visibility[end] > 0:
            a = tuple(np.rint(keypoints[start]).astype(int))
            b = tuple(np.rint(keypoints[end]).astype(int))
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
    for index, point in enumerate(keypoints):
        if visibility[index] > 0:
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3,
                       (255, 90, 40), -1, cv2.LINE_AA)
    return canvas
