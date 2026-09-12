"""Affine geometry used by the official HRNet MPII implementation."""

from __future__ import annotations

import cv2
import numpy as np


PIXEL_STD = 200.0


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.asarray([-direction[1], direction[0]], np.float32)


def _direction(point, rotation: float) -> np.ndarray:
    sine, cosine = np.sin(rotation), np.cos(rotation)
    return np.asarray([
        point[0] * cosine - point[1] * sine,
        point[0] * sine + point[1] * cosine,
    ], np.float32)


def affine_matrix(center, scale, rotation, output_size, inverse=False) -> np.ndarray:
    center = np.asarray(center, np.float32)
    scale = np.asarray(scale, np.float32)
    if scale.ndim == 0:
        scale = np.repeat(scale, 2)
    output_size = np.asarray(output_size, np.float32)
    source_width = scale[0] * PIXEL_STD
    radians = np.pi * float(rotation) / 180.0
    source_direction = _direction([0, -0.5 * source_width], radians)
    target_direction = np.asarray([0, -0.5 * output_size[0]], np.float32)
    source = np.zeros((3, 2), np.float32)
    target = np.zeros((3, 2), np.float32)
    source[0], source[1] = center, center + source_direction
    target[0] = 0.5 * output_size
    target[1] = target[0] + target_direction
    source[2] = _third_point(source[0], source[1])
    target[2] = _third_point(target[0], target[1])
    first, second = (target, source) if inverse else (source, target)
    return cv2.getAffineTransform(first, second)


def transform_point(point, matrix) -> np.ndarray:
    value = np.asarray([point[0], point[1], 1.0], np.float32)
    return (np.asarray(matrix, np.float32) @ value)[:2]


def transform_points(points, matrix) -> np.ndarray:
    points = np.asarray(points, np.float32)
    homogeneous = np.concatenate(
        (points[:, :2], np.ones((len(points), 1), np.float32)), axis=1,
    )
    return homogeneous @ np.asarray(matrix, np.float32).T
