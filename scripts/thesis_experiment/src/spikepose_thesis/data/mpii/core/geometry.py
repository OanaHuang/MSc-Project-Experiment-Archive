from __future__ import annotations

import cv2
import numpy as np


def person_box(center, scale, image_shape, expansion=1.25):
    """Return a clipped square MPII person box; scale is relative to 200 px."""
    h, w = image_shape[:2]
    side = max(float(scale) * 200.0 * float(expansion), 2.0)
    cx, cy = map(float, center)
    x1, y1 = cx - side / 2.0, cy - side / 2.0
    x2, y2 = cx + side / 2.0, cy + side / 2.0
    return np.asarray([max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)], np.float32)


def crop_person(image, keypoints, center, scale, image_size=224, expansion=1.25):
    box = person_box(center, scale, image.shape, expansion)
    x1, y1, x2, y2 = box
    ix1, iy1 = int(np.floor(x1)), int(np.floor(y1))
    ix2, iy2 = int(np.ceil(x2)), int(np.ceil(y2))
    crop = image[iy1:iy2 + 1, ix1:ix2 + 1]
    if crop.size == 0:
        raise ValueError("MPII person crop is empty")
    sx, sy = image_size / crop.shape[1], image_size / crop.shape[0]
    transformed = np.asarray(keypoints, np.float32).copy()
    transformed[:, 0] = (transformed[:, 0] - ix1) * sx
    transformed[:, 1] = (transformed[:, 1] - iy1) * sy
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    inverse = np.asarray([1.0 / sx, 1.0 / sy, float(ix1), float(iy1)], np.float32)
    return resized, transformed, box, inverse


def make_heatmaps(keypoints, visibility, image_size=224, heatmap_size=56, sigma=2.0):
    heatmaps = np.zeros((len(keypoints), heatmap_size, heatmap_size), np.float32)
    scale = heatmap_size / float(image_size)
    radius = int(3 * sigma)
    for j, ((x, y), visible) in enumerate(zip(keypoints, visibility)):
        if not visible or not np.isfinite((x, y)).all():
            continue
        x, y = x * scale, y * scale
        if not (0 <= x < heatmap_size and 0 <= y < heatmap_size):
            continue
        x0, y0 = int(round(x)), int(round(y))
        xa, xb = max(0, x0 - radius), min(heatmap_size, x0 + radius + 1)
        ya, yb = max(0, y0 - radius), min(heatmap_size, y0 + radius + 1)
        yy, xx = np.meshgrid(np.arange(ya, yb), np.arange(xa, xb), indexing="ij")
        heatmaps[j, ya:yb, xa:xb] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmaps


def _gaussian_blur_heatmaps(heatmaps, kernel=11):
    """Blur heatmaps while preserving each joint's original peak magnitude."""
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("DARK kernel must be an odd integer >= 3")
    blurred = np.empty_like(heatmaps, dtype=np.float32)
    for joint, heatmap in enumerate(heatmaps):
        maximum = float(np.max(heatmap))
        value = cv2.GaussianBlur(heatmap, (kernel, kernel), 0)
        blurred[joint] = value * (maximum / max(float(np.max(value)), 1e-12))
    return blurred


def _quarter_pixel_refine(heatmaps, xy):
    """Classic SimpleBaseline refinement from the local heatmap gradient."""
    _, height, width = heatmaps.shape
    refined = xy.copy()
    for joint, (x_value, y_value) in enumerate(xy):
        x, y = int(x_value), int(y_value)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            dx = heatmaps[joint, y, x + 1] - heatmaps[joint, y, x - 1]
            dy = heatmaps[joint, y + 1, x] - heatmaps[joint, y - 1, x]
            refined[joint] += np.sign((dx, dy)).astype(np.float32) * 0.25
    return refined


def _dark_refine(heatmaps, xy, kernel=11):
    """Distribution-aware sub-pixel Taylor refinement used by DARK/UDP."""
    logarithm = np.log(np.maximum(_gaussian_blur_heatmaps(heatmaps, kernel), 1e-10))
    _, height, width = logarithm.shape
    refined = xy.copy()
    for joint, (x_value, y_value) in enumerate(xy):
        x, y = int(x_value), int(y_value)
        if not (1 <= x < width - 1 and 1 <= y < height - 1):
            continue
        value = logarithm[joint]
        gradient = 0.5 * np.asarray(
            [value[y, x + 1] - value[y, x - 1],
             value[y + 1, x] - value[y - 1, x]], np.float32)
        dxx = value[y, x + 1] - 2.0 * value[y, x] + value[y, x - 1]
        dyy = value[y + 1, x] - 2.0 * value[y, x] + value[y - 1, x]
        dxy = 0.25 * (value[y + 1, x + 1] - value[y - 1, x + 1]
                      - value[y + 1, x - 1] + value[y - 1, x - 1])
        hessian = np.asarray([[dxx, dxy], [dxy, dyy]], np.float32)
        if abs(float(np.linalg.det(hessian))) > 1e-6:
            offset = -np.linalg.solve(hessian, gradient)
            if np.isfinite(offset).all() and np.max(np.abs(offset)) <= 1.5:
                refined[joint] += offset.astype(np.float32)
    return refined


def heatmaps_to_keypoints(heatmaps, image_size=224, method="argmax",
                          udp=False, dark_kernel=11):
    """Decode one ``[J,H,W]`` heatmap tensor into crop-space coordinates.

    ``method`` is ``argmax``, ``quarter``, or ``dark``. With ``udp=True``,
    heatmap coordinates are mapped using aligned endpoints, avoiding the
    biased ``image_size / heatmap_size`` scale used by conventional decoding.
    """
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape [J, H, W]")
    if method not in {"argmax", "quarter", "dark"}:
        raise ValueError("method must be 'argmax', 'quarter', or 'dark'")
    joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(joints, -1)
    indices = flat.argmax(axis=1)
    confidence = flat[np.arange(joints), indices]
    xy = np.stack((indices % width, indices // width), axis=1).astype(np.float32)
    if method == "quarter":
        xy = _quarter_pixel_refine(heatmaps, xy)
    elif method == "dark":
        xy = _dark_refine(heatmaps, xy, dark_kernel)
    if udp:
        xy[:, 0] *= float(image_size - 1) / max(width - 1, 1)
        xy[:, 1] *= float(image_size - 1) / max(height - 1, 1)
    else:
        xy[:, 0] *= float(image_size) / width
        xy[:, 1] *= float(image_size) / height
    return xy, confidence.astype(np.float32)
