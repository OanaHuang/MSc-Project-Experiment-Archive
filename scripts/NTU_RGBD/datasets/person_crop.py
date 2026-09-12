# scripts/NTU_RGBD/datasets/person_crop.py

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PersonCropResult:
    image: np.ndarray
    keypoints: np.ndarray
    visibility: np.ndarray
    bbox_xyxy: np.ndarray
    scale: float
    offset_xy: np.ndarray


def compute_person_bbox(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_width: int,
    image_height: int,
    expansion: float = 0.25,
    make_square: bool = True,
    minimum_size: float = 32.0,
) -> np.ndarray:
    """
    Compute a person bounding box from visible skeleton joints.

    Returns:
        [x1, y1, x2, y2] in original-image coordinates.
        x2 and y2 are exclusive crop boundaries.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=bool,
    )

    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(
            "keypoints must have shape [J, 2]"
        )

    if visibility.shape != (keypoints.shape[0],):
        raise ValueError(
            "visibility must have shape [J]"
        )

    valid = (
        visibility
        & np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] < image_width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] < image_height)
    )

    valid_points = keypoints[valid]

    if valid_points.shape[0] < 2:
        return np.array(
            [
                0.0,
                0.0,
                float(image_width),
                float(image_height),
            ],
            dtype=np.float32,
        )

    x_min = float(valid_points[:, 0].min())
    y_min = float(valid_points[:, 1].min())
    x_max = float(valid_points[:, 0].max())
    y_max = float(valid_points[:, 1].max())

    width = max(
        x_max - x_min,
        minimum_size,
    )

    height = max(
        y_max - y_min,
        minimum_size,
    )

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    width *= 1.0 + 2.0 * expansion
    height *= 1.0 + 2.0 * expansion

    if make_square:
        side = max(width, height)
        width = side
        height = side

    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_width), x2)
    y2 = min(float(image_height), y2)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.array(
            [
                0.0,
                0.0,
                float(image_width),
                float(image_height),
            ],
            dtype=np.float32,
        )

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def crop_and_resize_person(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    output_size: int = 224,
    expansion: float = 0.25,
    make_square: bool = True,
) -> PersonCropResult:
    """
    Crop the person using a skeleton-derived bounding box,
    resize the crop, and transform keypoints accordingly.
    """
    if image.ndim != 3:
        raise ValueError(
            "image must have shape [H, W, C]"
        )

    if output_size <= 0:
        raise ValueError(
            "output_size must be positive"
        )

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    ).copy()

    visibility = np.asarray(
        visibility,
        dtype=np.float32,
    ).copy()

    image_height, image_width = image.shape[:2]

    bbox = compute_person_bbox(
        keypoints=keypoints,
        visibility=visibility,
        image_width=image_width,
        image_height=image_height,
        expansion=expansion,
        make_square=make_square,
    )

    return crop_and_resize_person_with_bbox(
        image=image, keypoints=keypoints, visibility=visibility,
        bbox_xyxy=bbox, output_size=output_size,
    )


def crop_and_resize_person_with_bbox(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: int,
) -> PersonCropResult:
    """Crop one frame with a caller-provided box shared by a temporal clip."""
    if image.ndim != 3:
        raise ValueError("image must have shape [H, W, C]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    visibility = np.asarray(visibility, dtype=np.float32).copy()
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("bbox_xyxy must contain four finite values")
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    x1_int = int(np.floor(x1))
    y1_int = int(np.floor(y1))
    x2_int = int(np.ceil(x2))
    y2_int = int(np.ceil(y2))

    x1_int = int(
        np.clip(x1_int, 0, image_width - 1)
    )

    y1_int = int(
        np.clip(y1_int, 0, image_height - 1)
    )

    x2_int = int(
        np.clip(x2_int, x1_int + 1, image_width)
    )

    y2_int = int(
        np.clip(y2_int, y1_int + 1, image_height)
    )

    cropped_image = image[
        y1_int:y2_int,
        x1_int:x2_int,
    ]

    crop_height, crop_width = (
        cropped_image.shape[:2]
    )

    resized_image = cv2.resize(
        cropped_image,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = output_size / crop_width
    scale_y = output_size / crop_height

    transformed_keypoints = keypoints.copy()

    transformed_keypoints[:, 0] = (
        keypoints[:, 0] - x1_int
    ) * scale_x

    transformed_keypoints[:, 1] = (
        keypoints[:, 1] - y1_int
    ) * scale_y

    inside_crop = (
        np.isfinite(
            transformed_keypoints
        ).all(axis=1)
        & (transformed_keypoints[:, 0] >= 0)
        & (transformed_keypoints[:, 0] < output_size)
        & (transformed_keypoints[:, 1] >= 0)
        & (transformed_keypoints[:, 1] < output_size)
    )

    transformed_visibility = (
        visibility.astype(bool)
        & inside_crop
    ).astype(np.float32)

    return PersonCropResult(
        image=resized_image,
        keypoints=transformed_keypoints,
        visibility=transformed_visibility,
        bbox_xyxy=np.array(
            [
                x1_int,
                y1_int,
                x2_int,
                y2_int,
            ],
            dtype=np.float32,
        ),
        scale=float(
            min(scale_x, scale_y)
        ),
        offset_xy=np.array(
            [x1_int, y1_int],
            dtype=np.float32,
        ),
    )
