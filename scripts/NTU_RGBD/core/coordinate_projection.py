# scripts/NTU_RGBD/core/coordinate_projection.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import (
    TRACKING_STATE_NOT_TRACKED,
)


# ============================================================
# 1. Coordinate validation
# ============================================================

def validate_keypoints_2d(
    keypoints: np.ndarray,
) -> np.ndarray:
    """
    Validate and return 2D keypoints as float32.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.ndim < 2:
        raise ValueError(
            "keypoints must have at least two dimensions"
        )

    if keypoints.shape[-1] != 2:
        raise ValueError(
            "Final keypoint dimension must equal 2"
        )

    return keypoints


def validate_keypoints_3d(
    keypoints: np.ndarray,
) -> np.ndarray:
    """
    Validate and return 3D keypoints as float32.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.ndim < 2:
        raise ValueError(
            "keypoints must have at least two dimensions"
        )

    if keypoints.shape[-1] != 3:
        raise ValueError(
            "Final keypoint dimension must equal 3"
        )

    return keypoints


# ============================================================
# 2. Visibility
# ============================================================

def tracking_state_to_visibility(
    tracking_state: np.ndarray,
    include_inferred: bool = True,
) -> np.ndarray:
    """
    Convert NTU tracking state to a boolean visibility mask.

    NTU tracking state:
        0 = not tracked
        1 = inferred
        2 = tracked
    """
    tracking_state = np.asarray(
        tracking_state
    )

    if include_inferred:
        return (
            tracking_state
            > TRACKING_STATE_NOT_TRACKED
        )

    return tracking_state >= 2


def coordinate_visibility(
    keypoints: np.ndarray,
    tracking_state: Optional[
        np.ndarray
    ] = None,
    image_size: Optional[
        Tuple[int, int]
    ] = None,
    include_inferred: bool = True,
) -> np.ndarray:
    """
    Create a visibility mask based on finite coordinates,
    tracking state, and optional image boundaries.
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    visible = np.isfinite(
        keypoints
    ).all(axis=-1)

    if tracking_state is not None:
        visible &= (
            tracking_state_to_visibility(
                tracking_state,
                include_inferred=(
                    include_inferred
                ),
            )
        )

    if image_size is not None:
        width, height = image_size

        x = keypoints[..., 0]
        y = keypoints[..., 1]

        visible &= (
            (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )

    return visible


# ============================================================
# 3. Normalize and denormalize
# ============================================================

def normalize_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert pixel coordinates into normalized [0, 1] values.

    image_size uses:
        (width, height)
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    width, height = image_size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size: {image_size}"
        )

    normalized = keypoints.copy()

    normalized[..., 0] /= width
    normalized[..., 1] /= height

    return normalized


def denormalize_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert normalized coordinates into image pixels.
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    width, height = image_size

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image size: {image_size}"
        )

    denormalized = keypoints.copy()

    denormalized[..., 0] *= width
    denormalized[..., 1] *= height

    return denormalized


# ============================================================
# 4. Coordinate scaling
# ============================================================

def scale_keypoints(
    keypoints: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale keypoints between two image resolutions.
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    source_width, source_height = (
        source_size
    )

    target_width, target_height = (
        target_size
    )

    if (
        source_width <= 0
        or source_height <= 0
    ):
        raise ValueError(
            f"Invalid source size: {source_size}"
        )

    if (
        target_width <= 0
        or target_height <= 0
    ):
        raise ValueError(
            f"Invalid target size: {target_size}"
        )

    scaled = keypoints.copy()

    scaled[..., 0] *= (
        target_width / source_width
    )

    scaled[..., 1] *= (
        target_height / source_height
    )

    return scaled


# ============================================================
# 5. Clip coordinates
# ============================================================

def clip_keypoints_to_image(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Clip keypoints to valid image coordinates.
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    width, height = image_size

    clipped = keypoints.copy()

    clipped[..., 0] = np.clip(
        clipped[..., 0],
        0,
        max(width - 1, 0),
    )

    clipped[..., 1] = np.clip(
        clipped[..., 1],
        0,
        max(height - 1, 0),
    )

    return clipped


# ============================================================
# 6. Keypoint bounding box
# ============================================================

def keypoints_to_bbox(
    keypoints: np.ndarray,
    visibility: Optional[
        np.ndarray
    ] = None,
    padding_ratio: float = 0.1,
    image_size: Optional[
        Tuple[int, int]
    ] = None,
) -> Optional[
    Tuple[int, int, int, int]
]:
    """
    Compute a bounding box from visible keypoints.

    Returns:
        x_min, y_min, x_max, y_max
    """
    keypoints = validate_keypoints_2d(
        keypoints
    )

    if padding_ratio < 0:
        raise ValueError(
            "padding_ratio must be non-negative"
        )

    valid = np.isfinite(
        keypoints
    ).all(axis=-1)

    if visibility is not None:
        visibility = np.asarray(
            visibility
        ).astype(bool)

        if visibility.shape != valid.shape:
            raise ValueError(
                "visibility shape must match "
                "keypoint joint dimensions"
            )

        valid &= visibility

    selected = keypoints[valid]

    if selected.size == 0:
        return None

    x_min = float(
        selected[:, 0].min()
    )

    y_min = float(
        selected[:, 1].min()
    )

    x_max = float(
        selected[:, 0].max()
    )

    y_max = float(
        selected[:, 1].max()
    )

    box_width = max(
        x_max - x_min,
        1.0,
    )

    box_height = max(
        y_max - y_min,
        1.0,
    )

    x_padding = (
        box_width * padding_ratio
    )

    y_padding = (
        box_height * padding_ratio
    )

    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    if image_size is not None:
        width, height = image_size

        x_min = np.clip(
            x_min,
            0,
            width - 1,
        )

        x_max = np.clip(
            x_max,
            0,
            width - 1,
        )

        y_min = np.clip(
            y_min,
            0,
            height - 1,
        )

        y_max = np.clip(
            y_max,
            0,
            height - 1,
        )

    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max)),
        int(round(y_max)),
    )


# ============================================================
# 7. Root-relative 3D coordinates
# ============================================================

def root_center_3d_pose(
    keypoints_3d: np.ndarray,
    root_joint_index: int = 0,
) -> np.ndarray:
    """
    Subtract the root joint from all 3D joints.

    NTU joint 0 is spine base.
    """
    keypoints_3d = (
        validate_keypoints_3d(
            keypoints_3d
        )
    )

    num_joints = (
        keypoints_3d.shape[-2]
    )

    if not (
        0 <= root_joint_index
        < num_joints
    ):
        raise IndexError(
            f"Invalid root_joint_index: "
            f"{root_joint_index}"
        )

    root = keypoints_3d[
        ...,
        root_joint_index:
        root_joint_index + 1,
        :,
    ]

    return keypoints_3d - root
