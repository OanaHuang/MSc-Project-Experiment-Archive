# scripts/NTU_RGBD/core/person_selector.py

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .skeleton_reader import (
    NTUBody,
    NTUFrame,
    NTUSkeletonSequence,
)


# ============================================================
# 1. Body-count checks
# ============================================================

def get_frame_body_counts(
    sequence: NTUSkeletonSequence,
) -> np.ndarray:
    """
    Return body count for every frame.
    """
    return np.asarray(
        [
            len(frame.bodies)
            for frame in sequence.frames
        ],
        dtype=np.int32,
    )


def is_single_person_sequence(
    sequence: NTUSkeletonSequence,
    allow_empty_frames: bool = True,
) -> bool:
    """
    Determine whether a sequence contains at most one body
    in every frame.
    """
    counts = get_frame_body_counts(
        sequence
    )

    if not allow_empty_frames:
        if np.any(counts == 0):
            return False

    return bool(
        np.all(counts <= 1)
    )


def max_body_count(
    sequence: NTUSkeletonSequence,
) -> int:
    """
    Maximum number of bodies visible in any frame.
    """
    counts = get_frame_body_counts(
        sequence
    )

    if len(counts) == 0:
        return 0

    return int(
        counts.max()
    )


# ============================================================
# 2. Body occurrence statistics
# ============================================================

def count_body_occurrences(
    sequence: NTUSkeletonSequence,
) -> dict[str, int]:
    """
    Count how many frames each body ID appears in.
    """
    counter: Counter[str] = Counter()

    for frame in sequence.frames:
        for body in frame.bodies:
            counter[body.body_id] += 1

    return dict(counter)


def select_longest_visible_body_id(
    sequence: NTUSkeletonSequence,
) -> Optional[str]:
    """
    Select the body ID that appears in the most frames.
    """
    occurrences = (
        count_body_occurrences(
            sequence
        )
    )

    if not occurrences:
        return None

    return max(
        occurrences,
        key=occurrences.get,
    )


# ============================================================
# 3. Pose-size score
# ============================================================

def body_pose_spread_score(
    body: NTUBody,
) -> float:
    """
    Estimate body scale from 3D joint spread.

    Larger values usually indicate that the person is closer
    to the sensor or occupies more of the scene.
    """
    arrays = body.joint_arrays()

    camera_xyz = arrays[
        "camera_xyz"
    ]

    valid = np.isfinite(
        camera_xyz
    ).all(axis=-1)

    points = camera_xyz[valid]

    if len(points) < 2:
        return 0.0

    minimum = points.min(
        axis=0
    )

    maximum = points.max(
        axis=0
    )

    extent = maximum - minimum

    return float(
        np.linalg.norm(extent)
    )


def body_tracking_score(
    body: NTUBody,
) -> float:
    """
    Score a body using Kinect tracking-state values.

    Fully tracked joints receive more weight than inferred joints.
    """
    tracking_state = (
        body.joint_arrays()[
            "tracking_state"
        ]
    )

    tracked = np.sum(
        tracking_state == 2
    )

    inferred = np.sum(
        tracking_state == 1
    )

    return float(
        tracked + 0.25 * inferred
    )


# ============================================================
# 4. Select primary body in one frame
# ============================================================

def select_primary_body_in_frame(
    frame: NTUFrame,
    preferred_body_id: Optional[
        str
    ] = None,
) -> Optional[NTUBody]:
    """
    Select the primary body in one frame.

    Priority:
    1. Preferred body ID, when available.
    2. Best tracking score.
    3. Largest body pose spread.
    """
    if not frame.bodies:
        return None

    if preferred_body_id is not None:
        for body in frame.bodies:
            if (
                body.body_id
                == preferred_body_id
            ):
                return body

    return max(
        frame.bodies,
        key=lambda body: (
            body_tracking_score(body),
            body_pose_spread_score(body),
        ),
    )


# ============================================================
# 5. Select primary sequence track
# ============================================================

def select_primary_body_id(
    sequence: NTUSkeletonSequence,
) -> Optional[str]:
    """
    Select the main person across a sequence.

    The current method prioritizes:
    - body occurrence count
    - tracking quality
    - body scale
    """
    body_ids = sequence.body_ids

    if not body_ids:
        return None

    occurrence_counter = (
        count_body_occurrences(
            sequence
        )
    )

    tracking_scores = {
        body_id: 0.0
        for body_id in body_ids
    }

    spread_scores = {
        body_id: 0.0
        for body_id in body_ids
    }

    for frame in sequence.frames:
        for body in frame.bodies:
            tracking_scores[
                body.body_id
            ] += body_tracking_score(
                body
            )

            spread_scores[
                body.body_id
            ] += body_pose_spread_score(
                body
            )

    return max(
        body_ids,
        key=lambda body_id: (
            occurrence_counter.get(
                body_id,
                0,
            ),
            tracking_scores.get(
                body_id,
                0.0,
            ),
            spread_scores.get(
                body_id,
                0.0,
            ),
        ),
    )


# ============================================================
# 6. Extract primary pose sequence
# ============================================================

def extract_primary_pose_sequence(
    sequence: NTUSkeletonSequence,
) -> dict[str, np.ndarray | str]:
    """
    Extract the selected primary body across all frames.

    Missing frames are filled with NaN values.
    """
    primary_body_id = (
        select_primary_body_id(
            sequence
        )
    )

    if primary_body_id is None:
        raise ValueError(
            f"No valid body found in "
            f"{sequence.path}"
        )

    num_frames = sequence.num_frames
    num_joints = 25

    camera_xyz = np.full(
        (num_frames, num_joints, 3),
        np.nan,
        dtype=np.float32,
    )

    depth_xy = np.full(
        (num_frames, num_joints, 2),
        np.nan,
        dtype=np.float32,
    )

    color_xy = np.full(
        (num_frames, num_joints, 2),
        np.nan,
        dtype=np.float32,
    )

    tracking_state = np.zeros(
        (num_frames, num_joints),
        dtype=np.int8,
    )

    body_present = np.zeros(
        num_frames,
        dtype=bool,
    )

    for frame in sequence.frames:
        body = (
            select_primary_body_in_frame(
                frame,
                preferred_body_id=(
                    primary_body_id
                ),
            )
        )

        if body is None:
            continue

        if body.body_id != primary_body_id:
            continue

        arrays = body.joint_arrays()

        frame_index = (
            frame.frame_index
        )

        camera_xyz[
            frame_index
        ] = arrays["camera_xyz"]

        depth_xy[
            frame_index
        ] = arrays["depth_xy"]

        color_xy[
            frame_index
        ] = arrays["color_xy"]

        tracking_state[
            frame_index
        ] = arrays[
            "tracking_state"
        ]

        body_present[
            frame_index
        ] = True

    return {
        "body_id": primary_body_id,
        "camera_xyz": camera_xyz,
        "depth_xy": depth_xy,
        "color_xy": color_xy,
        "tracking_state": (
            tracking_state
        ),
        "body_present": body_present,
    }
