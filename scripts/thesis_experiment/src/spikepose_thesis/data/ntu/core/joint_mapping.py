from __future__ import annotations

import numpy as np

from .config import NTU_FLIP_PAIRS, NTU_JOINT_NAMES


MPII16_JOINT_NAMES = (
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax", "upper_neck",
    "head_top", "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
)

# MPII order -> Kinect V2 NTU-25 order. Pelvis/thorax/head-top use the nearest
# directly observed NTU joints: spine_base, spine_shoulder and head.
NTU25_TO_MPII16 = np.asarray(
    [18, 17, 16, 12, 13, 14, 0, 20, 2, 3, 10, 9, 8, 4, 5, 6],
    dtype=np.int64,
)
MPII16_FLIP_PAIRS = ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13))
MPII16_NECK_INDEX = 8
MPII16_HEAD_INDEX = 9
NTU25_NECK_INDEX = 2
NTU25_HEAD_INDEX = 3

NTU25_IDENTITY = "ntu25_identity_v1"
NTU25_TO_MPII16_LAYOUT = "ntu25_to_mpii16_v1"


def joint_layout(mapping: str) -> dict[str, object]:
    """Return the output-layout contract used by data and evaluation code."""
    if mapping == NTU25_TO_MPII16_LAYOUT:
        return {
            "num_joints": 16,
            "joint_names": MPII16_JOINT_NAMES,
            "flip_pairs": MPII16_FLIP_PAIRS,
            "head_index": MPII16_HEAD_INDEX,
            "neck_index": MPII16_NECK_INDEX,
            "map_to_mpii16": True,
        }
    if mapping == NTU25_IDENTITY:
        return {
            "num_joints": 25,
            "joint_names": NTU_JOINT_NAMES,
            "flip_pairs": NTU_FLIP_PAIRS,
            "head_index": NTU25_HEAD_INDEX,
            "neck_index": NTU25_NECK_INDEX,
            "map_to_mpii16": False,
        }
    raise ValueError(f"Unknown NTU joint layout: {mapping!r}")


def joint_names_for_count(num_joints: int) -> tuple[str, ...]:
    if int(num_joints) == len(MPII16_JOINT_NAMES):
        return MPII16_JOINT_NAMES
    if int(num_joints) == len(NTU_JOINT_NAMES):
        return NTU_JOINT_NAMES
    return tuple(f"joint_{index}" for index in range(int(num_joints)))


def map_ntu25_to_mpii16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-2] == 16:
        return value.copy()
    if value.shape[-2] != 25:
        raise ValueError(f"Expected 25 or 16 joints, got shape {value.shape}")
    return np.take(value, NTU25_TO_MPII16, axis=-2)
