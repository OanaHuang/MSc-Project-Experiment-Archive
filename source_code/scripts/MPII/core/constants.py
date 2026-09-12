"""Official MPII 0-based joint convention."""

MPII_JOINT_NAMES = (
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax",
    "upper_neck", "head_top", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist",
)
MPII_NUM_JOINTS = len(MPII_JOINT_NAMES)

MPII_FLIP_PAIRS = ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13))

MPII_SKELETON_EDGES = (
    (0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15),
)
