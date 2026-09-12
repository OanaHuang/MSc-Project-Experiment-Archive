# scripts/NTU_RGBD/core/skeleton_reader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import NTU_NUM_JOINTS


@dataclass
class NTUJoint:
    camera_xyz: np.ndarray
    depth_xy: np.ndarray
    color_xy: np.ndarray
    orientation_wxyz: np.ndarray
    tracking_state: int

    def to_dict(self) -> dict:
        return {
            "camera_xyz": self.camera_xyz,
            "depth_xy": self.depth_xy,
            "color_xy": self.color_xy,
            "orientation_wxyz": self.orientation_wxyz,
            "tracking_state": self.tracking_state,
        }


@dataclass
class NTUBody:
    body_id: str
    body_info: list[float | int | str]
    joints: list[NTUJoint]

    def joint_arrays(self) -> dict[str, np.ndarray]:
        return {
            "camera_xyz": np.stack(
                [joint.camera_xyz for joint in self.joints],
                axis=0,
            ).astype(np.float32),

            "depth_xy": np.stack(
                [joint.depth_xy for joint in self.joints],
                axis=0,
            ).astype(np.float32),

            "color_xy": np.stack(
                [joint.color_xy for joint in self.joints],
                axis=0,
            ).astype(np.float32),

            "orientation_wxyz": np.stack(
                [joint.orientation_wxyz for joint in self.joints],
                axis=0,
            ).astype(np.float32),

            "tracking_state": np.asarray(
                [joint.tracking_state for joint in self.joints],
                dtype=np.int8,
            ),
        }


@dataclass
class NTUFrame:
    frame_index: int
    bodies: list[NTUBody]


@dataclass
class NTUSkeletonSequence:
    path: Path
    frames: list[NTUFrame]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def max_bodies(self) -> int:
        return max(
            (len(frame.bodies) for frame in self.frames),
            default=0,
        )

    @property
    def body_ids(self) -> set[str]:
        return {
            body.body_id
            for frame in self.frames
            for body in frame.bodies
        }


def _parse_number(
    token: str,
) -> int | float | str:
    token = token.strip()

    try:
        if "." in token or "e" in token.lower():
            return float(token)

        return int(token)

    except ValueError:
        return token


def _read_nonempty_line(
    iterator: Iterator[str],
    context: str,
) -> str:
    try:
        line = next(iterator)

    except StopIteration as error:
        raise ValueError(
            f"Unexpected end of skeleton file while reading {context}"
        ) from error

    return line.strip()


def _parse_joint_line(
    line: str,
    path: Path,
    frame_index: int,
    body_index: int,
    joint_index: int,
) -> NTUJoint:
    tokens = line.split()

    if len(tokens) < 12:
        raise ValueError(
            f"Invalid joint line in {path}, "
            f"frame={frame_index}, "
            f"body={body_index}, "
            f"joint={joint_index}: "
            f"expected at least 12 values, got {len(tokens)}"
        )

    values = [
        float(value)
        for value in tokens[:11]
    ]

    tracking_state = int(
        float(tokens[11])
    )

    x, y, z = values[0:3]
    depth_x, depth_y = values[3:5]
    color_x, color_y = values[5:7]

    orientation_wxyz = np.asarray(
        values[7:11],
        dtype=np.float32,
    )

    return NTUJoint(
        camera_xyz=np.asarray(
            [x, y, z],
            dtype=np.float32,
        ),

        depth_xy=np.asarray(
            [depth_x, depth_y],
            dtype=np.float32,
        ),

        color_xy=np.asarray(
            [color_x, color_y],
            dtype=np.float32,
        ),

        orientation_wxyz=orientation_wxyz,

        tracking_state=tracking_state,
    )


def read_skeleton_file(
    skeleton_path: str | Path,
    strict_joint_count: bool = True,
) -> NTUSkeletonSequence:
    """
    Read one NTU RGB+D .skeleton file.
    """
    skeleton_path = Path(
        skeleton_path
    )

    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"Skeleton file does not exist: {skeleton_path}"
        )

    with skeleton_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        lines = iter(handle)

        first_line = _read_nonempty_line(
            lines,
            "number of frames",
        )

        try:
            num_frames = int(
                first_line
            )

        except ValueError as error:
            raise ValueError(
                f"Invalid frame count in {skeleton_path}: {first_line}"
            ) from error

        frames: list[NTUFrame] = []

        for frame_index in range(
            num_frames
        ):
            num_bodies_line = _read_nonempty_line(
                lines,
                f"number of bodies for frame {frame_index}",
            )

            try:
                num_bodies = int(
                    num_bodies_line
                )

            except ValueError as error:
                raise ValueError(
                    f"Invalid body count in {skeleton_path}, "
                    f"frame={frame_index}: {num_bodies_line}"
                ) from error

            bodies: list[NTUBody] = []

            for body_index in range(
                num_bodies
            ):
                body_info_line = _read_nonempty_line(
                    lines,
                    (
                        f"body information for frame {frame_index}, "
                        f"body {body_index}"
                    ),
                )

                body_tokens = body_info_line.split()

                if not body_tokens:
                    raise ValueError(
                        f"Empty body information line in {skeleton_path}, "
                        f"frame={frame_index}, body={body_index}"
                    )

                body_id = body_tokens[0]

                body_info = [
                    _parse_number(token)
                    for token in body_tokens
                ]

                num_joints_line = _read_nonempty_line(
                    lines,
                    (
                        f"joint count for frame {frame_index}, "
                        f"body {body_index}"
                    ),
                )

                try:
                    num_joints = int(
                        num_joints_line
                    )

                except ValueError as error:
                    raise ValueError(
                        f"Invalid joint count in {skeleton_path}, "
                        f"frame={frame_index}, body={body_index}: "
                        f"{num_joints_line}"
                    ) from error

                if (
                    strict_joint_count
                    and num_joints != NTU_NUM_JOINTS
                ):
                    raise ValueError(
                        f"Expected {NTU_NUM_JOINTS} joints, "
                        f"found {num_joints} in {skeleton_path}, "
                        f"frame={frame_index}, body={body_index}"
                    )

                joints: list[NTUJoint] = []

                for joint_index in range(
                    num_joints
                ):
                    joint_line = _read_nonempty_line(
                        lines,
                        (
                            f"joint {joint_index} "
                            f"for frame {frame_index}, "
                            f"body {body_index}"
                        ),
                    )

                    joint = _parse_joint_line(
                        joint_line,
                        path=skeleton_path,
                        frame_index=frame_index,
                        body_index=body_index,
                        joint_index=joint_index,
                    )

                    joints.append(
                        joint
                    )

                bodies.append(
                    NTUBody(
                        body_id=body_id,
                        body_info=body_info,
                        joints=joints,
                    )
                )

            frames.append(
                NTUFrame(
                    frame_index=frame_index,
                    bodies=bodies,
                )
            )

    return NTUSkeletonSequence(
        path=skeleton_path,
        frames=frames,
    )


def read_skeleton_summary(
    skeleton_path: str | Path,
) -> dict[str, int | bool | str]:
    """
    Quickly read frame and body counts without storing joint arrays.
    """
    skeleton_path = Path(
        skeleton_path
    )

    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"Skeleton file does not exist: {skeleton_path}"
        )

    frame_body_counts: list[int] = []

    with skeleton_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        lines = iter(handle)

        num_frames = int(
            _read_nonempty_line(
                lines,
                "number of frames",
            )
        )

        for frame_index in range(
            num_frames
        ):
            num_bodies = int(
                _read_nonempty_line(
                    lines,
                    f"number of bodies for frame {frame_index}",
                )
            )

            frame_body_counts.append(
                num_bodies
            )

            for body_index in range(
                num_bodies
            ):
                _ = _read_nonempty_line(
                    lines,
                    (
                        f"body information for frame {frame_index}, "
                        f"body {body_index}"
                    ),
                )

                num_joints = int(
                    _read_nonempty_line(
                        lines,
                        (
                            f"joint count for frame {frame_index}, "
                            f"body {body_index}"
                        ),
                    )
                )

                for _ in range(
                    num_joints
                ):
                    _read_nonempty_line(
                        lines,
                        "joint data",
                    )

    max_bodies = max(
        frame_body_counts,
        default=0,
    )

    nonempty_frames = sum(
        count > 0
        for count in frame_body_counts
    )

    return {
        "path": str(
            skeleton_path
        ),

        "num_frames": num_frames,

        "max_bodies": max_bodies,

        "nonempty_frames": (
            nonempty_frames
        ),

        "empty_frames": (
            num_frames
            - nonempty_frames
        ),

        "is_single_person": (
            max_bodies <= 1
        ),
    }


def extract_body_track(
    sequence: NTUSkeletonSequence,
    body_id: str,
) -> dict[str, np.ndarray | str]:
    """
    Extract one body ID over all frames.

    Missing frames are represented by NaN coordinates.
    """
    num_frames = sequence.num_frames

    camera_xyz = np.full(
        (
            num_frames,
            NTU_NUM_JOINTS,
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    depth_xy = np.full(
        (
            num_frames,
            NTU_NUM_JOINTS,
            2,
        ),
        np.nan,
        dtype=np.float32,
    )

    color_xy = np.full(
        (
            num_frames,
            NTU_NUM_JOINTS,
            2,
        ),
        np.nan,
        dtype=np.float32,
    )

    tracking_state = np.zeros(
        (
            num_frames,
            NTU_NUM_JOINTS,
        ),
        dtype=np.int8,
    )

    body_present = np.zeros(
        num_frames,
        dtype=bool,
    )

    for frame in sequence.frames:
        matching_body = next(
            (
                body
                for body in frame.bodies
                if body.body_id == body_id
            ),
            None,
        )

        if matching_body is None:
            continue

        arrays = (
            matching_body
            .joint_arrays()
        )

        frame_index = (
            frame.frame_index
        )

        camera_xyz[
            frame_index
        ] = arrays[
            "camera_xyz"
        ]

        depth_xy[
            frame_index
        ] = arrays[
            "depth_xy"
        ]

        color_xy[
            frame_index
        ] = arrays[
            "color_xy"
        ]

        tracking_state[
            frame_index
        ] = arrays[
            "tracking_state"
        ]

        body_present[
            frame_index
        ] = True

    return {
        "body_id": body_id,
        "camera_xyz": camera_xyz,
        "depth_xy": depth_xy,
        "color_xy": color_xy,
        "tracking_state": tracking_state,
        "body_present": body_present,
    }
