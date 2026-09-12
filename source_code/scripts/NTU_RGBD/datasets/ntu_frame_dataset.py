# scripts/NTU_RGBD/datasets/ntu_frame_dataset.py

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.NTU_RGBD.core import (
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)

from scripts.NTU_RGBD.datasets.person_crop import (
    compute_person_bbox,
    crop_and_resize_person,
    crop_and_resize_person_with_bbox,
)


# ============================================================
# 1. Extracted frame directory
# ============================================================

# ============================================================
# 2. NTU joint indices
# ============================================================

# NTU RGB+D Kinect V2 joint ordering:
# 0: SpineBase
# 1: SpineMid
# 2: Neck
# 3: Head

NTU_NECK_INDEX = 2
NTU_HEAD_INDEX = 3

HEAD_LENGTH_EPSILON = 1e-6
HEAD_NECK_SCALE_FACTOR = 1.0


# ============================================================
# 3. Head-length calculation
# ============================================================

def compute_head_length(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    head_index: int = NTU_HEAD_INDEX,
    neck_index: int = NTU_NECK_INDEX,
    epsilon: float = HEAD_LENGTH_EPSILON,
) -> np.float32:
    """
    Compute the 2D distance between the NTU head and neck joints.

    The returned length must be calculated in the same coordinate
    system as the prediction and target keypoints.

    Parameters
    ----------
    keypoints:
        Shape [J, 2].

    visibility:
        Shape [J].

    head_index:
        Index of the head joint.

    neck_index:
        Index of the neck joint.

    epsilon:
        Minimum valid head length.

    Returns
    -------
    np.float32
        Head-to-neck distance. Returns NaN when the required joints
        are unavailable or invalid.
    """
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
    ).astype(bool)

    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(
            "keypoints must have shape [J, 2]"
        )

    if visibility.shape != (keypoints.shape[0],):
        raise ValueError(
            "visibility must have shape [J]"
        )

    num_joints = keypoints.shape[0]

    if not (
        0 <= head_index < num_joints
    ):
        raise IndexError(
            f"head_index {head_index} is outside "
            f"the valid range [0, {num_joints - 1}]"
        )

    if not (
        0 <= neck_index < num_joints
    ):
        raise IndexError(
            f"neck_index {neck_index} is outside "
            f"the valid range [0, {num_joints - 1}]"
        )

    if not visibility[head_index]:
        return np.float32(
            np.nan
        )

    if not visibility[neck_index]:
        return np.float32(
            np.nan
        )

    head_point = keypoints[
        head_index
    ]

    neck_point = keypoints[
        neck_index
    ]

    if not np.isfinite(
        head_point
    ).all():
        return np.float32(
            np.nan
        )

    if not np.isfinite(
        neck_point
    ).all():
        return np.float32(
            np.nan
        )

    head_length = float(
    np.linalg.norm(
        head_point - neck_point
    )
    )

    head_length *= HEAD_NECK_SCALE_FACTOR

    if (
        not np.isfinite(head_length)
        or head_length <= epsilon
    ):
        return np.float32(
            np.nan
        )

    return np.float32(
        head_length
    )


# ============================================================
# 4. Heatmap generation
# ============================================================

def generate_gaussian_heatmaps(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    image_size: int = 224,
    heatmap_size: int = 56,
    sigma: float = 2.0,
) -> np.ndarray:
    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    visibility = np.asarray(
        visibility,
        dtype=bool,
    )

    num_joints = keypoints.shape[0]

    heatmaps = np.zeros(
        (
            num_joints,
            heatmap_size,
            heatmap_size,
        ),
        dtype=np.float32,
    )

    scale = heatmap_size / image_size
    radius = int(3 * sigma)

    for joint_index in range(num_joints):
        if not visibility[joint_index]:
            continue

        x = keypoints[
            joint_index,
            0,
        ] * scale

        y = keypoints[
            joint_index,
            1,
        ] * scale

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            continue

        center_x = int(
            round(x)
        )

        center_y = int(
            round(y)
        )

        if not (
            0 <= center_x < heatmap_size
            and 0 <= center_y < heatmap_size
        ):
            continue

        x_min = max(
            center_x - radius,
            0,
        )

        x_max = min(
            center_x + radius + 1,
            heatmap_size,
        )

        y_min = max(
            center_y - radius,
            0,
        )

        y_max = min(
            center_y + radius + 1,
            heatmap_size,
        )

        grid_x = np.arange(
            x_min,
            x_max,
            dtype=np.float32,
        )

        grid_y = np.arange(
            y_min,
            y_max,
            dtype=np.float32,
        )

        yy, xx = np.meshgrid(
            grid_y,
            grid_x,
            indexing="ij",
        )

        gaussian = np.exp(
            -(
                (xx - x) ** 2
                + (yy - y) ** 2
            )
            / (2 * sigma ** 2)
        )

        heatmaps[
            joint_index,
            y_min:y_max,
            x_min:x_max,
        ] = gaussian

    return heatmaps


# ============================================================
# 5. Dataset
# ============================================================

class NTUFrameDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        transform=None,
        image_size=224,
        heatmap_size=56,
        sigma=2.0,
        frame_stride=1,
        temporal_steps=1,
        temporal_frame_gap=1,
        minimum_temporal_history=0,
        preprocessed_pose_cache=False,
        return_temporal_sequence=False,
        return_temporal_targets=False,
        single_person_only=True,
        max_samples=None,
        skeleton_cache_size=8,
        extracted_frames_dir=None,
        person_crop: bool = True,
        bbox_expansion: float = 0.25,
        validate_pose_sequences: bool = True,
        head_index: int = NTU_HEAD_INDEX,
        neck_index: int = NTU_NECK_INDEX,
        path_root: Path | None = None,
        contiguous_clip_subdir: str | None = None,
        minimum_visible_joints: int = 1,
    ):
        super().__init__()

        self.metadata_csv = Path(
            metadata_csv
        )

        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.frame_stride = frame_stride
        self.temporal_steps = int(temporal_steps)
        self.temporal_frame_gap = int(temporal_frame_gap)
        self.minimum_temporal_history = int(minimum_temporal_history)
        self.preprocessed_pose_cache = bool(preprocessed_pose_cache)
        self.return_temporal_sequence = bool(return_temporal_sequence)
        self.return_temporal_targets = bool(return_temporal_targets)

        self.single_person_only = (
            single_person_only
        )

        self.max_samples = max_samples

        self.skeleton_cache_size = (
            skeleton_cache_size
        )

        self.person_crop = person_crop

        self.bbox_expansion = (
            bbox_expansion
        )

        self.validate_pose_sequences = (
            validate_pose_sequences
        )

        self.head_index = int(
            head_index
        )

        self.neck_index = int(
            neck_index
        )

        self.path_root = Path(path_root) if path_root is not None else Path.cwd()
        self.contiguous_clip_subdir = contiguous_clip_subdir
        self.minimum_visible_joints = int(minimum_visible_joints)
        self._frame_locations: dict[tuple[str, int], tuple[Path, str]] = {}
        if self.minimum_visible_joints < 1:
            raise ValueError("minimum_visible_joints must be positive")

        if self.head_index < 0:
            raise ValueError(
                "head_index must be non-negative"
            )

        if self.neck_index < 0:
            raise ValueError(
                "neck_index must be non-negative"
            )

        if self.bbox_expansion < 0:
            raise ValueError(
                "bbox_expansion must be non-negative"
            )

        if extracted_frames_dir is None:
            raise ValueError("extracted_frames_dir must be provided explicitly")

        self.extracted_frames_dir = Path(
            extracted_frames_dir
        )

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: "
                f"{self.metadata_csv}"
            )

        if not self.extracted_frames_dir.exists():
            raise FileNotFoundError(
                "Extracted frames directory "
                f"not found: "
                f"{self.extracted_frames_dir}\n"
                "Run 05a_Extract_RGB_Frames.py first."
            )

        if frame_stride <= 0:
            raise ValueError(
                "frame_stride must be positive"
            )

        if self.temporal_steps <= 0:
            raise ValueError("temporal_steps must be positive")

        if self.temporal_frame_gap <= 0:
            raise ValueError("temporal_frame_gap must be positive")

        if self.minimum_temporal_history < 0:
            raise ValueError("minimum_temporal_history must be non-negative")

        if skeleton_cache_size <= 0:
            raise ValueError(
                "skeleton_cache_size must be positive"
            )

        self._skeleton_cache = OrderedDict()

        if self.preprocessed_pose_cache and self.transform is not None:
            if (
                getattr(self.transform, "scale_range", None) is not None
                or getattr(self.transform, "rotation_degrees", 0.0) != 0.0
                or getattr(self.transform, "flip_probability", 0.0) != 0.0
            ):
                raise ValueError(
                    "Preprocessed pose caches require augmentation to be disabled"
                )

        (
            self.samples,
            self.skipped_samples,
        ) = self._load_metadata(
            single_person_only=(
                single_person_only
            ),
            max_samples=max_samples,
        )

        self.frame_index = (
            self._build_frame_index()
        )

        if not self.frame_index:
            raise RuntimeError(
                "No valid frames remained after "
                "dataset filtering."
            )

        print()
        print("=" * 70)
        print("NTUFrameDataset")
        print("=" * 70)

        print(
            f"Metadata CSV:        "
            f"{self.metadata_csv}"
        )

        print(
            f"Valid videos:        "
            f"{len(self.samples)}"
        )

        print(
            f"Skipped videos:      "
            f"{len(self.skipped_samples)}"
        )

        print(
            f"Usable frames:       "
            f"{len(self.frame_index)}"
        )

        print(f"Temporal steps:      {self.temporal_steps}")
        print(f"Temporal frame gap:  {self.temporal_frame_gap}")
        print(f"Minimum history:     {self.minimum_temporal_history}")
        print(f"Preprocessed cache:  {self.preprocessed_pose_cache}")
        print(f"Skeleton cache:      {self.skeleton_cache_size} videos/worker")
        print(f"Minimum joints:      {self.minimum_visible_joints} tracked joints/target")

        print(
            f"Extracted frames:    "
            f"{self.extracted_frames_dir}"
        )

        print(
            f"Person crop:         "
            f"{self.person_crop}"
        )

        print(
            f"Validate skeletons:  "
            f"{self.validate_pose_sequences}"
        )

        print(
            f"Head joint index:    "
            f"{self.head_index}"
        )

        print(
            f"Neck joint index:    "
            f"{self.neck_index}"
        )

        print(
            "Head scale:          "
            "2D head-to-neck distance"
        )

        if self.skipped_samples:
            print()
            print("Skipped sample examples:")

            for item in self.skipped_samples[:10]:
                print(
                    f"  {item['sample_id']} | "
                    f"{item['reason']}"
                )

            if len(self.skipped_samples) > 10:
                print(
                    f"  ... and "
                    f"{len(self.skipped_samples) - 10} "
                    f"more"
                )

        print("=" * 70)
        print()

    # ========================================================
    # 6. Metadata loading and validation
    # ========================================================

    @staticmethod
    def _string_to_bool(
        value,
    ) -> bool:
        return str(
            value
        ).strip().lower() in {
            "true",
            "1",
            "yes",
        }

    @staticmethod
    def _safe_int(
        value,
        default: int = 0,
    ) -> int:
        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return default

    def _metadata_pose_is_empty(
        self,
        row: dict,
    ) -> tuple[bool, str]:
        """
        Fast metadata-level rejection.

        Returns:
            is_empty
            rejection reason
        """
        max_bodies = self._safe_int(
            row.get(
                "max_bodies",
                0,
            )
        )

        skeleton_frames = self._safe_int(
            row.get(
                "skeleton_frames",
                0,
            )
        )

        empty_frames = self._safe_int(
            row.get(
                "empty_frames",
                0,
            )
        )

        if skeleton_frames <= 0:
            return (
                True,
                "skeleton_frames <= 0",
            )

        if max_bodies <= 0:
            return (
                True,
                "max_bodies <= 0",
            )

        if empty_frames >= skeleton_frames:
            return (
                True,
                "all skeleton frames are empty",
            )

        return (
            False,
            "",
        )

    def _validate_pose_sequence(
        self,
        skeleton_path: Path,
    ) -> tuple[bool, str]:
        """
        Strictly verify that the skeleton file contains a
        primary pose sequence that the Dataset can use.
        """
        if not skeleton_path.exists():
            return (
                False,
                f"skeleton file not found: "
                f"{skeleton_path}",
            )

        try:
            pose_sequence = (
                self._load_pose_sequence(
                    skeleton_path
                )
            )

        except Exception as error:
            return (
                False,
                f"{type(error).__name__}: "
                f"{error}",
            )

        if "color_xy" not in pose_sequence:
            return (
                False,
                "pose sequence has no color_xy",
            )

        if "tracking_state" not in pose_sequence:
            return (
                False,
                "pose sequence has no tracking_state",
            )

        color_xy = np.asarray(
            pose_sequence["color_xy"]
        )

        tracking_state = np.asarray(
            pose_sequence["tracking_state"]
        )

        if color_xy.ndim != 3:
            return (
                False,
                "color_xy has invalid shape: "
                f"{color_xy.shape}",
            )

        if tracking_state.ndim != 2:
            return (
                False,
                "tracking_state has invalid shape: "
                f"{tracking_state.shape}",
            )

        if color_xy.shape[0] <= 0:
            return (
                False,
                "pose sequence contains no frames",
            )

        if (
            tracking_state.shape[0]
            != color_xy.shape[0]
        ):
            return (
                False,
                "pose and tracking frame counts differ",
            )

        if not np.isfinite(
            color_xy
        ).any():
            return (
                False,
                "all pose coordinates are non-finite",
            )

        return (
            True,
            "",
        )

    def _load_metadata(
        self,
        single_person_only: bool,
        max_samples: Optional[int],
    ) -> tuple[
        list[dict],
        list[dict[str, str]],
    ]:
        valid_rows: list[dict] = []
        skipped_rows: list[
            dict[str, str]
        ] = []

        with self.metadata_csv.open(
            "r",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            for row in reader:
                sample_id = str(
                    row.get(
                        "sample_id",
                        "",
                    )
                ).strip()

                if not sample_id:
                    skipped_rows.append(
                        {
                            "sample_id": "<missing>",
                            "reason": (
                                "missing sample_id"
                            ),
                        }
                    )
                    continue

                if single_person_only:
                    is_single_person = (
                        self._string_to_bool(
                            row.get(
                                "is_single_person",
                                "",
                            )
                        )
                    )

                    if not is_single_person:
                        skipped_rows.append(
                            {
                                "sample_id": (
                                    sample_id
                                ),
                                "reason": (
                                    "not single-person"
                                ),
                            }
                        )
                        continue

                is_empty, empty_reason = (
                    self._metadata_pose_is_empty(
                        row
                    )
                )

                if is_empty:
                    skipped_rows.append(
                        {
                            "sample_id": (
                                sample_id
                            ),
                            "reason": (
                                empty_reason
                            ),
                        }
                    )
                    continue

                skeleton_path = Path(
                    str(
                        row.get(
                            "skeleton_path",
                            "",
                        )
                    )
                )
                if not skeleton_path.is_absolute():
                    skeleton_path = self.path_root / skeleton_path
                row["skeleton_path"] = str(skeleton_path)

                if self.validate_pose_sequences:
                    (
                        is_valid,
                        validation_reason,
                    ) = self._validate_pose_sequence(
                        skeleton_path
                    )

                    if not is_valid:
                        skipped_rows.append(
                            {
                                "sample_id": (
                                    sample_id
                                ),
                                "reason": (
                                    validation_reason
                                ),
                            }
                        )
                        continue

                valid_rows.append(
                    row
                )

                if (
                    max_samples is not None
                    and len(valid_rows)
                    >= max_samples
                ):
                    break

        if not valid_rows:
            raise RuntimeError(
                "No valid samples remained after "
                "metadata and pose validation."
            )

        return (
            valid_rows,
            skipped_rows,
        )

    # ========================================================
    # 7. Frame index
    # ========================================================

    def _build_frame_index(
        self,
    ) -> list[tuple[int, int]]:
        if not hasattr(self, "_frame_locations"):
            self._frame_locations = {}
        frame_index: list[
            tuple[int, int]
        ] = []

        valid_samples: list[dict] = []

        skipped_missing_frames: list[
            dict[str, str]
        ] = []

        for sample in self.samples:
            sample_id = str(
                sample["sample_id"]
            )

            rgb_frames = self._safe_int(
                sample.get(
                    "rgb_frames",
                    0,
                )
            )

            skeleton_frames = (
                self._safe_int(
                    sample.get(
                        "skeleton_frames",
                        0,
                    )
                )
            )

            usable_frames = min(
                rgb_frames,
                skeleton_frames,
            )

            if usable_frames <= 0:
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "usable frame count <= 0"
                        ),
                    }
                )
                continue

            sample_frame_dir = self._sample_frame_dir(sample_id)

            if not sample_frame_dir.exists():
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "extracted frame directory "
                            "not found"
                        ),
                    }
                )
                continue

            new_sample_index = len(
                valid_samples
            )

            sample_frame_indices = []

            locations = self._sample_frame_locations(sample_id, sample_frame_dir)
            minimum_visible = getattr(self, "minimum_visible_joints", None)
            pose = None
            if minimum_visible is not None and sample.get("skeleton_path"):
                pose = self._load_pose_sequence(Path(sample["skeleton_path"]))
            image_size = (
                self._safe_int(sample.get("width", 1920), 1920),
                self._safe_int(sample.get("height", 1080), 1080),
            )
            for frame_number in sorted(locations):
                if frame_number >= usable_frames or frame_number % self.frame_stride:
                    continue
                if pose is not None:
                    visibility = coordinate_visibility(
                        pose["color_xy"][frame_number],
                        tracking_state=pose["tracking_state"][frame_number],
                        image_size=image_size, include_inferred=False,
                    )
                    if int(np.count_nonzero(visibility)) < minimum_visible:
                        continue
                self._frame_locations[(sample_id, frame_number)] = locations[frame_number]
                sample_frame_indices.append((new_sample_index, frame_number))

            if self.temporal_steps > 1 or self.minimum_temporal_history > 0:
                available = {frame for _, frame in sample_frame_indices}
                history = (self.temporal_steps - 1) * self.temporal_frame_gap
                sample_frame_indices = [
                    (sample_index, target)
                    for sample_index, target in sample_frame_indices
                    if target >= max(history, self.minimum_temporal_history)
                    and all(
                        target - offset * self.temporal_frame_gap in available
                        for offset in range(self.temporal_steps)
                    )
                    and all(
                        target - offset in available
                        for offset in range(self.minimum_temporal_history + 1)
                    )
                    and self._history_stays_in_one_clip(
                        sample_id, target, self.temporal_steps,
                        self.temporal_frame_gap,
                    )
                    and self._history_stays_in_one_clip(
                        sample_id, target, self.minimum_temporal_history + 1, 1,
                    )
                ]

            if not sample_frame_indices:
                skipped_missing_frames.append(
                    {
                        "sample_id": sample_id,
                        "reason": (
                            "no extracted frames found"
                        ),
                    }
                )
                continue

            valid_samples.append(
                sample
            )

            frame_index.extend(
                sample_frame_indices
            )

        self.samples = valid_samples

        self.skipped_samples.extend(
            skipped_missing_frames
        )

        return frame_index

    def _sample_frame_dir(self, sample_id: str) -> Path:
        if getattr(self, "contiguous_clip_subdir", None):
            return (
                self.extracted_frames_dir / sample_id[:4]
                / self.contiguous_clip_subdir / sample_id
            )
        return self.extracted_frames_dir / sample_id

    def _sample_frame_locations(
        self, sample_id: str, sample_frame_dir: Path,
    ) -> dict[int, tuple[Path, str]]:
        locations: dict[int, tuple[Path, str]] = {}
        if getattr(self, "contiguous_clip_subdir", None):
            clip_dirs = sorted(path for path in sample_frame_dir.glob("clip_*") if path.is_dir())
        else:
            clip_dirs = [sample_frame_dir]
        for clip_dir in clip_dirs:
            clip_id = clip_dir.name if getattr(self, "contiguous_clip_subdir", None) else "flat"
            for frame_path in sorted(clip_dir.glob("frame_*.jpg")):
                try:
                    frame_number = int(frame_path.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if frame_number in locations:
                    raise RuntimeError(
                        f"duplicate sampled frame {frame_number} for {sample_id}"
                    )
                locations[frame_number] = (frame_path, clip_id)
        return locations

    def _history_stays_in_one_clip(
        self, sample_id: str, target: int, steps: int, gap: int | None = None,
    ) -> bool:
        gap = self.temporal_frame_gap if gap is None else gap
        frames = [
            target - offset * gap
            for offset in range(steps)
        ]
        locations = [self._frame_locations.get((sample_id, frame)) for frame in frames]
        return all(location is not None for location in locations) and len({
            location[1] for location in locations if location is not None
        }) == 1

    # ========================================================
    # 8. Skeleton loading and cache
    # ========================================================

    def _load_pose_sequence(
        self,
        skeleton_path: Path,
    ) -> dict:
        cache_key = str(
            skeleton_path
        )

        if cache_key in self._skeleton_cache:
            value = self._skeleton_cache.pop(
                cache_key
            )

            self._skeleton_cache[
                cache_key
            ] = value

            return value

        sequence = read_skeleton_file(
            skeleton_path
        )

        pose_sequence = (
            extract_primary_pose_sequence(
                sequence
            )
        )

        self._skeleton_cache[
            cache_key
        ] = pose_sequence

        while (
            len(self._skeleton_cache)
            > self.skeleton_cache_size
        ):
            self._skeleton_cache.popitem(
                last=False
            )

        return pose_sequence

    def _load_preprocessed_pose(self, sample_id: str) -> dict:
        cache_path = self.extracted_frames_dir / sample_id / "pose_cache.npz"
        cache_key = f"preprocessed:{cache_path}"
        if cache_key in self._skeleton_cache:
            value = self._skeleton_cache.pop(cache_key)
            self._skeleton_cache[cache_key] = value
            return value
        if not cache_path.is_file():
            raise FileNotFoundError(f"Preprocessed pose cache not found: {cache_path}")
        with np.load(cache_path) as data:
            value = {key: data[key].copy() for key in data.files}
        value["frame_lookup"] = {
            int(frame): index for index, frame in enumerate(value["frame_numbers"])
        }
        self._skeleton_cache[cache_key] = value
        while len(self._skeleton_cache) > self.skeleton_cache_size:
            self._skeleton_cache.popitem(last=False)
        return value

    # ========================================================
    # 9. Frame path
    # ========================================================

    def _get_frame_path(
        self,
        sample_id: str,
        frame_number: int,
    ) -> Path:
        location = self._frame_locations.get((sample_id, frame_number))
        if location is not None:
            return location[0]
        return self._sample_frame_dir(sample_id) / f"frame_{frame_number:06d}.jpg"

    # ========================================================
    # 10. Dataset interface
    # ========================================================

    def __len__(
        self,
    ) -> int:
        return len(
            self.frame_index
        )

    def _get_frame_item(
        self,
        sample_index: int,
        frame_number: int,
        transform_parameters: dict[str, float | bool] | None = None,
        include_target: bool = True,
        crop_bbox: np.ndarray | None = None,
    ) -> dict[str, object]:
        if self.preprocessed_pose_cache:
            return self._get_preprocessed_frame_item(
                sample_index, frame_number, transform_parameters, include_target,
            )
        sample = self.samples[
            sample_index
        ]

        sample_id = str(
            sample["sample_id"]
        )

        skeleton_path = Path(
            sample["skeleton_path"]
        )

        frame_path = self._get_frame_path(
            sample_id=sample_id,
            frame_number=frame_number,
        )

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Extracted frame not found: "
                f"{frame_path}"
            )

        image = cv2.imread(
            str(frame_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                "Could not read extracted frame: "
                f"{frame_path}"
            )

        pose_sequence = (
            self._load_pose_sequence(
                skeleton_path
            )
        )

        if (
            frame_number
            >= len(
                pose_sequence["color_xy"]
            )
        ):
            raise IndexError(
                f"Pose frame index out of range: "
                f"sample={sample_id}, "
                f"frame={frame_number}"
            )

        keypoints = pose_sequence[
            "color_xy"
        ][frame_number].copy()

        tracking_state = pose_sequence[
            "tracking_state"
        ][frame_number].copy()

        visibility = (
            coordinate_visibility(
                keypoints,
                tracking_state=tracking_state,
                image_size=(
                    image.shape[1],
                    image.shape[0],
                ),
                include_inferred=False,
            )
        ).astype(
            np.float32
        )

        original_keypoints = (
            keypoints.copy()
        )

        original_visibility = (
            visibility.copy()
        )

        if self.person_crop:
            crop_result = (
                crop_and_resize_person_with_bbox(
                    image=image, keypoints=keypoints, visibility=visibility,
                    bbox_xyxy=crop_bbox, output_size=self.image_size,
                ) if crop_bbox is not None else crop_and_resize_person(
                    image=image,
                    keypoints=keypoints,
                    visibility=visibility,
                    output_size=(
                        self.image_size
                    ),
                    expansion=(
                        self.bbox_expansion
                    ),
                    make_square=True,
                )
            )

            image = crop_result.image
            keypoints = crop_result.keypoints
            visibility = crop_result.visibility

            person_bbox = (
                crop_result.bbox_xyxy
            )

        else:
            person_bbox = np.array(
                [
                    0.0,
                    0.0,
                    float(image.shape[1]),
                    float(image.shape[0]),
                ],
                dtype=np.float32,
            )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                keypoints=keypoints,
                visibility=visibility,
                parameters=transform_parameters,
            )

            image_tensor = transformed[
                "image"
            ]

            keypoints_tensor = transformed[
                "keypoints"
            ]

            visibility_tensor = transformed[
                "visibility"
            ]

        else:
            original_height, original_width = (
                image.shape[:2]
            )

            image = cv2.resize(
                image,
                (
                    self.image_size,
                    self.image_size,
                ),
            )

            keypoints[:, 0] *= (
                self.image_size
                / original_width
            )

            keypoints[:, 1] *= (
                self.image_size
                / original_height
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            image_tensor = (
                torch.from_numpy(
                    np.transpose(
                        image,
                        (
                            2,
                            0,
                            1,
                        ),
                    )
                )
                .float()
                / 255.0
            )

            keypoints_tensor = (
                torch.from_numpy(
                    keypoints
                ).float()
            )

            visibility_tensor = (
                torch.from_numpy(
                    visibility
                ).float()
            )

        if not include_target:
            return {"image": image_tensor}

        # Calculate head length after all crop and resize
        # operations. This keeps it in the same coordinate system
        # as keypoints_tensor and decoded predictions.
        head_length = compute_head_length(
            keypoints=(
                keypoints_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            visibility=(
                visibility_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            head_index=self.head_index,
            neck_index=self.neck_index,
        )

        heatmaps = generate_gaussian_heatmaps(
            keypoints=(
                keypoints_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            visibility=(
                visibility_tensor
                .detach()
                .cpu()
                .numpy()
            ),
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        return {
            "image": image_tensor,

            "heatmaps": torch.from_numpy(
                heatmaps
            ).float(),

            "keypoints": keypoints_tensor,

            # Shared pose-dataset contract used by MPII evaluation and tooling.
            # Keep original_keypoints below as a compatibility alias.
            "keypoints_original": torch.from_numpy(
                original_keypoints
            ).float(),

            "visibility": visibility_tensor,

            "head_length": torch.tensor(
                head_length,
                dtype=torch.float32,
            ),

            "sample_id": sample_id,

            "frame_index": frame_number,

            "rgb_path": str(
                frame_path
            ),

            "person_bbox": torch.from_numpy(
                person_bbox
            ).float(),

            "original_keypoints": torch.from_numpy(
                original_keypoints
            ).float(),

            "original_visibility": torch.from_numpy(
                original_visibility
            ).float(),
        }

    def _get_preprocessed_frame_item(
        self,
        sample_index: int,
        frame_number: int,
        transform_parameters: dict[str, float | bool] | None = None,
        include_target: bool = True,
    ) -> dict[str, object]:
        sample = self.samples[sample_index]
        sample_id = str(sample["sample_id"])
        frame_path = self._get_frame_path(sample_id, frame_number)
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read preprocessed frame: {frame_path}")
        pose = self._load_preprocessed_pose(sample_id)
        try:
            pose_index = pose["frame_lookup"][frame_number]
        except KeyError as error:
            raise IndexError(
                f"Frame {frame_number} missing from pose cache for {sample_id}"
            ) from error
        keypoints = pose["keypoints"][pose_index].copy()
        visibility = pose["visibility"][pose_index].copy()
        if self.transform is not None:
            transformed = self.transform(
                image=image, keypoints=keypoints, visibility=visibility,
                parameters=transform_parameters,
            )
            image_tensor = transformed["image"]
            keypoints_tensor = transformed["keypoints"]
            visibility_tensor = transformed["visibility"]
        else:
            image_tensor = torch.from_numpy(
                np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2, 0, 1))
            ).float() / 255.0
            keypoints_tensor = torch.from_numpy(keypoints).float()
            visibility_tensor = torch.from_numpy(visibility).float()
        if not include_target:
            return {"image": image_tensor}
        heatmaps = generate_gaussian_heatmaps(
            keypoints=keypoints_tensor.numpy(),
            visibility=visibility_tensor.numpy(),
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )
        original_keypoints = pose["original_keypoints"][pose_index]
        original_visibility = pose["original_visibility"][pose_index]
        return {
            "image": image_tensor,
            "heatmaps": torch.from_numpy(heatmaps).float(),
            "keypoints": keypoints_tensor,
            "keypoints_original": torch.from_numpy(original_keypoints).float(),
            "visibility": visibility_tensor,
            "head_length": torch.tensor(
                float(pose["head_length"][pose_index]), dtype=torch.float32,
            ),
            "sample_id": sample_id,
            "frame_index": frame_number,
            "rgb_path": str(frame_path),
            "person_bbox": torch.from_numpy(pose["person_bbox"][pose_index]).float(),
            "original_keypoints": torch.from_numpy(original_keypoints).float(),
            "original_visibility": torch.from_numpy(original_visibility).float(),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, target_frame = self.frame_index[index]
        parameters = (
            self.transform.sample_parameters()
            if self.transform is not None
            and hasattr(self.transform, "sample_parameters")
            else None
        )
        frame_numbers = [
            target_frame - offset * self.temporal_frame_gap
            for offset in reversed(range(self.temporal_steps))
        ]
        crop_bbox = self._shared_temporal_bbox(sample_index, frame_numbers)
        items = []
        for order, frame in enumerate(frame_numbers):
            kwargs = {
                "include_target": (
                    getattr(self, "return_temporal_targets", False)
                    or order == len(frame_numbers) - 1
                ),
            }
            if crop_bbox is not None:
                kwargs["crop_bbox"] = crop_bbox
            items.append(self._get_frame_item(sample_index, frame, parameters, **kwargs))
        target = items[-1]
        if self.return_temporal_sequence:
            target["image"] = torch.stack(
                [item["image"] for item in items], dim=0,
            )
        if getattr(self, "return_temporal_targets", False):
            target["temporal_heatmaps"] = torch.stack(
                [item["heatmaps"] for item in items], dim=0,
            )
            target["temporal_visibility"] = torch.stack(
                [item["visibility"] for item in items], dim=0,
            )
        target["temporal_frame_indices"] = torch.tensor(
            frame_numbers, dtype=torch.int64,
        )
        return target

    def _shared_temporal_bbox(
        self, sample_index: int, frame_numbers: list[int],
    ) -> np.ndarray | None:
        if (
            not getattr(self, "person_crop", False)
            or getattr(self, "preprocessed_pose_cache", False)
            or len(frame_numbers) < 2
        ):
            return None
        sample = self.samples[sample_index]
        sample_id = str(sample["sample_id"])
        pose = self._load_pose_sequence(Path(sample["skeleton_path"]))
        first = cv2.imread(str(self._get_frame_path(sample_id, frame_numbers[0])))
        if first is None:
            raise RuntimeError(f"Could not read temporal frame for shared crop: {sample_id}")
        height, width = first.shape[:2]
        points = []
        visible = []
        for frame in frame_numbers:
            keypoints = pose["color_xy"][frame]
            visibility = coordinate_visibility(
                keypoints, tracking_state=pose["tracking_state"][frame],
                image_size=(width, height), include_inferred=False,
            )
            points.append(keypoints)
            visible.append(visibility)
        return compute_person_bbox(
            np.concatenate(points, axis=0), np.concatenate(visible, axis=0),
            width, height, expansion=self.bbox_expansion, make_square=True,
        )
