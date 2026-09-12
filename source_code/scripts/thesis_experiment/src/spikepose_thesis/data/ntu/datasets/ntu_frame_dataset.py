
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from spikepose_thesis.data.ntu.core import (
    coordinate_visibility,
    extract_body_pose_sequence,
    extract_primary_pose_sequence,
    read_skeleton_file,
)
from spikepose_thesis.data.ntu.core.joint_mapping import (
    MPII16_HEAD_INDEX,
    MPII16_NECK_INDEX,
    map_ntu25_to_mpii16,
)
from spikepose_thesis.data.ntu.preflight import validate_pose_file
from spikepose_thesis.data.ntu.runtime_cache import (
    RuntimeCacheReader,
    transform_keypoints_to_bbox,
)

from spikepose_thesis.data.ntu.datasets.person_crop import (
    compute_person_bbox,
    crop_and_resize_person,
    crop_and_resize_person_with_bbox,
    quantize_person_bbox,
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
        frame_layout: str = "flat",
        frame_clip_subdir: str = "contiguous_2x16",
        exclude_overlapping_clips: bool = False,
        person_crop: bool = True,
        bbox_expansion: float = 0.25,
        validate_pose_sequences: bool = True,
        head_index: int = MPII16_HEAD_INDEX,
        neck_index: int = MPII16_NECK_INDEX,
        map_to_mpii16: bool = True,
        tube_crop: bool = True,
        tube_crop_scope: str = "window",
        validated_sample_ids: frozenset[str] | None = None,
        generate_heatmaps: bool = True,
        runtime_cache_root: Path | None = None,
        runtime_spatial_crops: bool = False,
        allowed_setups: list[str] | tuple[str, ...] | None = None,
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
        self.validated_sample_ids = validated_sample_ids
        self.allowed_setups = (
            frozenset(str(value) for value in allowed_setups)
            if allowed_setups is not None else None
        )

        self.head_index = int(
            head_index
        )

        self.neck_index = int(
            neck_index
        )
        self.map_to_mpii16 = bool(map_to_mpii16)
        self.tube_crop = bool(tube_crop)
        self.tube_crop_scope = str(tube_crop_scope)
        self.generate_heatmaps = bool(generate_heatmaps)
        self.runtime_spatial_crops = bool(runtime_spatial_crops)
        self.runtime_cache = (
            RuntimeCacheReader(Path(runtime_cache_root))
            if runtime_cache_root is not None else None
        )
        if self.runtime_spatial_crops and self.runtime_cache is None:
            raise ValueError("runtime_spatial_crops requires runtime_cache_root")
        if self.tube_crop_scope not in {"window", "video"}:
            raise ValueError(
                "tube_crop_scope must be 'window' or 'video'; "
                f"got {self.tube_crop_scope!r}"
            )
        if self.tube_crop_scope == "video" and self.runtime_spatial_crops:
            raise ValueError(
                "tube_crop_scope='video' is incompatible with cached "
                "clip-level spatial crops"
            )
        if self.tube_crop_scope == "video" and self.preprocessed_pose_cache:
            raise ValueError(
                "tube_crop_scope='video' requires source frames rather than "
                "preprocessed frame-level crops"
            )

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
        self.frame_layout = str(frame_layout)
        self.frame_clip_subdir = str(frame_clip_subdir)
        self.exclude_overlapping_clips = bool(exclude_overlapping_clips)

        if self.frame_layout not in {"flat", "setup_contiguous_clips"}:
            raise ValueError(
                "frame_layout must be 'flat' or 'setup_contiguous_clips'; "
                f"got {self.frame_layout!r}"
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
        self._video_tube_bbox_cache: dict[tuple[int, str], np.ndarray] = {}
        self._available_frames_by_sample_clip: dict[
            tuple[int, str], tuple[int, ...]
        ] = {}

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
            "NTU setups:          "
            + (",".join(sorted(self.allowed_setups)) if self.allowed_setups else "full")
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
        print(f"Runtime pose cache:  {self.runtime_cache is not None}")
        print(f"Runtime crop cache:  {self.runtime_spatial_crops}")
        print(f"Dataset heatmaps:    {self.generate_heatmaps}")
        print(f"Skeleton cache:      {self.skeleton_cache_size} videos/worker")

        print(
            f"Extracted frames:    "
            f"{self.extracted_frames_dir}"
        )

        print(
            f"Person crop:         "
            f"{self.person_crop}"
        )

        print(f"Tube crop scope:     {self.tube_crop_scope}")

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
        return validate_pose_file(skeleton_path)

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

                if (
                    self.allowed_setups is not None
                    and sample_id[:4] not in self.allowed_setups
                ):
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

                if (
                    self.validated_sample_ids is not None
                    and sample_id not in self.validated_sample_ids
                ):
                    skipped_rows.append(
                        {
                            "sample_id": sample_id,
                            "reason": "not approved by pose preflight manifest",
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
    ) -> list[tuple[int, int, str | None, int]]:
        # Some repository tests construct a minimal dataset with ``__new__``
        # and call this index builder directly, so initialize derived geometry
        # metadata here as well as in ``__init__``.
        self._available_frames_by_sample_clip = {}
        frame_index: list[
            tuple[int, int, str | None, int]
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

            sample_frame_dir = self._get_sample_frame_dir(sample_id)

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

            if self.frame_layout == "setup_contiguous_clips":
                clip_frames = [
                    (
                        clip_dir.name,
                        sorted(
                            int(path.stem.removeprefix("frame_"))
                            for path in clip_dir.glob("*.jpg")
                            if path.stem.removeprefix("frame_").isdigit()
                        ),
                    )
                    for clip_dir in sorted(sample_frame_dir.glob("clip_*"))
                    if clip_dir.is_dir()
                ]
                if self.exclude_overlapping_clips and any(
                    set(left_frames) & set(right_frames)
                    for left_index, (_, left_frames) in enumerate(clip_frames)
                    for _, right_frames in clip_frames[left_index + 1:]
                ):
                    skipped_missing_frames.append({
                        "sample_id": sample_id,
                        "reason": "overlapping fixed 2x16 clips",
                    })
                    continue
            else:
                frame_numbers = [
                    frame_number
                    for frame_number in range(usable_frames)
                    if (sample_frame_dir / f"frame_{frame_number:06d}.jpg").exists()
                ]

                clip_frames = [(None, frame_numbers)]

            sample_frame_indices = []
            available_frames_by_clip: dict[str, tuple[int, ...]] = {}
            history = (self.temporal_steps - 1) * self.temporal_frame_gap
            for clip_name, clip_frame_numbers in clip_frames:
                selected = clip_frame_numbers[::self.frame_stride]
                available_frames_by_clip[str(clip_name or "full")] = tuple(selected)
                available = set(selected)
                sample_frame_indices.extend(
                    (new_sample_index, target, clip_name, position)
                    for position, target in enumerate(selected)
                    if (
                        self.temporal_steps == 1
                        and self.minimum_temporal_history == 0
                    ) or (
                        target >= max(history, self.minimum_temporal_history)
                        and all(
                            target - offset * self.temporal_frame_gap in available
                            for offset in range(self.temporal_steps)
                        )
                        and all(
                            target - offset in available
                            for offset in range(self.minimum_temporal_history + 1)
                        )
                    )
                )

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

            for clip_key, frame_numbers in available_frames_by_clip.items():
                self._available_frames_by_sample_clip[
                    (new_sample_index, clip_key)
                ] = frame_numbers

            frame_index.extend(
                sample_frame_indices
            )

        self.samples = valid_samples

        self.skipped_samples.extend(
            skipped_missing_frames
        )

        return frame_index

    # ========================================================
    # 8. Skeleton loading and cache
    # ========================================================

    def _load_pose_sequence(
        self,
        skeleton_path: Path,
        body_id: str | None = None,
    ) -> dict:
        cache_key = f"{skeleton_path}::body={body_id or 'primary'}"

        if cache_key in self._skeleton_cache:
            value = self._skeleton_cache.pop(
                cache_key
            )

            self._skeleton_cache[
                cache_key
            ] = value

            return value

        pose_sequence = None
        if self.runtime_cache is not None:
            cached = self.runtime_cache.load_pose(skeleton_path.stem)
            if cached is not None and (
                body_id is None or str(cached.get("body_id")) == str(body_id)
            ):
                pose_sequence = cached
        if pose_sequence is None:
            sequence = read_skeleton_file(
                skeleton_path
            )

            pose_sequence = (
                extract_body_pose_sequence(sequence, body_id)
                if body_id else extract_primary_pose_sequence(sequence)
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

    def _get_sample_frame_dir(self, sample_id: str) -> Path:
        if self.frame_layout == "setup_contiguous_clips":
            return (
                self.extracted_frames_dir
                / sample_id[:4]
                / self.frame_clip_subdir
                / sample_id
            )
        return self.extracted_frames_dir / sample_id

    def _get_frame_path(
        self,
        sample_id: str,
        frame_number: int,
        clip_name: str | None = None,
    ) -> Path:
        sample_dir = self._get_sample_frame_dir(sample_id)
        filename = f"frame_{frame_number:06d}.jpg"
        if self.frame_layout == "setup_contiguous_clips":
            if clip_name is not None:
                return sample_dir / clip_name / filename
            for clip_dir in sorted(sample_dir.glob("clip_*")):
                candidate = clip_dir / filename
                if candidate.is_file():
                    return candidate
            return sample_dir / "clip_01" / filename
        return sample_dir / filename

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
        shared_bbox: np.ndarray | None = None,
        clip_name: str | None = None,
    ) -> dict[str, object]:
        if self.preprocessed_pose_cache:
            return self._get_preprocessed_frame_item(
                sample_index, frame_number, transform_parameters, include_target,
                clip_name,
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
            clip_name=clip_name,
        )

        pose_sequence = self._load_pose_sequence(
            skeleton_path, str(sample.get("body_id", "")).strip() or None,
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

        cached_crop = None
        cached_crop_mode = None
        if self.runtime_cache is not None and self.runtime_spatial_crops:
            cached_crop_mode = self.runtime_cache.spatial_crop_mode(sample_id)
        if (
            self.runtime_cache is not None
            and self.runtime_spatial_crops
            and self.person_crop
            and cached_crop_mode == "clip_tube"
            and shared_bbox is not None
        ):
            cached_crop = self.runtime_cache.load_spatial_frame(
                sample_id, clip_name, frame_number,
            )
            if cached_crop is None:
                raise RuntimeError(
                    f"Required runtime crop is missing: {sample_id} "
                    f"{clip_name or ''} frame={frame_number}"
                )

        if cached_crop is not None:
            image, person_bbox = cached_crop
            image_width = self._safe_int(sample.get("width"), 0)
            image_height = self._safe_int(sample.get("height"), 0)
            expected_bbox = quantize_person_bbox(
                shared_bbox, image_width, image_height,
            )
            if not np.array_equal(person_bbox, expected_bbox):
                raise RuntimeError(
                    f"Cached tube bbox differs from runtime geometry: "
                    f"{sample_id} {clip_name or ''} frame={frame_number}"
                )
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Invalid frame dimensions in metadata: {sample_id}")
        else:
            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Extracted frame not found: {frame_path}"
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
            image_height, image_width = image.shape[:2]

        visibility = (
            coordinate_visibility(
                keypoints,
                tracking_state=tracking_state,
                image_size=(
                    image_width,
                    image_height,
                ),
                include_inferred=False,
            )
        ).astype(
            np.float32
        )

        original_keypoints = keypoints.copy()

        original_visibility = (
            visibility.copy()
        )

        if cached_crop is not None:
            keypoints, visibility = transform_keypoints_to_bbox(
                keypoints, visibility, person_bbox, self.image_size,
            )
        elif self.person_crop:
            crop_result = (
                crop_and_resize_person_with_bbox(
                    image=image, keypoints=keypoints, visibility=visibility,
                    bbox_xyxy=shared_bbox, output_size=self.image_size,
                ) if shared_bbox is not None else crop_and_resize_person(
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

        if self.map_to_mpii16:
            keypoints = map_ntu25_to_mpii16(keypoints)
            visibility = map_ntu25_to_mpii16(visibility[:, None])[:, 0]
            original_keypoints = map_ntu25_to_mpii16(original_keypoints)
            original_visibility = map_ntu25_to_mpii16(original_visibility[:, None])[:, 0]

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

        result = {
            "image": image_tensor,
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

            "sample_id": (
                f"{sample_id}::{clip_name}" if clip_name is not None else sample_id
            ),

            "person_id": str(sample.get("body_id", pose_sequence.get("body_id", "primary"))),

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
        if self.generate_heatmaps:
            heatmaps = generate_gaussian_heatmaps(
                keypoints=keypoints_tensor.detach().cpu().numpy(),
                visibility=visibility_tensor.detach().cpu().numpy(),
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
            )
            result["heatmaps"] = torch.from_numpy(heatmaps).float()
        return result

    def _get_preprocessed_frame_item(
        self,
        sample_index: int,
        frame_number: int,
        transform_parameters: dict[str, float | bool] | None = None,
        include_target: bool = True,
        clip_name: str | None = None,
    ) -> dict[str, object]:
        sample = self.samples[sample_index]
        sample_id = str(sample["sample_id"])
        frame_path = self._get_frame_path(sample_id, frame_number, clip_name)
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
        original_keypoints = pose["original_keypoints"][pose_index].copy()
        original_visibility = pose["original_visibility"][pose_index].copy()
        if self.map_to_mpii16:
            keypoints = map_ntu25_to_mpii16(keypoints)
            visibility = map_ntu25_to_mpii16(visibility[:, None])[:, 0]
            original_keypoints = map_ntu25_to_mpii16(original_keypoints)
            original_visibility = map_ntu25_to_mpii16(original_visibility[:, None])[:, 0]
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
        result = {
            "image": image_tensor,
            "keypoints": keypoints_tensor,
            "keypoints_original": torch.from_numpy(original_keypoints).float(),
            "visibility": visibility_tensor,
            "head_length": torch.tensor(
                float(pose["head_length"][pose_index]), dtype=torch.float32,
            ),
            "sample_id": (
                f"{sample_id}::{clip_name}" if clip_name is not None else sample_id
            ),
            "person_id": str(sample.get("body_id", "primary")),
            "frame_index": frame_number,
            "rgb_path": str(frame_path),
            "person_bbox": torch.from_numpy(pose["person_bbox"][pose_index]).float(),
            "original_keypoints": torch.from_numpy(original_keypoints).float(),
            "original_visibility": torch.from_numpy(original_visibility).float(),
        }
        if self.generate_heatmaps:
            heatmaps = generate_gaussian_heatmaps(
                keypoints=keypoints_tensor.numpy(),
                visibility=visibility_tensor.numpy(),
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
            )
            result["heatmaps"] = torch.from_numpy(heatmaps).float()
        return result

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, target_frame, clip_name, target_position = self.frame_index[index]
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
        shared_bbox = self._shared_temporal_bbox(
            sample_index, frame_numbers, clip_name,
        )
        items = [
            self._get_frame_item(
                sample_index, frame, parameters,
                include_target=(
                    getattr(self, "return_temporal_targets", False)
                    or order == len(frame_numbers) - 1
                ), shared_bbox=shared_bbox,
                clip_name=clip_name,
            )
            for order, frame in enumerate(frame_numbers)
        ]
        target = items[-1]
        if self.return_temporal_sequence:
            target["image"] = torch.stack(
                [item["image"] for item in items], dim=0,
            )
        if getattr(self, "return_temporal_targets", False):
            if self.generate_heatmaps:
                target["temporal_heatmaps"] = torch.stack(
                    [item["heatmaps"] for item in items], dim=0,
                )
            target["temporal_keypoints"] = torch.stack(
                [item["keypoints"] for item in items], dim=0,
            )
            target["temporal_visibility"] = torch.stack(
                [item["visibility"] for item in items], dim=0,
            )
        target["temporal_frame_indices"] = torch.tensor(
            frame_numbers, dtype=torch.int64,
        )
        # Explicit clip identity prevents temporal metrics from inferring
        # boundaries by parsing sample_id strings. Positions are cheap to
        # precompute in _build_frame_index and require no per-item directory IO.
        target["video_id"] = str(self.samples[sample_index]["sample_id"])
        target["clip_id"] = str(clip_name or "full")
        target["frame_position_in_clip"] = int(target_position)
        target["temporal_frame_positions"] = torch.tensor(
            [
                target_position - offset * self.temporal_frame_gap
                for offset in reversed(range(self.temporal_steps))
            ],
            dtype=torch.int64,
        )
        return target

    def _shared_temporal_bbox(
        self, sample_index: int, frame_numbers: list[int],
        clip_name: str | None = None,
    ) -> np.ndarray | None:
        """Resolve the geometry shared by every frame in a temporal sample.

        Cached temporal crops use one bbox computed over the complete prepared
        clip.  That cached bbox is authoritative even when a control consumes
        only a shorter window (for example V2U2); recomputing a window-local
        bbox would no longer match the cached images.
        """
        if (
            not self.person_crop
            or not self.tube_crop
            or len(frame_numbers) <= 1
            or self.preprocessed_pose_cache
        ):
            return None
        sample_id = str(self.samples[sample_index]["sample_id"])
        if getattr(self, "tube_crop_scope", "window") == "video":
            clip_key = str(clip_name or "full")
            cache_key = (sample_index, clip_key)
            if not hasattr(self, "_video_tube_bbox_cache"):
                self._video_tube_bbox_cache = {}
            cached = self._video_tube_bbox_cache.get(cache_key)
            if cached is not None:
                return cached
            video_frames = self._available_frames_by_sample_clip.get(cache_key)
            if not video_frames:
                raise RuntimeError(
                    f"No source frames recorded for video-level tube crop: "
                    f"{sample_id} {clip_key}"
                )
            bbox = self._tube_bbox(sample_index, list(video_frames), clip_name)
            self._video_tube_bbox_cache[cache_key] = bbox
            return bbox
        if (
            self.runtime_cache is not None
            and self.runtime_spatial_crops
            and self.runtime_cache.spatial_crop_mode(sample_id) == "clip_tube"
        ):
            bbox = self.runtime_cache.load_spatial_bbox(
                sample_id, clip_name, frame_numbers[0],
            )
            if bbox is None:
                raise RuntimeError(
                    f"Required runtime bbox is missing: {sample_id} "
                    f"{clip_name or ''} frame={frame_numbers[0]}"
                )
            return bbox
        return self._tube_bbox(sample_index, frame_numbers, clip_name)

    def _tube_bbox(self, sample_index: int, frame_numbers: list[int],
                   clip_name: str | None = None) -> np.ndarray:
        """Compute one union person box and reuse it for every frame in a clip."""
        sample = self.samples[sample_index]
        sample_id = str(sample["sample_id"])
        image_width = self._safe_int(sample.get("width"), 0)
        image_height = self._safe_int(sample.get("height"), 0)
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid frame dimensions in metadata: {sample_id}")
        pose = self._load_pose_sequence(
            Path(sample["skeleton_path"]),
            str(sample.get("body_id", "")).strip() or None,
        )
        points = []
        validities = []
        for frame_number in frame_numbers:
            keypoints = pose["color_xy"][frame_number]
            tracking_state = pose["tracking_state"][frame_number]
            visibility = coordinate_visibility(
                keypoints, tracking_state=tracking_state,
                image_size=(image_width, image_height),
                include_inferred=False,
            )
            points.append(keypoints)
            validities.append(visibility)
        return compute_person_bbox(
            np.concatenate(points, axis=0), np.concatenate(validities, axis=0),
            image_width, image_height, self.bbox_expansion, True,
        )
