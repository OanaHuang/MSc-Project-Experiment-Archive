from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

try:
    from scripts.NTU_RGBD.core import (
        coordinate_visibility, extract_primary_pose_sequence, read_skeleton_file,
    )
    from scripts.NTU_RGBD.datasets.ntu_frame_dataset import compute_head_length
    from scripts.NTU_RGBD.datasets.person_crop import (
        compute_person_bbox, crop_and_resize_person_with_bbox,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.NTU_RGBD.core import (
        coordinate_visibility, extract_primary_pose_sequence, read_skeleton_file,
    )
    from scripts.NTU_RGBD.datasets.ntu_frame_dataset import compute_head_length
    from scripts.NTU_RGBD.datasets.person_crop import (
        compute_person_bbox, crop_and_resize_person_with_bbox,
    )


DEFAULT_CLIP_LENGTH = 4
DEFAULT_CLIP_START_GAP = 16
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
MANIFEST_NAME = "manifest.json"


def load_samples(metadata_paths: list[Path]) -> list[dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    for metadata_path in metadata_paths:
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                samples[row["sample_id"]] = row
    return [samples[sample_id] for sample_id in sorted(samples)]


def clip_frame_groups(
    source_dir: Path, usable_frames: int, clip_length: int = DEFAULT_CLIP_LENGTH,
    clip_start_gap: int = DEFAULT_CLIP_START_GAP,
) -> list[tuple[Path, ...]]:
    groups = []
    for start in range(0, max(0, usable_frames - clip_length + 1), clip_start_gap):
        paths = tuple(
            source_dir / f"frame_{frame:06d}.jpg"
            for frame in range(start, start + clip_length)
        )
        if all(path.is_file() for path in paths):
            groups.append(paths)
    return groups


def preprocess_sample(
    row: dict[str, str], source_root: Path, output_root: Path,
    bbox_expansion: float, jpeg_quality: int, clip_length: int = DEFAULT_CLIP_LENGTH,
    clip_start_gap: int = DEFAULT_CLIP_START_GAP,
) -> dict[str, object]:
    sample_id = row["sample_id"]
    source_dir = source_root / sample_id
    usable_frames = min(int(row["rgb_frames"]), int(row["skeleton_frames"]))
    groups = clip_frame_groups(source_dir, usable_frames, clip_length, clip_start_gap)
    if not groups:
        raise RuntimeError(f"No complete {clip_length}-frame clips found for {sample_id}")
    output_dir = output_root / sample_id
    expected = {
        "sample_id": sample_id,
        "clips": len(groups),
        "saved_frames": len(groups) * clip_length,
        "clip_length": clip_length,
        "clip_start_gap": clip_start_gap,
        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "bbox_expansion": bbox_expansion,
        "jpeg_quality": jpeg_quality,
    }
    marker_path = output_dir / f".clip{clip_length}_256_complete.json"
    if marker_path.is_file() and (output_dir / "pose_cache.npz").is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if all(marker.get(key) == value for key, value in expected.items()):
            return {**marker, "status": "skipped"}
    if output_dir.exists():
        raise RuntimeError(
            f"Incomplete or incompatible output already exists: {output_dir}"
        )

    pose = extract_primary_pose_sequence(read_skeleton_file(Path(row["skeleton_path"])))
    temporary_root = output_root / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"{sample_id}.", dir=temporary_root))
    frame_numbers: list[int] = []
    keypoints_all: list[np.ndarray] = []
    visibility_all: list[np.ndarray] = []
    original_keypoints_all: list[np.ndarray] = []
    original_visibility_all: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []
    head_lengths: list[float] = []
    try:
        for paths in groups:
            images = []
            clip_keypoints = []
            clip_visibility = []
            clip_numbers = []
            for frame_path in paths:
                frame_number = int(frame_path.stem.rsplit("_", 1)[-1])
                image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not read {frame_path}")
                keypoints = pose["color_xy"][frame_number].copy()
                tracking = pose["tracking_state"][frame_number].copy()
                visibility = coordinate_visibility(
                    keypoints, tracking_state=tracking,
                    image_size=(image.shape[1], image.shape[0]),
                    include_inferred=False,
                ).astype(np.float32)
                images.append(image)
                clip_keypoints.append(keypoints)
                clip_visibility.append(visibility)
                clip_numbers.append(frame_number)

            shared_bbox = compute_person_bbox(
                np.concatenate(clip_keypoints, axis=0),
                np.concatenate(clip_visibility, axis=0),
                image_width=images[0].shape[1], image_height=images[0].shape[0],
                expansion=bbox_expansion, make_square=True,
            )
            for image, keypoints, visibility, frame_number in zip(
                images, clip_keypoints, clip_visibility, clip_numbers,
            ):
                crop = crop_and_resize_person_with_bbox(
                    image, keypoints, visibility, shared_bbox, IMAGE_SIZE,
                )
                destination = temporary_dir / f"frame_{frame_number:06d}.jpg"
                if not cv2.imwrite(
                    str(destination), crop.image,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                ):
                    raise RuntimeError(f"Could not write {destination}")
                frame_numbers.append(frame_number)
                keypoints_all.append(crop.keypoints.astype(np.float32))
                visibility_all.append(crop.visibility.astype(np.float32))
                original_keypoints_all.append(keypoints.astype(np.float32))
                original_visibility_all.append(visibility.astype(np.float32))
                bboxes.append(crop.bbox_xyxy.astype(np.float32))
                head_lengths.append(compute_head_length(crop.keypoints, crop.visibility))

        np.savez(
            temporary_dir / "pose_cache.npz",
            frame_numbers=np.asarray(frame_numbers, dtype=np.int32),
            keypoints=np.stack(keypoints_all),
            visibility=np.stack(visibility_all),
            original_keypoints=np.stack(original_keypoints_all),
            original_visibility=np.stack(original_visibility_all),
            person_bbox=np.stack(bboxes),
            head_length=np.asarray(head_lengths, dtype=np.float32),
        )
        (temporary_dir / marker_path.name).write_text(
            json.dumps(expected, indent=2), encoding="utf-8",
        )
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return {**expected, "status": "written"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic S010 contiguous 256x256 HPE clips",
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("Datasets/NTU_RGBD/extracted_frames_full"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("Datasets/NTU_RGBD/frames/S010/clip4_256"),
    )
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--bbox-expansion", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--clip-length", type=int, default=DEFAULT_CLIP_LENGTH)
    parser.add_argument("--clip-start-gap", type=int, default=DEFAULT_CLIP_START_GAP)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-samples", type=int,
        help="Process only the first N sorted videos for a smoke benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.clip_length < 1 or args.clip_start_gap < args.clip_length:
        raise ValueError("clip length must be positive and no larger than clip start gap")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must lie between 1 and 100")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.metadata)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        samples = samples[:args.max_samples]
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                preprocess_sample, row, source_root, output_root,
                args.bbox_expansion, args.jpeg_quality,
                args.clip_length, args.clip_start_gap,
            ): row["sample_id"]
            for row in samples
        }
        for index, future in enumerate(as_completed(pending), 1):
            sample_id = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"Preprocessing failed for {sample_id}: {error}") from error
            if index == 1 or index % 100 == 0 or index == len(samples):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[clip{args.clip_length}_256] {index}/{len(samples)} "
                    f"({index / elapsed:.2f} videos/s)", flush=True,
                )

    manifest = {
        "dataset": "ntu_rgbd", "setup": "S010",
        "name": f"clip{args.clip_length}_256",
        "source": str(source_root), "sampling": "uniform_contiguous_clips",
        "clip_length": args.clip_length, "clip_start_gap": args.clip_start_gap,
        "image_size": IMAGE_SIZE, "heatmap_size": HEATMAP_SIZE,
        "bbox": "shared_within_clip", "bbox_expansion": args.bbox_expansion,
        "jpeg_quality": args.jpeg_quality, "samples": len(results),
        "clips": sum(int(item["clips"]) for item in results),
        "frames": sum(int(item["saved_frames"]) for item in results),
        "written": sum(item["status"] == "written" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "metadata": [str(path.resolve()) for path in args.metadata],
    }
    temporary_manifest = output_root / f"{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(output_root / MANIFEST_NAME)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
