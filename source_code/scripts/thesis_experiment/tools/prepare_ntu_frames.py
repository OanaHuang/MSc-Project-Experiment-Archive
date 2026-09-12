from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile


SETUP_PATTERN = re.compile(r"S\d{3}")
VIDEO_SUFFIX = "_rgb.avi"
SAMPLE_MARKER = ".frames_complete.json"
SETUP_MARKER = "setup_complete.json"


def sample_id_from_video(path: Path) -> str:
    if not path.name.endswith(VIDEO_SUFFIX):
        raise ValueError(f"Unexpected NTU RGB filename: {path.name}")
    return path.name[:-len(VIDEO_SUFFIX)]


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d{6})\.jpg", path.name)
    if match is None:
        raise ValueError(f"Unexpected extracted-frame filename: {path.name}")
    return int(match.group(1))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def archive_video_count(archive: Path, setup: str) -> int:
    prefix = f"nturgb+d_rgb/{setup}"
    with zipfile.ZipFile(archive) as handle:
        return sum(
            name.startswith(prefix) and name.endswith(VIDEO_SUFFIX)
            for name in handle.namelist()
        )


def unpack_archive(archive: Path, raw_setup_root: Path, setup: str) -> list[Path]:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    expected = archive_video_count(archive, setup)
    if expected < 1:
        raise RuntimeError(f"Archive contains no {setup} RGB videos: {archive}")
    raw_setup_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unzip", "-n", "-q", str(archive), "-d", str(raw_setup_root)],
        check=True,
    )
    video_root = raw_setup_root / "nturgb+d_rgb"
    videos = sorted(video_root.glob(f"{setup}*{VIDEO_SUFFIX}"))
    if len(videos) != expected:
        raise RuntimeError(
            f"Expected {expected} {setup} videos after unpacking, found {len(videos)}"
        )
    return videos


def clear_incomplete_frames(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("frame_*.jpg"):
        if path.is_file() and not path.is_symlink():
            path.unlink()
    marker = output_dir / SAMPLE_MARKER
    if marker.exists() and marker.is_file() and not marker.is_symlink():
        marker.unlink()


def extract_video(video_path: Path, output_root: Path,
                  jpeg_quality: int) -> dict:
    import cv2

    sample_id = sample_id_from_video(video_path)
    output_dir = output_root / sample_id
    marker_path = output_dir / SAMPLE_MARKER
    source_size = video_path.stat().st_size
    marker = read_json(marker_path)
    if (marker.get("sample_id") == sample_id
            and marker.get("source_bytes") == source_size
            and marker.get("stride") == 1
            and marker.get("jpeg_quality") == jpeg_quality
            and marker.get("saved_frames", 0) > 0):
        return {**marker, "status": "skipped"}

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_incomplete_frames(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    saved = 0
    try:
        while True:
            success, image = capture.read()
            if not success or image is None:
                break
            output_path = output_dir / f"frame_{saved:06d}.jpg"
            if not cv2.imwrite(
                str(output_path), image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Could not write frame: {output_path}")
            saved += 1
    finally:
        capture.release()
    if saved < 1:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    payload = {
        "sample_id": sample_id,
        "source": str(video_path),
        "source_bytes": source_size,
        "stride": 1,
        "jpeg_quality": jpeg_quality,
        "saved_frames": saved,
    }
    write_json_atomic(marker_path, payload)
    return {**payload, "status": "written"}


def copy_sampled_frames(full_sample_dir: Path, sampled_root: Path,
                        stride: int) -> dict:
    if stride < 1:
        raise ValueError("stride must be positive")
    source_marker = read_json(full_sample_dir / SAMPLE_MARKER)
    if source_marker.get("saved_frames", 0) < 1:
        raise RuntimeError(f"Full-frame sample is incomplete: {full_sample_dir}")
    sample_id = full_sample_dir.name
    output_dir = sampled_root / sample_id
    marker_path = output_dir / SAMPLE_MARKER
    expected = (int(source_marker["saved_frames"]) + stride - 1) // stride
    marker = read_json(marker_path)
    if (marker.get("sample_id") == sample_id
            and marker.get("stride") == stride
            and marker.get("source_frames") == source_marker["saved_frames"]
            and marker.get("saved_frames") == expected):
        return {**marker, "status": "skipped"}

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_incomplete_frames(output_dir)
    copied = 0
    for source in sorted(full_sample_dir.glob("frame_*.jpg")):
        if frame_number(source) % stride:
            continue
        shutil.copy2(source, output_dir / source.name)
        copied += 1
    if copied != expected:
        raise RuntimeError(
            f"Expected {expected} sampled frames for {sample_id}, copied {copied}"
        )
    payload = {
        "sample_id": sample_id,
        "source": str(full_sample_dir),
        "source_frames": source_marker["saved_frames"],
        "stride": stride,
        "saved_frames": copied,
    }
    write_json_atomic(marker_path, payload)
    return {**payload, "status": "written"}


def run_parallel(function, items: list[Path], workers: int, label: str,
                 **kwargs) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    executor_type = ProcessPoolExecutor if label == "extract" else ThreadPoolExecutor
    results: list[dict] = []
    started = time.monotonic()
    with executor_type(max_workers=workers) as executor:
        pending = {executor.submit(function, item, **kwargs): item for item in items}
        for index, future in enumerate(as_completed(pending), 1):
            item = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"{label} failed for {item}: {error}") from error
            if index == 1 or index % 25 == 0 or index == len(items):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{label}] {index}/{len(items)} "
                    f"({index / elapsed:.2f} samples/s)", flush=True,
                )
    return results


def validate_setup(root: Path, setup: str, expected_samples: int,
                   stride: int) -> dict:
    sample_dirs = sorted(
        path for path in root.glob(f"{setup}*") if path.is_dir()
    )
    complete = [path for path in sample_dirs if (path / SAMPLE_MARKER).is_file()]
    frames = sum(int(read_json(path / SAMPLE_MARKER).get("saved_frames", 0))
                 for path in complete)
    if len(complete) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} completed samples in {root}, "
            f"found {len(complete)}"
        )
    payload = {
        "setup": setup,
        "stride": stride,
        "samples": len(complete),
        "frames": frames,
    }
    write_json_atomic(root / f".{setup.lower()}_{SETUP_MARKER}", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unpack one NTU setup, extract every frame, and copy a strided set",
    )
    parser.add_argument("--setup", default="S010")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument(
        "--stage", choices=("unpack", "extract", "sample", "all"), default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if SETUP_PATTERN.fullmatch(args.setup) is None:
        raise ValueError("setup must use the S### format")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must lie between 1 and 100")
    dataset_root = args.dataset_root.resolve()
    raw_root = dataset_root / "rgb_videos" / args.setup
    full_root = dataset_root / "extracted_frames_full"
    sampled_root = dataset_root / f"extracted_frames_stride{args.sample_stride}"

    videos = sorted(
        (raw_root / "nturgb+d_rgb").glob(f"{args.setup}*{VIDEO_SUFFIX}")
    )
    if args.stage in ("unpack", "all"):
        videos = unpack_archive(args.archive.resolve(), raw_root, args.setup)
        print(f"[unpack] validated {len(videos)} videos", flush=True)
    if args.stage in ("extract", "sample") and not videos:
        raise RuntimeError(f"No unpacked {args.setup} videos found in {raw_root}")
    expected = len(videos)

    if args.stage in ("extract", "all"):
        run_parallel(
            extract_video, videos, args.extract_workers, "extract",
            output_root=full_root, jpeg_quality=args.jpeg_quality,
        )
        print(json.dumps(validate_setup(full_root, args.setup, expected, 1), indent=2))
    if args.stage in ("sample", "all"):
        samples = sorted(
            path for path in full_root.glob(f"{args.setup}*") if path.is_dir()
        )
        if len(samples) != expected:
            raise RuntimeError(
                f"Expected {expected} full-frame sample directories, found {len(samples)}"
            )
        run_parallel(
            copy_sampled_frames, samples, args.copy_workers, "sample",
            sampled_root=sampled_root, stride=args.sample_stride,
        )
        print(json.dumps(
            validate_setup(sampled_root, args.setup, expected, args.sample_stride),
            indent=2,
        ))


if __name__ == "__main__":
    main()
