from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import tempfile
import time

try:
    from scripts.NTU_RGBD.prepare_setup_frames import (
        VIDEO_SUFFIX, sample_id_from_video, write_json_atomic,
    )
    from scripts.NTU_RGBD.sample_contiguous_frames import clip_starts
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.NTU_RGBD.prepare_setup_frames import (
        VIDEO_SUFFIX, sample_id_from_video, write_json_atomic,
    )
    from scripts.NTU_RGBD.sample_contiguous_frames import clip_starts


SAMPLE_MARKER = ".contiguous_sample_complete.json"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_clip(capture, start: int, length: int) -> list[tuple[int, object]]:
    import cv2

    if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise RuntimeError(f"could not seek to frame {start}")
    frames = []
    for frame_number in range(start, start + length):
        success, image = capture.read()
        if not success or image is None:
            raise RuntimeError(f"could not decode frame {frame_number}")
        frames.append((frame_number, image))
    return frames


def extract_video(
    video_path: Path, output_root: Path, clip_length: int, clips: int,
    jpeg_quality: int,
) -> dict:
    import cv2

    sample_id = sample_id_from_video(video_path)
    output_dir = output_root / sample_id
    marker_path = output_dir / SAMPLE_MARKER
    source_bytes = video_path.stat().st_size
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        starts = clip_starts(frame_count, clip_length, clips)
        clip_payloads = [
            {
                "clip_index": index,
                "start_frame": start,
                "end_frame": start + clip_length - 1,
                "frames": list(range(start, start + clip_length)),
            }
            for index, start in enumerate(starts)
        ]
        payload = {
            "sample_id": sample_id,
            "source": str(video_path.resolve()),
            "source_bytes": source_bytes,
            "source_frames": frame_count,
            "sampling": "uniform_temporal_centres_contiguous_clips",
            "clip_length": clip_length,
            "clips_per_video": clips,
            "saved_frames": clip_length * clips,
            "jpeg_quality": jpeg_quality,
            "clips": clip_payloads,
        }
        marker = read_json(marker_path)
        if marker == payload:
            expected_files = [
                output_dir / f"clip_{clip['clip_index']:02d}"
                / f"frame_{number:06d}.jpg"
                for clip in clip_payloads for number in clip["frames"]
            ]
            if all(path.is_file() for path in expected_files):
                return {**payload, "status": "skipped"}
        if output_dir.exists():
            raise RuntimeError(f"incomplete or incompatible output exists: {output_dir}")

        temporary_root = output_root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f"{sample_id}.", dir=temporary_root))
        try:
            for clip in clip_payloads:
                clip_dir = temporary_dir / f"clip_{clip['clip_index']:02d}"
                clip_dir.mkdir()
                decoded = read_clip(capture, clip["start_frame"], clip_length)
                for frame_number, image in decoded:
                    destination = clip_dir / f"frame_{frame_number:06d}.jpg"
                    if not cv2.imwrite(
                        str(destination), image,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    ):
                        raise RuntimeError(f"could not write frame: {destination}")
            write_json_atomic(temporary_dir / SAMPLE_MARKER, payload)
            temporary_dir.replace(output_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return {**payload, "status": "written"}
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deterministic contiguous clips directly from NTU AVI files",
    )
    parser.add_argument("--setup", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--clips-per-video", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_length < 1 or args.clips_per_video < 1 or args.workers < 1:
        raise ValueError("clip length, clips per video, and workers must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg quality must lie between 1 and 100")
    video_root = args.video_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_root.glob(f"{args.setup}*{VIDEO_SUFFIX}"))
    if not videos:
        raise RuntimeError(f"no {args.setup} videos found in {video_root}")

    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                extract_video, video, output_root, args.clip_length,
                args.clips_per_video, args.jpeg_quality,
            ): video.name
            for video in videos
        }
        for index, future in enumerate(as_completed(pending), 1):
            video_name = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"extraction failed for {video_name}: {error}") from error
            if index == 1 or index % 100 == 0 or index == len(videos):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[direct-contiguous] {index}/{len(videos)} "
                    f"({index / elapsed:.2f} videos/s)", flush=True,
                )

    results.sort(key=lambda item: item["sample_id"])
    manifest = {
        "dataset": "ntu_rgbd",
        "setup": args.setup,
        "sampling": "uniform_temporal_centres_contiguous_clips",
        "rule": (
            "Clip centres are fixed at (2k+1)/(2K) of each video; "
            "every clip contains L original consecutive frames with no stride; "
            "clips overlap only when a video is shorter than K*L frames."
        ),
        "clip_length": args.clip_length,
        "clips_per_video": args.clips_per_video,
        "frames_per_video": args.clip_length * args.clips_per_video,
        "jpeg_quality": args.jpeg_quality,
        "videos": len(results),
        "frames": sum(int(item["saved_frames"]) for item in results),
        "written": sum(item["status"] == "written" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "video_root": str(video_root),
        "output_root": str(output_root),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_root,
        prefix=".samples.", suffix=".tmp", delete=False,
    ) as handle:
        for result in results:
            handle.write(json.dumps({
                key: value for key, value in result.items() if key != "status"
            }) + "\n")
        temporary = Path(handle.name)
    temporary.replace(output_root / "samples.jsonl")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
