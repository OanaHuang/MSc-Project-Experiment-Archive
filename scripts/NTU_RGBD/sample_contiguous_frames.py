from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import tempfile
import time


SOURCE_MARKER = ".frames_complete.json"
SAMPLE_MARKER = ".contiguous_sample_complete.json"


def clip_starts(frame_count: int, clip_length: int, clips: int) -> list[int]:
    """Place clips at uniform temporal centres, overlapping only when necessary."""
    if frame_count < clip_length:
        raise ValueError(
            f"need at least {clip_length} frames for a contiguous clip, "
            f"found {frame_count}"
        )
    starts = [
        round(((2 * index + 1) * frame_count) / (2 * clips) - clip_length / 2)
        for index in range(clips)
    ]
    starts = [max(0, min(start, frame_count - clip_length)) for start in starts]
    required = clip_length * clips
    for previous, current in zip(starts, starts[1:]):
        if frame_count >= required and current < previous + clip_length:
            raise RuntimeError(
                f"computed overlapping clips for {frame_count} frames: {starts}"
            )
    return starts


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def sample_video(
    source_dir: Path, output_root: Path, clip_length: int, clips: int,
) -> dict:
    source_marker = read_json(source_dir / SOURCE_MARKER)
    frame_count = int(source_marker["saved_frames"])
    starts = clip_starts(frame_count, clip_length, clips)
    sample_id = source_dir.name
    output_dir = output_root / sample_id
    expected_clips = []
    for clip_index, start in enumerate(starts):
        expected_clips.append({
            "clip_index": clip_index,
            "start_frame": start,
            "end_frame": start + clip_length - 1,
            "frames": list(range(start, start + clip_length)),
        })
    payload = {
        "sample_id": sample_id,
        "source": str(source_dir.resolve()),
        "source_frames": frame_count,
        "sampling": "uniform_temporal_centres_contiguous_clips",
        "clip_length": clip_length,
        "clips_per_video": clips,
        "saved_frames": clip_length * clips,
        "clips": expected_clips,
    }
    marker_path = output_dir / SAMPLE_MARKER
    if marker_path.is_file() and read_json(marker_path) == payload:
        expected_files = [
            output_dir / f"clip_{clip['clip_index']:02d}"
            / f"frame_{frame_number:06d}.jpg"
            for clip in expected_clips for frame_number in clip["frames"]
        ]
        if all(path.is_file() for path in expected_files):
            return {**payload, "status": "skipped"}
    if output_dir.exists():
        raise RuntimeError(f"incomplete or incompatible output exists: {output_dir}")

    temporary_root = output_root / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"{sample_id}.", dir=temporary_root))
    try:
        for clip in expected_clips:
            clip_dir = temporary_dir / f"clip_{clip['clip_index']:02d}"
            clip_dir.mkdir()
            for frame_number in clip["frames"]:
                source = source_dir / f"frame_{frame_number:06d}.jpg"
                if not source.is_file():
                    raise FileNotFoundError(f"source frame not found: {source}")
                shutil.copy2(source, clip_dir / source.name)
        write_json_atomic(temporary_dir / SAMPLE_MARKER, payload)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return {**payload, "status": "written"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic contiguous clips from decoded NTU frames",
    )
    parser.add_argument("--setup", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--clips-per-video", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_length < 1 or args.clips_per_video < 1 or args.workers < 1:
        raise ValueError("clip length, clips per video, and workers must be positive")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_dirs = sorted(
        path for path in source_root.glob(f"{args.setup}*")
        if path.is_dir() and (path / SOURCE_MARKER).is_file()
    )
    if not source_dirs:
        raise RuntimeError(f"no completed {args.setup} sources found in {source_root}")

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {
            executor.submit(
                sample_video, source_dir, output_root,
                args.clip_length, args.clips_per_video,
            ): source_dir.name
            for source_dir in source_dirs
        }
        for index, future in enumerate(as_completed(pending), 1):
            sample_id = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"sampling failed for {sample_id}: {error}") from error
            if index == 1 or index % 100 == 0 or index == len(source_dirs):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[contiguous] {index}/{len(source_dirs)} "
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
        "videos": len(results),
        "frames": sum(int(item["saved_frames"]) for item in results),
        "written": sum(item["status"] == "written" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "source_root": str(source_root),
        "output_root": str(output_root),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    manifest_path = output_root / "samples.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_root,
        prefix=".samples.", suffix=".tmp", delete=False,
    ) as handle:
        for result in results:
            result = {key: value for key, value in result.items() if key != "status"}
            handle.write(json.dumps(result) + "\n")
        temporary = Path(handle.name)
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
