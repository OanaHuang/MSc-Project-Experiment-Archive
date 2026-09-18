#!/usr/bin/env python3
"""Prepare fixed NTU test videos and render prediction-only pose overlays."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE_ROOT = Path(os.environ.get(
    "NTU_ARCHIVE_ROOT",
    str(PROJECT_ROOT / "Datasets/NTU_RGBD/archives"),
))
THESIS_SRC = PROJECT_ROOT / "scripts" / "thesis_experiment" / "src"
if str(THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(THESIS_SRC))

SETUPS = tuple(f"S{index:03d}" for index in range(1, 18))
VIDEO_SUFFIX = "_rgb.avi"
FRAME_MARKER = ".frames_complete.json"
PCKHB_HEAD_INDEX = 9
PCKHB_NECK_INDEX = 8
PCKHN_CENTER_TO_HEAD_AXIS_RATIO = 0.75
PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO = 0.75
PCKHB_BOX_COLOR_BGR = (40, 220, 40)

# OpenCV uses BGR.  The palette follows the high-contrast qualitative style
# commonly used for MPII poses: each anatomical branch has a stable colour,
# while the centre line progresses from green through orange to red.
MPII_EDGE_COLORS_BGR = (
    (255, 255, 0), (255, 255, 0), (0, 220, 0),       # right leg -> pelvis
    (0, 220, 0), (255, 0, 255), (255, 0, 255),       # pelvis -> left leg
    (0, 165, 255), (0, 80, 255), (0, 0, 255),        # torso and head
    (0, 255, 255), (0, 255, 255), (0, 255, 255),     # right arm
    (255, 190, 0), (255, 190, 0), (255, 190, 0),     # left arm
)
MPII_JOINT_COLORS_BGR = (
    (255, 255, 0), (255, 255, 0), (255, 255, 0),
    (255, 0, 255), (255, 0, 255), (255, 0, 255),
    (0, 220, 0), (0, 165, 255), (0, 80, 255), (0, 0, 255),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (255, 190, 0), (255, 190, 0), (255, 190, 0),
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _localized_test_metadata(prepared_root: Path) -> tuple[Path, list[dict]]:
    """Resolve prepared assets below the current machine's prepared root."""
    source_path = prepared_root / "selected_test_metadata.csv"
    rows = _read_csv(source_path)
    localized = []
    for row in rows:
        value = dict(row)
        setup = f"S{int(row['setup']):03d}"
        sample_id = row["sample_id"]
        value["rgb_path"] = str(
            prepared_root / "source_videos" / setup / f"{sample_id}{VIDEO_SUFFIX}"
        )
        value["skeleton_path"] = str(
            prepared_root / "ground_truth" / setup / f"{sample_id}.skeleton"
        )
        value["full_frames_dir"] = str(prepared_root / "full_frames" / sample_id)
        localized.append(value)
    localized_path = prepared_root / "selected_test_metadata_local.csv"
    if localized:
        _write_csv(localized_path, localized, list(localized[0]))
    return localized_path, localized


def _selected_rows(rows: list[dict], sample_ids: list[str]) -> list[dict]:
    """Return an explicit incremental subset while preserving request order."""
    if not sample_ids:
        return rows
    requested = list(dict.fromkeys(sample_ids))
    by_sample = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in requested if sample_id not in by_sample]
    if missing:
        raise ValueError(f"Unknown sample IDs in prepared metadata: {missing}")
    return [by_sample[sample_id] for sample_id in requested]


def _merge_video_entries(
    manifest_path: Path, new_videos: list[dict], metadata_rows: list[dict],
) -> list[dict]:
    """Merge incremental video outputs into an existing render manifest."""
    merged = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged.update({video["sample_id"]: video for video in existing.get("videos", [])})
    merged.update({video["sample_id"]: video for video in new_videos})
    return [
        merged[row["sample_id"]]
        for row in metadata_rows
        if row["sample_id"] in merged
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _archive_member(archive: zipfile.ZipFile, sample_id: str) -> str | None:
    filename = f"{sample_id}{VIDEO_SUFFIX}"
    matches = [name for name in archive.namelist() if Path(name).name == filename]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate archive members for {sample_id}: {matches}")
    return matches[0] if matches else None


def _qualified_rows(
    rows: list[dict], setup: str, archive_path: Path, project_root: Path,
) -> list[tuple[dict, str]]:
    candidates = [
        row for row in rows
        if f"S{int(row['setup']):03d}" == setup
        and int(row.get("frame_difference", -1)) == 0
        and _is_true(row.get("is_single_person", False))
        and _resolve_project_path(row["skeleton_path"], project_root).is_file()
    ]
    qualified: list[tuple[dict, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for row in sorted(candidates, key=lambda item: item["sample_id"]):
            member = _archive_member(archive, row["sample_id"])
            if member is not None:
                qualified.append((row, member))
    return qualified


def _extract_member(
    archive_path: Path, member: str, output_path: Path,
) -> zipfile.ZipInfo:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if output_path.is_file() and output_path.stat().st_size == info.file_size:
            return info
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise RuntimeError(
                f"Incomplete extraction for {member}: "
                f"{temporary.stat().st_size} != {info.file_size}"
            )
        temporary.replace(output_path)
        return info


def _extract_frames(video_path: Path, frame_dir: Path, jpeg_quality: int) -> dict:
    import cv2

    marker_path = frame_dir / FRAME_MARKER
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            marker = {}
        saved = int(marker.get("saved_frames", 0))
        if (
            marker.get("source_bytes") == video_path.stat().st_size
            and marker.get("jpeg_quality") == jpeg_quality
            and saved > 0
            and len(list(frame_dir.glob("frame_*.jpg"))) == saved
        ):
            return {**marker, "status": "reused"}

    frame_dir.mkdir(parents=True, exist_ok=True)
    for path in frame_dir.glob("frame_*.jpg"):
        path.unlink()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            target = frame_dir / f"frame_{saved:06d}.jpg"
            if not cv2.imwrite(
                str(target), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise RuntimeError(f"Could not write frame: {target}")
            saved += 1
    finally:
        capture.release()
    if saved < 1:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    marker = {
        "source": str(video_path),
        "source_bytes": video_path.stat().st_size,
        "stride": 1,
        "jpeg_quality": jpeg_quality,
        "saved_frames": saved,
        "fps": fps,
        "width": width,
        "height": height,
    }
    _write_json(marker_path, marker)
    return {**marker, "status": "written"}


def prepare(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    metadata_path = args.test_metadata.resolve()
    archive_root = args.archive_root.resolve()
    rows = _read_csv(metadata_path)
    rng = random.Random(args.seed)
    overrides = {}
    for value in args.override:
        if "=" not in value:
            raise ValueError(f"Override must be SETUP=SAMPLE_ID, got {value!r}")
        setup, sample_id = value.split("=", 1)
        setup = setup.upper()
        if setup not in SETUPS:
            raise ValueError(f"Unknown setup in override: {setup}")
        overrides[setup] = sample_id
    selected: list[tuple[str, dict, Path, str]] = []
    for setup in SETUPS:
        archive_path = archive_root / f"nturgbd_rgb_s{int(setup[1:]):03d}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing NTU archive: {archive_path}")
        candidates = _qualified_rows(rows, setup, archive_path, project_root)
        if not candidates:
            raise RuntimeError(f"No exact-frame test candidates for {setup}")
        row, member = rng.choice(candidates)
        if setup in overrides:
            matches = [item for item in candidates if item[0]["sample_id"] == overrides[setup]]
            if len(matches) != 1:
                raise ValueError(
                    f"Override {overrides[setup]} is not a qualified official "
                    f"test sample for {setup}"
                )
            row, member = matches[0]
        selected.append((setup, row, archive_path, member))

    prepared_rows: list[dict] = []
    started = time.monotonic()
    for index, (setup, row, archive_path, member) in enumerate(selected, 1):
        sample_id = row["sample_id"]
        source_video = output_root / "source_videos" / setup / f"{sample_id}{VIDEO_SUFFIX}"
        copied_skeleton = output_root / "ground_truth" / setup / f"{sample_id}.skeleton"
        frame_dir = output_root / "full_frames" / sample_id
        info = _extract_member(archive_path, member, source_video)
        copied_skeleton.parent.mkdir(parents=True, exist_ok=True)
        skeleton_source = _resolve_project_path(row["skeleton_path"], project_root)
        shutil.copy2(skeleton_source, copied_skeleton)
        frame_report = _extract_frames(source_video, frame_dir, args.jpeg_quality)
        decoded = int(frame_report["saved_frames"])
        expected_rgb = int(row["rgb_frames"])
        expected_skeleton = int(row["skeleton_frames"])
        if decoded != expected_rgb or decoded != expected_skeleton:
            raise RuntimeError(
                f"Frame mismatch for {sample_id}: decoded={decoded}, "
                f"rgb={expected_rgb}, skeleton={expected_skeleton}"
            )
        prepared = {
            **row,
            "rgb_path": str(source_video),
            "skeleton_path": str(copied_skeleton),
            "archive_path": str(archive_path),
            "archive_member": member,
            "archive_crc32": f"{info.CRC:08x}",
            "source_video_sha256": _sha256(source_video),
            "full_frames_dir": str(frame_dir),
            "selection_seed": args.seed,
        }
        prepared_rows.append(prepared)
        _write_json(output_root / "samples" / f"{sample_id}.json", prepared)
        print(
            f"[prepare] {index:02d}/17 {setup} {sample_id} "
            f"frames={decoded} status={frame_report['status']}",
            flush=True,
        )

    original_fields = list(rows[0].keys())
    extra_fields = [
        "archive_path", "archive_member", "archive_crc32",
        "source_video_sha256", "full_frames_dir", "selection_seed",
    ]
    fields = original_fields + [field for field in extra_fields if field not in original_fields]
    _write_csv(output_root / "selected_test_metadata.csv", prepared_rows, fields)
    _write_csv(output_root / "selected_test_videos.csv", prepared_rows, fields)
    summary = {
        "source_test_metadata": str(metadata_path),
        "selection_seed": args.seed,
        "selection_policy": (
            "one deterministic random single-person exact-frame video per setup; "
            "selection is independent of model predictions; explicit replacements="
            f"{overrides}"
        ),
        "setups": list(SETUPS),
        "videos": len(prepared_rows),
        "total_frames": sum(int(row["rgb_frames"]) for row in prepared_rows),
        "elapsed_seconds": time.monotonic() - started,
        "sample_ids": [row["sample_id"] for row in prepared_rows],
    }
    _write_json(output_root / "preparation_manifest.json", summary)
    (output_root / "README.md").write_text(
        "# NTU Test Video Visualization\n\n"
        "This directory contains one deterministically selected official "
        "Cross-Subject test video from each NTU60 setup S001-S017. Frames are "
        "complete, native-resolution, stride-1 JPEG decodes of the archived AVI.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _restore_coordinates(points, bbox, image_size: int):
    import numpy as np

    value = np.asarray(points, dtype=np.float32).copy()
    box = np.asarray(bbox, dtype=np.float32)
    value[..., 0] = box[0] + value[..., 0] * (box[2] - box[0]) / image_size
    value[..., 1] = box[1] + value[..., 1] * (box[3] - box[1]) / image_size
    return value


def _draw_prediction_pose(image, points, confidence, thickness: int) -> None:
    import cv2
    import numpy as np
    from spikepose_thesis.data.mpii.core.constants import MPII_SKELETON_EDGES

    points = np.asarray(points)
    confidence = np.asarray(confidence)
    visible = (confidence > 0) & np.isfinite(points).all(axis=1)
    if len(MPII_SKELETON_EDGES) != len(MPII_EDGE_COLORS_BGR):
        raise RuntimeError("MPII edge palette does not cover the skeleton")
    if len(points) != len(MPII_JOINT_COLORS_BGR):
        raise ValueError(
            f"Expected {len(MPII_JOINT_COLORS_BGR)} MPII joints, found {len(points)}"
        )
    for (left, right), color in zip(MPII_SKELETON_EDGES, MPII_EDGE_COLORS_BGR):
        if visible[left] and visible[right]:
            cv2.line(
                image,
                tuple(np.rint(points[left]).astype(int)),
                tuple(np.rint(points[right]).astype(int)),
                color, thickness, cv2.LINE_AA,
            )
    joint_radius = max(2, thickness + 1)
    for point, is_visible, color in zip(points, visible, MPII_JOINT_COLORS_BGR):
        if is_visible:
            centre = tuple(np.rint(point).astype(int))
            cv2.circle(
                image, centre, joint_radius + 1, (0, 0, 0), -1, cv2.LINE_AA,
            )
            cv2.circle(
                image, centre, joint_radius, color, -1, cv2.LINE_AA,
            )


def _pckhb_proxy_headbox(pose, visibility):
    """Estimate an oriented display-only head box from NTU centre joints.

    The legacy PCKHN conversion defines the head-centre--neck-centre distance
    as 0.75 times the head-top--upper-neck distance at r=0.25.  We invert that
    relation, place the rendered head centre at the box centre, place the
    estimated upper neck at the lower-edge midpoint, and leave the rendered
    neck centre 0.25
    box lengths beyond that edge.  Box width is 0.75 times its length.  This
    geometry is only for visualization; PCK-HB/PCKHN evaluation is unchanged.
    """
    import numpy as np

    points = np.asarray(pose, dtype=np.float32)
    visible = np.asarray(visibility) > 0
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pose must have shape [J, 2]")
    if visible.shape != (len(points),):
        raise ValueError("visibility must have shape [J]")
    if len(points) <= max(PCKHB_HEAD_INDEX, PCKHB_NECK_INDEX):
        raise ValueError("pose does not contain PCK-HB head/neck joints")
    if not (visible[PCKHB_HEAD_INDEX] and visible[PCKHB_NECK_INDEX]):
        return None
    head_center = points[PCKHB_HEAD_INDEX]
    neck_center = points[PCKHB_NECK_INDEX]
    if not (np.isfinite(head_center).all() and np.isfinite(neck_center).all()):
        return None
    head_neck_distance = float(np.linalg.norm(head_center - neck_center))
    if not np.isfinite(head_neck_distance) or head_neck_distance <= 1e-6:
        return None
    box_length = head_neck_distance / PCKHN_CENTER_TO_HEAD_AXIS_RATIO
    box_width = PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO * box_length
    long_axis = (neck_center - head_center) / head_neck_distance
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float32)
    half_length = 0.5 * box_length * long_axis
    head_top = head_center - half_length
    upper_neck = head_center + half_length
    half_width = 0.5 * box_width * short_axis
    return np.stack(
        (
            head_top - half_width,
            head_top + half_width,
            upper_neck + half_width,
            upper_neck - half_width,
        ),
        axis=0,
    )


def _draw_dashed_line(image, start, end, color, thickness: int) -> None:
    import cv2
    import numpy as np

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        return
    direction = delta / length
    dash = float(max(5, 4 * thickness))
    gap = float(max(3, 2 * thickness))
    offset = 0.0
    while offset < length:
        segment_end = min(offset + dash, length)
        left = tuple(np.rint(start + direction * offset).astype(int))
        right = tuple(np.rint(start + direction * segment_end).astype(int))
        cv2.line(image, left, right, color, thickness, cv2.LINE_AA)
        offset += dash + gap


def _draw_pckhb_proxy_headbox(
    image, pose, visibility, thickness: int,
) -> bool:
    import numpy as np

    box = _pckhb_proxy_headbox(pose, visibility)
    if box is None:
        return False
    height, width = image.shape[:2]
    corners = np.asarray(box, dtype=np.float32).copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    corners = tuple(tuple(point) for point in corners)
    for start, end in zip(corners, corners[1:] + corners[:1]):
        _draw_dashed_line(
            image, start, end, PCKHB_BOX_COLOR_BGR, max(1, thickness),
        )
    return True


def _video_writer(output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import cv2

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open OpenCV video writer: {output_path}")
        return {"kind": "opencv", "writer": writer}
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8g}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Could not open ffmpeg input pipe")
    return {"kind": "ffmpeg", "process": process}


def _write_video_frame(writer: dict, image) -> None:
    if writer["kind"] == "opencv":
        writer["writer"].write(image)
    else:
        writer["process"].stdin.write(image.tobytes())


def _close_video_writer(writer: dict) -> None:
    if writer["kind"] == "opencv":
        writer["writer"].release()
        return
    process = writer["process"]
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with code {return_code}")


def _render_video(
    row: dict, predictions: dict[int, dict], output_path: Path, model_label: str,
    line_thickness: int, draw_pckhb_headbox: bool = False,
) -> dict:
    import cv2
    import numpy as np

    frame_dir = Path(row["full_frames_dir"])
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if len(frame_paths) != int(row["rgb_frames"]):
        raise RuntimeError(f"Incomplete full frames for {row['sample_id']}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {frame_paths[0]}")
    height, width = first.shape[:2]
    fps = float(row.get("fps", 30.0))
    writer = _video_writer(output_path, fps, width, height)
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {frame_path}")
            value = predictions.get(frame_index)
            if value is None:
                raise RuntimeError(
                    f"Missing prediction for {row['sample_id']} frame {frame_index}"
                )
            _draw_prediction_pose(
                image, value["prediction"], value["confidence"], line_thickness,
            )
            if draw_pckhb_headbox:
                _draw_pckhb_proxy_headbox(
                    image, value["prediction"], value["confidence"],
                    line_thickness,
                )
            _write_video_frame(writer, image)
    finally:
        _close_video_writer(writer)
    return {
        "sample_id": row["sample_id"],
        "setup": f"S{int(row['setup']):03d}",
        "frames": len(frame_paths),
        "fps": fps,
        "width": width,
        "height": height,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }


def render(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from spikepose_thesis.data.ntu.datasets.dataset import build_dataset
    from spikepose_thesis.evaluation.mpii.metrics import prediction_to_keypoints
    from spikepose_thesis.models import build_model
    from spikepose_thesis.training.checkpoint import load_checkpoint

    project_root = args.project_root.resolve()
    prepared_root = args.prepared_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    metadata_path, metadata_rows = _localized_test_metadata(prepared_root)
    rows = _selected_rows(metadata_rows, args.sample_id)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = deepcopy(checkpoint["config"])
    config["data"].update({
        "frames_dir": str(prepared_root / "full_frames"),
        "frame_layout": "flat",
        "frame_stride": 1,
        "minimum_temporal_history": int(config["model"]["num_steps"]) - 1,
        "preprocessed_pose_cache": False,
        "runtime_cache_mode": "disabled",
        "runtime_spatial_crops": False,
        "pose_validation_mode": "strict",
        "exclude_overlapping_clips": False,
        "setup_filter": None,
        "train_metadata": str(metadata_path),
        "validation_metadata": str(metadata_path),
        "test_metadata": str(metadata_path),
    })
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = build_dataset(project_root, config, "test")
    indices_by_sample: dict[str, list[int]] = defaultdict(list)
    for dataset_index, (sample_index, _, _, _) in enumerate(dataset.frame_index):
        indices_by_sample[str(dataset.samples[sample_index]["sample_id"])].append(
            dataset_index
        )
    expected_ids = {row["sample_id"] for row in metadata_rows}
    if set(indices_by_sample) != expected_ids:
        raise RuntimeError(
            f"Dataset/sample mismatch: dataset={sorted(indices_by_sample)} "
            f"metadata={sorted(expected_ids)}"
        )
    indices_by_sample = {
        row["sample_id"]: indices_by_sample[row["sample_id"]] for row in rows
    }

    decoder = config.get("evaluation", {}).get("main_decoder", "dark")
    image_size = int(config["data"]["image_size"])
    predictions: dict[str, dict[int, dict]] = defaultdict(dict)
    first_indices: set[int] = set()
    started = time.monotonic()
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            first_index = indices_by_sample[sample_id][0]
            first_indices.add(first_index)
            item = dataset[first_index]
            images = item["image"].unsqueeze(0).to(device)
            per_step = model.forward_per_step(images)[:, 0]
            coordinates, confidence = prediction_to_keypoints(
                per_step, image_size, decoder=decoder,
            )
            bbox = item["person_bbox"].numpy()
            restored = _restore_coordinates(coordinates, bbox, image_size)
            gt = _restore_coordinates(
                item["temporal_keypoints"].numpy(), bbox, image_size,
            )
            frames = item["temporal_frame_indices"].numpy()
            visibility = item["temporal_visibility"].numpy()
            for step, frame in enumerate(frames):
                predictions[sample_id][int(frame)] = {
                    "prediction": restored[step],
                    "confidence": confidence[step],
                    "gt": gt[step],
                    "visibility": visibility[step],
                }
            print(
                f"[infer-warmup] {number:02d}/{len(rows):02d} {sample_id} "
                f"frames={int(frames[0])}-{int(frames[-1])}",
                flush=True,
            )

        remaining = [
            index for indices in indices_by_sample.values() for index in indices
            if index not in first_indices
        ]
        loader = DataLoader(
            Subset(dataset, remaining), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        processed = 0
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            coordinates, confidence = prediction_to_keypoints(
                output, image_size, decoder=decoder,
            )
            bboxes = batch["person_bbox"].numpy()
            for item_index, sample_id in enumerate(batch["video_id"]):
                frame = int(batch["frame_index"][item_index])
                predictions[str(sample_id)][frame] = {
                    "prediction": _restore_coordinates(
                        coordinates[item_index], bboxes[item_index], image_size,
                    ),
                    "confidence": confidence[item_index],
                    "gt": batch["original_keypoints"][item_index].numpy(),
                    "visibility": batch["original_visibility"][item_index].numpy(),
                }
            processed += len(batch["video_id"])
            if processed == len(batch["video_id"]) or processed % 200 < len(batch["video_id"]):
                print(f"[infer] {processed}/{len(remaining)}", flush=True)

    model_root = prepared_root / "models" / config["id"]
    prediction_root = model_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = row["sample_id"]
        values = predictions[sample_id]
        ordered = sorted(values)
        np.savez_compressed(
            prediction_root / f"{sample_id}.npz",
            frame_index=np.asarray(ordered, dtype=np.int64),
            prediction=np.asarray([values[index]["prediction"] for index in ordered]),
            confidence=np.asarray([values[index]["confidence"] for index in ordered]),
            ground_truth=np.asarray([values[index]["gt"] for index in ordered]),
            visibility=np.asarray([values[index]["visibility"] for index in ordered]),
        )

    draw_pckhb_headbox = bool(args.draw_pckhb_headbox)
    video_root = model_root / (
        "videos_prediction_pckhb_headbox"
        if draw_pckhb_headbox else "videos_prediction_only"
    )
    videos = []
    for index, row in enumerate(rows, 1):
        suffix = "prediction_pckhb_headbox" if draw_pckhb_headbox else "prediction"
        output_path = video_root / f"{row['sample_id']}_{suffix}.mp4"
        videos.append(_render_video(
            row, predictions[row["sample_id"]], output_path, config["id"],
            args.line_thickness, draw_pckhb_headbox,
        ))
        print(
            f"[render] {index:02d}/{len(rows):02d} {row['sample_id']} "
            f"bytes={videos[-1]['bytes']}",
            flush=True,
        )
    manifest_path = model_root / (
        "render_manifest_prediction_pckhb_headbox.json"
        if draw_pckhb_headbox else "render_manifest_prediction_only.json"
    )
    videos = _merge_video_entries(manifest_path, videos, metadata_rows)
    manifest = {
        "experiment_id": config["id"],
        "model_name": config.get("name", config["id"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "decoder": decoder,
        "device": str(device),
        "full_frame_policy": (
            "frames 0-15 use causal per-step predictions from the first 16-frame "
            "window; later frames use the final prediction from each rolling "
            "16-frame window"
        ),
        "overlay_style": {
            "content": (
                "prediction_with_pose_derived_pckhb_proxy_box"
                if draw_pckhb_headbox else "prediction_only"
            ),
            "palette": "anatomical_multicolor_mpii16",
            "line_thickness_px": args.line_thickness,
            "joint_radius_px": max(2, args.line_thickness + 1),
            "joint_outline": "black_1px",
            "ground_truth_pose": False,
            "bounding_box": bool(draw_pckhb_headbox),
            "bounding_box_source": (
                "rendered_prediction_head_and_neck_centres"
                if draw_pckhb_headbox else None
            ),
            "bounding_box_definition": (
                "estimated_oriented_head_box; head_top--upper_neck_length="
                "prediction_head_centre--neck_centre_distance/0.75; "
                "head_centre_at_box_centre; upper_neck_at_lower_edge_midpoint; "
                "neck_centre=upper_neck+0.25*length; width=0.75*length"
                if draw_pckhb_headbox else None
            ),
            "text_overlay": False,
        },
        "videos": videos,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def rerender(args: argparse.Namespace) -> None:
    """Rebuild overlays from saved prediction matrices without re-running inference."""
    import numpy as np

    prepared_root = args.prepared_root.resolve()
    _, metadata_rows = _localized_test_metadata(prepared_root)
    rows = _selected_rows(metadata_rows, args.sample_id)
    model_root = prepared_root / "models" / args.model_id
    prediction_root = model_root / "predictions"
    draw_pckhb_headbox = bool(args.draw_pckhb_headbox)
    video_root = model_root / (
        "videos_prediction_pckhb_headbox"
        if draw_pckhb_headbox else "videos_prediction_only"
    )
    videos = []
    for number, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        archive = np.load(prediction_root / f"{sample_id}.npz")
        predictions = {}
        for offset, frame_index in enumerate(archive["frame_index"]):
            predictions[int(frame_index)] = {
                "prediction": archive["prediction"][offset],
                "confidence": archive["confidence"][offset],
            }
        suffix = "prediction_pckhb_headbox" if draw_pckhb_headbox else "prediction"
        output_path = video_root / f"{sample_id}_{suffix}.mp4"
        videos.append(_render_video(
            row, predictions, output_path, args.model_id, args.line_thickness,
            draw_pckhb_headbox,
        ))
        print(f"[rerender] {number:02d}/{len(rows):02d} {sample_id}", flush=True)
    source_manifest_path = model_root / "render_manifest_prediction_only.json"
    if not source_manifest_path.is_file():
        source_manifest_path = model_root / "render_manifest.json"
    manifest = json.loads(source_manifest_path.read_text())
    manifest["videos"] = videos
    manifest["overlay_style"] = {
        "content": (
            "prediction_with_pose_derived_pckhb_proxy_box"
            if draw_pckhb_headbox else "prediction_only"
        ),
        "palette": "anatomical_multicolor_mpii16",
        "line_thickness_px": args.line_thickness,
        "joint_radius_px": max(2, args.line_thickness + 1),
        "joint_outline": "black_1px",
        "ground_truth_pose": False,
        "bounding_box": bool(draw_pckhb_headbox),
        "bounding_box_source": (
            "rendered_prediction_head_and_neck_centres"
            if draw_pckhb_headbox else None
        ),
        "bounding_box_definition": (
            "estimated_oriented_head_box; head_top--upper_neck_length="
            "prediction_head_centre--neck_centre_distance/0.75; "
            "head_centre_at_box_centre; upper_neck_at_lower_edge_midpoint; "
            "neck_centre=upper_neck+0.25*length; width=0.75*length"
            if draw_pckhb_headbox else None
        ),
        "text_overlay": False,
    }
    manifest["source_render_manifest"] = str(source_manifest_path)
    output_manifest = model_root / (
        "render_manifest_prediction_pckhb_headbox.json"
        if draw_pckhb_headbox else "render_manifest_prediction_only.json"
    )
    _write_json(output_manifest, manifest)


def rerender_ground_truth(args: argparse.Namespace) -> None:
    """Rebuild GT skeleton videos from saved arrays without model inference."""
    import numpy as np

    prepared_root = args.prepared_root.resolve()
    metadata_path, metadata_rows = _localized_test_metadata(prepared_root)
    rows = _selected_rows(metadata_rows, args.sample_id)
    prediction_root = (
        prepared_root / "models" / args.source_model_id / "predictions"
    )
    video_root = prepared_root / "ground_truth_videos_skeleton_pckhb_headbox"
    manifest_path = prepared_root / "render_manifest_ground_truth_pckhb_headbox.json"
    videos = []
    for number, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        with np.load(prediction_root / f"{sample_id}.npz") as archive:
            frame_indices = archive["frame_index"].astype(np.int64)
            ground_truth = archive["ground_truth"].copy()
            visibility = archive["visibility"].copy()
        poses = {
            int(frame_index): {
                "prediction": ground_truth[offset],
                "confidence": visibility[offset],
            }
            for offset, frame_index in enumerate(frame_indices)
        }
        output_path = video_root / f"{sample_id}_ground_truth_pckhb_headbox.mp4"
        videos.append(_render_video(
            row, poses, output_path, "ground_truth", args.line_thickness, True,
        ))
        print(
            f"[rerender-ground-truth] {number:02d}/{len(rows):02d} {sample_id}",
            flush=True,
        )

    videos = _merge_video_entries(manifest_path, videos, metadata_rows)
    width_ratio = PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO
    manifest = {
        "artifact_id": "ntu_test_ground_truth_pckhb_headbox",
        "source_metadata": str(metadata_path),
        "source_ground_truth": "saved NTU RGB+D skeleton annotations",
        "source_prediction_archives": str(prediction_root),
        "overlay_style": {
            "content": "ground_truth_pose_with_pose_derived_oriented_headbox",
            "pose": "ground_truth",
            "palette": "anatomical_multicolor_mpii16",
            "line_thickness_px": args.line_thickness,
            "joint_radius_px": max(2, args.line_thickness + 1),
            "joint_outline": "black_1px",
            "bounding_box": True,
            "bounding_box_style": "green_dashed",
            "bounding_box_source": "ground_truth_ntu_head_and_neck_centres",
            "bounding_box_definition": (
                "estimated oriented head box; head-top--upper-neck length="
                "GT head-centre--neck-centre distance/0.75; head centre at box "
                "centre; upper neck at lower-edge midpoint; neck centre=upper "
                f"neck+0.25*length; width={width_ratio:g}*length; width axis "
                "perpendicular to head-centre--neck-centre"
            ),
            "metric_definition_changed": False,
            "model_prediction": False,
            "text_overlay": False,
        },
        "videos": videos,
    }
    _write_json(manifest_path, manifest)


def one_euro(args: argparse.Namespace) -> None:
    """Apply a causal One-Euro filter to saved full-frame source predictions."""
    import numpy as np

    from spikepose_thesis.refinement.filters import OneEuroFilter

    prepared_root = args.prepared_root.resolve()
    _, metadata_rows = _localized_test_metadata(prepared_root)
    rows = _selected_rows(metadata_rows, args.sample_id)
    source_root = prepared_root / "models" / args.source_model_id
    source_prediction_root = source_root / "predictions"
    source_manifest_path = source_root / "render_manifest_prediction_only.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing source render manifest: {source_manifest_path}"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    model_root = prepared_root / "models" / args.model_id
    prediction_root = model_root / "predictions"
    video_root = model_root / "videos_prediction_only"
    prediction_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    videos = []
    for number, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        source_path = source_prediction_root / f"{sample_id}.npz"
        with np.load(source_path) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        frame_indices = arrays["frame_index"].astype(np.int64)
        expected = np.arange(int(row["rgb_frames"]), dtype=np.int64)
        if not np.array_equal(frame_indices, expected):
            raise ValueError(
                f"Source predictions are not complete contiguous frames for {sample_id}"
            )
        frequency = float(row.get("fps", 30.0))
        refined = OneEuroFilter(
            args.min_cutoff, args.beta, args.derivative_cutoff, frequency,
        )(arrays["prediction"])
        arrays["prediction"] = refined
        np.savez_compressed(prediction_root / f"{sample_id}.npz", **arrays)
        predictions = {
            int(frame_index): {
                "prediction": refined[offset],
                "confidence": arrays["confidence"][offset],
            }
            for offset, frame_index in enumerate(frame_indices)
        }
        output_path = video_root / f"{sample_id}_prediction.mp4"
        videos.append(_render_video(
            row, predictions, output_path, args.model_id, args.line_thickness,
        ))
        print(
            f"[one-euro] {number:02d}/{len(rows):02d} {sample_id}", flush=True,
        )

    manifest_path = model_root / "render_manifest_prediction_only.json"
    videos = _merge_video_entries(manifest_path, videos, metadata_rows)
    manifest = {
        "experiment_id": args.model_id,
        "model_name": "One Euro",
        "method": "one_euro",
        "source_experiment_id": source_manifest["experiment_id"],
        "source_checkpoint": source_manifest["checkpoint"],
        "source_checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "source_checkpoint_epoch": source_manifest["checkpoint_epoch"],
        "source_inference_device": source_manifest["device"],
        "source_render_manifest": str(source_manifest_path),
        "causal": True,
        "look_ahead_frames": 0,
        "parameters": {
            "min_cutoff": args.min_cutoff,
            "beta": args.beta,
            "derivative_cutoff": args.derivative_cutoff,
            "frequency": "per-video FPS",
        },
        "full_frame_policy": (
            "complete contiguous source predictions followed by a causal One-Euro "
            "filter whose state resets at each video boundary"
        ),
        "overlay_style": {
            "content": "prediction_only",
            "palette": "anatomical_multicolor_mpii16",
            "line_thickness_px": args.line_thickness,
            "joint_radius_px": max(2, args.line_thickness + 1),
            "joint_outline": "black_1px",
            "ground_truth": False,
            "bounding_box": False,
            "text_overlay": False,
        },
        "videos": videos,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    prepare_parser.add_argument(
        "--test-metadata", type=Path,
        default=PROJECT_ROOT / "Datasets/NTU_RGBD/metadata/contiguous_xsub/test_split.csv",
    )
    prepare_parser.add_argument(
        "--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT,
    )
    prepare_parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "Video_Visualization",
    )
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--jpeg-quality", type=int, default=95)
    prepare_parser.add_argument(
        "--override", action="append", default=[], metavar="SETUP=SAMPLE_ID",
        help="Replace the seeded choice for one setup with a qualified test sample",
    )
    prepare_parser.set_defaults(function=prepare)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    render_parser.add_argument(
        "--prepared-root", type=Path, default=PROJECT_ROOT / "Video_Visualization",
    )
    render_parser.add_argument("--checkpoint", type=Path, required=True)
    render_parser.add_argument(
        "--device", default="cuda:0",
    )
    render_parser.add_argument("--batch-size", type=int, default=4)
    render_parser.add_argument("--num-workers", type=int, default=4)
    render_parser.add_argument(
        "--sample-id", action="append", default=[],
        help="Process only this prepared sample ID; repeat for incremental export",
    )
    render_parser.add_argument(
        "--line-thickness", type=int, default=2,
        help="Prediction pose line thickness in pixels",
    )
    render_parser.add_argument(
        "--draw-pckhb-headbox", action="store_true",
        help="Overlay a display-only proxy head box derived from the rendered pose",
    )
    render_parser.set_defaults(function=render)
    rerender_parser = subparsers.add_parser("rerender")
    rerender_parser.add_argument(
        "--prepared-root", type=Path, default=PROJECT_ROOT / "Video_Visualization",
    )
    rerender_parser.add_argument("--model-id", required=True)
    rerender_parser.add_argument(
        "--line-thickness", type=int, default=2,
        help="Prediction pose line thickness in pixels",
    )
    rerender_parser.add_argument(
        "--sample-id", action="append", default=[],
        help="Rerender only this prepared sample ID; repeat as needed",
    )
    rerender_parser.add_argument(
        "--draw-pckhb-headbox", action="store_true",
        help="Overlay a display-only proxy head box derived from the rendered pose",
    )
    rerender_parser.set_defaults(function=rerender)
    gt_parser = subparsers.add_parser("rerender-ground-truth")
    gt_parser.add_argument(
        "--prepared-root", type=Path, default=PROJECT_ROOT / "Video_Visualization",
    )
    gt_parser.add_argument("--source-model-id", default="mamv2_fullcs20")
    gt_parser.add_argument(
        "--sample-id", action="append", default=[],
        help="Rerender only this prepared sample ID; repeat as needed",
    )
    gt_parser.add_argument(
        "--line-thickness", type=int, default=2,
        help="Ground-truth pose line thickness in pixels",
    )
    gt_parser.set_defaults(function=rerender_ground_truth)
    one_euro_parser = subparsers.add_parser("one-euro")
    one_euro_parser.add_argument(
        "--prepared-root", type=Path, default=PROJECT_ROOT / "Video_Visualization",
    )
    one_euro_parser.add_argument("--source-model-id", required=True)
    one_euro_parser.add_argument("--model-id", default="pilot20_eval_one_euro")
    one_euro_parser.add_argument("--min-cutoff", type=float, default=3.0)
    one_euro_parser.add_argument("--beta", type=float, default=0.1)
    one_euro_parser.add_argument("--derivative-cutoff", type=float, default=1.0)
    one_euro_parser.add_argument(
        "--sample-id", action="append", default=[],
        help="Process only this prepared sample ID; repeat for incremental export",
    )
    one_euro_parser.add_argument(
        "--line-thickness", type=int, default=2,
        help="Prediction pose line thickness in pixels",
    )
    one_euro_parser.set_defaults(function=one_euro)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "jpeg_quality", 95) not in range(1, 101):
        raise ValueError("jpeg-quality must be between 1 and 100")
    if getattr(args, "line_thickness", 2) not in range(1, 9):
        raise ValueError("line-thickness must be between 1 and 8")
    if getattr(args, "min_cutoff", 3.0) <= 0:
        raise ValueError("min-cutoff must be positive")
    if getattr(args, "beta", 0.1) < 0:
        raise ValueError("beta must be non-negative")
    if getattr(args, "derivative_cutoff", 1.0) <= 0:
        raise ValueError("derivative-cutoff must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
