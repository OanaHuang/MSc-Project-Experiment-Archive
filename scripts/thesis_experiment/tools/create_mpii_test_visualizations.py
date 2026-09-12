#!/usr/bin/env python3
"""Prepare fixed MPII test images and render prediction-only pose overlays."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
THESIS_SRC = PROJECT_ROOT / "scripts" / "thesis_experiment" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(THESIS_SRC))

from scripts.thesis_experiment.tools.create_ntu_test_visualizations import (  # noqa: E402
    _draw_prediction_pose,
)
from spikepose_thesis.data.mpii.core.geometry import crop_person  # noqa: E402


MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty image selection")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _test_people(annotation_path: Path, images_dir: Path) -> list[dict]:
    values = json.loads(annotation_path.read_text(encoding="utf-8"))
    people_per_image: dict[str, int] = {}
    rows = []
    for official_index, value in enumerate(values):
        image_name = str(value["image"])
        person_index = people_per_image.get(image_name, 0)
        people_per_image[image_name] = person_index + 1
        image_path = images_dir / image_name
        if not image_path.is_file():
            continue
        center = value.get("center")
        scale = value.get("scale")
        if center is None or len(center) != 2 or scale is None:
            continue
        if not np.isfinite([float(center[0]), float(center[1]), float(scale)]).all():
            continue
        if float(scale) <= 0:
            continue
        rows.append({
            "official_test_index": official_index,
            "image": image_name,
            "person_index": person_index,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "scale": float(scale),
        })
    for row in rows:
        row["people_in_official_test_annotations"] = people_per_image[row["image"]]
    return rows


def _select_unique_images(
    rows: list[dict],
    images_dir: Path,
    count: int,
    seed: int,
    crop_expansion: float,
    excluded_images: set[str],
) -> list[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected = []
    used_images = set()
    for row in shuffled:
        image_name = str(row["image"])
        if image_name in used_images or image_name in excluded_images:
            continue
        # These source-only filters avoid tiny/background people and ambiguous
        # test annotations without using any model prediction to choose samples.
        if int(row["people_in_official_test_annotations"]) != 1:
            continue
        image = cv2.imread(
            str(images_dir / image_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            continue
        height, width = image.shape[:2]
        center_x = float(row["center_x"])
        center_y = float(row["center_y"])
        side = float(row["scale"]) * 200.0 * crop_expansion
        x1, y1 = center_x - side / 2.0, center_y - side / 2.0
        x2, y2 = center_x + side / 2.0, center_y + side / 2.0
        retained_area = (
            max(0.0, min(float(width), x2) - max(0.0, x1))
            * max(0.0, min(float(height), y2) - max(0.0, y1))
        )
        retained_fraction = retained_area / (side * side)
        crop_area_fraction = side * side / float(width * height)
        center_x_fraction = center_x / float(width)
        center_y_fraction = center_y / float(height)
        if not (
            0.15 <= center_x_fraction <= 0.85
            and 0.12 <= center_y_fraction <= 0.88
            and retained_fraction >= 0.72
            and 0.10 <= crop_area_fraction <= 0.90
        ):
            continue
        used_images.add(image_name)
        selected.append({
            **row,
            "selection_center_x_fraction": center_x_fraction,
            "selection_center_y_fraction": center_y_fraction,
            "selection_retained_crop_fraction": retained_fraction,
            "selection_crop_area_fraction": crop_area_fraction,
        })
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} unique qualified test images are available")


def prepare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    images_dir = args.images_dir.resolve()
    annotation_path = args.test_annotations.resolve()
    candidates = _test_people(annotation_path, images_dir)
    selected = _select_unique_images(
        candidates,
        images_dir,
        args.count,
        args.seed,
        args.crop_expansion,
        set(args.exclude_image),
    )
    prepared = []
    for order, row in enumerate(selected, 1):
        source = images_dir / row["image"]
        sample_id = f"{Path(row['image']).stem}_p{int(row['person_index']):02d}"
        target = output_root / "full_images" / f"{sample_id}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        image = cv2.imread(str(target), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise RuntimeError(f"Could not read selected image: {target}")
        height, width = image.shape[:2]
        crop, _, box, _ = crop_person(
            image, np.zeros((16, 2), np.float32),
            [row["center_x"], row["center_y"]], row["scale"],
            args.image_size, args.crop_expansion,
        )
        if crop.shape[:2] != (args.image_size, args.image_size):
            raise RuntimeError(f"Unexpected person crop for {sample_id}: {crop.shape}")
        value = {
            "sample_id": sample_id,
            **row,
            "source_path": str(source),
            "full_image_path": str(target),
            "source_sha256": _sha256(target),
            "width": width,
            "height": height,
            "person_box_x1": float(box[0]),
            "person_box_y1": float(box[1]),
            "person_box_x2": float(box[2]),
            "person_box_y2": float(box[3]),
            "selection_seed": args.seed,
        }
        prepared.append(value)
        print(f"[prepare] {order:02d}/{args.count:02d} {sample_id}", flush=True)

    _write_csv(output_root / "selected_test_metadata.csv", prepared)
    _write_csv(output_root / "selected_test_metadata_local.csv", prepared)
    _write_csv(output_root / "selected_test_images.csv", prepared)
    manifest = {
        "dataset": "MPII Human Pose official test split",
        "source_annotations": str(annotation_path),
        "selection_policy": (
            "deterministic seed-based selection from official test images with one "
            "annotated test person, a centered annotation, at least 72% retained "
            "person-crop area, and a 10%-90% crop-to-image area ratio; selection "
            "and source-clarity exclusions are independent of model predictions"
        ),
        "selection_seed": args.seed,
        "source_clarity_exclusions": sorted(set(args.exclude_image)),
        "images": len(prepared),
        "sample_ids": [row["sample_id"] for row in prepared],
        "ground_truth_included": False,
        "image_size": args.image_size,
        "crop_expansion": args.crop_expansion,
    }
    _write_json(output_root / "preparation_manifest.json", manifest)
    (output_root / "README.md").write_text(
        "# MPII Test Image Visualization\n\n"
        "This directory contains ten deterministically selected official MPII "
        "test images. Selection uses only official person metadata and source "
        "visibility, never model predictions. The official test split does not "
        "publish pose labels, so all rendered outputs are prediction-only. "
        "Original images are kept in `full_images/`; model outputs use the same "
        "multicolour MPII-16 skeleton style as the NTU video visualizations.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _model_input(image: np.ndarray, row: dict, config: dict):
    import torch

    crop, _, box, inverse = crop_person(
        image, np.zeros((16, 2), np.float32),
        [float(row["center_x"]), float(row["center_y"])], float(row["scale"]),
        int(config["data"]["image_size"]), float(config["data"]["crop_expansion"]),
    )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    value = np.transpose((rgb - MEAN) / STD, (2, 0, 1))
    return torch.from_numpy(value).float().unsqueeze(0), box, inverse


def render(args: argparse.Namespace) -> None:
    import torch

    from spikepose_thesis.evaluation.mpii.metrics import prediction_to_keypoints
    from spikepose_thesis.models import build_model
    from spikepose_thesis.training.checkpoint import load_checkpoint

    prepared_root = args.prepared_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    rows = _read_csv(prepared_root / "selected_test_metadata.csv")
    device = torch.device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = deepcopy(checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model_id = str(config["id"])
    model_root = prepared_root / "models" / model_id
    prediction_root = model_root / "predictions"
    image_root = model_root / "images_prediction_only"
    prediction_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    decoder = config.get("evaluation", {}).get("main_decoder", "dark")
    image_size = int(config["data"]["image_size"])
    outputs = []
    started = time.monotonic()
    with torch.inference_mode():
        for order, row in enumerate(rows, 1):
            source_path = Path(row["full_image_path"])
            image = cv2.imread(
                str(source_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                raise RuntimeError(f"Could not read image: {source_path}")
            tensor, box, inverse = _model_input(image, row, config)
            prediction = model(tensor.to(device))
            coordinates, confidence = prediction_to_keypoints(
                prediction, image_size, decoder=decoder,
            )
            points = coordinates[0].astype(np.float32)
            points[:, 0] = points[:, 0] * inverse[0] + inverse[2]
            points[:, 1] = points[:, 1] * inverse[1] + inverse[3]
            confidence_value = confidence[0].astype(np.float32)
            sample_id = row["sample_id"]
            np.savez_compressed(
                prediction_root / f"{sample_id}.npz",
                prediction=points,
                confidence=confidence_value,
                person_box=box,
                center=np.asarray([row["center_x"], row["center_y"]], np.float32),
                scale=np.asarray(float(row["scale"]), np.float32),
            )
            rendered = image.copy()
            _draw_prediction_pose(
                rendered, points, confidence_value, args.line_thickness,
            )
            output_path = image_root / f"{sample_id}_prediction.png"
            if not cv2.imwrite(str(output_path), rendered):
                raise RuntimeError(f"Could not write image: {output_path}")
            outputs.append({
                "sample_id": sample_id,
                "source_image": str(source_path),
                "path": str(output_path),
                "width": int(rendered.shape[1]),
                "height": int(rendered.shape[0]),
                "bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
            })
            print(f"[render] {order:02d}/{len(rows):02d} {sample_id}", flush=True)

    manifest = {
        "experiment_id": model_id,
        "model_name": config.get("name", model_id),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "decoder": decoder,
        "device": str(device),
        "dataset": "MPII Human Pose official test split",
        "ground_truth_included": False,
        "overlay_style": {
            "content": "prediction_only",
            "palette": "anatomical_multicolor_mpii16",
            "line_thickness_px": args.line_thickness,
            "joint_radius_px": max(2, args.line_thickness + 1),
            "joint_outline": "black_1px",
            "bounding_box": False,
            "text_overlay": False,
        },
        "images": outputs,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(model_root / "render_manifest_prediction_only.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--images-dir", type=Path, default=PROJECT_ROOT / "Datasets/MPII/images",
    )
    prepare_parser.add_argument(
        "--test-annotations", type=Path,
        default=PROJECT_ROOT / "Datasets/MPII/official_simplebaseline/annot/test.json",
    )
    prepare_parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "Image_Visualization",
    )
    prepare_parser.add_argument("--count", type=int, default=10)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--image-size", type=int, default=256)
    prepare_parser.add_argument("--crop-expansion", type=float, default=1.25)
    prepare_parser.add_argument(
        "--exclude-image", action="append", default=[],
        help="Source filename to omit for visual clarity; repeat as needed",
    )
    prepare_parser.set_defaults(function=prepare)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument(
        "--prepared-root", type=Path, default=PROJECT_ROOT / "Image_Visualization",
    )
    render_parser.add_argument("--checkpoint", type=Path, required=True)
    render_parser.add_argument("--device", default="mps")
    render_parser.add_argument("--line-thickness", type=int, default=2)
    render_parser.set_defaults(function=render)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "count", 10) < 1:
        raise ValueError("count must be positive")
    if getattr(args, "line_thickness", 2) not in range(1, 9):
        raise ValueError("line-thickness must be between 1 and 8")
    args.function(args)


if __name__ == "__main__":
    main()
