#!/usr/bin/env python3
"""Compose consistently cropped NTU frames into a paper-ready strip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--headbox-source", required=True)
    parser.add_argument("--headbox-geometry", required=True)
    parser.add_argument("--frame-id", type=int, action="append", required=True)
    parser.add_argument(
        "--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
    )
    parser.add_argument("--gap", type=int, default=14)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--label-height", type=int, default=74)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--box-width-ratio", type=float, default=0.75)
    return parser.parse_args()


def _load_panel(frame_root: Path, frame_id: int, crop: tuple[int, ...]):
    frame_path = frame_root / f"frame_{frame_id:06d}.png"
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {frame_path}")
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError(f"Crop {crop} is outside {frame_path} with shape {image.shape}")
    return image[y1:y2, x1:x2].copy()


def _write_pdf(png_path: Path, pdf_path: Path, dpi: int) -> None:
    sips = shutil.which("sips")
    if sips is None:
        raise RuntimeError("PDF export requires the macOS sips command")
    subprocess.run(
        [sips, "-s", "dpiWidth", str(dpi), "-s", "dpiHeight", str(dpi),
         str(png_path)],
        check=True, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sips, "-s", "format", "pdf", str(png_path), "--out", str(pdf_path)],
        check=True, stdout=subprocess.DEVNULL,
    )


def main() -> None:
    args = parse_args()
    crop = tuple(args.crop)
    panels = [_load_panel(args.frame_root, frame_id, crop) for frame_id in args.frame_id]
    panel_height, panel_width = panels[0].shape[:2]
    if any(panel.shape[:2] != (panel_height, panel_width) for panel in panels):
        raise ValueError("All cropped panels must have the same dimensions")

    width = 2 * args.margin + len(panels) * panel_width + (len(panels) - 1) * args.gap
    height = args.margin + panel_height + args.label_height
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    font_thickness = 2
    border_color = (102, 84, 66)

    for index, (frame_id, panel) in enumerate(zip(args.frame_id, panels)):
        left = args.margin + index * (panel_width + args.gap)
        top = args.margin
        canvas[top:top + panel_height, left:left + panel_width] = panel
        cv2.rectangle(
            canvas, (left, top), (left + panel_width - 1, top + panel_height - 1),
            border_color, 1, cv2.LINE_AA,
        )
        label = f"Frame {frame_id}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness,
        )
        text_x = left + (panel_width - text_width) // 2
        text_y = top + panel_height + (args.label_height + text_height) // 2
        cv2.putText(
            canvas, label, (text_x, text_y), font, font_scale, (0, 0, 0),
            font_thickness, cv2.LINE_AA,
        )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"Could not write {args.output_png}")
    if args.output_pdf is not None:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        _write_pdf(args.output_png, args.output_pdf, args.dpi)

    if args.manifest is not None:
        payload = {
            "sample_id": args.sample_id,
            "content": args.content,
            "headbox_source": args.headbox_source,
            "headbox_geometry": args.headbox_geometry,
            "head_centre_to_neck_centre_ratio": 0.75,
            "neck_centre_extension_beyond_upper_neck": 0.25,
            "box_width_to_length_ratio": args.box_width_ratio,
            "box_width_axis": "perpendicular_to_head_centre--neck_centre_axis",
            "metric_definition_changed": False,
            "frame_ids": args.frame_id,
            "source_frame_root": str(args.frame_root.resolve()),
            "consistent_crop_xyxy": list(crop),
            "panel_size_px": [panel_width, panel_height],
            "gap_px": args.gap,
            "outer_margin_px": args.margin,
            "figure_size_px": [width, height],
            "background": "white",
            "output_png": str(args.output_png.resolve()),
            "output_pdf": (
                str(args.output_pdf.resolve()) if args.output_pdf is not None else None
            ),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(args.manifest)


if __name__ == "__main__":
    main()
