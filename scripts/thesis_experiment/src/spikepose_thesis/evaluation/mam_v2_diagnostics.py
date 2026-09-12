from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.data.ntu.core.joint_mapping import joint_layout
from spikepose_thesis.evaluation.runner import (
    _append_ntu_temporal_predictions,
    _empty_prediction_result,
    summarize_predictions,
)
from spikepose_thesis.evaluation.temporal import temporal_summary
from spikepose_thesis.models.mam_v2 import MAMV2Config, MotionAlignedMembraneV2


def _finalize(value: dict[str, list]) -> dict[str, np.ndarray]:
    return {key: np.asarray(rows) for key, rows in value.items()}


def _terminal_rows(
    values: dict[str, np.ndarray], expected_length: int,
) -> dict[str, np.ndarray]:
    selected = values["frame_position_in_clip"] == expected_length - 1
    if not bool(selected.any()):
        raise ValueError("MAM diagnostic export contains no terminal clip frames")
    return {key: value[selected] for key, value in values.items()}


@torch.inference_mode()
def evaluate_mam_v2_diagnostic_grid(
    model,
    loader,
    config: dict,
    device: torch.device,
    variants: dict[str, MAMV2Config],
    output_dir: Path | None = None,
    decoder: str | None = None,
) -> dict[str, dict]:
    """Evaluate Reset/Low-Carry/GT-Warp with one spatial pass per batch.

    This path is diagnostic-only. It never installs oracle memory into the
    source model and never exposes GT to its spatial network.
    """
    if config["dataset"] == "mpii":
        raise ValueError("MAM V2 diagnostics require an NTU clip dataset")
    expected_length = int(config["temporal"]["video_frames"])
    image_size = int(config["data"]["image_size"])
    heatmap_size = int(config["data"]["heatmap_size"])
    layout = joint_layout(
        config["data"].get("joint_mapping", "ntu25_to_mpii16_v1"),
    )
    decoder = decoder or config.get("evaluation", {}).get("main_decoder", "dark")
    cells = {
        name: MotionAlignedMembraneV2(
            int(config["model"]["num_joints"]), variant,
        ).to(device).eval()
        for name, variant in variants.items()
    }
    if any(cell.mode not in {"reset", "carry_fixed", "gt_align_oracle"}
           for cell in cells.values()):
        raise ValueError(
            "D1 grid accepts only deterministic reset/carry_fixed/gt_align_oracle modes"
        )
    collected = {
        name: _empty_prediction_result(temporal_identity=True)
        for name in variants
    }
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        # Performance invariant: expensive SpikePose spatial inference runs
        # once, then all lightweight temporal variants reuse the heatmaps.
        spatial = model.forward_spatial_per_step(images)
        target_xy = batch["temporal_keypoints"].to(
            device, non_blocking=True,
        ).transpose(0, 1)
        target_xy = target_xy * (float(heatmap_size) / float(image_size))
        visibility = batch["temporal_visibility"].to(
            device, non_blocking=True,
        ).transpose(0, 1)
        for name, cell in cells.items():
            result = cell.forward_sequence(
                spatial,
                gt_keypoints_heatmap=target_xy if cell.mode == "gt_align_oracle" else None,
                visibility=visibility if cell.mode == "gt_align_oracle" else None,
                diagnostic=cell.mode == "gt_align_oracle",
            )
            _append_ntu_temporal_predictions(
                collected[name], result.heatmap, batch, image_size, decoder,
                int(layout["head_index"]), int(layout["neck_index"]),
            )

    reports = {}
    for name, rows in collected.items():
        values = _finalize(rows)
        terminal = _terminal_rows(values, expected_length)
        report = {
            "variant": name,
            "config": vars(variants[name]),
            "terminal": summarize_predictions(terminal),
            "all_frames": summarize_predictions(values),
            "temporal": temporal_summary(
                values, strict=True, expected_length=expected_length,
            ),
        }
        reports[name] = report
        if output_dir is not None:
            variant_dir = output_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                variant_dir / "predictions_per_frame.npz", **values,
            )
            (variant_dir / "summary.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
    return reports


def default_d1_variants() -> dict[str, MAMV2Config]:
    """Small default grid from the feedback checklist."""
    result = {
        "reset": MAMV2Config(
            mode="reset", residual_gamma_init=0.0,
            use_residual_offset=False, use_dynamic_gate=False,
        ),
    }
    for value in (0.02, 0.05, 0.10, 0.20):
        label = f"carry_{value:.2f}".replace(".", "p")
        result[label] = MAMV2Config(
            mode="carry_fixed", decay_init=value,
            residual_gamma_init=value,
            use_residual_offset=False, use_dynamic_gate=False,
        )
        label = f"gt_warp_{value:.2f}".replace(".", "p")
        result[label] = MAMV2Config(
            mode="gt_align_oracle", decay_init=value,
            residual_gamma_init=value,
            use_residual_offset=False, use_dynamic_gate=False,
        )
    return result
