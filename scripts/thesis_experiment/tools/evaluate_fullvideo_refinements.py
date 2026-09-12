#!/usr/bin/env python3
"""Apply the frozen Table-3 temporal refinements to complete-video predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.data.ntu.core.joint_mapping import MPII16_JOINT_NAMES
from spikepose_thesis.evaluation.runner import _ntu_group_summaries, summarize_predictions
from spikepose_thesis.evaluation.temporal import sequence_groups, temporal_summary
from spikepose_thesis.refinement.filters import (
    OneEuroFilter,
    ema_shared,
    savgol_causal,
)
from spikepose_thesis.refinement.jtr import JointwiseTemporalRefinement
from spikepose_thesis.refinement.runner import _apply_filter, _apply_module


METHODS = {
    "shared_ema": {"alpha": 0.8},
    "one_euro": {
        "min_cutoff": 3.0,
        "beta": 0.1,
        "derivative_cutoff": 1.0,
    },
    "causal_sg": {"window": 5, "order": 2},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--selection-provenance", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, required=True)
    parser.add_argument("--jointwise-config", type=Path, required=True)
    parser.add_argument("--jointwise-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _full_frame_pck(values: dict[str, np.ndarray]) -> dict:
    prediction = values["prediction"]
    target = values["target"]
    visibility = values["visibility"]
    scale = values["scale_hb"]
    distance = np.linalg.norm(prediction - target, axis=-1)
    valid = (
        (visibility > 0)
        & np.isfinite(distance)
        & np.isfinite(scale[:, None])
    )
    correct = distance <= 0.5 * scale[:, None]
    return {
        "pck_0.5": float(correct[valid].mean()),
        "samples": int(len(scale)),
        "per_joint": {
            name: {
                "pck_0.5": float(correct[:, joint][valid[:, joint]].mean()),
            }
            for joint, name in enumerate(MPII16_JOINT_NAMES)
        },
    }


def _validate_source(values: dict[str, np.ndarray], provenance: dict) -> None:
    required = {
        "prediction", "target", "visibility", "scale", "scale_hb",
        "sample_id", "person_id", "frame_index", "video_id", "clip_id",
        "frame_position_in_clip",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Source archive is missing {sorted(missing)}")
    expected_frames = int(provenance["expected_physical_frames"])
    expected_videos = int(provenance["videos"])
    if len(values["prediction"]) != expected_frames:
        raise ValueError("Source archive frame count differs from provenance")
    groups = sequence_groups(values)
    if len(groups) != expected_videos:
        raise ValueError("Source archive video count differs from provenance")
    for indices in groups:
        positions = values["frame_position_in_clip"][indices].astype(np.int64)
        if not np.array_equal(positions, np.arange(len(indices))):
            raise ValueError(
                f"Non-contiguous source video: {values['video_id'][indices[0]]}"
            )


def _summarize(values: dict[str, np.ndarray]) -> dict:
    result = summarize_predictions(values)
    result["pck_hb"] = summarize_predictions({
        **values, "scale": values["scale_hb"],
    })
    result["pck_hb_all_frames"] = _full_frame_pck(values)
    result["pck_hb_video_macro"] = {
        "pck_0.5": float(result["pck_hb"]["sequence_equal"]["pck_0.5"]),
        "videos": int(result["pck_hb"]["sequences"]),
    }
    result["temporal"] = temporal_summary(values, strict=False)
    result["groups"] = _ntu_group_summaries(values)
    result["head_bone_calibration_ready"] = True
    result["videos"] = int(result["pck_hb"]["sequences"])
    result["physical_frames"] = int(len(values["prediction"]))
    return result


def _write_method(
    output_root: Path, method: str, values: dict[str, np.ndarray],
    parameters: dict, common_provenance: dict,
) -> dict:
    output = output_root / method
    output.mkdir(parents=True, exist_ok=True)
    summary = _summarize(values)
    np.savez_compressed(output / "predictions.npz", **values)
    np.savez_compressed(output / "predictions_per_frame.npz", **values)
    (output / "parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    completion = {
        "method": method,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "videos": int(summary["videos"]),
        "physical_frames": int(summary["physical_frames"]),
        "pck_hb_all_frames": float(summary["pck_hb_all_frames"]["pck_0.5"]),
        "pck_hb_video_macro": float(summary["pck_hb_video_macro"]["pck_0.5"]),
        "parameters": parameters,
        **common_provenance,
    }
    (output / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8",
    )
    print(json.dumps(completion, indent=2), flush=True)
    return completion


def main() -> None:
    args = _parser().parse_args()
    paths = {
        "source": args.source.resolve(),
        "source_provenance": args.source_provenance.resolve(),
        "selection_provenance": args.selection_provenance.resolve(),
        "selection_results": args.selection_results.resolve(),
        "jointwise_config": args.jointwise_config.resolve(),
        "jointwise_checkpoint": args.jointwise_checkpoint.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source = _load(paths["source"])
    source_provenance = json.loads(
        paths["source_provenance"].read_text(encoding="utf-8")
    )
    selection_provenance = json.loads(
        paths["selection_provenance"].read_text(encoding="utf-8")
    )
    selection_results = json.loads(
        paths["selection_results"].read_text(encoding="utf-8")
    )
    _validate_source(source, source_provenance)
    if selection_provenance.get("selection_split") != "validation":
        raise ValueError("Filter parameters were not selected on validation")
    for method, parameters in METHODS.items():
        if selection_results.get(method, {}).get("parameters") != parameters:
            raise ValueError(f"Frozen validation parameters differ for {method}")

    common = {
        "source": str(paths["source"]),
        "source_sha256": _sha256(paths["source"]),
        "source_split_sha256": source_provenance["split_metadata_sha256"],
        "selection_provenance": str(paths["selection_provenance"]),
        "selection_provenance_sha256": _sha256(paths["selection_provenance"]),
        "selection_results": str(paths["selection_results"]),
        "selection_results_sha256": _sha256(paths["selection_results"]),
        "state_reset": "video_boundary_only",
        "physical_frames_scored_once": True,
        "git_commit": _git(["rev-parse", "HEAD"]),
    }
    results = {}
    functions = {
        "shared_ema": lambda value: ema_shared(value, METHODS["shared_ema"]["alpha"]),
        "one_euro": lambda value: OneEuroFilter(**METHODS["one_euro"])(value),
        "causal_sg": lambda value: savgol_causal(
            value, METHODS["causal_sg"]["window"], METHODS["causal_sg"]["order"],
        ),
    }
    for method, function in functions.items():
        print(f"applying method={method}", flush=True)
        refined = _apply_filter(source, function)
        results[method] = _write_method(
            output_root, method, refined, METHODS[method], common,
        )

    config = yaml.safe_load(paths["jointwise_config"].read_text(encoding="utf-8"))
    protocol = config["refinement_protocol"]
    refinement = config["refinement"]
    device = torch.device(args.device)
    module = JointwiseTemporalRefinement(
        int(refinement["joints"]), jointwise=True,
        alpha_min=float(protocol["alpha_min"]),
        alpha_max=float(protocol["alpha_max"]),
    ).to(device)
    payload = torch.load(
        paths["jointwise_checkpoint"], map_location=device, weights_only=False,
    )
    module.load_state_dict(payload["model_state_dict"], strict=True)
    module.eval()
    if any(not torch.isfinite(item).all() for item in module.parameters()):
        raise FloatingPointError("Joint-wise checkpoint contains non-finite values")
    print("applying method=jointwise_refinement", flush=True)
    refined = _apply_module(
        module, source, device, float(protocol["coordinate_scale"]),
    )
    joint_parameters = {
        "alpha": module.alpha.detach().cpu().tolist(),
        "coordinate_scale": float(protocol["coordinate_scale"]),
        "checkpoint": str(paths["jointwise_checkpoint"]),
        "checkpoint_sha256": _sha256(paths["jointwise_checkpoint"]),
    }
    results["jointwise_refinement"] = _write_method(
        output_root, "jointwise_refinement", refined, joint_parameters, common,
    )
    (output_root / "refinement_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
