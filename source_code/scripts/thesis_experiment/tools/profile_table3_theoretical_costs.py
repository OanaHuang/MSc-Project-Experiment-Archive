#!/usr/bin/env python3
"""Generate end-to-end theoretical cost tables for both thesis test sets.

The report deliberately contains no measured latency.  Model arithmetic is
counted at ATen dispatch level and the four coordinate refiners are added with
closed-form causal inference counts.  Frozen checkpoints are hashed for
provenance, although learned values do not affect the operation shapes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch
import yaml

from spikepose_thesis.analysis import (
    END_TO_END_FLOP_ASSUMPTIONS,
    profile_end_to_end_flops,
)
from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.models import build_model


RUN_ROOT = PROJECT_ROOT / "Outputs_Thesis_Pilot20" / "runs" / "ntu60_cs"
DEFAULT_OLD_METADATA = (
    PROJECT_ROOT / "Datasets/NTU_RGBD/metadata/contiguous_xsub/test_split.csv"
)
DEFAULT_FULLVIDEO_METADATA = (
    PROJECT_ROOT / "Datasets/NTU_RGBD/metadata/fullvideo_xsub35/test_split.csv"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    run: str
    call_unit: str


@dataclass(frozen=True)
class Protocol:
    key: str
    display: str
    sequence_lengths: tuple[int, ...]
    clip_invocations: int
    clip_length: int = 16

    @property
    def physical_frames(self) -> int:
        return sum(self.sequence_lengths)

    @property
    def reset_sequences(self) -> int:
        return len(self.sequence_lengths)


MODEL_SPECS = (
    ModelSpec(
        "simplebaseline", "SimpleBaseline",
        "paper_baseline/pilot20_ntu_simplebaseline_r50/seed_42", "frame",
    ),
    ModelSpec(
        "hrnet", "HRNet",
        "paper_baseline/pilot20_ntu_hrnet_w32/seed_42", "frame",
    ),
    ModelSpec(
        "framewise", "Frame-wise",
        "mam_v2_fullcs_source/mamv2_fullcs_p00_source/seed_42", "clip",
    ),
    ModelSpec(
        "core_mam", "Core MAM",
        "mam_v2_fullcs/mamv2_fullcs20/seed_42", "clip",
    ),
    ModelSpec(
        "mam_kpa", "MAM+KPA",
        "mam_ktp_pilot/pilot20_mamv2_kpa/seed_42", "clip",
    ),
    ModelSpec(
        "mam_tpa", "MAM+TPA",
        "mam_ktp_pilot/pilot20_mamv2_tpa/seed_42", "clip",
    ),
    ModelSpec(
        "mam_ktp", "MAM+KPA+TPA",
        "mam_ktp_pilot/pilot20_mamv2_ktp/seed_42", "clip",
    ),
)


TABLE_METHODS = (
    ("simplebaseline", "SimpleBaseline", "simplebaseline", None),
    ("hrnet", "HRNet", "hrnet", None),
    ("framewise", "Frame-wise", "framewise", None),
    ("shared_ema", "Shared EMA", "framewise", "shared_ema"),
    ("jointwise_refinement", "Joint-wise ref.", "framewise", "jointwise"),
    ("one_euro", "One Euro", "framewise", "one_euro"),
    ("causal_sg", "Causal SG", "framewise", "causal_sg"),
    ("core_mam", "Core MAM", "core_mam", None),
    ("mam_kpa", "MAM+KPA", "mam_kpa", None),
    ("mam_tpa", "MAM+TPA", "mam_tpa", None),
    ("mam_ktp", "MAM+KPA+TPA", "mam_ktp", None),
)


REFINER_ASSUMPTIONS = {
    "coordinates_per_frame": 32,
    "state_reset": "at each protocol sequence boundary",
    "shared_ema": "3 vector FLOPs plus one scalar complement per transition",
    "jointwise": (
        "sigmoid/range conversion once per sequence, then two products, one "
        "sum and one jointwise complement per transition"
    ),
    "one_euro": (
        "all arithmetic in the causal derivative, adaptive cutoff and value "
        "updates; each elementary function counts as one FLOP"
    ),
    "causal_sg": (
        "frozen polynomial coefficients applied as causal dot products; "
        "one-time coefficient construction is excluded"
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--old-metadata", type=Path, default=DEFAULT_OLD_METADATA)
    parser.add_argument(
        "--fullvideo-metadata", type=Path, default=DEFAULT_FULLVIDEO_METADATA,
    )
    parser.add_argument(
        "--allow-incomplete-training", action="store_true",
        help="Allow a model status other than completed (disabled by default).",
    )
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _protocols(old_metadata: Path, fullvideo_metadata: Path) -> tuple[Protocol, ...]:
    old_rows = _read_rows(old_metadata)
    fullvideo_rows = _read_rows(fullvideo_metadata)
    old_lengths = (16,) * (len(old_rows) * 2)
    try:
        full_lengths = tuple(int(row["skeleton_frames"]) for row in fullvideo_rows)
    except (KeyError, ValueError) as error:
        raise ValueError("Full-video metadata lacks valid skeleton_frames") from error
    protocols = (
        Protocol(
            "sampled_2x16", "Original sampled 2x16 test",
            old_lengths, len(old_lengths),
        ),
        Protocol(
            "fullvideo_xsub35", "Complete-video xsub35 test",
            full_lengths, sum(math.ceil(length / 16) for length in full_lengths),
        ),
    )
    expected = {
        "sampled_2x16": (13_124, 26_248, 419_968),
        "fullvideo_xsub35": (4_563, 28_404, 419_968),
    }
    observed = {
        "sampled_2x16": (
            len(old_rows), protocols[0].clip_invocations,
            protocols[0].physical_frames,
        ),
        "fullvideo_xsub35": (
            protocols[1].reset_sequences, protocols[1].clip_invocations,
            protocols[1].physical_frames,
        ),
    }
    if observed != expected:
        raise RuntimeError(
            f"Frozen protocol cardinality changed: observed={observed}, "
            f"expected={expected}"
        )
    return protocols


def _profile_model(spec: ModelSpec, device: torch.device,
                   allow_incomplete: bool) -> dict[str, Any]:
    run = RUN_ROOT / spec.run
    config_path = run / "resolved_config.yaml"
    checkpoint_path = run / "checkpoints/best.pt"
    status_path = run / "status.json"
    for path in (config_path, checkpoint_path, status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed" and not allow_incomplete:
        raise RuntimeError(
            f"Training is not complete for {spec.display}: {status.get('state')!r}"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(config).to(device).eval()
    size = int(config["data"]["image_size"])
    temporal = config["model"].get("temporal", {})
    if spec.call_unit == "clip":
        frames = int(temporal["video_frames"])
        image = torch.zeros(1, frames, 3, size, size, device=device)
    else:
        image = torch.zeros(1, 3, size, size, device=device)
    arithmetic = profile_end_to_end_flops(model, image, strict=True)
    result = {
        "key": spec.key,
        "display": spec.display,
        "call_unit": spec.call_unit,
        "run": str(run),
        "experiment": config["id"],
        "training_state": status.get("state"),
        "completed_epoch": status.get("completed_epoch"),
        "resolved_config": str(config_path),
        "resolved_config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        **arithmetic,
    }
    del model, image
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _refiner_flops(method: str | None, lengths: tuple[int, ...],
                   joints: int = 16) -> int:
    if method is None:
        return 0
    coordinates = joints * 2
    transitions = sum(max(length - 1, 0) for length in lengths)
    if method == "shared_ema":
        return transitions * (3 * coordinates + 1)
    if method == "jointwise":
        alpha_setup = len(lengths) * (3 * joints)
        per_transition = 3 * coordinates + joints
        return alpha_setup + transitions * per_transition
    if method == "one_euro":
        # See REFINER_ASSUMPTIONS. Scalar alpha arithmetic contributes 8.
        return transitions * (17 * coordinates + 8)
    if method == "causal_sg":
        total = 0
        for length in lengths:
            for index in range(1, length):
                history = min(index + 1, 5)
                per_coordinate = history if history <= 2 else 2 * history - 1
                total += per_coordinate * coordinates
        return total
    raise ValueError(f"Unknown refiner: {method}")


def _refiner_parameters(method: str | None) -> int:
    return 16 if method == "jointwise" else 0


def _row_for_protocol(method: tuple[str, str, str, str | None],
                      models: dict[str, dict[str, Any]],
                      protocol: Protocol) -> dict[str, Any]:
    key, display, model_key, refiner = method
    model = models[model_key]
    invocations = (
        protocol.physical_frames
        if model["call_unit"] == "frame"
        else protocol.clip_invocations
    )
    model_flops = int(model["flops"]) * invocations
    post_flops = _refiner_flops(refiner, protocol.sequence_lengths)
    total = model_flops + post_flops
    canonical_lengths = (protocol.clip_length,)
    canonical_model = (
        int(model["flops"]) * protocol.clip_length
        if model["call_unit"] == "frame"
        else int(model["flops"])
    )
    canonical = canonical_model + _refiner_flops(refiner, canonical_lengths)
    parameters = int(model["parameters"]) + _refiner_parameters(refiner)
    return {
        "method_key": key,
        "method": display,
        "protocol": protocol.key,
        "physical_frames": protocol.physical_frames,
        "sequence_resets": protocol.reset_sequences,
        "model_call_unit": model["call_unit"],
        "model_invocations": invocations,
        "canonical_16_frame_gflops": canonical / 1e9,
        "model_theoretical_flops": model_flops,
        "postprocess_theoretical_flops": post_flops,
        "total_theoretical_flops": total,
        "total_theoretical_pflops": total / 1e15,
        "average_gflops_per_unique_frame": total / protocol.physical_frames / 1e9,
        "parameters": parameters,
        "parameters_million": parameters / 1e6,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "complete_operator_coverage": bool(model["complete"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]], protocol: Protocol) -> str:
    selected = [row for row in rows if row["protocol"] == protocol.key]
    lines = [
        f"### {protocol.display}",
        "",
        "| Method | GFLOPs / 16 frames | Params (M) | Postprocess GFLOPs (test) | Total theoretical PFLOPs |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['method']} | {row['canonical_16_frame_gflops']:.6f} | "
        f"{row['parameters_million']:.6f} | "
        f"{row['postprocess_theoretical_flops'] / 1e9:.6f} | "
        f"{row['total_theoretical_pflops']:.9f} |"
        for row in selected
    )
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocols = _protocols(
        args.old_metadata.resolve(), args.fullvideo_metadata.resolve(),
    )
    device = torch.device(args.device)
    models = {}
    for spec in MODEL_SPECS:
        print(f"profiling {spec.display}", flush=True)
        models[spec.key] = _profile_model(
            spec, device, args.allow_incomplete_training,
        )
        (output / f"operator_breakdown_{spec.key}.json").write_text(
            json.dumps(models[spec.key], indent=2), encoding="utf-8",
        )
    rows = [
        _row_for_protocol(method, models, protocol)
        for protocol in protocols for method in TABLE_METHODS
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "theoretical_only_no_measured_latency",
        "finalized_after_training": all(
            model["training_state"] == "completed" for model in models.values()
        ),
        "incomplete_training_models": [
            model["display"] for model in models.values()
            if model["training_state"] != "completed"
        ],
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "model_flop_assumptions": END_TO_END_FLOP_ASSUMPTIONS,
        "refiner_assumptions": REFINER_ASSUMPTIONS,
        "protocols": [
            {
                "key": item.key,
                "display": item.display,
                "physical_frames": item.physical_frames,
                "sequence_resets": item.reset_sequences,
                "clip_invocations": item.clip_invocations,
                "clip_length": item.clip_length,
            }
            for item in protocols
        ],
        "models": models,
        "rows": rows,
    }
    (output / "theoretical_costs.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    _write_csv(output / "theoretical_costs.csv", rows)
    markdown = [
        "# Table 3 theoretical cost",
        "",
        "No measured latency is included.  One MAC is two FLOPs.",
        "",
        *[_markdown(rows, protocol) for protocol in protocols],
        "",
    ]
    (output / "theoretical_costs.md").write_text(
        "\n\n".join(markdown), encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "protocols": report["protocols"],
        "rows": len(rows),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
