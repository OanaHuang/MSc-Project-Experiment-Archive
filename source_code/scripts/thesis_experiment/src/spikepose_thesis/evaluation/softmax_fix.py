"""Fixed-heatmap diagnostics and declared validation-only candidate selection."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.models.mam_v2.coordinate_decoding import DecoderSpec, decode_coordinates


DECODERS = {
    "softmax": DecoderSpec("softmax"),
    "scaled10": DecoderSpec("softmax", beta=10),
    "scaled20": DecoderSpec("softmax", beta=20),
    "dark": DecoderSpec("dark"),
    "relu": DecoderSpec("relu"),
    "power2": DecoderSpec("power", alpha=2),
    "expm1": DecoderSpec("expm1"),
}
BINS = {"all": (0, float("inf")), "0_0.5": (0, 0.5), "0.5_2": (0.5, 2), "2_4": (2, 4), "4_plus": (4, float("inf"))}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MotionMetrics:
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, xy, valid, target, visible, final=None, residual=None):
        """T,B,J arrays; differences never cross B lanes / sampled clips."""
        xy, target = xy.detach().float(), target.detach().float()
        visible = visible.bool() & torch.isfinite(target).all(-1)
        coord_error = (xy - target).norm(dim=-1)
        self.values["coordinate_epe"].append(coord_error[visible].cpu().numpy())
        self.values["invalid"].append((~valid)[visible].cpu().numpy())
        dt = target[1:] - target[:-1]
        dc = xy[1:] - xy[:-1]
        pair_valid = valid[1:] & valid[:-1]
        # Same sentinel handling as the MAM feature path; invalid pairs are
        # included in GT-valid aggregate metrics, not silently dropped.
        dc = torch.where(pair_valid.unsqueeze(-1), dc, torch.zeros_like(dc))
        mask = visible[1:] & visible[:-1]
        error = dc - dt
        metrics = {
            "zero_epe": dt.norm(dim=-1),
            "coarse_epe": error.norm(dim=-1),
            "coverage2": error.abs().amax(-1) <= 2,
            "reachable_epe": (error.abs() - 2).clamp_min(0).norm(dim=-1),
            "coarse_length": dc.norm(dim=-1),
            "invalid_pair": ~pair_valid,
        }
        if final is not None:
            metrics["final_epe"] = (final[1:] - dt).norm(dim=-1)
            metrics["final_length"] = final[1:].norm(dim=-1)
        if residual is not None:
            metrics["residual_saturation"] = (residual[1:].abs() >= 1.98).any(-1)
        speed = dt.norm(dim=-1)
        for label, (low, high) in BINS.items():
            selected = mask & (speed >= low) & (speed < high)
            for key, value in metrics.items():
                self.values[f"{label}/{key}"].append(value[selected].detach().cpu().numpy())

    def summary(self):
        result = {}
        for key, arrays in self.values.items():
            values = np.concatenate(arrays) if arrays else np.array([])
            result[key] = {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None}
        return result


def choose_candidates(summaries, count=2):
    """Sum of ranks: coordinate EPE, balanced motion EPE, reachable EPE, invalidity.

    No requirement to beat zero motion. Tie order is stable by candidate name.
    This is a predeclared screening heuristic, not a claim about final accuracy.
    """
    candidates = {}
    for name in ("relu", "power2", "expm1"):
        summary = summaries[name]
        def mean_bins(metric):
            values = [summary[f"{label}/{metric}"]["mean"] for label in BINS if label != "all"]
            return float(np.mean([v for v in values if v is not None]))
        row = [summary["coordinate_epe"]["mean"], mean_bins("coarse_epe"),
               mean_bins("reachable_epe"), summary["invalid"]["mean"]]
        if row[-1] is not None and row[-1] < 1 and all(v is not None and np.isfinite(v) for v in row):
            candidates[name] = row
    if not candidates:
        raise RuntimeError("No numerically usable differentiable decoder")
    scores = {name: 0 for name in candidates}
    for column in range(4):
        for name, row in candidates.items():
            scores[name] += sum(other[column] < row[column] for other in candidates.values())
    selected = sorted(candidates, key=lambda name: (scores[name], name))[:count]
    return {"selected": selected, "rank_sums": scores, "screening_values": candidates,
            "zero_motion_is_a_diagnostic_not_a_gate": True}


def synthetic_checks():
    y, x = torch.meshgrid(torch.arange(64, dtype=torch.float64), torch.arange(64, dtype=torch.float64), indexing="ij")
    def gaussian(cx, cy):
        return torch.exp(-((x - cx).square() + (y - cy).square()) / 8)
    cases = {}
    for label, cx, cy in (("interior", 16.25, 24.25), ("edge", 0.25, 24.25)):
        for background in (0.0, 0.001):
            for secondary in (0.0, 0.3):
                shape = gaussian(cx, cy) + secondary * gaussian(cx + 8, cy) + background
                for amplitude in (0.1, 1.0, 2.0):
                    key = f"{label}_bg{background}_secondary{secondary}_amplitude{amplitude}"
                    heatmap = (amplitude * shape)[None]
                    cases[key] = {name: decode_coordinates(heatmap, spec)[0][0].tolist() for name, spec in DECODERS.items()}
    pair = torch.stack((gaussian(16.25, 24.25), gaussian(20.65, 24.25)))[:, None]
    table = {}
    for name, spec in {**DECODERS, "scaled100": DecoderSpec("softmax", beta=100)}.items():
        xy, _, _ = decode_coordinates(pair, spec)
        table[name] = {"x0": float(xy[0, 0, 0]), "dx": float(xy[1, 0, 0] - xy[0, 0, 0])}
    for name in ("relu", "power2", "expm1"):
        if abs(table[name]["dx"] - 4.4) > 1e-5:
            raise AssertionError(f"Gaussian displacement check failed: {name}")
    return {"kind": "synthetic_only", "document_table": table, "combined_stress_cases": cases,
            "decoder_specs": {name: asdict(spec) for name, spec in DECODERS.items()}}
