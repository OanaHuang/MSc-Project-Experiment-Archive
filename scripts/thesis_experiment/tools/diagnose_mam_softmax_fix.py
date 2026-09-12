#!/usr/bin/env python3
"""Cache identical validation clips, diagnose decoders, or evaluate a trained MAM.

Only canonical validation data are used. Never evaluates/chooses on test data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from spikepose_thesis.artifacts.manager import locate_source_run
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path
from spikepose_thesis.data import build_dataset
from spikepose_thesis.evaluation.softmax_fix import DECODERS, MotionMetrics, choose_candidates, sha256, synthetic_checks, write_json
from spikepose_thesis.evaluation.mpii.metrics import prediction_to_keypoints
from spikepose_thesis.models import build_model
from spikepose_thesis.models.mam_v2.coordinate_decoding import decode_coordinates
from spikepose_thesis.training.checkpoint import load_model, load_spatial_model

SOURCE = "mamv2_fullcs_p00_source"
SOURCE_SHA256 = "004ca2da72c22d8c77ccf9688a3e325adbbd61eada16a8df0842318d6d2f4990"
META_KEYS = ("temporal_keypoints", "temporal_visibility", "temporal_frame_indices", "temporal_frame_positions", "person_bbox", "video_id", "clip_id", "person_id", "sample_id")


def targets(batch):
    target = batch["temporal_keypoints"].transpose(0, 1).float() / 4
    valid = (batch["temporal_visibility"].transpose(0, 1) > 0) & torch.isfinite(target).all(-1)
    valid &= (target >= 0).all(-1) & (target < 64).all(-1)
    return target, valid


def dump_metrics(output, summaries):
    write_json(output / "metrics.json", summaries)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("decoder", "metric", "count", "mean"))
        for name, summary in summaries.items():
            for key, value in summary.items():
                writer.writerow((name, key, value["count"], value["mean"]))


@torch.inference_mode()
def diagnose(args):
    output = args.output.resolve()
    if (output / "complete.json").exists():
        raise FileExistsError(f"Completed diagnosis already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("Use a new diagnostic directory after an interrupted cache build")
    cache.mkdir(exist_ok=True)
    write_json(output / "synthetic.json", synthetic_checks())
    config = load_experiment("mam_softmax_fix_softmax")
    source = locate_source_run(SOURCE, 42) / "checkpoints" / "best.pt"
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError("Source checkpoint does not match the paper's independent frame model")
    model = build_model(config).to(args.device).eval()
    metadata, report = load_spatial_model(source, model, args.device)
    dataset = build_dataset(config, "validation")
    groups = {}
    for index, sample in enumerate(dataset.samples):
        sid = str(sample["sample_id"])
        groups.setdefault(sid[-4:], []).append((sid, index))
    rng = random.Random(42)
    selected = []
    for action in sorted(groups):
        options = sorted(groups[action])
        if len(options) < args.videos_per_action:
            raise RuntimeError(f"Insufficient validation videos for {action}")
        selected.extend(rng.sample(options, args.videos_per_action))
    chosen = {index for _, index in selected}
    indices = [i for i, item in enumerate(dataset.frame_index) if int(item[0]) in chosen]
    if len(indices) != len(selected) * 2:
        raise RuntimeError("Expected exactly two independent 16-frame clips per selected video")
    manifest = {"split": "validation", "selection_seed": 42, "videos": [sid for sid, _ in selected],
                "clips": len(indices), "frames": len(indices) * 16,
                "protocol": "canonical_validation_2x16_clip_tube_crop_no_cross_clip_differences",
                "frames_are_not_complete_videos": True}
    write_json(output / "manifest.json", manifest)
    provenance = {"source_checkpoint": str(source), "source_sha256": source_hash,
                  "source_epoch": int(metadata["epoch"]), "spatial_load_report": report,
                  "validation_metadata_sha256": sha256(resolve_project_path(config["data"]["validation_metadata"])),
                  "effective_config": config, "torch": torch.__version__, "python": sys.version,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
                  "source_files": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted((PROJECT_ROOT / "scripts/thesis_experiment/src").rglob("*.py"))},
                  "argv": sys.argv, "inference_dtype": "float32"}
    write_json(output / "provenance.json", provenance)
    loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=args.workers, shuffle=False)
    accumulators = {name: MotionMetrics() for name in DECODERS}
    batches = []
    for i, batch in enumerate(loader):
        frames = batch["temporal_frame_indices"]
        if not bool((frames[:, 1:] - frames[:, :-1] == 1).all()):
            raise RuntimeError("Diagnostic batch contains nonadjacent physical frames")
        raw = model.forward_spatial_per_step(batch["image"].to(args.device)).float()
        target, visible = targets(batch)
        for name, spec in DECODERS.items():
            xy, _, valid = decode_coordinates(raw, spec)
            accumulators[name].update(xy.cpu(), valid.cpu(), target, visible)
        saved = {key: batch[key] for key in META_KEYS}
        saved["raw_heatmaps"] = raw.cpu()
        path = cache / f"batch_{i:05d}.pt"
        torch.save(saved, path)
        batches.append({"name": path.name, "sha256": sha256(path)})
        print(f"diagnosis batch={i+1}/{len(loader)}", flush=True)
    summaries = {name: metric.summary() for name, metric in accumulators.items()}
    dump_metrics(output, summaries)
    selection = choose_candidates(summaries)
    write_json(output / "selection.json", selection)
    write_json(output / "complete.json", {"completed_at": time.time(), "batches": batches,
        "manifest_sha256": sha256(output / "manifest.json"), "selection": selection})
    print(json.dumps(selection), flush=True)


@torch.inference_mode()
def evaluate(args):
    config = load_experiment(args.experiment)
    run = locate_source_run(args.experiment, 42)
    checkpoint = run / "checkpoints" / "best.pt"
    model = build_model(config).to(args.device).eval()
    metadata = load_model(checkpoint, model, args.device)
    source = locate_source_run(SOURCE, 42) / "checkpoints" / "best.pt"
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError("Source fingerprint changed")
    original = torch.load(source, map_location="cpu", weights_only=False)["model_state_dict"]
    for name, value in model.state_dict().items():
        if not name.startswith("mam.") and not torch.equal(value.cpu(), original[name].cpu()):
            raise RuntimeError(f"Frozen spatial weights or buffers changed: {name}")
    diagnosis = args.diagnosis.resolve()
    completion = json.loads((diagnosis / "complete.json").read_text())
    if sha256(diagnosis / "manifest.json") != completion["manifest_sha256"]:
        raise RuntimeError("Diagnostic manifest changed")
    modes = ("trained", "warp_disabled") if model.mam.aligned else ("trained",)
    motions = {mode: MotionMetrics() for mode in modes}
    scores = {mode: {"correct": 0, "valid": 0, "velocity_sum": 0.0, "velocity_n": 0,
                     "acceleration_sum": 0.0, "acceleration_n": 0} for mode in modes}
    per_clip = []
    for entry in completion["batches"]:
        path = diagnosis / "cache" / entry["name"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Changed cached heatmaps: {path}")
        batch = torch.load(path, map_location="cpu", weights_only=False)
        raw = batch["raw_heatmaps"].to(args.device)
        target, visible = targets(batch)
        xy, _, valid = decode_coordinates(raw, model.mam.decoder_spec)
        for mode in modes:
            original_mode = model.mam.mode
            if mode == "warp_disabled":
                model.mam.mode = "mam_v2_noalign"
            try:
                result = model.mam.forward_sequence(raw)
            finally:
                model.mam.mode = original_mode
            motions[mode].update(xy.cpu(), valid.cpu(), target, visible, result.final_offset.cpu(), result.residual_offset.cpu())
            t, b, j, h, w = result.heatmap.shape
            decoded, _ = prediction_to_keypoints(result.heatmap.flatten(0, 1), 256, decoder="dark")
            decoded = decoded.reshape(t, b, j, 2)
            bbox = batch["person_bbox"].numpy()
            extent = (bbox[:, 2:] - bbox[:, :2])[None, :, None] / 256
            prediction = decoded * extent + bbox[None, :, None, :2]
            gt = target.numpy() * 4 * extent + bbox[None, :, None, :2]
            vis = visible.numpy()
            for lane in range(b):
                scale = np.linalg.norm(gt[:, lane, 9] - gt[:, lane, 8], axis=-1)
                distance = np.linalg.norm(prediction[:, lane] - gt[:, lane], axis=-1)
                mask = vis[:, lane] & vis[:, lane, 8:9] & vis[:, lane, 9:10] & (scale[:, None] > 0)
                correct = int(((distance <= scale[:, None] * .5) & mask).sum())
                score = scores[mode]
                score["correct"] += correct
                score["valid"] += int(mask.sum())
                row = {"mode": mode, "video": str(batch["video_id"][lane]), "clip": str(batch["clip_id"][lane]),
                       "pck_correct": correct, "pck_valid": int(mask.sum())}
                for order, label in ((1, "velocity"), (2, "acceleration")):
                    errors = np.linalg.norm(np.diff(prediction[:, lane] - gt[:, lane], n=order, axis=0), axis=-1)
                    vm = vis[:t-order, lane].copy()
                    for k in range(1, order+1):
                        vm &= vis[k:t-order+k, lane]
                    score[label + "_sum"] += float(errors[vm].sum())
                    score[label + "_n"] += int(vm.sum())
                    row[label + "_sum"] = float(errors[vm].sum())
                    row[label + "_n"] = int(vm.sum())
                per_clip.append(row)
    summary = {mode: {**score, "pckhb": score["correct"] / max(score["valid"], 1),
                     "mpjve": score["velocity_sum"] / max(score["velocity_n"], 1),
                     "mpjacce": score["acceleration_sum"] / max(score["acceleration_n"], 1)} for mode, score in scores.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    dump_metrics(args.output, {mode: metric.summary() for mode, metric in motions.items()})
    write_json(args.output / "summary.json", {"experiment": args.experiment, "checkpoint_sha256": sha256(checkpoint),
        "epoch": int(metadata["epoch"]), "protocol": "fixed_validation_2x16_clip_tube_crop_frame_weighted",
        "spatial_weights_unchanged": True, "scores": summary, "per_clip": per_clip})
    write_json(args.output / "complete.json", {"completed_at": time.time(), "experiment": args.experiment})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--videos-per-action", type=int, default=2)
    parser.add_argument("--diagnosis", type=Path)
    parser.add_argument("--experiment")
    args = parser.parse_args()
    if args.mode == "evaluate" and (not args.diagnosis or not args.experiment):
        parser.error("evaluate requires --diagnosis and --experiment")
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    if args.mode == "diagnose":
        diagnose(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
