from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.NTU_RGBD.evaluation.metrics import compute_pckhn
from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce, save_nacce
from scripts.NTU_RGBD.jatc_pilot import (
    BETA_GRID, JointAdaptiveCoordinateLIF, PilotWindowDataset, causal_ema,
    dynamic_decays, load_pilot_sequences, refine_causally,
)


EXPERIMENTS = ("jp0", "jp1", "jp2", "jp3", "jp4")


def evaluate(prediction: np.ndarray, sequences) -> dict:
    pck = compute_pckhn(prediction, sequences.gt, sequences.visibility,
                        sequences.head_length, 0.5)
    temporal = compute_nacce(
        prediction, sequences.gt, sequences.visibility, sequences.head_length,
        sequences.sample_ids, sequences.frame_indices, NTU_JOINT_NAMES,
    )
    valid, correct = pck["valid_mask"], pck["correct_matrix"]
    per_joint_pckhn = {
        name: float(correct[:, joint].sum() / max(int(valid[:, joint].sum()), 1))
        for joint, name in enumerate(NTU_JOINT_NAMES)
    }
    return {
        "pckhn": float(correct.sum() / max(int(valid.sum()), 1)),
        "valid_joints": int(valid.sum()), "per_joint_pckhn": per_joint_pckhn,
        **temporal,
    }


def save_result(output_dir: Path, prediction: np.ndarray, sequences, summary: dict,
                **extra_arrays: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics").mkdir()
    (output_dir / "metrics/pckhn_summary.json").write_text(
        json.dumps({key: value for key, value in summary.items()
                    if key in ("pckhn", "valid_joints", "per_joint_pckhn")}, indent=2),
        encoding="utf-8",
    )
    save_nacce(summary, output_dir / "metrics/temporal_metrics")
    np.savez_compressed(
        output_dir / "metrics/prediction_matrices.npz", pred=prediction,
        confidence=sequences.confidence, gt=sequences.gt,
        visibility=sequences.visibility, head_length=sequences.head_length,
        sample_ids=sequences.sample_ids, frame_indices=sequences.frame_indices,
        **extra_arrays,
    )
    (output_dir / "status.json").write_text(json.dumps({
        "status": "completed", "metrics": {
            "pckhn": summary["pckhn"], "nacce": summary["nacce"],
            "predicted_to_gt_acceleration_ratio": summary["predicted_to_gt_acceleration_ratio"],
        },
    }, indent=2), encoding="utf-8")


def constrained_choice(rows: list[dict], baseline_pck: float,
                       tolerance: float = 0.005) -> dict:
    finite = [row for row in rows if np.isfinite(row["pckhn"]) and np.isfinite(row["nacce"])]
    if not finite:
        raise ValueError("all temporal-decay candidates produced non-finite metrics")
    eligible = [row for row in finite if row["pckhn"] >= baseline_pck - tolerance]
    return min(eligible or finite, key=lambda row: (row["nacce"], -row["pckhn"]))


def run_jp0(validation, output_dir: Path) -> None:
    summary = evaluate(validation.pred, validation)
    save_result(output_dir, validation.pred, validation, summary)


def run_jp1(train, validation, output_dir: Path) -> None:
    raw = evaluate(train.pred, train)
    candidates = []
    for beta in BETA_GRID:
        prediction = causal_ema(train.pred, train.groups, float(beta))
        result = evaluate(prediction, train)
        candidates.append({"beta": float(beta), "pckhn": result["pckhn"],
                           "nacce": result["nacce"]})
    selected = constrained_choice(candidates, raw["pckhn"])
    prediction = causal_ema(validation.pred, validation.groups, selected["beta"])
    summary = evaluate(prediction, validation)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selection.json").write_text(json.dumps({
        "selection_split": "train", "pckhn_tolerance": 0.005,
        "selected_beta": selected["beta"], "candidates": candidates,
    }, indent=2), encoding="utf-8")
    # save_result owns directory creation, so complete into a temporary child then lift files.
    result_dir = output_dir / "result"
    save_result(result_dir, prediction, validation, summary)
    for child in result_dir.iterdir():
        child.rename(output_dir / child.name)
    result_dir.rmdir()


def run_jp2(train, validation, output_dir: Path) -> None:
    raw = evaluate(train.pred, train)
    candidate_results = []
    candidate_predictions = []
    for beta in BETA_GRID:
        prediction = causal_ema(train.pred, train.groups, float(beta))
        candidate_predictions.append(prediction)
        candidate_results.append(evaluate(prediction, train))
    selected = np.zeros(len(NTU_JOINT_NAMES), dtype=np.float32)
    selection = {}
    for joint, name in enumerate(NTU_JOINT_NAMES):
        rows = [{
            "beta": float(beta),
            "pckhn": result["per_joint_pckhn"][name],
            "nacce": result["per_joint"][name]["nacce"],
        } for beta, result in zip(BETA_GRID, candidate_results)]
        best = constrained_choice(rows, raw["per_joint_pckhn"][name])
        selected[joint] = best["beta"]
        selection[name] = {"selected_beta": best["beta"], "candidates": rows}
    prediction = causal_ema(validation.pred, validation.groups, selected)
    summary = evaluate(prediction, validation)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_beta_per_joint.json").write_text(
        json.dumps({"selection_split": "train", "joints": selection}, indent=2),
        encoding="utf-8",
    )
    result_dir = output_dir / "result"
    save_result(result_dir, prediction, validation, summary, selected_beta=selected)
    for child in result_dir.iterdir():
        child.rename(output_dir / child.name)
    result_dir.rmdir()


def load_selected_beta(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))["joints"]
    return np.asarray([value[name]["selected_beta"] for name in NTU_JOINT_NAMES],
                      dtype=np.float32)


def run_jp3(train, validation, output_dir: Path, beta_path: Path) -> None:
    base_beta = load_selected_beta(beta_path)
    raw = evaluate(train.pred, train)
    candidates = []
    for confidence_scale in (0.05, 0.10, 0.20):
        for velocity_scale in (0.05, 0.10, 0.20):
            decay = dynamic_decays(train, base_beta, confidence_scale, velocity_scale)
            prediction = causal_ema(train.pred, train.groups, base_beta, decay)
            result = evaluate(prediction, train)
            candidates.append({
                "confidence_scale": confidence_scale, "velocity_scale": velocity_scale,
                "pckhn": result["pckhn"], "nacce": result["nacce"],
            })
    selected = constrained_choice(candidates, raw["pckhn"])
    decay = dynamic_decays(validation, base_beta, selected["confidence_scale"],
                           selected["velocity_scale"])
    prediction = causal_ema(validation.pred, validation.groups, base_beta, decay)
    summary = evaluate(prediction, validation)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selection.json").write_text(json.dumps({
        "selection_split": "train", "selected": selected, "candidates": candidates,
    }, indent=2), encoding="utf-8")
    result_dir = output_dir / "result"
    save_result(result_dir, prediction, validation, summary, dynamic_beta=decay,
                selected_beta=base_beta)
    for child in result_dir.iterdir():
        child.rename(output_dir / child.name)
    result_dir.rmdir()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              visibility: torch.Tensor) -> torch.Tensor:
    mask = visibility[..., None].bool()
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    error = torch.where(mask, (prediction - target).abs(), 0.0)
    return error.sum() / mask.sum().clamp_min(1)


def run_jp4(train, validation, output_dir: Path, beta_path: Path, args) -> None:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base_beta = load_selected_beta(beta_path)
    model = JointAdaptiveCoordinateLIF(25, base_beta, args.hidden_channels).to(device)
    dataset = PilotWindowDataset(train, args.window_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    (output_dir / "resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_loss, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            pred = batch["pred"].to(device)
            gt = batch["gt"].to(device)
            visibility = batch["visibility"].to(device)
            head = batch["head_length"].to(device)
            refined, beta = model(pred, batch["confidence"].to(device), head)
            position = masked_l1(refined / 1000.0, gt / 1000.0, visibility)
            accel_visibility = visibility[:, :-2] * visibility[:, 1:-1] * visibility[:, 2:]
            acceleration = masked_l1(
                refined.diff(n=2, dim=1) / head[:, 1:-1, None, None].clamp_min(1e-6),
                gt.diff(n=2, dim=1) / head[:, 1:-1, None, None].clamp_min(1e-6),
                accel_visibility,
            )
            base_decay = model.beta_min + (model.beta_max - model.beta_min) * model.base_logit.sigmoid()
            regularization = (beta - base_decay[None, None]).square().mean()
            loss = position + args.acceleration_weight * acceleration + 1e-3 * regularization
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(pred); count += len(pred)
        scheduler.step(); epoch_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config,
                        "base_beta": base_beta}, output_dir / "best.pt")
        print(f"jp4 seed={args.seed} epoch={epoch}/{args.epochs} loss={epoch_loss:.6f}",
              flush=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    saved = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    prediction, beta_rows = refine_causally(model, validation, device)
    summary = evaluate(prediction, validation)
    result_dir = output_dir / "result"
    save_result(result_dir, prediction, validation, summary, dynamic_beta=beta_rows,
                selected_beta=base_beta)
    for child in result_dir.iterdir():
        child.rename(output_dir / child.name)
    result_dir.rmdir()
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    status.update({"id": "jp4", "seed": args.seed, "best_epoch": saved["epoch"]})
    (output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one NTU JATC pilot experiment")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--train-matrices", type=Path, required=True)
    parser.add_argument("--validation-matrices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jp2-beta", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--acceleration-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    require_confidence = args.experiment in {"jp3", "jp4"}
    train = load_pilot_sequences(args.train_matrices, require_confidence=require_confidence)
    validation = load_pilot_sequences(args.validation_matrices,
                                      require_confidence=require_confidence)
    if args.experiment == "jp0": run_jp0(validation, args.output_dir)
    elif args.experiment == "jp1": run_jp1(train, validation, args.output_dir)
    elif args.experiment == "jp2": run_jp2(train, validation, args.output_dir)
    elif args.experiment == "jp3":
        if args.jp2_beta is None: raise ValueError("jp3 requires --jp2-beta")
        run_jp3(train, validation, args.output_dir, args.jp2_beta)
    else:
        if args.jp2_beta is None: raise ValueError("jp4 requires --jp2-beta")
        run_jp4(train, validation, args.output_dir, args.jp2_beta, args)


if __name__ == "__main__":
    main()
