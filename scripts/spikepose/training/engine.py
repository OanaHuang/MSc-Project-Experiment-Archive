from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None,
              gradient_clip: float | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target_key = getattr(criterion, "target_key", "heatmaps")
            target = batch[target_key].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            temporal = getattr(getattr(model, "config", None), "temporal", None)
            early_weights = tuple(float(item) for item in getattr(
                temporal, "early_classifier_weights", (),
            ))
            auxiliary_weight = float(getattr(
                temporal,
                "intermediate_loss_weight", 0.0,
            ))
            kinematic_weight = float(getattr(
                temporal, "kinematic_loss_weight", 0.0,
            ))
            if training and early_weights:
                prediction, early_predictions = model(
                    image, return_early_classifiers=True,
                )
                final_weight = 1.0 - sum(early_weights)
                loss = final_weight * criterion(prediction, target, visibility)
                loss = loss + sum(
                    weight * criterion(item, target, visibility)
                    for weight, item in zip(early_weights, early_predictions)
                )
            elif auxiliary_weight > 0.0 or kinematic_weight > 0.0:
                prediction, temporal_predictions = model(
                    image, return_intermediates=True,
                )
                loss = criterion(prediction, target, visibility)
                intermediates = temporal_predictions[:-1]
                if intermediates:
                    if auxiliary_weight:
                        auxiliary = sum(
                            criterion(item, target, visibility) for item in intermediates
                        ) / len(intermediates)
                        loss = loss + auxiliary_weight * auxiliary
                if kinematic_weight:
                    predicted_sequence = torch.stack(temporal_predictions)
                    target_sequence = batch["temporal_heatmaps"].to(
                        device, non_blocking=True,
                    ).transpose(0, 1)
                    sequence_visibility = batch["temporal_visibility"].to(
                        device, non_blocking=True,
                    ).transpose(0, 1)
                    predicted_acceleration = predicted_sequence.diff(n=2, dim=0)
                    target_acceleration = target_sequence.diff(n=2, dim=0)
                    acceleration_visibility = (
                        sequence_visibility[:-2]
                        * sequence_visibility[1:-1]
                        * sequence_visibility[2:]
                    )
                    acceleration_loss = criterion(
                        predicted_acceleration.flatten(0, 1),
                        target_acceleration.flatten(0, 1),
                        acceleration_visibility.flatten(0, 1),
                    )
                    loss = loss + kinematic_weight * acceleration_loss
            else:
                loss = criterion(model(image), target, visibility)
            if training:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            total += float(loss.detach()) * len(image)
            count += len(image)
    return total / max(count, 1)


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs: int, run_dir: Path, config: dict, start_epoch: int = 1,
          history: list[dict] | None = None,
          best_metric: float = float("inf"),
          checkpoint_epochs: set[int] | None = None) -> list[dict]:
    """Train through ``epochs`` total epochs, optionally continuing a run."""
    history = list(history or [])
    best = float(best_metric)
    if start_epoch < 1:
        raise ValueError("start_epoch must be positive")
    if epochs < start_epoch:
        raise ValueError(
            f"Target total epochs ({epochs}) must be at least {start_epoch} "
            "when resuming"
        )
    checkpoint_dir = run_dir / "checkpoints"
    gradient_clip = config["training"].get("gradient_clip")
    checkpoint_epochs = set(checkpoint_epochs or ())
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        if hasattr(scheduler, "set_epoch"):
            scheduler.set_epoch(epoch)
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, gradient_clip)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler,
                        epoch, min(best, val_loss), config, history)
        if epoch in checkpoint_epochs:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch}.pt", model, optimizer,
                scheduler, epoch, min(best, val_loss), config, history,
            )
        if val_loss < best:
            best = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler,
                            epoch, best, config, history)
        write_history(run_dir / "logs" / "train.csv", history)
        print(f"Epoch {epoch}/{epochs} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
    return history
