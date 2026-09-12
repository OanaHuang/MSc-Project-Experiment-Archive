from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_metric: float, config: dict, history: list[dict]) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "config": config,
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def _load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location=device)


def load_model(path: Path, model, device):
    checkpoint = _load_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def resume_training(path: Path, model, optimizer, scheduler, device,
                    expected_experiment: str) -> dict:
    """Restore a full training state and return loop continuation metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = _load_checkpoint(path, device)
    saved_experiment = checkpoint.get("config", {}).get("id")
    if saved_experiment != expected_experiment:
        raise ValueError(
            f"Checkpoint experiment {saved_experiment!r} does not match "
            f"{expected_experiment!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    rng = checkpoint.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            # Loading the checkpoint with a CUDA map location also moves the
            # saved RNG tensors to CUDA. PyTorch's RNG restoration API expects
            # CPU ByteTensors, so normalize them before restoring each device.
            cuda_states = [state.detach().cpu().to(dtype=torch.uint8)
                           for state in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "completed_epoch": int(checkpoint["epoch"]),
        "best_metric": float(checkpoint["best_metric"]),
        "history": list(checkpoint.get("history", [])),
    }
