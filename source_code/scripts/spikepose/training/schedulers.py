from __future__ import annotations

import math

import torch


class PlateauScheduler:
    def __init__(self, optimizer, *, factor=0.5, patience=8, min_lr=1e-6):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, min_lr=min_lr)

    def set_epoch(self, epoch: int) -> None:
        return None

    def step(self, metric: float) -> None:
        self.scheduler.step(metric)

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.scheduler.load_state_dict(state)


class WarmupCosineScheduler:
    """Epoch scheduler matching the legacy BF/residual training scripts."""

    def __init__(self, optimizer, *, total_epochs: int, warmup_epochs=2,
                 base_lr=3e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.last_epoch = 0

    def _learning_rate(self, epoch: int) -> float:
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            return self.base_lr * epoch / self.warmup_epochs
        progress = ((epoch - self.warmup_epochs) /
                    max(self.total_epochs - self.warmup_epochs, 1))
        cosine = self.base_lr * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(self.min_lr, cosine)

    def set_epoch(self, epoch: int) -> None:
        self.last_epoch = int(epoch)
        learning_rate = self._learning_rate(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self, metric: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))


class ConstantScheduler:
    """Keep a fixed learning rate for every continuation epoch."""

    def __init__(self, optimizer, *, learning_rate: float):
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.last_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.last_epoch = int(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def step(self, metric: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state.get("last_epoch", 0))


class CosineRestartScheduler:
    """Cosine fine-tuning cycle beginning at an explicit global epoch."""

    def __init__(self, optimizer, *, start_epoch: int, end_epoch: int,
                 base_lr: float, min_lr: float):
        if end_epoch <= start_epoch:
            raise ValueError("cosine restart end_epoch must exceed start_epoch")
        self.optimizer = optimizer
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.last_epoch = self.start_epoch - 1

    def _learning_rate(self, epoch: int) -> float:
        progress = ((epoch - self.start_epoch) /
                    max(self.end_epoch - self.start_epoch, 1))
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress))

    def set_epoch(self, epoch: int) -> None:
        self.last_epoch = int(epoch)
        learning_rate = self._learning_rate(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self, metric: float) -> None:
        return None

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.last_epoch = int(state.get("last_epoch", self.start_epoch - 1))


def build_scheduler(optimizer, training: dict):
    config = training.get("scheduler", {"kind": "plateau"})
    kind = config.get("kind", "plateau")
    if kind == "plateau":
        return PlateauScheduler(
            optimizer, factor=config.get("factor", 0.5),
            patience=config.get("patience", 8), min_lr=config.get("min_lr", 1e-6))
    if kind == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer, total_epochs=training["epochs"],
            warmup_epochs=config.get("warmup_epochs", 2),
            base_lr=training["learning_rate"], min_lr=config.get("min_lr", 1e-6))
    if kind == "constant":
        return ConstantScheduler(
            optimizer, learning_rate=config.get(
                "learning_rate", training["learning_rate"]),
        )
    if kind == "cosine_restart":
        return CosineRestartScheduler(
            optimizer,
            start_epoch=config["start_epoch"],
            end_epoch=training["epochs"],
            base_lr=config.get("base_lr", training["learning_rate"]),
            min_lr=config.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unknown scheduler kind: {kind}")
