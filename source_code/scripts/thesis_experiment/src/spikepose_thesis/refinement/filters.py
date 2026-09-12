from __future__ import annotations

import math

import numpy as np


def ema_shared(value: np.ndarray, alpha: float) -> np.ndarray:
    """Apply one causal EMA coefficient to every coordinate and joint."""
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha must lie in (0, 1]")
    if not len(value):
        return value.copy()
    result = value.copy()
    for index in range(1, len(value)):
        result[index] = (
            float(alpha) * value[index]
            + (1.0 - float(alpha)) * result[index - 1]
        )
    return result.astype(np.float32)


def savgol_offline(value: np.ndarray, window: int, order: int) -> np.ndarray:
    if len(value) < window:
        return value.copy()
    radius = window // 2
    result = value.copy()
    for index in range(len(value)):
        start = max(0, min(index - radius, len(value) - window))
        history = value[start:start + window]
        x = np.arange(len(history), dtype=np.float64)
        evaluation = float(index - start)
        flat = history.reshape(len(history), -1)
        estimate = [
            np.polyval(np.polyfit(x, flat[:, column], order), evaluation)
            for column in range(flat.shape[1])
        ]
        result[index] = np.asarray(estimate).reshape(value.shape[1:])
    return result.astype(np.float32)


def savgol_causal(value: np.ndarray, window: int, order: int) -> np.ndarray:
    """Past-only polynomial fit evaluated at the newest observation."""
    result = value.copy()
    for index in range(1, len(value)):
        start = max(0, index - window + 1)
        history = value[start:index + 1]
        if len(history) <= order:
            result[index] = history.mean(0)
            continue
        x = np.arange(len(history), dtype=np.float64)
        flat = history.reshape(len(history), -1)
        estimate = [np.polyval(np.polyfit(x, flat[:, column], order), x[-1])
                    for column in range(flat.shape[1])]
        result[index] = np.asarray(estimate).reshape(value.shape[1:])
    return result.astype(np.float32)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float = 1.0,
                 frequency: float = 30.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.frequency = float(frequency)

    def _alpha(self, cutoff) -> np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff))
        return 1.0 / (1.0 + tau * self.frequency)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        if not len(value):
            return value.copy()
        result = value.copy()
        derivative = np.zeros_like(value[0])
        for index in range(1, len(value)):
            raw_derivative = (value[index] - value[index - 1]) * self.frequency
            derivative_alpha = self._alpha(self.derivative_cutoff)
            derivative = derivative_alpha * raw_derivative + (1 - derivative_alpha) * derivative
            cutoff = self.min_cutoff + self.beta * np.abs(derivative)
            alpha = self._alpha(cutoff)
            result[index] = alpha * value[index] + (1 - alpha) * result[index - 1]
        return result.astype(np.float32)
