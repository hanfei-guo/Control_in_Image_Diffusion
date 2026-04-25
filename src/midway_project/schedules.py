from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def sigmoid_weight(progress: float, tau: float, sharpness: float) -> float:
    return float(1.0 / (1.0 + np.exp(-sharpness * (progress - tau))))


def control_weight(progress: float, tau: float, sharpness: float, max_scale: float) -> float:
    return float(max_scale * (1.0 - sigmoid_weight(progress, tau, sharpness)))


def semantic_weight(progress: float, tau: float, sharpness: float, max_scale: float) -> float:
    return float(max_scale * sigmoid_weight(progress, tau, sharpness))


def make_control_staircase(
    tau: float,
    sharpness: float,
    max_scale: float,
    segments: int,
) -> tuple[list[float], list[float]]:
    boundaries = np.linspace(1.0 / segments, 1.0, segments)
    midpoints = np.linspace(0.5 / segments, 1.0 - 0.5 / segments, segments)
    desired = [control_weight(progress, tau, sharpness, max_scale) for progress in midpoints]

    deltas: list[float] = []
    for idx, value in enumerate(desired):
        next_value = desired[idx + 1] if idx + 1 < len(desired) else 0.0
        deltas.append(max(0.0, value - next_value))
    return deltas, boundaries.tolist()


@dataclass(slots=True)
class SmoothScheduleConfig:
    tau: float
    sharpness: float
    control_max_scale: float
    ip_max_scale: float


def format_mode_value(value: float) -> str:
    text = str(value).replace(".", "p")
    return text.replace("-", "m")


def build_mode_name(prefix: str, params: dict[str, float]) -> str:
    tokens = [prefix]
    for key, value in params.items():
        tokens.append(f"{key}_{format_mode_value(value)}")
    return "__".join(tokens)


def stage_best(summary, exclude_modes: Iterable[str] | None = None):
    exclude_modes = set(exclude_modes or [])
    filtered = summary.loc[~summary["mode"].isin(exclude_modes)].copy()
    if filtered.empty:
        raise ValueError("No modes available after filtering.")
    return filtered.sort_values("balanced_score", ascending=False).iloc[0]
