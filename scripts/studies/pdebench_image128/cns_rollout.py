"""Autoregressive CNS rollout logic independent of model family and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scripts.studies.pdebench_image128.cns_rollout_data import CnsTrajectoryWindow
from scripts.studies.pdebench_image128.normalization import denormalize_batch


@dataclass(frozen=True)
class CnsRolloutResult:
    initial_state_phys: np.ndarray
    true_phys: np.ndarray
    pred_phys: np.ndarray
    abs_error_phys: np.ndarray
    field_order: tuple[str, ...]
    frame_time_indices: tuple[int, ...]


def autoregressive_rollout(
    *,
    window: CnsTrajectoryWindow,
    predictor,
    state_stats: dict[str, Any],
) -> CnsRolloutResult:
    """Roll out a one-step CNS predictor without teacher forcing."""
    history = window.initial_history_norm.detach().cpu().float().clone()
    predictions: list[torch.Tensor] = []
    for _ in range(int(window.true_future_norm.shape[0])):
        pred_next = predictor(history).detach().cpu().float()
        predictions.append(pred_next)
        history = torch.cat([history[1:], pred_next.unsqueeze(0)], dim=0)
    pred_norm = torch.stack(predictions)
    pred_phys = denormalize_batch(pred_norm, state_stats).numpy()
    true_phys = window.true_future_phys.detach().cpu().numpy()
    initial_state = window.initial_history_phys[-1].detach().cpu().numpy()
    return CnsRolloutResult(
        initial_state_phys=initial_state,
        true_phys=true_phys,
        pred_phys=pred_phys,
        abs_error_phys=np.abs(pred_phys - true_phys),
        field_order=tuple(window.field_order),
        frame_time_indices=tuple(range(int(window.start_time), int(window.start_time + window.true_future_phys.shape[0]))),
    )
