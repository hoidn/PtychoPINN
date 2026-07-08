"""Model-based Born inverse baseline for the BRDT candidate lane.

This module solves the per-sample inverse problem directly against the
locked local BRDT forward operator. It is the non-learned classical
baseline for the repo's own measurement convention; it does not use
ODTbrain or target images during optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class ModelBasedInverseConfig:
    """Configuration for direct physical-q optimization."""

    steps: int = 300
    learning_rate: float = 5e-2
    tv_weight: float = 1e-4
    l2_weight: float = 1e-6
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    init: str = "zeros"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def relative_physics_l2(
    pred: torch.Tensor, obs: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Mean squared relative L2 residual between predicted and observed sinograms."""
    diff = pred - obs
    num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(obs.reshape(obs.shape[0], -1), dim=1).clamp_min(eps)
    return torch.mean((num / den) ** 2)


def total_variation_l1(q: torch.Tensor) -> torch.Tensor:
    """Anisotropic L1 total variation for physical-q images."""
    dy = torch.mean(torch.abs(q[..., 1:, :] - q[..., :-1, :]))
    dx = torch.mean(torch.abs(q[..., :, 1:] - q[..., :, :-1]))
    return dx + dy


def _apply_bounds(q: torch.Tensor, config: ModelBasedInverseConfig) -> torch.Tensor:
    if config.clamp_min is not None or config.clamp_max is not None:
        q = torch.clamp(q, min=config.clamp_min, max=config.clamp_max)
    return q


def optimize_born_inverse_batch(
    *,
    sinogram_obs: torch.Tensor,
    operator: torch.nn.Module,
    config: ModelBasedInverseConfig,
    initial_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Optimize physical ``q`` directly against observed BRDT sinograms."""
    device = sinogram_obs.device
    dtype = sinogram_obs.dtype
    if initial_q is None:
        batch = int(sinogram_obs.shape[0])
        grid_size = int(operator.grid_size)
        q_var = torch.zeros(batch, 1, grid_size, grid_size, device=device, dtype=dtype)
    else:
        q_var = initial_q.detach().to(device=device, dtype=dtype).clone()
    q_var.requires_grad_(True)

    with torch.no_grad():
        initial_loss = relative_physics_l2(
            operator(_apply_bounds(q_var, config)), sinogram_obs
        ).item()

    opt = torch.optim.Adam([q_var], lr=float(config.learning_rate))
    last_total = float("nan")
    last_phys = float("nan")
    for _ in range(int(config.steps)):
        q_eff = _apply_bounds(q_var, config)
        pred = operator(q_eff)
        phys = relative_physics_l2(pred, sinogram_obs)
        tv = total_variation_l1(q_eff)
        l2 = torch.mean(q_eff * q_eff)
        total = phys + float(config.tv_weight) * tv + float(config.l2_weight) * l2
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        with torch.no_grad():
            q_var.copy_(_apply_bounds(q_var, config))
        last_total = float(total.detach().cpu().item())
        last_phys = float(phys.detach().cpu().item())

    q_final = _apply_bounds(q_var.detach(), config)
    with torch.no_grad():
        final_phys = float(relative_physics_l2(operator(q_final), sinogram_obs).item())
    info = {
        "solver": "adam_direct_q",
        "config": config.as_dict(),
        "initial_relative_physics_l2": float(initial_loss),
        "final_relative_physics_l2": final_phys,
        "final_total_loss": float(last_total),
    }
    return q_final, info
