"""Reusable physics regularization helpers for PDEBench image-suite tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scripts.studies.pdebench_image128.normalization import denormalize_batch


def periodic_central_difference(field: torch.Tensor, *, spacing: float, dim: int) -> torch.Tensor:
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    return (torch.roll(field, shifts=-1, dims=dim) - torch.roll(field, shifts=1, dims=dim)) / (2.0 * spacing)


@dataclass(frozen=True)
class PhysicsRegularizationConfig:
    enabled: bool = False
    positivity_weight: float = 0.0
    continuity_weight: float = 0.0
    global_mass_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("positivity_weight", "continuity_weight", "global_mass_weight"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite, got {value}")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")

    def active_terms(self) -> list[str]:
        terms = []
        if self.positivity_weight > 0.0:
            terms.append("positivity")
        if self.continuity_weight > 0.0:
            terms.append("continuity")
        if self.global_mass_weight > 0.0:
            terms.append("global_mass")
        return terms

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "terms": self.active_terms(),
            "weights": {
                "positivity": float(self.positivity_weight),
                "continuity": float(self.continuity_weight),
                "global_mass": float(self.global_mass_weight),
            },
        }


@dataclass
class PhysicsLossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]

    def detached_payload(self) -> dict[str, Any]:
        return {
            "total": float(self.total.detach().cpu().item()),
            "terms": {name: float(value.detach().cpu().item()) for name, value in self.terms.items()},
            "weighted_terms": {
                name: float(value.detach().cpu().item()) for name, value in self.weighted_terms.items()
            },
        }


class DisabledPhysicsRegularizer:
    def __init__(self, config: PhysicsRegularizationConfig):
        self.config = config

    def compute(self, *, x_norm: torch.Tensor, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> PhysicsLossResult:
        del x_norm, target_norm
        zero = pred_norm.new_zeros(())
        return PhysicsLossResult(total=zero, terms={}, weighted_terms={})


class CFDCNSPhysicsRegularizer:
    def __init__(self, *, metadata: dict[str, Any], state_stats: dict[str, Any], config: PhysicsRegularizationConfig):
        field_order = [str(item) for item in metadata.get("field_order", [])]
        if field_order != ["density", "Vx", "Vy", "pressure"]:
            raise ValueError(f"unexpected CNS field order: {field_order}")
        if str(metadata.get("boundary_condition", "")).lower() != "periodic":
            raise ValueError("CNS physics regularization requires periodic boundaries")
        self.dx = float(metadata["dx"])
        self.dy = float(metadata["dy"])
        self.dt = float(metadata["dt"])
        self.history_len = int(metadata["history_len"])
        self.state_channels = int(len(field_order))
        self.state_stats = state_stats
        self.config = config

    def _denormalize(self, batch: torch.Tensor) -> torch.Tensor:
        return denormalize_batch(batch, self.state_stats)

    def _latest_history_state(self, x_norm: torch.Tensor) -> torch.Tensor:
        required_channels = self.history_len * self.state_channels
        if x_norm.shape[1] != required_channels:
            raise ValueError(
                f"x_norm has {x_norm.shape[1]} channels but expected {required_channels} for "
                f"history_len={self.history_len} and state_channels={self.state_channels}"
            )
        return x_norm[:, -self.state_channels :, :, :]

    @staticmethod
    def _split_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[:, 0], state[:, 1], state[:, 2], state[:, 3]

    def compute(self, *, x_norm: torch.Tensor, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> PhysicsLossResult:
        del target_norm
        previous = self._denormalize(self._latest_history_state(x_norm))
        predicted = self._denormalize(pred_norm)
        rho_prev, _, _, _ = self._split_state(previous)
        rho_hat, u_hat, v_hat, p_hat = self._split_state(predicted)

        zero = pred_norm.new_zeros(())
        terms: dict[str, torch.Tensor] = {}
        weighted_terms: dict[str, torch.Tensor] = {}

        positivity = torch.mean(torch.relu(-rho_hat).square()) + torch.mean(torch.relu(-p_hat).square())
        terms["positivity"] = positivity
        weighted_terms["positivity"] = positivity * float(self.config.positivity_weight)

        mass_flux_x = rho_hat * u_hat
        mass_flux_y = rho_hat * v_hat
        continuity_residual = (rho_hat - rho_prev) / self.dt
        continuity_residual = continuity_residual + periodic_central_difference(mass_flux_x, spacing=self.dx, dim=-1)
        continuity_residual = continuity_residual + periodic_central_difference(mass_flux_y, spacing=self.dy, dim=-2)
        continuity = torch.mean(continuity_residual.square())
        terms["continuity"] = continuity
        weighted_terms["continuity"] = continuity * float(self.config.continuity_weight)

        mass_prev = rho_prev.sum(dim=(-2, -1)) * (self.dx * self.dy)
        mass_hat = rho_hat.sum(dim=(-2, -1)) * (self.dx * self.dy)
        global_mass = torch.mean(torch.abs(mass_hat - mass_prev) / torch.clamp(torch.abs(mass_prev), min=1e-12))
        terms["global_mass"] = global_mass
        weighted_terms["global_mass"] = global_mass * float(self.config.global_mass_weight)

        total = zero
        for value in weighted_terms.values():
            total = total + value
        return PhysicsLossResult(total=total, terms=terms, weighted_terms=weighted_terms)


def build_physics_regularizer(
    *,
    task_id: str,
    metadata: dict[str, Any],
    state_stats: dict[str, Any],
    config: PhysicsRegularizationConfig,
):
    if not config.enabled:
        return DisabledPhysicsRegularizer(config)
    if task_id != "2d_cfd_cns":
        raise ValueError(f"physics regularization is not supported for task_id={task_id!r}")
    return CFDCNSPhysicsRegularizer(metadata=metadata, state_stats=state_stats, config=config)
