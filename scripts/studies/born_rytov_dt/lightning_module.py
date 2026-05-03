"""BRDT task-local training wrapper.

This module owns the supervised + Born-consistency loss for the bounded
preflight. It deliberately routes physical-q semantics through the
checked-in :mod:`scripts.studies.born_rytov_dt.dataset_contract`
helpers so the unnormalize-before-physics rule and the normalized-q
operator-input guard live in exactly one place.

Naming note: this is "lightning_module" only by file convention with the
PDEBench studies; it does not depend on PyTorch Lightning. The class
:class:`BRDTTrainingModule` is a plain ``torch.nn.Module`` so the
bounded preflight can drive it from a small pure-PyTorch loop without
adding optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.run_config import LossWeights


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV on a ``(B, 1, H, W)`` tensor (mean per pixel)."""
    if image.dim() != 4:
        raise ValueError(f"image must be (B,1,H,W); got {tuple(image.shape)}")
    dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
    dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    return dy.mean() + dx.mean()


@dataclass
class LossBreakdown:
    """Per-step loss breakdown for diagnostic logging."""

    image: float
    physics: float
    relative_physics: float
    tv: float
    positivity: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "image": float(self.image),
            "physics": float(self.physics),
            "relative_physics": float(self.relative_physics),
            "tv": float(self.tv),
            "positivity": float(self.positivity),
            "total": float(self.total),
        }


class BRDTTrainingModule(nn.Module):
    """Wrap a model adapter with the BRDT supervised + Born consistency loss.

    Parameters
    ----------
    model:
        Adapter producing ``q_pred_norm`` of shape ``(B, 1, N, N)``.
    operator:
        Locked :class:`BornRytovForward2D` instance.
    normalization:
        Train-only normalization stats (used to unnormalize before physics).
    weights:
        Loss weights (image / physics / relative_physics / tv / positivity).
    output_space:
        ``"normalized_q"`` (default) or ``"physical_q"``. The image-space
        loss is computed in this space; the physics loss always
        unnormalizes to physical q before invoking the operator.
    """

    SUPPORTED_OUTPUT_SPACES = ("normalized_q", "physical_q")
    TRAINING_LABEL = "supervised + Born consistency"

    def __init__(
        self,
        *,
        model: nn.Module,
        operator: BornRytovForward2D,
        normalization: dc.NormalizationStats,
        weights: Optional[LossWeights] = None,
        output_space: str = "normalized_q",
    ):
        super().__init__()
        if output_space not in self.SUPPORTED_OUTPUT_SPACES:
            raise ValueError(
                f"output_space={output_space!r} not in {self.SUPPORTED_OUTPUT_SPACES}"
            )
        self.model = model
        self.operator = operator
        self._mean = float(normalization.mean)
        self._std = float(normalization.safe_std)
        self.weights = weights or LossWeights()
        self.output_space = output_space

    def to_physical_q(self, q_pred: torch.Tensor) -> torch.Tensor:
        """Unnormalize the model output before invoking the operator.

        This routes through the same arithmetic as
        :func:`scripts.studies.born_rytov_dt.dataset_contract.unnormalize_q`
        and is the only place in the BRDT training stack that converts
        normalized predictions into physical q.
        """
        if self.output_space == "physical_q":
            return q_pred
        return q_pred * self._std + self._mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(
        self,
        *,
        q_pred: torch.Tensor,
        q_true_norm: torch.Tensor,
        q_true_physical: torch.Tensor,
        sinogram_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        """Compute the supervised + Born consistency loss.

        Parameters
        ----------
        q_pred:
            Model output in :attr:`output_space`.
        q_true_norm:
            Reference target in normalized q units.
        q_true_physical:
            Reference target in physical q units.
        sinogram_obs:
            Observed (noisy) sinogram, shape ``(B, A, D, 2)``.
        """
        if q_pred.dim() != 4 or q_pred.shape[1] != 1:
            raise ValueError(
                f"q_pred must be (B,1,N,N); got {tuple(q_pred.shape)}"
            )
        # Image-space supervised loss. Compute in the model's output space
        # so the gradient signal is on the same scale the model produces.
        if self.output_space == "normalized_q":
            target_image = q_true_norm
        else:
            target_image = q_true_physical
        image_loss = (q_pred - target_image).abs().mean()

        # Physics loss: enforce contract by routing the operator input
        # explicitly through the dataset-contract guard.
        q_phys = self.to_physical_q(q_pred)
        dc.reject_normalized_q_to_operator("physical_q")
        y_pred = self.operator(q_phys)
        if y_pred.shape != sinogram_obs.shape:
            raise ValueError(
                f"forward operator output {tuple(y_pred.shape)} != sinogram "
                f"shape {tuple(sinogram_obs.shape)}"
            )
        phys_residual = (y_pred - sinogram_obs).abs().mean()
        denom = sinogram_obs.norm() + 1e-8
        rel_phys = (y_pred - sinogram_obs).norm() / denom

        tv_term = _total_variation(q_phys)
        positivity_term = (q_phys.clamp(max=0.0) ** 2).mean()

        weights = self.weights
        total = (
            weights.image * image_loss
            + weights.physics * phys_residual
            + weights.relative_physics * rel_phys
            + weights.tv * tv_term
            + weights.positivity * positivity_term
        )
        breakdown = LossBreakdown(
            image=float(image_loss.detach().item()),
            physics=float(phys_residual.detach().item()),
            relative_physics=float(rel_phys.detach().item()),
            tv=float(tv_term.detach().item()),
            positivity=float(positivity_term.detach().item()),
            total=float(total.detach().item()),
        )
        return total, breakdown

    def loss_contract(self) -> Dict[str, Any]:
        """Snapshot of the loss contract for invocation/provenance artifacts."""
        return {
            "training_label": self.TRAINING_LABEL,
            "output_space": self.output_space,
            "weights": self.weights.as_dict(),
            "physics_loss_rule": dc.PHYSICS_LOSS_RULE,
            "operator_input_routing_guard": "dataset_contract.reject_normalized_q_to_operator",
            "unnormalize_helper": "BRDTTrainingModule.to_physical_q (mirrors dataset_contract.unnormalize_q)",
        }
