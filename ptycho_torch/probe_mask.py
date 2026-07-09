"""Probe mask helpers for Torch workflows.

This module defines the default Torch probe-mask semantics:
- `probe_mask=True` enables a centered disk mask
- default disk diameter is N/2 pixels (radius N/4)
- default edge smoothing is enabled (sigma=1 px, smooth edge)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


MaskLike = Union[torch.Tensor, np.ndarray]


def _gaussian_kernel_1d(
    sigma: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel = kernel / torch.sum(kernel)
    return kernel


def _smooth_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return mask
    kernel = _gaussian_kernel_1d(sigma, dtype=mask.dtype, device=mask.device)
    pad = int(kernel.numel() // 2)
    x = mask[None, None]
    x = F.conv2d(x, kernel.view(1, 1, 1, -1), padding=(0, pad))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1), padding=(pad, 0))
    return x[0, 0]


def make_soft_probe_mask_torch(
    n: int,
    *,
    diameter: Optional[float] = None,
    sigma: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build centered soft disk mask with max value normalized to 1."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if diameter is None:
        diameter = float(n) / 2.0
    if diameter <= 0:
        raise ValueError(f"diameter must be > 0, got {diameter}")

    centered = torch.arange(n, dtype=dtype, device=device) - (n // 2) + 0.5
    yy, xx = torch.meshgrid(centered, centered, indexing="ij")
    radius = float(diameter) / 2.0
    mask = (torch.sqrt(xx * xx + yy * yy) < radius).to(dtype=dtype)
    mask = _smooth_mask(mask, float(sigma))
    max_val = torch.max(mask)
    if float(max_val) > 0:
        mask = mask / max_val
    return torch.clamp(mask, min=0.0, max=1.0)


def make_soft_probe_mask_np(
    n: int,
    *,
    diameter: Optional[float] = None,
    sigma: float = 1.0,
) -> np.ndarray:
    mask = make_soft_probe_mask_torch(n, diameter=diameter, sigma=sigma)
    return mask.detach().cpu().numpy().astype(np.float32, copy=False)


def resolve_probe_mask_enabled(
    probe_mask: Optional[Union[bool, MaskLike]],
    probe_mask_tensor: Optional[MaskLike],
) -> bool:
    """Resolve final boolean mask-enable state from toggle + optional tensor."""
    if probe_mask_tensor is not None:
        return True
    if probe_mask is None:
        return False
    if isinstance(probe_mask, bool):
        return probe_mask
    return True


def _coerce_mask_tensor(
    mask_like: MaskLike,
    *,
    n: int,
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> torch.Tensor:
    mask = torch.as_tensor(mask_like, dtype=dtype, device=device)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask[0, 0]
    if mask.shape != (n, n):
        raise ValueError(f"probe mask must have shape ({n}, {n}), got {tuple(mask.shape)}")
    return torch.clamp(mask, min=0.0)


def resolve_probe_mask_torch(
    n: int,
    *,
    probe_mask: Optional[Union[bool, MaskLike]] = False,
    probe_mask_tensor: Optional[MaskLike] = None,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Resolve final probe mask tensor (ones when masking disabled)."""
    enabled = resolve_probe_mask_enabled(probe_mask, probe_mask_tensor)
    if not enabled:
        return torch.ones((n, n), dtype=dtype, device=device)

    if probe_mask_tensor is not None:
        return _coerce_mask_tensor(probe_mask_tensor, n=n, dtype=dtype, device=device)

    if probe_mask is not None and not isinstance(probe_mask, bool):
        return _coerce_mask_tensor(probe_mask, n=n, dtype=dtype, device=device)

    return make_soft_probe_mask_torch(
        n,
        diameter=probe_mask_diameter,
        sigma=probe_mask_sigma,
        dtype=dtype,
        device=device,
    )


def resolve_probe_mask_np(
    n: int,
    *,
    probe_mask: Optional[Union[bool, MaskLike]] = False,
    probe_mask_tensor: Optional[MaskLike] = None,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: Optional[float] = None,
) -> np.ndarray:
    mask = resolve_probe_mask_torch(
        n,
        probe_mask=probe_mask,
        probe_mask_tensor=probe_mask_tensor,
        probe_mask_sigma=probe_mask_sigma,
        probe_mask_diameter=probe_mask_diameter,
    )
    return mask.detach().cpu().numpy().astype(np.float32, copy=False)
