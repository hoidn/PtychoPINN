"""Tests for Adaptive Gradient Clipping (AGC).

See: Brock et al., 2021 (NFNet), Algorithm 2.
Contract: ptycho_torch/train_utils.py::adaptive_gradient_clip_
"""

import torch
from ptycho_torch.train_utils import adaptive_gradient_clip_


def test_agc_clips_large_gradients():
    """AGC should clip when gradient norm >> parameter norm * clip_factor."""
    p = torch.nn.Parameter(torch.ones(4) * 0.1)  # small param norm ~0.2
    p.grad = torch.ones(4) * 100.0  # large grad norm ~200

    adaptive_gradient_clip_([p], clip_factor=0.01)

    g_norm_after = p.grad.data.norm(2).item()
    p_norm = p.data.norm(2).item()
    max_allowed = p_norm * 0.01
    assert g_norm_after <= max_allowed + 1e-6, (
        f"Gradient norm {g_norm_after} exceeds max allowed {max_allowed}"
    )


def test_agc_preserves_small_gradients():
    """AGC should not modify gradients when gradient norm <= parameter norm * clip_factor."""
    p = torch.nn.Parameter(torch.ones(4) * 10.0)  # large param norm ~20
    p.grad = torch.ones(4) * 0.001  # small grad norm ~0.002

    grad_before = p.grad.data.clone()
    adaptive_gradient_clip_([p], clip_factor=0.01)

    assert torch.allclose(p.grad.data, grad_before), (
        "Small gradients should not be modified by AGC"
    )


def test_agc_handles_zero_params():
    """AGC should not crash when parameter tensor is all zeros (eps guard)."""
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.ones(4) * 5.0

    # Should not raise
    adaptive_gradient_clip_([p], clip_factor=0.01, eps=1e-3)

    # Grad should be clipped to eps * clip_factor direction
    assert torch.isfinite(p.grad.data).all(), "Gradients should remain finite"
