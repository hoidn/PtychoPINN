"""Patterson correlation feature computation for ptychographic input encoding.

Computes physically meaningful input features from raw diffraction intensities
and the known probe function, encoding overlap constraints directly in the
input representation via leave-one-out Patterson cross-products.
"""

import torch
from torch import nn


class PattersonFeaturizer(nn.Module):
    """Compute Patterson correlation features from diffraction intensities.

    Non-learnable module that transforms (B, C, N, N) raw diffraction
    intensities into (B, 4*C, N, N) feature channels:
      - I_k:       raw diffraction intensity (reciprocal space)
      - Omega_k:   probe overlap map (real space)
      - Re(Pi_k):  LOO-Patterson cross-product, real part (real space)
      - Im(Pi_k):  LOO-Patterson cross-product, imaginary part (real space)

    Channels are interleaved per-position:
      [I_0, Omega_0, Re(Pi_0), Im(Pi_0), I_1, Omega_1, ...]
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, intensities: torch.Tensor,
                probe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intensities: (B, C, N, N) float32 raw diffraction patterns
            probe: (B, C, P, N, N) complex64 probe function

        Returns:
            (B, 4*C, N, N) float32 Patterson feature tensor, detached
        """
        B, C, N, _ = intensities.shape
        assert C > 1, (
            f"Patterson featurizer requires C > 1 for leave-one-out, got C={C}"
        )

        # DataParallel scatters complex tensors as real-viewed (..., 2)
        if not probe.is_complex():
            probe = torch.view_as_complex(probe.contiguous())

        # 1. Patterson functions: A_k = IFFT(I_k)
        A = torch.fft.ifft2(intensities)  # (B, C, N, N) complex64

        # 2. Leave-one-out cross-products
        A_sum = A.sum(dim=1, keepdim=True)  # (B, 1, N, N)
        A_avg_neg_k = (A_sum - A) / (C - 1)  # (B, C, N, N)
        Pi = A * torch.conj(A_avg_neg_k)  # (B, C, N, N) complex64

        # 3. Probe overlap maps
        probe_intensity = (probe.abs() ** 2).sum(dim=2)  # (B, C, N, N)
        P_sum = probe_intensity.sum(dim=1, keepdim=True)  # (B, 1, N, N)
        P_avg_neg_k = (P_sum - probe_intensity) / (C - 1)  # (B, C, N, N)
        Omega = probe_intensity * P_avg_neg_k  # (B, C, N, N)

        # 4. Assemble interleaved feature tensor
        features = torch.stack([
            intensities,
            Omega,
            Pi.real,
            Pi.imag,
        ], dim=2)  # (B, C, 4, N, N)

        features = features.reshape(B, 4 * C, N, N)

        return features.detach()
