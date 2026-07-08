"""Differentiable 2D Born forward operator for the BRDT candidate lane.

This module implements ``BornRytovForward2D``, a differentiable 2D weak-
scattering forward operator used by the Born/Rytov diffraction tomography
candidate evidence lane in NEURIPS-HYBRID-RESNET-2026.

Locked operator contract (see ``docs/plans/NEURIPS-HYBRID-RESNET-2026/
born_rytov_dt_candidate_lane_design.md`` and the operator validation
report ``brdt_operator_validation_report.md`` for the authoritative
specification):

- Input ``q`` is the real physical scattering potential
  ``q(x,z) = k_m^2 ((n/n_m)^2 - 1)`` shaped ``(B, 1, N, N)`` with row index
  iterating over the ``z`` axis and column index iterating over the ``x``
  axis. Both axes are sampled at unit pixel spacing.
- Output is the complex scattered field on a detector line at ``z_D = 0``
  for each illumination angle, returned as ``(B, A, D, 2)`` with the last
  dimension carrying ``(real, imag)`` channels.
- Coordinate convention: ``z`` is the propagation axis. Angle ``theta``
  measures the rotation of the incident plane wave such that
  ``k_inc = k_m * (sin(theta), cos(theta))`` and ``theta = 0`` means the
  source is +z directed.
- Detector-frequency convention: ``k_x`` is sampled on the
  ``torch.fft.fftfreq(D)`` grid scaled to angular frequency
  ``2*pi * f`` (rad/pixel). Propagating components satisfy
  ``|k_x| < k_m`` with ``k_z = sqrt(k_m^2 - k_x^2)``; evanescent modes are
  zeroed by the validity mask.
- Object-spectrum sampling on the Ewald arc uses
  ``K_obj = (k_x - k_m sin(theta), k_z - k_m cos(theta))``.
- FFT normalization mode ``unitary_fft`` uses ``norm="ortho"`` for the
  object FFT and the detector inverse FFT and omits the
  ``(i / (2 k_z))`` Wolf prefactor. Mode ``odtbrain_compatible`` uses
  unnormalised forward FFTs, includes the ``(i / (2 k_z))`` prefactor, and
  uses the default ``ifft`` normalization so that the output matches the
  scalar 2D Born integral with Green's function
  ``G(r) = (i/4) H_0^{(1)}(k_m |r|)`` up to discretization error.
- Output layout ``(B, A, D, 2)`` is fixed; downstream consumers that need a
  complex tensor must call ``torch.complex(out[..., 0], out[..., 1])``.

Only the ``born`` mode is supported. ``rytov_linearized`` is reserved by the
constructor as an explicit boundary so downstream code cannot silently fall
through to it; selecting it raises ``NotImplementedError`` and emits a
clear message naming the design's preprocessing-gate requirement.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

NormalizeMode = Literal["unitary_fft", "odtbrain_compatible"]
OperatorMode = Literal["born", "rytov_linearized"]


class BornRytovForward2D(nn.Module):
    """Differentiable Born 2D forward operator.

    Parameters
    ----------
    grid_size:
        Object grid side ``N``. Object tensors must be shaped
        ``(B, 1, N, N)``.
    detector_size:
        Detector line length ``D`` in pixels.
    angles:
        Iterable or 1-D tensor of illumination angles in radians.
    wavelength_px:
        Wavelength in the medium expressed in pixels. The medium wave
        number is ``k_m = 2*pi / wavelength_px``.
    medium_ri:
        Surrounding-medium refractive index ``n_m``. Recorded in operator
        identity but not used by the forward path because ``q`` already
        encodes the scattering potential.
    mode:
        Either ``"born"`` (only supported mode) or ``"rytov_linearized"``
        (intentionally not implemented; raises ``NotImplementedError``).
    normalize:
        FFT normalization mode. ``unitary_fft`` (default) is preferred for
        downstream training because it keeps amplitudes bounded
        independent of grid size. ``odtbrain_compatible`` matches the
        scalar 2D direct Born integral to within discretization tolerance.
    device:
        Optional device hint. The forward path will follow the device of
        the input ``q`` regardless of the constructor hint.
    """

    def __init__(
        self,
        grid_size: int,
        detector_size: int,
        angles: Union[torch.Tensor, Iterable[float]],
        wavelength_px: float,
        medium_ri: float,
        mode: OperatorMode = "born",
        normalize: NormalizeMode = "unitary_fft",
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        super().__init__()
        if mode == "rytov_linearized":
            raise NotImplementedError(
                "Rytov mode is not supported by BornRytovForward2D. The BRDT "
                "candidate-lane design (born_rytov_dt_candidate_lane_design.md) "
                "requires a separate Rytov preprocessing gate before any "
                "Rytov-linearized operator can be exposed."
            )
        if mode != "born":
            raise ValueError(f"Unknown operator mode: {mode!r}")
        if normalize not in ("unitary_fft", "odtbrain_compatible"):
            raise ValueError(
                "normalize must be 'unitary_fft' or 'odtbrain_compatible'; "
                f"got {normalize!r}"
            )
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if detector_size <= 0:
            raise ValueError("detector_size must be positive")
        if wavelength_px <= 0:
            raise ValueError("wavelength_px must be positive")
        if medium_ri <= 0:
            raise ValueError("medium_ri must be positive")

        self.grid_size = int(grid_size)
        self.detector_size = int(detector_size)
        self.wavelength_px = float(wavelength_px)
        self.medium_ri = float(medium_ri)
        self.mode: OperatorMode = mode
        self.normalize: NormalizeMode = normalize
        self.k_m = 2.0 * math.pi / self.wavelength_px

        angles_t = torch.as_tensor(list(angles), dtype=torch.float64)
        if angles_t.dim() != 1:
            raise ValueError("angles must be a 1-D iterable")
        self.register_buffer("angles", angles_t)

        det_freqs = (
            2.0 * math.pi * torch.fft.fftfreq(self.detector_size, d=1.0).to(torch.float64)
        )
        self.register_buffer("det_freqs", det_freqs)

        kx = det_freqs.unsqueeze(0)  # (1, D)
        kx_sq = kx * kx
        propagating = (kx_sq < self.k_m * self.k_m).to(torch.float64)
        kz = torch.sqrt(torch.clamp(self.k_m * self.k_m - kx_sq, min=0.0))  # (1, D)

        cos_t = torch.cos(self.angles).unsqueeze(1)  # (A, 1)
        sin_t = torch.sin(self.angles).unsqueeze(1)

        kx_b = kx.expand(self.angles.shape[0], -1)  # (A, D)
        kz_b = kz.expand(self.angles.shape[0], -1)
        prop_b = propagating.expand_as(kx_b)
        Kx_obj = kx_b - self.k_m * sin_t  # (A, D)
        Kz_obj = kz_b - self.k_m * cos_t  # (A, D)

        N = self.grid_size
        N_half = N // 2
        idx_x = Kx_obj * N / (2.0 * math.pi) + N_half  # (A, D)
        idx_z = Kz_obj * N / (2.0 * math.pi) + N_half
        denom = float(N - 1) if N > 1 else 1.0
        grid_x = idx_x / denom * 2.0 - 1.0
        grid_z = idx_z / denom * 2.0 - 1.0
        in_bounds = (grid_x.abs() <= 1.0) & (grid_z.abs() <= 1.0)
        valid = (prop_b > 0) & in_bounds  # (A, D) bool

        # grid_sample expects last dim (x, y) with x → cols, y → rows; Q has
        # rows = kz_idx, cols = kx_idx, so x_grid = grid_x and y_grid = grid_z.
        sampling_grid = torch.stack([grid_x, grid_z], dim=-1)  # (A, D, 2)

        self.register_buffer("sampling_grid", sampling_grid)
        self.register_buffer("valid_mask", valid)

        if self.normalize == "odtbrain_compatible":
            kz_safe = torch.clamp(kz_b, min=1e-12)
            coeff_real = torch.zeros_like(kz_safe)
            coeff_imag = 1.0 / (2.0 * kz_safe)
        else:
            coeff_real = torch.zeros_like(kz_b)
            coeff_imag = torch.ones_like(kz_b)
        self.register_buffer("coeff_real", coeff_real)
        self.register_buffer("coeff_imag", coeff_imag)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Apply the Born forward operator.

        Parameters
        ----------
        q:
            Real scattering-potential tensor of shape
            ``(B, 1, grid_size, grid_size)``.

        Returns
        -------
        torch.Tensor
            ``(B, A, D, 2)`` real tensor whose last dim is the real and
            imaginary parts of the complex scattered field on the
            detector line for each illumination angle.
        """
        if q.dim() != 4 or q.shape[1] != 1 or q.shape[2] != self.grid_size or q.shape[3] != self.grid_size:
            raise ValueError(
                f"q must have shape (B, 1, {self.grid_size}, {self.grid_size}); "
                f"got {tuple(q.shape)}"
            )
        if not torch.is_floating_point(q):
            raise TypeError(f"q must be a floating-point tensor; got dtype={q.dtype}")

        target_dtype = q.dtype
        device = q.device

        if target_dtype == torch.float64:
            cdtype = torch.complex128
        else:
            cdtype = torch.complex64

        q_typed = q.squeeze(1).to(cdtype)

        if self.normalize == "unitary_fft":
            Q = torch.fft.fft2(q_typed, norm="ortho")
        else:
            Q = torch.fft.fft2(q_typed, norm="backward")
        Q = torch.fft.fftshift(Q, dim=(-2, -1))

        Q_ri = torch.stack([Q.real, Q.imag], dim=1)  # (B, 2, N, N)

        grid = self.sampling_grid.to(device=device, dtype=Q_ri.dtype)
        grid = grid.unsqueeze(0).expand(Q_ri.shape[0], -1, -1, -1)
        sampled = F.grid_sample(
            Q_ri, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled_real = sampled[:, 0]  # (B, A, D)
        sampled_imag = sampled[:, 1]

        mask = self.valid_mask.to(device=device, dtype=sampled_real.dtype)
        sampled_real = sampled_real * mask
        sampled_imag = sampled_imag * mask

        c_real = self.coeff_real.to(device=device, dtype=sampled_real.dtype)
        c_imag = self.coeff_imag.to(device=device, dtype=sampled_imag.dtype)
        out_real = sampled_real * c_real - sampled_imag * c_imag
        out_imag = sampled_real * c_imag + sampled_imag * c_real

        S = torch.complex(out_real, out_imag)
        if self.normalize == "unitary_fft":
            u_det = torch.fft.ifft(S, dim=-1, norm="ortho")
        else:
            u_det = torch.fft.ifft(S, dim=-1, norm="backward")

        out = torch.stack([u_det.real, u_det.imag], dim=-1)
        return out.to(target_dtype)

    # ------------------------------------------------------------------
    # Discoverability
    # ------------------------------------------------------------------
    def operator_contract(self) -> dict:
        """Return a snapshot of the locked operator contract.

        Useful for serialization in the validation report and JSON
        artifact so downstream BRDT plans can read the contract without
        reflecting on the module.
        """
        return {
            "module": __name__,
            "class": type(self).__name__,
            "mode": self.mode,
            "normalize": self.normalize,
            "grid_size": self.grid_size,
            "detector_size": self.detector_size,
            "wavelength_px": self.wavelength_px,
            "medium_ri": self.medium_ri,
            "k_m": self.k_m,
            "angle_count": int(self.angles.shape[0]),
            "angle_min_rad": float(self.angles.min().item()),
            "angle_max_rad": float(self.angles.max().item()),
            "coordinate_convention": "z propagation; theta rotates k_inc=k_m*(sin,cos)",
            "detector_frequency_convention": (
                "torch.fft.fftfreq(D) * 2*pi (rad/pixel)"
            ),
            "ewald_sampling": "K_obj = (k_x - k_m sin(theta), k_z - k_m cos(theta))",
            "fft_normalization": (
                "unitary ortho FFTs (no Wolf prefactor)"
                if self.normalize == "unitary_fft"
                else "default FFTs with (i / (2 k_z)) Wolf prefactor"
            ),
            "output_layout": "(B, A, D, 2) real/imag",
        }
