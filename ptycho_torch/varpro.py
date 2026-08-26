'''VarPro scaling cluster extracted from reassembly.py (W4 split).

Owns the VarPro scaler, its basis construction, the shared CI batch
preparation, canvas scaling, and fitted-count metrics.  Cross-module
references back into ``ptycho_torch.reassembly`` are resolved lazily inside
the two functions that need them, so ``reassembly`` stays the single
monkeypatch/ownership seam and no import cycle exists.
'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.probe_mask import resolve_probe_mask_torch
from ptycho_torch.reassembly_diagnostics import (
    FittedCountMetrics,
    NotApplicable,
    VarProSufficientStatistics,
    array_digest,
    not_applicable,
)
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    LEGACY_SCALE_CONTRACT,
)


class VarProScaler:
    """
    Solves for global scaling s1, s2 and background offset c4.
    Memory efficient: only stores 4x4 matrix and 4x1 vector.
    """
    def __init__(self, device):
        self.device = device
        self.ATA = torch.zeros((3, 3), device=device, dtype=torch.float64)
        self.ATb = torch.zeros((3, 1), device=device, dtype=torch.float64)

        # For autograd
        self.X1X1 = 0.0
        self.X2X2 = 0.0
        self.X3X3 = 0.0
        self.X1X2 = 0.0
        self.X1X3 = 0.0
        self.X2X3 = 0.0
        self.X1I = 0.0
        self.X2I = 0.0
        self.X3I = 0.0
        self.II = 0.0  # For computing residual
        self.n_pixels = 0

    @torch.no_grad()
    def accumulate_batch(self, I_raw, Psi_a, Psi_b,
    terms = 3):
        """
        I_raw: [B, H, W] or [B, C, H, W]
        Psi_a/Psi_b: detector waves matching I_raw, optionally with an
            additional incoherent-mode axis immediately before H, W.
        """
        X1 = torch.abs(Psi_a)**2
        X2 = torch.abs(Psi_b)**2
        X3 = 2 * torch.real(Psi_a * torch.conj(Psi_b))
        if X1.ndim == I_raw.ndim + 1:
            X1 = X1.sum(dim=-3)
            X2 = X2.sum(dim=-3)
            X3 = X3.sum(dim=-3)
        elif X1.ndim != I_raw.ndim:
            raise ValueError(
                "Psi_a/Psi_b must match I_raw dimensions or add one mode axis"
            )
        self.accumulate_batch_from_basis(I_raw, X1, X2, X3, terms=terms)

    @torch.no_grad()
    def accumulate_batch_from_basis(self, I_raw, X1, X2, X3, terms=3):
        """
        Accumulate from pre-computed mode-summed basis images.
        Used for incoherent multi-mode probes where:
            X1 = sum_p |F[p_p * a_tilde]|^2
            X2 = sum_p |F[j * p_p * b_tilde]|^2
            X3 = sum_p 2*Re(F[p_p * a_tilde] * conj(F[j * p_p * b_tilde]))

        I_raw: [B, H, W] or [B, C, H, W]
        X1, X2, X3: same shape as I_raw
        """
        intensity64 = I_raw.to(torch.float64)
        bases = [basis.to(torch.float64) for basis in (X1, X2, X3)]

        for i in range(terms):
            self.ATb[i] += torch.sum(bases[i] * intensity64).item()
            for j in range(i, terms):
                val = torch.sum(bases[i] * bases[j]).item()
                self.ATA[i, j] += val
                if i != j:
                    self.ATA[j, i] += val

        X1_64, X2_64, X3_64 = bases
        self.X1X1 += torch.sum(X1_64 * X1_64).item()
        self.X2X2 += torch.sum(X2_64 * X2_64).item()
        self.X3X3 += torch.sum(X3_64 * X3_64).item()
        self.X1X2 += torch.sum(X1_64 * X2_64).item()
        self.X1X3 += torch.sum(X1_64 * X3_64).item()
        self.X2X3 += torch.sum(X2_64 * X3_64).item()
        self.X1I += torch.sum(X1_64 * intensity64).item()
        self.X2I += torch.sum(X2_64 * intensity64).item()
        self.X3I += torch.sum(X3_64 * intensity64).item()
        self.II += torch.sum(intensity64 * intensity64).item()
        self.n_pixels += I_raw.numel()

    def swap_channels(self):
        """Transform accumulated statistics to correct for real/imag channel swap.

        Under channel swap, the basis images transform as:
        X1 <-> X2, X3 -> -X3. This applies the corresponding
        transformation T*ATA*T and T*ATb where T swaps indices 0,1
        and negates index 2.
        """
        # Transform ATA
        new_ATA = self.ATA.clone()
        new_ATA[0, 0] = self.ATA[1, 1]
        new_ATA[1, 1] = self.ATA[0, 0]
        # [0,1] and [1,0] unchanged
        new_ATA[0, 2] = -self.ATA[1, 2]
        new_ATA[2, 0] = -self.ATA[2, 1]
        new_ATA[1, 2] = -self.ATA[0, 2]
        new_ATA[2, 1] = -self.ATA[2, 0]
        self.ATA = new_ATA

        # Transform ATb
        new_ATb = self.ATb.clone()
        new_ATb[0] = self.ATb[1]
        new_ATb[1] = self.ATb[0]
        new_ATb[2] = -self.ATb[2]
        self.ATb = new_ATb

        # Transform autograd statistics
        X1X1_old, X1X3_old, X1I_old = self.X1X1, self.X1X3, self.X1I
        self.X1X1 = self.X2X2
        self.X2X2 = X1X1_old
        self.X1X3 = -self.X2X3
        self.X2X3 = -X1X3_old
        self.X1I = self.X2I
        self.X2I = X1I_old
        self.X3I = -self.X3I

    def get_condition_number(self):
        """Compute condition number of ATA matrix"""
        eigenvalues = torch.linalg.eigvalsh(self.ATA)
        cond_num = eigenvalues.max() / (eigenvalues.min() + 1e-15)
        return cond_num.item()

    def get_correlation_matrix(self):
        """Get correlation matrix to check channel correlation"""
        # Normalize to correlation matrix
        diag = torch.sqrt(torch.diag(self.ATA))
        corr = self.ATA / (diag.unsqueeze(1) * diag.unsqueeze(0) + 1e-15)
        return corr

    def sufficient_statistics(self) -> VarProSufficientStatistics:
        """Return a detached snapshot of all dataset-level fit evidence."""
        return VarProSufficientStatistics(
            ATA=self.ATA,
            ATb=self.ATb,
            sum_i2=self.II,
            n_pixels=self.n_pixels,
        )

    def solve(self, verbose = True):
        """Solves the system and returns (s1, s2, background)"""
        # Solve linear system c = (ATA^-1) * ATb
        # Add small epsilon to diagonal for numerical stability

        if verbose:
            cond_num = self.get_condition_number()
            corr = self.get_correlation_matrix()
            print(f"Condition number: {cond_num:.2e}")
            print(f"Correlation matrix:\n{corr}")

            if cond_num > 1e10:
                print("WARNING: Matrix is ill-conditioned!")

        reg = torch.eye(3, device=self.device) * (self.ATA.max() * 1e-9)
        c = torch.linalg.solve(self.ATA + reg, self.ATb).flatten() # [c1, c2, c3, c4]

        c1, c2, c3 = c[0], c[1], c[2]

        # Stage B: Eigen-projection for physical consistency (Object part)
        # Matrix C = [[c1, c3], [c3, c2]]
        disc = torch.sqrt((c1 - c2)**2 + 4 * c3**2)
        lambda_max = 0.5 * (c1 + c2 + disc)

        # Principal Eigenvector (v1, v2)
        v1 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), c3, lambda_max - c2)
        v2 = torch.where(torch.abs(c3) > torch.abs(lambda_max - c1), lambda_max - c1, c3)
        norm = torch.sqrt(v1**2 + v2**2 + 1e-9)

        # Final Scale factors
        mag = torch.sqrt(torch.clamp(lambda_max, min=0))
        s1 = (v1 / norm) * mag
        s2 = (v2 / norm) * mag

        return s1.float(), s2.float()

    def solve_quadratic_direct(self, max_iter=50, verbose=True):
        """
        Directly solve for s1, s2 using Newton's method on the quadratic objective.
        Minimizes: ||s1^2*X1 + s2^2*X2 + s1*s2*X3 - I||^2
        """
        # Extract statistics from ATA and ATb
        X1X1 = self.ATA[0, 0].item()
        X2X2 = self.ATA[1, 1].item()
        X3X3 = self.ATA[2, 2].item()
        X1X2 = self.ATA[0, 1].item()
        X1X3 = self.ATA[0, 2].item()
        X2X3 = self.ATA[1, 2].item()
        X1I = self.ATb[0].item()
        X2I = self.ATb[1].item()
        X3I = self.ATb[2].item()

        # Initialize with positive square roots of diagonal solution
        s1 = torch.sqrt(torch.tensor(max(X1I / (X1X1 + 1e-10), 1e-6), device=self.device))
        s2 = torch.sqrt(torch.tensor(max(X2I / (X2X2 + 1e-10), 1e-6), device=self.device))


        if verbose:
            print(f"Initial guess: s1={s1:.4f}, s2={s2:.4f}")

        # Newton's method iterations
        for iter_num in range(max_iter):
            # Current objective value
            obj = (s1**4 * X1X1 + s2**4 * X2X2 + s1**2 * s2**2 * X3X3 +
                   2 * s1**2 * s2**2 * X1X2 + 2 * s1**3 * s2 * X1X3 + 2 * s1 * s2**3 * X2X3 -
                   2 * s1**2 * X1I - 2 * s2**2 * X2I - 2 * s1 * s2 * X3I)

            # Gradient components
            g1 = (4 * s1**3 * X1X1 + 2 * s1 * s2**2 * X3X3 + 4 * s1 * s2**2 * X1X2 +
                  6 * s1**2 * s2 * X1X3 + 2 * s2**3 * X2X3 -
                  4 * s1 * X1I - 2 * s2 * X3I)

            g2 = (4 * s2**3 * X2X2 + 2 * s1**2 * s2 * X3X3 + 4 * s1**2 * s2 * X1X2 +
                  2 * s1**3 * X1X3 + 6 * s1 * s2**2 * X2X3 -
                  4 * s2 * X2I - 2 * s1 * X3I)

            # Hessian components
            H11 = 12 * s1**2 * X1X1 + 2 * s2**2 * X3X3 + 4 * s2**2 * X1X2 + 12 * s1 * s2 * X1X3
            H22 = 12 * s2**2 * X2X2 + 2 * s1**2 * X3X3 + 4 * s1**2 * X1X2 + 12 * s1 * s2 * X2X3
            H12 = 4 * s1 * s2 * X3X3 + 8 * s1 * s2 * X1X2 + 6 * s1**2 * X1X3 + 6 * s2**2 * X2X3 - 2 * X3I

            # Check convergence
            grad_norm = torch.sqrt(g1**2 + g2**2)
            if verbose and iter_num % 5 == 0:
                print(f"Iter {iter_num}: obj={obj:.6e}, |grad|={grad_norm:.6e}, s1={s1:.4f}, s2={s2:.4f}")

            if grad_norm < 1e-8:
                if verbose:
                    print(f"Converged at iteration {iter_num}")
                break

            # Solve Newton system: H * delta = -g
            det = H11 * H22 - H12**2
            if torch.abs(det) < 1e-10:
                if verbose:
                    print(f"Warning: Near-singular Hessian at iteration {iter_num}, using gradient descent")
                # Fall back to gradient descent
                alpha = 0.01 / (grad_norm + 1e-10)
                delta_s1 = -alpha * g1
                delta_s2 = -alpha * g2
            else:
                delta_s1 = (-g1 * H22 + g2 * H12) / det
                delta_s2 = (g1 * H12 - g2 * H11) / det

            # Line search with positivity constraint
            alpha = 1.0
            for _ in range(10):
                s1_new = s1 + alpha * delta_s1
                s2_new = s2 + alpha * delta_s2

                if s1_new > 0 and s2_new > 0:
                    # Check if objective decreases
                    obj_new = (s1_new**4 * X1X1 + s2_new**4 * X2X2 + s1_new**2 * s2_new**2 * X3X3 +
                              2 * s1_new**2 * s2_new**2 * X1X2 + 2 * s1_new**3 * s2_new * X1X3 +
                              2 * s1_new * s2_new**3 * X2X3 - 2 * s1_new**2 * X1I -
                              2 * s2_new**2 * X2I - 2 * s1_new * s2_new * X3I)

                    if obj_new < obj:
                        break

                alpha *= 0.5
                if alpha < 1e-6:
                    if verbose:
                        print(f"Line search failed at iteration {iter_num}")
                    break

            s1 = s1 + alpha * delta_s1
            s2 = s2 + alpha * delta_s2

            # Ensure positivity
            s1 = torch.clamp(s1, min=1e-6)
            s2 = torch.clamp(s2, min=1e-6)

        return s1.float(), s2.float()

    def solve_autograd(self, max_iter=100, lr=0.1, verbose=True):
        """
        Solve for s1, s2 using PyTorch autograd with accumulated statistics.
        """
        # Convert statistics to tensors
        X1X1 = torch.tensor(self.X1X1, device=self.device, dtype=torch.float64)
        X2X2 = torch.tensor(self.X2X2, device=self.device, dtype=torch.float64)
        X3X3 = torch.tensor(self.X3X3, device=self.device, dtype=torch.float64)
        X1X2 = torch.tensor(self.X1X2, device=self.device, dtype=torch.float64)
        X1X3 = torch.tensor(self.X1X3, device=self.device, dtype=torch.float64)
        X2X3 = torch.tensor(self.X2X3, device=self.device, dtype=torch.float64)
        X1I = torch.tensor(self.X1I, device=self.device, dtype=torch.float64)
        X2I = torch.tensor(self.X2I, device=self.device, dtype=torch.float64)
        X3I = torch.tensor(self.X3I, device=self.device, dtype=torch.float64)
        II = torch.tensor(self.II, device=self.device, dtype=torch.float64)

        # Initialize parameters
        s1_init = torch.sqrt(torch.clamp(X1I / (X1X1 + 1e-10), min=1e-6))
        s2_init = torch.sqrt(torch.clamp(X2I / (X2X2 + 1e-10), min=1e-6))

        s1 = torch.nn.Parameter(s1_init)
        s2 = torch.nn.Parameter(s2_init)

        optimizer = torch.optim.Adam([s1, s2], lr=lr)

        if verbose:
            print(f"Initial: s1={s1.item():.4f}, s2={s2.item():.4f}")
            print(f"Total pixels accumulated: {self.n_pixels}")

        for iter_num in range(max_iter):
            optimizer.zero_grad()

            loss = (s1**4 * X1X1 +
                   s2**4 * X2X2 +
                   (s1*s2)**2 * X3X3 +
                   2 * s1**2 * s2**2 * X1X2 +
                   2 * s1**3 * s2 * X1X3 +
                   2 * s1 * s2**3 * X2X3 -
                   2 * s1**2 * X1I -
                   2 * s2**2 * X2I -
                   2 * s1 * s2 * X3I +
                   II)

            # Normalize by number of pixels for scale-invariant loss
            loss = loss / self.n_pixels

            loss.backward()
            optimizer.step()

            # Ensure positivity
            with torch.no_grad():
                s1.clamp_(min=1e-6)
                s2.clamp_(min=1e-6)

            if verbose and iter_num % 20 == 0:
                grad_norm = torch.sqrt(s1.grad**2 + s2.grad**2)
                print(f"Iter {iter_num}: loss={loss.item():.6e}, |grad|={grad_norm.item():.6e}, "
                      f"s1={s1.item():.4f}, s2={s2.item():.4f}")

            # Check convergence
            if s1.grad is not None and s2.grad is not None:
                grad_norm = torch.sqrt(s1.grad**2 + s2.grad**2)
                if grad_norm < 1e-8:
                    if verbose:
                        print(f"Converged at iteration {iter_num}")
                    break

        return s1.detach().float(), s2.detach().float()

    def solve_lbfgs(self, max_iter=50, verbose=True):
        """
        Solve using L-BFGS - often faster convergence for quadratic problems.
        """
        # Convert statistics to tensors
        X1X1 = torch.tensor(self.X1X1, device=self.device, dtype=torch.float64)
        X2X2 = torch.tensor(self.X2X2, device=self.device, dtype=torch.float64)
        X3X3 = torch.tensor(self.X3X3, device=self.device, dtype=torch.float64)
        X1X2 = torch.tensor(self.X1X2, device=self.device, dtype=torch.float64)
        X1X3 = torch.tensor(self.X1X3, device=self.device, dtype=torch.float64)
        X2X3 = torch.tensor(self.X2X3, device=self.device, dtype=torch.float64)
        X1I = torch.tensor(self.X1I, device=self.device, dtype=torch.float64)
        X2I = torch.tensor(self.X2I, device=self.device, dtype=torch.float64)
        X3I = torch.tensor(self.X3I, device=self.device, dtype=torch.float64)
        II = torch.tensor(self.II, device=self.device, dtype=torch.float64)

        # Initialize
        s1_init = torch.sqrt(torch.clamp(X1I / (X1X1 + 1e-10), min=1e-6))
        s2_init = torch.sqrt(torch.clamp(X2I / (X2X2 + 1e-10), min=1e-6))

        s1 = torch.nn.Parameter(s1_init)
        s2 = torch.nn.Parameter(s2_init)

        s1_pos, s2_pos = s1, s2

        optimizer = torch.optim.LBFGS([s1, s2], max_iter=max_iter, line_search_fn='strong_wolfe')

        if verbose:
            print(f"Initial: s1={s1.item():.4f}, s2={s2.item():.4f}")

        def closure():
            optimizer.zero_grad()

            loss = (s1_pos**4 * X1X1 +
                   s2_pos**4 * X2X2 +
                   (s1_pos*s2_pos)**2 * X3X3 +
                   2 * s1_pos**2 * s2_pos**2 * X1X2 +
                   2 * s1_pos**3 * s2_pos * X1X3 +
                   2 * s1_pos * s2_pos**3 * X2X3 -
                   2 * s1_pos**2 * X1I -
                   2 * s2_pos**2 * X2I -
                   2 * s1_pos * s2_pos * X3I +
                   II) / self.n_pixels

            loss.backward()
            return loss

        # Run optimization
        optimizer.step(closure)

        print(f"uncorrected scalars: {s1, s2}")
        if s1 < 0:
            s1_final, s2_final = -s1, -s2  # flip both, convention: s1 > 0
        else:
            s1_final, s2_final = s1, s2

        if verbose:
            final_loss = closure()
            print(f"Final: s1={s1_final.item():.4f}, s2={s2_final.item():.4f}, loss={final_loss.item():.6e}")

        return s1_final.detach().float(), s2_final.detach().float()



def compute_varpro_basis(probe: torch.Tensor,
                          a_tilde: torch.Tensor,
                          b_tilde: torch.Tensor,
                          scale: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-mode VarPro exit-wave FFTs (Psi_a, Psi_b) and mode-summed basis
    images (X1, X2, X3) consumed by ``VarProScaler``.

    Uses ``norm='ortho'`` (energy-preserving / Parseval-exact), matching the
    convention used everywhere else this pattern appears -- this file's own
    ``detect_swap_probe_reference`` and ``ptycho_torch/model.py``'s
    ``Psi_a``/``Psi_b`` FFTs.

    Args:
        probe: (B,C,P,H,W) complex probe modes.
        a_tilde, b_tilde: (B,C,H,W) real/imag decoder textures.
        scale: optional output scale folded into the exit waves EXACTLY like
            the training forward (``RectangularScaledDiffraction.forward``,
            model.py:1395/1403: ``exit_wave = scale * probe * texture``), so
            the basis is in the same count units the training loss compared
            against (VARPRO-SOLVE-UNITS-001). A (B,1,1,1) tensor is broadcast
            over probe modes via ``unsqueeze(2)`` (model.py:1393 convention);
            ``None`` keeps the historical unscaled basis byte-for-byte. A
            dataset with ``physics_scaling_constant == 1.0`` (``normalize=
            'None'``, dataloader.py:736) yields ``output_scale`` ~= 1, i.e.
            no count-unit correction.

    Returns:
        (Psi_a, Psi_b, X1, X2, X3): Psi_a/Psi_b are (B,C,P,H,W) complex
        per-mode exit-wave FFTs; X1, X2, X3 are (B,C,H,W) real, mode-summed
        VarPro basis images.
    """
    # Unsqueeze texture for broadcasting with probe modes: (B,C,H,W) -> (B,C,1,H,W)
    a_5d = a_tilde.unsqueeze(2)
    b_5d = b_tilde.unsqueeze(2)

    # Per-mode exit waves: (B,C,P,H,W)
    exit_a = probe * a_5d
    exit_b = 1j * probe * b_5d
    if scale is not None:
        if torch.is_tensor(scale) and scale.dim() == 4:
            scale = scale.unsqueeze(2)  # (B,1,1,1) -> (B,1,1,1,1) over modes
        exit_a = scale * exit_a
        exit_b = scale * exit_b

    Psi_a = torch.fft.fftshift(torch.fft.fft2(exit_a, norm='ortho'), dim=(-2, -1))
    Psi_b = torch.fft.fftshift(torch.fft.fft2(exit_b, norm='ortho'), dim=(-2, -1))

    # Mode-summed basis images for VarPro: (B,C,P,H,W) -> (B,C,H,W)
    X1 = torch.sum(torch.abs(Psi_a)**2, dim=2)
    X2 = torch.sum(torch.abs(Psi_b)**2, dim=2)
    X3 = torch.sum(2 * torch.real(Psi_a * torch.conj(Psi_b)), dim=2)

    return Psi_a, Psi_b, X1, X2, X3



def _configured_probe_mask(
    reference: torch.Tensor,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> torch.Tensor:
    """Resolve the same effective probe mask used by training."""
    return resolve_probe_mask_torch(
        data_config.N,
        probe_mask=getattr(model_config, "probe_mask", False),
        probe_mask_tensor=getattr(model_config, "probe_mask_tensor", None),
        probe_mask_sigma=float(getattr(model_config, "probe_mask_sigma", 1.0)),
        probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
        dtype=reference.real.dtype if reference.is_complex() else reference.dtype,
        device=reference.device,
    )



def _apply_configured_probe_mask(
    probe: torch.Tensor,
    reference: torch.Tensor,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> torch.Tensor:
    """Apply the same effective probe-mask resolver used by training."""
    mask = _configured_probe_mask(reference, data_config, model_config)
    return probe * mask.view(1, 1, 1, data_config.N, data_config.N)



@dataclass(frozen=True)
class _PreparedCIVarProBatch:
    measured_intensity: torch.Tensor
    positions: torch.Tensor
    probe_physical: torch.Tensor
    input_scale: torch.Tensor
    texture_raw: torch.Tensor
    effective_mask: torch.Tensor
    effective_probe: torch.Tensor
    psi_a: torch.Tensor
    psi_b: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    inference_time: float
    assembly_start: float



def _prepare_ci_varpro_batch(
    model: nn.Module,
    batch_data: Any,
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    device: torch.device,
    precision: InferencePrecision,
    channels_swapped: bool,
    collect_timing: bool = False,
) -> _PreparedCIVarProBatch:
    """Prepare one CI batch identically for fitting and count evaluation."""
    from ptycho_torch import reassembly

    _forward_predict = reassembly._forward_predict
    _synchronize_cuda_for_timing = reassembly._synchronize_cuda_for_timing
    compute_varpro_basis = reassembly.compute_varpro_basis
    measured_intensity = batch_data["measured_intensity"].to(
        device, non_blocking=True
    )
    positions = batch_data["coords_relative"].to(device, non_blocking=True)
    probe_physical = batch_data["probe_physical"].to(device, non_blocking=True)
    input_scale = batch_data["rms_input_scale"].to(device, non_blocking=True)

    if collect_timing:
        _synchronize_cuda_for_timing(device)
        inference_start = time.time()
    else:
        inference_start = 0.0
    texture_raw = _forward_predict(
        model,
        measured_intensity,
        positions,
        probe_physical,
        input_scale,
        device=device,
        precision=precision,
    ).to(torch.complex64)
    if collect_timing:
        _synchronize_cuda_for_timing(device)
        inference_time = time.time() - inference_start
    else:
        inference_time = 0.0

    if channels_swapped:
        texture_raw = torch.complex(texture_raw.imag, texture_raw.real)
    effective_mask = _configured_probe_mask(
        measured_intensity,
        data_config,
        model_config,
    )
    effective_probe = (
        probe_physical
        * effective_mask.view(1, 1, 1, data_config.N, data_config.N)
    ).to(torch.complex64)

    if collect_timing:
        _synchronize_cuda_for_timing(device)
        assembly_start = time.time()
    else:
        assembly_start = 0.0
    psi_a, psi_b, x1, x2, x3 = compute_varpro_basis(
        effective_probe,
        texture_raw.real.float(),
        texture_raw.imag.float(),
    )
    return _PreparedCIVarProBatch(
        measured_intensity=measured_intensity,
        positions=positions,
        probe_physical=probe_physical,
        input_scale=input_scale,
        texture_raw=texture_raw,
        effective_mask=effective_mask,
        effective_probe=effective_probe,
        psi_a=psi_a,
        psi_b=psi_b,
        x1=x1,
        x2=x2,
        x3=x3,
        inference_time=inference_time,
        assembly_start=assembly_start,
    )



def apply_varpro_canvas_scaling(
    texture_canvas: torch.Tensor,
    scaler: VarProScaler,
    *,
    enabled: bool = True,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply VarPro real/imag scaling to a stitched canvas, or return identity."""
    if not enabled:
        one = torch.tensor(1.0, device=texture_canvas.device, dtype=torch.float32)
        return texture_canvas, one, one

    s1, s2 = scaler.solve_lbfgs(verbose=verbose)
    scaled_canvas = torch.complex(s1 * texture_canvas.real, s2 * texture_canvas.imag)
    return scaled_canvas, s1, s2



def evaluate_fitted_count_metrics(
    model: nn.Module,
    infer_loader: Any,
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    s1: Any,
    s2: Any,
    device: Union[str, torch.device],
    scale_profile: str,
    precision: Optional[InferencePrecision] = None,
    channels_swapped: bool = False,
    local_to_source_ids: Any = None,
) -> Union[FittedCountMetrics, NotApplicable]:
    """Stream a deterministic fitted count-space pass over every batch."""
    from ptycho_torch import reassembly

    resolve_inference_precision_for_device = reassembly.resolve_inference_precision_for_device
    _prepare_ci_varpro_batch = reassembly._prepare_ci_varpro_batch
    if scale_profile == LEGACY_SCALE_CONTRACT:
        return not_applicable()
    if scale_profile != CI_SCALE_CONTRACT:
        raise ValueError(f"Unsupported count-metric scale profile: {scale_profile!r}")

    device = torch.device(device)
    effective_precision = resolve_inference_precision_for_device(precision, device)
    s1_value = torch.as_tensor(s1, dtype=torch.float32, device=device).reshape(())
    s2_value = torch.as_tensor(s2, dtype=torch.float32, device=device).reshape(())
    squared_error_sum = torch.zeros((), dtype=torch.float64, device=device)
    measured_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    poisson_nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_samples = 0
    n_pixels = 0
    effective_mask_digest = None
    sample_id_batches: list[torch.Tensor] = []
    source_id_map = None
    if local_to_source_ids is not None:
        source_id_map = torch.as_tensor(
            local_to_source_ids, dtype=torch.int64, device=device
        ).reshape(-1)

    with torch.no_grad():
        for batch in infer_loader:
            batch_data = batch[0]
            raw_sample_ids = batch_data.get("nn_indices")
            if raw_sample_ids is None:
                batch_size = int(batch_data["measured_intensity"].shape[0])
                channels = int(batch_data["measured_intensity"].shape[1])
                start = sum(item.numel() for item in sample_id_batches)
                batch_sample_ids = torch.arange(
                    start, start + batch_size * channels,
                    dtype=torch.int64, device=device,
                )
            else:
                batch_sample_ids = raw_sample_ids.to(
                    device=device, dtype=torch.int64
                ).reshape(-1)
            if source_id_map is not None:
                if batch_sample_ids.numel() and (
                    bool(torch.any(batch_sample_ids < 0))
                    or bool(torch.any(batch_sample_ids >= source_id_map.numel()))
                ):
                    raise ValueError("Count-metric local sample id is out of range")
                batch_sample_ids = source_id_map[batch_sample_ids]
            sample_id_batches.append(batch_sample_ids)
            prepared = _prepare_ci_varpro_batch(
                model,
                batch_data,
                data_config,
                model_config,
                device=device,
                precision=effective_precision,
                channels_swapped=channels_swapped,
            )
            measured_intensity = prepared.measured_intensity
            batch_mask_digest = array_digest(prepared.effective_mask)
            if effective_mask_digest is None:
                effective_mask_digest = batch_mask_digest
            elif effective_mask_digest != batch_mask_digest:
                raise ValueError("Effective probe mask changed across count batches")
            prediction = (
                s1_value.square() * prepared.x1.float()
                + s2_value.square() * prepared.x2.float()
                + (s1_value * s2_value) * prepared.x3.float()
            )
            measured64 = measured_intensity.to(torch.float64)
            prediction64 = prediction.to(torch.float64)
            residual = prediction64 - measured64
            squared_error_sum += residual.square().sum()
            measured_square_sum += measured64.square().sum()
            poisson_nll_sum += (
                prediction64
                - measured64 * torch.log(torch.clamp(prediction64, min=1e-8))
            ).sum()
            n_samples += int(measured_intensity.shape[0] * measured_intensity.shape[1])
            n_pixels += int(measured_intensity.numel())

    if n_pixels == 0:
        raise ValueError("Count-metric loader yielded no detector pixels")
    if float(measured_square_sum.item()) <= 0.0:
        raise ValueError("Measured intensity has zero count-space energy")
    assert effective_mask_digest is not None
    sample_ids = tuple(
        int(item)
        for item in torch.cat(sample_id_batches).detach().cpu().tolist()
    )
    if len(sample_ids) != n_samples:
        raise ValueError("Count-metric sample identity does not match loader samples")
    relative_l2 = torch.sqrt(squared_error_sum / measured_square_sum)
    return FittedCountMetrics(
        relative_l2_intensity_error=float(relative_l2.item()),
        mean_raw_poisson_nll=float((poisson_nll_sum / n_pixels).item()),
        n_samples=n_samples,
        n_pixels=n_pixels,
        effective_mask_digest=effective_mask_digest,
        sample_ids=sample_ids,
    )

