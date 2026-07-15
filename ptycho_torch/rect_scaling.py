"""Global least-squares solver for the rectangular (s1, s2) scales.

Implements the PtychoPINN-CI paper Eq. (8): given basis intensities
``A = |Psi_a|^2``, ``B = |Psi_b|^2`` and cross term ``C = Re(conj(Psi_a) * Psi_b)``
together with measured intensities ``I``, find the real scales ``(s1, s2)`` that

    minimize  sum_pixels ( I - (s1^2 * A + 2*s1*s2*C + s2^2 * B) )^2 .

The objective is quartic in ``(s1, s2)`` but depends on the data only through
the lifted normal-equation moments ``M`` and ``b`` for
``q = (s1^2, s1*s2, s2^2)``.  A Euclidean projection of the unconstrained
lifted solution is not generally correct because the residual metric is ``M``.
Instead, write ``s = sqrt(rho) * (cos(theta), sin(theta))``.  For each direction
``d(theta) = (cos^2(theta), cos(theta)sin(theta), sin^2(theta))``, the optimal
nonnegative radial scale is

    rho(theta) = max(b.T @ d(theta), 0) / (d(theta).T @ M @ d(theta)).

The stationary directions are all real roots of a degree-at-most-four
polynomial in ``tan(theta)``.  Evaluating those roots plus the direction at
infinity gives the global constrained minimum.  ``(s1, s2)`` and
``(-s1, -s2)`` are physically identical, so ``s1 >= 0`` normalizes the result.

State contract (streaming refit): ``accumulate_rect_basis`` returns an opaque
``{"M": (3,3) float64, "b": (3,) float64}`` holding only the running normal-equation
moments -- never the per-pattern tensors -- so a per-dataset refit can stream
arbitrarily many batches at fixed memory and finish with ``solve_from_state``.
"""

import numpy as np
import torch

__all__ = ["solve_rect_scales", "accumulate_rect_basis", "solve_from_state"]


def _moments(A, C, B, I, weights):
    """Build normal-equation moments (M, b) for regressor rows r = (A, 2C, B).

    All sums are accumulated in float64 to avoid fp32 cancellation on
    count-scale intensities (up to ~1e6). M = sum w r^T r, b = sum w r^T I.
    Inputs of any mutually broadcastable shapes are broadcast before flattening.
    """
    if weights is None:
        A, C, B, I = torch.broadcast_tensors(A, C, B, I)
    else:
        A, C, B, I, weights = torch.broadcast_tensors(A, C, B, I, weights)
    A = A.reshape(-1).to(torch.float64)
    C = C.reshape(-1).to(torch.float64)
    B = B.reshape(-1).to(torch.float64)
    I = I.reshape(-1).to(torch.float64)
    R = torch.stack([A, 2.0 * C, B], dim=1)  # (N, 3)
    if weights is None:
        RI = I
        Rw = R
    else:
        w = weights.reshape(-1).to(torch.float64)
        RI = w * I
        Rw = R * w.unsqueeze(1)
    M = R.t() @ Rw           # (3, 3)
    b = R.t() @ RI           # (3,)
    return M, b


def _solve_rank1_residual(M, b):
    """Globally minimize the lifted residual over q=(s1^2,s1*s2,s2^2)."""
    if not (torch.isfinite(M).all() and torch.isfinite(b).all()):
        raise ValueError("non-finite normal equations (M or b contains nan/inf)")
    if int(torch.linalg.matrix_rank(M).item()) < 3:
        raise ValueError("singular normal matrix: rectangular scales are not identifiable")

    M_np = M.detach().cpu().numpy().astype(np.float64, copy=False)
    b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)

    beta = np.array([b_np[0], b_np[1], b_np[2]], dtype=np.float64)
    alpha = np.array(
        [
            M_np[0, 0],
            M_np[0, 1] + M_np[1, 0],
            M_np[0, 2] + M_np[1, 1] + M_np[2, 0],
            M_np[1, 2] + M_np[2, 1],
            M_np[2, 2],
        ],
        dtype=np.float64,
    )
    def _chart_roots(beta_coeffs, alpha_coeffs):
        """Stationary slopes in one bounded projective chart."""
        beta_scale = float(np.max(np.abs(beta_coeffs)))
        if beta_scale == 0.0:
            return []
        beta_coeffs = beta_coeffs / beta_scale
        alpha_coeffs = alpha_coeffs / np.max(np.abs(alpha_coeffs))
        beta_prime = np.array(
            [beta_coeffs[1], 2.0 * beta_coeffs[2]], dtype=np.float64
        )
        alpha_prime = np.array(
            [
                alpha_coeffs[1],
                2.0 * alpha_coeffs[2],
                3.0 * alpha_coeffs[3],
                4.0 * alpha_coeffs[4],
            ],
            dtype=np.float64,
        )
        polynomial = (
            2.0 * np.convolve(beta_prime, alpha_coeffs)
            - np.convolve(beta_coeffs, alpha_prime)
        )
        coefficient_scale = float(np.max(np.abs(polynomial)))
        if coefficient_scale == 0.0:
            return []
        polynomial /= coefficient_scale
        tolerance = 128.0 * np.finfo(np.float64).eps
        while polynomial.size > 1 and abs(polynomial[-1]) <= tolerance:
            polynomial = polynomial[:-1]
        if polynomial.size <= 1:
            return []

        derivative = np.arange(1, polynomial.size) * polynomial[1:]
        slopes = []
        for root in np.roots(polynomial[::-1]):
            if abs(root.imag) > 1e-7 * (1.0 + abs(root.real)):
                continue
            slope = float(root.real)
            for _ in range(2):
                value = np.polynomial.polynomial.polyval(slope, polynomial)
                gradient = np.polynomial.polynomial.polyval(slope, derivative)
                if gradient == 0.0:
                    break
                slope -= value / gradient
            if np.isfinite(slope) and abs(slope) <= 1.0 + 1e-8:
                slopes.append(slope)
        return slopes

    # Solve in both bounded charts so a valid direction never requires a huge
    # tan(theta), which would make the quartic coefficients ill-conditioned.
    directions = []
    for slope in [0.0, -1.0, 1.0, *_chart_roots(beta, alpha)]:
        directions.append(np.array([1.0, slope], dtype=np.float64))
    for slope in [0.0, -1.0, 1.0, *_chart_roots(beta[::-1], alpha[::-1])]:
        directions.append(np.array([slope, 1.0], dtype=np.float64))

    for direction in directions:
        direction /= np.linalg.norm(direction)
        if direction[0] < 0.0 or (
            direction[0] == 0.0 and direction[1] < 0.0
        ):
            direction *= -1.0

    best = None
    for direction in directions:
        d = np.array(
            [
                direction[0] ** 2,
                direction[0] * direction[1],
                direction[1] ** 2,
            ],
            dtype=np.float64,
        )
        denominator = float(d @ M_np @ d)
        numerator = float(b_np @ d)
        if denominator <= 0.0 or numerator <= 0.0:
            continue
        rho = numerator / denominator
        q = rho * d
        objective = float(q @ M_np @ q - 2.0 * b_np @ q)
        if best is None or objective < best[0]:
            root_rho = np.sqrt(rho)
            best = (objective, root_rho * direction)

    if best is None or not np.isfinite(best[1]).all():
        raise ValueError("non-positive rank-one optimum")
    return float(best[1][0]), float(best[1][1])


@torch.no_grad()
def solve_rect_scales(A, C, B, I, weights=None):
    """Return ``(s1, s2)`` minimizing Eq. (8) for basis intensities A, C, B and I.

    A = |Psi_a|^2, B = |Psi_b|^2, C = Re(conj(Psi_a) * Psi_b), all broadcastable
    to the shape of measured intensities ``I``. Optional ``weights`` (same shape
    as ``I``) applies a per-pixel weighted least squares. Raises ``ValueError`` on
    non-finite input, a singular normal matrix, or a non-positive rank-one
    optimum.
    """
    for name, t in (("A", A), ("C", C), ("B", B), ("I", I)):
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
    if weights is not None and not torch.isfinite(weights).all():
        raise ValueError("non-finite values in weights")
    M, b = _moments(A, C, B, I, weights)
    return _solve_rank1_residual(M, b)


@torch.no_grad()
def accumulate_rect_basis(psi_a, psi_b, I_meas, state=None):
    """Accumulate normal-equation moments from a batch of complex fields.

    Computes A = |psi_a|^2, B = |psi_b|^2, C = Re(conj(psi_a) * psi_b) internally,
    folds them into the running float64 moments, and returns the updated state
    (never retaining the per-pattern tensors). Pass ``state=None`` to start.
    Finish a stream with ``solve_from_state``. Raises ``ValueError`` on
    non-finite input so the offending batch is identified at accumulation time.
    """
    for name, t in (("psi_a", psi_a), ("psi_b", psi_b), ("I_meas", I_meas)):
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
    A = psi_a.abs().square()
    B = psi_b.abs().square()
    C = (psi_a.conj() * psi_b).real
    M, b = _moments(A, C, B, I_meas, None)
    if state is not None:
        M = M + state["M"]
        b = b + state["b"]
    return {"M": M, "b": b}


@torch.no_grad()
def solve_from_state(state):
    """Finalize a streaming refit: solve the accumulated moments for ``(s1, s2)``."""
    if state is None:
        raise ValueError("empty state: no batches were accumulated")
    return _solve_rank1_residual(state["M"], state["b"])
