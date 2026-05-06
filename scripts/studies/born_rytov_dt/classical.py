"""Classical Born backprop / Born initialization image for BRDT.

This module owns the derivation of the ``born_init_image`` shared
neural input from the observed complex sinogram, and the classical
reference reconstruction baseline. ODTbrain is the preferred backend
when available; if it is missing we fall back to a local backprop
adjoint implemented from the locked ``BornRytovForward2D`` Fourier-
diffraction-theorem path. The fallback is acceptable for the
adapter-readiness preflight only; benchmark-grade rows must record
that the ODTbrain dependency was not exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc


@dataclass(frozen=True)
class ClassicalBackendInfo:
    """Backend identification for the classical backprop path."""

    name: str  # "odtbrain" or "local_adjoint"
    reason: str  # short tag explaining why this backend was picked
    claim_boundary: str  # "feasibility_only" or "external_oracle"


def detect_classical_backend(prefer_odtbrain: bool = True) -> ClassicalBackendInfo:
    """Return the available classical backend without importing eagerly.

    The local adjoint backend is the differentiable adjoint of
    :class:`BornRytovForward2D`. It is sufficient as a Born-init-image
    derivation for the bounded preflight, but it shares spectral
    machinery with the forward operator and therefore cannot serve as
    an independent oracle. ODTbrain is preferred when present.
    """
    if prefer_odtbrain:
        try:  # pragma: no cover - environment dependent
            import odtbrain  # type: ignore  # noqa: F401

            return ClassicalBackendInfo(
                name="odtbrain",
                reason="odtbrain_import_succeeded",
                claim_boundary="external_oracle",
            )
        except Exception:
            pass
    return ClassicalBackendInfo(
        name="local_adjoint",
        reason="odtbrain_unavailable",
        claim_boundary="feasibility_only",
    )


def _build_local_operator(
    *,
    angles: Sequence[float] | torch.Tensor | None = None,
    device: Optional[torch.device | str] = None,
) -> BornRytovForward2D:
    """Build a fresh forward operator under the locked smoke geometry.

    Buffers are kept in float64; the operator forward path promotes
    sampling-grid buffers to the input dtype on each call.
    """
    angles_tensor = (
        torch.as_tensor(list(angles), dtype=torch.float64)
        if angles is not None
        else torch.from_numpy(dc.locked_angles())
    )
    op = BornRytovForward2D(
        grid_size=dc.LOCKED_GRID_SIZE,
        detector_size=dc.LOCKED_DETECTOR_SIZE,
        angles=angles_tensor,
        wavelength_px=dc.LOCKED_WAVELENGTH_PX,
        medium_ri=dc.LOCKED_MEDIUM_RI,
        mode="born",
        normalize="unitary_fft",
    )
    if device is not None:
        op = op.to(device)
    return op


def _local_adjoint_init(
    sinogram: torch.Tensor,
    *,
    operator: BornRytovForward2D,
) -> torch.Tensor:
    """Derive the Born initialization image via differentiable adjoint.

    For the linear forward map ``y = A(q)``, the initialization image
    used here is ``q_init = A^T y / scale`` where ``A^T`` is taken to be
    the autograd transpose of :class:`BornRytovForward2D`. The result is
    a real image in physical-q units approximating a Born backprop. It
    is not a regularized inverse; rows in the bounded preflight must
    treat it as initialization only, never as a reconstruction.

    ``sinogram`` is shaped ``(B, A, D, 2)`` matching the operator output
    layout.
    """
    if sinogram.dim() != 4 or sinogram.shape[-1] != 2:
        raise ValueError(
            "sinogram must have shape (B, A, D, 2); "
            f"got {tuple(sinogram.shape)}"
        )
    batch_size = sinogram.shape[0]
    device = sinogram.device
    dtype = sinogram.dtype

    q_zero = torch.zeros(
        (batch_size, 1, operator.grid_size, operator.grid_size),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    y_pred = operator(q_zero)
    # Inner product <y_pred, sinogram> gives the adjoint when differentiated
    # at q=0 because the forward map is linear in q.
    inner = (y_pred * sinogram).sum()
    grad = torch.autograd.grad(inner, q_zero, create_graph=False)[0]
    return grad.detach()


def _odtbrain_backprop(
    sinogram_np: np.ndarray,
    *,
    angles: np.ndarray,
) -> np.ndarray:  # pragma: no cover - environment dependent
    """ODTbrain-backed 2D Born backprop on a single sample.

    ``sinogram_np`` shape ``(A, D, 2)``; ``angles`` shape ``(A,)``.
    Returns a ``(N, N)`` real array in physical-q units.
    """
    import odtbrain  # type: ignore

    complex_sinogram = sinogram_np[..., 0] + 1j * sinogram_np[..., 1]
    # ODTbrain ``res`` is the vacuum wavelength in pixels. The locked BRDT
    # operator stores wavelength in the surrounding medium, so convert back.
    res = float(dc.LOCKED_WAVELENGTH_PX * dc.LOCKED_MEDIUM_RI)
    nm = float(dc.LOCKED_MEDIUM_RI)
    rec = odtbrain.backpropagate_2d(
        uSin=complex_sinogram,
        angles=angles,
        res=res,
        nm=nm,
        lD=0.0,
        coords=None,
        weight_angles=False,
        onlyreal=False,
        padding=True,
    )
    # ODTbrain returns the object function f(r), which is the same physical
    # scattering potential q used by this BRDT contract.
    q = rec.real if np.iscomplexobj(rec) else rec
    return np.asarray(q, dtype=np.float32)


def derive_born_init_image(
    sinogram: torch.Tensor,
    *,
    operator: Optional[BornRytovForward2D] = None,
    backend: Optional[ClassicalBackendInfo] = None,
) -> torch.Tensor:
    """Return ``born_init_image`` of shape ``(B, 1, N, N)`` in physical q units.

    If ``backend`` is ``None``, the backend is auto-detected. The local
    adjoint backend is differentiable end-to-end through the operator;
    the ODTbrain backend runs in NumPy and the result is cast back to a
    detached tensor.
    """
    if backend is None:
        backend = detect_classical_backend()
    if operator is None:
        operator = _build_local_operator(device=sinogram.device)
    assert operator is not None  # for type narrowers

    if backend.name == "odtbrain":  # pragma: no cover - environment dependent
        sino_np = sinogram.detach().cpu().numpy()
        angles_np = operator.angles.detach().cpu().numpy()
        out = np.empty(
            (sino_np.shape[0], 1, operator.grid_size, operator.grid_size),
            dtype=np.float32,
        )
        for b in range(sino_np.shape[0]):
            out[b, 0] = _odtbrain_backprop(sino_np[b], angles=angles_np)
        return torch.from_numpy(out).to(device=sinogram.device, dtype=sinogram.dtype)

    if backend.name == "local_adjoint":
        return _local_adjoint_init(sinogram, operator=operator)

    raise ValueError(f"unknown classical backend: {backend.name!r}")


def classical_reconstruction(
    sinogram: torch.Tensor,
    *,
    backend: Optional[ClassicalBackendInfo] = None,
) -> torch.Tensor:
    """Classical Born reconstruction used for the reference row.

    For the bounded preflight, this is the same backprop adjoint used to
    derive ``born_init_image`` so the classical row is a single
    well-defined operation rather than a tuned inverse solver. Rows must
    treat the result as ``feasibility_only`` unless ODTbrain is the
    selected backend.
    """
    return derive_born_init_image(sinogram, backend=backend)
