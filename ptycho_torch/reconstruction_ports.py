"""Backend-native ports for reconstruction calibration and presentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import torch

from ptycho.reconstruction_policy import CalibrationSpec, OutputSpec


VarProCalibrator = Callable[
    [torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def calibrate_reconstruction_canvas(
    canvas: torch.Tensor,
    spec: CalibrationSpec,
    *,
    varpro_calibrator: Optional[VarProCalibrator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the selected calibration after assembly, or preserve identity."""

    if spec.method == "identity_v1":
        one = torch.tensor(1.0, device=canvas.device, dtype=torch.float32)
        return canvas, one, one
    if varpro_calibrator is None:
        raise ValueError("VarPro calibration requires a varpro_calibrator")
    return varpro_calibrator(canvas)


def present_reconstruction_canvas(
    canvas: torch.Tensor,
    spec: OutputSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Present one calibrated complex canvas as amplitude and phase arrays."""

    if canvas.ndim == 3 and canvas.shape[0] == 1:
        canvas = canvas[0]
    if canvas.ndim != 2 or not torch.is_complex(canvas):
        raise ValueError(
            "Output presentation requires a single complex canvas with shape (H, W)"
        )
    if spec.representation != "amplitude_phase":
        raise ValueError(
            f"Unsupported output representation: {spec.representation!r}"
        )
    presented = canvas.detach().to("cpu")
    return torch.abs(presented).numpy(), torch.angle(presented).numpy()
