"""Focused contracts for calibration and reconstruction-output ports."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ptycho.reconstruction_policy import CalibrationSpec, OutputSpec


def test_identity_calibration_preserves_canvas_and_skips_varpro():
    from ptycho_torch.reconstruction_ports import calibrate_reconstruction_canvas

    canvas = torch.tensor([[1 + 2j, 3 + 4j]], dtype=torch.complex64)

    def forbidden(_canvas):
        raise AssertionError("identity calibration invoked VarPro")

    calibrated, s1, s2 = calibrate_reconstruction_canvas(
        canvas,
        CalibrationSpec(method="identity_v1"),
        varpro_calibrator=forbidden,
    )

    assert calibrated is canvas
    assert float(s1) == 1.0
    assert float(s2) == 1.0


def test_varpro_calibration_delegates_once():
    from ptycho_torch.reconstruction_ports import calibrate_reconstruction_canvas

    canvas = torch.tensor([[1 + 2j]], dtype=torch.complex64)
    expected = canvas * (2 - 1j)
    calls = []

    def calibrate(value):
        calls.append(value)
        return expected, torch.tensor(2.0), torch.tensor(3.0)

    calibrated, s1, s2 = calibrate_reconstruction_canvas(
        canvas,
        CalibrationSpec(method="varpro_s1s2_v1"),
        varpro_calibrator=calibrate,
    )

    assert calls == [canvas]
    assert calibrated is expected
    assert float(s1) == 2.0
    assert float(s2) == 3.0


@pytest.mark.parametrize("leading_batch", [False, True])
def test_output_port_emits_amplitude_and_phase(leading_batch):
    from ptycho_torch.reconstruction_ports import present_reconstruction_canvas

    canvas = torch.tensor(
        [[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 2j]],
        dtype=torch.complex64,
    )
    input_canvas = canvas.unsqueeze(0) if leading_batch else canvas

    amplitude, phase = present_reconstruction_canvas(input_canvas, OutputSpec())

    np.testing.assert_allclose(amplitude, torch.abs(canvas).numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(phase, torch.angle(canvas).numpy(), rtol=0, atol=0)


def test_output_port_rejects_multiple_canvases():
    from ptycho_torch.reconstruction_ports import present_reconstruction_canvas

    with pytest.raises(ValueError, match="single complex canvas"):
        present_reconstruction_canvas(
            torch.ones((2, 4, 4), dtype=torch.complex64),
            OutputSpec(),
        )
