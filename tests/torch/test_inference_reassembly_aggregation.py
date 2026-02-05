import numpy as np
import pytest


def test_inference_reassembles_all_patches():
    import torch

    from ptycho.raw_data import RawData
    from ptycho_torch.inference import _run_inference_and_reconstruct

    class DummyModel:
        torch_loss_mode = "poisson"

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward_predict(self, diffraction, positions, probe, input_scale_factor):
            batch, _, n, _ = diffraction.shape
            out = torch.zeros((batch, 1, n, n), dtype=torch.complex64, device=diffraction.device)
            if batch > 1:
                out[1, 0, n // 2, n // 2] = 1.0 + 0.0j
            return out

    n = 4
    diff3d = np.ones((2, n, n), dtype=np.float32)
    xcoords = np.array([0.0, 1.0], dtype=np.float32)
    ycoords = np.array([0.0, 0.0], dtype=np.float32)
    xcoords_start = xcoords.copy()
    ycoords_start = ycoords.copy()
    probe_guess = np.ones((n, n), dtype=np.complex64)
    scan_index = np.zeros(2, dtype=int)

    raw_data = RawData(
        xcoords,
        ycoords,
        xcoords_start,
        ycoords_start,
        diff3d,
        probe_guess,
        scan_index,
    )

    class SimpleConfig:
        n_groups = 2
        stitch_crop_size = 4

    amp, _ = _run_inference_and_reconstruct(
        model=DummyModel(),
        raw_data=raw_data,
        config=SimpleConfig(),
        execution_config=None,
        device="cpu",
        quiet=True,
    )

    assert np.any(amp > 1e-6)
