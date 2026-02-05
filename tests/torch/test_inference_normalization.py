import numpy as np
import torch
from types import SimpleNamespace

import ptycho_torch.inference as inference
from ptycho_torch import helper as hh


def test_inference_uses_training_normalization_scale(monkeypatch):
    diff = np.full((2, 4, 4), 2.0, dtype=np.float32)
    raw_data = SimpleNamespace(
        diff3d=diff,
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        xcoords=np.zeros(2, dtype=np.float32),
        ycoords=np.zeros(2, dtype=np.float32),
    )

    mean_sum = np.mean(np.sum(diff ** 2, axis=(1, 2)))
    expected_scale = np.sqrt(((4 / 2) ** 2) / mean_sum)
    expected_tensor = torch.full((2, 1, 1, 1), expected_scale, dtype=torch.float32)

    class DummyModel:
        torch_loss_mode = "poisson"

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward_predict(self, diffraction, positions, probe, input_scale_factor):
            assert torch.allclose(
                input_scale_factor, expected_tensor.to(input_scale_factor.device)
            )
            return torch.ones(
                (diffraction.shape[0], 1, 4, 4),
                dtype=torch.complex64,
                device=diffraction.device,
            )

    def fake_reassemble(patches, offsets, data_cfg, model_cfg, padded_size=None):
        b, _, h, w = patches.shape
        return torch.ones((b, h, w), dtype=patches.dtype, device=patches.device), None, h

    monkeypatch.setattr(hh, "reassemble_patches_position_real", fake_reassemble)

    config = SimpleNamespace(n_groups=2)
    inference._run_inference_and_reconstruct(
        DummyModel(),
        raw_data,
        config,
        execution_config=None,
        device=torch.device("cpu"),
        quiet=True,
        intensity_scale=1.0,
    )
