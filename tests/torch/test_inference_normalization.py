import numpy as np
import pytest
import torch
from types import SimpleNamespace

import ptycho_torch.inference as inference
from ptycho_torch import helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig


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

    def fake_reassemble(patches, offsets, data_cfg, model_cfg, padded_size=None, **_kwargs):
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


def test_simplified_ci_inference_fails_before_computing_output_scale(monkeypatch):
    raw_data = SimpleNamespace(
        diff3d=np.ones((2, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        xcoords=np.zeros(2, dtype=np.float32),
        ycoords=np.zeros(2, dtype=np.float32),
    )

    class CIModel:
        data_config = DataConfig(N=4, C=1, grid_size=(1, 1))
        model_config = ModelConfig(
            C_model=1,
            C_forward=1,
            physics_forward_mode="rectangular_scaled",
            cnn_output_mode="real_imag",
        )
        training_config = SimpleNamespace(torch_loss_mode="poisson")

        def to(self, _device):
            return self

        def eval(self):
            return self

    def fail_if_output_scale_is_computed(*_args, **_kwargs):
        raise AssertionError("simplified CI inference computed a discarded output scale")

    monkeypatch.setattr(
        hh,
        "get_physics_scaling_factor",
        fail_if_output_scale_is_computed,
    )

    with pytest.raises(RuntimeError, match="reconstruct_image_barycentric.*VarPro"):
        inference._run_inference_and_reconstruct(
            CIModel(),
            raw_data,
            SimpleNamespace(n_groups=2),
            execution_config=None,
            device=torch.device("cpu"),
            quiet=True,
        )


def test_explicit_legacy_simplified_inference_remains_available(monkeypatch):
    diff = np.ones((2, 4, 4), dtype=np.float32)
    raw_data = SimpleNamespace(
        diff3d=diff,
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        xcoords=np.zeros(2, dtype=np.float32),
        ycoords=np.zeros(2, dtype=np.float32),
    )

    class LegacyModel:
        torch_loss_mode = "poisson"
        data_config = DataConfig(
            N=4,
            C=1,
            grid_size=(1, 1),
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        )
        model_config = ModelConfig(
            C_model=1,
            C_forward=1,
            physics_forward_mode="rectangular_scaled",
            cnn_output_mode="real_imag",
        )

        def to(self, _device):
            return self

        def eval(self):
            return self

        def forward_predict(self, diffraction, positions, probe, input_scale_factor):
            return torch.ones(
                (diffraction.shape[0], 1, 4, 4),
                dtype=torch.complex64,
                device=diffraction.device,
            )

    def fake_reassemble(patches, offsets, data_cfg, model_cfg, padded_size=None, **_kwargs):
        b, _, h, w = patches.shape
        return torch.ones((b, h, w), dtype=patches.dtype), None, h

    monkeypatch.setattr(hh, "reassemble_patches_position_real", fake_reassemble)

    amplitude, phase = inference._run_inference_and_reconstruct(
        LegacyModel(),
        raw_data,
        SimpleNamespace(n_groups=2),
        execution_config=None,
        device=torch.device("cpu"),
        quiet=True,
        intensity_scale=1.0,
    )

    assert amplitude.shape == phase.shape
