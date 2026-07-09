# tests/torch/test_generator_adapter.py
"""
Tests for generator adapter path in ptycho_torch/model.py.

These tests verify the real/imag to complex channel-first conversion
used by FNO/Hybrid generators to integrate with PtychoPINN's physics pipeline.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest

FNO_REAL_IMAG_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "varpro_parity"
    / "fno_real_imag_tensor_path.npz"
)


def test_real_imag_to_complex_channel_first():
    """Test conversion from (B, H, W, C, 2) real/imag to (B, C, H, W) complex."""
    from ptycho_torch.model import _real_imag_to_complex_channel_first

    batch = 2
    H = 8
    W = 8
    C = 4
    x = torch.zeros(batch, H, W, C, 2)
    x[..., 0] = 1.0  # Real part = 1.0

    out = _real_imag_to_complex_channel_first(x)
    assert out.shape == (batch, C, H, W)
    assert out.is_complex()
    assert torch.allclose(out.real, torch.ones_like(out.real))
    assert torch.allclose(out.imag, torch.zeros_like(out.imag))


def test_real_imag_to_complex_channel_first_with_imag():
    """Test that imaginary part is correctly converted."""
    from ptycho_torch.model import _real_imag_to_complex_channel_first

    batch = 2
    H = 4
    W = 4
    C = 2
    x = torch.zeros(batch, H, W, C, 2)
    x[..., 0] = 2.0  # Real part = 2.0
    x[..., 1] = 3.0  # Imag part = 3.0

    out = _real_imag_to_complex_channel_first(x)
    assert out.shape == (batch, C, H, W)
    assert torch.allclose(out.real, torch.full_like(out.real, 2.0))
    assert torch.allclose(out.imag, torch.full_like(out.imag, 3.0))


def test_real_imag_to_complex_channel_first_invalid_shape():
    """Test that invalid input shapes raise ValueError."""
    from ptycho_torch.model import _real_imag_to_complex_channel_first

    # Wrong number of dimensions
    with pytest.raises(ValueError, match="Expected real/imag tensor"):
        _real_imag_to_complex_channel_first(torch.zeros(2, 8, 8, 4))

    # Wrong last dimension
    with pytest.raises(ValueError, match="Expected real/imag tensor"):
        _real_imag_to_complex_channel_first(torch.zeros(2, 8, 8, 4, 3))


def test_ptychopinn_with_custom_generator():
    """Test that PtychoPINN accepts a custom generator with real_imag output."""
    from ptycho_torch.model import PtychoPINN
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig

    # Create minimal configs
    model_config = ModelConfig(mode='Unsupervised', C_model=4)
    data_config = DataConfig(N=64, C=4, grid_size=(2, 2))
    training_config = TrainingConfig()

    # Mock generator that outputs (B, H, W, C, 2) real/imag format
    class MockRealImagGenerator(nn.Module):
        def __init__(self, C):
            super().__init__()
            self.C = C
            # Minimal trainable parameter to make it a valid nn.Module
            self.dummy = nn.Parameter(torch.ones(1))

        def forward(self, x):
            B = x.shape[0]
            H, W = 96, 96  # PtychoPINN expects N + N//2 due to padding
            # Output (B, H, W, C, 2) real/imag format
            return torch.randn(B, H, W, self.C, 2)

    generator = MockRealImagGenerator(C=4)

    # Create PtychoPINN with custom generator
    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        generator=generator,
        generator_output="real_imag",
    )

    # Verify generator is used instead of autoencoder
    assert model.autoencoder is generator
    assert model.generator_output == "real_imag"


def test_ptychopinn_predict_complex_real_imag():
    """Test _predict_complex method with real_imag output mode."""
    from ptycho_torch.model import PtychoPINN
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig

    model_config = ModelConfig(mode='Unsupervised', C_model=4)
    data_config = DataConfig(N=64, C=4, grid_size=(2, 2))
    training_config = TrainingConfig()

    # Mock generator returning fixed real/imag values
    class FixedGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(1))

        def forward(self, x):
            B = x.shape[0]
            out = torch.zeros(B, 96, 96, 4, 2)
            out[..., 0] = 2.0  # Real
            out[..., 1] = 3.0  # Imag
            return out

    generator = FixedGenerator()
    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        generator=generator,
        generator_output="real_imag",
    )

    # Create dummy input
    x = torch.randn(2, 4, 64, 64)

    # Call _predict_complex
    x_complex, amp, phase = model._predict_complex(x)

    # Verify output shapes
    assert x_complex.shape == (2, 4, 96, 96)
    assert amp.shape == (2, 4, 96, 96)
    assert phase.shape == (2, 4, 96, 96)

    # Verify complex values
    assert x_complex.is_complex()
    expected_amp = (2.0**2 + 3.0**2) ** 0.5
    assert torch.allclose(amp, torch.full_like(amp, expected_amp), atol=1e-5)


# --- Task 2.3 (B1): tuple vs tensor real_imag contract ----------------------


def test_predict_complex_patches_accepts_cnn_tuple():
    """_predict_complex_patches must combine a CNN (real, imag) tuple of channel-first
    tensors via torch.complex, alongside the FNO/Hybrid (B,H,W,C,2) tensor path."""
    from ptycho_torch.model import _predict_complex_patches, CombineComplex

    class TupleHead(nn.Module):
        def forward(self, x):
            b, c, h, w = x.shape
            real = torch.full((b, c, h, w), 2.0, dtype=x.dtype)
            imag = torch.full((b, c, h, w), 3.0, dtype=x.dtype)
            return real, imag

    x = torch.randn(2, 1, 8, 8)
    x_complex, amp, phase = _predict_complex_patches(
        TupleHead(), CombineComplex(), "real_imag", x
    )

    assert x_complex.is_complex()
    assert x_complex.shape == (2, 1, 8, 8)
    assert torch.allclose(x_complex.real, torch.full_like(x_complex.real, 2.0))
    assert torch.allclose(x_complex.imag, torch.full_like(x_complex.imag, 3.0))


class _DeterministicTensorGenerator(nn.Module):
    """FNO/Hybrid-style generator returning a deterministic (B,H,W,C,2) tensor."""

    def __init__(self, H, W, C):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))
        self.H, self.W, self.C = H, W, C

    def forward(self, x):
        b = x.shape[0]
        n = self.H * self.W * self.C * 2
        base = torch.linspace(-1.0, 1.0, steps=n, dtype=x.dtype)
        base = base.reshape(1, self.H, self.W, self.C, 2)
        return base.expand(b, -1, -1, -1, -1).contiguous()


def test_fno_real_imag_tensor_path_byte_identical():
    """Task 2.1-style frozen pin: the FNO/Hybrid real_imag TENSOR path (B,H,W,C,2)
    stays byte-identical after the tuple-vs-tensor change to _predict_complex_patches.

    Also asserts the tensor branch equals the untouched adapter, proving the added
    tuple branch cannot perturb it.
    """
    from ptycho_torch.model import PtychoPINN, _real_imag_to_complex_channel_first
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig

    H = W = 96
    C = 1
    generator = _DeterministicTensorGenerator(H, W, C)

    model_config = ModelConfig(architecture='fno', mode='Unsupervised', C_model=C)
    data_config = DataConfig(N=64, C=C, grid_size=(1, 1))
    training_config = TrainingConfig()

    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        generator=generator,
        generator_output="real_imag",
    )
    assert model.generator_output == "real_imag"

    x = torch.zeros(2, C, data_config.N, data_config.N)
    with torch.no_grad():
        x_complex, _amp, _phase = model._predict_complex(x)
        expected = _real_imag_to_complex_channel_first(generator(x))

    # Tensor branch is the untouched adapter, exactly.
    assert torch.equal(x_complex, expected)

    real = x_complex.real.numpy()
    imag = x_complex.imag.numpy()

    if not FNO_REAL_IMAG_FIXTURE.exists():
        FNO_REAL_IMAG_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FNO_REAL_IMAG_FIXTURE, real=real, imag=imag)
        return

    fixture = np.load(FNO_REAL_IMAG_FIXTURE)
    np.testing.assert_array_equal(
        real, fixture["real"],
        err_msg="FNO real_imag tensor path (real) drifted from frozen fixture",
    )
    np.testing.assert_array_equal(
        imag, fixture["imag"],
        err_msg="FNO real_imag tensor path (imag) drifted from frozen fixture",
    )
