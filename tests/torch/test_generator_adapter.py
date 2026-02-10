# tests/torch/test_generator_adapter.py
"""
Tests for generator adapter path in ptycho_torch/model.py.

These tests verify the real/imag to complex channel-first conversion
used by FNO/Hybrid generators to integrate with PtychoPINN's physics pipeline.
"""
import torch
import torch.nn as nn
import pytest


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
