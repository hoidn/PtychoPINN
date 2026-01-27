# tests/torch/test_fno_integration.py
"""Integration tests for FNO/Hybrid end-to-end training and inference."""
import numpy as np
import pytest
import torch


def test_synthetic_npz_fixture_contract(synthetic_ptycho_npz):
    """Verify synthetic fixture provides all required NPZ keys."""
    train_npz, _ = synthetic_ptycho_npz
    data = dict(np.load(train_npz))
    for key in ("diffraction", "Y_I", "Y_phi", "coords_nominal", "coords_true"):
        assert key in data, f"Missing required key: {key}"


def test_pinn_uses_fno_generator_when_selected():
    """Verify PtychoPINN instantiates FNO generator when architecture is fno."""
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig
    from ptycho_torch.model import PtychoPINN

    cfg = ModelConfig(architecture="fno")
    data_cfg = DataConfig(C=1, grid_size=(1, 1), N=64)
    train_cfg = TrainingConfig()

    model = PtychoPINN(cfg, data_cfg, train_cfg)
    assert model.generator is not None


@pytest.mark.slow
def test_fno_generator_forward_pass():
    """Verify FNO generator produces valid output shape and gradients."""
    from ptycho_torch.generators.fno import CascadedFNOGenerator

    # Create small model for testing
    model = CascadedFNOGenerator(
        in_channels=1,
        out_channels=2,
        hidden_channels=8,  # Small for speed
        fno_blocks=2,
        cnn_blocks=1,
        modes=4,
        C=1,  # gridsize=1
    )

    # Create synthetic input
    batch_size = 2
    N = 32
    C = 1
    x = torch.randn(batch_size, C, N, N, requires_grad=True)

    # Forward pass
    output = model(x)

    # Verify output shape: (B, N, N, C, 2) for real/imag
    assert output.shape == (batch_size, N, N, C, 2), f"Unexpected output shape: {output.shape}"

    # Verify gradients flow
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients did not flow to input"


@pytest.mark.slow
def test_hybrid_generator_forward_pass():
    """Verify Hybrid U-NO generator produces valid output shape and gradients."""
    from ptycho_torch.generators.fno import HybridUNOGenerator

    # Create small model for testing
    model = HybridUNOGenerator(
        in_channels=1,
        out_channels=2,
        hidden_channels=8,  # Small for speed
        n_blocks=2,
        modes=4,
        C=1,
    )

    # Create synthetic input
    batch_size = 2
    N = 32
    C = 1
    x = torch.randn(batch_size, C, N, N, requires_grad=True)

    # Forward pass
    output = model(x)

    # Verify output shape
    assert output.shape == (batch_size, N, N, C, 2)

    # Verify differentiability
    loss = output.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.slow
def test_neuraloperator_spectral_conv_available():
    """Verify HAS_NEURALOPERATOR flag is set correctly."""
    from ptycho_torch.generators.fno import HAS_NEURALOPERATOR

    # Just verify the flag exists and is a boolean
    assert isinstance(HAS_NEURALOPERATOR, bool)

    if HAS_NEURALOPERATOR:
        # If available, verify we can import the actual module
        from neuraloperator.layers.spectral_convolution import SpectralConv
        assert SpectralConv is not None


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__('ptycho_torch.generators.fno', fromlist=['HAS_NEURALOPERATOR']).HAS_NEURALOPERATOR,
    reason="neuraloperator not installed"
)
def test_ptychoblock_uses_neuraloperator_when_available():
    """When neuraloperator is available, PtychoBlock should use SpectralConv."""
    from ptycho_torch.generators.fno import PtychoBlock
    from neuraloperator.layers.spectral_convolution import SpectralConv

    block = PtychoBlock(channels=32, modes=8)

    # The spectral layer should be the neuraloperator SpectralConv
    assert isinstance(block.spectral, SpectralConv), \
        f"Expected SpectralConv but got {type(block.spectral)}"


@pytest.mark.slow
def test_fno_generator_training_loop():
    """Verify FNO generator can be trained with gradient descent."""
    from ptycho_torch.generators.fno import CascadedFNOGenerator

    model = CascadedFNOGenerator(
        in_channels=1,
        out_channels=2,
        hidden_channels=8,
        fno_blocks=2,
        cnn_blocks=1,
        modes=4,
        C=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Synthetic data
    batch_size = 4
    N = 32
    x = torch.randn(batch_size, 1, N, N)
    target = torch.randn(batch_size, N, N, 1, 2)

    # Training loop
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Verify loss was collected
    assert len(losses) == 3, f"Expected 3 loss values, got {len(losses)}"

    # Loss should be finite
    for i, loss_val in enumerate(losses):
        assert np.isfinite(loss_val), f"Loss at step {i} is not finite: {loss_val}"
