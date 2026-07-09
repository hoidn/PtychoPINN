# tests/torch/test_fno_generators.py
"""Tests for FNO generator implementations."""
import math
import pytest
import torch

from ptycho_torch.generators.fno import (
    InputTransform,
    SpatialLifter,
    PtychoBlock,
    StablePtychoBlock,
    CascadedFNOGenerator,
    FnoGenerator,
)
from ptycho_torch.generators.fno_vanilla import FnoVanillaGeneratorModule
from ptycho_torch.generators.registry import resolve_generator
from ptycho.config.config import TrainingConfig, ModelConfig


class TestSpatialLifter:
    """Tests for the SpatialLifter module."""

    def test_lifter_preserves_spatial_dims(self):
        """Lifter should preserve H, W dimensions."""
        lifter = SpatialLifter(in_channels=4, out_channels=32)
        x = torch.randn(2, 4, 64, 64)
        out = lifter(x)
        assert out.shape == (2, 32, 64, 64)

    def test_lifter_with_custom_hidden(self):
        """Lifter should work with custom hidden channels."""
        lifter = SpatialLifter(in_channels=4, out_channels=32, hidden_channels=64)
        x = torch.randn(2, 4, 64, 64)
        out = lifter(x)
        assert out.shape == (2, 32, 64, 64)


class TestInputTransform:
    """Tests for optional input dynamic-range transforms."""

    def test_input_transform_sqrt_matches_expected(self):
        x = torch.tensor([[[[0.0, 4.0], [9.0, 16.0]]]])
        transform = InputTransform(mode="sqrt", channels=1)
        out = transform(x)
        assert torch.allclose(out, torch.sqrt(x))

    def test_input_transform_log1p_matches_expected(self):
        x = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        transform = InputTransform(mode="log1p", channels=1)
        out = transform(x)
        assert torch.allclose(out, torch.log1p(x))


class TestPtychoBlock:
    """Tests for the PtychoBlock module."""

    def test_block_preserves_dims(self):
        """PtychoBlock should preserve all dimensions."""
        block = PtychoBlock(channels=32, modes=8)
        x = torch.randn(2, 32, 64, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_block_has_residual(self):
        """PtychoBlock should have residual connection."""
        block = PtychoBlock(channels=32, modes=8)
        x = torch.randn(2, 32, 64, 64)

        # With zero weights, output should approximately equal input (due to residual)
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        out = block(x)
        # Residual ensures output contains input information
        assert torch.allclose(out, x, atol=1e-5)


class TestCascadedFNOGenerator:
    """Tests for the CascadedFNOGenerator model."""

    def test_output_shape(self):
        """Generator should produce correct output shape."""
        model = CascadedFNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            fno_blocks=2,
            cnn_blocks=1,
            modes=8,
            C=4,
        )
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert out.shape == (2, 64, 64, 4, 2)

    def test_output_is_differentiable(self):
        """Output should be differentiable for training."""
        model = CascadedFNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            fno_blocks=2,
            cnn_blocks=1,
            modes=8,
            C=1,
        )
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestFnoVanillaGenerator:
    """Tests for the FnoVanillaGenerator module."""

    def test_output_shape_real_imag(self):
        """FnoVanillaGenerator should preserve resolution and emit real/imag output."""
        model = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)


class TestGeneratorRegistry:
    """Tests for generator registry with FNO generators."""

    @pytest.fixture
    def fno_config(self):
        """Create config for FNO generator."""
        return TrainingConfig(
            model=ModelConfig(architecture='fno', N=64, gridsize=1)
        )

    def test_resolve_fno_generator(self, fno_config):
        """Registry should resolve FNO generator."""
        gen = resolve_generator(fno_config)
        assert gen.name == 'fno'
        assert isinstance(gen, FnoGenerator)

    def test_fno_generator_builds_model(self, fno_config):
        """FNO generator should build a model."""
        from ptycho_torch.config_params import DataConfig, ModelConfig as PTModelConfig, TrainingConfig as PTTrainingConfig

        gen = resolve_generator(fno_config)

        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
        from ptycho_torch.model import PtychoPINN_Lightning

        pt_configs = {
            "data_config": DataConfig(N=64, C=4),
            "model_config": PTModelConfig(architecture='fno'),
            "training_config": PTTrainingConfig(),
            "inference_config": PTInferenceConfig(),
        }

        model = gen.build_model(pt_configs)
        assert isinstance(model, PtychoPINN_Lightning)
        assert isinstance(model.model.generator, CascadedFNOGenerator)

class TestStablePtychoBlock:
    """Tests for the StablePtychoBlock module.

    Task ID: FNO-STABILITY-OVERHAUL-001 Phase 5 (LayerScale)
    """

    def test_identity_init(self):
        """At initialization (layerscale ~1e-3), block(x) should be near x."""
        block = StablePtychoBlock(channels=32, modes=8)
        x = torch.randn(2, 32, 64, 64)
        with torch.no_grad():
            out = block(x)
        assert torch.allclose(out, x, atol=1e-2), (
            "StablePtychoBlock should be near-identity at init (small layerscale)"
        )
        # But NOT exactly identity — layerscale > 0 allows residual signal
        assert torch.max(torch.abs(out - x)) > 1e-6, (
            "LayerScale init > 0 should produce nonzero residual"
        )

    def test_zero_mean_update(self):
        """The norm-wrapped update should have zero spatial mean."""
        block = StablePtychoBlock(channels=32, modes=8)
        x = torch.randn(2, 32, 64, 64)
        with torch.no_grad():
            update = block(x) - x  # isolate the residual branch
        # Mean over spatial dims should be ~0 due to InstanceNorm
        spatial_mean = update.mean(dim=(2, 3))
        assert torch.allclose(spatial_mean, torch.zeros_like(spatial_mean), atol=1e-5), (
            "InstanceNorm should center the update to zero spatial mean"
        )

    def test_forward_shape(self):
        """StablePtychoBlock should preserve (B, C, H, W) shape."""
        block = StablePtychoBlock(channels=16, modes=4)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_layerscale_grad_flow(self):
        """LayerScale parameter should receive gradients during backprop."""
        block = StablePtychoBlock(channels=8, modes=4, layerscale_init=1e-3)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        loss = (block(x) ** 2).mean()
        loss.backward()
        assert block.layerscale.grad is not None
        assert block.layerscale.grad.norm().item() > 0


