# tests/torch/test_fno_generators.py
"""Tests for FNO/Hybrid generator implementations."""
import math
import pytest
import torch

from ptycho_torch.generators.fno import (
    InputTransform,
    SpatialLifter,
    PtychoBlock,
    StablePtychoBlock,
    HybridUNOGenerator,
    StableHybridUNOGenerator,
    CascadedFNOGenerator,
    HybridGenerator,
    StableHybridGenerator,
    FnoGenerator,
)
from ptycho_torch.generators.fno_vanilla import FnoVanillaGeneratorModule
from ptycho_torch.generators.hybrid_resnet import (
    HybridResnetEncoderBlock,
    HybridResnetGeneratorModule,
)
from ptycho_torch.generators.resnet_components import (
    ResnetBlock,
    ResnetBottleneck,
    CycleGanUpsampler,
)
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


class TestHybridUNOGenerator:
    """Tests for the HybridUNOGenerator model."""

    def test_output_shape(self):
        """Generator should produce correct output shape."""
        model = HybridUNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
        )
        x = torch.randn(2, 4, 64, 64)  # (B, C, H, W)
        out = model(x)
        # Expected: (B, H, W, C, 2)
        assert out.shape == (2, 64, 64, 4, 2)

    def test_output_contract_real_imag(self):
        """Output should have real/imag in last dimension."""
        model = HybridUNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=1,
        )
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        # Last dim should be 2 (real, imag)
        assert out.shape[-1] == 2


    def test_max_hidden_channel_cap(self):
        """HybridUNOGenerator with max_hidden_channels=512 keeps all layers ≤512 channels."""
        gen = HybridUNOGenerator(
            in_channels=1, out_channels=2, hidden_channels=32,
            n_blocks=6, modes=4, C=1, max_hidden_channels=512,
        )
        # Without cap, 6 blocks would reach 32*2^5=1024; cap should keep ≤512
        for ch in gen._encoder_channels:
            assert ch <= 512, f"Encoder channel {ch} exceeds cap 512"
        for name, module in gen.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                assert module.out_channels <= 512, f"{name} out_channels={module.out_channels} exceeds cap"
                assert module.in_channels <= 1024, f"{name} in_channels={module.in_channels} exceeds 2*cap (skip concat)"
        x = torch.randn(2, 1, 64, 64)
        y = gen(x)
        assert y.shape == (2, 64, 64, 1, 2), f"Output shape {y.shape} != expected (2, 64, 64, 1, 2)"


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


class TestHybridResnetGenerator:
    """Tests for the HybridResnetGenerator module."""

    def test_decoder_components_use_gelu_activations(self):
        block = ResnetBlock(channels=8)
        upsampler = CycleGanUpsampler(in_channels=8, out_channels=4)

        assert any(isinstance(module, torch.nn.GELU) for module in block.modules())
        assert not any(isinstance(module, torch.nn.ReLU) for module in block.modules())
        assert any(isinstance(module, torch.nn.GELU) for module in upsampler.modules())
        assert not any(isinstance(module, torch.nn.ReLU) for module in upsampler.modules())

    def test_decoder_upsampler_uses_standard_4x4_deconv(self):
        upsampler = CycleGanUpsampler(in_channels=8, out_channels=4)
        deconv = upsampler.block[0]

        assert isinstance(deconv, torch.nn.ConvTranspose2d)
        assert deconv.kernel_size == (4, 4)
        assert deconv.stride == (2, 2)
        assert deconv.padding == (1, 1)
        assert deconv.output_padding == (0, 0)

    def test_resnet_decoder_block_uses_scalar_layerscale(self):
        block = ResnetBlock(channels=8)
        assert block.layerscale.numel() == 1

    def test_resnet_bottleneck_shares_one_scalar_layerscale_across_blocks(self):
        bottleneck = ResnetBottleneck(channels=8, n_blocks=3)

        shared_gate = bottleneck.layerscale
        assert shared_gate.numel() == 1
        for block in bottleneck.blocks:
            assert block.layerscale is shared_gate

    def test_resnet_decoder_block_zero_layerscale_is_identity(self):
        block = ResnetBlock(channels=8)
        x = torch.randn(2, 8, 16, 16)

        with torch.no_grad():
            block.layerscale.zero_()

        out = block(x)
        assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)

    def test_hybrid_resnet_fixed_bottleneck_layerscale_is_non_trainable(self):
        bottleneck = ResnetBottleneck(
            channels=8,
            n_blocks=3,
            layerscale_mode="fixed",
            layerscale_value=1.0,
        )

        assert bottleneck.layerscale.numel() == 1
        assert bottleneck.layerscale.requires_grad is False
        assert bottleneck.layerscale.item() == pytest.approx(1.0)
        for block in bottleneck.blocks:
            assert block.layerscale is bottleneck.layerscale

    def test_hybrid_resnet_rejects_invalid_fixed_bottleneck_layerscale_value(self):
        with pytest.raises(ValueError, match="layerscale_value"):
            ResnetBottleneck(
                channels=8,
                n_blocks=3,
                layerscale_mode="fixed",
                layerscale_value=None,
            )

    def test_hybrid_resnet_rejects_invalid_bottleneck_layerscale_mode_combo(self):
        with pytest.raises(ValueError, match="layerscale_value"):
            ResnetBottleneck(
                channels=8,
                n_blocks=3,
                layerscale_mode="learned",
                layerscale_value=1.0,
            )

    def test_hybrid_resnet_fixed_bottleneck_layerscale_preserves_shape(self):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            bottleneck_layerscale_mode="fixed",
            bottleneck_layerscale_value=1.0,
        )
        x = torch.randn(2, 4, 32, 32)

        out = model(x)

        assert out.shape == (2, 32, 32, 4, 2)
        assert model.resnet.layerscale.requires_grad is False

    def test_output_shape_real_imag(self):
        """HybridResnetGenerator should preserve resolution and emit real/imag output."""
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)

    def test_amp_phase_bounds(self):
        """amp_phase output should be bounded by sigmoid/tanh scaling."""
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            output_mode="amp_phase",
        )
        x = torch.randn(1, 4, 32, 32)
        amp, phase = model(x)
        assert amp.min().item() >= 0.0
        assert amp.max().item() <= 1.0
        assert phase.min().item() >= -math.pi - 1e-3
        assert phase.max().item() <= math.pi + 1e-3

    @pytest.mark.parametrize("hybrid_downsample_steps", [1, 2])
    def test_hybrid_downsample_steps_skip_connections_forward(self, hybrid_downsample_steps):
        """Skip-enabled forward pass should work for both supported downsample schedules."""
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_downsample_steps=hybrid_downsample_steps,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)

    def test_hybrid_downsample_steps_metadata_drives_skip_fusion_points(self):
        """Fusion-point plan should derive from stage metadata, not fixed tap indices."""
        model_one = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_downsample_steps=1,
        )
        model_two = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_downsample_steps=2,
        )
        taps_one = [item["resolution_divisor"] for item in model_one.encoder_tap_plan]
        taps_two = [item["resolution_divisor"] for item in model_two.encoder_tap_plan]

        assert taps_one == [item["resolution_divisor"] for item in model_one.stage_metadata[:-1]]
        assert taps_two == [item["resolution_divisor"] for item in model_two.stage_metadata[:-1]]
        assert [item["resolution_divisor"] for item in model_one.skip_fusion_plan] == list(reversed(taps_one))
        assert [item["resolution_divisor"] for item in model_two.skip_fusion_plan] == list(reversed(taps_two))

    def test_skip_topology_derivation_uses_stage_metadata_values(self):
        tap_plan, skip_plan = HybridResnetGeneratorModule._derive_skip_topology_from_stage_metadata(
            [
                {"resolution_divisor": 1, "channels": 16},
                {"resolution_divisor": 3, "channels": 32},
                {"resolution_divisor": 7, "channels": 64},
            ],
            downsample_steps=2,
        )

        assert [item["resolution_divisor"] for item in tap_plan] == [1, 3]
        assert [item["key"] for item in tap_plan] == ["d1", "d3"]
        assert [item["resolution_divisor"] for item in skip_plan] == [3, 1]
        assert [item["key"] for item in skip_plan] == ["d3", "d1"]

    def test_invalid_hybrid_downsample_steps_raises(self):
        with pytest.raises(ValueError, match="hybrid_downsample_steps"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                hybrid_downsample_steps=3,
            )

    @pytest.mark.parametrize("hybrid_downsample_op", ["stride_conv", "avgpool_conv", "blurpool_conv"])
    def test_hybrid_downsample_op_shape_invariance(self, hybrid_downsample_op):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_downsample_steps=2,
            hybrid_downsample_op=hybrid_downsample_op,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)

    def test_invalid_hybrid_downsample_op_raises(self):
        with pytest.raises(ValueError, match="hybrid_downsample_op"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                hybrid_downsample_op="bad_op",
            )

    def test_hybrid_downsample_op_branches_are_distinct(self):
        x = torch.randn(2, 4, 32, 32)

        torch.manual_seed(19)
        stride_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_downsample_op="stride_conv",
        )
        torch.manual_seed(19)
        avg_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_downsample_op="avgpool_conv",
        )
        torch.manual_seed(19)
        blur_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_downsample_op="blurpool_conv",
        )

        assert stride_model.downsample_layers[0].__class__ is not avg_model.downsample_layers[0].__class__
        assert stride_model.downsample_layers[0].__class__ is not blur_model.downsample_layers[0].__class__
        assert avg_model.downsample_layers[0].__class__ is not blur_model.downsample_layers[0].__class__

        y_stride = stride_model(x)
        y_avg = avg_model(x)
        y_blur = blur_model(x)
        assert not torch.allclose(y_stride, y_avg, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(y_stride, y_blur, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(y_avg, y_blur, atol=1e-6, rtol=1e-6)

    def test_hybrid_encoder_conv_hidden_scale_default_parity(self):
        torch.manual_seed(7)
        model_default = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
        )
        torch.manual_seed(7)
        model_explicit = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_encoder_conv_hidden_scale=1.0,
            hybrid_encoder_spectral_hidden_scale=1.0,
        )
        x = torch.randn(2, 4, 32, 32)
        assert torch.allclose(model_default(x), model_explicit(x), atol=1e-6, rtol=1e-6)

    def test_hybrid_encoder_conv_hidden_scale_changes_output(self):
        torch.manual_seed(17)
        model_default = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
        )
        torch.manual_seed(17)
        model_custom = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_encoder_conv_hidden_scale=2.0,
        )
        x = torch.randn(2, 4, 32, 32)
        assert not torch.allclose(model_default(x), model_custom(x), atol=1e-6, rtol=1e-6)

    def test_hybrid_encoder_spectral_hidden_scale_changes_output(self):
        torch.manual_seed(23)
        model_default = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
        )
        torch.manual_seed(23)
        model_custom = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_encoder_spectral_hidden_scale=0.5,
        )
        x = torch.randn(2, 4, 32, 32)
        assert not torch.allclose(model_default(x), model_custom(x), atol=1e-6, rtol=1e-6)

    def test_hybrid_encoder_conv_hidden_scale_resolved_width_mapping(self):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            hybrid_encoder_conv_hidden_scale=0.5,
            hybrid_encoder_spectral_hidden_scale=2.0,
        )
        assert model.encoder_stage_channels == [16, 32, 64]
        assert model.encoder_conv_hidden_resolved_per_block == [8, 16, 32]
        assert model.encoder_spectral_hidden_resolved_per_block == [32, 64, 128]

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_hybrid_encoder_conv_hidden_scale_raises(self, bad_scale):
        with pytest.raises(ValueError, match="hybrid_encoder_conv_hidden_scale"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                hybrid_encoder_conv_hidden_scale=bad_scale,
            )

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_hybrid_encoder_spectral_hidden_scale_raises(self, bad_scale):
        with pytest.raises(ValueError, match="hybrid_encoder_spectral_hidden_scale"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                hybrid_encoder_spectral_hidden_scale=bad_scale,
            )

    def test_invalid_resnet_width_raises(self):
        with pytest.raises(ValueError, match="resnet_width"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                resnet_width=255,
            )

    @pytest.mark.parametrize("hybrid_resnet_blocks", [4, 6, 8])
    def test_hybrid_resnet_blocks_shape_invariance(self, hybrid_resnet_blocks):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            resnet_blocks=hybrid_resnet_blocks,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)

    @pytest.mark.parametrize("hybrid_skip_style", ["add", "concat", "gated_add"])
    def test_skip_style_shape_contract(self, hybrid_skip_style):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_skip_style=hybrid_skip_style,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)

    def test_invalid_skip_style_raises(self):
        with pytest.raises(ValueError, match="hybrid_skip_style"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                skip_connections=True,
                hybrid_skip_style="bad_style",
            )

    def test_skip_style_branches_are_distinct(self):
        x = torch.randn(2, 4, 32, 32)

        torch.manual_seed(31)
        add_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_skip_style="add",
        )
        torch.manual_seed(31)
        concat_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_skip_style="concat",
        )
        torch.manual_seed(31)
        gated_model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_skip_style="gated_add",
        )

        y_add = add_model(x)
        y_concat = concat_model(x)
        y_gated = gated_model(x)
        assert not torch.allclose(y_add, y_concat, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(y_add, y_gated, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(y_concat, y_gated, atol=1e-6, rtol=1e-6)

    def test_gated_add_skip_style_starts_with_small_positive_gate(self):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            skip_connections=True,
            hybrid_skip_style="gated_add",
        )

        assert sorted(model.skip_fusion_gates.keys()) == ["d1", "d2"]
        for gate in model.skip_fusion_gates.values():
            assert gate.item() == pytest.approx(0.1)

    def test_hybrid_resnet_encoder_local_conv_uses_reflect_padding(self):
        block = HybridResnetEncoderBlock(channels=16, modes=4)
        assert block.local_conv.padding_mode == "reflect"

    def test_hybrid_resnet_encoder_baseline_has_no_fusion_scalars(self):
        block = HybridResnetEncoderBlock(channels=16, modes=4)
        assert block.encoder_fusion_mode == "baseline"
        assert block.encoder_layerscale is None
        assert block.spectral_branch_gate is None
        assert block.local_branch_gate is None

    def test_hybrid_resnet_encoder_layerscale_mode_initializes_per_block_scalar(self):
        block = HybridResnetEncoderBlock(
            channels=16, modes=4, encoder_fusion_mode="layerscale"
        )
        assert block.encoder_layerscale is not None
        assert block.encoder_layerscale.shape == (1,)
        assert float(block.encoder_layerscale.item()) == pytest.approx(0.1)
        assert float(block.encoder_layerscale.item()) > 0.0
        assert block.spectral_branch_gate is None
        assert block.local_branch_gate is None

    def test_hybrid_resnet_encoder_branch_gated_mode_initializes_per_block_gates(self):
        block = HybridResnetEncoderBlock(
            channels=16, modes=4, encoder_fusion_mode="branch_gated"
        )
        assert block.encoder_layerscale is None
        assert block.spectral_branch_gate is not None
        assert block.local_branch_gate is not None
        assert float(block.spectral_branch_gate.item()) == pytest.approx(0.1)
        assert float(block.local_branch_gate.item()) == pytest.approx(0.1)
        assert float(block.spectral_branch_gate.item()) > 0.0
        assert float(block.local_branch_gate.item()) > 0.0

    def test_hybrid_resnet_encoder_branch_gated_layerscale_has_all_three_per_block_scalars(self):
        block = HybridResnetEncoderBlock(
            channels=16, modes=4, encoder_fusion_mode="branch_gated_layerscale"
        )
        assert block.encoder_layerscale is not None
        assert block.spectral_branch_gate is not None
        assert block.local_branch_gate is not None
        for scalar in (
            block.encoder_layerscale,
            block.spectral_branch_gate,
            block.local_branch_gate,
        ):
            assert float(scalar.item()) == pytest.approx(0.1)
            assert float(scalar.item()) > 0.0

    @pytest.mark.parametrize(
        "fusion_mode",
        ["baseline", "layerscale", "branch_gated", "branch_gated_layerscale"],
    )
    def test_hybrid_resnet_encoder_fusion_mode_preserves_shape(self, fusion_mode):
        torch.manual_seed(11)
        block = HybridResnetEncoderBlock(
            channels=16, modes=4, encoder_fusion_mode=fusion_mode
        )
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize(
        "fusion_mode",
        ["baseline", "layerscale", "branch_gated", "branch_gated_layerscale"],
    )
    def test_hybrid_resnet_encoder_fusion_mode_preserves_identity_residual(self, fusion_mode):
        """The identity residual path x_next = x + (...) must hold for all encoder fusion modes."""
        torch.manual_seed(13)
        block = HybridResnetEncoderBlock(
            channels=16, modes=4, encoder_fusion_mode=fusion_mode
        )
        block.eval()
        # Zero out all branch parameters so the update path emits a zero contribution
        # and we can confirm the residual passthrough.
        with torch.no_grad():
            for p in block.spectral.parameters():
                p.zero_()
            for p in block.local_conv.parameters():
                p.zero_()
            # spectral_in / spectral_out / conv_in / conv_out are nn.Identity unless widths differ;
            # with channels==hidden==16 there are no extra projections to zero.
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        # GELU(0) = 0, so the update is zero and y == x.
        assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

    def test_hybrid_resnet_encoder_invalid_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="encoder_fusion_mode"):
            HybridResnetEncoderBlock(channels=16, modes=4, encoder_fusion_mode="bogus")

    @pytest.mark.parametrize("bad_value", [0.0, -0.1, math.inf, math.nan])
    def test_hybrid_resnet_encoder_invalid_layerscale_init_raises(self, bad_value):
        with pytest.raises(ValueError, match="encoder_layerscale_init"):
            HybridResnetEncoderBlock(
                channels=16,
                modes=4,
                encoder_fusion_mode="layerscale",
                encoder_layerscale_init=bad_value,
            )

    @pytest.mark.parametrize("bad_value", [0.0, -0.1, math.inf, math.nan])
    def test_hybrid_resnet_encoder_invalid_branch_gate_init_raises(self, bad_value):
        with pytest.raises(ValueError, match="encoder_branch_gate_init"):
            HybridResnetEncoderBlock(
                channels=16,
                modes=4,
                encoder_fusion_mode="branch_gated",
                encoder_branch_gate_init=bad_value,
            )

    @pytest.mark.parametrize(
        "fusion_mode",
        ["layerscale", "branch_gated", "branch_gated_layerscale"],
    )
    def test_hybrid_resnet_module_propagates_fusion_mode_per_block(self, fusion_mode):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            encoder_fusion_mode=fusion_mode,
        )
        assert model.encoder_fusion_mode == fusion_mode
        for block in model.encoder_blocks:
            assert block.encoder_fusion_mode == fusion_mode
            if fusion_mode in {"layerscale", "branch_gated_layerscale"}:
                assert block.encoder_layerscale is not None
            else:
                assert block.encoder_layerscale is None
            if fusion_mode in {"branch_gated", "branch_gated_layerscale"}:
                assert block.spectral_branch_gate is not None
                assert block.local_branch_gate is not None
            else:
                assert block.spectral_branch_gate is None
                assert block.local_branch_gate is None

    def test_hybrid_resnet_module_invalid_fusion_mode_raises(self):
        with pytest.raises(ValueError, match="encoder_fusion_mode"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                encoder_fusion_mode="bogus",
            )

    def test_hybrid_resnet_module_default_fusion_mode_is_baseline(self):
        """Default behavior is unchanged: no encoder-fusion scalars under default construction."""
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
        )
        assert model.encoder_fusion_mode == "baseline"
        for block in model.encoder_blocks:
            assert block.encoder_layerscale is None
            assert block.spectral_branch_gate is None
            assert block.local_branch_gate is None

    @pytest.mark.parametrize("select", ["both", "conv_only", "spectral_only"])
    def test_hybrid_resnet_encoder_branch_select_preserves_shape(self, select):
        torch.manual_seed(17)
        block = HybridResnetEncoderBlock(channels=16, modes=4, encoder_branch_select=select)
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        assert y.shape == x.shape
        assert block.encoder_branch_select == select

    def test_hybrid_resnet_encoder_branch_select_default_is_both(self):
        block = HybridResnetEncoderBlock(channels=16, modes=4)
        assert block.encoder_branch_select == "both"

    def test_hybrid_resnet_encoder_branch_select_conv_only_drops_spectral_params(self):
        block = HybridResnetEncoderBlock(channels=16, modes=4, encoder_branch_select="conv_only")
        # Spectral submodules are nn.Identity placeholders carrying no learnable params.
        assert isinstance(block.spectral, torch.nn.Identity)
        assert isinstance(block.spectral_in, torch.nn.Identity)
        assert isinstance(block.spectral_out, torch.nn.Identity)
        # Conv branch must remain a learnable Conv2d.
        assert isinstance(block.local_conv, torch.nn.Conv2d)
        spectral_params = sum(p.numel() for p in block.spectral.parameters()) + \
                          sum(p.numel() for p in block.spectral_in.parameters()) + \
                          sum(p.numel() for p in block.spectral_out.parameters())
        assert spectral_params == 0

    def test_hybrid_resnet_encoder_branch_select_spectral_only_drops_conv_params(self):
        block = HybridResnetEncoderBlock(channels=16, modes=4, encoder_branch_select="spectral_only")
        assert isinstance(block.local_conv, torch.nn.Identity)
        assert isinstance(block.conv_in, torch.nn.Identity)
        assert isinstance(block.conv_out, torch.nn.Identity)
        # Spectral branch must remain a non-Identity learnable module.
        assert not isinstance(block.spectral, torch.nn.Identity)
        local_params = sum(p.numel() for p in block.local_conv.parameters()) + \
                       sum(p.numel() for p in block.conv_in.parameters()) + \
                       sum(p.numel() for p in block.conv_out.parameters())
        assert local_params == 0

    def test_hybrid_resnet_encoder_branch_select_invalid_raises(self):
        with pytest.raises(ValueError, match="encoder_branch_select"):
            HybridResnetEncoderBlock(channels=16, modes=4, encoder_branch_select="bogus")

    def test_hybrid_resnet_encoder_branch_select_orthogonal_to_branch_gated(self):
        """Combining branch_gated fusion with branch_select must instantiate gates only for active branches."""
        block_conv = HybridResnetEncoderBlock(
            channels=16,
            modes=4,
            encoder_fusion_mode="branch_gated",
            encoder_branch_select="conv_only",
        )
        assert block_conv.spectral_branch_gate is None
        assert block_conv.local_branch_gate is not None
        block_spec = HybridResnetEncoderBlock(
            channels=16,
            modes=4,
            encoder_fusion_mode="branch_gated",
            encoder_branch_select="spectral_only",
        )
        assert block_spec.spectral_branch_gate is not None
        assert block_spec.local_branch_gate is None

    def test_hybrid_resnet_encoder_branch_select_conv_only_zero_means_residual_passthrough(self):
        """Zeroing the active conv branch reduces the block to identity (deterministic ablation)."""
        torch.manual_seed(31)
        block = HybridResnetEncoderBlock(channels=16, modes=4, encoder_branch_select="conv_only")
        block.eval()
        with torch.no_grad():
            for p in block.local_conv.parameters():
                p.zero_()
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("select", ["both", "conv_only", "spectral_only"])
    def test_hybrid_resnet_module_propagates_branch_select_per_block(self, select):
        model = HybridResnetGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=3,
            modes=4,
            C=4,
            encoder_branch_select=select,
        )
        assert model.encoder_branch_select == select
        for block in model.encoder_blocks:
            assert block.encoder_branch_select == select

    def test_hybrid_resnet_module_invalid_branch_select_raises(self):
        with pytest.raises(ValueError, match="encoder_branch_select"):
            HybridResnetGeneratorModule(
                in_channels=1,
                out_channels=2,
                hidden_channels=16,
                n_blocks=3,
                modes=4,
                C=4,
                encoder_branch_select="bogus",
            )

    def test_hybrid_resnet_ffno_ptychoblock_encoder_module_records_fixed_recipe(self):
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetFfnoPtychoBlockEncoderGeneratorModule,
        )

        torch.manual_seed(19)
        model = HybridResnetFfnoPtychoBlockEncoderGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            modes=4,
            C=4,
            resnet_blocks=6,
            skip_connections=True,
            hybrid_skip_style="add",
        )
        x = torch.randn(2, 4, 32, 32)
        y = model(x)

        assert y.shape == (2, 32, 32, 4, 2)
        assert model.encoder_variant == "ffno_ptychoblock_encoder"
        assert model.ffno_encoder_blocks == 2
        assert model.ffno_encoder_modes == 4
        assert model.ffno_encoder_share_weights is True
        assert model.ffno_encoder_gate_init == pytest.approx(0.1)
        assert model.ffno_encoder_norm == "instance"
        assert model.ffno_encoder_mlp_ratio == pytest.approx(2.0)
        assert model.ptychoblock_stage_count == 2
        assert model.hybrid_downsample_steps == 2


class TestGeneratorRegistry:
    """Tests for generator registry with FNO/Hybrid generators."""

    @pytest.fixture
    def fno_config(self):
        """Create config for FNO generator."""
        return TrainingConfig(
            model=ModelConfig(architecture='fno', N=64, gridsize=1)
        )

    @pytest.fixture
    def hybrid_config(self):
        """Create config for Hybrid generator."""
        return TrainingConfig(
            model=ModelConfig(architecture='hybrid', N=64, gridsize=1)
        )

    def test_resolve_fno_generator(self, fno_config):
        """Registry should resolve FNO generator."""
        gen = resolve_generator(fno_config)
        assert gen.name == 'fno'
        assert isinstance(gen, FnoGenerator)

    def test_resolve_hybrid_generator(self, hybrid_config):
        """Registry should resolve Hybrid generator."""
        gen = resolve_generator(hybrid_config)
        assert gen.name == 'hybrid'
        assert isinstance(gen, HybridGenerator)

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

    def test_hybrid_generator_builds_model(self, hybrid_config):
        """Hybrid generator should build a model."""
        from ptycho_torch.config_params import DataConfig, ModelConfig as PTModelConfig, TrainingConfig as PTTrainingConfig

        gen = resolve_generator(hybrid_config)

        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
        from ptycho_torch.model import PtychoPINN_Lightning

        pt_configs = {
            "data_config": DataConfig(N=64, C=4),
            "model_config": PTModelConfig(architecture='hybrid'),
            "training_config": PTTrainingConfig(),
            "inference_config": PTInferenceConfig(),
        }

        model = gen.build_model(pt_configs)
        assert isinstance(model, PtychoPINN_Lightning)
        assert isinstance(model.model.generator, HybridUNOGenerator)

    def test_hybrid_resnet_generator_builds_fixed_residual_from_execution_overrides(self):
        """Hybrid ResNet fixed residual control should stay out of PTModelConfig but still build."""
        from ptycho.config.config import PyTorchExecutionConfig
        from ptycho_torch.config_params import (
            DataConfig,
            InferenceConfig as PTInferenceConfig,
            ModelConfig as PTModelConfig,
            TrainingConfig as PTTrainingConfig,
        )
        from ptycho_torch.generators.hybrid_resnet import HybridResnetGenerator
        from ptycho_torch.model import PtychoPINN_Lightning

        config = TrainingConfig(
            model=ModelConfig(architecture="hybrid_resnet", N=64, gridsize=1)
        )
        gen = HybridResnetGenerator(config)
        execution_config = PyTorchExecutionConfig(
            hybrid_resnet_bottleneck_layerscale_mode="fixed",
            hybrid_resnet_bottleneck_layerscale_value=1.0,
        )
        pt_model_config = PTModelConfig(architecture="hybrid_resnet")
        pt_configs = {
            "data_config": DataConfig(N=64, C=1),
            "model_config": pt_model_config,
            "training_config": PTTrainingConfig(),
            "inference_config": PTInferenceConfig(),
            "execution_config": execution_config,
        }

        model = gen.build_model(pt_configs)

        assert isinstance(model, PtychoPINN_Lightning)
        assert model.model.generator.resnet.layerscale.requires_grad is False
        assert model.hparams["generator_overrides"]["hybrid_resnet_bottleneck_layerscale_mode"] == "fixed"
        assert model.hparams["generator_overrides"]["hybrid_resnet_bottleneck_layerscale_value"] == 1.0
        assert "hybrid_resnet_bottleneck_layerscale_mode" not in model.hparams["model_config"]
        assert "hybrid_resnet_bottleneck_layerscale_value" not in model.hparams["model_config"]

    def test_resolve_stable_hybrid_generator(self):
        """Registry should resolve stable_hybrid generator."""
        config = TrainingConfig(
            model=ModelConfig(architecture='stable_hybrid', N=64, gridsize=1)
        )
        gen = resolve_generator(config)
        assert gen.name == 'stable_hybrid'
        assert isinstance(gen, StableHybridGenerator)


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


class TestStableHybridUNOGenerator:
    """Tests for the StableHybridUNOGenerator model.

    Task ID: FNO-STABILITY-OVERHAUL-001 Task 2.2
    """

    def test_stable_hybrid_generator_output_shape(self):
        """StableHybridUNOGenerator should produce (B, H, W, C, 2) output."""
        model = StableHybridUNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
        )
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert out.shape == (2, 64, 64, 4, 2)


class TestConvNextBottleneck:
    """Tests for the ConvNeXt-style bottleneck added for the SRU-Net ConvNeXt
    bottleneck ablation (architecture id ``hybrid_resnet_convnext_bottleneck``).
    """

    def test_convnext_block_preserves_shape(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneckBlock

        block = ConvNextBottleneckBlock(channels=32)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert tuple(y.shape) == (2, 32, 16, 16)

    def test_convnext_bottleneck_preserves_shape(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneck

        bottleneck = ConvNextBottleneck(channels=32, n_blocks=4)
        x = torch.randn(2, 32, 16, 16)
        y = bottleneck(x)
        assert tuple(y.shape) == (2, 32, 16, 16)

    def test_convnext_block_layerscale_default_matches_srunet_convention(self):
        """First ConvNeXt row uses the SRU-Net LayerScale convention 0.1."""
        from ptycho_torch.generators.resnet_components import ConvNextBottleneckBlock

        block = ConvNextBottleneckBlock(channels=16)
        assert block.layerscale.requires_grad is True
        assert math.isclose(float(block.layerscale.detach().item()), 0.1, abs_tol=1e-6)

    def test_convnext_bottleneck_layerscale_default_matches_srunet_convention(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneck

        bottleneck = ConvNextBottleneck(channels=16, n_blocks=3)
        assert bottleneck.layerscale.requires_grad is True
        assert math.isclose(float(bottleneck.layerscale.detach().item()), 0.1, abs_tol=1e-6)

    def test_convnext_bottleneck_shares_layerscale_across_blocks(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneck

        bottleneck = ConvNextBottleneck(channels=16, n_blocks=4)
        ids = {id(block.layerscale) for block in bottleneck.blocks}
        assert len(ids) == 1
        assert id(bottleneck.layerscale) in ids

    def test_convnext_bottleneck_fixed_layerscale_disables_grad(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneck

        bottleneck = ConvNextBottleneck(
            channels=8,
            n_blocks=2,
            layerscale_mode="fixed",
            layerscale_value=0.05,
        )
        assert bottleneck.layerscale.requires_grad is False
        assert math.isclose(float(bottleneck.layerscale.detach().item()), 0.05, abs_tol=1e-6)

    def test_convnext_block_invalid_kernel_size_raises(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneckBlock

        with pytest.raises(ValueError, match="kernel_size"):
            ConvNextBottleneckBlock(channels=8, kernel_size=4)

    def test_convnext_block_invalid_layerscale_raises(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneckBlock

        with pytest.raises(ValueError, match="layerscale_init"):
            ConvNextBottleneckBlock(channels=8, layerscale_init=0.0)

    def test_convnext_bottleneck_invalid_n_blocks_raises(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneck

        with pytest.raises(ValueError, match="n_blocks"):
            ConvNextBottleneck(channels=8, n_blocks=0)

    def test_convnext_block_has_expand_act_project_feedforward_path(self):
        from ptycho_torch.generators.resnet_components import ConvNextBottleneckBlock

        block = ConvNextBottleneckBlock(channels=16)
        assert isinstance(block.depthwise, torch.nn.Conv2d)
        assert block.depthwise.groups == 16
        assert isinstance(block.expand, torch.nn.Conv2d)
        assert isinstance(block.act, torch.nn.GELU)
        assert isinstance(block.project, torch.nn.Conv2d)


class TestHybridResnetConvNextBottleneckGenerator:
    """End-to-end shape and contract tests for the new architecture."""

    def test_hybrid_resnet_convnext_bottleneck_module_forward_preserves_shape(self):
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetConvNextBottleneckGeneratorModule,
        )

        model = HybridResnetConvNextBottleneckGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=32,
            n_blocks=4,
            modes=12,
            C=1,
            hybrid_downsample_steps=2,
            resnet_blocks=6,
        )
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        assert tuple(y.shape) == (2, 128, 128, 1, 2)

    def test_hybrid_resnet_convnext_bottleneck_swaps_only_bottleneck(self):
        """Architecture must keep the lifter/encoder/downsample/decoder/output
        intact and only replace ``self.resnet`` with ConvNextBottleneck."""
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetConvNextBottleneckGeneratorModule,
            HybridResnetGeneratorModule,
        )
        from ptycho_torch.generators.resnet_components import (
            ConvNextBottleneck,
            ResnetBottleneck,
        )

        baseline = HybridResnetGeneratorModule(
            in_channels=1, out_channels=2, hidden_channels=32, n_blocks=4, modes=12, C=1,
            hybrid_downsample_steps=2, resnet_blocks=6,
        )
        convnext = HybridResnetConvNextBottleneckGeneratorModule(
            in_channels=1, out_channels=2, hidden_channels=32, n_blocks=4, modes=12, C=1,
            hybrid_downsample_steps=2, resnet_blocks=6,
        )
        # Bottleneck swapped
        assert isinstance(baseline.resnet, ResnetBottleneck)
        assert isinstance(convnext.resnet, ConvNextBottleneck)
        # Shell preserved (same submodule classes)
        for attr in ("lifter", "encoder_blocks", "downsample_layers", "adapter",
                    "upsample_layers", "output_proj"):
            assert type(getattr(baseline, attr)) is type(getattr(convnext, attr))
        # Bottleneck width matches
        assert baseline.bottleneck_channels == convnext.bottleneck_channels

    def test_hybrid_resnet_convnext_bottleneck_layerscale_default_is_srunet_convention(self):
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetConvNextBottleneckGeneratorModule,
        )

        model = HybridResnetConvNextBottleneckGeneratorModule(
            in_channels=1, out_channels=2, hidden_channels=32, n_blocks=4, modes=12, C=1,
            hybrid_downsample_steps=2, resnet_blocks=6,
        )
        assert math.isclose(float(model.resnet.layerscale.detach().item()), 0.1, abs_tol=1e-6)
        assert model.resnet.layerscale.requires_grad is True

    def test_registry_resolves_hybrid_resnet_convnext_bottleneck(self):
        cfg = TrainingConfig(model=ModelConfig(architecture="hybrid_resnet_convnext_bottleneck"))
        generator = resolve_generator(cfg)
        assert generator.name == "hybrid_resnet_convnext_bottleneck"

    def test_hybrid_resnet_convnext_bottleneck_generator_serializes_overrides(self):
        """ConvNeXt-bottleneck generator must record execution-time bottleneck/encoder/
        ConvNeXt overrides into ``model.hparams['generator_overrides']`` so the saved
        provenance proves the row's explicit settings, not just the architecture id."""
        from ptycho.config.config import PyTorchExecutionConfig
        from ptycho_torch.config_params import (
            DataConfig,
            InferenceConfig as PTInferenceConfig,
            ModelConfig as PTModelConfig,
            TrainingConfig as PTTrainingConfig,
        )
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetConvNextBottleneckGenerator,
        )
        from ptycho_torch.model import PtychoPINN_Lightning

        config = TrainingConfig(
            model=ModelConfig(architecture="hybrid_resnet_convnext_bottleneck", N=64, gridsize=1)
        )
        gen = HybridResnetConvNextBottleneckGenerator(config)
        execution_config = PyTorchExecutionConfig(
            hybrid_resnet_bottleneck_layerscale_mode="fixed",
            hybrid_resnet_bottleneck_layerscale_value=1.0,
        )
        pt_configs = {
            "data_config": DataConfig(N=64, C=1),
            "model_config": PTModelConfig(architecture="hybrid_resnet_convnext_bottleneck"),
            "training_config": PTTrainingConfig(),
            "inference_config": PTInferenceConfig(),
            "execution_config": execution_config,
        }

        model = gen.build_model(pt_configs)

        assert isinstance(model, PtychoPINN_Lightning)
        overrides = model.hparams["generator_overrides"]
        assert overrides["hybrid_resnet_bottleneck_layerscale_mode"] == "fixed"
        assert overrides["hybrid_resnet_bottleneck_layerscale_value"] == 1.0
        assert overrides["hybrid_encoder_fusion_mode"] == "baseline"
        assert overrides["hybrid_encoder_layerscale_init"] == 0.1
        assert overrides["hybrid_encoder_branch_gate_init"] == 0.1
        assert overrides["hybrid_encoder_branch_select"] == "both"
        assert overrides["convnext_bottleneck_layerscale_init"] == 0.1
        assert overrides["convnext_bottleneck_mlp_ratio"] == 4.0
        assert overrides["convnext_bottleneck_kernel_size"] == 7
        # ConvNeXt-specific knobs must not leak into PTModelConfig hparams (same
        # discipline already enforced for the baseline hybrid_resnet path).
        assert "convnext_bottleneck_layerscale_init" not in model.hparams["model_config"]
        assert "convnext_bottleneck_mlp_ratio" not in model.hparams["model_config"]
        assert "convnext_bottleneck_kernel_size" not in model.hparams["model_config"]
        # The fixed-residual flag must be honored end-to-end on the new path too.
        assert model.model.generator.resnet.layerscale.requires_grad is False

    def test_build_generator_module_from_config_rebuilds_hybrid_resnet_convnext_bottleneck(self):
        """The checkpoint-rebuild path must accept the new architecture so
        ``load_from_checkpoint`` can reconstruct the generator without ValueError."""
        from ptycho_torch.config_params import (
            DataConfig as PTDataConfig,
            ModelConfig as PTModelConfig,
        )
        from ptycho_torch.generators.hybrid_resnet import (
            HybridResnetConvNextBottleneckGeneratorModule,
        )
        from ptycho_torch.model import _build_generator_module_from_config

        model_config = PTModelConfig(architecture="hybrid_resnet_convnext_bottleneck")
        data_config = PTDataConfig(N=64, C=1)
        overrides = {
            "hybrid_resnet_bottleneck_layerscale_mode": "fixed",
            "hybrid_resnet_bottleneck_layerscale_value": 1.0,
            "hybrid_encoder_fusion_mode": "baseline",
            "hybrid_encoder_layerscale_init": 0.1,
            "hybrid_encoder_branch_gate_init": 0.1,
            "hybrid_encoder_branch_select": "both",
            "convnext_bottleneck_layerscale_init": 0.1,
            "convnext_bottleneck_mlp_ratio": 4.0,
            "convnext_bottleneck_kernel_size": 7,
        }

        module = _build_generator_module_from_config(
            model_config,
            data_config,
            generator_output="real_imag",
            generator_overrides=overrides,
        )

        assert isinstance(module, HybridResnetConvNextBottleneckGeneratorModule)
        assert module.resnet.layerscale.requires_grad is False
