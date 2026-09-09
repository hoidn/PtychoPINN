"""Contract tests for the matched global-attention CDI baseline."""

import math

import pytest
import torch

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.application_factory import build_ptychopinn_from_configs


def test_vit_match_module_implements_locked_vit5_contract():
    from ptycho_torch.generators.vit import VitGeneratorModule

    model = VitGeneratorModule(
        width=32,
        depth=2,
        heads=4,
        patch_size=4,
        output_mode="amp_phase",
    )

    assert model.patch_size == 4
    assert model.num_heads == 4
    assert model.num_register_tokens == 4
    assert model.absolute_position.shape == (1, 1028, 32)
    assert model.absolute_position.requires_grad
    assert len(model.blocks) == 2
    assert isinstance(model.blocks[0].norm1, torch.nn.RMSNorm)
    assert isinstance(model.blocks[0].attention.q_norm, torch.nn.RMSNorm)
    assert isinstance(model.blocks[0].attention.k_norm, torch.nn.RMSNorm)
    assert model.blocks[0].attention.qkv.bias is None
    assert torch.count_nonzero(model.blocks[0].attention.rope_sin) > 0
    assert torch.all(model.blocks[0].scale_attention != 0)
    assert torch.all(model.blocks[0].scale_mlp != 0)
    assert isinstance(model.pixel_shuffle, torch.nn.PixelShuffle)


def test_vit_match_amp_phase_forward_preserves_cdi_shape_and_bounds():
    from ptycho_torch.generators.vit import VitGeneratorModule

    model = VitGeneratorModule(
        width=32,
        depth=1,
        heads=4,
        patch_size=4,
        output_mode="amp_phase",
    ).eval()

    with torch.no_grad():
        amplitude, phase = model(torch.rand(1, 1, 128, 128))

    assert amplitude.shape == (1, 1, 128, 128)
    assert phase.shape == amplitude.shape
    assert torch.isfinite(amplitude).all()
    assert torch.isfinite(phase).all()
    assert torch.all((0 <= amplitude) & (amplitude <= 1))
    assert torch.all((-math.pi <= phase) & (phase <= math.pi))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"C": 2}, "C=1"),
        ({"width": 30}, "divisible"),
        ({"depth": 0}, "positive"),
    ],
)
def test_vit_match_rejects_out_of_contract_construction(kwargs, message):
    from ptycho_torch.generators.vit import VitGeneratorModule

    with pytest.raises(ValueError, match=message):
        VitGeneratorModule(**kwargs)


def test_vit_structural_config_defaults_are_explicit():
    config = ModelConfig()

    assert config.vit_patch_size == 4
    assert config.vit_width == 256
    assert config.vit_depth == 12
    assert config.vit_heads == 4


def test_vit_match_factory_builds_canonical_lightning_application():
    pt_configs = {
        "data_config": DataConfig(N=128, gridsize=1),
        "model_config": ModelConfig(
            architecture="vit",
            vit_patch_size=4,
            vit_width=32,
            vit_depth=1,
            vit_heads=4,
            generator_output_mode="amp_phase",
        ),
        "training_config": TrainingConfig(),
        "inference_config": InferenceConfig(),
    }

    application = build_ptychopinn_from_configs(pt_configs)

    assert type(application.model.generator).__name__ == "VitGeneratorModule"
