"""Slice 3A contracts for Torch structural configuration ownership."""

from pathlib import Path

import numpy as np
import pytest

from ptycho.config.config import ModelConfig, PyTorchExecutionConfig, TrainingConfig
from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig as TorchModelConfig,
    TrainingConfig as TorchTrainingConfig,
)


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


def test_deprecated_execution_topology_alias_populates_structural_model(
    training_npz: Path, tmp_path: Path
) -> None:
    execution = PyTorchExecutionConfig(
        hybrid_resnet_bottleneck_layerscale_mode="fixed",
        hybrid_resnet_bottleneck_layerscale_value=1.0,
        hybrid_encoder_fusion_mode="branch_gated",
    )

    with pytest.warns(DeprecationWarning, match="structural ModelConfig"):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "architecture": "hybrid_resnet",
            },
            execution_config=execution,
        )

    assert payload.pt_model_config.hybrid_resnet_bottleneck_layerscale_mode == "fixed"
    assert payload.pt_model_config.hybrid_resnet_bottleneck_layerscale_value == 1.0
    assert payload.pt_model_config.hybrid_encoder_fusion_mode == "branch_gated"


def test_equal_old_and_new_structural_inputs_are_accepted(
    training_npz: Path, tmp_path: Path
) -> None:
    execution = PyTorchExecutionConfig(hybrid_skip_style="concat")

    with pytest.warns(DeprecationWarning):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "hybrid_skip_style": "concat",
            },
            execution_config=execution,
        )

    assert payload.pt_model_config.hybrid_skip_style == "concat"


def test_conflicting_old_and_new_structural_inputs_fail_closed(
    training_npz: Path, tmp_path: Path
) -> None:
    execution = PyTorchExecutionConfig(hybrid_skip_style="concat")

    with pytest.raises(ValueError, match="hybrid_skip_style.*conflict"):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "hybrid_skip_style": "gated_add",
            },
            execution_config=execution,
        )


def test_explicit_old_default_still_conflicts_with_new_structural_value(
    training_npz: Path, tmp_path: Path
) -> None:
    execution = PyTorchExecutionConfig(hybrid_skip_style="add")

    with pytest.raises(ValueError, match="hybrid_skip_style.*conflict"):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "hybrid_skip_style": "gated_add",
            },
            execution_config=execution,
        )


@pytest.mark.parametrize(
    ("field_name", "alias_value"),
    [
        ("hybrid_skip_connections", True),
        ("hybrid_downsample_steps", 1),
        ("hybrid_downsample_op", "avgpool_conv"),
        ("hybrid_encoder_conv_hidden_scale", 0.5),
        ("hybrid_encoder_spectral_hidden_scale", 2.0),
        ("hybrid_encoder_conv_hidden_channels", 16),
        ("hybrid_encoder_spectral_hidden_channels", 16),
        ("hybrid_resnet_blocks", 8),
        ("hybrid_skip_style", "concat"),
        ("hybrid_encoder_fusion_mode", "layerscale"),
        ("hybrid_encoder_layerscale_init", 0.2),
        ("hybrid_encoder_branch_gate_init", 0.2),
        ("hybrid_encoder_branch_select", "conv_only"),
        ("ffno_encoder_blocks", 12),
        ("ffno_encoder_modes", 10),
        ("ffno_encoder_share_weights", False),
        ("ffno_encoder_gate_init", 0.2),
        ("ffno_encoder_norm", "layer"),
        ("ffno_encoder_mlp_ratio", 3.0),
        ("spectral_bottleneck_blocks", 8),
        ("spectral_bottleneck_modes", 10),
        ("spectral_bottleneck_share_weights", False),
        ("spectral_bottleneck_gate_init", 0.2),
        ("spectral_bottleneck_gate_mode", "per_block"),
    ],
)
def test_each_independent_execution_topology_alias_maps_to_model_owner(
    training_npz: Path,
    tmp_path: Path,
    field_name: str,
    alias_value: object,
) -> None:
    execution = PyTorchExecutionConfig(**{field_name: alias_value})

    with pytest.warns(DeprecationWarning):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={"n_groups": 4, "gridsize": 1},
            execution_config=execution,
        )

    assert getattr(payload.pt_model_config, field_name) == alias_value


def test_default_execution_alias_does_not_override_explicit_structural_value(
    training_npz: Path, tmp_path: Path
) -> None:
    payload = create_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path,
        overrides={
            "n_groups": 4,
            "gridsize": 1,
            "hybrid_skip_style": "gated_add",
        },
        execution_config=PyTorchExecutionConfig(),
    )

    assert payload.pt_model_config.hybrid_skip_style == "gated_add"


def test_training_factory_rejects_unknown_override(
    training_npz: Path, tmp_path: Path
) -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown training input field\(s\).*hybrid_skip_stlye",
    ):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={"n_groups": 4, "hybrid_skip_stlye": "concat"},
        )


def test_generator_topology_reads_model_config_not_execution_side_channel() -> None:
    from ptycho_torch.generators.hybrid_resnet import HybridResnetGenerator

    generator = HybridResnetGenerator(
        TrainingConfig(
            model=ModelConfig(architecture="hybrid_resnet", N=64, gridsize=1)
        )
    )
    structural = TorchModelConfig(
        architecture="hybrid_resnet",
        C_model=1,
        C_forward=1,
        hybrid_resnet_bottleneck_layerscale_mode="fixed",
        hybrid_resnet_bottleneck_layerscale_value=1.0,
    )
    configs = {
        "data_config": DataConfig(N=64, C=1),
        "model_config": structural,
        "training_config": TorchTrainingConfig(),
        "inference_config": InferenceConfig(),
        "execution_config": PyTorchExecutionConfig(
            hybrid_resnet_bottleneck_layerscale_mode="learned",
            hybrid_resnet_bottleneck_layerscale_value=None,
        ),
    }

    model = generator.build_model(configs)

    assert model.model.generator.resnet.layerscale.requires_grad is False
    assert model.hparams["model_config"][
        "hybrid_resnet_bottleneck_layerscale_mode"
    ] == "fixed"
    assert model.hparams["model_spec"]["model_config"][
        "hybrid_resnet_bottleneck_layerscale_mode"
    ] == "fixed"
