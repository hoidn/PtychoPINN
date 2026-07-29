"""Contracts for singular Torch structural configuration ownership."""

from __future__ import annotations

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
from ptycho_torch.execution_request import ExecutionRequest


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


@pytest.mark.parametrize(
    ("field_name", "value"),
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
def test_each_topology_field_enters_through_canonical_model_patch(
    training_npz: Path,
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    payload = create_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path,
        overrides={"n_groups": 4, "gridsize": 1, field_name: value},
        execution_config=ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )

    assert getattr(payload.pt_model_config, field_name) == value
    assert field_name not in payload.overrides_applied.get(
        "topology_compatibility",
        {},
    )


def test_training_factory_rejects_unknown_structural_override(
    training_npz: Path,
    tmp_path: Path,
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


def test_generator_topology_reads_model_config_only() -> None:
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
        "execution_config": PyTorchExecutionConfig(),
    }

    model = generator.build_model(configs)

    assert model.model.generator.resnet.layerscale.requires_grad is False
    assert model.hparams["model_config"][
        "hybrid_resnet_bottleneck_layerscale_mode"
    ] == "fixed"
    assert model.hparams["model_spec"]["model_config"][
        "hybrid_resnet_bottleneck_layerscale_mode"
    ] == "fixed"
