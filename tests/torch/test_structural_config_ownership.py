"""Main-side contracts for Torch structural configuration ownership."""

from pathlib import Path

import numpy as np
import pytest

from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_factory import create_training_payload


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


def test_deprecated_execution_topology_alias_populates_structural_model(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    execution = PyTorchExecutionConfig(ffno_encoder_blocks=12)

    with pytest.warns(DeprecationWarning, match="structural ModelConfig"):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "architecture": "ffno",
            },
            execution_config=execution,
        )

    assert payload.pt_model_config.ffno_encoder_blocks == 12


def test_equal_old_and_new_structural_inputs_are_accepted(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    execution = PyTorchExecutionConfig(spectral_bottleneck_modes=10)

    with pytest.warns(DeprecationWarning):
        payload = create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "spectral_bottleneck_modes": 10,
            },
            execution_config=execution,
        )

    assert payload.pt_model_config.spectral_bottleneck_modes == 10


def test_conflicting_old_and_new_structural_inputs_fail_closed(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    execution = PyTorchExecutionConfig(spectral_bottleneck_modes=10)

    with pytest.raises(ValueError, match="spectral_bottleneck_modes.*conflict"):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "gridsize": 1,
                "spectral_bottleneck_modes": 8,
            },
            execution_config=execution,
        )


def test_default_execution_alias_does_not_override_explicit_structural_value(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = create_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path,
        overrides={
            "n_groups": 4,
            "gridsize": 1,
            "spectral_bottleneck_modes": 8,
        },
        execution_config=PyTorchExecutionConfig(),
    )

    assert payload.pt_model_config.spectral_bottleneck_modes == 8


def test_training_factory_rejects_unknown_override(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unknown training override.*spectral_bottleneck_modse",
    ):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "spectral_bottleneck_modse": 10,
            },
        )
