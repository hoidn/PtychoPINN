"""Public contracts for singular Torch structural configuration ownership."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.execution_request import ExecutionRequest


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("fno_modes", 10),
        ("fno_width", 48),
        ("fno_blocks", 6),
        ("fno_cnn_blocks", 3),
        ("fno_input_transform", "log1p"),
        ("max_hidden_channels", 64),
        ("spectral_bottleneck_blocks", 8),
        ("spectral_bottleneck_modes", 10),
        ("spectral_bottleneck_share_weights", False),
        ("spectral_bottleneck_gate_init", 0.2),
        ("spectral_bottleneck_gate_mode", "per_block"),
    ],
)
def test_each_public_topology_field_enters_through_canonical_model_patch(
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
        match=r"unknown training input field\(s\).*spectral_bottleneck_modse",
    ):
        create_training_payload(
            train_data_file=training_npz,
            output_dir=tmp_path,
            overrides={
                "n_groups": 4,
                "spectral_bottleneck_modse": 10,
            },
        )
