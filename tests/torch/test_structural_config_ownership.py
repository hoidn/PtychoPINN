"""Main-side contracts for Torch structural configuration ownership."""

from pathlib import Path

import numpy as np
import pytest
import warnings

from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.execution_request import ExecutionRequest


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


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


def test_execution_topology_explicit_default_is_consumed_but_omitted_is_not(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    explicit = ExecutionRequest(
        values={"accelerator": "cpu", "spectral_bottleneck_modes": 12},
        explicit_fields=frozenset({"spectral_bottleneck_modes"}),
    )
    omitted = ExecutionRequest(
        values={"accelerator": "cpu", "spectral_bottleneck_modes": 12},
        explicit_fields=frozenset(),
    )

    with pytest.warns(DeprecationWarning, match="spectral_bottleneck_modes"):
        consumed = create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4, "spectral_bottleneck_modes": 12},
            explicit,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        untouched = create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4, "spectral_bottleneck_modes": 8},
            omitted,
        )

    assert consumed.pt_model_config.spectral_bottleneck_modes == 12
    assert untouched.pt_model_config.spectral_bottleneck_modes == 8
    assert [
        item
        for item in caught
        if item.category is DeprecationWarning
        and "topology fields" in str(item.message)
    ] == []


def test_execution_topology_maps_representable_ffno_aliases(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    request = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "ffno_encoder_blocks": 7,
            "ffno_encoder_modes": 9,
        },
        explicit_fields=frozenset(
            {"ffno_encoder_blocks", "ffno_encoder_modes"}
        ),
    )

    with pytest.warns(DeprecationWarning):
        payload = create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4},
            request,
        )

    assert payload.pt_model_config.fno_blocks == 7
    assert payload.pt_model_config.fno_modes == 9
    assert payload.overrides_applied["topology_compatibility"] == {
        "ffno_encoder_blocks": "fno_blocks",
        "ffno_encoder_modes": "fno_modes",
    }


@pytest.mark.parametrize(
    "field_name",
    [
        "ffno_encoder_share_weights",
        "ffno_encoder_gate_init",
        "ffno_encoder_norm",
        "ffno_encoder_mlp_ratio",
    ],
)
def test_execution_topology_rejects_unowned_ffno_alias(
    training_npz: Path,
    tmp_path: Path,
    field_name: str,
) -> None:
    request = ExecutionRequest(
        values={"accelerator": "cpu", field_name: 1},
        explicit_fields=frozenset({field_name}),
    )

    with pytest.raises(ValueError, match=f"{field_name}.*no ModelConfig owner"):
        create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4},
            request,
        )


def test_execution_topology_resolution_is_pure_and_warning_is_deferred(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    overrides = {"spectral_bottleneck_modes": 12}
    request_values = {
        "accelerator": "cpu",
        "spectral_bottleneck_modes": 12,
    }
    request = ExecutionRequest(
        values=request_values,
        explicit_fields=frozenset({"spectral_bottleneck_modes"}),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="n_groups is required"):
            create_training_payload(
                training_npz,
                tmp_path,
                overrides,
                request,
            )

    assert overrides == {"spectral_bottleneck_modes": 12}
    assert request_values == {
        "accelerator": "cpu",
        "spectral_bottleneck_modes": 12,
    }
    assert caught == []


def test_execution_topology_resolved_carrier_does_not_fabricate_aliases(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = create_training_payload(
        training_npz,
        tmp_path,
        {"n_groups": 4},
        ExecutionRequest(
            values={"accelerator": "cpu"},
            explicit_fields=frozenset({"accelerator"}),
        ),
    )

    assert payload.execution_config._explicit_structural_aliases == frozenset()
    reused = create_training_payload(
        training_npz,
        tmp_path,
        {"n_groups": 4},
        payload.execution_config,
    )
    assert reused.pt_model_config.fno_blocks == 4


def test_execution_topology_resolved_carrier_retains_actual_alias(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    with pytest.warns(DeprecationWarning):
        payload = create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4},
            ExecutionRequest(
                values={
                    "accelerator": "cpu",
                    "spectral_bottleneck_modes": 10,
                },
                explicit_fields=frozenset({"spectral_bottleneck_modes"}),
            ),
        )

    assert payload.execution_config._explicit_structural_aliases == frozenset(
        {"spectral_bottleneck_modes"}
    )
    with pytest.warns(DeprecationWarning):
        reused = create_training_payload(
            training_npz,
            tmp_path,
            {"n_groups": 4},
            payload.execution_config,
        )
    assert reused.pt_model_config.spectral_bottleneck_modes == 10
