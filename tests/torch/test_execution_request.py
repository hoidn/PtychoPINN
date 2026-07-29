"""Focused contracts for pure execution request and environment resolution."""

from __future__ import annotations

from dataclasses import fields
import warnings

import pytest


EXPECTED_TOPOLOGY_ALIASES = frozenset(
    {
        "hybrid_skip_connections",
        "hybrid_downsample_steps",
        "hybrid_downsample_op",
        "hybrid_encoder_conv_hidden_scale",
        "hybrid_encoder_spectral_hidden_scale",
        "hybrid_encoder_conv_hidden_channels",
        "hybrid_encoder_spectral_hidden_channels",
        "hybrid_resnet_blocks",
        "hybrid_skip_style",
        "hybrid_resnet_bottleneck_layerscale_mode",
        "hybrid_resnet_bottleneck_layerscale_value",
        "hybrid_encoder_fusion_mode",
        "hybrid_encoder_layerscale_init",
        "hybrid_encoder_branch_gate_init",
        "hybrid_encoder_branch_select",
        "ffno_encoder_blocks",
        "ffno_encoder_modes",
        "ffno_encoder_share_weights",
        "ffno_encoder_gate_init",
        "ffno_encoder_norm",
        "ffno_encoder_mlp_ratio",
        "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes",
        "spectral_bottleneck_share_weights",
        "spectral_bottleneck_gate_init",
        "spectral_bottleneck_gate_mode",
    }
)

EXPECTED_OPTIMIZER_ALIASES = frozenset(
    {
        "learning_rate",
        "scheduler",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accum_steps",
    }
)


def test_execution_request_defensively_copies_values_and_provenance():
    from ptycho_torch.execution_request import ExecutionRequest, ResolutionNotice

    source = {"accelerator": "auto", "devices": "auto"}
    explicit_fields = {"accelerator"}
    notices = [ResolutionNotice(DeprecationWarning, "deprecated input")]
    request = ExecutionRequest(
        values=source,
        explicit_fields=explicit_fields,
        notices=notices,
    )

    source["accelerator"] = "cpu"
    explicit_fields.add("devices")
    notices.append(ResolutionNotice(UserWarning, "late mutation"))

    assert request.as_dict() == {
        "accelerator": "auto",
        "devices": "auto",
    }
    assert request.explicit_fields == frozenset({"accelerator"})
    assert len(request.notices) == 1
    with pytest.raises(TypeError):
        request.values["accelerator"] = "cuda"


def test_execution_request_freezes_reconstruction_indices_through_resolution():
    from ptycho_torch.execution_request import (
        ExecutionRequest,
        normalize_execution_input,
        resolve_runtime_execution_request,
    )

    source_indices = [1, 3]
    request = ExecutionRequest(
        values={
            "accelerator": "cpu",
            "recon_log_fixed_indices": source_indices,
        },
        explicit_fields=frozenset(
            {"accelerator", "recon_log_fixed_indices"}
        ),
    )

    source_indices.append(5)
    returned = request.as_dict()
    returned["recon_log_fixed_indices"].append(7)
    normalized = normalize_execution_input(request, mode="training")

    assert request.values["recon_log_fixed_indices"] == (1, 3)
    assert returned["recon_log_fixed_indices"] == [1, 3, 7]
    assert normalized is not None
    assert normalized.values["recon_log_fixed_indices"] == (1, 3)

    resolved = resolve_runtime_execution_request(
        normalized,
        mode="training",
    )

    assert resolved.config.recon_log_fixed_indices == [1, 3]
    assert isinstance(resolved.config.recon_log_fixed_indices, list)
    resolved.config.recon_log_fixed_indices.append(9)
    assert normalized.values["recon_log_fixed_indices"] == (1, 3)


def test_bare_execution_config_normalization_snapshots_reconstruction_indices():
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import (
        normalize_execution_input,
        resolve_runtime_execution_request,
    )

    config = PyTorchExecutionConfig(
        accelerator="cpu",
        recon_log_fixed_indices=[2, 4],
    )
    normalized = normalize_execution_input(config, mode="training")
    assert normalized is not None

    config.recon_log_fixed_indices.append(6)

    assert normalized.values["recon_log_fixed_indices"] == (2, 4)
    resolved = resolve_runtime_execution_request(
        normalized,
        mode="training",
    )
    assert resolved.config.recon_log_fixed_indices == [2, 4]
    assert isinstance(resolved.config.recon_log_fixed_indices, list)


def test_environment_resolution_snapshots_mutable_execution_values():
    from ptycho_torch.execution_request import EnvironmentResolution

    requested_indices = [1, 2]
    resolved_indices = [3, 4]
    resolution = EnvironmentResolution(
        requested={"recon_log_fixed_indices": requested_indices},
        resolved={"recon_log_fixed_indices": resolved_indices},
        capabilities=None,
    )

    requested_indices.append(5)
    resolved_indices.append(6)

    assert resolution.requested["recon_log_fixed_indices"] == (1, 2)
    assert resolution.resolved["recon_log_fixed_indices"] == (3, 4)


@pytest.mark.parametrize(
    ("values", "explicit_fields", "message"),
    [
        ({"bogus": "value"}, frozenset(), "unknown execution request field"),
        (
            {"accelerator": "cpu"},
            frozenset({"bogus"}),
            "unknown explicit execution field",
        ),
        (
            {"accelerator": "cpu"},
            frozenset({"devices"}),
            "explicit execution field.*absent",
        ),
    ],
)
def test_execution_request_rejects_invalid_field_provenance(
    values,
    explicit_fields,
    message,
):
    from ptycho_torch.execution_request import ExecutionRequest

    with pytest.raises(ValueError, match=message):
        ExecutionRequest(values=values, explicit_fields=explicit_fields)


def test_execution_request_construction_is_pure(monkeypatch):
    import torch

    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    def fail_if_called(*args, **kwargs):
        raise AssertionError("request construction must remain pure")

    monkeypatch.setattr(PyTorchExecutionConfig, "__init__", fail_if_called)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)

    request = ExecutionRequest(
        values={"accelerator": "auto"},
        explicit_fields=frozenset({"accelerator"}),
    )

    assert request.as_dict() == {"accelerator": "auto"}


def test_execution_request_compatibility_field_sets_match_internal_carrier():
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import (
        OPTIMIZER_EXECUTION_COMPAT_FIELDS,
        TOPOLOGY_EXECUTION_COMPAT_FIELDS,
    )

    assert len(fields(PyTorchExecutionConfig)) == 55
    assert TOPOLOGY_EXECUTION_COMPAT_FIELDS == EXPECTED_TOPOLOGY_ALIASES
    assert OPTIMIZER_EXECUTION_COMPAT_FIELDS == EXPECTED_OPTIMIZER_ALIASES


def test_request_structure_fails_before_capability_observation(monkeypatch):
    import ptycho_torch.execution_request as execution_request

    observed = False

    def fail_if_observed():
        nonlocal observed
        observed = True
        raise AssertionError("capabilities observed before structure validated")

    monkeypatch.setattr(
        execution_request,
        "observe_execution_capabilities",
        fail_if_observed,
    )
    request = execution_request.ExecutionRequest(
        values={
            "accelerator": "auto",
            "hybrid_downsample_steps": 0,
        },
        explicit_fields=frozenset(
            {"accelerator", "hybrid_downsample_steps"}
        ),
    )

    with pytest.raises(ValueError, match="hybrid_downsample_steps"):
        execution_request.resolve_runtime_execution_request(
            request,
            mode="training",
        )

    assert observed is False


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"num_workers": 0, "persistent_workers": True},
            "persistent_workers",
        ),
        ({"logger_backend": "none"}, "logger_backend"),
    ],
)
def test_request_resolution_enforces_selected_runtime_contract(
    values,
    message,
):
    from ptycho_torch.execution_request import (
        ExecutionRequest,
        resolve_runtime_execution_request,
    )

    request_values = {"accelerator": "cpu", **values}
    with pytest.raises(ValueError, match=message):
        resolve_runtime_execution_request(
            ExecutionRequest(
                values=request_values,
                explicit_fields=frozenset(values),
            ),
            mode="training",
        )


@pytest.mark.parametrize(
    "logger_backend",
    ["csv", "tensorboard", "mlflow", None],
)
def test_request_resolution_accepts_canonical_logger_backends(logger_backend):
    from ptycho_torch.execution_request import (
        ExecutionRequest,
        resolve_runtime_execution_request,
    )

    result = resolve_runtime_execution_request(
        ExecutionRequest(
            values={
                "accelerator": "cpu",
                "logger_backend": logger_backend,
            },
            explicit_fields=frozenset({"logger_backend"}),
        ),
        mode="training",
    )

    assert result.config.logger_backend == logger_backend


def test_cpu_environment_resolution_is_injected_immutable_and_deferred():
    from ptycho_torch.execution_request import (
        ExecutionCapabilities,
        ExecutionRequest,
        resolve_runtime_execution_request,
    )

    request = ExecutionRequest(
        values={
            "accelerator": "auto",
            "devices": "auto",
            "pin_memory": True,
            "precision": "16-mixed",
        },
        explicit_fields=frozenset(
            {"accelerator", "devices", "pin_memory", "precision"}
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = resolve_runtime_execution_request(
            request,
            mode="training",
            execution_capabilities=ExecutionCapabilities(
                cuda_available=False,
                cuda_device_count=0,
            ),
        )

    assert caught == []
    assert dict(result.environment.requested) == {
        "accelerator": "auto",
        "devices": "auto",
        "pin_memory": True,
        "precision": "16-mixed",
    }
    assert dict(result.environment.resolved) == {
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "precision": "bf16-mixed",
    }
    assert result.config.accelerator == "cpu"
    assert result.config.devices == 1
    assert result.config.pin_memory is False
    assert result.config.precision == "bf16-mixed"
    assert [notice.category for notice in result.notices] == [
        UserWarning,
        UserWarning,
    ]
    with pytest.raises(TypeError):
        result.environment.resolved["devices"] = 2


def test_cuda_environment_resolution_uses_injected_device_count():
    from ptycho_torch.execution_request import (
        ExecutionCapabilities,
        ExecutionRequest,
        resolve_runtime_execution_request,
    )

    result = resolve_runtime_execution_request(
        ExecutionRequest(
            values={
                "accelerator": "auto",
                "devices": "auto",
                "pin_memory": True,
                "precision": "16-mixed",
            },
            explicit_fields=frozenset(),
        ),
        mode="training",
        execution_capabilities=ExecutionCapabilities(
            cuda_available=True,
            cuda_device_count=3,
        ),
    )

    assert dict(result.environment.resolved) == {
        "accelerator": "cuda",
        "devices": 3,
        "pin_memory": True,
        "precision": "16-mixed",
    }
    assert result.notices == ()


def test_inference_rejects_training_fields_before_capability_observation(
    monkeypatch,
):
    import ptycho_torch.execution_request as execution_request

    def fail_if_observed():
        raise AssertionError("capabilities observed before phase validation")

    monkeypatch.setattr(
        execution_request,
        "observe_execution_capabilities",
        fail_if_observed,
    )
    request = execution_request.ExecutionRequest(
        values={
            "accelerator": "auto",
            "hybrid_skip_connections": True,
        },
        explicit_fields=frozenset({"hybrid_skip_connections"}),
    )

    with pytest.raises(
        ValueError,
        match="inference execution request.*hybrid_skip_connections",
    ):
        execution_request.resolve_runtime_execution_request(
            request,
            mode="inference",
        )


def test_bare_execution_config_preserves_compatibility_provenance():
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import (
        normalize_execution_input,
        resolve_runtime_execution_request,
    )

    config = PyTorchExecutionConfig(
        accelerator="cpu",
        hybrid_skip_connections=True,
        ffno_encoder_modes=8,
        spectral_bottleneck_modes=10,
    )
    normalized = normalize_execution_input(config, mode="training")

    assert normalized is not None
    assert normalized.legacy_config is config
    assert normalized.accelerator_already_resolved is True
    assert normalized.explicit_fields == (
        EXPECTED_OPTIMIZER_ALIASES
        | {
            "hybrid_skip_connections",
            "ffno_encoder_modes",
            "spectral_bottleneck_modes",
        }
    )
    assert len(normalized.values) == 55

    result = resolve_runtime_execution_request(
        normalized,
        mode="training",
    )

    assert result.config._explicit_structural_aliases == frozenset(
        {
            "hybrid_skip_connections",
            "ffno_encoder_modes",
            "spectral_bottleneck_modes",
        }
    )


def test_explicit_cpu_devices_auto_needs_no_capability_observation(monkeypatch):
    import ptycho_torch.execution_request as execution_request

    def fail_if_observed():
        raise AssertionError("explicit CPU resolution must stay pure")

    monkeypatch.setattr(
        execution_request,
        "observe_execution_capabilities",
        fail_if_observed,
    )
    result = execution_request.resolve_runtime_execution_request(
        execution_request.ExecutionRequest(
            values={"accelerator": "cpu", "devices": "auto"},
            explicit_fields=frozenset({"accelerator", "devices"}),
        ),
        mode="training",
    )

    assert result.config.accelerator == "cpu"
    assert result.config.devices == 1
