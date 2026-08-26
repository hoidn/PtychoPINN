"""Focused contracts for unresolved runtime requests and environment resolution."""

from __future__ import annotations

from dataclasses import fields
import warnings

import pytest


def test_execution_request_defensively_copies_values_and_provenance() -> None:
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


def test_execution_request_accepts_only_runtime_carrier_fields() -> None:
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    runtime_fields = {item.name for item in fields(PyTorchExecutionConfig)}
    request = ExecutionRequest(
        values={"accelerator": "auto", "devices": "auto"},
        explicit_fields=frozenset({"accelerator", "devices"}),
    )

    assert set(request.values) <= runtime_fields
    for retired_field in (
        "learning_rate",
        "scheduler",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accum_steps",
        "hybrid_skip_style",
        "ffno_encoder_modes",
    ):
        with pytest.raises(
            ValueError,
            match="unknown execution request field",
        ):
            ExecutionRequest(
                values={retired_field: object()},
                explicit_fields=frozenset({retired_field}),
            )


def test_execution_request_freezes_reconstruction_indices_through_resolution() -> None:
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
    resolved.config.recon_log_fixed_indices.append(9)
    assert normalized.values["recon_log_fixed_indices"] == (1, 3)


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
) -> None:
    from ptycho_torch.execution_request import ExecutionRequest

    with pytest.raises(ValueError, match=message):
        ExecutionRequest(values=values, explicit_fields=explicit_fields)


def test_execution_request_construction_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import ExecutionRequest

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("request construction must remain pure")

    monkeypatch.setattr(PyTorchExecutionConfig, "__init__", fail_if_called)
    monkeypatch.setattr(torch.cuda, "is_available", fail_if_called)
    monkeypatch.setattr(torch.cuda, "device_count", fail_if_called)

    request = ExecutionRequest(
        values={"accelerator": "auto"},
        explicit_fields=frozenset({"accelerator"}),
    )

    assert request.as_dict() == {"accelerator": "auto"}


def test_request_structure_fails_before_capability_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "num_workers": 0,
            "persistent_workers": True,
        },
        explicit_fields=frozenset(
            {"accelerator", "num_workers", "persistent_workers"}
        ),
    )

    with pytest.raises(ValueError, match="persistent_workers"):
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
) -> None:
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


def test_cpu_environment_resolution_is_injected_immutable_and_deferred() -> None:
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


def test_cuda_environment_resolution_uses_injected_device_count() -> None:
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


def test_explicit_cpu_devices_auto_needs_no_capability_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_default_request_resolves_gpu_first_but_default_carrier_is_cpu() -> None:
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import (
        ExecutionCapabilities,
        resolve_runtime_execution_request,
    )

    assert PyTorchExecutionConfig().accelerator == "cpu"
    result = resolve_runtime_execution_request(
        None,
        mode="training",
        execution_capabilities=ExecutionCapabilities(
            cuda_available=True,
            cuda_device_count=2,
        ),
    )

    assert result.environment.requested["accelerator"] == "auto"
    assert result.config.accelerator == "cuda"


def test_bare_resolved_carrier_is_not_an_unresolved_request() -> None:
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.execution_request import (
        normalize_execution_input,
        resolve_runtime_execution_request,
    )

    carrier = PyTorchExecutionConfig()

    with pytest.raises(TypeError, match="ExecutionRequest"):
        normalize_execution_input(carrier, mode="training")
    with pytest.raises(TypeError, match="ExecutionRequest"):
        resolve_runtime_execution_request(carrier, mode="training")
