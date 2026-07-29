"""Primitive execution-request and environment-resolution records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from types import MappingProxyType
from typing import Any

from ptycho.config.config import (
    PyTorchExecutionConfig,
    _validate_execution_post_environment_values,
    _validate_execution_pre_environment_values,
)


OPTIMIZER_EXECUTION_COMPAT_FIELDS = frozenset(
    {
        "learning_rate",
        "scheduler",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accum_steps",
    }
)
TOPOLOGY_EXECUTION_COMPAT_FIELDS = frozenset(
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
ENVIRONMENT_EXECUTION_FIELDS = (
    "accelerator",
    "devices",
    "pin_memory",
    "precision",
)


def _freeze_execution_values(
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Copy execution values into an immutable internal snapshot."""

    copied = dict(values)
    fixed_indices = copied.get("recon_log_fixed_indices")
    if isinstance(fixed_indices, (list, tuple)):
        copied["recon_log_fixed_indices"] = tuple(fixed_indices)
    return MappingProxyType(copied)


def _thaw_execution_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Copy internal values into compatibility-dataclass representations."""

    copied = dict(values)
    fixed_indices = copied.get("recon_log_fixed_indices")
    if isinstance(fixed_indices, (list, tuple)):
        copied["recon_log_fixed_indices"] = list(fixed_indices)
    return copied


@dataclass(frozen=True)
class ExecutionCapabilities:
    """Hardware capabilities observed after pure request resolution."""

    cuda_available: bool
    cuda_device_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.cuda_available, bool):
            raise ValueError("cuda_available must be a bool")
        if (
            isinstance(self.cuda_device_count, bool)
            or not isinstance(self.cuda_device_count, int)
            or self.cuda_device_count < 0
        ):
            raise ValueError("cuda_device_count must be a non-negative integer")


@dataclass(frozen=True)
class ResolutionNotice:
    """A deferred warning produced while resolving an execution request."""

    category: type[Warning]
    message: str


@dataclass(frozen=True)
class EnvironmentResolution:
    """Requested and environment-resolved primitive execution values."""

    requested: Mapping[str, Any]
    resolved: Mapping[str, Any]
    capabilities: ExecutionCapabilities | None
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested",
            _freeze_execution_values(self.requested),
        )
        object.__setattr__(
            self,
            "resolved",
            _freeze_execution_values(self.resolved),
        )
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class ExecutionRequest:
    """Canonical primitive execution values with explicit-input provenance."""

    values: Mapping[str, Any]
    explicit_fields: frozenset[str]
    notices: tuple[ResolutionNotice, ...] = ()

    def __post_init__(self) -> None:
        known_fields = {item.name for item in fields(PyTorchExecutionConfig)}
        copied_values = dict(self.values)
        copied_explicit_fields = frozenset(self.explicit_fields)
        copied_notices = tuple(self.notices)

        unknown_values = set(copied_values) - known_fields
        if unknown_values:
            raise ValueError(
                "unknown execution request field(s): "
                + ", ".join(sorted(unknown_values))
            )

        unknown_explicit = copied_explicit_fields - known_fields
        if unknown_explicit:
            raise ValueError(
                "unknown explicit execution field(s): "
                + ", ".join(sorted(unknown_explicit))
            )

        absent_explicit = copied_explicit_fields - set(copied_values)
        if absent_explicit:
            raise ValueError(
                "explicit execution field(s) absent from request values: "
                + ", ".join(sorted(absent_explicit))
            )

        object.__setattr__(
            self,
            "values",
            _freeze_execution_values(copied_values),
        )
        object.__setattr__(self, "explicit_fields", copied_explicit_fields)
        object.__setattr__(self, "notices", copied_notices)

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the canonical primitive values."""

        return _thaw_execution_values(self.values)


@dataclass(frozen=True)
class NormalizedExecutionInput:
    """Internal complete primitive candidate used by factory resolution."""

    values: Mapping[str, Any]
    explicit_fields: frozenset[str]
    notices: tuple[ResolutionNotice, ...]
    legacy_config: PyTorchExecutionConfig | None = None
    accelerator_already_resolved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _freeze_execution_values(self.values),
        )
        object.__setattr__(
            self,
            "explicit_fields",
            frozenset(self.explicit_fields),
        )
        object.__setattr__(self, "notices", tuple(self.notices))


@dataclass(frozen=True)
class ResolvedRuntimeExecution:
    """Resolved compatibility config plus deterministic runtime audit."""

    config: PyTorchExecutionConfig
    environment: EnvironmentResolution
    explicit_fields: frozenset[str]
    notices: tuple[ResolutionNotice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "explicit_fields",
            frozenset(self.explicit_fields),
        )
        object.__setattr__(self, "notices", tuple(self.notices))

    def audit_dict(self) -> dict[str, Any]:
        """Return primitive requested/resolved runtime audit values."""

        capabilities = self.environment.capabilities
        return {
            "explicit_fields": sorted(self.explicit_fields),
            "requested": dict(self.environment.requested),
            "resolved": dict(self.environment.resolved),
            "capabilities": (
                {
                    "cuda_available": capabilities.cuda_available,
                    "cuda_device_count": capabilities.cuda_device_count,
                }
                if capabilities is not None
                else None
            ),
        }


def _execution_default_values() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_info in fields(PyTorchExecutionConfig):
        if field_info.default is not MISSING:
            values[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:
            values[field_info.name] = field_info.default_factory()
        else:  # pragma: no cover - all compatibility fields currently default
            raise TypeError(
                f"execution field {field_info.name!r} has no primitive default"
            )
    return values


def normalize_execution_input(
    value: ExecutionRequest | PyTorchExecutionConfig | None,
    *,
    mode: str,
) -> NormalizedExecutionInput | None:
    """Copy an accepted execution input without observing the environment."""

    if mode not in {"training", "inference"}:
        raise ValueError(
            f"Invalid mode: {mode}. Expected 'training' or 'inference'."
        )
    if value is None:
        return None

    defaults = _execution_default_values()
    if isinstance(value, ExecutionRequest):
        defaults.update(value.as_dict())
        return NormalizedExecutionInput(
            values=defaults,
            explicit_fields=value.explicit_fields,
            notices=value.notices,
        )
    if isinstance(value, PyTorchExecutionConfig):
        copied = {
            field_info.name: getattr(value, field_info.name)
            for field_info in fields(PyTorchExecutionConfig)
        }
        explicit_aliases = frozenset(
            getattr(value, "_explicit_structural_aliases", frozenset())
        )
        explicit_fields = explicit_aliases
        if mode == "training":
            explicit_fields |= OPTIMIZER_EXECUTION_COMPAT_FIELDS
        return NormalizedExecutionInput(
            values=copied,
            explicit_fields=explicit_fields,
            notices=(),
            legacy_config=value,
            accelerator_already_resolved=True,
        )
    raise TypeError(
        "execution_config must be an ExecutionRequest, "
        "PyTorchExecutionConfig, or None"
    )


def materialize_default_execution_input(*, mode: str) -> NormalizedExecutionInput:
    """Create a complete, environment-unresolved default request."""

    normalized = normalize_execution_input(
        ExecutionRequest(values={}, explicit_fields=frozenset()),
        mode=mode,
    )
    assert normalized is not None
    return normalized


def validate_execution_input_structure(
    normalized: NormalizedExecutionInput,
) -> None:
    """Run all pure execution checks before capability observation."""

    _validate_execution_pre_environment_values(normalized.values)
    _validate_execution_post_environment_values(normalized.values)


def validate_execution_input_phase(
    normalized: NormalizedExecutionInput,
    *,
    mode: str,
) -> None:
    """Reject training-only compatibility inputs in inference requests."""

    if mode == "training":
        return
    if mode != "inference":
        raise ValueError(
            f"Invalid mode: {mode}. Expected 'training' or 'inference'."
        )
    invalid = normalized.explicit_fields & (
        OPTIMIZER_EXECUTION_COMPAT_FIELDS
        | TOPOLOGY_EXECUTION_COMPAT_FIELDS
    )
    if invalid:
        raise ValueError(
            "inference execution request cannot consume training-only "
            "optimizer/topology field(s): "
            + ", ".join(sorted(invalid))
        )


def observe_execution_capabilities() -> ExecutionCapabilities:
    """Observe the minimum CUDA facts required by runtime resolution."""

    import torch

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    return ExecutionCapabilities(
        cuda_available=available,
        cuda_device_count=count,
    )


def observe_cuda_device_count() -> int:
    """Observe CUDA cardinality when accelerator availability is already known."""

    import torch

    return int(torch.cuda.device_count())


def execution_capabilities_required(
    normalized: NormalizedExecutionInput,
) -> bool:
    """Return whether an unresolved runtime value needs CUDA facts."""

    accelerator = normalized.values["accelerator"]
    if not normalized.accelerator_already_resolved and accelerator == "auto":
        return True
    return (
        accelerator in {"gpu", "cuda"}
        and normalized.values["devices"] == "auto"
    )


def _policy_cpu_fallback_notice() -> ResolutionNotice:
    return ResolutionNotice(
        UserWarning,
        "POLICY-001: PyTorch backend defaults to GPU execution. "
        "No CUDA device detected; falling back to CPU. "
        "For production workloads, ensure CUDA is available or explicitly "
        "set accelerator='cpu'.",
    )


def resolve_execution_environment(
    normalized: NormalizedExecutionInput,
    *,
    capabilities: ExecutionCapabilities | None,
) -> EnvironmentResolution:
    """Resolve the four environment-dependent runtime fields purely."""

    requested = {
        name: normalized.values[name]
        for name in ENVIRONMENT_EXECUTION_FIELDS
    }
    resolved = dict(requested)
    notices: list[ResolutionNotice] = []

    if (
        not normalized.accelerator_already_resolved
        and resolved["accelerator"] == "auto"
    ):
        if capabilities is None:
            raise RuntimeError(
                "execution capabilities are required to resolve accelerator='auto'"
            )
        if capabilities.cuda_available:
            resolved["accelerator"] = "cuda"
        else:
            resolved["accelerator"] = "cpu"
            notices.append(_policy_cpu_fallback_notice())

    if resolved["devices"] == "auto":
        if resolved["accelerator"] in {"gpu", "cuda"}:
            if capabilities is None:
                raise RuntimeError(
                    "execution capabilities are required to resolve CUDA devices='auto'"
                )
            if capabilities.cuda_device_count <= 0:
                raise ValueError(
                    "CUDA devices='auto' requires a positive CUDA device count"
                )
            resolved["devices"] = capabilities.cuda_device_count
        else:
            resolved["devices"] = 1

    if (
        resolved["pin_memory"]
        and resolved["accelerator"] not in {"gpu", "cuda"}
    ):
        notices.append(
            ResolutionNotice(
                UserWarning,
                "pin_memory=True is unavailable for "
                f"accelerator={resolved['accelerator']!r}; using False.",
            )
        )
        resolved["pin_memory"] = False

    if (
        resolved["accelerator"] == "cpu"
        and resolved["precision"] == "16-mixed"
    ):
        resolved["precision"] = "bf16-mixed"

    return EnvironmentResolution(
        requested=requested,
        resolved=resolved,
        capabilities=capabilities,
        notices=tuple(notices),
    )


def resolve_runtime_execution_request(
    value: (
        ExecutionRequest
        | PyTorchExecutionConfig
        | NormalizedExecutionInput
        | None
    ),
    *,
    mode: str,
    execution_capabilities: ExecutionCapabilities | None = None,
) -> ResolvedRuntimeExecution:
    """Resolve a validated request to the compatible runtime carrier."""

    if isinstance(value, NormalizedExecutionInput):
        normalized = value
    else:
        normalized = normalize_execution_input(value, mode=mode)
        if normalized is None:
            normalized = materialize_default_execution_input(mode=mode)

    validate_execution_input_structure(normalized)
    validate_execution_input_phase(normalized, mode=mode)

    capabilities = None
    if execution_capabilities_required(normalized):
        if execution_capabilities is not None:
            capabilities = execution_capabilities
        elif (
            normalized.accelerator_already_resolved
            and normalized.values["accelerator"] in {"gpu", "cuda"}
            and normalized.values["devices"] == "auto"
        ):
            capabilities = ExecutionCapabilities(
                cuda_available=True,
                cuda_device_count=observe_cuda_device_count(),
            )
        else:
            capabilities = observe_execution_capabilities()

    environment = resolve_execution_environment(
        normalized,
        capabilities=capabilities,
    )
    resolved_values = _thaw_execution_values(normalized.values)
    resolved_values.update(environment.resolved)
    config = PyTorchExecutionConfig(**resolved_values)
    object.__setattr__(
        config,
        "_explicit_structural_aliases",
        frozenset(
            normalized.explicit_fields & TOPOLOGY_EXECUTION_COMPAT_FIELDS
        ),
    )
    notices = normalized.notices + environment.notices
    return ResolvedRuntimeExecution(
        config=config,
        environment=environment,
        explicit_fields=normalized.explicit_fields,
        notices=notices,
    )
