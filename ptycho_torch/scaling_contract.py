"""Configuration contract for PyTorch absolute-intensity scaling profiles."""

from dataclasses import dataclass
from typing import Any, Optional


CI_SCALE_CONTRACT = "ci_intensity_v2"
LEGACY_SCALE_CONTRACT = "legacy_v1"
COUNT_INTENSITY = "count_intensity"
NORMALIZED_AMPLITUDE = "normalized_amplitude"


@dataclass(frozen=True)
class ResolvedScaleContract:
    version: str
    measurement_domain: str


def resolve_scale_contract(
    version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
) -> ResolvedScaleContract:
    """Apply CI defaults independently, then require a supported profile pair."""
    resolved = ResolvedScaleContract(
        version=CI_SCALE_CONTRACT if version is None else version,
        measurement_domain=(
            COUNT_INTENSITY if measurement_domain is None else measurement_domain
        ),
    )
    supported = {
        ResolvedScaleContract(CI_SCALE_CONTRACT, COUNT_INTENSITY),
        ResolvedScaleContract(LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE),
    }
    if resolved not in supported:
        raise ValueError(
            "Unsupported scale contract profile: "
            f"version={resolved.version!r}, "
            f"measurement_domain={resolved.measurement_domain!r}. "
            "Expected ('ci_intensity_v2', 'count_intensity') or "
            "('legacy_v1', 'normalized_amplitude')."
        )
    return resolved


def ci_scaling_active(model_config: Any) -> bool:
    """Return whether the rectangular scaling path activates contract validation."""
    return getattr(model_config, "physics_forward_mode", "amplitude") == "rectangular_scaled"


def validate_scale_contract(
    data_config: Any,
    model_config: Any,
    training_config: Any,
) -> Optional[ResolvedScaleContract]:
    """Validate the active rectangular profile and its CI training constraints."""
    if not ci_scaling_active(model_config):
        return None

    resolved = resolve_scale_contract(
        getattr(data_config, "scale_contract_version", None),
        getattr(data_config, "measurement_domain", None),
    )
    if resolved.version != CI_SCALE_CONTRACT:
        return resolved

    mode = getattr(model_config, "mode", None)
    if mode != "Unsupervised":
        raise ValueError(
            "ci_intensity_v2 requires ModelConfig.mode='Unsupervised'; "
            f"got {mode!r}."
        )

    torch_loss_mode = getattr(training_config, "torch_loss_mode", None)
    if torch_loss_mode != "poisson":
        raise ValueError(
            "ci_intensity_v2 requires TrainingConfig.torch_loss_mode='poisson'; "
            f"got {torch_loss_mode!r}. ModelConfig.loss_function does not override "
            "the Lightning primary loss."
        )

    return resolved
