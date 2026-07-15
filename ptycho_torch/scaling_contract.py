"""Configuration contract for PyTorch absolute-intensity scaling profiles."""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch


CI_SCALE_CONTRACT = "ci_intensity_v2"
LEGACY_SCALE_CONTRACT = "legacy_v1"
COUNT_INTENSITY = "count_intensity"
NORMALIZED_AMPLITUDE = "normalized_amplitude"


@dataclass(frozen=True)
class ResolvedScaleContract:
    version: str
    measurement_domain: str


@dataclass(frozen=True)
class CIExperimentStatistics:
    rms_input_scale: torch.Tensor
    mean_measured_intensity: torch.Tensor


def _require_real_floating_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError(f"{name} must be a real floating-point tensor.")
    return value


def _coerce_positive_scalar(
    value: Any,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor.")
        if torch.is_complex(value) or value.dtype == torch.bool:
            raise TypeError(f"{name} must be a real scalar.")
        if value.device != reference.device:
            raise ValueError(
                f"{name} must be on device {reference.device}; got {value.device}."
            )
        scalar = value.to(dtype=reference.dtype)
    else:
        try:
            scalar = reference.new_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(f"{name} must be a real scalar.") from exc
        if scalar.ndim != 0 or torch.is_complex(scalar):
            raise TypeError(f"{name} must be a real scalar.")

    if not bool(torch.isfinite(scalar)) or not bool(scalar > 0):
        raise ValueError(f"{name} must be positive and finite.")
    return scalar


def derive_ci_experiment_statistics(
    measured_intensity: torch.Tensor,
    N: Any,
) -> CIExperimentStatistics:
    """Derive experiment-level CI input and loss normalization statistics."""
    measured_intensity = _require_real_floating_tensor(
        measured_intensity,
        "measured_intensity",
    )
    if measured_intensity.ndim != 4:
        raise ValueError("measured_intensity must have shape (B, C, H, W).")
    if not bool(torch.isfinite(measured_intensity).all()):
        raise ValueError("measured_intensity must contain only finite values.")
    if bool((measured_intensity < 0).any()):
        raise ValueError("measured_intensity must contain nonnegative counts.")

    n_scalar = _coerce_positive_scalar(N, "N", measured_intensity)
    mean_squared_energy = measured_intensity.square().sum(dim=(-2, -1)).mean()
    mean_measured_intensity = measured_intensity.mean()
    if not bool(torch.isfinite(mean_squared_energy)) or not bool(
        mean_squared_energy > 0
    ):
        raise ValueError("measured_intensity has zero or degenerate energy.")
    if not bool(torch.isfinite(mean_measured_intensity)) or not bool(
        mean_measured_intensity > 0
    ):
        raise ValueError("measured_intensity must have a positive finite mean.")

    target_energy = (n_scalar / 2).square()
    if not bool(torch.isfinite(target_energy)):
        raise ValueError("CI target energy must be finite.")
    rms_input_scale = torch.sqrt(target_energy / mean_squared_energy)
    if not bool(torch.isfinite(rms_input_scale)):
        raise ValueError("rms_input_scale must be finite.")
    return CIExperimentStatistics(
        rms_input_scale=rms_input_scale,
        mean_measured_intensity=mean_measured_intensity,
    )


def normalize_ci_poisson_per_sample(
    raw_nll: torch.Tensor,
    mean_measured_intensity: torch.Tensor,
) -> torch.Tensor:
    """Normalize per-sample count NLL by detached physical mean intensity."""
    raw_nll = _require_real_floating_tensor(raw_nll, "raw_nll")
    mean_measured_intensity = _require_real_floating_tensor(
        mean_measured_intensity,
        "mean_measured_intensity",
    )
    if raw_nll.ndim != 1:
        raise ValueError("raw_nll must have shape (B,).")
    if mean_measured_intensity.device != raw_nll.device:
        raise ValueError("mean_measured_intensity must be on the raw_nll device.")
    if mean_measured_intensity.dtype != raw_nll.dtype:
        raise ValueError("mean_measured_intensity must match raw_nll dtype.")

    batch_size = raw_nll.shape[0]
    if mean_measured_intensity.numel() == 1:
        denominator = mean_measured_intensity.reshape(())
    elif (
        mean_measured_intensity.shape[0] == batch_size
        and all(size == 1 for size in mean_measured_intensity.shape[1:])
    ):
        denominator = mean_measured_intensity.reshape(batch_size)
    else:
        raise ValueError(
            "mean_measured_intensity must be scalar or have shape "
            "(B, 1, ...) matching raw_nll."
        )
    denominator = denominator.detach()
    torch._assert_async(
        torch.isfinite(denominator).all(),
        "mean_measured_intensity must be finite.",
    )
    torch._assert_async(
        (denominator > 0).all(),
        "mean_measured_intensity must be positive.",
    )
    return raw_nll / denominator


def adapt_normalized_amplitude_to_ci(
    amplitude: torch.Tensor,
    probe: torch.Tensor,
    count_amplitude_scale: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized-amplitude data and probe to physical CI units."""
    amplitude = _require_real_floating_tensor(amplitude, "amplitude")
    if not isinstance(probe, torch.Tensor):
        raise TypeError("probe must be a torch.Tensor.")
    if not (torch.is_floating_point(probe) or torch.is_complex(probe)):
        raise TypeError("probe must be a floating-point or complex tensor.")
    if probe.device != amplitude.device:
        raise ValueError("amplitude and probe must be on the same device.")

    scale = _coerce_positive_scalar(
        count_amplitude_scale,
        "count_amplitude_scale",
        amplitude,
    )
    if not bool(torch.isfinite(amplitude).all()):
        raise ValueError("amplitude must contain only finite values.")
    if bool((amplitude < 0).any()):
        raise ValueError("amplitude must contain nonnegative values.")
    if not bool((amplitude != 0).any()):
        raise ValueError("amplitude must have nonzero energy.")

    probe_is_finite = torch.isfinite(probe.real).all()
    if torch.is_complex(probe):
        probe_is_finite = probe_is_finite & torch.isfinite(probe.imag).all()
    if not bool(probe_is_finite):
        raise ValueError("probe real and imaginary components must be finite.")
    if not bool((probe != 0).any()):
        raise ValueError("probe must have nonzero energy.")

    intensity = (scale * amplitude).square()
    probe_physical = scale * probe
    if not bool(torch.isfinite(intensity).all()):
        raise ValueError("converted intensity must contain only finite values.")

    converted_probe_is_finite = torch.isfinite(probe_physical.real).all()
    if torch.is_complex(probe_physical):
        converted_probe_is_finite = (
            converted_probe_is_finite & torch.isfinite(probe_physical.imag).all()
        )
    if not bool(converted_probe_is_finite):
        raise ValueError(
            "converted probe real and imaginary components must be finite."
        )
    return intensity, probe_physical


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


def validate_amplitude_physics_gain(model_config: Any) -> float:
    """Validate ``ModelConfig.amplitude_physics_gain`` (PROBE-RANK-001 §3.3).

    The explicit gain replaces the accidental flat-probe xB amplitude gain
    (docs/specs/spec-ptycho-torch-probe-layout.md). It must be finite and
    > 0 in every mode; whenever the rectangular/CI scaling path is active it
    must be exactly 1.0 (fail-closed — the gain is an amplitude-forward-only
    training-objective device and those modes must remain untouched by it).
    Configs without the attribute (pre-fix checkpoints, duck-typed test
    stand-ins) resolve to the 1.0 default.
    """
    gain = getattr(model_config, "amplitude_physics_gain", 1.0)
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        raise TypeError(
            f"amplitude_physics_gain must be a real number; got {gain!r}."
        )
    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(
            f"amplitude_physics_gain must be positive and finite; got {gain!r}."
        )
    if ci_scaling_active(model_config) and gain != 1.0:
        raise ValueError(
            "amplitude_physics_gain must be 1.0 when "
            "physics_forward_mode='rectangular_scaled' (rectangular/CI "
            f"scaling contract, fail-closed); got {gain!r}. The explicit "
            "gain applies only to the amplitude-mode training forward "
            "(PROBE-RANK-001)."
        )
    return gain


def validate_contract_coherence(
    data_config: Any,
    model_config: Any,
    training_config: Any,
) -> None:
    """Fail-closed coherence validation across the three config objects.

    Conformance D3 (Theme 3, docs/superpowers/plans/
    2026-07-14-ci-paper-conformance-audit.md): a single unconditional entry
    point that raises ``ValueError`` on ACTIVE contradictions:

    - ``physics_forward_mode='rectangular_scaled'`` with an unsupported
      (``scale_contract_version``, ``measurement_domain``) pair;
    - active ``ci_intensity_v2`` with supervised mode or a non-poisson primary
      loss — this covers ``measurement_domain='count_intensity'`` combined
      with ``torch_loss_mode='mae'`` under the rectangular forward, because
      ``count_intensity`` only resolves inside the CI profile;
    - a non-1.0 ``amplitude_physics_gain`` wherever the contract forbids it
      (every rectangular/CI mode; validated in every mode for finiteness).

    Deliberately a no-op pass for BOTH coherent bundles:

    - coherent legacy: the amplitude forward ignores the CI-flavored
      ``DataConfig`` defaults by design (2026-07-09 CI ablation design,
      "Amplitude mode does not activate CI even when absent profile fields
      receive CI defaults"), so bare-default construction stays valid;
    - coherent CI: rectangular + ``ci_intensity_v2``/``count_intensity`` +
      ``torch_loss_mode='poisson'``.

    Explicit-intent detection for half-configured CI (CI-only knobs passed
    without the rectangular forward) lives at the factory layer
    (``ptycho_torch.config_factory``), where override explicitness is
    knowable; bare dataclasses cannot distinguish defaults from intent.
    """
    validate_scale_contract(data_config, model_config, training_config)
    return None


def validate_scale_contract(
    data_config: Any,
    model_config: Any,
    training_config: Any,
) -> Optional[ResolvedScaleContract]:
    """Validate the active rectangular profile and its CI training constraints.

    Also validates ``amplitude_physics_gain`` in EVERY mode (PROBE-RANK-001
    §3.3): finite and > 0, and exactly 1.0 for rectangular_scaled/CI modes.
    """
    validate_amplitude_physics_gain(model_config)
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
