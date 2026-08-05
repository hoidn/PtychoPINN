"""Configuration contract for PyTorch absolute-intensity scaling profiles."""

import json
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import numpy as np
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


@dataclass(frozen=True)
class AmplitudePhysicsGainRecord:
    """Plain persisted resolution of one training-time amplitude gain."""

    value: float
    provenance: str
    method: str
    version: str
    input_statistics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("value must be a positive real scalar")
        value = float(self.value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("value must be positive and finite")
        for name in ("provenance", "method", "version"):
            field_value = getattr(self, name)
            if not isinstance(field_value, str) or not field_value:
                raise TypeError(f"{name} must be a non-empty string")
        if not isinstance(self.input_statistics, Mapping):
            raise TypeError("input_statistics must be a mapping")
        statistics = dict(self.input_statistics)
        try:
            json.dumps(statistics, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "input_statistics must be finite JSON-native values"
            ) from error
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "input_statistics", statistics)
        _validate_gain_record_semantics(self)

    def factory_overrides(self) -> dict[str, float]:
        """Return the existing Torch factory patch owned by this record."""

        return {"amplitude_physics_gain": self.value}

    def to_metadata(self) -> dict[str, Any]:
        """Return a detached JSON-native copy of the plain persisted record."""

        return asdict(self)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "AmplitudePhysicsGainRecord":
        """Decode the five persisted record fields."""

        if not isinstance(metadata, Mapping):
            raise ValueError("amplitude physics gain record must be a mapping")
        try:
            return cls(
                value=metadata["value"],
                provenance=metadata["provenance"],
                method=metadata["method"],
                version=metadata["version"],
                input_statistics=metadata["input_statistics"],
            )
        except KeyError as error:
            raise ValueError(
                f"amplitude physics gain record is missing {error.args[0]!r}"
            ) from error


def amplitude_physics_gain_record_to_json(
    record: AmplitudePhysicsGainRecord,
) -> str:
    """Encode the plain five-field record stored in the bundle."""

    return json.dumps(
        record.to_metadata(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def amplitude_physics_gain_record_from_json(
    encoded: Any,
) -> AmplitudePhysicsGainRecord:
    """Decode one plain gain-record JSON sidecar."""

    if not isinstance(encoded, (str, bytes, bytearray)):
        raise TypeError("amplitude physics gain sidecar JSON must be text or bytes")
    try:
        record = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid amplitude physics gain sidecar JSON") from exc
    return AmplitudePhysicsGainRecord.from_metadata(record)


_GAIN_METHOD = "normalized_amplitude_physical_gain"
_GAIN_VERSION = "legacy-amplitude-physics-gain-v1"
_GAIN_RESOLUTION_VERSION = "amplitude-physics-gain-resolution-v1"


def _validate_gain_record_semantics(record: AmplitudePhysicsGainRecord) -> None:
    allowed = {
        "derived": (_GAIN_METHOD, _GAIN_VERSION),
        "override": ("advanced_model_override", _GAIN_RESOLUTION_VERSION),
        "scale_contract_fixed": (
            "rectangular_scale_contract_fixed",
            _GAIN_RESOLUTION_VERSION,
        ),
    }
    expected = allowed.get(record.provenance)
    if expected is None:
        raise ValueError(f"unsupported amplitude gain provenance {record.provenance!r}")
    if (record.method, record.version) != expected:
        raise ValueError(
            f"provenance {record.provenance!r} requires method/version {expected!r}"
        )
    if record.provenance == "scale_contract_fixed" and record.value != 1.0:
        raise ValueError("scale_contract_fixed value must be exactly 1.0")


def _mask_control_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def _gain_record(
    *,
    value: float,
    provenance: str,
    method: str,
    version: str,
    input_statistics: Mapping[str, Any],
) -> AmplitudePhysicsGainRecord:
    value = validate_amplitude_physics_gain(
        SimpleNamespace(
            amplitude_physics_gain=value,
            physics_forward_mode=(
                "rectangular_scaled"
                if provenance == "scale_contract_fixed"
                else "amplitude"
            ),
        )
    )
    return AmplitudePhysicsGainRecord(
        value=value,
        provenance=provenance,
        method=method,
        version=version,
        input_statistics=dict(input_statistics),
    )


def _canonical_detector_samples(
    value: np.ndarray,
    *,
    name: str,
    complex_values: bool,
) -> np.ndarray:
    dtype = np.complex128 if complex_values else np.float64
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 3:
        samples = array
    elif array.ndim == 4 and array.shape[1] == array.shape[2]:
        # Grouped loader layout (B, N, N, C).
        samples = np.moveaxis(array, -1, 1).reshape(
            -1, array.shape[1], array.shape[2]
        )
    elif array.ndim == 4 and array.shape[-2] == array.shape[-1]:
        # Torch-native grouped layout (B, C, N, N).
        samples = array.reshape(-1, array.shape[-2], array.shape[-1])
    else:
        raise ValueError(
            f"{name} must have flat shape (S, N, N) or grouped shape "
            "(B, N, N, C)/(B, C, N, N)"
        )
    samples = np.ascontiguousarray(samples)
    if samples.shape[1] != samples.shape[2]:
        raise ValueError(f"{name} detector patterns must be square")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name} must contain only finite values")
    if not complex_values and np.any(samples < 0):
        raise ValueError(f"{name} must contain nonnegative amplitudes")
    return samples


def _canonical_probe_modes(stored_probe: np.ndarray, N: int) -> np.ndarray:
    probe = np.asarray(stored_probe, dtype=np.complex64)
    if probe.shape == (N, N):
        probe = probe[None, ...]
    elif probe.shape == (N, N, 1):
        probe = probe[..., 0][None, ...]
    elif probe.ndim == 3 and probe.shape[-2:] == (N, N):
        pass
    else:
        raise ValueError(
            "stored_probe must have shape (N, N), (N, N, 1), or (P, N, N)"
        )
    if not np.isfinite(probe).all():
        raise ValueError("stored_probe must contain only finite values")
    if not np.any(probe):
        raise ValueError("stored_probe must have nonzero energy")
    return np.ascontiguousarray(probe)


def derive_legacy_amplitude_physics_gain(
    measured_amplitude: np.ndarray,
    object_patches: np.ndarray,
    stored_probe: np.ndarray,
    *,
    probe_scale: float,
    probe_mask: Any = False,
    probe_mask_tensor: Any = None,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: float | None = None,
) -> AmplitudePhysicsGainRecord:
    """Derive the documented legacy normalized-amplitude physical gain."""

    if isinstance(probe_scale, bool) or not isinstance(
        probe_scale, (int, float, np.integer, np.floating)
    ):
        raise TypeError("probe_scale must be a real scalar")
    probe_scale = float(probe_scale)
    if not math.isfinite(probe_scale) or probe_scale <= 0:
        raise ValueError("probe_scale must be positive and finite")

    measured = _canonical_detector_samples(
        measured_amplitude,
        name="measured_amplitude",
        complex_values=False,
    )
    truth = _canonical_detector_samples(
        object_patches,
        name="object_patches",
        complex_values=True,
    )
    if measured.shape != truth.shape:
        raise ValueError(
            "measured_amplitude and object_patches must resolve to matching "
            f"sample shapes, got {measured.shape} and {truth.shape}"
        )
    if not np.any(measured):
        raise ValueError("measured_amplitude must have positive nonzero energy")
    N = measured.shape[-1]
    probe_modes = _canonical_probe_modes(stored_probe, N)

    from ptycho_torch.helper import normalize_probe_like_tf
    from ptycho_torch.probe_mask import resolve_probe_mask_np

    normalized_probe, normalization_multiplier = normalize_probe_like_tf(
        probe_modes,
        probe_scale=probe_scale,
        probe_mask=probe_mask,
        probe_mask_tensor=probe_mask_tensor,
        probe_mask_sigma=probe_mask_sigma,
        probe_mask_diameter=probe_mask_diameter,
    )
    resolved_mask = resolve_probe_mask_np(
        N,
        probe_mask=probe_mask,
        probe_mask_tensor=probe_mask_tensor,
        probe_mask_sigma=probe_mask_sigma,
        probe_mask_diameter=probe_mask_diameter,
    )
    effective_probe = (
        np.asarray(normalized_probe, dtype=np.complex128)
        * np.asarray(resolved_mask, dtype=np.float64)[None, ...]
    ) / probe_scale
    coherent_field = np.fft.fft2(
        truth[:, None, ...] * effective_probe[None, ...], axes=(-2, -1)
    ).sum(axis=1)
    forward_amplitude = np.fft.fftshift(
        np.abs(coherent_field), axes=(-2, -1)
    ) / float(N)

    measured_squared = np.square(measured, dtype=np.float64)
    measured_energy = float(measured_squared.sum(dtype=np.float64))
    mean_sample_energy = float(
        measured_squared.sum(axis=(-2, -1), dtype=np.float64).mean()
    )
    forward_energy = float(
        np.square(forward_amplitude, dtype=np.float64).sum(dtype=np.float64)
    )
    if not math.isfinite(mean_sample_energy) or mean_sample_energy <= 0:
        raise ValueError("measured_amplitude has degenerate sample energy")
    if not math.isfinite(forward_energy) or forward_energy <= 0:
        raise ValueError("object_patches/stored_probe forward has degenerate energy")
    r = math.sqrt((N**2) / mean_sample_energy)
    value = r * math.sqrt(measured_energy / forward_energy)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("derived amplitude_physics_gain must be positive and finite")

    statistics = {
        "N": N,
        "sample_count": measured.shape[0],
        "probe_mode_count": effective_probe.shape[0],
        "probe_scale": probe_scale,
        "probe_normalization_multiplier": float(normalization_multiplier),
        "effective_probe_multiplier": float(normalization_multiplier)
        / probe_scale,
        "probe_mask_settings": {
            "probe_mask": _mask_control_metadata(probe_mask),
            "probe_mask_tensor": _mask_control_metadata(probe_mask_tensor),
            "probe_mask_sigma": float(probe_mask_sigma),
            "probe_mask_diameter": (
                None
                if probe_mask_diameter is None
                else float(probe_mask_diameter)
            ),
        },
        "mean_sample_measured_energy": mean_sample_energy,
        "measured_energy": measured_energy,
        "forward_energy": forward_energy,
        "r": r,
    }
    return _gain_record(
        value=value,
        provenance="derived",
        method=_GAIN_METHOD,
        version=_GAIN_VERSION,
        input_statistics=statistics,
    )


derive_amplitude_physics_gain = derive_legacy_amplitude_physics_gain


def resolve_amplitude_physics_gain(
    measured_amplitude: np.ndarray | None = None,
    object_patches: np.ndarray | None = None,
    stored_probe: np.ndarray | None = None,
    *,
    probe_scale: float,
    override: float | None = None,
    physics_forward_mode: str = "amplitude",
    probe_mask: Any = False,
    probe_mask_tensor: Any = None,
    probe_mask_sigma: float = 1.0,
    probe_mask_diameter: float | None = None,
) -> AmplitudePhysicsGainRecord:
    """Resolve override/fixed/derived gain without relabeling provenance."""

    if physics_forward_mode not in {"amplitude", "rectangular_scaled"}:
        raise ValueError(
            f"unsupported physics_forward_mode {physics_forward_mode!r}"
        )
    if physics_forward_mode == "rectangular_scaled":
        if override is not None:
            validate_amplitude_physics_gain(
                SimpleNamespace(
                    amplitude_physics_gain=override,
                    physics_forward_mode=physics_forward_mode,
                )
            )
        return _gain_record(
            value=1.0,
            provenance="scale_contract_fixed",
            method="rectangular_scale_contract_fixed",
            version=_GAIN_RESOLUTION_VERSION,
            input_statistics={},
        )
    if override is not None:
        value = validate_amplitude_physics_gain(
            SimpleNamespace(
                amplitude_physics_gain=override,
                physics_forward_mode=physics_forward_mode,
            )
        )
        return _gain_record(
            value=value,
            provenance="override",
            method="advanced_model_override",
            version=_GAIN_RESOLUTION_VERSION,
            input_statistics={},
        )
    if measured_amplitude is None or object_patches is None or stored_probe is None:
        raise ValueError(
            "legacy amplitude gain derivation requires measured_amplitude, "
            "object_patches, and stored_probe"
        )
    return derive_legacy_amplitude_physics_gain(
        measured_amplitude,
        object_patches,
        stored_probe,
        probe_scale=probe_scale,
        probe_mask=probe_mask,
        probe_mask_tensor=probe_mask_tensor,
        probe_mask_sigma=probe_mask_sigma,
        probe_mask_diameter=probe_mask_diameter,
    )


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
    - ``rect_s1s2_init='dose_closure'`` outside the complete rectangular
      ``ci_intensity_v2``/``count_intensity``/Poisson contract;
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
    rect_s1s2_init = getattr(model_config, "rect_s1s2_init", "ones")
    if not ci_scaling_active(model_config):
        if rect_s1s2_init == "dose_closure":
            raise ValueError(
                "rect_s1s2_init='dose_closure' requires the coherent "
                "ci_intensity_v2/count_intensity rectangular-scaled contract; "
                "physics_forward_mode must be 'rectangular_scaled'."
            )
        return None

    resolved = resolve_scale_contract(
        getattr(data_config, "scale_contract_version", None),
        getattr(data_config, "measurement_domain", None),
    )
    if resolved.version != CI_SCALE_CONTRACT:
        if rect_s1s2_init == "dose_closure":
            raise ValueError(
                "rect_s1s2_init='dose_closure' requires the coherent "
                "ci_intensity_v2/count_intensity contract; got "
                f"{resolved.version!r}/{resolved.measurement_domain!r}."
            )
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
