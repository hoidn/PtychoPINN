"""Immutable, JSON-safe diagnostics for probe-weighted reassembly."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class NotApplicable:
    """Immutable typed marker for diagnostics outside a scale profile."""

    status: str = field(default="not_applicable", init=False)
    reason: str = field(default="legacy_normalized_amplitude", init=False)

    def to_jsonable(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ConditionStatus:
    """Explicit JSON-native state for a nonfinite condition number."""

    status: str = field(default="rank_deficient", init=False)
    value: None = field(default=None, init=False)
    reason: str = field(default="nonfinite_condition_number", init=False)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": self.value,
            "reason": self.reason,
        }


def not_applicable() -> NotApplicable:
    """Return the typed marker used by legacy normalized-amplitude profiles."""
    return NotApplicable()


@dataclass(frozen=True, slots=True)
class NotEvaluated:
    """Typed marker for evidence intentionally deferred to a canonical pass."""

    status: str = field(default="not_evaluated", init=False)
    reason: str = field(default="deferred_to_canonical_runtime", init=False)

    def to_jsonable(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


def not_evaluated() -> NotEvaluated:
    """Return the typed marker for intentionally deferred diagnostics."""
    return NotEvaluated()


def _snapshot_tensor(value: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    snapshot = value.detach().to(device="cpu", dtype=dtype).clone().contiguous()
    snapshot.requires_grad_(False)
    return snapshot


def _finite_float(value: Any, name: str) -> float:
    try:
        if isinstance(value, torch.Tensor):
            result = float(value.detach().cpu().item())
        else:
            result = float(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_int(value: Any, name: str) -> int:
    numeric = _finite_float(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _require_finite_array(value: Any, name: str) -> None:
    tensor = torch.as_tensor(value).detach()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def array_metadata(value: Any) -> dict[str, Any]:
    """Return stable shape, dtype, and content hash metadata for an array."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    dtype = str(array.dtype)
    shape = [int(size) for size in array.shape]
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {"shape": shape, "dtype": dtype, "sha256": digest.hexdigest()}


def array_digest(value: Any) -> str:
    return array_metadata(value)["sha256"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON diagnostic values must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-safe")


@dataclass(frozen=True, init=False, slots=True)
class VarProSufficientStatistics:
    """Detached aggregate least-squares evidence for the three VarPro bases."""

    _ata_values: Tuple[Tuple[float, float, float], ...]
    _atb_values: Tuple[float, float, float]
    sum_i2: float
    n_pixels: int

    def __init__(
        self,
        ATA: Optional[torch.Tensor] = None,
        ATb: Optional[torch.Tensor] = None,
        sum_i2: float = 0.0,
        n_pixels: int = 0,
        *,
        ata: Optional[torch.Tensor] = None,
        atb: Optional[torch.Tensor] = None,
    ) -> None:
        if ATA is not None and ata is not None:
            raise TypeError("Specify only one of ATA or ata")
        if ATb is not None and atb is not None:
            raise TypeError("Specify only one of ATb or atb")
        resolved_ata = ATA if ATA is not None else ata
        resolved_atb = ATb if ATb is not None else atb
        if resolved_ata is None or resolved_atb is None:
            raise TypeError("ATA and ATb are required")

        ata_snapshot = _snapshot_tensor(resolved_ata, dtype=torch.float64)
        atb_snapshot = _snapshot_tensor(resolved_atb, dtype=torch.float64)
        if ata_snapshot.shape != (3, 3):
            raise ValueError("ATA must have shape (3, 3)")
        if atb_snapshot.shape == (3,):
            atb_snapshot = atb_snapshot[:, None]
        if atb_snapshot.shape != (3, 1):
            raise ValueError("ATb must have shape (3, 1) or (3,)")
        if not bool(torch.isfinite(ata_snapshot).all()):
            raise ValueError("ATA must contain only finite values")
        if not bool(torch.isfinite(atb_snapshot).all()):
            raise ValueError("ATb must contain only finite values")
        sum_i2_value = _finite_float(sum_i2, "sum_i2")
        if sum_i2_value < 0:
            raise ValueError("sum_i2 must be nonnegative")
        n_pixels_value = _finite_int(n_pixels, "n_pixels")
        if n_pixels_value <= 0:
            raise ValueError("n_pixels must be positive")

        object.__setattr__(
            self,
            "_ata_values",
            tuple(
                tuple(float(value) for value in row)
                for row in ata_snapshot.tolist()
            ),
        )
        object.__setattr__(
            self,
            "_atb_values",
            tuple(float(value) for value in atb_snapshot.flatten().tolist()),
        )
        object.__setattr__(self, "sum_i2", sum_i2_value)
        object.__setattr__(self, "n_pixels", n_pixels_value)

    @property
    def ATA(self) -> torch.Tensor:
        return torch.tensor(self._ata_values, dtype=torch.float64)

    @property
    def ATb(self) -> torch.Tensor:
        return torch.tensor(self._atb_values, dtype=torch.float64)[:, None]

    @property
    def ata(self) -> torch.Tensor:
        return self.ATA

    @property
    def atb(self) -> torch.Tensor:
        return self.ATb

    def objective_for_z(self, z: Any) -> float:
        coefficients = torch.as_tensor(z, dtype=torch.float64, device="cpu").reshape(-1)
        if coefficients.shape != (3,):
            raise ValueError("z must contain exactly three coefficients")
        if not bool(torch.isfinite(coefficients).all()):
            raise ValueError("objective coefficients must be finite")
        value = (
            coefficients @ self.ATA @ coefficients
            - 2.0 * coefficients @ self.ATb.flatten()
            + self.sum_i2
        ) / self.n_pixels
        result = float(value.item())
        if not math.isfinite(result):
            raise ValueError("VarPro objective must be finite")
        roundoff = 1e-14 * max(1.0, abs(self.sum_i2 / self.n_pixels))
        if result < 0.0 and abs(result) <= roundoff:
            return 0.0
        return result

    def objective(self, s1: Any, s2: Any) -> float:
        s1_value = _finite_float(s1, "s1")
        s2_value = _finite_float(s2, "s2")
        return self.objective_for_z(
            [s1_value * s1_value, s2_value * s2_value, s1_value * s2_value]
        )

    def condition_number(self) -> float:
        return float(torch.linalg.cond(self.ATA).item())

    def condition_record(self) -> Union[float, ConditionStatus]:
        condition = self.condition_number()
        if math.isfinite(condition):
            return condition
        return ConditionStatus()

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ATA": [list(row) for row in self._ata_values],
            "ATb": [[value] for value in self._atb_values],
            "sum_i2": self.sum_i2,
            "n_pixels": self.n_pixels,
        }


@dataclass(frozen=True, slots=True)
class FittedCountMetrics:
    relative_l2_intensity_error: float
    mean_raw_poisson_nll: float
    n_samples: int
    n_pixels: int
    effective_mask_digest: str
    sample_ids: Tuple[int, ...] = ()
    sample_identity_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.effective_mask_digest, str):
            raise TypeError("effective_mask_digest must be a string")
        object.__setattr__(
            self,
            "relative_l2_intensity_error",
            float(self.relative_l2_intensity_error),
        )
        object.__setattr__(
            self,
            "mean_raw_poisson_nll",
            float(self.mean_raw_poisson_nll),
        )
        object.__setattr__(self, "n_samples", _finite_int(self.n_samples, "n_samples"))
        object.__setattr__(self, "n_pixels", _finite_int(self.n_pixels, "n_pixels"))
        sample_ids = tuple(_finite_int(item, "sample_ids") for item in self.sample_ids)
        if any(item < 0 for item in sample_ids):
            raise ValueError("sample_ids must be nonnegative")
        if sample_ids and len(sample_ids) != self.n_samples:
            raise ValueError("sample_ids must match n_samples including multiplicity")
        object.__setattr__(self, "sample_ids", sample_ids)
        if sample_ids:
            digest = array_digest(torch.as_tensor(sample_ids, dtype=torch.int64))
            if self.sample_identity_digest and self.sample_identity_digest != digest:
                raise ValueError("sample_identity_digest does not match sample_ids")
            object.__setattr__(self, "sample_identity_digest", digest)
        elif self.sample_identity_digest:
            raise ValueError("sample_identity_digest requires sample_ids")
        if not math.isfinite(self.relative_l2_intensity_error):
            raise ValueError("relative_l2_intensity_error must be finite")
        if not math.isfinite(self.mean_raw_poisson_nll):
            raise ValueError("mean_raw_poisson_nll must be finite")
        if self.n_samples < 0 or self.n_pixels < 0:
            raise ValueError("Count metric sample and pixel counts must be nonnegative")

    @property
    def mask_digest(self) -> str:
        return self.effective_mask_digest

    @property
    def effective_probe_mask_digest(self) -> str:
        return self.effective_mask_digest

    def to_jsonable(self) -> dict[str, Any]:
        return _json_safe(self)


StatisticsRecord = Union[VarProSufficientStatistics, NotApplicable]
MetricRecord = Union[FittedCountMetrics, NotApplicable, NotEvaluated, None]
ConditionRecord = Union[float, ConditionStatus, NotApplicable]
ObjectiveRecord = Union[float, NotApplicable]


@dataclass(frozen=True, init=False, slots=True)
class ReassemblyDiagnostics:
    """Schema-v1 dataset-level reassembly and VarPro diagnostics."""

    schema_version: int
    inference_time: float
    assembly_time: float
    solve_time: float
    s1: float
    s2: float
    sufficient_statistics: StatisticsRecord
    condition: ConditionRecord
    unit_objective: ObjectiveRecord
    fitted_objective: ObjectiveRecord
    scale_profile: str
    mask_digest: str
    _canvas_anchor_json: str
    _canvas_weights_data: bytes
    _canvas_weights_shape: Tuple[int, ...]
    _canvas_weights_dtype: str
    _canvas_weights_sha256: str
    accepted_patches: int
    total_patches: int
    used_scan_ids: Tuple[int, ...]
    used_center_scan_ids: Tuple[int, ...]
    center_identity_available: bool
    expected_scan_ids: Tuple[int, ...]
    filtered_eligible_scan_ids: Tuple[int, ...]
    decoder_real_saturation_fraction: Optional[float]
    decoder_imag_saturation_fraction: Optional[float]
    decoder_real_lower_saturation_fraction: Optional[float]
    decoder_real_upper_saturation_fraction: Optional[float]
    decoder_imag_lower_saturation_fraction: Optional[float]
    decoder_imag_upper_saturation_fraction: Optional[float]
    count_metrics: MetricRecord
    effective_precision: str

    def __init__(
        self,
        *,
        inference_time: float,
        assembly_time: float,
        solve_time: float,
        s1: Any,
        s2: Any,
        sufficient_statistics: StatisticsRecord,
        condition: ConditionRecord,
        unit_objective: ObjectiveRecord,
        fitted_objective: ObjectiveRecord,
        scale_profile: str,
        mask_digest: str,
        canvas_anchor: Mapping[str, Any],
        canvas_weights: torch.Tensor,
        accepted_patches: int,
        total_patches: int,
        count_metrics: MetricRecord,
        used_scan_ids: Any = (),
        used_center_scan_ids: Any = (),
        center_identity_available: bool = True,
        expected_scan_ids: Any = (),
        filtered_eligible_scan_ids: Any = (),
        decoder_real_saturation_fraction: Any = None,
        decoder_imag_saturation_fraction: Any = None,
        decoder_real_lower_saturation_fraction: Any = None,
        decoder_real_upper_saturation_fraction: Any = None,
        decoder_imag_lower_saturation_fraction: Any = None,
        decoder_imag_upper_saturation_fraction: Any = None,
        effective_precision: str = "32-true",
        schema_version: int = 1,
    ) -> None:
        if schema_version != 1:
            raise ValueError("ReassemblyDiagnostics supports schema_version=1 only")
        accepted = _finite_int(accepted_patches, "accepted_patches")
        total = _finite_int(total_patches, "total_patches")
        if accepted < 0 or total < 0 or accepted > total:
            raise ValueError("Patch counts must satisfy 0 <= accepted <= total")
        try:
            used_scans = tuple(_finite_int(item, "used_scan_ids") for item in used_scan_ids)
            used_centers = tuple(
                _finite_int(item, "used_center_scan_ids")
                for item in used_center_scan_ids
            )
            expected_scans = tuple(
                _finite_int(item, "expected_scan_ids") for item in expected_scan_ids
            )
            filtered_scans = tuple(
                _finite_int(item, "filtered_eligible_scan_ids")
                for item in filtered_eligible_scan_ids
            )
        except TypeError as error:
            raise TypeError("scan ids must be iterable integers") from error
        if not filtered_scans and expected_scans:
            filtered_scans = expected_scans
        if type(center_identity_available) is not bool:
            raise TypeError("center_identity_available must be boolean")
        if not center_identity_available and used_centers:
            raise ValueError("unavailable center identity cannot carry used center ids")
        if (
            any(item < 0 for item in (*used_scans, *used_centers, *filtered_scans, *expected_scans))
            or len(set(used_scans)) != len(used_scans)
            or len(set(used_centers)) != len(used_centers)
            or len(set(filtered_scans)) != len(filtered_scans)
            or len(set(expected_scans)) != len(expected_scans)
            or (
                center_identity_available
                and not set(used_centers).issubset(filtered_scans)
            )
            or not set(used_scans).issubset(expected_scans)
            or not set(filtered_scans).issubset(expected_scans)
        ):
            raise ValueError(
                "scan ids must be unique nonnegative values with centers within filtered and participants within source"
            )
        saturation_values = []
        for name, value in (
            ("decoder_real_saturation_fraction", decoder_real_saturation_fraction),
            ("decoder_imag_saturation_fraction", decoder_imag_saturation_fraction),
            ("decoder_real_lower_saturation_fraction", decoder_real_lower_saturation_fraction),
            ("decoder_real_upper_saturation_fraction", decoder_real_upper_saturation_fraction),
            ("decoder_imag_lower_saturation_fraction", decoder_imag_lower_saturation_fraction),
            ("decoder_imag_upper_saturation_fraction", decoder_imag_upper_saturation_fraction),
        ):
            if value is None:
                saturation_values.append(None)
                continue
            number = _finite_float(value, name)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            saturation_values.append(number)
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(
            self,
            "inference_time",
            _finite_float(inference_time, "inference_time"),
        )
        object.__setattr__(
            self,
            "assembly_time",
            _finite_float(assembly_time, "assembly_time"),
        )
        object.__setattr__(self, "solve_time", _finite_float(solve_time, "solve_time"))
        object.__setattr__(self, "s1", _finite_float(s1, "s1"))
        object.__setattr__(self, "s2", _finite_float(s2, "s2"))
        if not isinstance(
            sufficient_statistics,
            (VarProSufficientStatistics, NotApplicable),
        ):
            raise TypeError("sufficient_statistics must be immutable diagnostics")
        if not isinstance(condition, (float, int, ConditionStatus, NotApplicable)):
            raise TypeError("condition must be numeric or a typed status")
        if isinstance(condition, (float, int)):
            condition = _finite_float(condition, "condition")
        for name, record in (
            ("unit_objective", unit_objective),
            ("fitted_objective", fitted_objective),
        ):
            if not isinstance(record, (float, int, NotApplicable)):
                raise TypeError(f"{name} must be numeric or NotApplicable")
            if isinstance(record, (float, int)):
                _finite_float(record, name)
        if count_metrics is not None and not isinstance(
            count_metrics, (FittedCountMetrics, NotApplicable, NotEvaluated)
        ):
            raise TypeError("count_metrics must be immutable diagnostics")
        if effective_precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("effective_precision is not a supported precision")
        object.__setattr__(self, "sufficient_statistics", sufficient_statistics)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "unit_objective", unit_objective)
        object.__setattr__(self, "fitted_objective", fitted_objective)
        object.__setattr__(self, "scale_profile", str(scale_profile))
        object.__setattr__(self, "mask_digest", str(mask_digest))
        anchor_json = json.dumps(
            _json_safe(canvas_anchor),
            sort_keys=True,
            separators=(",", ":"),
        )
        weights_snapshot = _snapshot_tensor(canvas_weights)
        _require_finite_array(weights_snapshot, "canvas_weights")
        weights_array = weights_snapshot.numpy()
        weights_metadata = array_metadata(weights_array)
        object.__setattr__(self, "_canvas_anchor_json", anchor_json)
        object.__setattr__(self, "_canvas_weights_data", weights_array.tobytes(order="C"))
        object.__setattr__(
            self,
            "_canvas_weights_shape",
            tuple(int(size) for size in weights_array.shape),
        )
        object.__setattr__(self, "_canvas_weights_dtype", str(weights_array.dtype))
        object.__setattr__(
            self,
            "_canvas_weights_sha256",
            weights_metadata["sha256"],
        )
        object.__setattr__(self, "accepted_patches", accepted)
        object.__setattr__(self, "total_patches", total)
        object.__setattr__(self, "used_scan_ids", used_scans)
        object.__setattr__(self, "used_center_scan_ids", used_centers)
        object.__setattr__(self, "center_identity_available", center_identity_available)
        object.__setattr__(self, "expected_scan_ids", expected_scans)
        object.__setattr__(self, "filtered_eligible_scan_ids", filtered_scans)
        object.__setattr__(
            self, "decoder_real_saturation_fraction", saturation_values[0]
        )
        object.__setattr__(
            self, "decoder_imag_saturation_fraction", saturation_values[1]
        )
        object.__setattr__(self, "decoder_real_lower_saturation_fraction", saturation_values[2])
        object.__setattr__(self, "decoder_real_upper_saturation_fraction", saturation_values[3])
        object.__setattr__(self, "decoder_imag_lower_saturation_fraction", saturation_values[4])
        object.__setattr__(self, "decoder_imag_upper_saturation_fraction", saturation_values[5])
        object.__setattr__(self, "count_metrics", count_metrics)
        object.__setattr__(self, "effective_precision", effective_precision)

    @property
    def canvas_weights(self) -> torch.Tensor:
        array = np.frombuffer(
            self._canvas_weights_data,
            dtype=np.dtype(self._canvas_weights_dtype),
        ).reshape(self._canvas_weights_shape)
        return torch.from_numpy(array.copy())

    @property
    def canvas_anchor(self) -> Mapping[str, Any]:
        return json.loads(self._canvas_anchor_json)

    @property
    def canvas_weights_metadata(self) -> dict[str, Any]:
        return {
            "shape": list(self._canvas_weights_shape),
            "dtype": self._canvas_weights_dtype,
            "sha256": self._canvas_weights_sha256,
        }

    @property
    def profile(self) -> str:
        return self.scale_profile

    @property
    def condition_number(self) -> ConditionRecord:
        return self.condition

    @property
    def timing(self) -> dict[str, float]:
        return {
            "inference_time": self.inference_time,
            "assembly_time": self.assembly_time,
            "solve_time": self.solve_time,
        }

    @property
    def patches_accepted(self) -> int:
        return self.accepted_patches

    @property
    def patches_total(self) -> int:
        return self.total_patches

    @classmethod
    def from_statistics(
        cls,
        statistics: VarProSufficientStatistics,
        **kwargs: Any,
    ) -> "ReassemblyDiagnostics":
        profile = kwargs.pop("profile", None)
        scale_profile = kwargs.get("scale_profile")
        if profile is not None and scale_profile is not None and profile != scale_profile:
            raise ValueError("profile and scale_profile must agree")
        if profile is not None:
            kwargs["scale_profile"] = profile
        s1 = kwargs["s1"]
        s2 = kwargs["s2"]
        unit_objective = statistics.objective(1.0, 1.0)
        fitted_objective = statistics.objective(s1, s2)
        if not math.isfinite(unit_objective) or not math.isfinite(fitted_objective):
            raise ValueError("unit and fitted objectives must be finite")
        tolerance = 1e-12 + 1e-10 * abs(unit_objective)
        if fitted_objective > unit_objective + tolerance:
            raise ValueError(
                "fitted objective exceeds the unit objective contract: "
                f"{fitted_objective} > {unit_objective} + {tolerance}"
            )
        mask = kwargs.pop("effective_probe_mask")
        _require_finite_array(mask, "effective_probe_mask")
        return cls(
            sufficient_statistics=statistics,
            condition=statistics.condition_record(),
            unit_objective=unit_objective,
            fitted_objective=fitted_objective,
            mask_digest=array_digest(mask),
            **kwargs,
        )

    @classmethod
    def legacy_not_applicable(
        cls,
        *,
        effective_probe_mask: torch.Tensor,
        **kwargs: Any,
    ) -> "ReassemblyDiagnostics":
        marker = not_applicable()
        _require_finite_array(effective_probe_mask, "effective_probe_mask")
        return cls(
            sufficient_statistics=marker,
            condition=marker,
            unit_objective=marker,
            fitted_objective=marker,
            mask_digest=array_digest(effective_probe_mask),
            **kwargs,
        )

    def to_jsonable(self) -> dict[str, Any]:
        statistics = self.sufficient_statistics
        if isinstance(statistics, VarProSufficientStatistics):
            statistics_payload = statistics.to_jsonable()
        else:
            statistics_payload = statistics.to_jsonable()
        count_metrics = self.count_metrics
        if isinstance(count_metrics, FittedCountMetrics):
            count_payload = count_metrics.to_jsonable()
        elif isinstance(count_metrics, (NotApplicable, NotEvaluated)):
            count_payload = count_metrics.to_jsonable()
        else:
            count_payload = _json_safe(count_metrics)
        return {
            "schema_version": self.schema_version,
            "inference_time": self.inference_time,
            "assembly_time": self.assembly_time,
            "solve_time": self.solve_time,
            "s1": self.s1,
            "s2": self.s2,
            "sufficient_statistics": statistics_payload,
            "condition": (
                self.condition.to_jsonable()
                if isinstance(self.condition, (ConditionStatus, NotApplicable))
                else self.condition
            ),
            "unit_objective": (
                self.unit_objective.to_jsonable()
                if isinstance(self.unit_objective, NotApplicable)
                else self.unit_objective
            ),
            "fitted_objective": (
                self.fitted_objective.to_jsonable()
                if isinstance(self.fitted_objective, NotApplicable)
                else self.fitted_objective
            ),
            "scale_profile": self.scale_profile,
            "mask_digest": self.mask_digest,
            "canvas_anchor": self.canvas_anchor,
            "canvas_weights": self.canvas_weights_metadata,
            "accepted_patches": self.accepted_patches,
            "total_patches": self.total_patches,
            "used_scan_ids": list(self.used_scan_ids),
            "used_center_scan_ids": list(self.used_center_scan_ids),
            "center_identity_available": self.center_identity_available,
            "expected_scan_ids": list(self.expected_scan_ids),
            "filtered_eligible_scan_ids": list(self.filtered_eligible_scan_ids),
            "decoder_real_saturation_fraction": self.decoder_real_saturation_fraction,
            "decoder_imag_saturation_fraction": self.decoder_imag_saturation_fraction,
            "decoder_real_lower_saturation_fraction": self.decoder_real_lower_saturation_fraction,
            "decoder_real_upper_saturation_fraction": self.decoder_real_upper_saturation_fraction,
            "decoder_imag_lower_saturation_fraction": self.decoder_imag_lower_saturation_fraction,
            "decoder_imag_upper_saturation_fraction": self.decoder_imag_upper_saturation_fraction,
            "count_metrics": count_payload,
            "effective_precision": self.effective_precision,
        }


__all__ = [
    "ConditionStatus",
    "FittedCountMetrics",
    "NotApplicable",
    "NotEvaluated",
    "ReassemblyDiagnostics",
    "VarProSufficientStatistics",
    "array_digest",
    "array_metadata",
    "not_applicable",
    "not_evaluated",
]
