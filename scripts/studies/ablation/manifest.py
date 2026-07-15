"""Versioned TOML manifests and deterministic ablation matrix expansion.

Type-check with ``mypy --explicit-package-bases scripts/studies/ablation/__init__.py
scripts/studies/ablation/manifest.py`` from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ptycho.config import (
    SimulationConfig,
    simulation_config_from_mapping,
    simulation_config_sha256,
    simulation_config_to_dict,
)

from .dataset_schema import (
    DatasetDescriptor,
    DatasetError,
    parse_checked_dataset_descriptor,
)

if TYPE_CHECKING:
    from .verdicts import IntegrationBridgeRequirement


class ManifestError(ValueError):
    """Raised when a manifest violates the versioned study contract."""


class FrozenDict(Mapping[str, Any]):
    """A recursively immutable mapping with normal mapping equality."""

    __slots__ = ("_data",)
    _data: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        values = {} if values is None else values
        if any(not isinstance(key, str) for key in values):
            raise TypeError("FrozenDict keys must be strings")
        object.__setattr__(
            self,
            "_data",
            MappingProxyType({key: _freeze(value) for key, value in values.items()}),
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")


class _ManifestRaw(dict[str, Any]):
    """Mutable manifest data retaining out-of-band source identity."""

    def __init__(
        self,
        values: dict[str, Any],
        *,
        source_bytes: bytes | None,
        source_stat: tuple[int, int, int, int, int] | None,
    ) -> None:
        super().__init__(values)
        self.source_bytes = source_bytes
        self.source_stat = source_stat


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class DatasetDeclaration:
    id: str
    kind: str
    truth: str
    capabilities: frozenset[str]
    metadata: FrozenDict


@dataclass(frozen=True)
class DimensionValue:
    id: str
    overrides: FrozenDict


@dataclass(frozen=True)
class Dimension:
    name: str
    values: tuple[DimensionValue, ...]


@dataclass(frozen=True)
class MatrixInclude:
    assignment: FrozenDict
    overrides: FrozenDict


@dataclass(frozen=True)
class Gate:
    id: str
    target: FrozenDict
    operator: str
    metric: str | None = None
    aggregation: str | None = None
    threshold: int | float | None = None
    min_successful: int | None = None
    status: str | None = None
    requested: int | None = None
    requires: tuple[str, ...] = ()
    when_dataset_kind: str | None = None
    on_missing_capability: str = "error"


@dataclass(frozen=True)
class Comparison:
    id: str
    left: FrozenDict
    right: FrozenDict
    operator: str
    metric: str
    aggregation: str
    threshold: int | float
    min_pairs: int
    requires: tuple[str, ...] = ()
    when_dataset_kind: str | None = None
    on_missing_capability: str = "error"
    diagnostic: bool = False


@dataclass(frozen=True)
class Diagnostics:
    """Optional exact post-epoch checkpoints evaluated by the canonical runtime."""

    milestones: tuple[int, ...] = ()


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    study_id: str
    seeds: tuple[int, ...]
    output_root: str | None
    require_all_explicit: bool
    diagnostics: Diagnostics
    base_overrides: FrozenDict
    datasets: tuple[DatasetDeclaration, ...]
    dimensions: tuple[Dimension, ...]
    excludes: tuple[FrozenDict, ...]
    includes: tuple[MatrixInclude, ...]
    gates: tuple[Gate, ...]
    comparisons: tuple[Comparison, ...]
    expected_protocol_sha256: str | None
    budget_threshold_contract_locked: bool
    integration_bridge_requirement: IntegrationBridgeRequirement | None
    simulation_config: SimulationConfig | None
    source_bytes: bytes | None
    source_sha256: str | None
    source_stat: tuple[int, int, int, int, int] | None
    canonical_json: str
    _raw: FrozenDict

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-native copy of the parsed TOML document."""
        return _ManifestRaw(
            _thaw(self._raw),
            source_bytes=self.source_bytes,
            source_stat=self.source_stat,
        )


@dataclass(frozen=True)
class LogicalArm:
    id: str
    study_id: str
    dataset_id: str
    dataset: DatasetDeclaration
    dimensions: FrozenDict
    dimension_options: FrozenDict
    overrides: FrozenDict
    output_root: str | None


@dataclass(frozen=True)
class ResolvedRun:
    id: str
    arm_id: str
    study_id: str
    dataset_id: str
    dataset: DatasetDeclaration
    seed: int
    dimensions: FrozenDict
    dimension_options: FrozenDict
    overrides: FrozenDict
    output_root: str | None


class RuleApplicability(str, Enum):
    ACTIVE = "active"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ResolvedGate:
    gate: Gate
    dataset_id: str
    target_arm_id: str
    applicability: RuleApplicability
    reason: str | None = None

    @property
    def id(self) -> str:
        return self.gate.id


@dataclass(frozen=True)
class ResolvedComparison:
    comparison: Comparison
    left_dataset_id: str
    right_dataset_id: str
    left_arm_id: str
    right_arm_id: str
    applicability: RuleApplicability
    reason: str | None = None

    @property
    def id(self) -> str:
        return self.comparison.id

    @property
    def dataset_id(self) -> str | None:
        if self.left_dataset_id == self.right_dataset_id:
            return self.left_dataset_id
        return None


@dataclass(frozen=True)
class ResolvedStudy:
    manifest: Manifest
    arms: tuple[LogicalArm, ...]
    runs: tuple[ResolvedRun, ...]
    gates: tuple[ResolvedGate, ...]
    comparisons: tuple[ResolvedComparison, ...]


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESERVED_SEED_COMPONENT_RE = re.compile(r"^seed-[0-9]+$")
_OVERRIDE_RE = re.compile(
    r"^(?:dataset\.id|(?:data|model|training|inference|execution)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+)$"
)
_DATASET_KINDS = frozenset({"synthetic", "experimental"})
_TRUTH_KINDS = frozenset({"object_truth", "reference_reconstruction", "none"})
_AGGREGATIONS = frozenset({"median", "mean", "cv", "all_successful"})
_GATE_OPERATORS = frozenset({"ge", "le", "finite", "status_count_ge", "manual_review"})
_CAPABILITIES = frozenset(
    {
        "has_object_truth",
        "has_reference",
        "has_physical_probe",
        "supports_count_metrics",
        "supports_dose_sweep",
        "supports_grouping_C",
    }
)
CLAIM_GRADE_DISQUALIFYING_REASONS = (
    "epochs_override",
    "seeds_override",
    "matrix_filter",
    "dataset_override",
    "external_dataset_spec",
    "dirty_checkout",
    "manifest_budget_mismatch",
    "fixture_dataset",
    "integration_bridge_prerequisite",
    "unlocked_budget_threshold_contract",
)
_CLAIM_GRADE_SEEDS = (3, 17, 29)
_CLAIM_GRADE_OBJECT_FAMILIES = frozenset({"deadleaves", "lines"})
_CLAIM_GRADE_ARMS = frozenset(
    {
        ("hybrid_resnet", "ci_nll"),
        ("cnn", "ci_nll"),
        ("hybrid_resnet", "legacy_nll"),
        ("hybrid_resnet", "legacy_mae"),
        ("cnn", "legacy_nll"),
        ("cnn", "legacy_mae"),
    }
)

_IMAGE_METRICS = {
    "amp_mean_ratio",
    "amp_quantile_ratio_p05",
    "amp_quantile_ratio_p50",
    "amp_quantile_ratio_p95",
    "amp_pearson",
    "amp_ssim",
    "amp_ms_ssim",
    "phase_ssim",
    "phase_ms_ssim",
    "amp_frc50",
    "amp_frc1over7",
    "phase_frc50",
    "phase_frc1over7",
    "amp_frc_curve",
    "phase_frc_curve",
    "phase_wrapped_mae",
    "patch_amp_pearson",
}
METRIC_PATHS = frozenset(
    {
        *(f"truth_quality.{name}" for name in _IMAGE_METRICS),
        "truth_quality.absolute_amp_mae",
        "truth_quality.absolute_amp_nrmse",
        "truth_quality.absolute_complex_nrmse",
        *(f"reference_agreement.{name}" for name in _IMAGE_METRICS),
        "measurement_consistency.relative_l2_intensity_error",
        "measurement_consistency.mean_raw_poisson_nll",
        "measurement_consistency.poisson_oracle_relative_l2_error",
        "measurement_consistency.model_to_poisson_oracle_error_ratio",
        "measurement_consistency.varpro.s1",
        "measurement_consistency.varpro.s2",
        "measurement_consistency.varpro.condition",
        "measurement_consistency.varpro.unit_objective",
        "measurement_consistency.varpro.fitted_objective",
        "measurement_consistency.dose.object_scale",
        "measurement_consistency.dose.object_scale_cv",
        "stability.finite",
        "stability.loss_final",
        "stability.loss_all_finite",
        "stability.gradient_norm_mean",
        "stability.gradient_norm_median",
        "stability.gradient_norm_p99",
        "stability.gradient_norm_max",
        "stability.gradient_norm_final",
        "stability.gradient_norm_all_finite",
        "stability.clip_fraction",
        "stability.amp_variance",
        "stability.phase_variance",
        "stability.amp_dynamic_range",
        "stability.phase_dynamic_range",
        "stability.real_head_saturation_fraction",
        "stability.imag_head_saturation_fraction",
        "stability.real_head_lower_saturation_fraction",
        "stability.real_head_upper_saturation_fraction",
        "stability.imag_head_lower_saturation_fraction",
        "stability.imag_head_upper_saturation_fraction",
        "stability.loss_tail_relative_improvement",
        "stability.loss_tail_normalized_slope",
        "stability.budget_boundary_improving",
        "stability.validation_loss_final",
        "stability.validation_loss_all_finite",
        "stability.validation_loss_tail_relative_improvement",
        "stability.validation_loss_tail_normalized_slope",
        "stability.validation_budget_boundary_improving",
        "stability.learning_rate_initial",
        "stability.learning_rate_final",
        "stability.learning_rate_min",
        "stability.learning_rate_reduction_count",
        "stability.checkpoint_identity_present",
        "stability.checkpoint_epoch",
        "stability.scan_utilization_fraction",
        "stability.unique_scans_used",
        "stability.unique_centers_used",
        "stability.unique_scans_expected",
        "stability.unique_scans_filtered_eligible",
        "stability.filtered_scan_utilization_fraction",
        "stability.cross_patch_cv",
        "stability.spatial_gradient_energy",
        "stability.reload_max_abs_error",
        "stability.reload_allclose",
        "stability.patches_accepted",
        "stability.patches_total",
        "stability.coverage_fraction",
        "runtime.train_seconds",
        "runtime.inference_seconds",
        "runtime.assembly_seconds",
        "runtime.peak_memory_bytes",
        *(f"truth_quality.pre_varpro.{name}" for name in (
            "amp_pearson", "amp_ssim", "phase_ssim", "phase_wrapped_mae"
        )),
        *(f"truth_quality.post_varpro.{name}" for name in (
            "amp_pearson", "amp_ssim", "phase_ssim", "phase_wrapped_mae"
        )),
    }
)
# Compatibility alias for callers predating the public registry name.
_METRIC_PATHS = METRIC_PATHS


def claim_grade_eligibility(
    manifest: Manifest,
    full_runs: Iterable[ResolvedRun],
    selected_runs: Iterable[ResolvedRun],
    *,
    epochs_override: bool,
    seeds_override: bool,
    matrix_filter: bool,
    dataset_override: bool,
    external_dataset_spec: bool,
    dirty_checkout: bool,
    resolved_run_configs: Mapping[str, Any] | None = None,
    dataset_profiles: Mapping[str, str | None] | None = None,
    integration_bridge_evidence: Any | None = None,
    integration_bridge_locked_ssim_floors: Any | None = None,
    extra_reasons: Iterable[str] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Return canonical protocol eligibility and closed ordered reasons."""
    full_runs = tuple(full_runs)
    selected_runs = tuple(selected_runs)
    profiles = _normalized_dataset_profiles(dataset_profiles, selected_runs)
    flags = {
        "epochs_override": epochs_override,
        "seeds_override": seeds_override,
        "matrix_filter": matrix_filter,
        "dataset_override": dataset_override,
        "external_dataset_spec": external_dataset_spec,
        "dirty_checkout": dirty_checkout,
    }
    supplied = tuple(extra_reasons)
    unknown = set(supplied) - set(CLAIM_GRADE_DISQUALIFYING_REASONS)
    if unknown:
        raise ManifestError(
            "unknown claim-grade disqualifying reasons: " + ", ".join(sorted(unknown))
        )
    full_ids = tuple(run.id for run in full_runs)
    selected_ids = tuple(run.id for run in selected_runs)
    canonical_run_keys = {
        (
            run.dimensions.get("object_family"),
            run.dimensions.get("architecture"),
            run.dimensions.get("physics_profile"),
            run.seed,
        )
        for run in full_runs
    }
    expected_run_keys = {
        (family, architecture, profile, seed)
        for family in _CLAIM_GRADE_OBJECT_FAMILIES
        for architecture, profile in _CLAIM_GRADE_ARMS
        for seed in _CLAIM_GRADE_SEEDS
    }
    protocol_matches = False
    if resolved_run_configs is not None and manifest.expected_protocol_sha256:
        protocol_matches = (
            protocol_fingerprint(
                manifest,
                full_runs,
                resolved_run_configs,
                dataset_profiles=None if dataset_profiles is None else profiles,
            )
            == manifest.expected_protocol_sha256
        )
    budget_mismatch = (
        manifest.seeds != _CLAIM_GRADE_SEEDS
        or len(full_ids) != 36
        or len(set(full_ids)) != 36
        or canonical_run_keys != expected_run_keys
        or selected_ids != full_ids
        or not protocol_matches
    )
    present = set(supplied)
    present.update(reason for reason, active in flags.items() if active)
    if budget_mismatch:
        present.add("manifest_budget_mismatch")
    if "fixture" in profiles.values():
        present.add("fixture_dataset")
    if manifest.integration_bridge_requirement is not None:
        from .verdicts import (
            IntegrationBridgeEvidence,
            Verdict,
            evaluate_integration_bridge,
        )

        bridge_passed = False
        if isinstance(integration_bridge_evidence, IntegrationBridgeEvidence):
            bridge_passed = (
                evaluate_integration_bridge(
                    manifest.integration_bridge_requirement,
                    integration_bridge_evidence,
                    locked_ssim_floors=integration_bridge_locked_ssim_floors,
                ).verdict
                is Verdict.PASS
            )
        if not bridge_passed:
            present.add("integration_bridge_prerequisite")
    if not manifest.budget_threshold_contract_locked:
        present.add("unlocked_budget_threshold_contract")
    reasons = tuple(
        reason for reason in CLAIM_GRADE_DISQUALIFYING_REASONS if reason in present
    )
    return not reasons, reasons


def protocol_fingerprint(
    manifest: Manifest,
    runs: Iterable[ResolvedRun],
    resolved_run_configs: Mapping[str, Any],
    *,
    dataset_profiles: Mapping[str, str | None] | None = None,
) -> str:
    """Hash the generic fully resolved study protocol, excluding its declaration."""
    runs = tuple(runs)
    run_ids = {run.id for run in runs}
    if set(resolved_run_configs) != run_ids:
        raise ManifestError(
            "resolved protocol configs must cover exactly the expanded run ids"
        )
    configs = {run_id: _thaw(config) for run_id, config in resolved_run_configs.items()}
    _validate_json_native(configs, path="resolved_protocol_configs")
    profiles = _normalized_dataset_profiles(dataset_profiles, runs, require_exact=False)
    payload = {
        "schema_version": "ablation_claim_grade_protocol_v3",
        "study_id": manifest.study_id,
        "seeds": list(manifest.seeds),
        "epochs": manifest.base_overrides.get("training.epochs"),
        "dimensions": [
            {
                "name": dimension.name,
                "values": [value.id for value in dimension.values],
            }
            for dimension in manifest.dimensions
        ],
        "gates": [
            {
                "id": gate.id,
                "target": _thaw(gate.target),
                "operator": gate.operator,
                "metric": gate.metric,
                "aggregation": gate.aggregation,
                "threshold": gate.threshold,
                "min_successful": gate.min_successful,
                "status": gate.status,
                "requested": gate.requested,
                "requires": list(gate.requires),
                "when_dataset_kind": gate.when_dataset_kind,
                "on_missing_capability": gate.on_missing_capability,
            }
            for gate in manifest.gates
        ],
        "comparisons": [
            {
                "id": comparison.id,
                "left": _thaw(comparison.left),
                "right": _thaw(comparison.right),
                "operator": comparison.operator,
                "metric": comparison.metric,
                "aggregation": comparison.aggregation,
                "threshold": comparison.threshold,
                "min_pairs": comparison.min_pairs,
                "requires": list(comparison.requires),
                "when_dataset_kind": comparison.when_dataset_kind,
                "on_missing_capability": comparison.on_missing_capability,
                "diagnostic": comparison.diagnostic,
            }
            for comparison in manifest.comparisons
        ],
        "integration_bridge_requirement": (
            None
            if manifest.integration_bridge_requirement is None
            else manifest.integration_bridge_requirement.to_mapping()
        ),
        "runs": [
            {
                "id": run.id,
                "arm_id": run.arm_id,
                "dataset_id": run.dataset_id,
                "seed": run.seed,
                "dimensions": _thaw(run.dimensions),
                "resolved_config": configs[run.id],
                "dataset": {
                    "id": run.dataset.id,
                    "kind": run.dataset.kind,
                    "truth": run.dataset.truth,
                    "metadata": _thaw(run.dataset.metadata),
                },
            }
            for run in runs
        ],
    }
    if manifest.schema_version == 2:
        if manifest.simulation_config is None:
            raise ManifestError("schema-v2 manifest is missing simulation_config")
        payload["simulation"] = simulation_config_to_dict(
            manifest.simulation_config
        )
    if dataset_profiles is not None:
        payload["dataset_materialization_profiles"] = profiles
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def claim_grade_protocol_fingerprints(
    manifest: Manifest,
    runs: Iterable[ResolvedRun],
    resolved_run_configs: Mapping[str, Any],
    *,
    dataset_profiles: Mapping[str, str | None] | None = None,
) -> tuple[str, str | None]:
    """Return the actual and declared expected protocol fingerprints."""
    return (
        protocol_fingerprint(
            manifest,
            runs,
            resolved_run_configs,
            dataset_profiles=dataset_profiles,
        ),
        manifest.expected_protocol_sha256,
    )


def _normalized_dataset_profiles(
    value: Mapping[str, str | None] | None,
    runs: Iterable[ResolvedRun],
    *,
    require_exact: bool = True,
) -> dict[str, str | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError("dataset materialization profiles must be a mapping")
    selected_ids = {run.dataset_id for run in runs}
    if (require_exact and set(value) != selected_ids) or not set(value) <= selected_ids:
        raise ManifestError(
            "dataset materialization profiles must match the selected dataset ids"
        )
    profiles = {dataset_id: value[dataset_id] for dataset_id in sorted(value)}
    if any(
        profile not in {None, "claim_grade", "fixture"} for profile in profiles.values()
    ):
        raise ManifestError(
            "dataset materialization profile must be claim_grade, fixture, or null"
        )
    return profiles


def _closed_table(
    value: Any, *, path: str, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    missing = required - set(value)
    if missing:
        raise ManifestError(
            f"{path} is missing required fields: {', '.join(sorted(missing))}"
        )
    return value


def _list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{path} must be an array")
    return value


def _identifier(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ManifestError(f"{path} must be a nonempty path-safe string id")
    return value


def _id_component(value: Any, *, path: str) -> str:
    identifier = _identifier(value, path=path)
    if "--" in identifier or _RESERVED_SEED_COMPONENT_RE.fullmatch(identifier):
        raise ManifestError(f"{path} contains a reserved arm/run id component pattern")
    return identifier


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ManifestError(f"{path} must be a nonnegative integer")
    return value


def _number(value: Any, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{path} must be numeric")
    if not math.isfinite(value):
        raise ManifestError(f"{path} must be finite")
    return value


def _json_scalar(value: Any, *, path: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(f"{path} must be finite")
        return value
    raise ManifestError(f"{path} must be a JSON-native scalar")


def _override_value(key: str, value: Any, *, path: str) -> Any:
    """Parse the structured runtime overrides owned by manifest v1."""
    if key not in {"data.grid_size", "data.x_bounds", "data.y_bounds"}:
        return _json_scalar(value, path=path)
    if not isinstance(value, list) or len(value) != 2:
        raise ManifestError(f"{path} must be a two-item positive-integer array")
    if key == "data.grid_size":
        if any(type(item) is not int or item <= 0 for item in value):
            raise ManifestError(f"{path} must be a two-item positive-integer array")
        return list(value)
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ManifestError(f"{path} must be a two-item finite numeric array")
    if any(not math.isfinite(item) for item in value):
        raise ManifestError(f"{path} must be a two-item finite numeric array")
    return list(value)


def _json_scalars_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        left, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) == json.dumps(right, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _parse_overrides(value: Any, *, path: str) -> FrozenDict:
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must be a table")
    parsed: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not _OVERRIDE_RE.fullmatch(key):
            raise ManifestError(
                f"{path} has unknown or malformed override field {key!r}"
            )
        parsed_value = _override_value(key, item, path=f"{path}.{key}")
        if key == "dataset.id" and not isinstance(parsed_value, str):
            raise ManifestError(f"{path}.dataset.id must be a string id")
        parsed[key] = parsed_value
    return FrozenDict(parsed)


def _validate_json_native(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError(f"{path} contains a non-string key")
            _validate_json_native(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_native(item, path=f"{path}[{index}]")
        return
    _json_scalar(value, path=path)


def _stat_identity(value: Any) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load a schema-v1/v2 study manifest using Python 3.11 ``tomllib``."""
    source = Path(path)
    try:
        before = source.stat()
        data = source.read_bytes()
        after = source.stat()
        if _stat_identity(before) != _stat_identity(after):
            raise ManifestError(f"manifest changed while being read: {path}")
        raw = tomllib.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"could not load TOML manifest {path}: {exc}") from exc
    return _parse_manifest(raw, source_bytes=data, source_stat=_stat_identity(after))


def loads_manifest(text: str) -> Manifest:
    """Parse a schema-v1/v2 manifest from TOML text."""
    if not isinstance(text, str):
        raise TypeError("manifest text must be str")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"could not parse TOML manifest: {exc}") from exc
    data = text.encode("utf-8")
    return _parse_manifest(raw, source_bytes=data)


def _parse_manifest(
    raw: dict[str, Any],
    *,
    dataset_descriptors: Mapping[str, DatasetDescriptor] | None = None,
    source_bytes: bytes | None = None,
    source_stat: tuple[int, int, int, int, int] | None = None,
) -> Manifest:
    if source_bytes is None and isinstance(raw, _ManifestRaw):
        source_bytes = raw.source_bytes
        source_stat = raw.source_stat
    _validate_json_native(raw)
    if not isinstance(raw, dict) or "schema" not in raw:
        raise ManifestError("manifest must contain a schema table")
    schema = _closed_table(
        raw["schema"], path="schema", allowed={"version"}, required={"version"}
    )
    version = schema["version"]
    if type(version) is not int or version not in {1, 2}:
        raise ManifestError("schema.version must be the integer 1 or 2")
    allowed_top_level = {
        "schema",
        "study",
        "base",
        "datasets",
        "matrix",
        "gates",
        "comparisons",
        "diagnostics",
        "claim_grade",
    }
    required_top_level = {"schema", "study", "datasets", "matrix"}
    if version == 2:
        allowed_top_level.add("simulation")
        required_top_level.add("simulation")
    _closed_table(
        raw,
        path="manifest",
        allowed=allowed_top_level,
        required=required_top_level,
    )
    simulation_config = None
    if version == 2:
        try:
            simulation_config = simulation_config_from_mapping(raw["simulation"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"simulation is invalid: {exc}") from exc

    claim_grade = _closed_table(
        raw.get("claim_grade", {}),
        path="claim_grade",
        allowed={
            "expected_protocol_sha256",
            "budget_threshold_contract_locked",
            "integration_bridge",
        },
        required=set(),
    )
    expected_protocol_sha256 = claim_grade.get("expected_protocol_sha256")
    budget_threshold_contract_locked = claim_grade.get(
        "budget_threshold_contract_locked", False
    )
    if type(budget_threshold_contract_locked) is not bool:
        raise ManifestError(
            "claim_grade.budget_threshold_contract_locked must be a boolean"
        )
    if expected_protocol_sha256 is not None and (
        not isinstance(expected_protocol_sha256, str)
        or len(expected_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_protocol_sha256
        )
    ):
        raise ManifestError(
            "claim_grade.expected_protocol_sha256 must be a lowercase SHA-256"
        )
    integration_bridge_requirement = None
    if "integration_bridge" in claim_grade:
        from .verdicts import IntegrationBridgeRequirement

        bridge_payload = claim_grade["integration_bridge"]
        if not isinstance(bridge_payload, dict):
            raise ManifestError("claim_grade.integration_bridge must be a table")
        bridge_payload = dict(bridge_payload)
        bridge_payload.setdefault("probe_mask_diameter", None)
        bridge_payload.setdefault("ssim_floor_reference_artifact_sha256", None)
        try:
            integration_bridge_requirement = (
                IntegrationBridgeRequirement.from_mapping(bridge_payload)
            )
        except ValueError as error:
            raise ManifestError(
                f"claim_grade.integration_bridge is invalid: {error}"
            ) from error

    study = _closed_table(
        raw["study"],
        path="study",
        allowed={"id", "seeds", "output_root", "require_all_explicit"},
        required={"id", "seeds"},
    )
    study_id = _id_component(study["id"], path="study.id")
    seeds = tuple(
        _nonnegative_int(seed, path="study.seeds[]")
        for seed in _list(study["seeds"], path="study.seeds")
    )
    if not seeds:
        raise ManifestError("study.seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ManifestError("study.seeds would create duplicate run ids")
    output_root = study.get("output_root")
    if output_root is not None and (
        not isinstance(output_root, str) or not output_root
    ):
        raise ManifestError("study.output_root must be a nonempty string")
    require_all_explicit = study.get("require_all_explicit", False)
    if type(require_all_explicit) is not bool:
        raise ManifestError("study.require_all_explicit must be a boolean")

    diagnostics_raw = _closed_table(
        raw.get("diagnostics", {}),
        path="diagnostics",
        allowed={"milestones"},
        required={"milestones"} if "diagnostics" in raw else set(),
    )
    milestones = tuple(
        _positive_int(epoch, path="diagnostics.milestones[]")
        for epoch in _list(
            diagnostics_raw.get("milestones", []),
            path="diagnostics.milestones",
        )
    )
    if "diagnostics" in raw and not milestones:
        raise ManifestError("diagnostics.milestones must not be empty")
    if tuple(sorted(set(milestones))) != milestones:
        raise ManifestError("diagnostics.milestones must be strictly increasing")
    diagnostics = Diagnostics(milestones)

    base_raw = _closed_table(
        raw.get("base", {}), path="base", allowed={"overrides"}, required=set()
    )
    base_overrides = _parse_overrides(
        base_raw.get("overrides", {}), path="base.overrides"
    )
    training_epochs = base_overrides.get("training.epochs")
    if milestones and type(training_epochs) is int and milestones[-1] > training_epochs:
        raise ManifestError("diagnostics.milestones must not exceed training.epochs")
    datasets = _parse_datasets(
        raw["datasets"],
        parsed_descriptors=dataset_descriptors,
        simulation_digest=(
            None
            if simulation_config is None
            else simulation_config_sha256(simulation_config)
        ),
    )
    dataset_ids = {dataset.id for dataset in datasets}

    matrix = _closed_table(
        raw["matrix"],
        path="matrix",
        allowed={"dimensions", "exclude", "include"},
        required={"dimensions"},
    )
    dimensions = _parse_dimensions(matrix["dimensions"], dataset_ids=dataset_ids)
    dimension_values = {
        dim.name: {value.id for value in dim.values} for dim in dimensions
    }
    excludes = tuple(
        _parse_selector(
            item,
            path=f"matrix.exclude[{index}]",
            dimension_values=dimension_values,
            dataset_ids=dataset_ids,
            allow_dataset=False,
        )
        for index, item in enumerate(
            _list(matrix.get("exclude", []), path="matrix.exclude")
        )
    )
    includes = _parse_includes(matrix.get("include", []), dimensions, dataset_ids)
    _validate_declared_dataset_overrides(
        base_overrides, dimensions, includes, dataset_ids
    )

    gates = tuple(
        _parse_gate(
            item,
            index=index,
            dimension_values=dimension_values,
            dataset_ids=dataset_ids,
        )
        for index, item in enumerate(_list(raw.get("gates", []), path="gates"))
    )
    comparisons = tuple(
        _parse_comparison(
            item,
            index=index,
            dimension_values=dimension_values,
            dataset_ids=dataset_ids,
        )
        for index, item in enumerate(
            _list(raw.get("comparisons", []), path="comparisons")
        )
    )
    rule_ids: list[str] = [gate.id for gate in gates]
    rule_ids.extend(comparison.id for comparison in comparisons)
    duplicate_rule = next(
        (rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1), None
    )
    if duplicate_rule is not None:
        raise ManifestError(f"duplicate gate/comparison id {duplicate_rule!r}")

    canonical = json.dumps(
        raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return Manifest(
        schema_version=version,
        study_id=study_id,
        seeds=seeds,
        output_root=output_root,
        require_all_explicit=require_all_explicit,
        diagnostics=diagnostics,
        base_overrides=base_overrides,
        datasets=datasets,
        dimensions=dimensions,
        excludes=excludes,
        includes=includes,
        gates=gates,
        comparisons=comparisons,
        expected_protocol_sha256=expected_protocol_sha256,
        budget_threshold_contract_locked=budget_threshold_contract_locked,
        integration_bridge_requirement=integration_bridge_requirement,
        simulation_config=simulation_config,
        source_bytes=source_bytes,
        source_sha256=(
            None if source_bytes is None else hashlib.sha256(source_bytes).hexdigest()
        ),
        source_stat=source_stat,
        canonical_json=canonical,
        _raw=FrozenDict(raw),
    )


def _parse_datasets(
    value: Any,
    *,
    parsed_descriptors: Mapping[str, DatasetDescriptor] | None = None,
    simulation_digest: str | None = None,
) -> tuple[DatasetDeclaration, ...]:
    if not isinstance(value, dict) or not value:
        raise ManifestError("datasets must be a nonempty table")
    supplied = {} if parsed_descriptors is None else dict(parsed_descriptors)
    unknown_supplied = set(supplied) - set(value)
    if unknown_supplied:
        raise ManifestError(
            f"parsed dataset descriptors are not declared: {sorted(unknown_supplied)}"
        )
    datasets: list[DatasetDeclaration] = []
    synthetic_count = 0
    for dataset_id, metadata in value.items():
        identifier = _id_component(dataset_id, path="datasets id")
        if not isinstance(metadata, dict):
            raise ManifestError(f"datasets.{identifier} must be a table")
        descriptor = supplied.get(identifier)
        if descriptor is None:
            try:
                descriptor = parse_checked_dataset_descriptor(
                    identifier,
                    metadata,
                    repo_root=Path.cwd(),
                )
            except DatasetError as exc:
                raise ManifestError(f"datasets.{identifier}: {exc}") from exc
        elif descriptor.id != identifier:
            raise ManifestError(
                f"datasets.{identifier}: parsed descriptor id {descriptor.id!r} does not match"
            )
        kind = descriptor.kind
        truth = descriptor.truth
        if kind == "synthetic":
            synthetic_count += 1
            if simulation_digest is not None:
                expected_parent = f"simulation-{simulation_digest}"
                for field in ("train", "test"):
                    declaration = getattr(descriptor.path_declarations, field)
                    if Path(declaration).parent.name != expected_parent:
                        raise ManifestError(
                            f"datasets.{identifier}.{field} must be directly under "
                            f"{expected_parent!r}, derived from simulation_config_sha256"
                        )
        capabilities: set[str] = set()
        if truth == "object_truth":
            capabilities.add("has_object_truth")
        elif truth == "reference_reconstruction":
            capabilities.add("has_reference")
        if (
            descriptor.measurement_domain == "count_intensity"
            and descriptor.probe.calibration == "count_amplitude"
            and descriptor.probe.gauge == "physical_count_amplitude"
        ):
            # Dry-run may route only declared capabilities. Content preflight still
            # verifies physical probe arrays and calibrated count measurements.
            capabilities.update({"has_physical_probe", "supports_count_metrics"})
        datasets.append(
            DatasetDeclaration(
                id=identifier,
                kind=kind,
                truth=truth,
                capabilities=frozenset(capabilities),
                metadata=FrozenDict(metadata),
            )
        )
    if simulation_digest is not None and synthetic_count == 0:
        raise ManifestError(
            "schema-v2 manifest requires at least one synthetic dataset to bind "
            "to simulation_config_sha256"
        )
    return tuple(datasets)


def _parse_dimensions(value: Any, *, dataset_ids: set[str]) -> tuple[Dimension, ...]:
    dimensions_raw = _list(value, path="matrix.dimensions")
    if not dimensions_raw:
        raise ManifestError("matrix.dimensions must not be empty")
    dimensions: list[Dimension] = []
    dimension_names: set[str] = set()
    value_ids: set[str] = set()
    for dim_index, raw_dimension in enumerate(dimensions_raw):
        table = _closed_table(
            raw_dimension,
            path=f"matrix.dimensions[{dim_index}]",
            allowed={"name", "values"},
            required={"name", "values"},
        )
        name = _id_component(table["name"], path=f"matrix.dimensions[{dim_index}].name")
        if name in dimension_names:
            raise ManifestError(f"duplicate dimension id {name!r}")
        dimension_names.add(name)
        values: list[DimensionValue] = []
        raw_values = _list(
            table["values"], path=f"matrix.dimensions[{dim_index}].values"
        )
        if not raw_values:
            raise ManifestError(f"dimension {name!r} must declare values")
        for value_index, raw_value in enumerate(raw_values):
            value_table = _closed_table(
                raw_value,
                path=f"matrix.dimensions[{dim_index}].values[{value_index}]",
                allowed={"id", "overrides"},
                required={"id"},
            )
            value_id = _id_component(
                value_table["id"], path=f"dimension {name} value id"
            )
            if value_id in value_ids:
                raise ManifestError(f"duplicate dimension value id {value_id!r}")
            value_ids.add(value_id)
            overrides = _parse_overrides(
                value_table.get("overrides", {}),
                path=f"dimension {name}.{value_id}.overrides",
            )
            if name == "dataset":
                if value_id not in dataset_ids:
                    raise ManifestError(
                        f"dataset dimension has unknown dataset value {value_id!r}"
                    )
                overridden = overrides.get("dataset.id", value_id)
                if overridden != value_id:
                    raise ManifestError(
                        "dataset dimension value id must equal its dataset.id override"
                    )
            values.append(DimensionValue(id=value_id, overrides=overrides))
        dimensions.append(Dimension(name=name, values=tuple(values)))
    return tuple(dimensions)


def _parse_selector(
    value: Any,
    *,
    path: str,
    dimension_values: Mapping[str, set[str]],
    dataset_ids: set[str],
    allow_dataset: bool = True,
) -> FrozenDict:
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{path} selector must be a nonempty table")
    parsed: dict[str, str] = {}
    for name, selected in value.items():
        if not isinstance(selected, str):
            raise ManifestError(f"{path} selector values must be strings")
        if name == "dataset" and allow_dataset:
            if selected not in dataset_ids:
                raise ManifestError(f"{path} has unknown dataset value {selected!r}")
        elif name not in dimension_values:
            raise ManifestError(f"{path} has unknown dimension {name!r}")
        elif selected not in dimension_values[name]:
            raise ManifestError(
                f"{path} has unknown value {selected!r} for dimension {name!r}"
            )
        parsed[name] = selected
    return FrozenDict(parsed)


def _parse_includes(
    value: Any, dimensions: tuple[Dimension, ...], dataset_ids: set[str]
) -> tuple[MatrixInclude, ...]:
    dimension_values = {
        dim.name: {item.id for item in dim.values} for dim in dimensions
    }
    expected = set(dimension_values)
    includes: list[MatrixInclude] = []
    for index, item in enumerate(_list(value, path="matrix.include")):
        if not isinstance(item, dict):
            raise ManifestError(f"matrix.include[{index}] must be a table")
        unknown = set(item) - expected - {"overrides"}
        if unknown:
            raise ManifestError(
                f"matrix.include[{index}] has unknown fields: {', '.join(sorted(unknown))}"
            )
        assignment_raw = {key: val for key, val in item.items() if key != "overrides"}
        if set(assignment_raw) != expected:
            raise ManifestError(
                "matrix include must name exactly one value from every dimension"
            )
        assignment = _parse_selector(
            assignment_raw,
            path=f"matrix.include[{index}]",
            dimension_values=dimension_values,
            dataset_ids=dataset_ids,
            allow_dataset=False,
        )
        overrides = _parse_overrides(
            item.get("overrides", {}), path=f"matrix.include[{index}].overrides"
        )
        includes.append(MatrixInclude(assignment=assignment, overrides=overrides))
    return tuple(includes)


def _validate_declared_dataset_overrides(
    base: FrozenDict,
    dimensions: tuple[Dimension, ...],
    includes: tuple[MatrixInclude, ...],
    dataset_ids: set[str],
) -> None:
    sources: list[FrozenDict] = [base]
    sources.extend(value.overrides for dim in dimensions for value in dim.values)
    sources.extend(include.overrides for include in includes)
    for overrides in sources:
        if "dataset.id" in overrides and overrides["dataset.id"] not in dataset_ids:
            raise ManifestError(
                f"override has unknown dataset.id {overrides['dataset.id']!r}"
            )


def _parse_conditions(
    table: dict[str, Any], *, path: str
) -> tuple[tuple[str, ...], str | None, str]:
    requires_raw = table.get("requires", [])
    requires = tuple(_list(requires_raw, path=f"{path}.requires"))
    if any(not isinstance(item, str) or item not in _CAPABILITIES for item in requires):
        raise ManifestError(f"{path}.requires contains an unknown capability")
    if len(set(requires)) != len(requires):
        raise ManifestError(f"{path}.requires contains duplicate capabilities")
    when = table.get("when_dataset_kind")
    if when is not None and (not isinstance(when, str) or when not in _DATASET_KINDS):
        raise ManifestError(
            f"{path}.when_dataset_kind must be synthetic or experimental"
        )
    on_missing = table.get("on_missing_capability", "error")
    if not isinstance(on_missing, str) or on_missing not in {
        "error",
        "not_applicable",
    }:
        raise ManifestError(
            f"{path}.on_missing_capability must be error or not_applicable"
        )
    return requires, when, on_missing


def _parse_gate(
    value: Any,
    *,
    index: int,
    dimension_values: Mapping[str, set[str]],
    dataset_ids: set[str],
) -> Gate:
    path = f"gates[{index}]"
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must be a table")
    operator = value.get("operator")
    if not isinstance(operator, str) or operator not in _GATE_OPERATORS:
        raise ManifestError(
            f"{path}.operator must be one of ge, le, finite, status_count_ge, manual_review"
        )
    common = {
        "id",
        "target",
        "operator",
        "requires",
        "when_dataset_kind",
        "on_missing_capability",
    }
    fields = {
        "ge": (
            {"metric", "aggregation", "threshold", "min_successful"},
            {"metric", "aggregation", "threshold", "min_successful"},
        ),
        "le": (
            {"metric", "aggregation", "threshold", "min_successful"},
            {"metric", "aggregation", "threshold", "min_successful"},
        ),
        "finite": (
            {"metric", "aggregation", "min_successful"},
            {"metric", "min_successful"},
        ),
        "status_count_ge": (
            {"status", "threshold", "requested"},
            {"status", "threshold", "requested"},
        ),
        "manual_review": (set(), set()),
    }
    optional, required_specific = fields[operator]
    table = _closed_table(
        value,
        path=path,
        allowed=common | optional,
        required={"id", "target", "operator"} | required_specific,
    )
    gate_id = _identifier(table["id"], path=f"{path}.id")
    target = _parse_selector(
        table["target"],
        path=f"{path}.target",
        dimension_values=dimension_values,
        dataset_ids=dataset_ids,
    )
    requires, when, on_missing = _parse_conditions(table, path=path)
    metric = table.get("metric")
    if metric is not None and (
        not isinstance(metric, str) or metric not in METRIC_PATHS
    ):
        raise ManifestError(
            f"{path}.metric is not in the closed metric registry: {metric!r}"
        )
    aggregation = table.get("aggregation")
    if aggregation is not None and (
        not isinstance(aggregation, str) or aggregation not in _AGGREGATIONS
    ):
        raise ManifestError(
            f"{path}.aggregation must be median, mean, cv, or all_successful"
        )
    if aggregation == "all_successful" and operator not in {"finite", "ge", "le"}:
        raise ManifestError(
            f"{path}.aggregation all_successful requires finite, ge, or le"
        )
    threshold = table.get("threshold")
    if threshold is not None:
        threshold = _number(threshold, path=f"{path}.threshold")
    min_successful = table.get("min_successful")
    if min_successful is not None:
        min_successful = _positive_int(min_successful, path=f"{path}.min_successful")
    status = table.get("status")
    requested = table.get("requested")
    if operator == "status_count_ge":
        if not isinstance(status, str) or status != "success":
            raise ManifestError(f"{path}.status must equal success")
        requested = _positive_int(requested, path=f"{path}.requested")
        if type(threshold) is not int or threshold < 0 or threshold > requested:
            raise ManifestError(
                f"{path}.threshold must be an integer from zero through requested"
            )
    return Gate(
        id=gate_id,
        target=target,
        operator=operator,
        metric=metric,
        aggregation=aggregation,
        threshold=threshold,
        min_successful=min_successful,
        status=status,
        requested=requested,
        requires=requires,
        when_dataset_kind=when,
        on_missing_capability=on_missing,
    )


def _parse_comparison(
    value: Any,
    *,
    index: int,
    dimension_values: Mapping[str, set[str]],
    dataset_ids: set[str],
) -> Comparison:
    path = f"comparisons[{index}]"
    table = _closed_table(
        value,
        path=path,
        allowed={
            "id",
            "left",
            "right",
            "operator",
            "metric",
            "aggregation",
            "threshold",
            "min_pairs",
            "requires",
            "when_dataset_kind",
            "on_missing_capability",
            "diagnostic",
        },
        required={
            "id",
            "left",
            "right",
            "operator",
            "metric",
            "aggregation",
            "threshold",
            "min_pairs",
        },
    )
    if not isinstance(table["operator"], str) or table["operator"] != "paired_ratio_ge":
        raise ManifestError(f"{path}.operator must equal paired_ratio_ge")
    comparison_id = _identifier(table["id"], path=f"{path}.id")
    left = _parse_selector(
        table["left"],
        path=f"{path}.left",
        dimension_values=dimension_values,
        dataset_ids=dataset_ids,
    )
    right = _parse_selector(
        table["right"],
        path=f"{path}.right",
        dimension_values=dimension_values,
        dataset_ids=dataset_ids,
    )
    metric = table["metric"]
    if not isinstance(metric, str) or metric not in METRIC_PATHS:
        raise ManifestError(
            f"{path}.metric is not in the closed metric registry: {metric!r}"
        )
    aggregation = table["aggregation"]
    if not isinstance(aggregation, str) or aggregation not in _AGGREGATIONS:
        raise ManifestError(f"{path}.aggregation must be median, mean, or cv")
    threshold = _number(table["threshold"], path=f"{path}.threshold")
    min_pairs = _positive_int(table["min_pairs"], path=f"{path}.min_pairs")
    diagnostic = table.get("diagnostic", False)
    if type(diagnostic) is not bool:
        raise ManifestError(f"{path}.diagnostic must be a boolean")
    requires, when, on_missing = _parse_conditions(table, path=path)
    return Comparison(
        id=comparison_id,
        left=left,
        right=right,
        operator="paired_ratio_ge",
        metric=metric,
        aggregation=aggregation,
        threshold=threshold,
        min_pairs=min_pairs,
        requires=requires,
        when_dataset_kind=when,
        on_missing_capability=on_missing,
        diagnostic=diagnostic,
    )


@dataclass(frozen=True)
class _Assignment:
    dimensions: FrozenDict
    values: tuple[DimensionValue, ...]
    include_overrides: FrozenDict


def _cartesian_assignments(manifest: Manifest) -> tuple[_Assignment, ...]:
    assignments: list[_Assignment] = []
    for selected_values in product(
        *(dimension.values for dimension in manifest.dimensions)
    ):
        selected = FrozenDict(
            {
                dimension.name: value.id
                for dimension, value in zip(manifest.dimensions, selected_values)
            }
        )
        assignments.append(_Assignment(selected, tuple(selected_values), FrozenDict()))
    return tuple(assignments)


def _matrix_assignments(manifest: Manifest) -> tuple[_Assignment, ...]:
    assignments = [
        assignment
        for assignment in _cartesian_assignments(manifest)
        if not any(
            all(assignment.dimensions[key] == value for key, value in exclude.items())
            for exclude in manifest.excludes
        )
    ]

    seen = {
        tuple(item.dimensions[dimension.name] for dimension in manifest.dimensions)
        for item in assignments
    }
    values_by_dimension = {
        dimension.name: {value.id: value for value in dimension.values}
        for dimension in manifest.dimensions
    }
    for include in manifest.includes:
        key = tuple(
            include.assignment[dimension.name] for dimension in manifest.dimensions
        )
        if key in seen:
            raise ManifestError(
                f"matrix include creates duplicate resulting assignment {key!r}"
            )
        seen.add(key)
        selected_values = tuple(
            values_by_dimension[dimension.name][include.assignment[dimension.name]]
            for dimension in manifest.dimensions
        )
        assignments.append(
            _Assignment(include.assignment, selected_values, include.overrides)
        )
    return tuple(assignments)


def _merge_assignment(manifest: Manifest, assignment: _Assignment) -> FrozenDict:
    merged = dict(manifest.base_overrides)
    assigned: dict[str, tuple[Any, str]] = {}
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for dimension, value in zip(manifest.dimensions, assignment.values):
        dimension_overrides = dict(value.overrides)
        if dimension.name == "dataset":
            dimension_overrides.setdefault("dataset.id", value.id)
        sources.append((f"dimension {dimension.name}={value.id}", dimension_overrides))
    sources.append(("matrix include", assignment.include_overrides))
    for source, source_overrides in sources:
        for key, value in source_overrides.items():
            if key in assigned:
                original_value, original_source = assigned[key]
                if not _json_scalars_equal(original_value, value):
                    raise ManifestError(
                        f"conflicting non-base override for {key!r}: "
                        f"{original_value!r} from {original_source} versus "
                        f"{value!r} from {source}"
                    )
            else:
                assigned[key] = (value, source)
            merged[key] = value
    return FrozenDict(merged)


def _cli_seeds(
    seeds: Iterable[int] | None, defaults: tuple[int, ...]
) -> tuple[int, ...]:
    if seeds is None:
        return defaults
    if isinstance(seeds, (str, bytes)):
        raise ManifestError("CLI seeds must be an iterable of nonnegative integers")
    parsed = tuple(_nonnegative_int(seed, path="CLI seeds") for seed in seeds)
    if not parsed:
        raise ManifestError("CLI seeds must not be empty")
    if len(set(parsed)) != len(parsed):
        raise ManifestError("CLI seeds would create duplicate run ids")
    return parsed


def _dataset_dimension(manifest: Manifest) -> Dimension | None:
    return next(
        (dimension for dimension in manifest.dimensions if dimension.name == "dataset"),
        None,
    )


def _filter_assignments_for_dataset(
    assignments: tuple[_Assignment, ...],
    *,
    dataset: str | None,
    dataset_dimension: Dimension | None,
) -> tuple[_Assignment, ...]:
    if dataset is not None and dataset_dimension is not None:
        dataset_value_ids = {value.id for value in dataset_dimension.values}
        if dataset not in dataset_value_ids:
            raise ManifestError(
                f"dataset dimension has no value {dataset!r} requested by CLI"
            )
        assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.dimensions["dataset"] == dataset
        )
    if assignments:
        return assignments
    if dataset is not None:
        raise ManifestError(
            f"empty matrix expansion after dataset selection {dataset!r}; "
            "no logical arms remain after exclusions and CLI filtering"
        )
    raise ManifestError(
        "empty matrix expansion after exclusions; no logical arms remain"
    )


def _resolve_assignment_dataset_id(
    manifest: Manifest,
    merged: dict[str, Any],
    *,
    dataset: str | None,
    dataset_dimension: Dimension | None,
    datasets_by_id: Mapping[str, DatasetDeclaration],
) -> str:
    if dataset is not None and dataset_dimension is None:
        merged["dataset.id"] = dataset
    dataset_id = merged.get("dataset.id")
    if dataset_id is None and len(manifest.datasets) == 1:
        dataset_id = manifest.datasets[0].id
        merged["dataset.id"] = dataset_id
    if not isinstance(dataset_id, str) or dataset_id not in datasets_by_id:
        raise ManifestError(
            f"matrix assignment does not resolve a declared dataset.id: {dataset_id!r}"
        )
    if dataset is not None and dataset_id != dataset:
        raise ManifestError(
            f"dataset dimension did not resolve selected dataset {dataset!r}"
        )
    return dataset_id


def _build_logical_arms(
    manifest: Manifest,
    assignments: tuple[_Assignment, ...],
    *,
    dataset: str | None,
    dataset_dimension: Dimension | None,
    datasets_by_id: Mapping[str, DatasetDeclaration],
    epochs: int | None,
    output_root: str | None,
) -> tuple[LogicalArm, ...]:
    dimension_options = FrozenDict(
        {
            dimension.name: tuple(value.id for value in dimension.values)
            for dimension in manifest.dimensions
        }
    )
    arms: list[LogicalArm] = []
    arm_ids: set[str] = set()
    for assignment in assignments:
        merged = dict(_merge_assignment(manifest, assignment))
        dataset_id = _resolve_assignment_dataset_id(
            manifest,
            merged,
            dataset=dataset,
            dataset_dimension=dataset_dimension,
            datasets_by_id=datasets_by_id,
        )
        if epochs is not None:
            merged["training.epochs"] = epochs

        id_parts = [manifest.study_id, dataset_id]
        id_parts.extend(
            assignment.dimensions[dimension.name]
            for dimension in manifest.dimensions
            if dimension.name != "dataset"
        )
        arm_id = "--".join(id_parts)
        if arm_id in arm_ids:
            raise ManifestError(f"duplicate logical arm id {arm_id!r}")
        arm_ids.add(arm_id)
        arms.append(
            LogicalArm(
                id=arm_id,
                study_id=manifest.study_id,
                dataset_id=dataset_id,
                dataset=datasets_by_id[dataset_id],
                dimensions=assignment.dimensions,
                dimension_options=dimension_options,
                overrides=FrozenDict(merged),
                output_root=output_root,
            )
        )
    return tuple(arms)


def _build_runs(
    arms: tuple[LogicalArm, ...], seeds: tuple[int, ...]
) -> tuple[ResolvedRun, ...]:
    runs: list[ResolvedRun] = []
    run_ids: set[str] = set()
    for arm in arms:
        for seed in seeds:
            run_id = f"{arm.id}--seed-{seed}"
            if run_id in run_ids:
                raise ManifestError(f"duplicate run id {run_id!r}")
            run_ids.add(run_id)
            runs.append(
                ResolvedRun(
                    id=run_id,
                    arm_id=arm.id,
                    study_id=arm.study_id,
                    dataset_id=arm.dataset_id,
                    dataset=arm.dataset,
                    seed=seed,
                    dimensions=arm.dimensions,
                    dimension_options=arm.dimension_options,
                    overrides=arm.overrides,
                    output_root=arm.output_root,
                )
            )
    arm_ids = {arm.id for arm in arms}
    namespace_collisions = arm_ids & run_ids
    if namespace_collisions:
        raise ManifestError(
            f"arm/run id namespace collision: {sorted(namespace_collisions)[0]!r}"
        )
    return tuple(runs)


def _expand(
    manifest: Manifest,
    *,
    dataset: str | None,
    epochs: int | None,
    seeds: Iterable[int] | None,
    output_root: str | Path | None,
) -> tuple[tuple[LogicalArm, ...], tuple[ResolvedRun, ...], tuple[str, ...]]:
    datasets_by_id = {item.id: item for item in manifest.datasets}
    if dataset is not None:
        if not isinstance(dataset, str) or dataset not in datasets_by_id:
            raise ManifestError(f"unknown CLI dataset {dataset!r}")
    if epochs is not None:
        epochs = _positive_int(epochs, path="CLI epochs")
    selected_seeds = _cli_seeds(seeds, manifest.seeds)
    selected_output = manifest.output_root if output_root is None else str(output_root)
    if selected_output is not None and not selected_output:
        raise ManifestError("CLI output-root must be nonempty")

    dataset_dimension = _dataset_dimension(manifest)
    assignments = _filter_assignments_for_dataset(
        _matrix_assignments(manifest),
        dataset=dataset,
        dataset_dimension=dataset_dimension,
    )
    arms = _build_logical_arms(
        manifest,
        assignments,
        dataset=dataset,
        dataset_dimension=dataset_dimension,
        datasets_by_id=datasets_by_id,
        epochs=epochs,
        output_root=selected_output,
    )
    runs = _build_runs(arms, selected_seeds)
    active_dataset_ids = tuple(dict.fromkeys(arm.dataset_id for arm in arms))
    return arms, runs, active_dataset_ids


def expand_arms(
    manifest: Manifest,
    *,
    dataset: str | None = None,
    epochs: int | None = None,
    output_root: str | Path | None = None,
) -> tuple[LogicalArm, ...]:
    """Expand logical arms with deterministic matrix and CLI precedence."""
    arms, _, _ = _expand(
        manifest,
        dataset=dataset,
        epochs=epochs,
        seeds=manifest.seeds,
        output_root=output_root,
    )
    return arms


def expand_runs(
    manifest: Manifest,
    *,
    dataset: str | None = None,
    epochs: int | None = None,
    seeds: Iterable[int] | None = None,
    output_root: str | Path | None = None,
) -> tuple[ResolvedRun, ...]:
    """Expand and validate all runs, including dataset-relative rule targets."""
    return resolve_manifest(
        manifest, dataset=dataset, epochs=epochs, seeds=seeds, output_root=output_root
    ).runs


def _resolve_selector(
    arms: tuple[LogicalArm, ...],
    selector: Mapping[str, str],
    *,
    path: str,
    selected_dataset_ids: tuple[str, ...],
) -> tuple[LogicalArm, ...]:
    narrowed_dataset = selector.get("dataset")
    matches = tuple(
        arm
        for arm in arms
        if (narrowed_dataset is None or arm.dataset_id == narrowed_dataset)
        and all(
            key == "dataset" or arm.dimensions.get(key) == value
            for key, value in selector.items()
        )
    )
    if not matches:
        dataset_context = ",".join(selected_dataset_ids)
        raise ManifestError(
            f"{path} must resolve at least one logical arm for selected datasets "
            f"{dataset_context!r}; got zero matches"
        )
    counts_by_dataset: dict[str, int] = {}
    for arm in matches:
        counts_by_dataset[arm.dataset_id] = counts_by_dataset.get(arm.dataset_id, 0) + 1
    ambiguous_datasets = [
        dataset_id for dataset_id, count in counts_by_dataset.items() if count > 1
    ]
    if ambiguous_datasets:
        raise ManifestError(
            f"{path} must resolve one logical arm globally or per dataset; got multiple "
            f"matches for datasets {','.join(ambiguous_datasets)!r}"
        )
    return matches


def _applicability(
    *,
    rule_id: str,
    dataset: DatasetDeclaration,
    requires: tuple[str, ...],
    when_dataset_kind: str | None,
    on_missing_capability: str,
) -> tuple[RuleApplicability, str | None]:
    if when_dataset_kind is not None and dataset.kind != when_dataset_kind:
        return RuleApplicability.NOT_APPLICABLE, "dataset_kind"
    missing = [
        capability for capability in requires if capability not in dataset.capabilities
    ]
    if not missing:
        return RuleApplicability.ACTIVE, None
    if on_missing_capability == "not_applicable":
        return RuleApplicability.NOT_APPLICABLE, f"missing_capability:{missing[0]}"
    raise ManifestError(
        f"rule {rule_id!r} requires missing dataset capability {missing[0]!r} on {dataset.id!r}"
    )


def _comparison_applicability(
    comparison: Comparison,
    left: LogicalArm,
    right: LogicalArm,
    datasets_by_id: Mapping[str, DatasetDeclaration],
) -> tuple[RuleApplicability, str | None]:
    dataset_ids = tuple(sorted({left.dataset_id, right.dataset_id}))
    datasets = tuple(datasets_by_id[dataset_id] for dataset_id in dataset_ids)
    if comparison.when_dataset_kind is not None and any(
        dataset.kind != comparison.when_dataset_kind for dataset in datasets
    ):
        return RuleApplicability.NOT_APPLICABLE, "dataset_kind"
    for capability in comparison.requires:
        missing_dataset_ids = [
            dataset.id for dataset in datasets if capability not in dataset.capabilities
        ]
        if not missing_dataset_ids:
            continue
        if comparison.on_missing_capability == "not_applicable":
            return RuleApplicability.NOT_APPLICABLE, f"missing_capability:{capability}"
        raise ManifestError(
            f"rule {comparison.id!r} requires missing dataset capability "
            f"{capability!r} on {missing_dataset_ids[0]!r}"
        )
    return RuleApplicability.ACTIVE, None


def _pair_comparison_targets(
    comparison: Comparison,
    left_targets: tuple[LogicalArm, ...],
    right_targets: tuple[LogicalArm, ...],
) -> tuple[tuple[LogicalArm, LogicalArm], ...]:
    pairs: tuple[tuple[LogicalArm, LogicalArm], ...]
    if len(left_targets) == len(right_targets) == 1:
        pairs = ((left_targets[0], right_targets[0]),)
    else:
        left_by_dataset = {arm.dataset_id: arm for arm in left_targets}
        right_by_dataset = {arm.dataset_id: arm for arm in right_targets}
        if set(left_by_dataset) != set(right_by_dataset):
            raise ManifestError(
                f"comparison {comparison.id!r} dataset-grouped selectors have zero "
                "paired matches for one or more datasets"
            )
        pairs = tuple(
            (left, right_by_dataset[left.dataset_id]) for left in left_targets
        )
    self_ratio = next((left for left, right in pairs if left.id == right.id), None)
    if self_ratio is not None:
        raise ManifestError(
            f"comparison {comparison.id!r} self-ratio uses the same arm "
            f"{self_ratio.id!r} on both sides"
        )
    return pairs


def _resolve_rules(
    manifest: Manifest,
    arms: tuple[LogicalArm, ...],
    selected_dataset_ids: tuple[str, ...],
) -> tuple[tuple[ResolvedGate, ...], tuple[ResolvedComparison, ...]]:
    datasets_by_id = {dataset.id: dataset for dataset in manifest.datasets}
    resolved_gates: list[ResolvedGate] = []
    for gate in manifest.gates:
        targets = _resolve_selector(
            arms,
            gate.target,
            path=f"gate {gate.id!r} target",
            selected_dataset_ids=selected_dataset_ids,
        )
        for target in targets:
            applicability, reason = _applicability(
                rule_id=gate.id,
                dataset=datasets_by_id[target.dataset_id],
                requires=gate.requires,
                when_dataset_kind=gate.when_dataset_kind,
                on_missing_capability=gate.on_missing_capability,
            )
            resolved_gates.append(
                ResolvedGate(gate, target.dataset_id, target.id, applicability, reason)
            )

    resolved_comparisons: list[ResolvedComparison] = []
    for comparison in manifest.comparisons:
        left_targets = _resolve_selector(
            arms,
            comparison.left,
            path=f"comparison {comparison.id!r} left",
            selected_dataset_ids=selected_dataset_ids,
        )
        right_targets = _resolve_selector(
            arms,
            comparison.right,
            path=f"comparison {comparison.id!r} right",
            selected_dataset_ids=selected_dataset_ids,
        )
        for left, right in _pair_comparison_targets(
            comparison, left_targets, right_targets
        ):
            applicability, reason = _comparison_applicability(
                comparison, left, right, datasets_by_id
            )
            resolved_comparisons.append(
                ResolvedComparison(
                    comparison,
                    left.dataset_id,
                    right.dataset_id,
                    left.id,
                    right.id,
                    applicability,
                    reason,
                )
            )
    return tuple(resolved_gates), tuple(resolved_comparisons)


def resolve_manifest(
    manifest: Manifest,
    *,
    dataset: str | None = None,
    epochs: int | None = None,
    seeds: Iterable[int] | None = None,
    output_root: str | Path | None = None,
) -> ResolvedStudy:
    """Resolve runs and dataset-relative gates/comparisons after CLI selection."""
    if not isinstance(manifest, Manifest):
        raise TypeError("manifest must be a Manifest")
    arms, runs, selected_dataset_ids = _expand(
        manifest, dataset=dataset, epochs=epochs, seeds=seeds, output_root=output_root
    )
    gates, comparisons = _resolve_rules(manifest, arms, selected_dataset_ids)
    return ResolvedStudy(manifest, arms, runs, gates, comparisons)


def select_runs(
    runs: Iterable[ResolvedRun], only: str | None
) -> tuple[ResolvedRun, ...]:
    """Select runs by exact arm/run id or strict dimension conjunction."""
    ordered = tuple(runs)
    if only is None:
        return ordered
    if not isinstance(only, str) or not only:
        raise ManifestError("--only must be a nonempty selector")
    if "=" not in only and "," not in only:
        return _select_runs_by_exact_id(ordered, only)
    parsed = _parse_only_conjunction(ordered, only)
    matches = tuple(
        run
        for run in ordered
        if all(run.dimensions.get(name) == value for name, value in parsed.items())
    )
    if not matches:
        raise ManifestError(f"--only selector {only!r} matched no runs")
    return matches


def _select_runs_by_exact_id(
    runs: tuple[ResolvedRun, ...], exact_id: str
) -> tuple[ResolvedRun, ...]:
    run_matches = tuple(run for run in runs if run.id == exact_id)
    arm_matches = tuple(run for run in runs if run.arm_id == exact_id)
    if run_matches and arm_matches:
        raise ManifestError(
            f"--only exact id {exact_id!r} is ambiguous between arm and run namespaces"
        )
    if len(run_matches) > 1:
        raise ManifestError(f"--only exact run id {exact_id!r} is ambiguous")
    matches = run_matches or arm_matches
    if not matches:
        raise ManifestError(f"--only exact id {exact_id!r} matched no arm or run")
    return matches


def _dimension_options(runs: tuple[ResolvedRun, ...]) -> dict[str, set[str]]:
    options: dict[str, set[str]] = {}
    for run in runs:
        for name, values in run.dimension_options.items():
            options.setdefault(name, set()).update(values)
    return options


def _parse_only_conjunction(runs: tuple[ResolvedRun, ...], only: str) -> dict[str, str]:
    if any(character.isspace() for character in only):
        raise ManifestError("--only conjunction may not contain whitespace")
    terms = only.split(",")
    if not terms or any(not term or term.count("=") != 1 for term in terms):
        raise ManifestError(
            "--only must be an exact id or comma conjunction of dimension=value"
        )
    parsed: dict[str, str] = {}
    dimension_options = _dimension_options(runs)
    for term in terms:
        name, value = term.split("=")
        if not name or not value or name in parsed:
            raise ManifestError(
                "--only conjunction has an empty or duplicate dimension"
            )
        if name not in dimension_options:
            raise ManifestError(f"--only has unknown dimension {name!r}")
        if value not in dimension_options[name]:
            raise ManifestError(
                f"--only has unknown value {value!r} for dimension {name!r}"
            )
        parsed[name] = value
    return parsed
