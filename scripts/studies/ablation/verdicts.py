"""Typed, conservative verdict evaluation for resolved ablation rules.

This module deliberately consumes resolved manifest rules and flattened metric
values. It does not parse manifests or reproduce metric calculations.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from numbers import Real
from types import MappingProxyType
import numpy as np

from .manifest import ResolvedComparison, ResolvedGate, RuleApplicability
from .visual_review import (
    PENDING_REVIEW_SCHEMA_VERSION,
    REVIEW_SCHEMA_VERSION,
    ReviewDecision,
    ReviewError,
    VisualReview,
    parse_review,
    pending_review_template,
)

INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION = "hybrid_resnet_integration_bridge_v4"
INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION = (
    "hybrid_resnet_integration_bridge_evidence_v3"
)
_SSIM_FLOOR_SOURCES = frozenset({"fixture", "frozen_reference_artifact"})
_DIFFERENCE_CLASSIFICATIONS = frozenset(
    {"harmless", "performance_relevant", "comparison_invalidating"}
)
# Command operands of the declared reference condition: a difference here is a
# violated experimental condition and can never be classified as diagnostic.
_BRIDGE_COMMAND_FIELDS = (
    "architecture",
    "generator_output_mode",
    "hybrid_encoder_conv_hidden_scale",
    "training_patch_weighting",
    "physics_forward_mode",
    "amplitude_physics_gain",
    "torch_loss_mode",
    "seed",
    "epochs",
)
# Locked gate floors/tolerances: a difference here is gate tampering and can
# never be classified as diagnostic.
_BRIDGE_GATE_CONTRACT_FIELDS = (
    "ssim_floor_source",
    "ssim_floor_reference_artifact_sha256",
    "fixture_amp_mae_max",
    "fixture_phase_mae_max",
    "fixture_amp_ssim_min",
    "fixture_phase_ssim_min",
)

__all__ = [
    "PENDING_REVIEW_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "AttemptRow",
    "AttemptStatus",
    "CompletionState",
    "ComparisonResult",
    "GateResult",
    "IntegrationBridgeEvidence",
    "IntegrationBridgeRequirement",
    "LockedSsimFloors",
    "RecordedDifference",
    "ReviewDecision",
    "ReviewError",
    "Verdict",
    "VerdictInputError",
    "VisualReview",
    "aggregate_verdict",
    "evaluate_comparison",
    "evaluate_gate",
    "evaluate_integration_bridge",
    "parse_review",
    "pending_review_template",
]


class VerdictInputError(ValueError):
    """Raised when runtime evidence cannot be unambiguously adjudicated."""


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class CompletionState(str, Enum):
    TERMINAL = "terminal"
    INCOMPLETE = "incomplete"


_COMPLETION_BY_STATUS = {
    AttemptStatus.SUCCESS: CompletionState.TERMINAL,
    AttemptStatus.FAILED: CompletionState.TERMINAL,
    AttemptStatus.INCOMPLETE: CompletionState.INCOMPLETE,
}


@dataclass(frozen=True)
class _IntegrationBridgeContract:
    """Closed performance-reference contract for a grid-lines reference run.

    Locked amplitude/phase SSIM floors are the primary qualification gate and
    the fixture MAE tolerances remain a supporting guard. Every provenance
    field stays required evidence, but exact historical/generic equality is
    diagnostic rather than a gate: differences must be surfaced and classified.
    """

    schema_version: str
    bridge_id: str
    ssim_floor_source: str
    ssim_floor_reference_artifact_sha256: str | None
    probe_source_kind: str
    loader_kind: str
    raw_probe_archive_sha256: str
    raw_probe_array_sha256: str
    transform_order: tuple[str, ...]
    probe_smoothing_sigma: float
    probe_target_n: int
    transformed_probe_sha256: str
    probe_normalize: bool
    probe_scale: float
    dictionary_probe_normalization_policy: str
    dictionary_effective_probe_sha256: str
    mmap_probe_normalization_policy: str
    mmap_same_config_equivalent: bool
    probe_mask: bool
    probe_mask_tensor_is_none: bool
    resolved_probe_mask_kind: str
    resolved_probe_mask_sha256: str
    probe_mask_sigma: float
    probe_mask_diameter: float | None
    model_edge_pad: int
    grid_lines_size: int
    grid_lines_n: int
    grid_lines_gridsize: int
    grid_lines_offset: int
    outer_offset_train: int
    outer_offset_test: int
    nimgs_train: int
    nimgs_test: int
    historical_crop_border: int
    historical_effective_patch_n: int
    generic_position_crop_border: int
    generic_effective_patch_n: int
    crop_boundaries_equivalent: bool
    architecture: str
    generator_output_mode: str
    hybrid_encoder_conv_hidden_scale: float
    training_patch_weighting: str
    physics_forward_mode: str
    amplitude_physics_gain: float
    torch_loss_mode: str
    seed: int
    epochs: int
    fixture_amp_mae_max: float
    fixture_phase_mae_max: float
    fixture_amp_ssim_min: float
    fixture_phase_ssim_min: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise VerdictInputError("integration bridge contract must be a mapping")
        version = value.get("schema_version")
        if version != INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION:
            raise VerdictInputError(
                f"integration bridge contract schema_version {version!r} is not "
                "supported; the prior contract is superseded; regenerate the "
                "reference contract and its sealed "
                f"evidence for {INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION!r} "
                "(older bridge artifacts are sealed history)"
            )
        expected = tuple(cls.__dataclass_fields__)
        if set(value) != set(expected):
            raise VerdictInputError(
                "integration bridge contract fields do not match the closed schema"
            )

        def text_field(name: str) -> str:
            item = value[name]
            if not isinstance(item, str) or not item:
                raise VerdictInputError(f"integration bridge {name} must be text")
            return item

        def sha_field(name: str) -> str:
            item = text_field(name)
            if len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
                raise VerdictInputError(f"integration bridge {name} must be SHA-256")
            if item == "0" * 64:
                raise VerdictInputError(
                    f"integration bridge {name} must be non-sentinel"
                )
            return item

        def bool_field(name: str) -> bool:
            item = value[name]
            if type(item) is not bool:
                raise VerdictInputError(f"integration bridge {name} must be boolean")
            return item

        def int_field(name: str) -> int:
            item = value[name]
            if type(item) is not int or item < 0:
                raise VerdictInputError(
                    f"integration bridge {name} must be a nonnegative integer"
                )
            return item

        def float_field(name: str) -> float:
            item = value[name]
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                raise VerdictInputError(f"integration bridge {name} must be finite")
            return float(item)

        raw_order = value["transform_order"]
        if (
            not isinstance(raw_order, (list, tuple))
            or not raw_order
            or any(not isinstance(item, str) or not item for item in raw_order)
        ):
            raise VerdictInputError(
                "integration bridge transform_order must be nonempty text"
            )
        diameter = value["probe_mask_diameter"]
        if diameter is not None:
            diameter = float_field("probe_mask_diameter")
        floor_source = text_field("ssim_floor_source")
        if floor_source not in _SSIM_FLOOR_SOURCES:
            raise VerdictInputError(
                "integration bridge ssim_floor_source must be 'fixture' or "
                "'frozen_reference_artifact'"
            )
        floor_reference = value["ssim_floor_reference_artifact_sha256"]
        if floor_source == "fixture":
            if floor_reference is not None:
                raise VerdictInputError(
                    "integration bridge fixture-sourced floors must not declare "
                    "a frozen reference artifact"
                )
        else:
            floor_reference = sha_field("ssim_floor_reference_artifact_sha256")
        amplitude_physics_gain = float_field("amplitude_physics_gain")
        if amplitude_physics_gain <= 0.0:
            raise VerdictInputError(
                "integration bridge amplitude_physics_gain must be positive"
            )
        physics_forward_mode = text_field("physics_forward_mode")
        if physics_forward_mode == "amplitude" and amplitude_physics_gain != 16.0:
            raise VerdictInputError(
                "integration bridge amplitude reference arms require "
                "amplitude_physics_gain exactly 16"
            )
        return cls(
            schema_version=text_field("schema_version"),
            bridge_id=text_field("bridge_id"),
            ssim_floor_source=floor_source,
            ssim_floor_reference_artifact_sha256=floor_reference,
            probe_source_kind=text_field("probe_source_kind"),
            loader_kind=text_field("loader_kind"),
            raw_probe_archive_sha256=sha_field("raw_probe_archive_sha256"),
            raw_probe_array_sha256=sha_field("raw_probe_array_sha256"),
            transform_order=tuple(raw_order),
            probe_smoothing_sigma=float_field("probe_smoothing_sigma"),
            probe_target_n=int_field("probe_target_n"),
            transformed_probe_sha256=sha_field("transformed_probe_sha256"),
            probe_normalize=bool_field("probe_normalize"),
            probe_scale=float_field("probe_scale"),
            dictionary_probe_normalization_policy=text_field(
                "dictionary_probe_normalization_policy"
            ),
            dictionary_effective_probe_sha256=sha_field(
                "dictionary_effective_probe_sha256"
            ),
            mmap_probe_normalization_policy=text_field(
                "mmap_probe_normalization_policy"
            ),
            mmap_same_config_equivalent=bool_field(
                "mmap_same_config_equivalent"
            ),
            probe_mask=bool_field("probe_mask"),
            probe_mask_tensor_is_none=bool_field("probe_mask_tensor_is_none"),
            resolved_probe_mask_kind=text_field("resolved_probe_mask_kind"),
            resolved_probe_mask_sha256=sha_field("resolved_probe_mask_sha256"),
            probe_mask_sigma=float_field("probe_mask_sigma"),
            probe_mask_diameter=diameter,
            model_edge_pad=int_field("model_edge_pad"),
            grid_lines_size=int_field("grid_lines_size"),
            grid_lines_n=int_field("grid_lines_n"),
            grid_lines_gridsize=int_field("grid_lines_gridsize"),
            grid_lines_offset=int_field("grid_lines_offset"),
            outer_offset_train=int_field("outer_offset_train"),
            outer_offset_test=int_field("outer_offset_test"),
            nimgs_train=int_field("nimgs_train"),
            nimgs_test=int_field("nimgs_test"),
            historical_crop_border=int_field("historical_crop_border"),
            historical_effective_patch_n=int_field(
                "historical_effective_patch_n"
            ),
            generic_position_crop_border=int_field(
                "generic_position_crop_border"
            ),
            generic_effective_patch_n=int_field("generic_effective_patch_n"),
            crop_boundaries_equivalent=bool_field("crop_boundaries_equivalent"),
            architecture=text_field("architecture"),
            generator_output_mode=text_field("generator_output_mode"),
            hybrid_encoder_conv_hidden_scale=float_field(
                "hybrid_encoder_conv_hidden_scale"
            ),
            training_patch_weighting=text_field("training_patch_weighting"),
            physics_forward_mode=physics_forward_mode,
            amplitude_physics_gain=amplitude_physics_gain,
            torch_loss_mode=text_field("torch_loss_mode"),
            seed=int_field("seed"),
            epochs=int_field("epochs"),
            fixture_amp_mae_max=float_field("fixture_amp_mae_max"),
            fixture_phase_mae_max=float_field("fixture_phase_mae_max"),
            fixture_amp_ssim_min=float_field("fixture_amp_ssim_min"),
            fixture_phase_ssim_min=float_field("fixture_phase_ssim_min"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class IntegrationBridgeRequirement(_IntegrationBridgeContract):
    """Declared reference-performance configuration and locked gate floors."""


# Contract fields whose historical/generic differences are recordable
# diagnostics once classified. Identity, command operands, and gate floors are
# binding and excluded.
_CLASSIFIABLE_BRIDGE_CONTRACT_FIELDS = tuple(
    name
    for name in _IntegrationBridgeContract.__dataclass_fields__
    if name
    not in {
        "schema_version",
        "bridge_id",
        *_BRIDGE_COMMAND_FIELDS,
        *_BRIDGE_GATE_CONTRACT_FIELDS,
    }
)
# Valid ids for recorded-difference classification entries: classifiable
# contract fields plus the dual-evaluator canvas/mask equivalence diagnostics.
BRIDGE_DIFFERENCE_IDS = frozenset(_CLASSIFIABLE_BRIDGE_CONTRACT_FIELDS) | {
    "canvas_equivalence",
    "mask_equivalence",
}


@dataclass(frozen=True)
class LockedSsimFloors:
    """Final SSIM floors locked from a frozen reference artifact.

    The floors are parsed from the frozen artifact's bytes and bound to its
    recomputed SHA-256, so approximate declarations in a contract can never
    substitute for the locked finals.
    """

    amp_ssim_min: float
    phase_ssim_min: float
    source_artifact_sha256: str

    @classmethod
    def from_frozen_reference_artifact_bytes(cls, value: bytes) -> LockedSsimFloors:
        if not isinstance(value, bytes) or not value:
            raise VerdictInputError(
                "frozen reference artifact must be nonempty bytes"
            )
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerdictInputError(
                "frozen reference artifact is invalid JSON"
            ) from error
        if not isinstance(payload, Mapping):
            raise VerdictInputError("frozen reference artifact must be an object")
        floors: dict[str, float] = {}
        for name in ("amp_ssim_min", "phase_ssim_min"):
            item = payload.get(name)
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
                or not -1.0 <= float(item) <= 1.0
            ):
                raise VerdictInputError(
                    f"frozen reference artifact {name} must be an SSIM floor "
                    "in [-1, 1]"
                )
            floors[name] = float(item)
        return cls(
            amp_ssim_min=floors["amp_ssim_min"],
            phase_ssim_min=floors["phase_ssim_min"],
            source_artifact_sha256=hashlib.sha256(value).hexdigest(),
        )


@dataclass(frozen=True)
class RecordedDifference:
    """One surfaced historical/generic difference with its declared class."""

    field: str
    classification: str
    justification: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RecordedDifference:
        if not isinstance(value, Mapping) or set(value) != {
            "field",
            "classification",
            "justification",
        }:
            raise VerdictInputError(
                "integration bridge recorded difference must declare exactly "
                "field, classification, and justification"
            )
        name = value["field"]
        if not isinstance(name, str) or name not in BRIDGE_DIFFERENCE_IDS:
            raise VerdictInputError(
                f"integration bridge recorded difference field {name!r} is not "
                "classifiable; identity, command-operand, and gate-floor "
                "differences are binding"
            )
        classification = value["classification"]
        if classification not in _DIFFERENCE_CLASSIFICATIONS:
            raise VerdictInputError(
                "integration bridge difference classification must be one of "
                + ", ".join(sorted(_DIFFERENCE_CLASSIFICATIONS))
            )
        justification = value["justification"]
        if not isinstance(justification, str) or not justification:
            raise VerdictInputError(
                "integration bridge recorded difference justification must be "
                "nonempty text"
            )
        return cls(
            field=name,
            classification=str(classification),
            justification=justification,
        )


@dataclass(frozen=True)
class IntegrationBridgeEvidence:
    """Execution results extracted from a sealed reference-qualification run."""

    schema_version: str
    contract: IntegrationBridgeRequirement
    checkpoint_sha256: str
    selected_checkpoint: str
    train_npz_sha256: str
    test_npz_sha256: str
    pre_stitch_patch_sha256: str
    historical_canvas_sha256: str
    ground_truth_sha256: str
    generic_canvas_sha256: str
    historical_mask_sha256: str
    generic_mask_sha256: str
    canvases_equivalent: bool
    masks_equivalent: bool
    no_resize_asserted: bool
    gauge_handling: str
    recorded_differences: tuple[RecordedDifference, ...]
    fixture_amp_mae: float
    fixture_phase_mae: float
    fixture_amp_ssim: float
    fixture_phase_ssim: float
    architecture: str
    generator_output_mode: str
    hybrid_encoder_conv_hidden_scale: float
    training_patch_weighting: str
    physics_forward_mode: str
    amplitude_physics_gain: float
    torch_loss_mode: str
    seed: int
    epochs: int
    sealed_artifact_sha256: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]):
        if not isinstance(value, Mapping):
            raise VerdictInputError(
                "integration bridge evidence must be a mapping"
            )
        version = value.get("schema_version")
        if version != INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION:
            raise VerdictInputError(
                f"integration bridge evidence schema_version {version!r} is not "
                "supported; the prior evidence is superseded; regenerate "
                "sealed execution evidence for "
                f"{INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION!r} under the "
                "performance-reference contract"
            )
        expected = {item.name for item in fields(cls) if item.init}
        if set(value) != expected:
            raise VerdictInputError(
                "integration bridge evidence fields do not match the execution schema"
            )

        def text(name: str) -> str:
            item = value[name]
            if not isinstance(item, str) or not item:
                raise VerdictInputError(f"integration bridge evidence {name} must be text")
            return item

        def sha(name: str) -> str:
            item = text(name)
            if len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item):
                raise VerdictInputError(
                    f"integration bridge evidence {name} must be SHA-256"
                )
            return item

        def boolean(name: str) -> bool:
            item = value[name]
            if type(item) is not bool:
                raise VerdictInputError(
                    f"integration bridge evidence {name} must be boolean"
                )
            return item

        def number(name: str) -> float:
            item = value[name]
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                raise VerdictInputError(
                    f"integration bridge evidence {name} must be finite"
                )
            return float(item)

        def integer(name: str) -> int:
            item = value[name]
            if type(item) is not int or item < 0:
                raise VerdictInputError(
                    f"integration bridge evidence {name} must be nonnegative integer"
                )
            return item

        contract = value["contract"]
        if not isinstance(contract, Mapping):
            raise VerdictInputError("integration bridge evidence contract must be a mapping")
        amp_mae = number("fixture_amp_mae")
        phase_mae = number("fixture_phase_mae")
        amp_ssim = number("fixture_amp_ssim")
        phase_ssim = number("fixture_phase_ssim")
        amplitude_physics_gain = number("amplitude_physics_gain")
        if amp_mae < 0.0 or phase_mae < 0.0:
            raise VerdictInputError("integration bridge fixture MAE must be nonnegative")
        if not -1.0 <= amp_ssim <= 1.0 or not -1.0 <= phase_ssim <= 1.0:
            raise VerdictInputError("integration bridge fixture SSIM must be in [-1, 1]")
        if amplitude_physics_gain <= 0.0:
            raise VerdictInputError(
                "integration bridge evidence amplitude_physics_gain must be positive"
            )
        raw_differences = value["recorded_differences"]
        if not isinstance(raw_differences, (list, tuple)):
            raise VerdictInputError(
                "integration bridge evidence recorded_differences must be a list"
            )
        differences = tuple(
            RecordedDifference.from_mapping(item) for item in raw_differences
        )
        difference_fields = [item.field for item in differences]
        if len(set(difference_fields)) != len(difference_fields):
            raise VerdictInputError(
                "integration bridge evidence recorded_differences contains "
                "duplicate fields"
            )
        return cls(
            schema_version=text("schema_version"),
            contract=IntegrationBridgeRequirement.from_mapping(contract),
            checkpoint_sha256=sha("checkpoint_sha256"),
            selected_checkpoint=text("selected_checkpoint"),
            train_npz_sha256=sha("train_npz_sha256"),
            test_npz_sha256=sha("test_npz_sha256"),
            pre_stitch_patch_sha256=sha("pre_stitch_patch_sha256"),
            historical_canvas_sha256=sha("historical_canvas_sha256"),
            ground_truth_sha256=sha("ground_truth_sha256"),
            generic_canvas_sha256=sha("generic_canvas_sha256"),
            historical_mask_sha256=sha("historical_mask_sha256"),
            generic_mask_sha256=sha("generic_mask_sha256"),
            canvases_equivalent=boolean("canvases_equivalent"),
            masks_equivalent=boolean("masks_equivalent"),
            no_resize_asserted=boolean("no_resize_asserted"),
            gauge_handling=text("gauge_handling"),
            recorded_differences=differences,
            fixture_amp_mae=amp_mae,
            fixture_phase_mae=phase_mae,
            fixture_amp_ssim=amp_ssim,
            fixture_phase_ssim=phase_ssim,
            architecture=text("architecture"),
            generator_output_mode=text("generator_output_mode"),
            hybrid_encoder_conv_hidden_scale=number(
                "hybrid_encoder_conv_hidden_scale"
            ),
            training_patch_weighting=text("training_patch_weighting"),
            physics_forward_mode=text("physics_forward_mode"),
            amplitude_physics_gain=amplitude_physics_gain,
            torch_loss_mode=text("torch_loss_mode"),
            seed=integer("seed"),
            epochs=integer("epochs"),
        )

    @classmethod
    def from_sealed_artifact_bytes(cls, value: bytes):
        if not isinstance(value, bytes) or not value:
            raise VerdictInputError("sealed integration bridge artifact must be bytes")
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerdictInputError("sealed integration bridge artifact is invalid JSON") from error
        evidence = cls.from_mapping(payload)
        object.__setattr__(
            evidence, "sealed_artifact_sha256", hashlib.sha256(value).hexdigest()
        )
        return evidence


@dataclass(frozen=True)
class AttemptRow:
    """One requested runtime attempt, carrying flat metric operands."""

    run_id: str
    arm_id: str
    dataset_id: str
    seed: int
    status: AttemptStatus
    completion: CompletionState
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("run_id", "arm_id", "dataset_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise VerdictInputError(f"{name} must be a nonempty string")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise VerdictInputError("seed must be a nonnegative integer")
        if not isinstance(self.status, AttemptStatus):
            raise VerdictInputError("status must be AttemptStatus")
        if not isinstance(self.completion, CompletionState):
            raise VerdictInputError("completion must be CompletionState")
        if self.completion is not _COMPLETION_BY_STATUS[self.status]:
            raise VerdictInputError(
                "status/completion must be success|failed/terminal or "
                "incomplete/incomplete"
            )
        if not isinstance(self.metrics, Mapping) or any(
            not isinstance(key, str) or not key for key in self.metrics
        ):
            raise VerdictInputError(
                "metrics must be a mapping with nonempty string paths"
            )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def terminal_success(self) -> bool:
        return (
            self.completion is CompletionState.TERMINAL
            and self.status is AttemptStatus.SUCCESS
        )


@dataclass(frozen=True)
class GateResult:
    id: str
    applicability: RuleApplicability
    verdict: Verdict | None
    category: str = "numeric"
    reason: str | None = None
    observed: float | int | None = None
    threshold: float | int | None = None
    contributing_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise VerdictInputError("result id must be a nonempty string")
        if not isinstance(self.category, str) or not self.category:
            raise VerdictInputError("result category must be a nonempty string")
        if not isinstance(self.applicability, RuleApplicability):
            raise VerdictInputError("result applicability must be RuleApplicability")
        if self.applicability is RuleApplicability.ACTIVE and not isinstance(
            self.verdict, Verdict
        ):
            raise VerdictInputError("active result requires a Verdict")
        if (
            self.applicability is RuleApplicability.NOT_APPLICABLE
            and self.verdict is not None
        ):
            raise VerdictInputError("not-applicable result must not have a Verdict")
        for name in ("observed", "threshold"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise VerdictInputError(f"{name} must be a finite numeric value")
        try:
            run_ids = tuple(self.contributing_run_ids)
        except TypeError as error:
            raise VerdictInputError("contributing_run_ids must be iterable") from error
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or len(
            set(run_ids)
        ) != len(run_ids):
            raise VerdictInputError(
                "contributing_run_ids must contain unique nonempty strings"
            )
        object.__setattr__(self, "contributing_run_ids", run_ids)

    @classmethod
    def active(
        cls,
        id: str,
        verdict: Verdict,
        *,
        category: str = "numeric",
        reason: str | None = None,
        observed: float | int | None = None,
        threshold: float | int | None = None,
        contributing_run_ids: Iterable[str] = (),
    ) -> GateResult:
        return cls(
            id=id,
            applicability=RuleApplicability.ACTIVE,
            verdict=verdict,
            category=category,
            reason=reason,
            observed=observed,
            threshold=threshold,
            contributing_run_ids=tuple(contributing_run_ids),
        )

    @classmethod
    def not_applicable(
        cls, id: str, reason: str | None, *, category: str = "numeric"
    ) -> GateResult:
        return cls(id, RuleApplicability.NOT_APPLICABLE, None, category, reason=reason)


ComparisonResult = GateResult


def _bridge_difference_reason(
    requirement: IntegrationBridgeRequirement, evidence: IntegrationBridgeEvidence
) -> str | None:
    """Adjudicate recorded provenance differences against fail-closed rules."""
    observed = {
        name
        for name in _CLASSIFIABLE_BRIDGE_CONTRACT_FIELDS
        if getattr(evidence.contract, name) != getattr(requirement, name)
    }
    if (
        not evidence.canvases_equivalent
        or evidence.historical_canvas_sha256 != evidence.generic_canvas_sha256
    ):
        observed.add("canvas_equivalence")
    if (
        not evidence.masks_equivalent
        or evidence.historical_mask_sha256 != evidence.generic_mask_sha256
    ):
        observed.add("mask_equivalence")
    if any(
        entry.classification == "comparison_invalidating"
        for entry in evidence.recorded_differences
    ):
        return "integration_bridge_comparison_invalidating_difference"
    classified = {entry.field for entry in evidence.recorded_differences}
    if observed - classified:
        return "integration_bridge_unclassified_difference"
    if classified - observed:
        return "integration_bridge_spurious_difference_classification"
    return None


def evaluate_integration_bridge(
    requirement: IntegrationBridgeRequirement,
    evidence: IntegrationBridgeEvidence | None,
    *,
    locked_ssim_floors: LockedSsimFloors | None = None,
) -> GateResult:
    """Gate claim-grade eligibility on reference performance qualification.

    Locked amplitude/phase SSIM floors are the primary gate and the MAE
    tolerances remain a supporting guard. Provenance differences are recorded
    diagnostics that must be classified: an unclassified difference fails
    closed and a comparison-invalidating one fails regardless of metrics.
    Command operands, gate floors, the sealed schema, and the no-hidden-resize
    assertion stay binding.
    """
    if not isinstance(requirement, IntegrationBridgeRequirement):
        raise VerdictInputError(
            "integration bridge requirement must be IntegrationBridgeRequirement"
        )
    if locked_ssim_floors is not None and not isinstance(
        locked_ssim_floors, LockedSsimFloors
    ):
        raise VerdictInputError("locked_ssim_floors must be LockedSsimFloors")
    floors_reason = None
    amp_floor = requirement.fixture_amp_ssim_min
    phase_floor = requirement.fixture_phase_ssim_min
    if requirement.ssim_floor_source == "fixture":
        if locked_ssim_floors is not None:
            raise VerdictInputError(
                "fixture-sourced SSIM floors are locked by the requirement and "
                "do not accept external locked floors"
            )
    elif locked_ssim_floors is None:
        floors_reason = "integration_bridge_ssim_floors_unlocked"
    elif (
        locked_ssim_floors.source_artifact_sha256
        != requirement.ssim_floor_reference_artifact_sha256
    ):
        floors_reason = "integration_bridge_ssim_floor_artifact_mismatch"
    else:
        amp_floor = locked_ssim_floors.amp_ssim_min
        phase_floor = locked_ssim_floors.phase_ssim_min
    if evidence is None:
        return GateResult.active(
            requirement.bridge_id,
            Verdict.INCONCLUSIVE,
            category="claim_prerequisite",
            reason="missing_integration_bridge_evidence",
        )
    if not isinstance(evidence, IntegrationBridgeEvidence):
        raise VerdictInputError(
            "integration bridge evidence must be IntegrationBridgeEvidence"
        )
    artifact_hashes = (
        evidence.checkpoint_sha256,
        evidence.pre_stitch_patch_sha256,
        evidence.historical_canvas_sha256,
        evidence.ground_truth_sha256,
        evidence.generic_canvas_sha256,
        evidence.historical_mask_sha256,
        evidence.generic_mask_sha256,
    )
    if evidence.sealed_artifact_sha256 is None:
        reason = "integration_bridge_unsealed_evidence"
    elif any(value == "0" * 64 for value in artifact_hashes):
        reason = "integration_bridge_artifact_identity_missing"
    elif evidence.contract.bridge_id != requirement.bridge_id:
        reason = "integration_bridge_identity_mismatch"
    elif any(
        getattr(evidence.contract, name) != getattr(requirement, name)
        for name in _BRIDGE_GATE_CONTRACT_FIELDS
    ):
        reason = "integration_bridge_gate_contract_mismatch"
    elif any(
        getattr(evidence, name) != getattr(requirement, name)
        for name in _BRIDGE_COMMAND_FIELDS
    ) or any(
        getattr(evidence.contract, name) != getattr(requirement, name)
        for name in _BRIDGE_COMMAND_FIELDS
    ):
        reason = "integration_bridge_command_mismatch"
    elif floors_reason is not None:
        reason = floors_reason
    elif not evidence.no_resize_asserted:
        reason = "integration_bridge_resize_detected"
    else:
        reason = _bridge_difference_reason(requirement, evidence)
    if reason is None:
        if (
            evidence.fixture_amp_ssim < amp_floor
            or evidence.fixture_phase_ssim < phase_floor
        ):
            reason = "integration_bridge_ssim_floor_failed"
        elif (
            evidence.fixture_amp_mae > requirement.fixture_amp_mae_max
            or evidence.fixture_phase_mae > requirement.fixture_phase_mae_max
        ):
            reason = "integration_bridge_mae_guard_failed"
    return GateResult.active(
        requirement.bridge_id,
        Verdict.PASS if reason is None else Verdict.FAIL,
        category="claim_prerequisite",
        reason=reason,
    )


def _target_rows(rows: Iterable[AttemptRow], arm_id: str) -> tuple[AttemptRow, ...]:
    selected = tuple(row for row in rows if row.arm_id == arm_id)
    identities: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    for row in selected:
        identity = (row.arm_id, row.seed)
        if identity in identities:
            raise VerdictInputError(f"duplicate seed identity for arm {row.arm_id!r}")
        if row.run_id in run_ids:
            raise VerdictInputError(f"duplicate run identity {row.run_id!r}")
        identities.add(identity)
        run_ids.add(row.run_id)
    return selected


def _requested(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in values
    ):
        raise VerdictInputError("requested seeds must be nonnegative integers")
    if len(set(values)) != len(values):
        raise VerdictInputError("requested seeds contain duplicates")
    return values


def _validate_rows_for_requested(
    rows: tuple[AttemptRow, ...], requested: tuple[int, ...]
) -> None:
    unexpected = sorted({row.seed for row in rows} - set(requested))
    if unexpected:
        raise VerdictInputError(f"unexpected seed rows: {unexpected!r}")


def _operand(row: AttemptRow, metric: str) -> tuple[str, float | None]:
    if metric not in row.metrics or row.metrics[metric] is None:
        return "missing", None
    value = row.metrics[metric]
    if isinstance(value, bool) or not isinstance(value, Real):
        return "malformed", None
    number = float(value)
    if not math.isfinite(number):
        return "nonfinite", number
    return "finite", number


def _aggregate(values: tuple[float, ...], aggregation: str) -> float | None:
    if aggregation == "mean":
        return float(sum(values) / len(values))
    if aggregation == "median":
        return float(np.median(np.asarray(values, dtype=np.float64)))
    if aggregation == "cv":
        if len(values) < 2:
            return None
        mean = float(sum(values) / len(values))
        scale = max(1.0, *(abs(value) for value in values))
        if abs(mean) <= 32.0 * np.finfo(np.float64).eps * scale:
            return None
        return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / abs(mean))
    raise VerdictInputError(f"unsupported aggregation {aggregation!r}")


def _numeric_result(
    gate: ResolvedGate,
    rows: tuple[AttemptRow, ...],
    status_result: GateResult | None,
) -> GateResult:
    spec = gate.gate
    if status_result is None or status_result.verdict is not Verdict.PASS:
        return GateResult.active(
            spec.id, Verdict.INCONCLUSIVE, reason="status_prerequisite"
        )
    if spec.metric is None or spec.min_successful is None:
        raise VerdictInputError(f"numeric gate {spec.id!r} is missing metric operands")
    successful = tuple(row for row in rows if row.terminal_success)
    operands = tuple(_operand(row, spec.metric) for row in successful)
    if any(state != "finite" for state, _ in operands):
        return GateResult.active(
            spec.id,
            Verdict.FAIL
            if spec.aggregation == "all_successful"
            else Verdict.INCONCLUSIVE,
            reason="missing_or_invalid_operand",
        )
    values = tuple(value for _, value in operands if value is not None)
    if len(values) < spec.min_successful:
        return GateResult.active(
            spec.id, Verdict.INCONCLUSIVE, reason="insufficient_successful_values"
        )
    if spec.aggregation is None or spec.threshold is None:
        raise VerdictInputError(
            f"numeric gate {spec.id!r} is missing aggregation operands"
        )
    if spec.aggregation == "all_successful":
        passed_values = tuple(
            value >= spec.threshold
            if spec.operator == "ge"
            else value <= spec.threshold
            for value in values
        )
        observed = min(values) if spec.operator == "ge" else max(values)
        return GateResult.active(
            spec.id,
            Verdict.PASS if all(passed_values) else Verdict.FAIL,
            observed=observed,
            threshold=spec.threshold,
            contributing_run_ids=(row.run_id for row in successful),
        )
    observed = _aggregate(values, spec.aggregation)
    if observed is None or not math.isfinite(observed):
        return GateResult.active(
            spec.id, Verdict.INCONCLUSIVE, reason="undefined_aggregation"
        )
    passed = (
        observed >= spec.threshold
        if spec.operator == "ge"
        else observed <= spec.threshold
    )
    return GateResult.active(
        spec.id,
        Verdict.PASS if passed else Verdict.FAIL,
        observed=observed,
        threshold=spec.threshold,
        contributing_run_ids=(row.run_id for row in successful),
    )


def evaluate_gate(
    gate: ResolvedGate,
    rows: Iterable[AttemptRow],
    *,
    requested_seeds: Iterable[int],
    status_result: GateResult | None = None,
    review: VisualReview | None = None,
) -> GateResult:
    """Evaluate one resolved manifest gate without reopening applicability."""
    if not isinstance(gate, ResolvedGate):
        raise VerdictInputError("gate must be ResolvedGate")
    requested = _requested(requested_seeds)
    spec = gate.gate
    if gate.applicability is RuleApplicability.NOT_APPLICABLE:
        return GateResult.not_applicable(gate.id, gate.reason, category=spec.operator)
    if spec.operator == "status_count_ge" and (
        spec.requested is None or len(requested) != spec.requested
    ):
        raise VerdictInputError("requested seed count does not match gate.requested")
    arm_rows = _target_rows(rows, gate.target_arm_id)
    _validate_rows_for_requested(arm_rows, requested)
    if spec.operator == "status_count_ge":
        if spec.threshold is None:
            raise VerdictInputError(f"status gate {spec.id!r} lacks requested operands")
        if len(arm_rows) != len(requested) or any(
            row.completion is not CompletionState.TERMINAL
            or row.status is AttemptStatus.INCOMPLETE
            for row in arm_rows
        ):
            return GateResult.active(
                spec.id, Verdict.INCONCLUSIVE, reason="missing_or_incomplete_attempt"
            )
        successes = sum(row.status is AttemptStatus.SUCCESS for row in arm_rows)
        return GateResult.active(
            spec.id,
            Verdict.PASS if successes >= spec.threshold else Verdict.FAIL,
            observed=successes,
            threshold=spec.threshold,
            contributing_run_ids=(row.run_id for row in arm_rows),
        )
    if spec.aggregation == "all_successful" and (
        len(arm_rows) != len(requested)
        or any(row.completion is not CompletionState.TERMINAL for row in arm_rows)
    ):
        return GateResult.active(
            spec.id, Verdict.INCONCLUSIVE, reason="missing_or_incomplete_attempt"
        )
    if spec.operator in {"ge", "le"}:
        return _numeric_result(gate, arm_rows, status_result)
    if spec.operator == "finite":
        if status_result is None or status_result.verdict is not Verdict.PASS:
            return GateResult.active(
                spec.id, Verdict.INCONCLUSIVE, reason="status_prerequisite"
            )
        if spec.metric is None or spec.min_successful is None:
            raise VerdictInputError(
                f"finite gate {spec.id!r} is missing metric operands"
            )
        successful = tuple(row for row in arm_rows if row.terminal_success)
        operands = tuple(_operand(row, spec.metric) for row in successful)
        if any(state in {"missing", "malformed"} for state, _ in operands):
            return GateResult.active(
                spec.id,
                Verdict.FAIL
                if spec.aggregation == "all_successful"
                else Verdict.INCONCLUSIVE,
                reason="missing_or_invalid_operand",
            )
        if any(state == "nonfinite" for state, _ in operands):
            return GateResult.active(spec.id, Verdict.FAIL, reason="nonfinite_operand")
        if len(successful) < spec.min_successful:
            return GateResult.active(
                spec.id, Verdict.INCONCLUSIVE, reason="insufficient_successful_values"
            )
        return GateResult.active(
            spec.id,
            Verdict.PASS,
            contributing_run_ids=(row.run_id for row in successful),
        )
    if spec.operator == "manual_review":
        if review is None:
            return GateResult.active(
                spec.id,
                Verdict.INCONCLUSIVE,
                category="manual_review",
                reason="pending_review",
            )
        family = spec.target.get("object_family")
        family_review = (
            review.families.get(family)
            if family is not None
            else review.legacy
        )
        if family is None and family_review is None and review.families:
            return GateResult.active(
                spec.id,
                Verdict.PASS
                if review.decision is ReviewDecision.APPROVE
                else Verdict.FAIL,
                category="manual_review",
            )
        if family_review is None:
            return GateResult.active(
                spec.id,
                Verdict.INCONCLUSIVE,
                category="manual_review",
                reason="missing_family_review",
            )
        return GateResult.active(
            spec.id,
            Verdict.PASS
            if family_review.decision is ReviewDecision.APPROVE
            else Verdict.FAIL,
            category="manual_review",
        )
    raise VerdictInputError(f"unsupported gate operator {spec.operator!r}")


def evaluate_comparison(
    comparison: ResolvedComparison,
    rows: Iterable[AttemptRow],
    *,
    requested_seeds: Iterable[int],
) -> ComparisonResult:
    """Evaluate one seed-paired ratio comparison using only terminal successes."""
    if not isinstance(comparison, ResolvedComparison):
        raise VerdictInputError("comparison must be ResolvedComparison")
    scored_category = (
        "diagnostic_comparison"
        if comparison.comparison.diagnostic
        else "mandatory_comparison"
    )
    evidence_category = (
        "required_diagnostic_evidence"
        if comparison.comparison.diagnostic
        else "mandatory_comparison"
    )
    if comparison.applicability is RuleApplicability.NOT_APPLICABLE:
        return GateResult.not_applicable(
            comparison.id, comparison.reason, category=scored_category
        )
    requested = _requested(requested_seeds)
    left_rows = _target_rows(rows, comparison.left_arm_id)
    right_rows = _target_rows(rows, comparison.right_arm_id)
    _validate_rows_for_requested(left_rows, requested)
    _validate_rows_for_requested(right_rows, requested)
    left_by_seed = {row.seed: row for row in left_rows}
    right_by_seed = {row.seed: row for row in right_rows}
    ratios: list[float] = []
    contributing: list[str] = []
    for seed in requested:
        left = left_by_seed.get(seed)
        right = right_by_seed.get(seed)
        if (
            left is None
            or right is None
            or left.completion is not CompletionState.TERMINAL
            or right.completion is not CompletionState.TERMINAL
        ):
            return GateResult.active(
                comparison.id,
                Verdict.INCONCLUSIVE,
                category=evidence_category,
                reason="missing_or_incomplete_attempt",
            )
        if not left.terminal_success or not right.terminal_success:
            if comparison.comparison.diagnostic:
                return GateResult.active(
                    comparison.id,
                    Verdict.INCONCLUSIVE,
                    category=evidence_category,
                    reason="failed_matched_control",
                )
            # Terminal failures are adjudicated by the status gate; the paired
            # ratio simply drops the seed and scores the remaining clean pairs.
            continue
        left_state, left_value = _operand(left, comparison.comparison.metric)
        right_state, right_value = _operand(right, comparison.comparison.metric)
        if (
            left_state != "finite"
            or right_state != "finite"
            or right_value is None
            or left_value is None
        ):
            return GateResult.active(
                comparison.id,
                Verdict.INCONCLUSIVE,
                category=evidence_category,
                reason="missing_or_invalid_operand",
            )
        ratios.append(left_value / max(right_value, 1e-12))
        contributing.extend((left.run_id, right.run_id))
    if len(ratios) < comparison.comparison.min_pairs:
        return GateResult.active(
            comparison.id,
            Verdict.INCONCLUSIVE,
            category=evidence_category,
            reason="insufficient_pairs",
        )
    observed = _aggregate(tuple(ratios), comparison.comparison.aggregation)
    if observed is None or not math.isfinite(observed):
        return GateResult.active(
            comparison.id,
            Verdict.INCONCLUSIVE,
            category=evidence_category,
            reason="undefined_aggregation",
        )
    return GateResult.active(
        comparison.id,
        Verdict.PASS if observed >= comparison.comparison.threshold else Verdict.FAIL,
        category=scored_category,
        observed=observed,
        threshold=comparison.comparison.threshold,
        contributing_run_ids=contributing,
    )


def aggregate_verdict(results: Iterable[GateResult]) -> Verdict:
    """Return the conservative aggregate while preserving skipped rows externally."""
    active = tuple(
        result
        for result in results
        if result.applicability is RuleApplicability.ACTIVE
        and result.category != "diagnostic_comparison"
    )
    if any(result.verdict is Verdict.FAIL for result in active):
        return Verdict.FAIL
    if any(result.verdict is Verdict.INCONCLUSIVE for result in active):
        return Verdict.INCONCLUSIVE
    if any(result.verdict is Verdict.PASS for result in active):
        return Verdict.PASS
    return Verdict.INCONCLUSIVE
