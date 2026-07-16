"""Versioned mapping for the legacy object_big compatibility switch.

This module records the three independent meanings historically selected by
object_big without changing runtime routing or the public configuration
surface. It is a pure compatibility proof for later model/artifact migrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias
import warnings

from ptycho.reconstruction_policy import (
    TrainingAssemblySpec,
    resolve_training_assembly_spec,
)


ObjectLayout: TypeAlias = Literal[
    "single_patch_components_v1",
    "grouped_patch_components_v1",
]
TrainingCanvas: TypeAlias = Literal[
    "independent_patch_v1",
    "relative_overlap_canvas_v1",
]
PublicObjectLayout: TypeAlias = Literal["single_patch", "grouped_patches"]
PublicTrainingCanvas: TypeAlias = Literal["independent", "relative_overlap"]
PublicTrainingPatchWeighting: TypeAlias = Literal[
    "central_mask",
    "uniform",
    "probe",
]

CURRENT_OBJECT_COMPATIBILITY_VERSION = "object-compatibility-v1"

_OBJECT_LAYOUTS = {
    "single_patch_components_v1",
    "grouped_patch_components_v1",
}
_TRAINING_CANVASES = {
    "independent_patch_v1",
    "relative_overlap_canvas_v1",
}
_TRAINING_PATCH_WEIGHTINGS = {"central_mask", "uniform", "probe"}
_PUBLIC_LAYOUTS = {"single_patch", "grouped_patches"}
_PUBLIC_CANVASES = {"independent", "relative_overlap"}
_PUBLIC_BACKENDS = {None, "tensorflow", "torch"}


@dataclass(frozen=True)
class LegacyObjectFields:
    """Exact legacy inputs needed to derive the separated identities."""

    object_big: bool
    training_patch_weighting: str
    pad_object: bool
    probe_big: bool

    def __post_init__(self) -> None:
        for name in ("object_big", "pad_object", "probe_big"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.training_patch_weighting not in _TRAINING_PATCH_WEIGHTINGS:
            raise ValueError(
                "training_patch_weighting must be 'central_mask', 'uniform', or "
                f"'probe', got {self.training_patch_weighting!r}"
            )


@dataclass(frozen=True)
class ResolvedPublicObjectPolicy:
    """Fully materialized public policy plus its versioned interpretation."""

    object_big: bool
    object_layout: PublicObjectLayout
    training_canvas: PublicTrainingCanvas
    training_patch_weighting: PublicTrainingPatchWeighting
    compatibility: "ObjectCompatibilitySpec"


@dataclass(frozen=True)
class ObjectCompatibilitySpec:
    """Separated, frozen interpretation of one legacy object configuration."""

    schema_version: str
    layout: ObjectLayout
    training_canvas: TrainingCanvas
    training_assembly: TrainingAssemblySpec
    pad_object: bool
    probe_big: bool

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_OBJECT_COMPATIBILITY_VERSION:
            raise ValueError(
                f"unsupported object compatibility schema {self.schema_version!r}; "
                f"expected {CURRENT_OBJECT_COMPATIBILITY_VERSION!r}"
            )
        if self.layout not in _OBJECT_LAYOUTS:
            raise ValueError(f"unsupported object layout {self.layout!r}")
        if self.training_canvas not in _TRAINING_CANVASES:
            raise ValueError(
                f"unsupported object training canvas {self.training_canvas!r}"
            )
        if not isinstance(self.training_assembly, TrainingAssemblySpec):
            raise TypeError("training_assembly must be a TrainingAssemblySpec")
        for name in ("pad_object", "probe_big"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")

        grouped = self.layout == "grouped_patch_components_v1"
        if grouped:
            if self.training_canvas != "relative_overlap_canvas_v1":
                raise ValueError(
                    "grouped_patch_components_v1 requires "
                    "relative_overlap_canvas_v1"
                )
        elif self.training_canvas != "independent_patch_v1":
            raise ValueError(
                "single_patch_components_v1 requires independent_patch_v1"
            )

        expected_assembly = resolve_training_assembly_spec(
            grouped,
            self.training_assembly.configured_weighting,
        )
        if self.training_assembly != expected_assembly:
            raise ValueError(
                f"{self.layout} conflicts with training assembly "
                f"{self.training_assembly.mode!r}; expected "
                f"{expected_assembly.mode!r}"
            )

    def to_legacy_fields(self) -> LegacyObjectFields:
        return LegacyObjectFields(
            object_big=self.layout == "grouped_patch_components_v1",
            training_patch_weighting=self.training_assembly.configured_weighting,
            pad_object=self.pad_object,
            probe_big=self.probe_big,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "training_canvas": self.training_canvas,
            "training_assembly": {
                "mode": self.training_assembly.mode,
                "configured_weighting": self.training_assembly.configured_weighting,
            },
            "pad_object": self.pad_object,
            "probe_big": self.probe_big,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ObjectCompatibilitySpec":
        if not isinstance(payload, Mapping):
            raise TypeError("object compatibility payload must be a mapping")
        expected = {
            "schema_version",
            "layout",
            "training_canvas",
            "training_assembly",
            "pad_object",
            "probe_big",
        }
        received = set(payload)
        if received != expected:
            raise ValueError(
                "object compatibility payload keys are not exact; "
                f"missing={sorted(expected - received)}, "
                f"unknown={sorted(received - expected)}"
            )
        assembly_payload = payload["training_assembly"]
        if not isinstance(assembly_payload, Mapping):
            raise ValueError("training_assembly payload must be a mapping")
        assembly_expected = {"mode", "configured_weighting"}
        assembly_received = set(assembly_payload)
        if assembly_received != assembly_expected:
            raise ValueError(
                "training_assembly payload keys are not exact; "
                f"missing={sorted(assembly_expected - assembly_received)}, "
                f"unknown={sorted(assembly_received - assembly_expected)}"
            )
        return cls(
            schema_version=payload["schema_version"],
            layout=payload["layout"],
            training_canvas=payload["training_canvas"],
            training_assembly=TrainingAssemblySpec(
                mode=assembly_payload["mode"],
                configured_weighting=assembly_payload["configured_weighting"],
            ),
            pad_object=payload["pad_object"],
            probe_big=payload["probe_big"],
        )


def resolve_object_compatibility_spec(
    legacy: LegacyObjectFields,
) -> ObjectCompatibilitySpec:
    """Derive the exact separated identities selected by legacy fields."""
    if not isinstance(legacy, LegacyObjectFields):
        raise TypeError("legacy must be a LegacyObjectFields")
    if legacy.object_big:
        layout: ObjectLayout = "grouped_patch_components_v1"
        training_canvas: TrainingCanvas = "relative_overlap_canvas_v1"
    else:
        layout = "single_patch_components_v1"
        training_canvas = "independent_patch_v1"
    return ObjectCompatibilitySpec(
        schema_version=CURRENT_OBJECT_COMPATIBILITY_VERSION,
        layout=layout,
        training_canvas=training_canvas,
        training_assembly=resolve_training_assembly_spec(
            legacy.object_big,
            legacy.training_patch_weighting,
        ),
        pad_object=legacy.pad_object,
        probe_big=legacy.probe_big,
    )


def resolve_public_object_policy(
    *,
    object_big: bool | None,
    object_layout: PublicObjectLayout | None,
    training_canvas: PublicTrainingCanvas | None,
    training_patch_weighting: PublicTrainingPatchWeighting | None,
    pad_object: bool,
    probe_big: bool,
    backend: Literal["tensorflow", "torch"] | None = None,
    warn_deprecated: bool = True,
) -> ResolvedPublicObjectPolicy:
    """Resolve unset-aware public fields into the closed v1 identity."""
    if backend not in _PUBLIC_BACKENDS:
        raise ValueError(f"unsupported object-policy backend {backend!r}")
    if object_big is not None and type(object_big) is not bool:
        raise TypeError("object_big must be bool or None")
    if object_layout is not None and object_layout not in _PUBLIC_LAYOUTS:
        raise ValueError(f"unsupported object_layout {object_layout!r}")
    if training_canvas is not None and training_canvas not in _PUBLIC_CANVASES:
        raise ValueError(f"unsupported training_canvas {training_canvas!r}")
    if (
        training_patch_weighting is not None
        and training_patch_weighting not in _TRAINING_PATCH_WEIGHTINGS
    ):
        raise ValueError(
            "training_patch_weighting must be 'central_mask', 'uniform', or "
            f"'probe', got {training_patch_weighting!r}"
        )
    for name, value in (("pad_object", pad_object), ("probe_big", probe_big)):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool")

    if object_big is not None and warn_deprecated:
        warnings.warn(
            "ModelConfig.object_big is deprecated; use object_layout and "
            "training_canvas",
            DeprecationWarning,
            stacklevel=2,
        )

    layout_is_set = object_layout is not None
    canvas_is_set = training_canvas is not None
    if layout_is_set != canvas_is_set:
        raise ValueError(
            "object_layout and training_canvas must be supplied together"
        )

    if not layout_is_set:
        resolved_big = True if object_big is None else object_big
        resolved_layout: PublicObjectLayout = (
            "grouped_patches" if resolved_big else "single_patch"
        )
        resolved_canvas: PublicTrainingCanvas = (
            "relative_overlap" if resolved_big else "independent"
        )
    else:
        pair = (object_layout, training_canvas)
        if pair == ("single_patch", "independent"):
            resolved_big = False
        elif pair == ("grouped_patches", "relative_overlap"):
            resolved_big = True
        else:
            raise ValueError(
                "unsupported object_layout/training_canvas pair "
                f"{pair!r}"
            )
        resolved_layout = object_layout
        resolved_canvas = training_canvas
        if object_big is not None and object_big is not resolved_big:
            raise ValueError(
                f"object_big={object_big!r} conflicts with "
                f"object_layout={object_layout!r}"
            )

    resolved_weighting: PublicTrainingPatchWeighting = (
        "central_mask"
        if training_patch_weighting is None
        else training_patch_weighting
    )
    if backend == "tensorflow" and resolved_weighting != "central_mask":
        raise ValueError(
            "TensorFlow supports training_patch_weighting='central_mask' only; "
            f"got {resolved_weighting!r}"
        )

    compatibility = resolve_object_compatibility_spec(
        LegacyObjectFields(
            object_big=resolved_big,
            training_patch_weighting=resolved_weighting,
            pad_object=pad_object,
            probe_big=probe_big,
        )
    )
    return ResolvedPublicObjectPolicy(
        object_big=resolved_big,
        object_layout=resolved_layout,
        training_canvas=resolved_canvas,
        training_patch_weighting=resolved_weighting,
        compatibility=compatibility,
    )


def reconcile_object_compatibility(
    spec: ObjectCompatibilitySpec,
    legacy: LegacyObjectFields,
) -> ObjectCompatibilitySpec:
    """Fail closed when dual legacy/new representations disagree."""
    if not isinstance(spec, ObjectCompatibilitySpec):
        raise TypeError("spec must be an ObjectCompatibilitySpec")
    if not isinstance(legacy, LegacyObjectFields):
        raise TypeError("legacy must be a LegacyObjectFields")
    if spec.to_legacy_fields() != legacy:
        raise ValueError(
            f"legacy object fields {legacy!r} conflicts with versioned object "
            f"compatibility {spec.to_legacy_fields()!r}"
        )
    return spec
