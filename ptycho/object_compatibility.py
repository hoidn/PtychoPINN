"""Versioned mapping for the legacy object_big compatibility switch.

This module records the three independent meanings historically selected by
object_big without changing runtime routing or the public configuration
surface. It is a pure compatibility proof for later model/artifact migrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

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
