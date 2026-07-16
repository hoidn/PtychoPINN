"""Versioned compatibility mapping for the legacy ``object_big`` switch."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_object_compatibility_imports_without_tensorflow_or_torch():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_frameworks(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError(f"object compatibility imported TensorFlow: {name}")
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"object compatibility imported Torch: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_frameworks

from ptycho.object_compatibility import ObjectCompatibilitySpec

assert ObjectCompatibilitySpec.__module__ == "ptycho.object_compatibility"
assert not any(
    name == "tensorflow" or name.startswith("tensorflow.")
    or name == "torch" or name.startswith("torch.")
    for name in sys.modules
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("object_big", [False, True])
@pytest.mark.parametrize("training_patch_weighting", ["central_mask", "uniform", "probe"])
@pytest.mark.parametrize("pad_object", [False, True])
@pytest.mark.parametrize("probe_big", [False, True])
def test_legacy_object_fields_map_and_round_trip_exactly(
    object_big,
    training_patch_weighting,
    pad_object,
    probe_big,
):
    from ptycho.object_compatibility import (
        LegacyObjectFields,
        resolve_object_compatibility_spec,
    )

    legacy = LegacyObjectFields(
        object_big=object_big,
        training_patch_weighting=training_patch_weighting,
        pad_object=pad_object,
        probe_big=probe_big,
    )
    spec = resolve_object_compatibility_spec(legacy)

    if object_big:
        assert spec.layout == "grouped_patch_components_v1"
        assert spec.training_canvas == "relative_overlap_canvas_v1"
        expected_mode = (
            "central_mask_overlap_v1"
            if training_patch_weighting == "central_mask"
            else "weighted_overlap_v1"
        )
    else:
        assert spec.layout == "single_patch_components_v1"
        assert spec.training_canvas == "independent_patch_v1"
        expected_mode = "pass_through_v1"

    assert spec.training_assembly.mode == expected_mode
    assert spec.training_assembly.configured_weighting == training_patch_weighting
    assert spec.pad_object is pad_object
    assert spec.probe_big is probe_big
    assert spec.to_legacy_fields() == legacy


def test_object_compatibility_payload_round_trip_is_exact_and_versioned():
    from ptycho.object_compatibility import (
        CURRENT_OBJECT_COMPATIBILITY_VERSION,
        LegacyObjectFields,
        ObjectCompatibilitySpec,
        resolve_object_compatibility_spec,
    )

    spec = resolve_object_compatibility_spec(
        LegacyObjectFields(
            object_big=True,
            training_patch_weighting="probe",
            pad_object=False,
            probe_big=True,
        )
    )
    payload = spec.to_payload()

    assert payload == {
        "schema_version": CURRENT_OBJECT_COMPATIBILITY_VERSION,
        "layout": "grouped_patch_components_v1",
        "training_canvas": "relative_overlap_canvas_v1",
        "training_assembly": {
            "mode": "weighted_overlap_v1",
            "configured_weighting": "probe",
        },
        "pad_object": False,
        "probe_big": True,
    }
    assert ObjectCompatibilitySpec.from_payload(payload) == spec


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "object-compatibility-v999", "schema"),
        ("layout", "unknown", "layout"),
        ("training_canvas", "unknown", "canvas"),
    ],
)
def test_object_compatibility_payload_rejects_unknown_identity(field, value, match):
    from ptycho.object_compatibility import (
        LegacyObjectFields,
        ObjectCompatibilitySpec,
        resolve_object_compatibility_spec,
    )

    payload = resolve_object_compatibility_spec(
        LegacyObjectFields(False, "central_mask", False, False)
    ).to_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        ObjectCompatibilitySpec.from_payload(payload)


def test_object_compatibility_rejects_unknown_payload_keys():
    from ptycho.object_compatibility import (
        LegacyObjectFields,
        ObjectCompatibilitySpec,
        resolve_object_compatibility_spec,
    )

    payload = resolve_object_compatibility_spec(
        LegacyObjectFields(False, "central_mask", False, False)
    ).to_payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        ObjectCompatibilitySpec.from_payload(payload)


def test_object_compatibility_rejects_contradictory_axes_and_legacy_fields():
    from ptycho.object_compatibility import (
        CURRENT_OBJECT_COMPATIBILITY_VERSION,
        LegacyObjectFields,
        ObjectCompatibilitySpec,
        reconcile_object_compatibility,
        resolve_object_compatibility_spec,
    )
    from ptycho.reconstruction_policy import TrainingAssemblySpec

    with pytest.raises(ValueError, match="single_patch_components_v1.*independent"):
        ObjectCompatibilitySpec(
            schema_version=CURRENT_OBJECT_COMPATIBILITY_VERSION,
            layout="single_patch_components_v1",
            training_canvas="relative_overlap_canvas_v1",
            training_assembly=TrainingAssemblySpec(
                mode="pass_through_v1",
                configured_weighting="central_mask",
            ),
            pad_object=False,
            probe_big=False,
        )

    legacy = LegacyObjectFields(False, "uniform", False, True)
    spec = resolve_object_compatibility_spec(legacy)
    contradictory = LegacyObjectFields(True, "uniform", False, True)
    with pytest.raises(ValueError, match="conflicts with versioned object compatibility"):
        reconcile_object_compatibility(spec, contradictory)
