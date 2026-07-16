"""Focused public contracts for the Slice 7D object-policy migration."""

from __future__ import annotations

import pytest


def test_default_public_object_policy_materializes_established_behavior():
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    raw = ModelConfig()
    assert raw.object_big is None
    assert raw.object_layout is None
    assert raw.training_canvas is None
    assert raw.training_patch_weighting is None

    resolved = resolve_model_object_policy(raw)

    assert resolved.object_big is True
    assert resolved.object_layout == "grouped_patches"
    assert resolved.training_canvas == "relative_overlap"
    assert resolved.training_patch_weighting == "central_mask"
    assert raw.object_big is None


@pytest.mark.parametrize(
    ("object_big", "layout", "canvas"),
    [
        (False, "single_patch", "independent"),
        (True, "grouped_patches", "relative_overlap"),
    ],
)
def test_legacy_only_public_input_maps_exactly_with_deprecation_signal(
    object_big,
    layout,
    canvas,
):
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    with pytest.warns(DeprecationWarning, match="object_big"):
        resolved = resolve_model_object_policy(ModelConfig(object_big=object_big))

    assert resolved.object_big is object_big
    assert resolved.object_layout == layout
    assert resolved.training_canvas == canvas
    assert resolved.training_patch_weighting == "central_mask"


@pytest.mark.parametrize(
    ("layout", "canvas", "object_big"),
    [
        ("single_patch", "independent", False),
        ("grouped_patches", "relative_overlap", True),
    ],
)
def test_new_only_public_input_derives_legacy_boolean(
    layout,
    canvas,
    object_big,
):
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    resolved = resolve_model_object_policy(
        ModelConfig(
            object_layout=layout,
            training_canvas=canvas,
            training_patch_weighting="probe",
        )
    )

    assert resolved.object_big is object_big
    assert resolved.object_layout == layout
    assert resolved.training_canvas == canvas
    assert resolved.training_patch_weighting == "probe"


def test_coherent_dual_public_input_is_accepted():
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    with pytest.warns(DeprecationWarning, match="object_big"):
        resolved = resolve_model_object_policy(
            ModelConfig(
                object_big=False,
                object_layout="single_patch",
                training_canvas="independent",
                training_patch_weighting="uniform",
            )
        )

    assert resolved.object_big is False
    assert resolved.training_patch_weighting == "uniform"


def test_conflicting_dual_public_input_fails_closed():
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    with pytest.warns(DeprecationWarning, match="object_big"):
        with pytest.raises(ValueError, match="object_big.*conflicts"):
            resolve_model_object_policy(
                ModelConfig(
                    object_big=True,
                    object_layout="single_patch",
                    training_canvas="independent",
                )
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"object_layout": "single_patch"},
        {"training_canvas": "independent"},
        {
            "object_layout": "single_patch",
            "training_canvas": "relative_overlap",
        },
        {
            "object_layout": "grouped_patches",
            "training_canvas": "independent",
        },
    ],
)
def test_partial_or_unsupported_public_policy_fails_closed(kwargs):
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    with pytest.raises(ValueError, match="object_layout|training_canvas|unsupported"):
        resolve_model_object_policy(ModelConfig(**kwargs))


def test_legacy_bridge_projects_derived_object_big_exactly():
    from ptycho.config.config import (
        ModelConfig,
        dataclass_to_legacy_dict,
    )

    payload = dataclass_to_legacy_dict(
        ModelConfig(
            object_layout="single_patch",
            training_canvas="independent",
            training_patch_weighting="probe",
        )
    )

    assert payload["object.big"] is False
    assert payload["object_layout"] == "single_patch"
    assert payload["training_canvas"] == "independent"
    assert payload["training_patch_weighting"] == "probe"
    assert "object_big" not in payload


def test_tensorflow_backend_accepts_only_central_mask_weighting():
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    resolved = resolve_model_object_policy(
        ModelConfig(
            object_layout="grouped_patches",
            training_canvas="relative_overlap",
            training_patch_weighting="central_mask",
        ),
        backend="tensorflow",
    )
    assert resolved.training_patch_weighting == "central_mask"

    for weighting in ("uniform", "probe"):
        with pytest.raises(ValueError, match="TensorFlow.*central_mask"):
            resolve_model_object_policy(
                ModelConfig(
                    object_layout="grouped_patches",
                    training_canvas="relative_overlap",
                    training_patch_weighting=weighting,
                ),
                backend="tensorflow",
            )


def test_torch_backend_accepts_all_public_weightings():
    from ptycho.config.config import ModelConfig, resolve_model_object_policy

    for weighting in ("central_mask", "uniform", "probe"):
        resolved = resolve_model_object_policy(
            ModelConfig(
                object_layout="grouped_patches",
                training_canvas="relative_overlap",
                training_patch_weighting=weighting,
            ),
            backend="torch",
        )
        assert resolved.training_patch_weighting == weighting
