"""TensorFlow construction capability checks for the public object policy."""

from __future__ import annotations

import pytest


@pytest.fixture
def params_snapshot():
    from ptycho import params

    snapshot = dict(params.cfg)
    try:
        params.cfg.setdefault("intensity_scale", 1.0)
        params.cfg.setdefault("n_filters_scale", 2)
        params.cfg.setdefault("offset", 4)
        params.cfg.setdefault("probe.big", True)
        params.cfg.setdefault("pad_object", True)
        yield params
    finally:
        params.cfg.clear()
        params.cfg.update(snapshot)


@pytest.mark.parametrize(
    (
        "gridsize",
        "layout",
        "canvas",
        "expected_channels",
        "expected_layer_type",
    ),
    [
        (1, "single_patch", "independent", 1, "PadReconstructionLayer"),
        (
            2,
            "grouped_patches",
            "relative_overlap",
            4,
            "ReassemblePatchesLayer",
        ),
    ],
)
def test_tensorflow_constructs_both_supported_public_object_policies(
    params_snapshot,
    gridsize,
    layout,
    canvas,
    expected_channels,
    expected_layer_type,
):
    from ptycho import model

    params_snapshot.cfg.pop("object.big", None)
    autoencoder, diffraction_to_obj = model.create_model_with_gridsize(
        gridsize,
        64,
        object_layout=layout,
        training_canvas=canvas,
        training_patch_weighting="central_mask",
    )

    assert autoencoder.get_layer("amp").output.shape[-1] == expected_channels
    assert type(diffraction_to_obj.get_layer("padded_obj_2")).__name__ == (
        expected_layer_type
    )


@pytest.mark.parametrize("weighting", ["uniform", "probe"])
def test_tensorflow_rejects_unsupported_weighting_before_model_construction(
    params_snapshot,
    weighting,
):
    from ptycho import model

    params_snapshot.cfg.pop("object.big", None)
    with pytest.raises(ValueError, match="TensorFlow.*central_mask"):
        model.create_model_with_gridsize(
            2,
            64,
            object_layout="grouped_patches",
            training_canvas="relative_overlap",
            training_patch_weighting=weighting,
        )
