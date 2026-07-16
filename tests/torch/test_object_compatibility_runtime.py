"""Torch topology/forward adoption of the versioned object compatibility map."""

from __future__ import annotations

import pytest

from ptycho_torch.config_params import DataConfig, ModelConfig


@pytest.mark.parametrize(
    ("object_big", "expected_channels", "layout", "canvas", "merge_mode"),
    [
        (
            False,
            1,
            "single_patch_components_v1",
            "independent_patch_v1",
            "pass_through_v1",
        ),
        (
            True,
            4,
            "grouped_patch_components_v1",
            "relative_overlap_canvas_v1",
            "weighted_overlap_v1",
        ),
    ],
)
def test_model_topology_and_forward_share_object_compatibility_identity(
    object_big,
    expected_channels,
    layout,
    canvas,
    merge_mode,
):
    from ptycho_torch.model import Encoder, ForwardModel

    model = ModelConfig(
        C_model=4,
        C_forward=4,
        object_big=object_big,
        training_patch_weighting="probe",
        pad_object=False,
        probe_big=True,
    )
    data = DataConfig(N=64, C=4, grid_size=(2, 2))

    encoder = Encoder(model, data)
    forward = ForwardModel(model, data)

    assert encoder.filters[0] == expected_channels
    assert encoder.object_compatibility.layout == layout
    assert forward.object_compatibility.layout == layout
    assert forward.object_compatibility.training_canvas == canvas
    assert forward.training_assembly_spec.mode == merge_mode
    assert forward.training_assembly_spec is forward.object_compatibility.training_assembly
    assert forward.object_big is object_big
