"""Tests for strict tiled reconstruction ordering and geometry."""

from dataclasses import fields
import numpy as np
import pytest
from types import SimpleNamespace


def test_canonicalize_tiled_patch_order_uses_scan_identity_not_loader_order():
    from ptycho_torch.inference import _canonicalize_tiled_patch_order

    loader_ids = np.asarray([2, 0, 3, 1], dtype=np.int64)
    patches = np.stack(
        [np.full((2, 2), scan_id, dtype=np.complex64) for scan_id in loader_ids]
    )
    expected_x = np.asarray([0.5, 2.5, 0.5, 2.5])
    expected_y = np.asarray([0.5, 0.5, 2.5, 2.5])
    loader_coords = np.column_stack(
        [expected_x[loader_ids], expected_y[loader_ids]]
    )

    ordered, ordered_ids = _canonicalize_tiled_patch_order(
        patches,
        loader_ids,
        loader_coords,
        expected_x=expected_x,
        expected_y=expected_y,
    )

    np.testing.assert_array_equal(ordered_ids, np.arange(4))
    np.testing.assert_array_equal(ordered[:, 0, 0], np.arange(4))


@pytest.mark.parametrize(
    "ids, message",
    [
        ([0, 1, 1, 3], "bijection"),
        ([0, 1, 2], "bijection"),
    ],
)
def test_canonicalize_tiled_patch_order_rejects_incomplete_identity(ids, message):
    from ptycho_torch.inference import _canonicalize_tiled_patch_order

    ids = np.asarray(ids, dtype=np.int64)
    patches = np.ones((ids.size, 2, 2), dtype=np.complex64)
    coordinates = np.zeros((ids.size, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        _canonicalize_tiled_patch_order(
            patches,
            ids,
            coordinates,
            expected_x=np.asarray([0.5, 2.5, 0.5, 2.5]),
            expected_y=np.asarray([0.5, 0.5, 2.5, 2.5]),
        )


def test_tiled_public_adapter_requests_tiled_bundle_identity(
    tmp_path,
    monkeypatch,
):
    from ptycho_torch import inference
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.workflows import components

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"bundle")
    test_npz = tmp_path / "test.npz"
    test_npz.write_bytes(b"npz")
    model = SimpleNamespace(
        data_config=object(),
        inference_config=InferenceConfig(),
    )
    monkeypatch.setattr("ptycho_torch.workflows.bundle_io.load_inference_bundle_torch",
        lambda *_args, **_kwargs: ({"diffraction_to_obj": model}, {}),
    )
    captured = {}

    def fake_identity(*_args, **_kwargs):
        pass

    def fake_workflow(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(
        inference,
        "_validate_loaded_reconstruction_identity",
        fake_identity,
    )
    monkeypatch.setattr(
        inference,
        "validate_bundle_matches_workflow",
        fake_workflow,
    )
    expected = SimpleNamespace(
        inference=SimpleNamespace(reconstruction_method="tiled")
    )

    with pytest.raises(RuntimeError, match="stop"):
        inference.reconstruct_npz_tiled(
            bundle,
            test_npz,
            run_root=tmp_path,
            expected_workflow=expected,
        )

    assert captured["reconstruction_method"] == "tiled"


def test_tiled_public_adapter_crops_restores_gauge_and_authenticates_mmap_coords(
    tmp_path,
    monkeypatch,
):
    from ptycho_torch import inference
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.workflows import components

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"bundle")
    test_npz = tmp_path / "test.npz"
    test_npz.write_bytes(b"staged-by-fake-base")
    runtime_inference = InferenceConfig(
        middle_trim=32,
        patch_weighting="uniform",
        varpro_scaling=False,
    )
    model = SimpleNamespace(
        data_config=object(),
        inference_config=runtime_inference,
    )
    monkeypatch.setattr("ptycho_torch.workflows.bundle_io.load_inference_bundle_torch",
        lambda *_args, **_kwargs: ({"diffraction_to_obj": model}, {}),
    )
    monkeypatch.setattr(
        inference,
        "_validate_loaded_reconstruction_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        inference,
        "validate_bundle_matches_workflow",
        lambda *_args, **_kwargs: None,
    )

    normalized = np.block(
        [
            [np.ones((2, 2)), np.full((2, 2), 2.0)],
            [np.full((2, 2), 3.0), np.full((2, 2), 4.0)],
        ]
    ).astype(np.complex64)
    padded = np.zeros((6, 6), dtype=np.complex64)
    padded[1:5, 1:5] = normalized
    weights = np.zeros((6, 6), dtype=np.float32)
    weights[1:5, 1:5] = 1.0
    expected_x = np.asarray([2.0, 4.0, 2.0, 4.0])
    expected_y = np.asarray([2.0, 2.0, 4.0, 4.0])
    loader_ids = np.asarray([[2], [0], [3], [1]], dtype=np.int64)
    loader_coords = np.stack(
        [
            np.asarray([expected_x[index], expected_y[index]])
            for index in loader_ids.reshape(-1)
        ]
    ).reshape(4, 1, 2)
    diagnostics = {
        "accepted_patches": 4,
        "total_patches": 4,
        "used_scan_ids": [0, 1, 2, 3],
        "used_center_scan_ids": [0, 1, 2, 3],
        "expected_scan_ids": [0, 1, 2, 3],
        "filtered_eligible_scan_ids": [0, 1, 2, 3],
        "s1": 1.0,
        "s2": 1.0,
        "effective_precision": "32-true",
        "count_metrics": {
            "status": "not_applicable",
            "reason": "legacy_normalized_amplitude",
        },
    }
    base = SimpleNamespace(
        complex_canvas=padded,
        prescale_canvas=padded * np.complex64(0.5),
        canvas_weights=weights,
        channel_indices=loader_ids,
        channel_coordinates=loader_coords,
        source_metadata={
            "xcoords": expected_x,
            "ycoords": expected_y,
            "object_amplitude_scale": np.asarray(3.0, dtype=np.float64),
        },
        effective_data_config=object(),
        reassembly=SimpleNamespace(to_jsonable=lambda: dict(diagnostics)),
    )
    monkeypatch.setattr(
        inference,
        "_stage_and_construct_reconstruction_dataset",
        lambda *_args, **_kwargs: (
            SimpleNamespace(), {}, SimpleNamespace(), SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        inference,
        "reconstruct_from_dataset",
        lambda *_args, **_kwargs: base,
    )
    simulation = SimpleNamespace(
        N=4,
        scan=SimpleNamespace(grid_size=(1, 1), outer_offset_test=4),
        object=SimpleNamespace(
            diffractions_per_object=4,
            image_size=(6, 6),
            patch_amplitude_normalization="mean_patch_max",
        ),
    )
    expected = SimpleNamespace(
        inference=SimpleNamespace(
            reconstruction_method="tiled",
            groups_per_center=1,
        ),
        simulation=SimpleNamespace(
            test=simulation,
            measurement_domain="normalized_amplitude",
        ),
    )

    result = inference.reconstruct_npz_tiled(
        bundle,
        test_npz,
        run_root=tmp_path,
        expected_workflow=expected,
    )

    np.testing.assert_array_equal(result.measurement_gauge_canvas, normalized)
    np.testing.assert_array_equal(result.complex_canvas, normalized * 3.0)
    np.testing.assert_array_equal(result.canvas_weights, np.ones((4, 4)))
    assert result.canvas_anchor == {
        "scan_com": [3.0, 3.0],
        "canvas_shape": [4, 4],
        "canvas_origin_offset": [-1.0, -1.0],
        "truth_origin": [1, 1],
        "assembly_method": "tiled_raster_v1",
    }
    assert result.reassembly["object_gauge"] == {
        "inference_canvas_before_publication": "split_normalized",
        "published_canvas": "raw_source",
        "published_scale_factor": 3.0,
        "count_diagnostics_canvas": "split_normalized",
    }
    assert result.reassembly["count_metrics"] == diagnostics["count_metrics"]


def test_tiled_public_adapter_rejects_mmap_coordinate_identity_drift():
    """The tiled guard must inspect actual mmap coordinates, not source lookup."""

    from ptycho_torch.inference import _canonicalize_tiled_patch_order

    ids = np.arange(4, dtype=np.int64)
    expected_x = np.asarray([2.0, 4.0, 2.0, 4.0])
    expected_y = np.asarray([2.0, 2.0, 4.0, 4.0])
    actual = np.column_stack((expected_x, expected_y))
    actual[2, 0] += 1.0

    with pytest.raises(ValueError, match="mmap|coordinates|fixed-pitch"):
        _canonicalize_tiled_patch_order(
            np.ones((4, 2, 2), dtype=np.complex64),
            ids,
            actual,
            expected_x=expected_x,
            expected_y=expected_y,
        )


@pytest.mark.parametrize(
    "frame_order_recipe",
    ["object-major-v1", "coordinate-major-interleaved-v1"],
)
def test_tiled_public_adapter_runs_real_mmap_grouping_and_accumulator(
    tmp_path,
    monkeypatch,
    frame_order_recipe,
):
    import torch

    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions
    from ptycho.workflows.synthetic_config import (
        materialize_data_config,
        resolve_synthetic_workflow,
    )
    from ptycho.workflows.training import _torch_model_from_snapshot
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.config_params import InferenceConfig, TrainingConfig
    from ptycho_torch.inference import reconstruct_npz_tiled
    from ptycho_torch.model_spec import derive_model_spec
    from ptycho_torch.workflows import components

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "seed": 5,
                "train_patterns": 4,
                "test_patterns": 4,
                "frame_order_recipe": frame_order_recipe,
                "object": {
                    "patch_amplitude_normalization": "mean_patch_max",
                },
                "scan": {"position_layout": "fixed_pitch_raster"},
            },
            "model": {"amplitude_physics_gain": 1.0},
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 1,
                "neighbor_pool_size": 1,
            },
            "inference": {
                "batch_size": 4,
                "reconstruction_method": "tiled",
                "patch_weighting": "uniform",
                "groups_per_center": 1,
                "varpro_scaling": False,
            },
        }
    )
    acquisition = generate_flat_acquisitions(resolved, tmp_path / "datasets")
    data_config = materialize_data_config(resolved)
    snapshot_model_config = _torch_model_from_snapshot(resolved)
    model_spec = derive_model_spec(
        to_model_config(data_config, snapshot_model_config),
        snapshot_model_config,
        data_config,
    )
    model_config = model_spec.to_model_config()
    training_values = {
        item.name: getattr(resolved.training, item.name)
        for item in fields(TrainingConfig)
        if hasattr(resolved.training, item.name)
    }
    training_values.update(
        device="cpu",
        num_workers=0,
        training_groups=resolved.training.training_groups,
    )
    training_config = TrainingConfig(**training_values)
    inference_config = InferenceConfig(
        **{
            item.name: getattr(resolved.inference, item.name)
            for item in fields(InferenceConfig)
            if hasattr(resolved.inference, item.name)
        }
    )
    values = torch.arange(1, 5, dtype=torch.float32).view(4, 1, 1, 1)
    textures = torch.complex(
        values.expand(4, 1, 64, 64),
        (values / 10.0).expand(4, 1, 64, 64),
    )

    class FixedTextureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("textures", textures)
            self.data_config = data_config
            self.model_config = model_config
            self.training_config = training_config
            self.inference_config = inference_config
            self._model_spec = model_spec

        def forward_predict(self, intensity, positions, probe, input_scale):
            return self.textures[: intensity.shape[0]]

    model = FixedTextureModel()
    bundle = tmp_path / "training"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"strict-test-bundle")
    monkeypatch.setattr("ptycho_torch.workflows.bundle_io.load_inference_bundle_torch",
        lambda *_args, **_kwargs: ({"diffraction_to_obj": model}, {}),
    )

    result = reconstruct_npz_tiled(
        bundle,
        acquisition.test_path,
        run_root=tmp_path / "run",
        expected_workflow=resolved,
        dataset_manifest_path=acquisition.manifest_path,
        device="cpu",
        quiet=True,
    )

    np.testing.assert_array_equal(
        result.channel_indices,
        np.arange(4, dtype=np.int64).reshape(4, 1),
    )
    expected_coordinates = {
        "object-major-v1": np.asarray(
            [[32.0, 32.0], [42.0, 32.0], [32.0, 42.0], [42.0, 42.0]]
        ),
        "coordinate-major-interleaved-v1": np.asarray(
            [[32.0, 32.0], [32.0, 42.0], [42.0, 32.0], [42.0, 42.0]]
        ),
    }[frame_order_recipe]
    np.testing.assert_array_equal(
        result.channel_coordinates.reshape(4, 2),
        expected_coordinates,
    )
    tile_values = (
        ((1.0, 2.0), (3.0, 4.0))
        if frame_order_recipe == "object-major-v1"
        else ((1.0, 3.0), (2.0, 4.0))
    )
    expected = np.block(
        [
            [
                np.full((10, 10), tile_values[0][0] * (1.0 + 0.1j)),
                np.full((10, 10), tile_values[0][1] * (1.0 + 0.1j)),
            ],
            [
                np.full((10, 10), tile_values[1][0] * (1.0 + 0.1j)),
                np.full((10, 10), tile_values[1][1] * (1.0 + 0.1j)),
            ],
        ]
    ).astype(np.complex64)
    np.testing.assert_allclose(result.measurement_gauge_canvas, expected)
    np.testing.assert_allclose(
        result.complex_canvas,
        expected * result.reassembly["object_amplitude_scale"],
    )
    np.testing.assert_array_equal(result.canvas_weights, np.ones((20, 20)))
    assert result.reassembly["accepted_patches"] == 4
    assert result.reassembly["total_patches"] == 4
    assert result.reassembly["assembly_method"] == "tiled_raster_v1"
    assert result.reassembly["effective_tile_size"] == 10
    assert result.reassembly["lattice_shape"] == [2, 2]
    assert result.reassembly["lattice_pitch"] == [10.0, 10.0]
