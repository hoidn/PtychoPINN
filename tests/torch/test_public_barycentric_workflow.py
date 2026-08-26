"""Public strict-load mmap barycentric reconstruction contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model_spec import derive_model_spec
from ptycho_torch.reassembly_diagnostics import (
    ReassemblyDiagnostics,
    not_applicable,
)


def _write_flat_npz(
    path: Path,
    *,
    N: int = 8,
    count: int = 4,
    diffraction_key: str = "diff3d",
    probe_layout: str = "single",
) -> None:
    rng = np.random.default_rng(41)
    diff3d = rng.random((count, N, N), dtype=np.float32)
    probe = (
        rng.random((N, N), dtype=np.float32) + 1j * rng.random((N, N), dtype=np.float32)
    ).astype(np.complex64)
    obj = np.ones((N * 2, N * 2), dtype=np.complex64)
    if probe_layout == "modes":
        probe = np.stack((probe, probe * np.complex64(0.5)), axis=0)
    elif probe_layout == "legacy-singleton":
        probe = probe[..., None]
    elif probe_layout != "single":
        raise ValueError(f"unsupported test probe layout: {probe_layout}")
    np.savez(
        path,
        **{diffraction_key: diff3d},
        xcoords=np.arange(count, dtype=np.float64),
        ycoords=np.arange(count, dtype=np.float64),
        probeGuess=probe,
        objectGuess=obj,
        scan_index=np.arange(count, dtype=np.int64),
    )


def _model_stub(
    *,
    N: int = 8,
    C: int = 4,
    physics_forward_mode: str = "amplitude",
    training_groups: int | None = None,
):
    data_config = DataConfig(
        N=N,
        neighbor_count=max(C, 4),
        gridsize=2,
        n_raw_frames_selected=4096,
        subsample_seed=31415,
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(
        architecture="cnn",
        object_layout="grouped_patches",
        training_canvas="relative_overlap",
        physics_forward_mode=physics_forward_mode,
    )
    canonical = to_model_config(data_config, model_config)
    model_spec = derive_model_spec(canonical, model_config, data_config)
    # The production loader reconstructs the live config from ModelSpec, which
    # also resolves derived compatibility defaults such as training weighting.
    model_config = model_spec.to_model_config()
    return SimpleNamespace(
        data_config=data_config,
        model_config=model_config,
        training_config=TrainingConfig(
            device="cpu", num_workers=0, training_groups=training_groups
        ),
        inference_config=InferenceConfig(
            patch_weighting="probe",
            varpro_scaling=False,
            middle_trim=4,
            batch_size=1,
        ),
        _model_spec=model_spec,
        eval=MagicMock(),
        to=MagicMock(),
    )


@dataclass(frozen=True)
class _ExpectedWorkflow:
    data: object
    model: object
    training: object
    inference: object
    workflow: object


def _expected_workflow_for_model(
    model,
    *,
    groups_per_center: int = 1,
    training_groups: int | None = None,
):
    training_values = {
        name: value
        for name, value in vars(model.training_config).items()
        if name != "training_groups"
    }
    return _ExpectedWorkflow(
        data=SimpleNamespace(**vars(model.data_config)),
        model=SimpleNamespace(
            **vars(model.model_config),
            amplitude_physics_gain_provenance=None,
        ),
        training=SimpleNamespace(
            **training_values,
            training_groups=training_groups,
        ),
        inference=SimpleNamespace(
            **vars(model.inference_config),
            reconstruction_method="barycentric",
            groups_per_center=groups_per_center,
        ),
        workflow=SimpleNamespace(
            num_workers=0,
            precision="32-true",
            accelerator="cpu",
        ),
    )


def _diagnostics(*, shape: tuple[int, int] = (10, 10)) -> ReassemblyDiagnostics:
    return ReassemblyDiagnostics.legacy_not_applicable(
        effective_probe_mask=torch.ones((8, 8)),
        inference_time=0.1,
        assembly_time=0.2,
        solve_time=0.0,
        s1=1.0,
        s2=1.0,
        scale_profile="legacy_v1",
        canvas_anchor={
            "scan_com": [1.5, 1.5],
            "canvas_shape": list(shape),
            "canvas_origin_offset": [0.0, 0.0],
        },
        canvas_weights=torch.ones(shape, dtype=torch.float32),
        accepted_patches=4,
        total_patches=4,
        count_metrics=not_applicable(),
        used_scan_ids=(0, 1, 2, 3),
        used_center_scan_ids=(0,),
        expected_scan_ids=(0, 1, 2, 3),
        filtered_eligible_scan_ids=(0,),
        effective_precision="32-true",
    )


def _dataset_stub():
    coords = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]],
        dtype=torch.float32,
    )
    return SimpleNamespace(
        mmap_ptycho={
            "images": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "nn_indices": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            "coords_global": coords,
        }
    )


def _install_stubs(monkeypatch, model, *, reconstruct=None):
    from ptycho_torch import dataloader, inference, reassembly
    from ptycho_torch.workflows import components

    loader = MagicMock(
        return_value=(
            {"diffraction_to_obj": model},
            {"amplitude_physics_gain_record": None},
        )
    )
    dataset_paths: list[Path] = []
    dataset_configs: list[DataConfig] = []

    def build_dataset(ptycho_dir, _model_config, data_config, *_args, **_kwargs):
        staged = Path(ptycho_dir)
        dataset_paths.append(staged)
        dataset_configs.append(data_config)
        assert [path.name for path in staged.iterdir()] == ["test.npz"]
        return _dataset_stub()

    if reconstruct is None:
        canvas = torch.complex(
            torch.ones((10, 10), dtype=torch.float32),
            torch.zeros((10, 10), dtype=torch.float32),
        )
        prescale = canvas * (2.0 + 0.0j)
        reconstruct = MagicMock(
            return_value=(canvas, SimpleNamespace(), _diagnostics(), prescale)
        )
    monkeypatch.setattr("ptycho_torch.workflows.bundle_io.load_inference_bundle_torch", loader)
    monkeypatch.setattr(dataloader, "PtychoDataset", build_dataset)
    monkeypatch.setattr(reassembly, "reconstruct_image_barycentric", reconstruct)
    monkeypatch.setattr(
        inference,
        "present_reconstruction_canvas",
        lambda canvas, _spec: (
            np.abs(np.asarray(canvas)),
            np.angle(np.asarray(canvas)),
        ),
    )
    return loader, dataset_paths, dataset_configs, reconstruct


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "training"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"strict-bundle")
    test_npz = tmp_path / "test.npz"
    _write_flat_npz(test_npz)
    run_root = tmp_path / "run"
    run_root.mkdir()
    return bundle, test_npz, run_root


def test_public_workflow_strict_loads_and_returns_frozen_snapshots(
    tmp_path, monkeypatch
):
    from ptycho_torch.inference import (
        BarycentricReconstructionResult,
        reconstruct_npz_barycentric,
    )

    bundle, test_npz, run_root = _paths(tmp_path)
    sentinel = run_root / "keep-me.txt"
    sentinel.write_text("caller-owned", encoding="utf-8")
    model = _model_stub()
    loader, dataset_paths, runtime_configs, reconstruct = _install_stubs(
        monkeypatch, model
    )

    result = reconstruct_npz_barycentric(
        bundle,
        test_npz,
        run_root=run_root,
        groups_per_center=1,
        device="cpu",
        quiet=True,
    )

    assert isinstance(result, BarycentricReconstructionResult)
    with pytest.raises(FrozenInstanceError):
        result.amplitude = np.zeros_like(result.amplitude)
    assert loader.call_count == 1
    assert loader.call_args.args[0] == bundle
    assert reconstruct.call_args.kwargs["structured_diagnostics"] is True
    forwarded_dataset = reconstruct.call_args.args[1]
    assert forwarded_dataset.mmap_ptycho["nn_indices"].tolist() == [[0, 1, 2, 3]]
    assert forwarded_dataset.mmap_ptycho["coords_global"].reshape(4, 2).tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    assert model.data_config.n_raw_frames_selected == 4096
    assert runtime_configs[0] is model.data_config
    assert result.effective_data_config is runtime_configs[0]
    assert result.complex_canvas.shape == (10, 10)
    assert result.prescale_canvas.shape == (10, 10)
    assert result.amplitude.shape == result.phase.shape == (10, 10)
    assert np.all(result.canvas_weights > 0)
    assert result.canvas_anchor["scan_com"] == [1.5, 1.5]
    assert result.channel_indices.dtype == np.int64
    assert result.channel_indices.tolist() == [[0, 1, 2, 3]]
    assert result.channel_indices.flags.writeable is False
    assert result.reassembly.used_scan_ids == (0, 1, 2, 3)
    assert sentinel.read_text(encoding="utf-8") == "caller-owned"
    assert all(not path.exists() for path in dataset_paths)
    assert not (run_root / "reconstruction" / "diagnostics.json").exists()


def test_workspace_cleanup_preserves_sentinel_when_reassembly_fails(
    tmp_path, monkeypatch
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    sentinel = run_root / "keep-me.txt"
    sentinel.write_text("caller-owned", encoding="utf-8")
    model = _model_stub()
    failing = MagicMock(side_effect=RuntimeError("reassembly exploded"))
    _, dataset_paths, _, _ = _install_stubs(monkeypatch, model, reconstruct=failing)

    with pytest.raises(RuntimeError, match="reassembly exploded"):
        reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root, quiet=True)

    assert sentinel.read_text(encoding="utf-8") == "caller-owned"
    assert all(not path.exists() for path in dataset_paths)


def test_bundle_modelspec_must_agree_with_dual_written_model_config(
    tmp_path, monkeypatch
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    # Channel twins are retired; provoke the ModelSpec/dual-write disagreement
    # through a surviving structural field instead.
    model.model_config = replace(model.model_config, n_filters_scale=7)
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    with pytest.raises(ValueError, match=r"ModelSpec.*n_filters_scale"):
        reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root)
    reconstruct.assert_not_called()


def test_npz_detector_shape_must_match_loaded_data_config(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    _write_flat_npz(test_npz, N=7)
    model = _model_stub(N=8)
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    with pytest.raises(ValueError, match=r"diff3d.*N=8"):
        reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root)
    reconstruct.assert_not_called()


def test_loaded_identity_must_agree_with_resolved_workflow(tmp_path, monkeypatch):
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub(N=8)
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)
    resolved = resolve_synthetic_workflow()

    with pytest.raises(ValueError, match=r"resolved_workflow\.data\."):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            expected_workflow=resolved,
        )
    reconstruct.assert_not_called()


def test_training_groups_alias_must_match_loaded_training_config(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub(training_groups=7)
    expected = _expected_workflow_for_model(model, training_groups=8)
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    with pytest.raises(
        ValueError, match=r"resolved_workflow\.training\.training_groups mismatch"
    ):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            expected_workflow=expected,
        )
    reconstruct.assert_not_called()


@pytest.mark.parametrize("runtime_groups", [2, 3])
def test_resolved_workflow_rejects_runtime_groups_per_center_drift(
    tmp_path, monkeypatch, runtime_groups
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub(training_groups=7)
    expected = _expected_workflow_for_model(
        model, groups_per_center=1, training_groups=7
    )
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    with pytest.raises(ValueError, match="groups_per_center"):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            expected_workflow=expected,
            groups_per_center=runtime_groups,
        )
    reconstruct.assert_not_called()


def test_resolved_workflow_rejects_runtime_inference_override_drift(
    tmp_path, monkeypatch
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub(training_groups=7)
    expected = _expected_workflow_for_model(model, training_groups=7)
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    with pytest.raises(ValueError, match=r"runtime inference.*patch_weighting"):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            expected_workflow=expected,
            inference_config=replace(model.inference_config, patch_weighting="uniform"),
        )
    reconstruct.assert_not_called()


def _write_dataset_manifest(
    manifest_path: Path,
    test_npz: Path,
    model,
    *,
    photons_per_pattern: float | None = None,
    npz_sha256: str | None = None,
    expected_workflow=None,
) -> None:
    from ptycho.simulation.identity import (
        array_sha256,
        canonical_sha256,
        file_sha256,
    )

    with np.load(test_npz, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    photons = (
        float(model.data_config.nphotons)
        if photons_per_pattern is None
        else float(photons_per_pattern)
    )
    array_hashes = {name: array_sha256(value) for name, value in arrays.items()}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    dtypes = {name: value.dtype.name for name, value in arrays.items()}
    split_recipe_sha256 = "1" * 64
    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": array_hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }
    manifest = {
        "schema_version": "flat-acquisition-manifest-v1",
        "storage_layout": "flat_acquisition_v1",
        "splits": {
            "test": {
                "artifact_path": test_npz.name,
                "npz_sha256": npz_sha256 or file_sha256(test_npz),
                "array_sha256": array_hashes,
                "split_recipe_sha256": split_recipe_sha256,
                "dataset_identity": dataset_identity,
                "dataset_sha256": canonical_sha256(dataset_identity),
                "shapes": shapes,
                "dtypes": dtypes,
                "measurement_identity": {
                    "scale_contract_version": "legacy_v1",
                    "measurement_domain": "normalized_amplitude",
                    "photons_per_pattern": photons,
                },
            }
        },
    }
    if expected_workflow is not None:
        from ptycho.config import (
            simulation_config_sha256,
            simulation_config_to_dict,
        )
        from ptycho.simulation.flat_acquisition import derive_seed_lineage
        from ptycho.workflows.synthetic_config import synthetic_workflow_to_dict

        semantic = synthetic_workflow_to_dict(expected_workflow)
        seed_lineage = derive_seed_lineage(expected_workflow.simulation.train.seed)
        test_simulation = expected_workflow.simulation.test
        manifest.update(
            profile=expected_workflow.profile,
            recipe_version=expected_workflow.recipe_version,
            simulation=semantic["simulation"],
            seed_lineage=seed_lineage,
        )
        manifest["splits"]["test"].update(
            simulation_config=simulation_config_to_dict(test_simulation),
            simulation_config_sha256=simulation_config_sha256(test_simulation),
            seed_lineage=seed_lineage,
            coordinate_seed=seed_lineage["test_coordinates"],
            detector_seed=seed_lineage["test_noise"],
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_flat_npz_compatibility_rejects_conflicting_diffraction_aliases(tmp_path):
    from ptycho_torch.inference import _validate_flat_npz

    path = tmp_path / "conflicting_aliases.npz"
    _write_flat_npz(path, count=3)
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["diffraction"] = arrays["diff3d"] + np.float32(1)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="conflicting diffraction"):
        _validate_flat_npz(path, _model_stub().data_config)


def test_flat_npz_compatibility_decodes_alias_layout_and_trailing_rows(tmp_path):
    from ptycho_torch.inference import _validate_flat_npz

    path = tmp_path / "legacy_trailing.npz"
    _write_flat_npz(path, count=3)
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["diffraction"] = np.transpose(arrays.pop("diff3d"), (1, 2, 0))
    arrays["xcoords"] = np.arange(5, dtype=np.float64)
    arrays["ycoords"] = np.arange(5, dtype=np.float64)
    arrays["scan_index"] = np.arange(5, dtype=np.int64)
    arrays["object_index"] = np.zeros(5, dtype=np.int64)
    np.savez(path, **arrays)

    with pytest.warns(RuntimeWarning, match="dropping the trailing 2 positions"):
        _validate_flat_npz(path, _model_stub().data_config)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("Y", np.ones((2, 8, 8), dtype=np.complex64), "Y must have shape"),
        ("object_index", np.zeros(2, dtype=np.int64), "object_index must have shape"),
    ],
)
def test_flat_npz_compatibility_uses_canonical_optional_shape_validation(
    tmp_path, name, value, message
):
    from ptycho_torch.inference import _validate_flat_npz

    path = tmp_path / f"bad_{name}.npz"
    _write_flat_npz(path, count=3)
    with np.load(path) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays[name] = value
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match=message):
        _validate_flat_npz(path, _model_stub().data_config)


def test_flat_v1_still_requires_canonical_key_layout_and_coordinates(tmp_path):
    from ptycho_torch.inference import _validate_flat_npz

    alias_path = tmp_path / "alias_only.npz"
    _write_flat_npz(alias_path, count=3, diffraction_key="diffraction")
    with pytest.raises(ValueError, match="flat-v1.*diff3d"):
        _validate_flat_npz(
            alias_path,
            _model_stub().data_config,
            dataset_manifest_path=tmp_path / "missing.json",
        )

    legacy_path = tmp_path / "legacy_diff3d.npz"
    _write_flat_npz(legacy_path, count=3)
    with np.load(legacy_path) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["diff3d"] = np.transpose(arrays["diff3d"], (1, 2, 0))
    np.savez(legacy_path, **arrays)
    with pytest.raises(ValueError, match="diff3d must have shape"):
        _validate_flat_npz(
            legacy_path,
            _model_stub().data_config,
            dataset_manifest_path=tmp_path / "missing.json",
        )

    trailing_path = tmp_path / "trailing_coordinates.npz"
    _write_flat_npz(trailing_path, count=3)
    with np.load(trailing_path) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["xcoords"] = np.arange(4, dtype=np.float64)
    arrays["ycoords"] = np.arange(4, dtype=np.float64)
    np.savez(trailing_path, **arrays)
    with pytest.raises(ValueError, match="strict coordinate policy"):
        _validate_flat_npz(
            trailing_path,
            _model_stub().data_config,
            dataset_manifest_path=tmp_path / "missing.json",
        )


def test_manifest_semantics_must_match_expected_workflow(tmp_path):
    from ptycho.workflows.synthetic_config import (
        materialize_data_config,
        resolve_synthetic_workflow,
    )
    from ptycho_torch.inference import _validate_flat_npz

    expected = resolve_synthetic_workflow(file_values={"simulation": {"seed": 13}})
    swapped = resolve_synthetic_workflow(file_values={"simulation": {"seed": 17}})
    test_npz = tmp_path / "test.npz"
    manifest_path = tmp_path / "manifest.json"
    _write_flat_npz(test_npz, N=swapped.data.N)
    _write_dataset_manifest(
        manifest_path,
        test_npz,
        SimpleNamespace(data_config=materialize_data_config(swapped)),
        expected_workflow=swapped,
    )

    _validate_flat_npz(
        test_npz,
        materialize_data_config(swapped),
        dataset_manifest_path=manifest_path,
        expected_workflow=swapped,
    )
    with pytest.raises(ValueError, match="simulation disagrees"):
        _validate_flat_npz(
            test_npz,
            materialize_data_config(expected),
            dataset_manifest_path=manifest_path,
            expected_workflow=expected,
        )


def test_manifest_seed_lineage_must_match_expected_workflow(tmp_path):
    from ptycho.workflows.synthetic_config import (
        materialize_data_config,
        resolve_synthetic_workflow,
    )
    from ptycho_torch.inference import _validate_flat_npz

    expected = resolve_synthetic_workflow(file_values={"simulation": {"seed": 13}})
    data_config = materialize_data_config(expected)
    test_npz = tmp_path / "test.npz"
    manifest_path = tmp_path / "manifest.json"
    _write_flat_npz(test_npz, N=expected.data.N)
    _write_dataset_manifest(
        manifest_path,
        test_npz,
        SimpleNamespace(data_config=data_config),
        expected_workflow=expected,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed_lineage"]["test_noise"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="seed_lineage"):
        _validate_flat_npz(
            test_npz,
            data_config,
            dataset_manifest_path=manifest_path,
            expected_workflow=expected,
        )


def test_selected_reconstruction_requires_npz_to_match_manifest_digest(
    tmp_path, monkeypatch
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)
    manifest_path = tmp_path / "manifest.json"
    _write_dataset_manifest(
        manifest_path,
        test_npz,
        model,
        npz_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="npz_sha256 mismatch"):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            dataset_manifest_path=manifest_path,
            quiet=True,
        )

    reconstruct.assert_not_called()


def test_dataset_manifest_photon_identity_must_match_bundle(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)
    manifest_path = tmp_path / "manifest.json"
    _write_dataset_manifest(
        manifest_path,
        test_npz,
        model,
        photons_per_pattern=model.data_config.nphotons * 2,
    )

    with pytest.raises(ValueError, match="photons_per_pattern"):
        reconstruct_npz_barycentric(
            bundle,
            test_npz,
            run_root=run_root,
            dataset_manifest_path=manifest_path,
        )
    reconstruct.assert_not_called()


def test_historical_rectangular_legacy_bundle_is_supported(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub(physics_forward_mode="rectangular_scaled")
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root, quiet=True)

    reconstruct.assert_called_once()


@pytest.mark.parametrize("probe_layout", ["modes", "legacy-singleton"])
def test_compatibility_npz_accepts_alias_and_supported_probe_layouts(
    tmp_path, monkeypatch, probe_layout
):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    _write_flat_npz(
        test_npz,
        diffraction_key="diffraction",
        probe_layout=probe_layout,
    )
    model = _model_stub()
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)

    reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root, quiet=True)

    reconstruct.assert_called_once()


def test_explicit_scaling_profile_overrides_reach_strict_loader(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    loader, _, _, _ = _install_stubs(monkeypatch, model)

    reconstruct_npz_barycentric(
        bundle,
        test_npz,
        run_root=run_root,
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        quiet=True,
    )

    assert loader.call_args.kwargs["scale_contract_version"] == "legacy_v1"
    assert loader.call_args.kwargs["measurement_domain"] == "normalized_amplitude"


def test_collapsed_or_duplicated_c4_channels_fail_before_reassembly(
    tmp_path, monkeypatch
):
    from ptycho_torch import dataloader
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    _, _, _, reconstruct = _install_stubs(monkeypatch, model)
    collapsed = _dataset_stub()
    collapsed.mmap_ptycho["nn_indices"] = torch.tensor([[0, 0, 0, 0]])
    collapsed.mmap_ptycho["coords_global"][:] = 0
    monkeypatch.setattr(dataloader, "PtychoDataset", lambda *_a, **_k: collapsed)

    with pytest.raises(ValueError, match=r"C4.*channel"):
        reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root)
    reconstruct.assert_not_called()


def test_empty_canvas_weights_are_rejected(tmp_path, monkeypatch):
    from ptycho_torch.inference import reconstruct_npz_barycentric

    bundle, test_npz, run_root = _paths(tmp_path)
    model = _model_stub()
    diagnostics = _diagnostics()
    object.__setattr__(diagnostics, "_canvas_weights_data", bytes(10 * 10 * 4))
    canvas = torch.ones((10, 10), dtype=torch.complex64)
    reconstruct = MagicMock(
        return_value=(canvas, SimpleNamespace(), diagnostics, canvas.clone())
    )
    _install_stubs(monkeypatch, model, reconstruct=reconstruct)

    with pytest.raises(ValueError, match="canvas weights"):
        reconstruct_npz_barycentric(bundle, test_npz, run_root=run_root)
