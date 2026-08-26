"""Embedding smoke test for the dataset-in reconstruction kernel (Task 1).

The programmatic reconstruction kernel is ``reconstruct_from_dataset``: a
loaded model and an already-grouped ``PtychoDataset`` in, frozen amplitude /
phase snapshots out. This smoke test drives the full load → validate →
construct → delegate shell (``reconstruct_npz_barycentric``) over a real
fixed-pitch scan and asserts the two invariants the extraction was meant to
guarantee:

- the model bundle is read exactly once (no CLI double bundle read), and
- the held-out NPZ is symlink-staged, never copied (no temp-dir staging copy).
"""

import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch


def _fixed_texture_model_and_scan(tmp_path):
    """Build a deterministic 4-patch fixed-pitch scan + FixedTextureModel."""
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions
    from ptycho.workflows.synthetic_config import (
        materialize_data_config,
        resolve_synthetic_workflow,
    )
    from ptycho.workflows.training import _torch_model_from_snapshot
    from ptycho_torch.config_bridge import to_model_config
    from ptycho_torch.config_params import InferenceConfig, TrainingConfig
    from ptycho_torch.model_spec import derive_model_spec

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "seed": 5,
                "train_patterns": 4,
                "test_patterns": 4,
                "object": {"patch_amplitude_normalization": "none"},
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
                "reconstruction_method": "barycentric",
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
        device="cpu", num_workers=0, training_groups=resolved.training.training_groups
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
            self.torch_loss_mode = "poisson"

        def forward_predict(self, intensity, positions, probe, input_scale):
            return self.textures[: intensity.shape[0]]

    return resolved, acquisition, FixedTextureModel()


def test_dataset_in_arrays_out_no_copy_single_bundle_read(tmp_path, monkeypatch):
    """Real dataset in, arrays out; no NPZ copy; the bundle is read once."""
    from ptycho_torch.workflows import components

    resolved, acquisition, model = _fixed_texture_model_and_scan(tmp_path)

    bundle = tmp_path / "training"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"strict-smoke-bundle")

    loader_calls = []

    def counting_loader(*args, **kwargs):
        loader_calls.append(args)
        return ({"diffraction_to_obj": model}, {})

    monkeypatch.setattr("ptycho_torch.workflows.bundle_io.load_inference_bundle_torch", counting_loader)

    copy_calls = []
    monkeypatch.setattr(shutil, "copy2", lambda *a, **k: copy_calls.append(a))
    monkeypatch.setattr(shutil, "copy", lambda *a, **k: copy_calls.append(a))

    from ptycho_torch.inference import reconstruct_npz_barycentric

    result = reconstruct_npz_barycentric(
        bundle,
        acquisition.test_path,
        run_root=tmp_path / "run",
        expected_workflow=resolved,
        dataset_manifest_path=acquisition.manifest_path,
        device="cpu",
        quiet=True,
    )

    assert result.amplitude.ndim == 2
    assert result.phase.ndim == 2
    assert result.amplitude.shape == result.phase.shape == (74, 74)
    assert np.isfinite(result.amplitude).all()
    assert np.isfinite(result.phase).all()
    assert result.amplitude.max() > 0.0, "amplitude canvas must carry the object"

    assert len(loader_calls) == 1, (
        f"expected exactly one bundle read, got {len(loader_calls)}"
    )
    assert copy_calls == [], (
        "the held-out NPZ must be symlink-staged, never copied"
    )


def test_validate_authentic_channels_maps_ungrouped_ids_to_source_space():
    """Ungrouped datasets expose LOCAL row ids in nn_indices; the coverage
    validator must speak SOURCE scan ids like the reassembly identity
    evidence (regression: full-dataset gridsize=1 inference on a bounded
    scan falsely rejected every dropped position)."""
    from types import SimpleNamespace

    from ptycho_torch.inference import _validate_authentic_channels

    source_ids = np.array([3, 7, 11, 20], dtype=np.int64)
    groups = len(source_ids)
    dataset = SimpleNamespace(
        mmap_ptycho={
            "images": torch.zeros(groups, 1, 2, 2),
            "nn_indices": torch.arange(groups, dtype=torch.int64)[:, None],
            "coords_global": torch.arange(groups * 2, dtype=torch.float32).view(
                groups, 1, 1, 2
            ),
        },
        group_coords_enabled=lambda: False,
        valid_indices_per_file=[source_ids],
    )
    data_config = SimpleNamespace(gridsize=1)

    expected, count, channel_indices, _ = _validate_authentic_channels(
        dataset, data_config
    )

    assert expected == set(source_ids.tolist())
    assert count == groups
    assert channel_indices.reshape(-1).tolist() == source_ids.tolist()


def test_arrays_in_arrays_out_equivalence(tmp_path, monkeypatch):
    """The arrays-in seam reproduces the NPZ-path reconstruction result."""
    from ptycho_torch.inference import (
        ReconstructionRuntimeParams,
        reconstruct_from_arrays,
        reconstruct_npz_barycentric,
    )

    resolved, acquisition, model = _fixed_texture_model_and_scan(tmp_path)

    bundle = tmp_path / "training"
    bundle.mkdir()
    (bundle / "wts.h5.zip").write_bytes(b"strict-smoke-bundle")
    monkeypatch.setattr(
        "ptycho_torch.workflows.bundle_io.load_inference_bundle_torch",
        lambda *a, **k: ({"diffraction_to_obj": model}, {}),
    )

    reference = reconstruct_npz_barycentric(
        bundle,
        acquisition.test_path,
        run_root=tmp_path / "run_ref",
        expected_workflow=resolved,
        dataset_manifest_path=acquisition.manifest_path,
        device="cpu",
        quiet=True,
    )

    with np.load(acquisition.test_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    result = reconstruct_from_arrays(
        model,
        arrays,
        runtime_params=ReconstructionRuntimeParams(
            data_config=model.data_config,
            training_config=model.training_config,
            inference_config=model.inference_config,
            source_metadata={},
            quiet=True,
        ),
        workspace=tmp_path / "arrays_workspace",
        device="cpu",
    )
    assert result.amplitude.shape == reference.amplitude.shape
    assert np.allclose(result.amplitude, reference.amplitude)
    assert np.allclose(result.phase, reference.phase)
