"""Lightweight regression coverage for dose_response_study inference workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.raw_data import RawData
from ptycho.loader import compute_dataset_intensity_stats


def test_run_inference_caps_groups(monkeypatch, tmp_path):
    """run_inference should cap groups to available test images."""
    from scripts.studies import dose_response_study as study
    from ptycho import loader
    from ptycho import nbutils
    from ptycho import tf_helper
    from ptycho.workflows import backend_selector

    N = 4
    gridsize = 2
    n_images = 3
    coords = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    diff3d = np.ones((n_images, N, N), dtype=np.float32)
    probe = np.ones((N, N), dtype=np.complex64)
    raw = RawData(
        xcoords=coords,
        ycoords=coords,
        xcoords_start=coords,
        ycoords_start=coords,
        diff3d=diff3d,
        probeGuess=probe,
        scan_index=np.zeros(n_images, dtype=int),
    )

    config = TrainingConfig(
        model=ModelConfig(N=N, gridsize=gridsize),
        n_groups=10,
        neighbor_count=4,
        output_dir=Path(tmp_path) / "arm",
    )

    results = {
        "arm": {
            "config": config,
            "test_data": raw,
            "success": True,
        }
    }

    captured = {}

    def fake_load_inference_bundle_with_backend(bundle_dir, config, model_name="diffraction_to_obj"):
        class DummyModel:
            def predict(self, *args, **kwargs):
                return np.zeros((1, N, N, 1), dtype=np.float32)

        return DummyModel(), {"N": N, "gridsize": gridsize}

    def fake_generate_grouped_data(self, N, K, nsamples, **kwargs):
        captured["nsamples"] = nsamples
        return {"dummy": True}

    class DummyContainer:
        def __init__(self):
            self.X = np.zeros((1, N, N, 1), dtype=np.float32)
            self.coords_nominal = np.zeros((1, 1, 2, 1), dtype=np.float32)
            self.global_offsets = np.zeros((1, 1, 1, 2), dtype=np.float32)
            self.local_offsets = np.zeros((1, 1, 1, 2), dtype=np.float32)
            self.probe = probe

    def fake_load(cb, probeGuess, which=None, create_split=False):
        return DummyContainer()

    def fake_reconstruct_image(container, diffraction_to_obj=None):
        return np.zeros((1, N, N, 1), dtype=np.float32), container.global_offsets

    def fake_reassemble_position(obj_tensor_full, global_offsets, M=20):
        return np.zeros((N, N), dtype=np.float32)

    monkeypatch.setattr(
        backend_selector,
        "load_inference_bundle_with_backend",
        fake_load_inference_bundle_with_backend,
    )
    monkeypatch.setattr(RawData, "generate_grouped_data", fake_generate_grouped_data, raising=True)
    monkeypatch.setattr(loader, "load", fake_load, raising=True)
    monkeypatch.setattr(nbutils, "reconstruct_image", fake_reconstruct_image, raising=True)
    monkeypatch.setattr(tf_helper, "reassemble_position", fake_reassemble_position, raising=True)

    updated = study.run_inference(results)

    assert captured["nsamples"] == n_images
    recon = updated["arm"]["reconstruction"]
    assert recon is not None
    assert recon["amplitude"].shape == (N, N)


def test_simulate_datasets_grid_mode_attaches_dataset_stats(monkeypatch, tmp_path):
    """Grid-mode containers should have dataset_intensity_stats from D4f.3.

    Per Phase D4f.3: simulate_datasets_grid_mode() must call compute_dataset_intensity_stats
    on the raw diffraction arrays (or back-compute from normalized data + intensity_scale)
    and pass the resulting dict into each PtychoDataContainer so training/inference
    never falls back to the 988.21 constant.

    See: specs/spec-ptycho-core.md §Normalization Invariants
    """
    from scripts.studies import dose_response_study as study
    from ptycho import params as p
    from ptycho.diffsim import mk_simdata

    # Constants matching the function
    GRID_N = 64
    GRID_SIZE = 2
    nimgs_train = 2
    nimgs_test = 2

    # Create deterministic mk_simdata mock outputs
    np.random.seed(42)

    # mk_simdata returns: X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, coords
    # X is normalized diffraction, shape (B * gridsize^2, N, N, gridsize^2)
    B_train = nimgs_train * (GRID_SIZE ** 2)  # 8 groups total
    B_test = nimgs_test * (GRID_SIZE ** 2)

    # Create deterministic diffraction data
    X_train = np.random.rand(B_train, GRID_N, GRID_N, GRID_SIZE ** 2).astype(np.float32) + 0.1
    X_test = np.random.rand(B_test, GRID_N, GRID_N, GRID_SIZE ** 2).astype(np.float32) + 0.1

    Y_I_train = np.random.rand(B_train, GRID_N, GRID_N, GRID_SIZE ** 2).astype(np.float32)
    Y_I_test = np.random.rand(B_test, GRID_N, GRID_N, GRID_SIZE ** 2).astype(np.float32)

    Y_phi_train = np.zeros_like(Y_I_train)
    Y_phi_test = np.zeros_like(Y_I_test)

    # The intensity_scale returned by mk_simdata
    mock_intensity_scale = 500.0

    YY_full_train = np.ones((1, 200, 200), dtype=np.complex64)
    YY_full_test = np.ones((1, 200, 200), dtype=np.complex64)

    norm_Y_I_train = 1.0
    norm_Y_I_test = 1.0

    # Coords: shape (2, B, 1, 2, C) where [0]=nominal, [1]=true
    coords_train = np.zeros((2, B_train, 1, 2, GRID_SIZE ** 2), dtype=np.float32)
    coords_test = np.zeros((2, B_test, 1, 2, GRID_SIZE ** 2), dtype=np.float32)

    call_count = {"train": 0, "test": 0}

    def fake_mk_simdata(n, size, probe, outer_offset, which, intensity_scale=None, jitter_scale=0.0):
        """Mock mk_simdata that returns deterministic outputs."""
        if which == "train":
            call_count["train"] += 1
            return (X_train, Y_I_train, Y_phi_train, mock_intensity_scale,
                    YY_full_train, norm_Y_I_train, coords_train)
        else:
            call_count["test"] += 1
            return (X_test, Y_I_test, Y_phi_test, mock_intensity_scale,
                    YY_full_test, norm_Y_I_test, coords_test)

    # Monkeypatch mk_simdata
    monkeypatch.setattr(study, "mk_simdata", fake_mk_simdata, raising=False)
    # Also need to patch in diffsim module which is imported inside the function
    from ptycho import diffsim
    monkeypatch.setattr(diffsim, "mk_simdata", fake_mk_simdata, raising=True)

    # Create a probe matching GRID_N
    probe = np.ones((GRID_N, GRID_N), dtype=np.complex64)

    # Run the function
    datasets = study.simulate_datasets_grid_mode(
        probeGuess=probe,
        base_output_dir=Path(tmp_path),
        nepochs=1,
        nimgs_train=nimgs_train,
        nimgs_test=nimgs_test
    )

    # Verify dataset_intensity_stats is attached to each container
    for arm_name, arm_data in datasets.items():
        train_container = arm_data['train_container']
        test_container = arm_data['test_container']

        # Check train container has stats
        assert hasattr(train_container, 'dataset_intensity_stats'), \
            f"{arm_name} train_container missing dataset_intensity_stats"
        assert train_container.dataset_intensity_stats is not None, \
            f"{arm_name} train_container.dataset_intensity_stats is None"

        train_stats = train_container.dataset_intensity_stats
        assert 'batch_mean_sum_intensity' in train_stats, \
            f"{arm_name} train stats missing batch_mean_sum_intensity"
        assert 'n_samples' in train_stats, \
            f"{arm_name} train stats missing n_samples"

        # Verify the stats are reasonable (positive, non-zero)
        assert train_stats['batch_mean_sum_intensity'] > 0, \
            f"{arm_name} train batch_mean should be positive"
        assert train_stats['n_samples'] == B_train, \
            f"{arm_name} train n_samples mismatch"

        # Check test container has stats
        assert hasattr(test_container, 'dataset_intensity_stats'), \
            f"{arm_name} test_container missing dataset_intensity_stats"
        assert test_container.dataset_intensity_stats is not None, \
            f"{arm_name} test_container.dataset_intensity_stats is None"

        test_stats = test_container.dataset_intensity_stats
        assert test_stats['batch_mean_sum_intensity'] > 0, \
            f"{arm_name} test batch_mean should be positive"
        assert test_stats['n_samples'] == B_test, \
            f"{arm_name} test n_samples mismatch"

        # Verify stats match compute_dataset_intensity_stats with is_normalized=True
        expected_train_stats = compute_dataset_intensity_stats(
            X_train, intensity_scale=mock_intensity_scale, is_normalized=True
        )
        assert abs(train_stats['batch_mean_sum_intensity'] - expected_train_stats['batch_mean_sum_intensity']) < 1e-6, \
            f"{arm_name} train stats don't match compute_dataset_intensity_stats output"
