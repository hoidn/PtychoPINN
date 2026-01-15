"""Lightweight regression coverage for dose_response_study inference workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.raw_data import RawData


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
