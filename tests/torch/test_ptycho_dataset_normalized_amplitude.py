"""Regression test for a dataloader defect found while building the Task 1.5
varpro/probe ablation harness.

``ptycho_torch.dataloader.PtychoDataset.memory_map_data`` used to call
``.round()`` on the raw diffraction array before computing RMS/physics
scaling factors. Per ``docs/data_contracts.md`` ("Diffraction patterns MUST
be normalized" / "DO NOT pre-apply photon scaling"), diffraction data is
normalized amplitude (typically max < 1.0) with photon count carried only as
a separate config parameter -- never baked into the array. Rounding such an
array to the nearest integer zeros it out entirely, which makes
``get_rms_scaling_factor`` divide by zero and silently return ``inf``
(confirmed empirically against a real fly64/1e9-photon fixture -- see
``.superpowers/sdd/task-1.5-report.md``).

This test constructs a tiny, spec-compliant (normalized-amplitude) synthetic
NPZ and asserts the resulting memory-mapped scaling constants are finite and
the diffraction values are not zeroed.
"""
import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import PtychoDataset


def _write_synthetic_npz(path, n_images=40, N=32, rng=None):
    rng = rng or np.random.default_rng(0)
    # Normalized amplitude: unit L2 norm per pattern, max well under 1.0 --
    # matches docs/data_contracts.md's normalization requirement.
    raw = rng.random((n_images, N, N)).astype(np.float32)
    norms = np.sqrt((raw ** 2).sum(axis=(-2, -1), keepdims=True))
    diff3d = raw / norms

    xcoords = np.linspace(0.0, 10.0, n_images).astype(np.float64)
    ycoords = np.linspace(0.0, 10.0, n_images).astype(np.float64)
    probe = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex128)
    obj = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex128)

    np.savez(
        path, xcoords=xcoords, ycoords=ycoords, diff3d=diff3d,
        probeGuess=probe, objectGuess=obj,
    )


def test_ptycho_dataset_does_not_zero_normalized_amplitude_data(tmp_path):
    ptycho_dir = tmp_path / "npz_dir"
    ptycho_dir.mkdir()
    _write_synthetic_npz(ptycho_dir / "fixture.npz", n_images=40, N=32)

    data_config = DataConfig(N=32, grid_size=(1, 1), C=1, K=4)
    model_config = ModelConfig(C_model=1, C_forward=1)
    training_config = TrainingConfig(batch_size=8)

    dataset = PtychoDataset(
        ptycho_dir=str(ptycho_dir), model_config=model_config, data_config=data_config,
        training_config=training_config, data_dir=str(tmp_path / "memmap"), remake_map=True,
    )

    images = dataset.mmap_ptycho["images"]
    rms = dataset.mmap_ptycho["rms_scaling_constant"]
    physics = dataset.mmap_ptycho["physics_scaling_constant"]

    assert torch.isfinite(torch.as_tensor(rms[:])).all(), "rms_scaling_constant contains inf/nan"
    assert torch.isfinite(torch.as_tensor(physics[:])).all(), "physics_scaling_constant contains inf/nan"
    assert float(torch.as_tensor(images[:]).abs().sum()) > 0.0, "diffraction images were zeroed"
