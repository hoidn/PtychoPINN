"""Regression test for the batch[2] scale semantics mismatch root-caused in
``.superpowers/sdd/ext/task-realimag-collapse-rca.md`` (H1, CONFIRMED).

``PtychoDataset.__getitem__`` (``ptycho_torch/dataloader.py``) returned the
RMS ``scaling_const`` (``dataloader.py`` diffraction-normalization constant)
as the third tuple element, but the B5-ported rectangular_scaled loss
(``ptycho_torch/model.py`` ``compute_loss``, ``scale = batch[2]``) consumes
that element as ``probe_scaling`` (the probe-normalization inverse from
``normalize_probe_like_tf``). origin/main fixed this exact bug in
``9824d7a5`` ("Fix intensity scaling bugs causing 400x loss inflation", bug
1) by returning ``probe_scaling`` instead of the RMS constant. This test
pins the post-fix contract on fno-stable: ``PtychoDataset.__getitem__``'s
third element must equal the dataset's computed ``probe_scaling``, and must
NOT equal the RMS ``scaling_const`` when the two differ numerically.
"""
import numpy as np
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
    # Probe with non-unit mean-abs energy so normalize_probe_like_tf's
    # probe_scaling differs numerically from the RMS scaling_constant.
    probe = (5.0 * (rng.random((N, N)) + 1j * rng.random((N, N)))).astype(np.complex128)
    obj = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex128)

    np.savez(
        path, xcoords=xcoords, ycoords=ycoords, diff3d=diff3d,
        probeGuess=probe, objectGuess=obj,
    )


def _build_dataset(tmp_path):
    ptycho_dir = tmp_path / "npz_dir"
    ptycho_dir.mkdir()
    _write_synthetic_npz(ptycho_dir / "fixture.npz", n_images=40, N=32)

    data_config = DataConfig(N=32, grid_size=(1, 1), C=1, K=4)
    model_config = ModelConfig(C_model=1, C_forward=1)
    training_config = TrainingConfig(batch_size=8)

    return PtychoDataset(
        ptycho_dir=str(ptycho_dir), model_config=model_config, data_config=data_config,
        training_config=training_config, data_dir=str(tmp_path / "memmap"), remake_map=True,
    )


def test_getitem_batch2_is_probe_scaling_not_rms_scaling_const(tmp_path):
    dataset = _build_dataset(tmp_path)

    idx = torch.arange(8)
    _, _, scale = dataset[idx]

    expected_probe_scaling = dataset.data_dict['probe_scaling'][
        torch.zeros_like(dataset.mmap_ptycho['experiment_id'][idx])
    ].view(-1, 1, 1, 1)
    rms_scaling_const = dataset.data_dict['scaling_constant'][
        torch.zeros_like(dataset.mmap_ptycho['experiment_id'][idx])
    ].view(-1, 1, 1, 1)

    # Guard: the fixture must actually distinguish the two constants, or this
    # test would pass vacuously regardless of which one __getitem__ returns.
    assert not torch.allclose(expected_probe_scaling, rms_scaling_const), (
        "fixture's probe_scaling and RMS scaling_constant coincide numerically; "
        "adjust the fixture so the two constants differ"
    )

    assert torch.allclose(scale, expected_probe_scaling), (
        f"batch[2] = {scale.flatten().tolist()} != probe_scaling "
        f"{expected_probe_scaling.flatten().tolist()}"
    )
    assert not torch.allclose(scale, rms_scaling_const), (
        f"batch[2] = {scale.flatten().tolist()} incorrectly equals the RMS "
        f"scaling_const {rms_scaling_const.flatten().tolist()} instead of probe_scaling"
    )
