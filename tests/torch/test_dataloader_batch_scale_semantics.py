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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.data_container_bridge import PtychoDataContainerTorch
from ptycho_torch.dataloader import PtychoDataset
from ptycho_torch.helper import normalize_probe_like_tf
from ptycho_torch.workflows.components import _build_lightning_dataloaders


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

    data_config = DataConfig(N=32, gridsize=1, neighbor_count=4)
    model_config = ModelConfig()
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


def _make_ram_container(n_samples=4, N=4, rng=None):
    """Tiny RAM ``PtychoDataContainerTorch`` with a probe whose mean-abs energy
    makes the probe-normalization multiplier numerically distinct from 1."""
    rng = rng or np.random.default_rng(7)
    gridsize = 1
    X = rng.random((n_samples, N, N, gridsize)).astype(np.float32)
    coords_relative = np.zeros((n_samples, 1, 2, gridsize), dtype=np.float32)
    coords_offsets = np.zeros((n_samples, 1, 2, 1), dtype=np.float64)
    nn_indices = np.zeros((n_samples, gridsize), dtype=np.int32)
    probe = (5.0 * (rng.random((N, N)) + 1j * rng.random((N, N)))).astype(
        np.complex64
    )
    grouped = {
        "X_full": X,
        "Y": None,
        "coords_relative": coords_relative,
        "coords_offsets": coords_offsets,
        "nn_indices": nn_indices,
    }
    return PtychoDataContainerTorch(grouped, probe=probe), probe


def _canonical_normalization(probe, data_config, model_config):
    """The mmap canonical TF target (dataloader.py PtychoDataset.__init__)."""
    return normalize_probe_like_tf(
        probe,
        probe_scale=data_config.probe_scale,
        probe_mask=getattr(model_config, "probe_mask", False),
        probe_mask_tensor=getattr(model_config, "probe_mask_tensor", None),
        probe_mask_sigma=getattr(model_config, "probe_mask_sigma", 1.0),
        probe_mask_diameter=getattr(model_config, "probe_mask_diameter", None),
    )


@pytest.mark.parametrize("probe_normalize", [True, False])
def test_ram_lightning_batch_probe_normalize_contract(probe_normalize):
    """The real ``_build_lightning_dataloaders`` RAM path must honor
    ``DataConfig.probe_normalize`` like the mmap path does: with True, the
    emitted batch probe is the canonical TF-normalized probe and batch item 2
    is its normalization multiplier (not 1); with False, the raw probe and
    factor 1 are preserved."""
    probe_scale = 2.5
    container, raw_probe = _make_ram_container()
    data_config = DataConfig(
        N=4,
        gridsize=1,
        neighbor_count=4,
        probe_normalize=probe_normalize,
        probe_scale=probe_scale,
    )
    model_config = ModelConfig()
    payload = SimpleNamespace(
        pt_data_config=data_config,
        pt_model_config=model_config,
        pt_training_config=TrainingConfig(batch_size=4),
    )

    train_loader, _ = _build_lightning_dataloaders(
        container, None, config=None, payload=payload
    )
    batch = next(iter(train_loader))
    probes, scaling = batch[1], batch[2]

    raw_probe_t = torch.from_numpy(np.asarray(raw_probe, dtype=np.complex64))
    raw_bank = raw_probe_t.view(1, 1, 1, *raw_probe_t.shape).expand_as(probes)

    if probe_normalize:
        expected_probe, expected_multiplier = _canonical_normalization(
            raw_probe, data_config, model_config
        )
        expected_probe_t = torch.from_numpy(expected_probe)
        expected_bank = expected_probe_t.view(
            1, 1, 1, *expected_probe_t.shape
        ).expand_as(probes)
        # Guard: the fixture must distinguish normalized from raw, or the
        # probe assertion would pass vacuously.
        assert expected_multiplier != 1.0, (
            "fixture's probe-normalization multiplier is 1; adjust the "
            "probe/probe_scale so normalization changes the probe"
        )
        assert not torch.allclose(expected_bank, raw_bank), (
            "fixture's normalization does not change the probe; adjust the "
            "probe/probe_scale"
        )
        assert torch.allclose(probes, expected_bank), (
            "RAM batch probe must equal the canonical TF-normalized probe "
            "when DataConfig.probe_normalize=True"
        )
        assert torch.allclose(
            scaling, torch.full_like(scaling, expected_multiplier)
        ), (
            "RAM batch item 2 must be the probe-normalization multiplier "
            f"{expected_multiplier} from normalize_probe_like_tf when "
            "DataConfig.probe_normalize=True"
        )
        assert not torch.allclose(scaling, torch.ones_like(scaling)), (
            "RAM batch item 2 must not be the default factor 1 when "
            "DataConfig.probe_normalize=True"
        )
    else:
        assert torch.allclose(probes, raw_bank), (
            "RAM batch probe must be the raw probe when "
            "DataConfig.probe_normalize=False"
        )
        assert torch.allclose(scaling, torch.ones_like(scaling)), (
            "RAM batch item 2 must be factor 1 when "
            "DataConfig.probe_normalize=False"
        )
