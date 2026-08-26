"""Regression tests for incoherent multi-mode probe support.

Multi-mode probe support existed on mainline (synced from CDI-PINN) and was
lost in the fno-stable dataloader rework: the probe buffer was allocated with
a hardcoded single mode axis and a (P, N, N) ``probeGuess`` skipped
normalization (leaving ``probe_scaling`` at 0) before crashing on the
mode-axis write.

These tests pin the retained behavior:
- ``normalize_probe_like_tf`` accepts a (P, N, N) stack, applying one joint
  norm (P=1 stack bit-matches the 2D path).
- ``memory_map_data`` pre-scans for the max mode count, allocates
  (n_files, max_modes, N, N), and normalizes multi-mode probes.
"""
import numpy as np
import pytest
import torch

import ptycho_torch.helper as hh
from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
from ptycho_torch.dataloader import PtychoDataset


N_PIX = 32


def _rng():
    return np.random.default_rng(7)


def _make_arrays(n_images=40, N=N_PIX, n_modes=None, rng=None):
    rng = rng or _rng()
    raw = rng.random((n_images, N, N)).astype(np.float32)
    norms = np.sqrt((raw ** 2).sum(axis=(-2, -1), keepdims=True))
    diff3d = raw / norms

    xcoords = np.linspace(0.0, 10.0, n_images).astype(np.float64)
    ycoords = np.linspace(0.0, 10.0, n_images).astype(np.float64)
    if n_modes is None:
        probe = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex128)
    else:
        probe = (rng.random((n_modes, N, N)) + 1j * rng.random((n_modes, N, N))).astype(np.complex128)
    obj = (rng.random((N, N)) + 1j * rng.random((N, N))).astype(np.complex128)
    return diff3d, xcoords, ycoords, probe, obj


def _make_grouped_arrays(n_modes=None):
    rng = _rng()
    raw = rng.random((64, N_PIX, N_PIX)).astype(np.float32)
    norms = np.sqrt((raw ** 2).sum(axis=(-2, -1), keepdims=True))
    diff3d = raw / norms

    xcoords, ycoords = np.meshgrid(
        np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64))
    xcoords = xcoords.ravel()
    ycoords = ycoords.ravel()
    if n_modes is None:
        probe = (rng.random((N_PIX, N_PIX)) + 1j * rng.random((N_PIX, N_PIX))).astype(np.complex128)
    else:
        probe = (rng.random((n_modes, N_PIX, N_PIX)) +
                 1j * rng.random((n_modes, N_PIX, N_PIX))).astype(np.complex128)
    obj = (rng.random((N_PIX, N_PIX)) + 1j * rng.random((N_PIX, N_PIX))).astype(np.complex128)
    return diff3d, xcoords, ycoords, probe, obj


def _write_npz(path, diff3d, xcoords, ycoords, probe, obj):
    np.savez(path, xcoords=xcoords, ycoords=ycoords, diff3d=diff3d,
             probeGuess=probe, objectGuess=obj)


def _configs(**data_overrides):
    data_config = DataConfig(N=N_PIX, gridsize=1, neighbor_count=4, **data_overrides)
    model_config = ModelConfig()
    training_config = TrainingConfig(batch_size=8)
    return data_config, model_config, training_config


def _group_configs(normalize='Group', data_scaling='Parseval'):
    data_config = DataConfig(
        N=N_PIX, gridsize=2, neighbor_count=6, n_raw_frames_selected=1,
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), normalize=normalize,
        data_scaling=data_scaling)
    model_config = ModelConfig(object_big=True)
    training_config = TrainingConfig(batch_size=8)
    return data_config, model_config, training_config


def _build_file_dataset(tmp_path, npz_payloads, data_config, model_config, training_config):
    ptycho_dir = tmp_path / "npz_dir"
    ptycho_dir.mkdir()
    for name, payload in npz_payloads.items():
        _write_npz(ptycho_dir / name, *payload)
    return PtychoDataset(
        ptycho_dir=str(ptycho_dir), model_config=model_config, data_config=data_config,
        training_config=training_config, data_dir=str(tmp_path / "memmap"), remake_map=True,
    )


# ---------------------------------------------------------------------------
# normalize_probe_like_tf multi-mode semantics
# ---------------------------------------------------------------------------

def test_normalize_probe_like_tf_p1_stack_matches_2d():
    rng = _rng()
    probe = (rng.random((N_PIX, N_PIX)) + 1j * rng.random((N_PIX, N_PIX))).astype(np.complex64)

    out_2d, scale_2d = hh.normalize_probe_like_tf(probe, probe_scale=4.0)
    out_stack, scale_stack = hh.normalize_probe_like_tf(probe[None], probe_scale=4.0)

    assert out_stack.shape == (1, N_PIX, N_PIX)
    np.testing.assert_allclose(out_stack[0], out_2d, rtol=1e-6)
    assert scale_stack == pytest.approx(scale_2d, rel=1e-6)


def test_normalize_probe_like_tf_multimode_applies_one_joint_norm():
    rng = _rng()
    probe = (rng.random((2, N_PIX, N_PIX)) + 1j * rng.random((2, N_PIX, N_PIX))).astype(np.complex64)

    out, scale = hh.normalize_probe_like_tf(probe, probe_scale=4.0)

    assert out.shape == (2, N_PIX, N_PIX)
    assert out.dtype == np.complex64
    assert scale > 0
    # One shared scalar: each mode is the input divided by the same norm, so
    # relative mode powers are preserved.
    np.testing.assert_allclose(out[0], probe[0] * scale, rtol=1e-5)
    np.testing.assert_allclose(out[1], probe[1] * scale, rtol=1e-5)


# ---------------------------------------------------------------------------
# memory_map_data multi-mode probes
# ---------------------------------------------------------------------------

def test_memory_map_multimode_probe(tmp_path):
    data_config, model_config, training_config = _configs()
    payload = _make_arrays(n_modes=2)
    dataset = _build_file_dataset(tmp_path, {"multi.npz": payload},
                                  data_config, model_config, training_config)

    probes = dataset.data_dict['probes']
    assert probes.shape == (1, 2, N_PIX, N_PIX)
    assert probes[0, 0].abs().sum() > 0
    assert probes[0, 1].abs().sum() > 0
    assert float(dataset.data_dict['probe_scaling'][0]) > 0


def test_memory_map_mixed_mode_directory(tmp_path):
    data_config, model_config, training_config = _configs()
    single = _make_arrays(rng=np.random.default_rng(1))
    multi = _make_arrays(n_modes=2, rng=np.random.default_rng(2))
    dataset = _build_file_dataset(tmp_path, {"a_single.npz": single, "b_multi.npz": multi},
                                  data_config, model_config, training_config)

    probes = dataset.data_dict['probes']
    assert probes.shape == (2, 2, N_PIX, N_PIX)
    # Single-mode file: mode 0 populated, mode 1 zero-padded.
    assert probes[0, 0].abs().sum() > 0
    assert probes[0, 1].abs().sum() == 0
    # Multi-mode file: both modes populated.
    assert probes[1, 1].abs().sum() > 0
    scalings = dataset.data_dict['probe_scaling']
    assert float(scalings[0]) > 0 and float(scalings[1]) > 0


# ---------------------------------------------------------------------------
# from_np restoration
# ---------------------------------------------------------------------------

def test_multifile_getitem_preserves_scalar_and_tensor_probe_identity(tmp_path):
    data_config, _, training_config = _configs(
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), n_raw_frames_selected=1)
    model_config = ModelConfig(object_big=False)
    payload_exp0 = _make_arrays(n_images=12, rng=np.random.default_rng(11))
    payload_exp1 = list(_make_arrays(n_images=12, rng=np.random.default_rng(22)))
    payload_exp1[3] *= 2.5
    dataset = _build_file_dataset(
        tmp_path,
        {"experiment_0.npz": payload_exp0, "experiment_1.npz": tuple(payload_exp1)},
        data_config, model_config, training_config)

    experiment_ids = torch.as_tensor(dataset.mmap_ptycho['experiment_id'])
    idx_exp0 = int(torch.where(experiment_ids == 0)[0][0])
    idx_exp1 = int(torch.where(experiment_ids == 1)[0][0])

    td, probes_indexed, probe_scaling = dataset[idx_exp1]

    assert td.batch_size == torch.Size([])
    assert probes_indexed.shape == (1, 1, N_PIX, N_PIX)
    assert probe_scaling.shape == (1, 1, 1)
    torch.testing.assert_close(probes_indexed, dataset.data_dict['probes'][1].unsqueeze(0))
    torch.testing.assert_close(
        probe_scaling, dataset.data_dict['probe_scaling'][1].view(1, 1, 1))

    _, probes_indexed, probe_scaling = dataset[torch.tensor([idx_exp0, idx_exp1])]

    assert probes_indexed.shape == (2, 1, 1, N_PIX, N_PIX)
    assert probe_scaling.shape == (2, 1, 1, 1)
    torch.testing.assert_close(probes_indexed[0], dataset.data_dict['probes'][0].unsqueeze(0))
    torch.testing.assert_close(probes_indexed[1], dataset.data_dict['probes'][1].unsqueeze(0))
    torch.testing.assert_close(
        probe_scaling[0], dataset.data_dict['probe_scaling'][0].view(1, 1, 1))
    torch.testing.assert_close(
        probe_scaling[1], dataset.data_dict['probe_scaling'][1].view(1, 1, 1))
