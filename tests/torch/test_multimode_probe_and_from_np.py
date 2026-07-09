"""Regression tests for incoherent multi-mode probe support and the restored
``PtychoDataset.from_np`` in-memory constructor.

Multi-mode probe support existed on mainline (synced from CDI-PINN) and was
lost in the fno-stable dataloader rework: the probe buffer was allocated with
a hardcoded single mode axis and a (P, N, N) ``probeGuess`` skipped
normalization (leaving ``probe_scaling`` at 0) before crashing on the
mode-axis write. ``from_np`` (in-memory construction for inference workflows,
bypassing NPZ I/O and the on-disk memmap) was dropped in the same rework.

These tests pin the restored behavior:
- ``normalize_probe_like_tf`` accepts a (P, N, N) stack, applying one joint
  norm (P=1 stack bit-matches the 2D path).
- ``memory_map_data`` pre-scans for the max mode count, allocates
  (n_files, max_modes, N, N), and normalizes multi-mode probes.
- ``from_np`` reproduces the file-based pipeline tensors on identical data
  and supports multi-mode probes.
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
    data_config = DataConfig(N=N_PIX, grid_size=(1, 1), C=1, K=4, **data_overrides)
    model_config = ModelConfig(C_model=1, C_forward=1)
    training_config = TrainingConfig(batch_size=8)
    return data_config, model_config, training_config


def _group_configs(normalize='Group', data_scaling='Parseval'):
    data_config = DataConfig(
        N=N_PIX, grid_size=(2, 2), C=4, K=6, n_subsample=1,
        neighbor_function='4_quadrant', scan_pattern='Isotropic',
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), normalize=normalize,
        data_scaling=data_scaling)
    model_config = ModelConfig(C_model=4, C_forward=4, object_big=True)
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

def test_from_np_matches_file_dataset(tmp_path):
    # x/y bounds widened so no positions are filtered: the file-based path
    # computes RMS/physics constants on the grouped (permuted) stack, which
    # only matches the from_np full-set computation when the sets are equal.
    data_config, model_config, training_config = _configs(
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    diff3d, xcoords, ycoords, probe, obj = _make_arrays()

    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": (diff3d, xcoords, ycoords, probe, obj)},
                                  data_config, model_config, training_config)
    positions = np.stack([ycoords, xcoords], axis=1)
    mem_ds = PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)

    assert len(mem_ds) == len(file_ds)
    td_file, td_mem = file_ds.mmap_ptycho, mem_ds.mmap_ptycho
    for key in ("images", "coords_relative", "coords_center",
                "rms_scaling_constant", "physics_scaling_constant"):
        torch.testing.assert_close(torch.as_tensor(td_mem[key]),
                                   torch.as_tensor(td_file[key][:]),
                                   rtol=1e-5, atol=1e-6, msg=f"mismatch in {key}")
    torch.testing.assert_close(mem_ds.data_dict['probes'][0],
                               file_ds.data_dict['probes'][0], rtol=1e-6, atol=1e-7)
    assert float(mem_ds.data_dict['probe_scaling'][0]) == pytest.approx(
        float(file_ds.data_dict['probe_scaling'][0]), rel=1e-6)


def test_from_np_multimode_probe_and_getitem():
    data_config, model_config, _ = _configs()
    diff3d, xcoords, ycoords, probe, _ = _make_arrays(n_modes=3)
    positions = np.stack([ycoords, xcoords], axis=1)

    dataset = PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)

    assert dataset.data_dict['probes'].shape == (1, 3, N_PIX, N_PIX)
    td, probes_indexed, probe_scaling = dataset[0:4]
    assert td["images"].shape[0] == 4
    # (B, C, P, N, N): probe expanded across channels with modes preserved.
    assert probes_indexed.shape == (4, 1, 3, N_PIX, N_PIX)
    assert probe_scaling.shape == (4, 1, 1, 1)


def test_from_np_scalar_getitem_single_mode_shapes():
    data_config, _, _ = _configs()
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    diff3d, xcoords, ycoords, probe, _ = _make_arrays(n_images=12)
    positions = np.stack([ycoords, xcoords], axis=1)

    dataset = PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)
    td, probes_indexed, probe_scaling = dataset[0]

    assert td['images'].shape == (1, N_PIX, N_PIX)
    assert probes_indexed.shape == (1, 1, N_PIX, N_PIX)
    assert probe_scaling.shape == (1, 1, 1)


def test_from_np_scalar_getitem_multimode_expands_channels():
    data_config, model_config, _ = _group_configs()
    diff3d, xcoords, ycoords, probe, _ = _make_grouped_arrays(n_modes=3)
    positions = np.stack([ycoords, xcoords], axis=1)

    np.random.seed(123)
    dataset = PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)
    td, probes_indexed, probe_scaling = dataset[0]

    assert td['images'].shape == (4, N_PIX, N_PIX)
    assert probes_indexed.shape == (4, 3, N_PIX, N_PIX)
    assert probe_scaling.shape == (1, 1, 1)


def test_multifile_getitem_preserves_scalar_and_tensor_probe_identity(tmp_path):
    data_config, _, training_config = _configs(
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), n_subsample=1)
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
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


def test_from_np_rejects_supervised_mode():
    data_config, model_config, _ = _configs()
    model_config = ModelConfig(C_model=1, C_forward=1, mode='Supervised')
    diff3d, xcoords, ycoords, probe, _ = _make_arrays(n_images=12)
    positions = np.stack([ycoords, xcoords], axis=1)

    with pytest.raises(ValueError, match="Unsupervised"):
        PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)


# ---------------------------------------------------------------------------
# Group-aware from_np normalization
# ---------------------------------------------------------------------------

def test_from_np_group_normalization_matches_file_dataset(tmp_path):
    data_config, model_config, training_config = _group_configs()
    payload = _make_grouped_arrays()
    diff3d, xcoords, ycoords, probe, _ = payload

    np.random.seed(123)
    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": payload},
                                  data_config, model_config, training_config)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1), model_config, data_config)

    for key in ("images", "nn_indices", "rms_scaling_constant", "physics_scaling_constant"):
        torch.testing.assert_close(torch.as_tensor(mem_ds.mmap_ptycho[key]),
                                   torch.as_tensor(file_ds.mmap_ptycho[key][:]),
                                   rtol=1e-5, atol=1e-6, msg=f"mismatch in {key}")

    grouped_images = torch.as_tensor(file_ds.mmap_ptycho["images"][:])
    expected_rms = torch.sqrt(
        (N_PIX * N_PIX) /
        grouped_images.square().sum(dim=(-2, -1)).mean(dim=1)
    ).view(-1, 1, 1, 1)
    expected_physics = grouped_images.sum(dim=(-2, -1)).mean(dim=1).reciprocal()
    expected_physics = expected_physics.view(-1, 1, 1, 1)
    for dataset in (file_ds, mem_ds):
        torch.testing.assert_close(
            torch.as_tensor(dataset.mmap_ptycho["rms_scaling_constant"]),
            expected_rms)
        torch.testing.assert_close(
            torch.as_tensor(dataset.mmap_ptycho["physics_scaling_constant"]),
            expected_physics)


def test_from_np_group_normalization_omits_undefined_legacy_scalar():
    data_config, model_config, _ = _group_configs()
    diff3d, xcoords, ycoords, probe, _ = _make_grouped_arrays()

    dataset = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1), model_config, data_config)

    assert "scaling_constant" not in dataset.data_dict
    assert dataset.mmap_ptycho["rms_scaling_constant"].shape == (len(dataset), 1, 1, 1)
    _, probes_indexed, probe_scaling = dataset[torch.arange(4)]
    assert probes_indexed.shape[:2] == (4, 4)
    assert probe_scaling.shape == (4, 1, 1, 1)
    assert "scaling_constant" not in dataset.get_experiment_dataset(0).data_dict


def test_from_np_coords_relative_uses_tf_sign():
    payload = _make_grouped_arrays()
    data_config, model_config, _ = _group_configs()
    diff3d, xcoords, ycoords, probe, _ = payload
    positions = np.stack([ycoords, xcoords], axis=1)

    np.random.seed(123)
    dataset = PtychoDataset.from_np(
        diff3d, probe, positions, model_config, data_config)

    coords_global = dataset.mmap_ptycho["coords_global"]
    coords_center = dataset.mmap_ptycho["coords_center"]
    coords_relative = dataset.mmap_ptycho["coords_relative"]
    expected = -(coords_global - coords_center)

    torch.testing.assert_close(coords_relative, expected, rtol=0, atol=1e-6)
    assert coords_relative.abs().max() > 0


def test_from_np_group_rms_override_preserves_group_physics_factors():
    data_config, model_config, _ = _group_configs()
    diff3d, xcoords, ycoords, probe, _ = _make_grouped_arrays()
    positions = np.stack([ycoords, xcoords], axis=1)

    np.random.seed(123)
    baseline = PtychoDataset.from_np(diff3d, probe, positions, model_config, data_config)
    np.random.seed(123)
    override = PtychoDataset.from_np(
        diff3d, probe, positions, model_config, data_config, scaling_constant=7.5)

    assert torch.all(override.mmap_ptycho["rms_scaling_constant"] == 7.5)
    torch.testing.assert_close(override.mmap_ptycho["physics_scaling_constant"],
                               baseline.mmap_ptycho["physics_scaling_constant"])
    torch.testing.assert_close(override.data_dict["scaling_constant"], torch.tensor([7.5]))


def test_batch_max_factors_match_mmap_and_from_np(tmp_path):
    data_config, model_config, training_config = _configs(
        normalize='Batch', data_scaling='Max',
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    payload = _make_arrays()
    diff3d, xcoords, ycoords, probe, _ = payload

    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": payload},
                                  data_config, model_config, training_config)
    mem_ds = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1),
        model_config, data_config)

    expected_scalar = 1 / torch.from_numpy(diff3d).sum(dim=(-2, -1)).max()
    expected_shape = (len(file_ds), 1, 1, 1)
    for key in ("rms_scaling_constant", "physics_scaling_constant"):
        file_factors = torch.as_tensor(file_ds.mmap_ptycho[key][:])
        mem_factors = torch.as_tensor(mem_ds.mmap_ptycho[key])
        assert file_factors.shape == mem_factors.shape == expected_shape
        torch.testing.assert_close(file_factors, mem_factors)
        torch.testing.assert_close(file_factors,
                                   expected_scalar.expand(expected_shape))


def test_group_max_factors_match_mmap_and_from_np(tmp_path):
    data_config, model_config, training_config = _group_configs(
        data_scaling='Max')
    payload = _make_grouped_arrays()
    diff3d, xcoords, ycoords, probe, _ = payload

    np.random.seed(123)
    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": payload},
                                  data_config, model_config, training_config)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1),
        model_config, data_config)

    expected_shape = (len(file_ds), 1, 1, 1)
    image_sums = torch.as_tensor(file_ds.mmap_ptycho["images"][:]).sum(dim=(-2, -1))
    expected = image_sums.max(dim=1).values.reciprocal().view(-1, 1, 1, 1)
    for key in ("rms_scaling_constant", "physics_scaling_constant"):
        file_factors = torch.as_tensor(file_ds.mmap_ptycho[key][:])
        mem_factors = torch.as_tensor(mem_ds.mmap_ptycho[key])
        assert file_factors.shape == mem_factors.shape == expected_shape
        torch.testing.assert_close(file_factors, mem_factors)
        torch.testing.assert_close(file_factors, expected)


@pytest.mark.parametrize(
    "factor_fn",
    [hh.get_rms_scaling_factor, hh.get_physics_scaling_factor],
)
@pytest.mark.parametrize(
    ("normalize", "shape"),
    [("Batch", (3, 1, N_PIX, N_PIX)),
     ("Group", (3, 4, N_PIX, N_PIX))],
)
def test_max_scaling_rejects_zero_selected_denominators(
        factor_fn, normalize, shape):
    data_config = DataConfig(
        N=N_PIX, C=shape[1], normalize=normalize, data_scaling="Max")

    with pytest.raises(
        ValueError,
        match=r"Max scaling.*finite and strictly positive",
    ):
        factor_fn(torch.zeros(shape), data_config)


@pytest.mark.parametrize("source", ["mmap", "from_np"])
@pytest.mark.parametrize("normalize", ["Batch", "Group"])
def test_zero_energy_max_is_rejected_before_factor_persistence(
        tmp_path, source, normalize):
    if normalize == "Batch":
        data_config, _, training_config = _configs(
            normalize="Batch", data_scaling="Max",
            x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
        model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
        payload = _make_arrays()
    else:
        data_config, model_config, training_config = _group_configs(
            data_scaling="Max")
        payload = _make_grouped_arrays()

    diff3d, xcoords, ycoords, probe, obj = payload
    zero_payload = (np.zeros_like(diff3d), xcoords, ycoords, probe, obj)

    with pytest.raises(
        ValueError,
        match=r"Max scaling.*finite and strictly positive",
    ):
        if source == "mmap":
            np.random.seed(123)
            _build_file_dataset(
                tmp_path, {"zero.npz": zero_payload},
                data_config, model_config, training_config)
        else:
            np.random.seed(123)
            PtychoDataset.from_np(
                zero_payload[0], probe,
                np.stack([ycoords, xcoords], axis=1),
                model_config, data_config)


def test_from_np_group_max_rms_override_preserves_shaped_physics_factors():
    data_config, model_config, _ = _group_configs(data_scaling='Max')
    diff3d, xcoords, ycoords, probe, _ = _make_grouped_arrays()
    positions = np.stack([ycoords, xcoords], axis=1)

    np.random.seed(123)
    baseline = PtychoDataset.from_np(
        diff3d, probe, positions, model_config, data_config)
    np.random.seed(123)
    override = PtychoDataset.from_np(
        diff3d, probe, positions, model_config, data_config,
        scaling_constant=7.5)

    expected_shape = (len(baseline), 1, 1, 1)
    assert baseline.mmap_ptycho["physics_scaling_constant"].shape == expected_shape
    assert override.mmap_ptycho["physics_scaling_constant"].shape == expected_shape
    assert override.mmap_ptycho["rms_scaling_constant"].shape == expected_shape
    assert torch.all(override.mmap_ptycho["rms_scaling_constant"] == 7.5)
    torch.testing.assert_close(
        override.mmap_ptycho["physics_scaling_constant"],
        baseline.mmap_ptycho["physics_scaling_constant"])


def test_c1_group_uses_effective_batch_factors_in_both_paths(tmp_path):
    data_config, model_config, training_config = _configs(
        normalize='Group', x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    payload = _make_arrays()
    diff3d, xcoords, ycoords, probe, _ = payload

    np.random.seed(123)
    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": payload},
                                  data_config, model_config, training_config)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1), model_config, data_config)

    for key in ("rms_scaling_constant", "physics_scaling_constant"):
        torch.testing.assert_close(torch.as_tensor(mem_ds.mmap_ptycho[key]),
                                   torch.as_tensor(file_ds.mmap_ptycho[key][:]),
                                   rtol=1e-5, atol=1e-6, msg=f"mismatch in {key}")
    torch.testing.assert_close(file_ds.data_dict["scaling_constant"],
                               file_ds.mmap_ptycho["rms_scaling_constant"][0].reshape(1))
    torch.testing.assert_close(mem_ds.data_dict["scaling_constant"],
                               mem_ds.mmap_ptycho["rms_scaling_constant"][0].reshape(1))


def test_c1_none_uses_unit_factors_and_legacy_scalar_in_both_paths(tmp_path):
    data_config, model_config, training_config = _configs(
        normalize='None', x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    payload = _make_arrays()
    diff3d, xcoords, ycoords, probe, _ = payload

    np.random.seed(123)
    file_ds = _build_file_dataset(tmp_path, {"fixture.npz": payload},
                                  data_config, model_config, training_config)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        diff3d, probe, np.stack([ycoords, xcoords], axis=1), model_config, data_config)

    for dataset in (file_ds, mem_ds):
        assert torch.all(dataset.mmap_ptycho["rms_scaling_constant"] == 1)
        assert torch.all(dataset.mmap_ptycho["physics_scaling_constant"] == 1)
        torch.testing.assert_close(dataset.data_dict["scaling_constant"], torch.tensor([1.0]))
