"""Native Torch CLI mmap-ingestion contracts."""

import errno
import os
import pickle
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import pytest
import torch


_SAFE_HARD_LINK_COPY_ERRNOS = sorted(
    {
        getattr(errno, name)
        for name in (
            "EXDEV",
            "EPERM",
            "EOPNOTSUPP",
            "ENOTSUP",
            "ENOSYS",
            "EMLINK",
        )
        if hasattr(errno, name)
    }
)


@pytest.fixture
def payload():
    return SimpleNamespace(
        pt_data_config=SimpleNamespace(
            N=64,
            grid_size=2,
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        ),
        pt_model_config=SimpleNamespace(
            physics_forward_mode="amplitude",
            loss_function="MAE",
            amplitude_physics_gain=1.0,
            rect_s1s2_trainable=False,
            rect_s1s2_init="ones",
            cnn_output_mode="amplitude",
        ),
        pt_training_config=SimpleNamespace(
            epochs=1,
            n_groups=3,
            batch_size=2,
            sequential_sampling=True,
            torch_loss_mode="mae",
            learning_rate=0.001,
        ),
        pt_inference_config=SimpleNamespace(
            log_patch_stats=False,
            patch_stats_limit=None,
        ),
        tf_training_config=SimpleNamespace(
            output_dir="",
            sequential_sampling=True,
        ),
        execution_config=SimpleNamespace(
            accelerator="cpu",
            deterministic=True,
        ),
    )


def _source_file(tmp_path: Path, name: str = "selected.npz") -> Path:
    source = tmp_path / "source" / name
    source.parent.mkdir()
    source.write_bytes(b"selected payload")
    return source


def test_mmap_adapter_isolates_selected_basename_and_removes_stale_workspace(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    (source.parent / "unrelated.npz").write_bytes(b"unrelated payload")
    output_dir = tmp_path / "output"
    stale_file = (
        output_dir / "mmap_workspace" / "train" / "stale" / "old-content"
    )
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale")
    sentinel = object()

    def fake_dataset(**kwargs):
        staged_dir = Path(kwargs["ptycho_dir"])
        expected_staged_dir = (
            output_dir / "mmap_workspace" / "train" / "staged"
        )
        assert str(staged_dir).startswith("/proc/self/fd/")
        assert staged_dir.resolve() == expected_staged_dir
        assert [entry.name for entry in staged_dir.iterdir()] == [source.name]
        exact_files = kwargs["exact_npz_files"]
        assert len(exact_files) == 1
        exact_file = Path(exact_files[0])
        assert str(exact_file).startswith("/proc/self/fd/")
        assert exact_file.resolve() == expected_staged_dir / source.name
        assert exact_file.read_bytes() == source.read_bytes()
        assert not stale_file.exists()
        data_dir = Path(kwargs["data_dir"])
        expected_data_dir = (
            output_dir / "mmap_workspace" / "train" / "mmap" / "memmap"
        )
        assert str(data_dir).startswith("/proc/self/fd/")
        assert data_dir.resolve() == expected_data_dir
        data_prefix = Path(kwargs["data_prefix_dir"])
        assert str(data_prefix).startswith("/proc/self/fd/")
        assert data_prefix.resolve() == expected_data_dir.parent
        assert kwargs["remake_map"] is True
        assert kwargs["group_limit"] == payload.pt_training_config.n_groups
        assert kwargs["model_config"] is payload.pt_model_config
        assert kwargs["data_config"] is payload.pt_data_config
        assert kwargs["training_config"] is payload.pt_training_config
        assert (
            kwargs["sequential_sampling"]
            is payload.tf_training_config.sequential_sampling
        )
        assert data_dir.is_dir()
        return sentinel

    monkeypatch.setattr(dataloader, "PtychoDataset", fake_dataset)

    result = build_cli_mmap_dataset(
        source,
        payload=payload,
        output_dir=output_dir,
        role="train",
    )

    assert result is sentinel
    assert not (output_dir / "mmap_workspace" / "train" / "staged").exists()
    assert (output_dir / "mmap_workspace" / "train" / "mmap").is_dir()


def test_mmap_adapter_constructor_workspace_swap_stays_descriptor_anchored(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    workspace_root = output_dir / "mmap_workspace"
    detached_workspace = tmp_path / "detached-workspace"
    victim_workspace = tmp_path / "external-victim"
    victim_role = victim_workspace / "train"
    victim_staged = victim_role / "staged"
    victim_map = victim_role / "mmap" / "memmap"
    victim_staged.mkdir(parents=True)
    victim_map.mkdir(parents=True)
    (victim_staged / source.name).write_bytes(b"attacker-selected payload")
    (victim_map / "sentinel").write_bytes(b"external mmap contents")
    victim_snapshot = {
        path.relative_to(victim_workspace): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(victim_workspace.rglob("*"))
    }
    observed = {}

    def swap_then_construct(**kwargs):
        workspace_root.rename(detached_workspace)
        workspace_root.symlink_to(victim_workspace, target_is_directory=True)

        staged_file = Path(kwargs["exact_npz_files"][0])
        observed["source_bytes"] = staged_file.read_bytes()
        data_dir = Path(kwargs["data_dir"])
        (data_dir / "partial").mkdir(parents=True)
        (data_dir / "partial" / "created").write_bytes(b"anchored partial map")
        return object()

    monkeypatch.setattr(dataloader, "PtychoDataset", swap_then_construct)

    with pytest.raises(ValueError, match="symlink"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert observed["source_bytes"] == source.read_bytes()
    assert workspace_root.is_symlink()
    assert not (detached_workspace / "train").exists()
    assert {
        path.relative_to(victim_workspace): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(victim_workspace.rglob("*"))
    } == victim_snapshot


def test_mmap_adapter_pins_exact_staged_file_when_entry_is_replaced(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    staged_entry = (
        output_dir
        / "mmap_workspace"
        / "train"
        / "staged"
        / source.name
    )
    external_dir = tmp_path / "external-secret"
    external_dir.mkdir()
    external_secret = external_dir / "secret.npz"
    external_secret.write_bytes(b"external secret payload")
    external_snapshot = {
        path.relative_to(external_dir): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(external_dir.rglob("*"))
    }
    observed = {}

    def replace_entry_then_construct(**kwargs):
        staged_entry.unlink()
        staged_entry.symlink_to(external_secret)

        exact_files = kwargs["exact_npz_files"]
        assert len(exact_files) == 1
        observed["source_bytes"] = Path(exact_files[0]).read_bytes()
        (Path(kwargs["data_dir"]) / "partial").mkdir(parents=True)
        return object()

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        replace_entry_then_construct,
    )

    with pytest.raises(ValueError, match="staged.*(symlink|replaced)"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert observed["source_bytes"] == source.read_bytes()
    assert not (output_dir / "mmap_workspace" / "train").exists()
    assert {
        path.relative_to(external_dir): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(external_dir.rglob("*"))
    } == external_snapshot


@pytest.mark.parametrize("swapped_child", ["mmap", "memmap"])
def test_mmap_adapter_pins_map_children_when_visible_entry_is_replaced(
    tmp_path, monkeypatch, payload, swapped_child
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    role_workspace = output_dir / "mmap_workspace" / "train"
    mmap_dir = role_workspace / "mmap"
    memmap_dir = mmap_dir / "memmap"
    external_dir = tmp_path / "external-victim"
    external_dir.mkdir()
    (external_dir / "sentinel").write_bytes(b"external contents")
    if swapped_child == "mmap":
        (external_dir / "memmap").mkdir()
    external_snapshot = {
        path.relative_to(external_dir): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(external_dir.rglob("*"))
    }

    def replace_map_child_then_construct(**kwargs):
        if swapped_child == "mmap":
            mmap_dir.rename(role_workspace / "original-mmap")
            mmap_dir.symlink_to(external_dir, target_is_directory=True)
        else:
            memmap_dir.rename(mmap_dir / "original-memmap")
            memmap_dir.symlink_to(external_dir, target_is_directory=True)

        data_dir = Path(kwargs["data_dir"])
        (data_dir / "partial").mkdir(parents=True)
        (data_dir / "partial" / "created").write_bytes(b"partial map")
        data_prefix = Path(
            kwargs.get("data_prefix_dir", data_dir.parent)
        )
        (data_prefix / "partial-state").write_bytes(b"partial state")
        return object()

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        replace_map_child_then_construct,
    )

    with pytest.raises(ValueError, match=f"{swapped_child}.*(symlink|replaced)"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert not role_workspace.exists()
    assert {
        path.relative_to(external_dir): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(external_dir.rglob("*"))
    } == external_snapshot


def test_mmap_adapter_real_dataset_survives_descriptor_close_with_canonical_paths(
    tmp_path,
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig
    from ptycho_torch.dataloader import PtychoDataset

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "selected.npz"
    n_images = 2
    n_pixels = 8
    diffraction = np.arange(
        1,
        n_images * n_pixels * n_pixels + 1,
        dtype=np.float32,
    ).reshape(n_images, n_pixels, n_pixels)
    np.savez(
        source,
        diff3d=diffraction,
        xcoords=np.linspace(0.25, 0.75, n_images),
        ycoords=np.linspace(0.25, 0.75, n_images),
        probeGuess=np.ones((n_pixels, n_pixels), dtype=np.complex64),
        objectGuess=np.ones((n_pixels, n_pixels), dtype=np.complex64),
    )
    payload = SimpleNamespace(
        pt_data_config=DataConfig(
            N=n_pixels,
            C=1,
            grid_size=(1, 1),
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
            normalize="None",
            probe_normalize=False,
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        ),
        pt_model_config=ModelConfig(
            C_model=1,
            C_forward=1,
            object_big=False,
        ),
        pt_training_config=TrainingConfig(
            orchestrator="Mlflow",
            n_groups=None,
            num_workers=0,
        ),
        tf_training_config=SimpleNamespace(sequential_sampling=True),
    )
    output_dir = tmp_path / "output"
    canonical_map = (
        output_dir / "mmap_workspace" / "train" / "mmap" / "memmap"
    )

    dataset = build_cli_mmap_dataset(
        source,
        payload=payload,
        output_dir=output_dir,
        role="train",
    )

    assert dataset.data_dir == str(canonical_map)
    assert dataset.data_dir_path == canonical_map
    assert dataset.state_path == canonical_map.parent / "state_files.npz"
    assert dataset.manifest_path == canonical_map.parent / "mmap_manifest.json"
    assert dataset.ptycho_dir == str(source.parent.resolve())
    assert dataset.file_list == [source.resolve()]
    assert dataset.n_files == 1
    assert "/proc/self/fd" not in dataset.ptycho_dir
    assert all("/proc/self/fd" not in str(path) for path in dataset.file_list)
    assert dataset.mmap_ptycho._memmap_prefix == canonical_map
    for value in dataset.mmap_ptycho.values(
        include_nested=True,
        leaves_only=True,
    ):
        filename = getattr(value, "_filename", None)
        if filename is not None:
            assert Path(filename).is_relative_to(canonical_map)
            assert "/proc/self/fd" not in filename

    images = dataset.mmap_ptycho["images"]
    assert float(images.abs().sum()) > 0.0
    serialized_dataset = pickle.dumps(dataset)
    assert b"/proc/self/fd" not in serialized_dataset
    reloaded_dataset = pickle.loads(serialized_dataset)
    assert reloaded_dataset.data_dir_path == canonical_map
    assert reloaded_dataset.ptycho_dir == str(source.parent.resolve())
    assert reloaded_dataset.file_list == [source.resolve()]
    torch.testing.assert_close(
        reloaded_dataset.mmap_ptycho["images"],
        images,
    )
    reopened_dataset = PtychoDataset.from_existing_map(
        canonical_map,
        payload.pt_model_config,
        payload.pt_data_config,
    )
    torch.testing.assert_close(
        reopened_dataset.mmap_ptycho["images"],
        images,
    )
    assert not (
        output_dir / "mmap_workspace" / "train" / "staged"
    ).exists()


def test_mmap_adapter_fails_closed_without_fd_backed_path_support(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        mmap_ingestion,
        "_FD_DIRECTORY_ROOT",
        tmp_path / "missing-fd-directory-root",
    )
    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Linux.*procfs.*[/]proc/self/fd",
    ):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert not output_dir.exists()


def test_mmap_adapter_rejects_non_linux_before_any_output_mutation(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(mmap_ingestion.sys, "platform", "darwin")
    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Linux.*procfs.*[/]proc/self/fd",
    ):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert not output_dir.exists()


def test_mmap_adapter_retains_all_fds_through_rebase_and_closes_before_return(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    observed = {}
    real_rebase = mmap_ingestion._rebase_dataset_paths

    def fake_dataset(**kwargs):
        observed["constructor_paths"] = {
            "staged": Path(kwargs["ptycho_dir"]),
            "staged_file": Path(kwargs["exact_npz_files"][0]),
            "mmap": Path(kwargs["data_prefix_dir"]),
            "memmap": Path(kwargs["data_dir"]),
        }
        return object()

    def verify_open_descriptors(dataset, **rebase_kwargs):
        fd_root = mmap_ingestion._FD_DIRECTORY_ROOT
        constructor_fds = {
            name: int(path.relative_to(fd_root).parts[0])
            for name, path in observed["constructor_paths"].items()
        }
        assert all(
            len(path.relative_to(fd_root).parts) == 1
            for path in observed["constructor_paths"].values()
        )
        assert stat.S_ISDIR(os.fstat(constructor_fds["staged"]).st_mode)
        assert stat.S_ISREG(
            os.fstat(constructor_fds["staged_file"]).st_mode
        )
        assert stat.S_ISDIR(os.fstat(constructor_fds["mmap"]).st_mode)
        assert stat.S_ISDIR(os.fstat(constructor_fds["memmap"]).st_mode)
        assert (
            output_dir / "mmap_workspace" / "train" / "staged"
        ).is_dir()

        retained_fds = set(constructor_fds.values())
        identity_paths = (
            source,
            output_dir,
            output_dir / "mmap_workspace",
            output_dir / "mmap_workspace" / "train",
        )
        for identity_path in identity_paths:
            identity_stat = os.stat(identity_path)
            for candidate in fd_root.iterdir():
                try:
                    if os.path.samestat(
                        identity_stat,
                        os.fstat(int(candidate.name)),
                    ):
                        retained_fds.add(int(candidate.name))
                except (OSError, ValueError):
                    continue
        observed["fds"] = retained_fds
        return real_rebase(
            dataset,
            **rebase_kwargs,
        )

    monkeypatch.setattr(dataloader, "PtychoDataset", fake_dataset)
    monkeypatch.setattr(
        mmap_ingestion,
        "_rebase_dataset_paths",
        verify_open_descriptors,
    )

    build_cli_mmap_dataset(
        source,
        payload=payload,
        output_dir=output_dir,
        role="train",
    )

    for fd in observed["fds"]:
        with pytest.raises(OSError) as closed_fd:
            os.fstat(fd)
        assert closed_fd.value.errno == errno.EBADF


def test_mmap_adapter_hard_links_source_when_possible(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    observed = {}

    def fake_dataset(**kwargs):
        staged = Path(kwargs["exact_npz_files"][0])
        observed["same_inode"] = staged.stat().st_ino == source.stat().st_ino
        return object()

    monkeypatch.setattr(dataloader, "PtychoDataset", fake_dataset)

    build_cli_mmap_dataset(
        source,
        payload=payload,
        output_dir=tmp_path / "output",
        role="train",
    )

    assert observed["same_inode"] is True


def test_mmap_adapter_rejects_source_path_swap_before_constructor(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    selected_bytes = source.read_bytes()
    detached_source = source.parent / "detached-selected.npz"
    external_secret = tmp_path / "external-secret.npz"
    external_secret.write_bytes(b"external secret payload")
    output_dir = tmp_path / "output"
    real_link = os.link
    constructor_called = False

    def link_then_swap_source(source_path, staged_name, **kwargs):
        result = real_link(source_path, staged_name, **kwargs)
        source.rename(detached_source)
        source.symlink_to(external_secret)
        return result

    def unexpected_dataset(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        return object()

    monkeypatch.setattr("os.link", link_then_swap_source)
    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="source.*(symlink|replaced)"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert constructor_called is False
    assert source.is_symlink()
    assert detached_source.read_bytes() == selected_bytes
    assert external_secret.read_bytes() == b"external secret payload"
    assert not (output_dir / "mmap_workspace" / "train").exists()


@pytest.mark.parametrize(
    "link_errno",
    _SAFE_HARD_LINK_COPY_ERRNOS,
    ids=lambda value: errno.errorcode[value],
)
def test_mmap_adapter_copies_source_when_hard_link_is_unsupported(
    tmp_path, monkeypatch, payload, link_errno
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    observed = {}
    source.chmod(0o640)

    def fail_link(*args, **kwargs):
        raise OSError(link_errno, "hard link unavailable")

    def fake_dataset(**kwargs):
        staged = Path(kwargs["exact_npz_files"][0])
        observed["content"] = staged.read_bytes()
        observed["different_inode"] = staged.stat().st_ino != source.stat().st_ino
        observed["mode"] = stat.S_IMODE(staged.stat().st_mode)
        return object()

    monkeypatch.setattr("os.link", fail_link)
    monkeypatch.setattr(dataloader, "PtychoDataset", fake_dataset)

    build_cli_mmap_dataset(
        source,
        payload=payload,
        output_dir=tmp_path / "output",
        role="test",
    )

    assert observed == {
        "content": source.read_bytes(),
        "different_inode": True,
        "mode": 0o640,
    }


def test_mmap_adapter_pins_copy_fallback_before_staged_entry_replacement(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    staged_entry = (
        output_dir
        / "mmap_workspace"
        / "test"
        / "staged"
        / source.name
    )
    detached_copy = tmp_path / "detached-copy.npz"
    external_secret = tmp_path / "external-secret.npz"
    external_secret.write_bytes(b"external secret payload")
    real_copy = mmap_ingestion._copy_file_exclusive
    real_link = os.link
    constructor_called = False

    def fail_link(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    def copy_then_replace_entry(source_fd, staged_name, *, staged_fd):
        copied_fd = real_copy(
            source_fd,
            staged_name,
            staged_fd=staged_fd,
        )
        staged_entry.rename(detached_copy)
        real_link(external_secret, staged_entry)
        return copied_fd

    def unexpected_dataset(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        return object()

    monkeypatch.setattr("os.link", fail_link)
    monkeypatch.setattr(
        mmap_ingestion,
        "_copy_file_exclusive",
        copy_then_replace_entry,
    )
    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="staged.*replaced"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="test",
        )

    assert constructor_called is False
    assert detached_copy.read_bytes() == source.read_bytes()
    assert external_secret.read_bytes() == b"external secret payload"
    assert not (output_dir / "mmap_workspace" / "test").exists()


@pytest.mark.parametrize("symlink_entry", ["mmap_workspace", "role_workspace"])
def test_mmap_adapter_rejects_managed_chain_symlink_without_touching_victim(
    tmp_path, monkeypatch, payload, symlink_entry
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    victim = tmp_path / "external-victim"
    victim.mkdir()
    if symlink_entry == "mmap_workspace":
        victim_role = victim / "train"
        victim_role.mkdir()
        managed_link = output_dir / "mmap_workspace"
        managed_link.symlink_to(victim, target_is_directory=True)
    else:
        workspace_root = output_dir / "mmap_workspace"
        workspace_root.mkdir()
        victim_role = victim
        managed_link = workspace_root / "train"
        managed_link.symlink_to(victim_role, target_is_directory=True)
    sentinel = victim_role / "do-not-delete"
    sentinel.write_bytes(b"external contents")

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(ValueError, match="symlink"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert managed_link.is_symlink()
    assert sentinel.read_bytes() == b"external contents"


@pytest.mark.parametrize("symlink_entry", ["mmap_workspace", "role_workspace"])
def test_mmap_adapter_rejects_dangling_managed_chain_symlink(
    tmp_path, monkeypatch, payload, symlink_entry
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    missing_target = tmp_path / "missing-target"
    if symlink_entry == "mmap_workspace":
        managed_link = output_dir / "mmap_workspace"
    else:
        workspace_root = output_dir / "mmap_workspace"
        workspace_root.mkdir()
        managed_link = workspace_root / "train"
    managed_link.symlink_to(missing_target, target_is_directory=True)

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(ValueError, match="symlink"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert managed_link.is_symlink()


def test_mmap_adapter_parent_swap_during_role_deletion_cannot_touch_victim(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    workspace_root = output_dir / "mmap_workspace"
    stale_role = workspace_root / "train"
    stale_role.mkdir(parents=True)
    (stale_role / "stale-map").write_bytes(b"replace me")

    detached_workspace = tmp_path / "detached-workspace"
    victim_workspace = tmp_path / "external-victim"
    victim_role = victim_workspace / "train"
    (victim_role / "nested").mkdir(parents=True)
    (victim_role / "sentinel").write_bytes(b"victim sentinel")
    (victim_role / "nested" / "payload").write_bytes(b"nested victim")
    victim_snapshot = {
        path.relative_to(victim_workspace): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(victim_workspace.rglob("*"))
    }

    real_rmtree = shutil.rmtree
    injected = False
    deletion_dir_fds = []
    constructor_called = False

    def swap_parent_then_delete(path, *args, **kwargs):
        nonlocal injected
        if not injected:
            workspace_root.rename(detached_workspace)
            workspace_root.symlink_to(victim_workspace, target_is_directory=True)
            injected = True
            deletion_dir_fds.append(kwargs.get("dir_fd"))
        return real_rmtree(path, *args, **kwargs)

    def unexpected_dataset(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        return object()

    monkeypatch.setattr("shutil.rmtree", swap_parent_then_delete)
    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="symlink"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert injected is True
    assert deletion_dir_fds[0] is not None
    with pytest.raises(OSError) as closed_fd:
        os.fstat(deletion_dir_fds[0])
    assert closed_fd.value.errno == errno.EBADF
    assert constructor_called is False
    assert workspace_root.is_symlink()
    assert {
        path.relative_to(victim_workspace): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(victim_workspace.rglob("*"))
    } == victim_snapshot


def test_mmap_adapter_workspace_swap_during_first_revalidation_cleans_anchor(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    workspace_root = output_dir / "mmap_workspace"
    stale_role = workspace_root / "train"
    stale_role.mkdir(parents=True)
    (stale_role / "stale-map").write_bytes(b"replace me")

    detached_workspace = tmp_path / "detached-workspace"
    victim_workspace = tmp_path / "external-victim"
    victim_role = victim_workspace / "train"
    victim_role.mkdir(parents=True)
    sentinel = victim_role / "sentinel"
    sentinel.write_bytes(b"victim sentinel")

    real_validate = mmap_ingestion._validate_visible_directory
    injected = False
    constructor_called = False

    def swap_on_workspace_validation(path, fd, *, label):
        nonlocal injected
        if label == "workspace" and not injected:
            workspace_root.rename(detached_workspace)
            workspace_root.symlink_to(victim_workspace, target_is_directory=True)
            injected = True
        return real_validate(path, fd, label=label)

    def unexpected_dataset(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        return object()

    monkeypatch.setattr(
        mmap_ingestion,
        "_validate_visible_directory",
        swap_on_workspace_validation,
    )
    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="symlink"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert injected is True
    assert constructor_called is False
    assert sentinel.read_bytes() == b"victim sentinel"
    assert not (detached_workspace / "train").exists()


def test_mmap_adapter_rejects_lexically_contained_source_symlink_before_cleanup(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    output_dir = tmp_path / "output"
    role_workspace = output_dir / "mmap_workspace" / "train"
    role_workspace.mkdir(parents=True)
    target = tmp_path / "external-source.npz"
    target.write_bytes(b"source target contents")
    source_link = role_workspace / "selected.npz"
    source_link.symlink_to(target)

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(ValueError, match="must be outside"):
        build_cli_mmap_dataset(
            source_link,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert source_link.is_symlink()
    assert source_link.resolve() == target
    assert target.read_bytes() == b"source target contents"


def test_mmap_adapter_rejects_source_inside_symlinked_output_alias_before_cleanup(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    canonical_output = tmp_path / "canonical-output"
    role_workspace = canonical_output / "mmap_workspace" / "train"
    role_workspace.mkdir(parents=True)
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(canonical_output, target_is_directory=True)
    target = tmp_path / "external-source.npz"
    target.write_bytes(b"source target contents")
    source_link = output_alias / "mmap_workspace" / "train" / "selected.npz"
    source_link.symlink_to(target)

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(ValueError, match="must be outside"):
        build_cli_mmap_dataset(
            source_link,
            payload=payload,
            output_dir=output_alias,
            role="train",
        )

    assert source_link.is_symlink()
    assert source_link.resolve() == target
    assert target.read_bytes() == b"source target contents"


def test_mmap_adapter_rejects_source_inside_arbitrary_role_alias_before_cleanup(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    output_dir = tmp_path / "output"
    role_workspace = output_dir / "mmap_workspace" / "train"
    role_workspace.mkdir(parents=True)
    role_alias = tmp_path / "role-alias"
    role_alias.symlink_to(role_workspace, target_is_directory=True)
    target = tmp_path / "external-source.npz"
    target.write_bytes(b"source target contents")
    source_link = role_alias / "selected.npz"
    source_link.symlink_to(target)

    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(ValueError, match="must be outside"):
        build_cli_mmap_dataset(
            source_link,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert source_link.is_symlink()
    assert source_link.resolve() == target
    assert target.read_bytes() == b"source target contents"


@pytest.mark.parametrize("link_errno", [errno.EEXIST, errno.EXDEV])
def test_mmap_adapter_destination_collision_does_not_overwrite_symlink_target(
    tmp_path, monkeypatch, payload, link_errno
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    victim = tmp_path / "external-victim"
    victim.write_bytes(b"victim contents")

    def inject_collision(source_path, staged_path, **kwargs):
        destination_dir_fd = kwargs.get("dst_dir_fd")
        if destination_dir_fd is None:
            Path(staged_path).symlink_to(victim)
        else:
            os.symlink(victim, staged_path, dir_fd=destination_dir_fd)
        raise OSError(link_errno, "link failed after destination collision", staged_path)

    monkeypatch.setattr("os.link", inject_collision)
    monkeypatch.setattr(
        dataloader,
        "PtychoDataset",
        lambda **kwargs: pytest.fail("dataset constructor must not run"),
    )

    with pytest.raises(OSError) as exc_info:
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert exc_info.value.errno == errno.EEXIST
    assert victim.read_bytes() == b"victim contents"
    assert not (output_dir / "mmap_workspace" / "train").exists()


def test_mmap_adapter_removes_partial_role_workspace_on_constructor_failure(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    expected = RuntimeError("dataset construction failed")

    def fail_dataset(**kwargs):
        (Path(kwargs["data_dir"]) / "partial").mkdir(parents=True)
        raise expected

    monkeypatch.setattr(dataloader, "PtychoDataset", fail_dataset)

    with pytest.raises(RuntimeError) as exc_info:
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="test",
        )

    assert exc_info.value is expected
    assert not (output_dir / "mmap_workspace" / "test").exists()


def test_mmap_adapter_removes_partial_workspace_on_constructor_base_exception(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    expected = KeyboardInterrupt("construction interrupted")

    def interrupt_dataset(**kwargs):
        (Path(kwargs["data_dir"]) / "partial").mkdir(parents=True)
        raise expected

    monkeypatch.setattr(dataloader, "PtychoDataset", interrupt_dataset)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="test",
        )

    assert exc_info.value is expected
    assert not (output_dir / "mmap_workspace" / "test").exists()


def test_mmap_adapter_cleanup_failure_does_not_mask_constructor_base_exception(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    role_workspace = output_dir / "mmap_workspace" / "test"
    expected = KeyboardInterrupt("construction interrupted")
    real_rmtree = shutil.rmtree

    def fail_role_cleanup(path, *args, **kwargs):
        if (
            Path(path) in {role_workspace, Path("test")}
            and kwargs.get("dir_fd") is not None
        ):
            raise OSError("role cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    def interrupt_dataset(**kwargs):
        (Path(kwargs["data_dir"]) / "partial").mkdir(parents=True)
        raise expected

    monkeypatch.setattr("shutil.rmtree", fail_role_cleanup)
    monkeypatch.setattr(dataloader, "PtychoDataset", interrupt_dataset)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="test",
        )

    assert exc_info.value is expected
    assert role_workspace.exists()


def test_mmap_adapter_surfaces_success_cleanup_failure_and_removes_role(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    source = _source_file(tmp_path)
    output_dir = tmp_path / "output"
    role_workspace = output_dir / "mmap_workspace" / "train"
    staged_dir = role_workspace / "staged"
    real_rmtree = shutil.rmtree

    def fail_staged_cleanup(path, *args, **kwargs):
        if Path(path) == staged_dir or (
            Path(path) == Path("staged")
            and kwargs.get("dir_fd") is not None
        ):
            if kwargs.get("ignore_errors") is True:
                return None
            raise OSError("staged cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    def successful_dataset(**kwargs):
        assert Path(kwargs["data_dir"]).is_dir()
        return object()

    monkeypatch.setattr("shutil.rmtree", fail_staged_cleanup)
    monkeypatch.setattr(dataloader, "PtychoDataset", successful_dataset)

    with pytest.raises(OSError, match="staged cleanup failed"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert not os.path.lexists(role_workspace)


def test_mmap_adapter_rejects_source_inside_role_workspace_before_cleanup(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    output_dir = tmp_path / "output"
    source = output_dir / "mmap_workspace" / "train" / "input.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"must survive validation")

    def unexpected_dataset(**kwargs):
        pytest.fail("dataset constructor must not run")

    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="must be outside"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert source.read_bytes() == b"must survive validation"


def test_mmap_adapter_rejects_source_inside_other_role_before_any_mutation(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.cli.mmap_ingestion import build_cli_mmap_dataset
    from ptycho_torch import dataloader

    output_dir = tmp_path / "output"
    source = (
        output_dir
        / "mmap_workspace"
        / "test"
        / "staged"
        / "input.npz"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"must survive cross-role validation")
    workspace_root = output_dir / "mmap_workspace"
    workspace_snapshot = {
        path.relative_to(workspace_root): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(workspace_root.rglob("*"))
    }
    constructor_called = False

    def unexpected_dataset(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        return object()

    monkeypatch.setattr(dataloader, "PtychoDataset", unexpected_dataset)

    with pytest.raises(ValueError, match="workspace"):
        build_cli_mmap_dataset(
            source,
            payload=payload,
            output_dir=output_dir,
            role="train",
        )

    assert constructor_called is False
    assert source.read_bytes() == b"must survive cross-role validation"
    assert {
        path.relative_to(workspace_root): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(workspace_root.rglob("*"))
    } == workspace_snapshot


def test_mmap_source_preflight_rejects_source_under_resolved_workspace_alias(
    tmp_path,
):
    from ptycho_torch.cli.mmap_ingestion import validate_cli_mmap_sources

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    external_workspace = tmp_path / "external-workspace"
    source = external_workspace / "train" / "selected.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"must survive resolved-alias validation")
    workspace_link = output_dir / "mmap_workspace"
    workspace_link.symlink_to(external_workspace, target_is_directory=True)

    with pytest.raises(ValueError, match="workspace"):
        validate_cli_mmap_sources((source,), output_dir=output_dir)

    assert workspace_link.is_symlink()
    assert source.read_bytes() == b"must survive resolved-alias validation"


@pytest.mark.parametrize(
    "protected_source_role",
    ["test-source-in-train", "train-source-in-test"],
)
def test_native_cli_preflights_both_sources_before_any_role_build(
    tmp_path, monkeypatch, payload, protected_source_role
):
    from ptycho_torch.cli import mmap_ingestion
    from ptycho_torch import dataloader
    from ptycho_torch.train import cli_main
    from ptycho_torch.workflows import components as workflow_components

    output_dir = tmp_path / "output"
    workspace_root = output_dir / "mmap_workspace"
    external_sources = tmp_path / "external-sources"
    external_sources.mkdir()
    if protected_source_role == "test-source-in-train":
        train_file = external_sources / "train.npz"
        test_file = (
            workspace_root / "train" / "stale-source" / "test.npz"
        )
    else:
        train_file = workspace_root / "test" / "stale-source" / "train.npz"
        test_file = external_sources / "test.npz"
    train_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_bytes(b"selected train payload")
    test_file.write_bytes(b"selected test payload")
    (workspace_root / "workspace-sentinel").write_bytes(b"preserve workspace")
    workspace_snapshot = {
        path.relative_to(workspace_root): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(workspace_root.rglob("*"))
    }
    selected_snapshots = {
        train_file: train_file.read_bytes(),
        test_file: test_file.read_bytes(),
    }
    payload.tf_training_config.output_dir = str(output_dir)
    real_build = mmap_ingestion.build_cli_mmap_dataset
    build_calls = []
    constructor_calls = []

    def tracking_build(npz_file, **kwargs):
        build_calls.append((Path(npz_file), kwargs["role"]))
        return real_build(npz_file, **kwargs)

    def fake_dataset(**kwargs):
        constructor_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_data_file",
            str(train_file),
            "--test_data_file",
            str(test_file),
            "--output_dir",
            str(output_dir),
            "--n_images",
            "3",
            "--max_epochs",
            "1",
        ],
    )
    monkeypatch.setattr(dataloader, "PtychoDataset", fake_dataset)

    with patch(
        "ptycho_torch.config_factory.create_training_payload",
        return_value=payload,
    ), patch(
        "ptycho_torch.cli.mmap_ingestion.build_cli_mmap_dataset",
        side_effect=tracking_build,
    ), patch.object(
        workflow_components,
        "run_cdi_example_torch",
        return_value=(None, None, {"models": {}}),
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli_main()

    assert exc_info.value.code == 1
    assert build_calls == []
    assert constructor_calls == []
    for selected_source, expected_bytes in selected_snapshots.items():
        assert selected_source.read_bytes() == expected_bytes
    assert {
        path.relative_to(workspace_root): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(workspace_root.rglob("*"))
    } == workspace_snapshot


def test_native_cli_routes_exact_mmap_datasets_and_resolved_payload(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.train import cli_main

    train_file = _source_file(tmp_path, "train.npz")
    test_file = train_file.parent / "test.npz"
    test_file.write_bytes(b"test payload")
    output_dir = tmp_path / "output"
    payload.tf_training_config.output_dir = str(output_dir)
    train_dataset = object()
    test_dataset = object()
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return None, None, {"models": {}}

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_data_file",
            str(train_file),
            "--test_data_file",
            str(test_file),
            "--output_dir",
            str(output_dir),
            "--n_images",
            "3",
            "--max_epochs",
            "1",
        ],
    )

    with patch(
        "ptycho_torch.config_factory.create_training_payload",
        return_value=payload,
    ), patch(
        "ptycho_torch.cli.mmap_ingestion.build_cli_mmap_dataset",
        side_effect=[train_dataset, test_dataset],
    ) as build_dataset, patch(
        "ptycho_torch.workflows.components.run_cdi_example_torch",
        side_effect=fake_run,
    ), patch(
        "ptycho.raw_data.RawData.from_file",
        side_effect=AssertionError("RawData fallback is forbidden"),
    ):
        cli_main()

    assert build_dataset.call_args_list == [
        call(
            train_file,
            payload=payload,
            output_dir=output_dir,
            role="train",
        ),
        call(
            test_file,
            payload=payload,
            output_dir=output_dir,
            role="test",
        ),
    ]
    assert captured["train_data"] is train_dataset
    assert captured["test_data"] is test_dataset
    assert captured["resolved_payload"] is payload


def test_native_cli_patch_stats_reads_mmap_images(
    tmp_path, monkeypatch, payload
):
    from ptycho_torch.train import cli_main

    train_file = _source_file(tmp_path, "train.npz")
    output_dir = tmp_path / "output"
    payload.tf_training_config.output_dir = str(output_dir)
    payload.pt_inference_config.log_patch_stats = True
    images = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
    train_dataset = SimpleNamespace(mmap_ptycho={"images": images})
    captured = {}

    class FakeLogger:
        def __init__(self, **kwargs):
            captured["logger_kwargs"] = kwargs

        def log_batch(self, tensor, **kwargs):
            captured["tensor"] = tensor.detach().cpu().numpy()
            captured["log_kwargs"] = kwargs

        def finalize(self):
            captured["finalized"] = True

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_data_file",
            str(train_file),
            "--output_dir",
            str(output_dir),
            "--n_images",
            "3",
            "--max_epochs",
            "1",
            "--log-patch-stats",
        ],
    )

    with patch(
        "ptycho_torch.config_factory.create_training_payload",
        return_value=payload,
    ), patch(
        "ptycho_torch.cli.mmap_ingestion.build_cli_mmap_dataset",
        return_value=train_dataset,
    ), patch(
        "ptycho_torch.workflows.components.run_cdi_example_torch",
        return_value=(None, None, {"train_container": train_dataset}),
    ), patch(
        "ptycho_torch.patch_stats_instrumentation.PatchStatsLogger",
        FakeLogger,
    ):
        cli_main()

    np.testing.assert_array_equal(
        captured["tensor"],
        images[: payload.pt_training_config.batch_size],
    )
    assert captured["log_kwargs"] == {"phase": "train", "batch_idx": 0}
    assert captured["finalized"] is True
