"""Exact-file mmap ingestion for the native Torch CLI."""

import errno
import os
import shutil
import stat
import sys
from pathlib import Path


_HARD_LINK_COPY_ERRNOS = frozenset(
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
)
_RMTREE_AVOIDS_SYMLINK_ATTACKS = bool(
    getattr(shutil.rmtree, "avoids_symlink_attacks", False)
)
_HAS_REQUIRED_DIR_FD_SUPPORT = all(
    function in os.supports_dir_fd
    for function in (os.mkdir, os.open, os.stat, os.link)
)
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_FD_DIRECTORY_ROOT = Path("/proc/self/fd")
_DESCRIPTOR_SAFETY_ERROR = (
    "Native Torch CLI mmap ingestion requires Linux with procfs mounted and "
    "accessible at /proc/self/fd, descriptor-relative filesystem operations, "
    "and no-follow path handling; no path-based fallback is supported."
)


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def validate_cli_mmap_sources(
    npz_files,
    *,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Validate selected sources against the whole managed mmap workspace."""
    lexical_workspace_root = (
        _lexical_absolute(Path(output_dir)) / "mmap_workspace"
    )
    workspace_entry = (
        Path(output_dir).resolve(strict=False) / "mmap_workspace"
    )
    resolved_workspace_root = workspace_entry.resolve(strict=False)
    workspace_roots = (
        lexical_workspace_root,
        workspace_entry,
        resolved_workspace_root,
    )
    canonical_sources = []
    for npz_file in npz_files:
        npz_file = Path(npz_file)
        lexical_source = _lexical_absolute(npz_file)
        if any(
            _is_within(lexical_source, workspace_root)
            for workspace_root in workspace_roots
        ):
            raise ValueError(
                f"Source NPZ {lexical_source} must be outside the mmap "
                f"workspace {resolved_workspace_root}."
            )

        canonical_source_entry = (
            npz_file.parent.resolve(strict=True) / npz_file.name
        )
        if any(
            _is_within(canonical_source_entry, workspace_root)
            for workspace_root in workspace_roots
        ):
            raise ValueError(
                f"Source NPZ {canonical_source_entry} must be outside the mmap "
                f"workspace {resolved_workspace_root}."
            )

        source_path = npz_file.resolve(strict=True)
        if any(
            _is_within(source_path, workspace_root)
            for workspace_root in workspace_roots
        ):
            raise ValueError(
                f"Source NPZ {source_path} must be outside the mmap "
                f"workspace {resolved_workspace_root}."
            )
        if not source_path.is_file():
            raise ValueError(f"Source NPZ is not a regular file: {source_path}.")
        canonical_sources.append(source_path)

    return tuple(canonical_sources)


def _validate_managed_role(
    *,
    output_root: Path,
    workspace_root: Path,
    role_workspace: Path,
    role: str,
) -> None:
    expected_workspace_root = output_root / "mmap_workspace"
    expected_role_workspace = expected_workspace_root / role
    if (
        workspace_root != expected_workspace_root
        or role_workspace != expected_role_workspace
        or role_workspace.relative_to(output_root).parts
        != ("mmap_workspace", role)
    ):
        raise ValueError(
            f"Refusing unsafe mmap role workspace {role_workspace}; "
            f"expected {expected_role_workspace}."
        )

    for managed_path in (workspace_root, role_workspace):
        if managed_path.is_symlink():
            raise ValueError(
                f"Refusing symlink in managed mmap workspace chain: "
                f"{managed_path}."
            )

    if (
        workspace_root.resolve(strict=False) != workspace_root
        or role_workspace.resolve(strict=False) != role_workspace
    ):
        raise ValueError(
            f"Managed mmap role workspace {role_workspace} must remain beneath "
            f"the trusted output root {output_root} without symlink traversal."
        )


def _require_descriptor_safety() -> None:
    if (
        not sys.platform.startswith("linux")
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or not _RMTREE_AVOIDS_SYMLINK_ATTACKS
        or not _HAS_REQUIRED_DIR_FD_SUPPORT
    ):
        raise RuntimeError(_DESCRIPTOR_SAFETY_ERROR)

    probe_fd = None
    try:
        probe_fd = os.open(".", _DIRECTORY_OPEN_FLAGS)
        descriptor_path_stat = os.stat(
            _FD_DIRECTORY_ROOT / str(probe_fd),
            follow_symlinks=True,
        )
        descriptor_stat = os.fstat(probe_fd)
        if (
            not stat.S_ISDIR(descriptor_path_stat.st_mode)
            or not os.path.samestat(descriptor_path_stat, descriptor_stat)
        ):
            raise OSError("procfs descriptor path does not match its live fd")
    except OSError as exc:
        raise RuntimeError(_DESCRIPTOR_SAFETY_ERROR) from exc
    finally:
        _close_fd(probe_fd)


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _open_directory(path, *, dir_fd: int | None = None) -> int:
    fd = os.open(path, _DIRECTORY_OPEN_FLAGS, dir_fd=dir_fd)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(f"Managed mmap path is not a directory: {path}.")
        return fd
    except BaseException:
        _close_fd(fd)
        raise


def _open_regular_file(path, *, dir_fd: int | None = None) -> int:
    fd = os.open(path, _FILE_OPEN_FLAGS, dir_fd=dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Managed mmap path is not a file: {path}.")
        return fd
    except BaseException:
        _close_fd(fd)
        raise


def _entry_lstat(parent_fd: int, name: str):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _validate_directory_entry(
    parent_fd: int,
    name: str,
    fd: int,
    *,
    label: str,
) -> None:
    entry_stat = _entry_lstat(parent_fd, name)
    if entry_stat is None:
        raise ValueError(f"Managed mmap {label} was removed: {name}.")
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is a symlink: {name}.")
    if not stat.S_ISDIR(entry_stat.st_mode):
        raise ValueError(
            f"Managed mmap {label} is not a directory: {name}."
        )
    if not os.path.samestat(entry_stat, os.fstat(fd)):
        raise ValueError(
            f"Managed mmap {label} was replaced after opening: {name}."
        )


def _validate_file_entry(
    parent_fd: int,
    name: str,
    fd: int,
    *,
    label: str,
) -> None:
    entry_stat = _entry_lstat(parent_fd, name)
    if entry_stat is None:
        raise ValueError(f"Managed mmap {label} was removed: {name}.")
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is a symlink: {name}.")
    if not stat.S_ISREG(entry_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is not a file: {name}.")
    if not os.path.samestat(entry_stat, os.fstat(fd)):
        raise ValueError(
            f"Managed mmap {label} was replaced after opening: {name}."
        )


def _validate_visible_directory(path: Path, fd: int, *, label: str) -> None:
    try:
        visible_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Managed mmap {label} was replaced: {path}.") from exc

    if stat.S_ISLNK(visible_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is a symlink: {path}.")
    if not stat.S_ISDIR(visible_stat.st_mode):
        raise ValueError(
            f"Managed mmap {label} is not a directory: {path}."
        )
    if not os.path.samestat(visible_stat, os.fstat(fd)):
        raise ValueError(
            f"Managed mmap {label} was replaced after opening: {path}."
        )


def _validate_visible_file(path: Path, fd: int, *, label: str) -> None:
    try:
        visible_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Managed mmap {label} was replaced: {path}.") from exc

    if stat.S_ISLNK(visible_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is a symlink: {path}.")
    if not stat.S_ISREG(visible_stat.st_mode):
        raise ValueError(f"Managed mmap {label} is not a file: {path}.")
    if not os.path.samestat(visible_stat, os.fstat(fd)):
        raise ValueError(
            f"Managed mmap {label} was replaced after opening: {path}."
        )


def _verified_fd_directory_path(path: Path, fd: int, *, label: str) -> Path:
    expected_path = _FD_DIRECTORY_ROOT / str(fd)
    if path != expected_path:
        raise RuntimeError(
            f"Refusing non-descriptor-backed mmap {label} path: {path}."
        )
    try:
        path_stat = os.stat(path, follow_symlinks=True)
        descriptor_stat = os.fstat(fd)
    except OSError as exc:
        raise RuntimeError(
            "Safe CLI mmap construction requires accessible descriptor-backed "
            f"paths under {_FD_DIRECTORY_ROOT}: {label}."
        ) from exc

    if not stat.S_ISDIR(path_stat.st_mode) or not stat.S_ISDIR(
        descriptor_stat.st_mode
    ):
        raise RuntimeError(
            f"Descriptor-backed mmap {label} is not a directory: {path}."
        )
    if not os.path.samestat(path_stat, descriptor_stat):
        raise RuntimeError(
            f"Descriptor-backed mmap {label} does not identify its retained "
            f"directory descriptor: {path}."
        )
    return path


def _verified_fd_file_path(path: Path, fd: int, *, label: str) -> Path:
    expected_path = _FD_DIRECTORY_ROOT / str(fd)
    if path != expected_path:
        raise RuntimeError(
            f"Refusing non-descriptor-backed mmap {label} path: {path}."
        )
    try:
        path_stat = os.stat(path, follow_symlinks=True)
        descriptor_stat = os.fstat(fd)
    except OSError as exc:
        raise RuntimeError(
            "Safe CLI mmap construction requires accessible descriptor-backed "
            f"paths under {_FD_DIRECTORY_ROOT}: {label}."
        ) from exc

    if not stat.S_ISREG(path_stat.st_mode) or not stat.S_ISREG(
        descriptor_stat.st_mode
    ):
        raise RuntimeError(
            f"Descriptor-backed mmap {label} is not a file: {path}."
        )
    if not os.path.samestat(path_stat, descriptor_stat):
        raise RuntimeError(
            f"Descriptor-backed mmap {label} does not identify its retained "
            f"file descriptor: {path}."
        )
    return path


def _rebase_path(path, *, anchored_root: Path, canonical_root: Path) -> Path:
    try:
        relative_path = Path(path).relative_to(anchored_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Dataset mmap path escaped its descriptor-backed root: {path}."
        ) from exc
    if os.pardir in relative_path.parts:
        raise RuntimeError(
            f"Dataset mmap path escaped its descriptor-backed root: {path}."
        )
    return canonical_root / relative_path


def _rebase_tensordict_paths(
    tensordict,
    *,
    anchored_map: Path,
    canonical_map: Path,
) -> None:
    from tensordict.memmap import MemoryMappedTensor

    prefix_updates = []
    nested_tensordicts = [tensordict]
    nested_tensordicts.extend(
        value
        for value in tensordict.values(
            include_nested=True,
            leaves_only=False,
        )
        if hasattr(value, "_memmap_prefix")
    )
    seen = set()
    for nested_tensordict in nested_tensordicts:
        if id(nested_tensordict) in seen:
            continue
        seen.add(id(nested_tensordict))
        prefix = getattr(nested_tensordict, "_memmap_prefix", None)
        if prefix is not None:
            prefix_updates.append(
                (
                    nested_tensordict,
                    _rebase_path(
                        prefix,
                        anchored_root=anchored_map,
                        canonical_root=canonical_map,
                    ),
                )
            )

    filename_updates = []
    for value in tensordict.values(
        include_nested=True,
        leaves_only=True,
    ):
        if not isinstance(value, MemoryMappedTensor):
            continue
        filename = getattr(value, "_filename", None)
        if filename is not None:
            filename_updates.append(
                (
                    value,
                    os.fspath(
                        _rebase_path(
                            filename,
                            anchored_root=anchored_map,
                            canonical_root=canonical_map,
                        )
                    ),
                )
            )

    for nested_tensordict, prefix in prefix_updates:
        nested_tensordict._memmap_prefix = prefix
    for value, filename in filename_updates:
        value._filename = filename


def _rebase_dataset_paths(
    dataset,
    *,
    anchored_map: Path,
    anchored_prefix: Path,
    canonical_map: Path,
    anchored_ptycho_dir: Path,
    anchored_npz_file: Path,
    canonical_npz_file: Path,
) -> None:
    path_attributes = (
        "data_dir",
        "data_dir_path",
        "state_path",
        "manifest_path",
    )
    present_attributes = [
        attribute for attribute in path_attributes if hasattr(dataset, attribute)
    ]
    if not present_attributes:
        return
    if len(present_attributes) != len(path_attributes):
        raise RuntimeError(
            "Constructed mmap dataset exposes an incomplete persistent-path "
            "contract."
        )

    expected_paths = {
        "data_dir": anchored_map,
        "data_dir_path": anchored_map,
        "state_path": anchored_prefix / "state_files.npz",
        "manifest_path": anchored_prefix / "mmap_manifest.json",
    }
    for attribute, expected_path in expected_paths.items():
        if Path(getattr(dataset, attribute)) != expected_path:
            raise RuntimeError(
                f"Constructed mmap dataset changed {attribute} outside its "
                "descriptor-backed role workspace."
            )

    if not hasattr(dataset, "ptycho_dir") or not hasattr(dataset, "file_list"):
        raise RuntimeError(
            "Constructed mmap dataset exposes an incomplete source-path "
            "contract."
        )
    if Path(dataset.ptycho_dir) != anchored_ptycho_dir:
        raise RuntimeError(
            "Constructed mmap dataset changed ptycho_dir outside its "
            "descriptor-backed staging directory."
        )
    if [Path(path) for path in dataset.file_list] != [anchored_npz_file]:
        raise RuntimeError(
            "Constructed mmap dataset changed file_list outside its exact "
            "descriptor-backed NPZ."
        )
    if getattr(dataset, "n_files", None) != 1:
        raise RuntimeError(
            "Constructed CLI mmap dataset must retain exactly one source file."
        )

    mmap_ptycho = getattr(dataset, "mmap_ptycho", None)
    if mmap_ptycho is not None:
        _rebase_tensordict_paths(
            mmap_ptycho,
            anchored_map=anchored_map,
            canonical_map=canonical_map,
        )

    dataset.data_dir = os.fspath(canonical_map)
    dataset.data_dir_path = canonical_map
    dataset.state_path = canonical_map.parent / "state_files.npz"
    dataset.manifest_path = canonical_map.parent / "mmap_manifest.json"
    dataset.ptycho_dir = os.fspath(canonical_npz_file.parent)
    dataset.file_list = [canonical_npz_file]


def _create_and_open_workspace(output_fd: int) -> int:
    try:
        os.mkdir("mmap_workspace", dir_fd=output_fd)
    except FileExistsError:
        pass

    workspace_stat = _entry_lstat(output_fd, "mmap_workspace")
    if workspace_stat is None:
        raise ValueError("Managed mmap workspace disappeared during creation.")
    if stat.S_ISLNK(workspace_stat.st_mode):
        raise ValueError("Refusing symlink at managed mmap workspace.")
    if not stat.S_ISDIR(workspace_stat.st_mode):
        raise ValueError("Managed mmap workspace is not a directory.")

    workspace_fd = _open_directory("mmap_workspace", dir_fd=output_fd)
    try:
        if not os.path.samestat(workspace_stat, os.fstat(workspace_fd)):
            raise ValueError(
                "Managed mmap workspace was replaced while opening."
            )
        return workspace_fd
    except BaseException:
        _close_fd(workspace_fd)
        raise


def _remove_tree_fd(parent_fd: int, name: str, *, label: str) -> None:
    entry_stat = _entry_lstat(parent_fd, name)
    if entry_stat is None:
        return
    if stat.S_ISLNK(entry_stat.st_mode):
        raise ValueError(f"Refusing symlink at managed mmap {label}: {name}.")
    if not stat.S_ISDIR(entry_stat.st_mode):
        raise ValueError(
            f"Managed mmap {label} is not a directory: {name}."
        )
    if not _RMTREE_AVOIDS_SYMLINK_ATTACKS:
        raise RuntimeError(
            "Refusing unsafe mmap cleanup because shutil.rmtree is not "
            "symlink-resistant on this platform."
        )

    shutil.rmtree(name, dir_fd=parent_fd)
    if _entry_lstat(parent_fd, name) is not None:
        raise OSError(f"Failed to remove managed mmap {label}: {name}.")


def _copy_file_exclusive(
    source_fd: int,
    staged_name: str,
    *,
    staged_fd: int,
) -> int:
    source_stat = os.fstat(source_fd)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    destination_fd = os.open(
        staged_name,
        flags,
        stat.S_IMODE(source_stat.st_mode),
        dir_fd=staged_fd,
    )
    try:
        with os.fdopen(os.dup(source_fd), "rb") as source, os.fdopen(
            destination_fd,
            "wb",
            closefd=False,
        ) as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
        os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        os.utime(
            destination_fd,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
        return destination_fd
    except BaseException:
        _close_fd(destination_fd)
        raise


def build_cli_mmap_dataset(
    npz_file: Path,
    *,
    payload,
    output_dir: Path,
    role: str,
):
    """Build a fresh role-local mmap dataset from one selected NPZ."""
    if role not in {"train", "test"}:
        raise ValueError(f"Unsupported mmap dataset role: {role!r}")

    from ptycho_torch.dataloader import PtychoDataset

    npz_file = Path(npz_file)
    output_root = Path(output_dir).resolve(strict=False)
    workspace_root = output_root / "mmap_workspace"
    role_workspace = workspace_root / role
    staged_dir = role_workspace / "staged"
    mmap_dir = role_workspace / "mmap"
    memmap_dir = mmap_dir / "memmap"
    staged_name = npz_file.name

    _validate_managed_role(
        output_root=output_root,
        workspace_root=workspace_root,
        role_workspace=role_workspace,
        role=role,
    )

    (source_path,) = validate_cli_mmap_sources(
        (npz_file,),
        output_dir=output_dir,
    )

    _require_descriptor_safety()

    source_fd = None
    output_fd = None
    workspace_fd = None
    role_fd = None
    staged_fd = None
    staged_file_fd = None
    mmap_fd = None
    memmap_fd = None
    try:
        source_fd = _open_regular_file(source_path)
        _validate_visible_file(source_path, source_fd, label="source NPZ")
        output_root.mkdir(parents=True, exist_ok=True)
        output_fd = _open_directory(output_root)
        _validate_visible_directory(output_root, output_fd, label="output root")
        workspace_fd = _create_and_open_workspace(output_fd)

        try:
            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )
            _remove_tree_fd(workspace_fd, role, label=f"{role} role")
            _validate_visible_directory(
                output_root,
                output_fd,
                label="output root",
            )
            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )

            os.mkdir(role, dir_fd=workspace_fd)
            role_fd = _open_directory(role, dir_fd=workspace_fd)
            os.mkdir("staged", dir_fd=role_fd)
            staged_fd = _open_directory("staged", dir_fd=role_fd)
            os.mkdir("mmap", dir_fd=role_fd)
            mmap_fd = _open_directory("mmap", dir_fd=role_fd)
            os.mkdir("memmap", dir_fd=mmap_fd)
            memmap_fd = _open_directory("memmap", dir_fd=mmap_fd)
            _verified_fd_directory_path(
                _FD_DIRECTORY_ROOT / str(role_fd),
                role_fd,
                label=f"{role} role",
            )
            descriptor_staged = _verified_fd_directory_path(
                _FD_DIRECTORY_ROOT / str(staged_fd),
                staged_fd,
                label=f"{role} staging directory",
            )
            descriptor_mmap = _verified_fd_directory_path(
                _FD_DIRECTORY_ROOT / str(mmap_fd),
                mmap_fd,
                label=f"{role} mmap directory",
            )
            descriptor_memmap = _verified_fd_directory_path(
                _FD_DIRECTORY_ROOT / str(memmap_fd),
                memmap_fd,
                label=f"{role} memmap directory",
            )

            _validate_visible_directory(
                output_root,
                output_fd,
                label="output root",
            )
            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )
            _validate_visible_directory(
                role_workspace,
                role_fd,
                label=f"{role} role",
            )
            _validate_visible_directory(
                staged_dir,
                staged_fd,
                label=f"{role} staging directory",
            )
            _validate_directory_entry(
                role_fd,
                "mmap",
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_directory_entry(
                mmap_fd,
                "memmap",
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _validate_visible_directory(
                mmap_dir,
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_visible_directory(
                memmap_dir,
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _validate_visible_file(
                source_path,
                source_fd,
                label="source NPZ",
            )

            hard_linked = False
            try:
                os.link(
                    source_path,
                    staged_name,
                    dst_dir_fd=staged_fd,
                )
                hard_linked = True
            except OSError as exc:
                if exc.errno not in _HARD_LINK_COPY_ERRNOS:
                    raise
                staged_file_fd = _copy_file_exclusive(
                    source_fd,
                    staged_name,
                    staged_fd=staged_fd,
                )

            if staged_file_fd is None:
                staged_file_fd = _open_regular_file(
                    staged_name,
                    dir_fd=staged_fd,
                )
            _validate_file_entry(
                staged_fd,
                staged_name,
                staged_file_fd,
                label=f"{role} staged NPZ",
            )
            if hard_linked and not os.path.samestat(
                os.fstat(source_fd),
                os.fstat(staged_file_fd),
            ):
                raise ValueError(
                    f"Managed mmap {role} staged NPZ does not match the "
                    "retained source inode."
                )
            descriptor_staged_file = _verified_fd_file_path(
                _FD_DIRECTORY_ROOT / str(staged_file_fd),
                staged_file_fd,
                label=f"{role} staged NPZ",
            )
            _validate_visible_file(
                source_path,
                source_fd,
                label="source NPZ",
            )
            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )
            _validate_visible_directory(
                role_workspace,
                role_fd,
                label=f"{role} role",
            )
            _validate_visible_directory(
                staged_dir,
                staged_fd,
                label=f"{role} staging directory",
            )
            _validate_directory_entry(
                role_fd,
                "mmap",
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_directory_entry(
                mmap_fd,
                "memmap",
                memmap_fd,
                label=f"{role} memmap directory",
            )

            dataset = PtychoDataset(
                ptycho_dir=str(descriptor_staged),
                exact_npz_files=[descriptor_staged_file],
                model_config=payload.pt_model_config,
                data_config=payload.pt_data_config,
                training_config=payload.pt_training_config,
                data_dir=str(descriptor_memmap),
                data_prefix_dir=str(descriptor_mmap),
                remake_map=True,
                group_limit=payload.pt_training_config.n_groups,
                sequential_sampling=(
                    payload.tf_training_config.sequential_sampling
                ),
            )

            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )
            _validate_visible_directory(
                role_workspace,
                role_fd,
                label=f"{role} role",
            )
            _validate_directory_entry(
                role_fd,
                "staged",
                staged_fd,
                label=f"{role} staging directory",
            )
            _validate_file_entry(
                staged_fd,
                staged_name,
                staged_file_fd,
                label=f"{role} staged NPZ",
            )
            _validate_directory_entry(
                role_fd,
                "mmap",
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_directory_entry(
                mmap_fd,
                "memmap",
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _validate_visible_directory(
                mmap_dir,
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_visible_directory(
                memmap_dir,
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _validate_visible_file(
                source_path,
                source_fd,
                label="source NPZ",
            )
            _rebase_dataset_paths(
                dataset,
                anchored_map=descriptor_memmap,
                anchored_prefix=descriptor_mmap,
                canonical_map=memmap_dir,
                anchored_ptycho_dir=descriptor_staged,
                anchored_npz_file=descriptor_staged_file,
                canonical_npz_file=source_path,
            )
            _validate_file_entry(
                staged_fd,
                staged_name,
                staged_file_fd,
                label=f"{role} staged NPZ",
            )
            _validate_directory_entry(
                role_fd,
                "mmap",
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_directory_entry(
                mmap_fd,
                "memmap",
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _validate_visible_file(
                source_path,
                source_fd,
                label="source NPZ",
            )

            _close_fd(staged_file_fd)
            staged_file_fd = None
            _close_fd(staged_fd)
            staged_fd = None
            _close_fd(source_fd)
            source_fd = None
            _remove_tree_fd(
                role_fd,
                "staged",
                label=f"{role} staging directory",
            )
            _validate_visible_directory(
                workspace_root,
                workspace_fd,
                label="workspace",
            )
            _validate_visible_directory(
                role_workspace,
                role_fd,
                label=f"{role} role",
            )
            _validate_directory_entry(
                role_fd,
                "mmap",
                mmap_fd,
                label=f"{role} mmap directory",
            )
            _validate_directory_entry(
                mmap_fd,
                "memmap",
                memmap_fd,
                label=f"{role} memmap directory",
            )
            _close_fd(memmap_fd)
            memmap_fd = None
            _close_fd(mmap_fd)
            mmap_fd = None
            _close_fd(role_fd)
            role_fd = None
            return dataset
        except BaseException:
            _close_fd(staged_file_fd)
            staged_file_fd = None
            _close_fd(memmap_fd)
            memmap_fd = None
            _close_fd(mmap_fd)
            mmap_fd = None
            _close_fd(staged_fd)
            staged_fd = None
            _close_fd(source_fd)
            source_fd = None
            _close_fd(role_fd)
            role_fd = None
            try:
                _remove_tree_fd(workspace_fd, role, label=f"{role} role")
            except BaseException:
                pass
            raise
    finally:
        _close_fd(staged_file_fd)
        _close_fd(memmap_fd)
        _close_fd(mmap_fd)
        _close_fd(staged_fd)
        _close_fd(role_fd)
        _close_fd(workspace_fd)
        _close_fd(output_fd)
        _close_fd(source_fd)
