import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_swe_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = np.arange(3 * 4 * 5 * 6 * 2, dtype=np.float32).reshape(3, 4, 5, 6, 2)
        ds = handle.create_dataset("data", data=data)
        ds.attrs["units"] = "synthetic"
        handle.create_dataset("x-coordinate", data=np.arange(5, dtype=np.float32))


def test_inspect_hdf5_records_recursive_dataset_metadata(tmp_path):
    from scripts.studies.pdebench_swe.manifest import inspect_hdf5

    path = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(path)

    metadata = inspect_hdf5(path)

    data_record = next(item for item in metadata["datasets"] if item["path"] == "data")
    assert data_record["shape"] == [3, 4, 5, 6, 2]
    assert data_record["dtype"] == "float32"
    assert data_record["attrs"]["units"] == "synthetic"
    assert any(item["path"] == "x-coordinate" for item in metadata["datasets"])


def test_file_identity_records_size_mtime_and_sha256(tmp_path):
    from scripts.studies.pdebench_swe.manifest import file_identity

    path = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(path)

    identity = file_identity(path)

    assert identity["path"] == str(path.resolve())
    assert identity["filename"] == "2D_rdb_NA_NA.h5"
    assert identity["size_bytes"] > 0
    assert identity["mtime_ns"] > 0
    assert len(identity["sha256"]) == 64


def test_select_state_dataset_prefers_unambiguous_data_dataset(tmp_path):
    from scripts.studies.pdebench_swe.manifest import inspect_hdf5, select_state_dataset

    path = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(path)

    selected = select_state_dataset(inspect_hdf5(path))

    assert selected["path"] == "data"
    assert selected["shape"] == [3, 4, 5, 6, 2]
    assert selected["axis_order"] == "NTHWC"


def test_select_state_dataset_raises_blocker_for_ambiguous_layout(tmp_path):
    from scripts.studies.pdebench_swe.manifest import (
        ManifestBlocker,
        inspect_hdf5,
        select_state_dataset,
    )

    path = tmp_path / "ambiguous.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("state_a", data=np.zeros((2, 3, 4, 4), dtype=np.float32))
        handle.create_dataset("state_b", data=np.zeros((2, 3, 4, 4), dtype=np.float32))

    with pytest.raises(ManifestBlocker) as exc_info:
        select_state_dataset(inspect_hdf5(path))

    payload = exc_info.value.to_payload(run_id="red-test")
    assert payload["run_id"] == "red-test"
    assert payload["reason"] == "ambiguous_state_dataset"
    assert "state_a" in payload["candidate_datasets"]
    assert "state_b" in payload["candidate_datasets"]


def test_select_state_dataset_detects_repeated_trajectory_groups(tmp_path):
    from scripts.studies.pdebench_swe.manifest import inspect_hdf5, select_state_dataset

    path = tmp_path / "grouped.h5"
    with h5py.File(path, "w") as handle:
        for trajectory_id in range(3):
            group = handle.create_group(f"{trajectory_id:04d}")
            group.create_dataset("data", data=np.zeros((4, 5, 6, 1), dtype=np.float32))
            grid = group.create_group("grid")
            grid.create_dataset("t", data=np.arange(4, dtype=np.float32))

    selected = select_state_dataset(inspect_hdf5(path))

    assert selected["path"] == "*/data"
    assert selected["path_pattern"] == "{trajectory_id:04d}/data"
    assert selected["trajectory_count"] == 3
    assert selected["trajectory_shape"] == [4, 5, 6, 1]
    assert selected["shape"] == [3, 4, 5, 6, 1]
    assert selected["axis_order"] == "NTHWC"


def test_write_dataset_manifests_persists_source_identity_and_run_id(tmp_path):
    from scripts.studies.pdebench_swe.manifest import write_dataset_manifests

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    _write_swe_h5(data_file)
    output_root = tmp_path / "artifacts"

    dataset_path, metadata_path = write_dataset_manifests(
        data_file=data_file,
        output_root=output_root,
        dataset_source="PDEBench",
        dataset_source_url="https://github.com/pdebench/PDEBench",
        dataset_darus_id="133021",
        license_note="synthetic test",
        run_id="manifest-test",
    )

    dataset_manifest = json.loads(dataset_path.read_text())
    hdf5_metadata = json.loads(metadata_path.read_text())
    assert dataset_manifest["run_id"] == "manifest-test"
    assert dataset_manifest["source"]["darus_id"] == "133021"
    assert dataset_manifest["file_identity"]["filename"] == "2D_rdb_NA_NA.h5"
    assert hdf5_metadata["run_id"] == "manifest-test"
    assert hdf5_metadata["selected_state_dataset"]["path"] == "data"
