"""Tests for the offline bundle migration CLI ``ptycho_torch.migrate_bundle``."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import era_fixtures as ef


def _read_manifest(bundle_dir: Path) -> dict:
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "r") as archive:
        return json.loads(archive.read("manifest.json"))


def _member_names(bundle_dir: Path) -> set:
    with zipfile.ZipFile(bundle_dir / "wts.h5.zip", "r") as archive:
        return set(archive.namelist())


def test_migrate_metadata_free_to_current_era(tmp_path):
    """Era detection: a dill metadata-free bundle promotes to torch-artifact-v5."""
    from ptycho_torch.migrate_bundle import migrate_bundle

    out_dir = tmp_path / "migrated"
    migrate_bundle(ef.metadata_free_bundle(tmp_path), out_dir)

    manifest = _read_manifest(out_dir)
    assert manifest["artifact_schema_version"] == "torch-artifact-v5"
    names = _member_names(out_dir)
    assert "manifest.dill" not in names
    assert not any(name.endswith("params.dill") for name in names)


def test_migrate_is_idempotent(tmp_path):
    """Re-migrating a current-era bundle is a no-op that preserves the archive."""
    from ptycho_torch.migrate_bundle import migrate_bundle

    first = tmp_path / "first"
    second = tmp_path / "second"
    migrate_bundle(ef.metadata_free_bundle(tmp_path), first)
    migrate_bundle(first, second)

    assert _read_manifest(second)["artifact_schema_version"] == "torch-artifact-v5"
    assert _member_names(second) == _member_names(first)


def test_migrate_v1_json_bundle_promotes_to_v5_and_loads(tmp_path):
    """The migrator's v1 JSON branch re-seals a C1 pre-v3 identity to v5."""
    from ptycho_torch.migrate_bundle import migrate_bundle
    from ptycho_torch.workflows.bundle_io import _decode_bundle_metadata

    out_dir = tmp_path / "migrated-v1"
    migrate_bundle(ef.v1_c1_bundle(tmp_path), out_dir)

    assert _read_manifest(out_dir)["artifact_schema_version"] == "torch-artifact-v5"
    metadata = ef.read_bundle_metadata(out_dir)
    assert metadata["schema_version"] == "torch-artifact-v5"
    identity = _decode_bundle_metadata(metadata)
    assert identity.data_config.gridsize == 1
    assert identity.data_config.N == 64


def test_migrate_v2_json_bundle_promotes_to_v5_and_loads(tmp_path):
    """The migrator's v2 JSON branch re-seals a C1 pre-v3 identity to v5."""
    from ptycho_torch.migrate_bundle import migrate_bundle
    from ptycho_torch.workflows.bundle_io import _decode_bundle_metadata

    out_dir = tmp_path / "migrated-v2"
    migrate_bundle(ef.v2_c1_bundle(tmp_path), out_dir)

    assert _read_manifest(out_dir)["artifact_schema_version"] == "torch-artifact-v5"
    metadata = ef.read_bundle_metadata(out_dir)
    assert metadata["schema_version"] == "torch-artifact-v5"
    identity = _decode_bundle_metadata(metadata)
    assert identity.data_config.gridsize == 1
    assert identity.data_config.N == 64


@pytest.mark.parametrize(
    "bundle_builder",
    [ef.v1_bundle, ef.v2_bundle],
    ids=["v1", "v2"],
)
def test_migrate_rejects_multi_channel_pre_v5_identity(tmp_path, bundle_builder):
    """The frozen C=4 v1/v2 identities fail with the retrain diagnostic."""
    from ptycho_torch.migrate_bundle import migrate_bundle

    with pytest.raises(
        ValueError, match=r"supports exactly one derived channel"
    ):
        migrate_bundle(bundle_builder(tmp_path), tmp_path / "out")


def test_migrate_rejects_missing_bundle(tmp_path):
    """A directory with no wts.h5.zip is rejected with a typed error."""
    from ptycho_torch.migrate_bundle import migrate_bundle

    empty = tmp_path / "no-bundle"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Legacy bundle not found"):
        migrate_bundle(empty, tmp_path / "out")


def test_cli_round_trip(tmp_path):
    """``python -m ptycho_torch.migrate_bundle`` migrates a dill bundle offline."""
    repo_root = Path(__file__).resolve().parents[2]
    out = tmp_path / "migrated"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptycho_torch.migrate_bundle",
            str(ef.metadata_free_bundle(tmp_path)),
            str(out),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _read_manifest(out)["artifact_schema_version"] == "torch-artifact-v5"


def test_cli_rejects_non_bundle(tmp_path):
    """The CLI exits non-zero on a directory with no bundle."""
    repo_root = Path(__file__).resolve().parents[2]
    empty = tmp_path / "no-bundle"
    empty.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptycho_torch.migrate_bundle",
            str(empty),
            str(tmp_path / "out"),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode != 0
