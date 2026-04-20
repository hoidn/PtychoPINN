from pathlib import Path

import pytest


def test_file_identity_records_size_mtime_and_sha256(tmp_path):
    from scripts.studies.openfwi_flatvel_a.manifest import file_identity

    path = tmp_path / "data1.npy"
    path.write_bytes(b"abc")

    payload = file_identity(path)

    assert payload["filename"] == "data1.npy"
    assert payload["size_bytes"] == 3
    assert payload["sha256"]


def test_required_shards_missing_writes_blocker(tmp_path):
    from scripts.studies.openfwi_flatvel_a.manifest import (
        OpenFWIManifestBlocker,
        resolve_required_shards,
    )

    with pytest.raises(OpenFWIManifestBlocker) as exc:
        resolve_required_shards(tmp_path)

    assert exc.value.reason == "missing_required_shards"
    assert "data1.npy" in exc.value.missing


def test_data_root_inside_repo_is_rejected_unless_ignored(tmp_path):
    from scripts.studies.openfwi_flatvel_a.manifest import (
        OpenFWIManifestBlocker,
        validate_data_root_policy,
    )

    repo_root = tmp_path / "repo"
    data_root = repo_root / "data" / "openfwi"
    data_root.mkdir(parents=True)

    with pytest.raises(OpenFWIManifestBlocker) as exc:
        validate_data_root_policy(data_root, repo_root=repo_root)

    assert exc.value.reason == "data_root_inside_repo"


def test_build_data_manifest_records_required_shard_roles(tmp_path):
    from scripts.studies.openfwi_flatvel_a.manifest import (
        REQUIRED_SHARDS,
        build_data_manifest,
        resolve_required_shards,
    )

    for name in REQUIRED_SHARDS:
        (tmp_path / name).write_bytes(name.encode("ascii"))

    shards = resolve_required_shards(tmp_path)
    payload = build_data_manifest(
        data_root=tmp_path,
        shards=shards,
        source_url="https://openfwi-lanl.github.io/docs/data.html",
        license_note="CC BY-NC-SA 4.0",
        access_note="synthetic shards for test",
        run_id="manifest-test",
    )

    assert payload["run_id"] == "manifest-test"
    assert payload["dataset_variant"] == "FlatVel-A"
    assert {item["filename"] for item in payload["shards"]} == set(REQUIRED_SHARDS)
    assert {item["split_role"] for item in payload["shards"]} == {"train", "validation_test"}
