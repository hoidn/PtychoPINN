"""Task 15 two-family CI compatibility materializer contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.studies import materialize_ci_compatibility_datasets as mat
from scripts.studies.ablation.datasets import (
    DatasetCompatibilityError,
    DatasetError,
    file_sha256,
    load_checked_dataset_bundle,
)

DETECTOR = 16
OBJECT_RESOLUTION = 64
EXPECTED_IDS = {
    "deadleaves_ci_3p5m",
    "deadleaves_legacy_amp",
    "lines_ci_3p5m",
    "lines_legacy_amp",
}


def _spec() -> "mat.MaterializationSpec":
    return mat.MaterializationSpec(
        detector_size=DETECTOR,
        object_resolution=OBJECT_RESOLUTION,
        object_seed=101,
        train_positions=16,
        test_positions=16,
        train_coordinate_seed=7,
        test_coordinate_seed=8,
        scan_jitter=1.5,
        measurement_seed=900,
        probe_source="synthetic_defocused_gaussian_16",
    )


def _probe() -> np.ndarray:
    yy, xx = np.mgrid[:DETECTOR, :DETECTOR]
    center = (DETECTOR - 1) / 2
    r2 = (xx - center) ** 2 + (yy - center) ** 2
    return np.asarray(
        np.exp(-r2 / (2 * (DETECTOR / 4) ** 2)) * np.exp(1j * (0.35 * r2 + 0.2 * xx)),
        dtype=np.complex64,
    )


def _materialize(root: Path):
    return mat.materialize_ci_compatibility_datasets(
        _spec(), raw_probe=_probe(), output_root=root
    )


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("ci_compat_v3")
    return root, _materialize(root)


def _load(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / name, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _outputs(descriptors: dict[str, dict]) -> set[str]:
    return {
        mat.PROVENANCE_FILENAME,
        mat.DESCRIPTORS_FILENAME,
        *(
            descriptor[split]
            for descriptor in descriptors.values()
            for split in ("train", "test")
        ),
    }


def _rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _dose(measurement: np.ndarray) -> dict[str, int | float]:
    photons = measurement.astype(np.float64).sum(axis=(-2, -1))
    dtype_max = int(np.iinfo(measurement.dtype).max)
    return {
        "counts_mean": float(measurement.mean(dtype=np.float64)),
        "photons_per_image_min": float(photons.min()),
        "photons_per_image_mean": float(photons.mean()),
        "max_observed_count": int(measurement.max()),
        "dtype_max": dtype_max,
        "saturation_fraction": float(
            np.count_nonzero(measurement == dtype_max) / measurement.size
        ),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(path, **arrays)


def _refresh_mutated_bundle(
    root: Path,
    descriptors: dict[str, dict],
    mutated_splits: list[tuple[str, str]],
) -> None:
    provenance_path = root / mat.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    for dataset_id, split in mutated_splits:
        descriptor = descriptors[dataset_id]
        arrays = _load(root, descriptor[split])
        split_claim = provenance["datasets"][dataset_id]["splits"][split]
        split_hash = file_sha256(root / descriptor[split])
        descriptor[f"{split}_sha256"] = split_hash
        split_claim.update(
            file_sha256=split_hash,
            truth_sha256=mat.v2_array_sha256(arrays["objectGuess"]),
            xcoords_sha256=mat.v2_array_sha256(arrays["xcoords"]),
            ycoords_sha256=mat.v2_array_sha256(arrays["ycoords"]),
            raw_probe_sha256=mat.v2_array_sha256(arrays["probeGeometry"]),
            stored_probe_sha256=mat.v2_array_sha256(arrays["probeGuess"]),
        )
        if dataset_id.endswith("ci_3p5m"):
            statistics = _dose(arrays["diff3d"])
            descriptor["dose"][split] = statistics
            split_claim["dose"] = statistics

    first_arrays = _load(root, descriptors["deadleaves_ci_3p5m"]["train"])
    provenance["probe_geometries"]["raw_probe"]["sha256"] = mat.v2_array_sha256(
        first_arrays["probeGeometry"]
    )
    provenance_path.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    provenance_sha = file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_sha


def _set_materialization_profile(
    root: Path, descriptors: dict[str, dict], profile: object
) -> None:
    provenance_path = root / mat.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text())
    provenance["materialization_profile"] = profile
    provenance_path.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    provenance_sha = file_sha256(provenance_path)
    for descriptor in descriptors.values():
        descriptor["provenance_sha256"] = provenance_sha


@pytest.mark.parametrize("with_previous", [False, True])
def test_preflight_failure_never_publishes_and_preserves_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_previous: bool
) -> None:
    root = tmp_path / "final"
    before = None
    if with_previous:
        descriptors = _materialize(root)
        before = {name: (root / name).read_bytes() for name in _outputs(descriptors)}

    def forced_failure(*_args, **_kwargs):
        raise DatasetError("forced staged preflight failure")

    monkeypatch.setattr(mat, "load_checked_dataset_bundle", forced_failure)
    with pytest.raises(mat.MaterializationError, match="forced staged preflight"):
        _materialize(root)

    if before is None:
        assert not root.exists()
    else:
        assert {name: (root / name).read_bytes() for name in before} == before


def test_atomic_publication_failure_rolls_back_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "final"
    output_root.mkdir()
    (output_root / "previous.txt").write_bytes(b"previous bundle")
    staging_root = tmp_path / "private-staging"
    staging_root.mkdir()
    (staging_root / "new.txt").write_bytes(b"validated bundle")
    real_replace = mat.os.replace

    def fail_stage_publish(source, destination):
        if Path(source) == staging_root:
            raise OSError("forced publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(mat.os, "replace", fail_stage_publish)
    with pytest.raises(OSError, match="forced publication"):
        mat._atomic_guarded_publish(
            staging_root,
            output_root,
            {"new.txt": hashlib.sha256(b"validated bundle").hexdigest()},
        )

    assert (output_root / "previous.txt").read_bytes() == b"previous bundle"
    assert not (output_root / "new.txt").exists()
    assert not (tmp_path / ".final.publish-backup").exists()


def test_rejects_lexical_symlink_output_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output_link = tmp_path / "bundle-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(mat.MaterializationError, match="symlink"):
        _materialize(output_link)

    assert list(target.iterdir()) == []


def test_startup_recovers_backup_when_public_root_is_absent(
    tmp_path: Path, materialized
) -> None:
    source, descriptors = materialized
    output_root = tmp_path / "bundle"
    backup_root = mat._publication_backup_path(output_root)
    shutil.copytree(source, backup_root)
    expected = {
        name: (backup_root / name).read_bytes() for name in _outputs(descriptors)
    }

    mat._recover_publication(output_root)

    assert not backup_root.exists()
    assert {name: (output_root / name).read_bytes() for name in expected} == expected


def test_startup_keeps_valid_public_bundle_and_removes_stale_backup(
    tmp_path: Path, materialized
) -> None:
    source, descriptors = materialized
    output_root = tmp_path / "bundle"
    backup_root = mat._publication_backup_path(output_root)
    shutil.copytree(source, output_root)
    shutil.copytree(source, backup_root)
    expected = {
        name: (output_root / name).read_bytes() for name in _outputs(descriptors)
    }

    mat._recover_publication(output_root)

    assert not backup_root.exists()
    assert {name: (output_root / name).read_bytes() for name in expected} == expected


def test_startup_restores_backup_when_both_present_public_is_invalid(
    tmp_path: Path, materialized
) -> None:
    source, descriptors = materialized
    output_root = tmp_path / "bundle"
    backup_root = mat._publication_backup_path(output_root)
    shutil.copytree(source, output_root)
    shutil.copytree(source, backup_root)
    expected = {
        name: (backup_root / name).read_bytes() for name in _outputs(descriptors)
    }
    victim = output_root / descriptors["lines_ci_3p5m"]["test"]
    victim.write_bytes(b"interrupted publication")

    mat._recover_publication(output_root)

    assert not backup_root.exists()
    assert {name: (output_root / name).read_bytes() for name in expected} == expected


def test_materializes_exactly_four_loadable_descriptors(materialized) -> None:
    root, descriptors = materialized

    assert set(descriptors) == EXPECTED_IDS
    bundle = load_checked_dataset_bundle(descriptors, repo_root=root)
    assert set(bundle) == EXPECTED_IDS
    assert bundle.materialization_profile == "fixture"
    assert all(
        item.bundle.materialization_profile == "fixture" for item in bundle.values()
    )
    object.__setattr__(bundle, "materialization_profile", "claim_grade")
    try:
        with pytest.raises(DatasetCompatibilityError, match="seal"):
            bundle._assert_sealed()
    finally:
        object.__setattr__(bundle, "materialization_profile", "fixture")
    bundle._assert_sealed()
    provenance = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    assert provenance["materialization_profile"] == "fixture"
    for dataset_id in EXPECTED_IDS:
        descriptor = descriptors[dataset_id]
        assert descriptor["truth"] == "object_truth"
        assert descriptor["probe"]["source"] == _spec().probe_source
        if dataset_id.endswith("ci_3p5m"):
            assert descriptor["scale_contract_version"] == "ci_intensity_v2"
            assert descriptor["measurement_domain"] == "count_intensity"
            assert descriptor["probe"]["calibration"] == "count_amplitude"
            assert "dose" in descriptor
        else:
            assert descriptor["scale_contract_version"] == "legacy_v1"
            assert descriptor["measurement_domain"] == "normalized_amplitude"
            assert descriptor["probe"]["calibration"] == "legacy_normalized"
            assert "dose" not in descriptor


def test_family_truth_and_coordinates_are_twins_but_morphologies_are_distinct(
    materialized,
) -> None:
    root, descriptors = materialized
    arrays = {
        (dataset_id, split): _load(root, descriptor[split])
        for dataset_id, descriptor in descriptors.items()
        for split in ("train", "test")
    }

    for family in ("deadleaves", "lines"):
        ci_id = f"{family}_ci_3p5m"
        legacy_id = f"{family}_legacy_amp"
        for split in ("train", "test"):
            ci = arrays[(ci_id, split)]
            legacy = arrays[(legacy_id, split)]
            np.testing.assert_array_equal(ci["objectGuess"], legacy["objectGuess"])
            np.testing.assert_array_equal(ci["xcoords"], legacy["xcoords"])
            np.testing.assert_array_equal(ci["ycoords"], legacy["ycoords"])
            np.testing.assert_array_equal(ci["probeGeometry"], legacy["probeGeometry"])

    dead = arrays[("deadleaves_ci_3p5m", "train")]
    lines = arrays[("lines_ci_3p5m", "train")]
    assert mat.v2_array_sha256(dead["objectGuess"]) != mat.v2_array_sha256(
        lines["objectGuess"]
    )
    np.testing.assert_array_equal(dead["xcoords"], lines["xcoords"])
    np.testing.assert_array_equal(dead["ycoords"], lines["ycoords"])
    np.testing.assert_array_equal(dead["probeGeometry"], lines["probeGeometry"])


def test_lines_object_uses_canonical_morphology_with_bounded_rectangular_mapping() -> (
    None
):
    spec = _spec()
    crop_start = spec.object_resolution // 2
    raw = mat.diffsim.mk_lines_img(
        N=2 * spec.object_resolution,
        nlines=400,
        rng=np.random.RandomState(spec.object_seed),
    )[
        crop_start : crop_start + spec.object_resolution,
        crop_start : crop_start + spec.object_resolution,
        0,
    ]
    normalized = (raw - raw.min()) / (raw.max() - raw.min())
    amplitude = 0.3 + 0.7 * normalized
    phase = 0.5 * (2.0 * normalized - 1.0)
    expected = np.ascontiguousarray(amplitude * np.exp(1j * phase), dtype=np.complex64)

    actual = mat._lines_object(spec)

    assert actual.dtype == np.complex64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert float(np.abs(actual).min()) == pytest.approx(0.3, abs=1e-7)
    assert float(np.abs(actual).max()) == pytest.approx(1.0, abs=1e-7)
    assert float(np.angle(actual).min()) == pytest.approx(-0.5, abs=1e-7)
    assert float(np.angle(actual).max()) == pytest.approx(0.5, abs=1e-7)


@pytest.mark.parametrize("bad_value", [np.nan, 2.0])
def test_lines_object_rejects_nonfinite_or_constant_morphology(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    def bad_morphology(*, N: int, nlines: int, rng: object) -> np.ndarray:
        assert N == 2 * _spec().object_resolution
        assert nlines == 400
        assert isinstance(rng, np.random.RandomState)
        return np.full((N, N, 3), bad_value, dtype=np.float64)

    monkeypatch.setattr(mat.diffsim, "mk_lines_img", bad_morphology)

    with pytest.raises(mat.MaterializationError, match="morphology.*finite|constant"):
        mat._lines_object(_spec())


def test_lines_object_normalizes_extreme_finite_morphology_without_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extreme: float = float(np.finfo(np.float64).max)

    def extreme_morphology(*, N: int, nlines: int, rng: object) -> np.ndarray:
        assert nlines == 400
        assert isinstance(rng, np.random.RandomState)
        morphology = np.full((N, N, 3), -extreme, dtype=np.float64)
        morphology[:, N // 2 :, :] = extreme
        return morphology

    monkeypatch.setattr(mat.diffsim, "mk_lines_img", extreme_morphology)

    actual = mat._lines_object(_spec())

    assert np.isfinite(actual.real).all()
    assert np.isfinite(actual.imag).all()
    assert float(np.abs(actual).min()) == pytest.approx(0.3, abs=1e-7)
    assert float(np.abs(actual).max()) == pytest.approx(1.0, abs=1e-7)
    assert float(np.angle(actual).min()) == pytest.approx(-0.5, abs=1e-7)
    assert float(np.angle(actual).max()) == pytest.approx(0.5, abs=1e-7)


def test_lines_generation_is_deterministic_and_preserves_ambient_rng() -> None:
    before = np.random.get_state()

    first = mat._lines_object(_spec())
    after = np.random.get_state()
    second = mat._lines_object(_spec())

    assert _rng_equal(before, after)
    np.testing.assert_array_equal(first, second)


def test_deadleaves_object_has_no_task30_drift() -> None:
    actual = mat._deadleaves_object(_spec())

    assert mat.v2_array_sha256(actual, np.complex64) == (
        "76cd1596ccdf618829bba5b95550e814241629befff4a8f46b15cb69ceb18288"
    )
    np.testing.assert_array_equal(
        actual[[0, 17, -1], [0, 29, -1]],
        np.array(
            [
                0.5749765 - 0.18715148j,
                0.93184793 + 0.30911058j,
                0.96534085 - 0.52736807j,
            ],
            dtype=np.complex64,
        ),
    )


def test_generation_is_local_rng_deterministic_and_preserves_ambient_state(
    tmp_path: Path,
) -> None:
    before = np.random.get_state()
    first = _materialize(tmp_path / "first")
    after = np.random.get_state()
    second = _materialize(tmp_path / "second")

    assert _rng_equal(before, after)
    assert first == second
    for name in _outputs(first):
        assert file_sha256(tmp_path / "first" / name) == file_sha256(
            tmp_path / "second" / name
        )


def test_materialization_streams_one_batched_split_without_blob_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_writer = mat._write_dataset_split
    real_extract = mat.M.extract_object_patches
    active = 0
    max_active = 0
    writes = []
    batch_sizes = []

    def instrumented_writer(*args, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            artifact = real_writer(*args, **kwargs)
            writes.append(artifact)
            assert not isinstance(artifact, (bytes, bytearray, np.ndarray, dict))
            return artifact
        finally:
            active -= 1

    def instrumented_extract(obj, xcoords, ycoords, detector_size):
        batch_sizes.append(len(xcoords))
        return real_extract(obj, xcoords, ycoords, detector_size)

    monkeypatch.setattr(mat, "GENERATION_BATCH_SIZE", 4)
    monkeypatch.setattr(mat, "_write_dataset_split", instrumented_writer)
    monkeypatch.setattr(mat.M, "extract_object_patches", instrumented_extract)

    _materialize(tmp_path / "streamed")

    assert len(writes) == 8
    assert max_active == 1
    assert batch_sizes and max(batch_sizes) <= 4
    assert not hasattr(mat, "_generate_measurements")


def test_ci_families_are_independently_calibrated_with_floor_and_no_saturation(
    materialized,
) -> None:
    root, descriptors = materialized
    scales = {}
    for family in ("deadleaves", "lines"):
        dataset_id = f"{family}_ci_3p5m"
        for split in ("train", "test"):
            arrays = _load(root, descriptors[dataset_id][split])
            counts = arrays["diff3d"]
            photons = counts.astype(np.float64).sum(axis=(-2, -1))
            metadata = json.loads(arrays["_metadata"].item())
            scales[(family, split)] = metadata["probe_calibration"][
                "dose_amplitude_scale"
            ]
            assert float(photons.mean()) == pytest.approx(3_538_944, rel=0.03)
            assert float(photons.min()) >= 1_000_000
            assert int(counts.max()) < np.iinfo(counts.dtype).max
            assert np.count_nonzero(counts == np.iinfo(counts.dtype).max) == 0
    assert scales[("deadleaves", "train")] != scales[("lines", "train")]


def test_v3_provenance_is_closed_and_hashes_every_split_array(materialized) -> None:
    from scripts.studies.ablation.dataset_provenance import parse_provenance_v3

    root, descriptors = materialized
    payload = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    parsed = parse_provenance_v3(payload)

    assert payload["schema_version"] == "ci_compatibility_provenance_v3"
    assert payload["materializer_version"] == 3
    assert payload["expected_dataset_ids"] == sorted(EXPECTED_IDS)
    assert set(payload["source_objects"]) == {"deadleaves", "lines"}
    assert payload["source_objects"]["lines"]["generator"] == (
        "grid_lines_rectangular_v1"
    )
    assert payload["source_objects"]["lines"]["parameters"] == {
        "amplitude_max": 1.0,
        "amplitude_min": 0.3,
        "canvas_size": 128,
        "crop_start": 32,
        "crop_stop": 96,
        "mapping": "rectangular_v1",
        "nlines": 400,
        "object_resolution": 64,
        "phase_max": 0.5,
        "phase_min": -0.5,
        "seed": 101,
    }
    assert parsed.expected_dataset_ids == tuple(sorted(EXPECTED_IDS))
    for dataset_id, record in payload["datasets"].items():
        for split in ("train", "test"):
            arrays = _load(root, descriptors[dataset_id][split])
            claim = record["splits"][split]
            assert claim["file_sha256"] == file_sha256(
                root / descriptors[dataset_id][split]
            )
            assert claim["truth_sha256"] == mat.v2_array_sha256(arrays["objectGuess"])
            assert claim["xcoords_sha256"] == mat.v2_array_sha256(arrays["xcoords"])
            assert claim["ycoords_sha256"] == mat.v2_array_sha256(arrays["ycoords"])
            assert claim["raw_probe_sha256"] == mat.v2_array_sha256(
                arrays["probeGeometry"]
            )
            assert claim["stored_probe_sha256"] == mat.v2_array_sha256(
                arrays["probeGuess"]
            )


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("seeds",),
        ("source_objects", "lines"),
        ("source_objects", "lines", "parameters"),
        ("coordinate_sets", "shared_scan", "train"),
        ("probe_geometries", "raw_probe"),
        ("datasets", "lines_ci_3p5m"),
        ("datasets", "lines_ci_3p5m", "splits"),
        ("datasets", "lines_ci_3p5m", "splits", "train"),
        ("datasets", "lines_ci_3p5m", "splits", "train", "dose"),
    ],
)
def test_v3_provenance_rejects_unknown_fields_at_every_level(
    materialized, path
) -> None:
    from scripts.studies.ablation.dataset_provenance import parse_provenance_v3

    root, _ = materialized
    payload = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    target = payload
    for key in path:
        target = target[key]
    target["unknown"] = True

    with pytest.raises(DatasetError, match="unknown field"):
        parse_provenance_v3(payload)


@pytest.mark.parametrize(
    ("path", "mode", "value"),
    [
        (("datasets", "lines_ci_3p5m", "splits", "train", "path"), "delete", None),
        (("datasets", "lines_ci_3p5m", "splits", "train", "path"), "set", None),
        (("coordinate_sets", "shared_scan", "train", "count"), "delete", None),
        (("coordinate_sets", "shared_scan", "train", "count"), "set", "16"),
        (("probe_geometries", "raw_probe", "shape"), "set", [1, "16", 16]),
        (
            (
                "datasets",
                "lines_ci_3p5m",
                "splits",
                "train",
                "dose",
                "saturation_fraction",
            ),
            "set",
            "zero",
        ),
    ],
)
def test_v3_provenance_rejects_nested_missing_and_wrong_types(
    materialized, path, mode, value
) -> None:
    from scripts.studies.ablation.dataset_provenance import parse_provenance_v3

    root, _ = materialized
    payload = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    if mode == "delete":
        del parent[path[-1]]
    else:
        parent[path[-1]] = value

    with pytest.raises(DatasetError, match="required|must be|positive|number"):
        parse_provenance_v3(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source_objects", "lines", "generator"), "grid_lines_set_phi"),
        (("source_objects", "lines", "parameters", "mapping"), "other"),
        (("source_objects", "lines", "parameters", "amplitude_min"), 0.31),
        (("source_objects", "lines", "parameters", "amplitude_max"), 0.99),
        (("source_objects", "lines", "parameters", "phase_min"), -0.49),
        (("source_objects", "lines", "parameters", "phase_max"), 0.49),
    ],
)
def test_v3_provenance_rejects_inexact_bounded_lines_declaration(
    materialized, path: tuple[str, ...], value: object
) -> None:
    from scripts.studies.ablation.dataset_provenance import parse_provenance_v3

    root, _ = materialized
    payload = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(DatasetError, match="generator|rectangular|parameters"):
        parse_provenance_v3(payload)


def test_genuine_v2_historical_bundle_remains_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation.dataset_content import ArtifactCache
    from scripts.studies.ablation.dataset_provenance import (
        CI_COMPATIBILITY_PROVENANCE_V2,
        ProvenanceV2,
        load_provenance,
    )

    real_build_provenance = mat._build_provenance

    def historical_lines_object(spec: mat.MaterializationSpec) -> np.ndarray:
        crop_start = spec.object_resolution // 2
        crop_stop = crop_start + spec.object_resolution
        amplitude = mat.diffsim.mk_lines_img(
            N=2 * spec.object_resolution,
            nlines=400,
            rng=np.random.RandomState(spec.object_seed),
        )[crop_start:crop_stop, crop_start:crop_stop, 0]
        phase = np.asarray(mat.diffsim.dummy_phi(amplitude), dtype=np.float32)
        return np.ascontiguousarray(amplitude * np.exp(1j * phase), dtype=np.complex64)

    def historical_v2_provenance(*args, **kwargs) -> dict:
        payload = real_build_provenance(*args, **kwargs)
        spec = args[0]
        crop_start = spec.object_resolution // 2
        payload.update(
            schema_version=CI_COMPATIBILITY_PROVENANCE_V2,
            materializer_id="ci_compatibility_twins_v2",
            materializer_version=2,
        )
        payload["source_objects"]["lines"].update(
            generator="grid_lines_set_phi",
            parameters={
                "canvas_size": 2 * spec.object_resolution,
                "object_resolution": spec.object_resolution,
                "crop_start": crop_start,
                "crop_stop": crop_start + spec.object_resolution,
                "nlines": 400,
                "set_phi": True,
                "seed": spec.object_seed,
            },
        )
        return payload

    root = tmp_path / "historical_v2"
    with monkeypatch.context() as context:
        context.setattr(mat, "_lines_object", historical_lines_object)
        context.setattr(mat, "_build_provenance", historical_v2_provenance)
        descriptors = _materialize(root)

    provenance_path = root / mat.PROVENANCE_FILENAME
    provenance = load_provenance(
        provenance_path,
        file_sha256(provenance_path),
        ArtifactCache(),
    )
    bundle = load_checked_dataset_bundle(
        copy.deepcopy(descriptors),
        repo_root=root,
    )
    lines = _load(root, descriptors["lines_ci_3p5m"]["train"])["objectGuess"]

    assert isinstance(provenance, ProvenanceV2)
    assert provenance.schema_version == CI_COMPATIBILITY_PROVENANCE_V2
    assert set(bundle) == EXPECTED_IDS
    assert float(np.abs(lines).max()) > 1.0


def test_preflight_recomputes_relationship_hashes_before_acceptance(
    tmp_path: Path, materialized
) -> None:
    source, descriptors = materialized
    root = tmp_path / "corrupt"
    shutil.copytree(source, root)
    victim = descriptors["lines_legacy_amp"]["test"]
    arrays = _load(root, victim)
    arrays["xcoords"] = arrays["xcoords"].copy()
    arrays["xcoords"][0] += 1
    np.savez(root / victim, **arrays)

    altered = copy.deepcopy(descriptors)
    altered["lines_legacy_amp"]["test_sha256"] = file_sha256(root / victim)
    with pytest.raises(DatasetError, match="provenance|coordinate|sha256"):
        load_checked_dataset_bundle(altered, repo_root=root)


def test_preflight_accepts_complete_bundle_relocation(materialized, tmp_path: Path):
    source, descriptors = materialized
    relocated = tmp_path / "elsewhere" / "bundle"
    shutil.copytree(source, relocated)

    bundle = load_checked_dataset_bundle(
        copy.deepcopy(descriptors), repo_root=relocated
    )

    assert set(bundle) == EXPECTED_IDS


def test_preflight_rejects_in_bundle_file_relocation_with_stale_provenance(
    tmp_path: Path, materialized
) -> None:
    source, source_descriptors = materialized
    root = tmp_path / "relocated"
    shutil.copytree(source, root)
    nested = root / "datasets" / "v2"
    nested.mkdir(parents=True)
    descriptors = copy.deepcopy(source_descriptors)
    descriptor = descriptors["lines_ci_3p5m"]
    original = descriptor["train"]
    shutil.move(root / original, nested / original)
    descriptor["train"] = f"datasets/v2/{original}"

    with pytest.raises(
        DatasetError, match="split path|repository.*relative|relocation"
    ):
        load_checked_dataset_bundle(descriptors, repo_root=root)


def test_fixture_bundle_cannot_claim_claim_grade(tmp_path: Path, materialized) -> None:
    source, source_descriptors = materialized
    root = tmp_path / "false_claim_grade"
    shutil.copytree(source, root)
    descriptors = copy.deepcopy(source_descriptors)
    _set_materialization_profile(root, descriptors, "claim_grade")

    with pytest.raises(DatasetError, match="claim_grade.*64|5000|320"):
        load_checked_dataset_bundle(descriptors, repo_root=root)


@pytest.mark.parametrize("mutation", ["missing", None, True, "development"])
def test_materialization_profile_is_required_closed_enum(materialized, mutation):
    from scripts.studies.ablation.dataset_provenance import parse_provenance_v3

    root, _ = materialized
    payload = json.loads((root / mat.PROVENANCE_FILENAME).read_text())
    if mutation == "missing":
        del payload["materialization_profile"]
    else:
        payload["materialization_profile"] = mutation

    with pytest.raises(DatasetError, match="materialization_profile|required|enum"):
        parse_provenance_v3(payload)


def test_preflight_rejects_self_consistent_complex128_raw_probe(
    tmp_path: Path, materialized
) -> None:
    source, source_descriptors = materialized
    root = tmp_path / "complex128_raw_probe"
    shutil.copytree(source, root)
    descriptors = copy.deepcopy(source_descriptors)
    mutated = []
    for dataset_id, descriptor in descriptors.items():
        for split in ("train", "test"):
            arrays = _load(root, descriptor[split])
            arrays["probeGeometry"] = arrays["probeGeometry"].astype(np.complex128)
            _write_npz(root / descriptor[split], arrays)
            mutated.append((dataset_id, split))
    _refresh_mutated_bundle(root, descriptors, mutated)

    with pytest.raises(DatasetError, match="probeGeometry.*complex64|dtype"):
        load_checked_dataset_bundle(descriptors, repo_root=root)


@pytest.mark.parametrize(
    ("case", "replacement", "match"),
    [
        (
            "zero_counts",
            lambda counts: np.zeros_like(counts),
            "mean photons|weakest|positive",
        ),
        (
            "wrong_dose",
            lambda counts: np.full_like(counts, 8_000),
            "mean photons.*target",
        ),
        (
            "saturated",
            lambda counts: np.full_like(counts, np.iinfo(counts.dtype).max),
            "saturat|max count",
        ),
    ],
)
def test_preflight_rejects_self_consistent_invalid_ci_physics(
    tmp_path: Path, materialized, case, replacement, match
) -> None:
    source, source_descriptors = materialized
    root = tmp_path / case
    shutil.copytree(source, root)
    descriptors = copy.deepcopy(source_descriptors)
    dataset_id, split = "lines_ci_3p5m", "train"
    arrays = _load(root, descriptors[dataset_id][split])
    arrays["diff3d"] = replacement(arrays["diff3d"])
    _write_npz(root / descriptors[dataset_id][split], arrays)
    _refresh_mutated_bundle(root, descriptors, [(dataset_id, split)])

    with pytest.raises(DatasetError, match=match):
        load_checked_dataset_bundle(descriptors, repo_root=root)


def test_refuses_to_overwrite_mismatched_output(tmp_path: Path, materialized) -> None:
    source, descriptors = materialized
    root = tmp_path / "guarded"
    shutil.copytree(source, root)
    victim = root / descriptors["lines_ci_3p5m"]["test"]
    victim.write_bytes(b"different")

    with pytest.raises(mat.MaterializationError, match="refus"):
        _materialize(root)
    assert (
        hashlib.sha256(victim.read_bytes()).hexdigest()
        == hashlib.sha256(b"different").hexdigest()
    )


def test_cli_defaults_are_claim_grade() -> None:
    args = mat._build_parser().parse_args(["--output-root", "unused"])
    assert args.detector_size == 64
    assert args.object_resolution == 320
    assert args.train_positions == 5000
    assert args.test_positions == 1250
    claim_spec = mat.MaterializationSpec(
        detector_size=args.detector_size,
        object_resolution=args.object_resolution,
        object_seed=args.object_seed,
        train_positions=args.train_positions,
        test_positions=args.test_positions,
        train_coordinate_seed=args.train_coordinate_seed,
        test_coordinate_seed=args.test_coordinate_seed,
        scan_jitter=args.scan_jitter,
        measurement_seed=args.measurement_seed,
        probe_source=str(args.probe_src),
    )
    assert mat._materialization_profile(claim_spec) == "claim_grade"
    assert mat._materialization_profile(_spec()) == "fixture"
