"""Deterministic production flat-acquisition generation contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ptycho.config.config import (
    DataConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
)
from ptycho.workflows.synthetic_config import resolve_synthetic_workflow


def _small_request(seed: int) -> dict[str, object]:
    return {
        "profile": "synthetic-lines",
        "file_values": {
            "simulation": {
                "N": 64,
                "seed": seed,
                "train_patterns": 2,
                "test_patterns": 2,
            },
            "training": {
                "train_raw_selection": 2,
                "training_groups": 2,
                "validation_groups": 2,
                "neighbor_count": 1,
                "neighbor_pool_size": 1,
            },
        },
    }


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _run_worker(tmp_path: Path, *, name: str, seed: int) -> Path:
    request_path = tmp_path / f"{name}.json"
    request_path.write_text(
        json.dumps(_small_request(seed), sort_keys=True), encoding="utf-8"
    )
    output_root = tmp_path / name
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "7"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.simulation.synthetic_simulation_worker",
            "--request-json",
            str(request_path),
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output_root / "datasets"


def test_seed_lineage_uses_seven_named_seed_sequence_children_in_fixed_order():
    from ptycho.simulation.flat_acquisition import derive_seed_lineage

    base_seed = 173
    expected_children = np.random.SeedSequence(base_seed).spawn(7)
    expected = {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(
            (
                "object",
                "train_coordinates",
                "train_noise",
                "test_coordinates",
                "test_noise",
                "grouping",
                "torch",
            ),
            expected_children,
            strict=True,
        )
    }

    lineage = derive_seed_lineage(base_seed)

    assert lineage["base_seed"] == base_seed
    assert {name: lineage[name] for name in expected} == expected


def test_nongrid_public_adapter_forwards_the_callers_seed(monkeypatch):
    from ptycho import nongrid_simulation

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_legacy(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        nongrid_simulation, "_generate_simulated_data_legacy_params", fake_legacy
    )
    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        sampling=SamplingConfig(n_groups=1),
    )

    result = nongrid_simulation.generate_simulated_data(
        config,
        np.ones((16, 16), dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
        4,
        return_patches=False,
        random_seed=982451653,
    )

    assert result is sentinel
    assert captured["random_seed"] == 982451653


def test_nongrid_rejects_ambiguous_coordinate_seed_sources():
    from ptycho import nongrid_simulation

    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        sampling=SamplingConfig(n_groups=1),
    )

    with pytest.raises(ValueError, match="random_seed.*coordinate_rng"):
        nongrid_simulation._generate_simulated_data_legacy_params(
            config,
            np.ones((16, 16), dtype=np.complex64),
            np.ones((8, 8), dtype=np.complex64),
            4,
            random_seed=3,
            coordinate_rng=np.random.default_rng(4),
        )


def test_lines_object_uses_the_locked_producers_crop_and_complex64(monkeypatch):
    from ptycho import diffsim
    from ptycho.simulation.flat_acquisition import build_lines_object

    rng = np.random.default_rng(29)
    calls: dict[str, object] = {}
    morphology = np.arange(784 * 784 * 3, dtype=np.float32).reshape(784, 784, 3)

    def fake_lines(size, *, nlines, rng):
        calls.update(size=size, nlines=nlines, rng=rng)
        return morphology

    def fake_phi(amplitude):
        calls["phase_input"] = np.array(amplitude, copy=True)
        return np.full_like(amplitude, np.pi / 4, dtype=np.float32)

    monkeypatch.setattr(diffsim, "mk_lines_img", fake_lines)
    monkeypatch.setattr(diffsim, "dummy_phi", fake_phi)

    ambient_before = np.random.get_state()
    result = build_lines_object(rng)
    ambient_after = np.random.get_state()

    expected_amplitude = morphology[196:-196, 196:-196, 0]
    expected = expected_amplitude * np.exp(1j * np.pi / 4)
    assert calls["size"] == 784
    assert calls["nlines"] == 400
    assert calls["rng"] is rng
    assert np.array_equal(calls["phase_input"], expected_amplitude)
    assert result.recipe == "lines-object-v1"
    assert result.array.dtype == np.complex64
    assert result.array.shape == (392, 392)
    assert np.allclose(result.array, expected.astype(np.complex64))
    assert result.producer_symbols == (
        "ptycho.diffsim.mk_lines_img",
        "ptycho.diffsim.dummy_phi",
    )
    assert ambient_before[0] == ambient_after[0]
    assert np.array_equal(ambient_before[1], ambient_after[1])
    assert ambient_before[2:] == ambient_after[2:]


def test_canonical_flat_schema_casts_exactly_and_rejects_pregrouped_diffraction():
    from ptycho.simulation.flat_acquisition import canonicalize_flat_acquisition

    raw = SimpleNamespace(
        diff3d=np.ones((3, 8, 8), dtype=np.float64),
        xcoords=np.arange(3, dtype=np.float32),
        ycoords=np.arange(3, dtype=np.float32) + 1,
        xcoords_start=np.arange(3, dtype=np.float32) + 2,
        ycoords_start=np.arange(3, dtype=np.float32) + 3,
        probeGuess=np.ones((8, 8), dtype=np.complex128),
        objectGuess=None,
        scan_index=np.arange(3, dtype=np.int32),
    )
    truth = np.ones((32, 32), dtype=np.complex128)

    payload = canonicalize_flat_acquisition(raw, object_guess=truth)

    assert {name: value.dtype for name, value in payload.items()} == {
        "diff3d": np.dtype("float32"),
        "xcoords": np.dtype("float64"),
        "ycoords": np.dtype("float64"),
        "xcoords_start": np.dtype("float64"),
        "ycoords_start": np.dtype("float64"),
        "probeGuess": np.dtype("complex64"),
        "objectGuess": np.dtype("complex64"),
        "scan_index": np.dtype("int64"),
    }
    assert payload["diff3d"].shape == (3, 8, 8)
    assert payload["probeGuess"].shape == (8, 8)
    assert payload["objectGuess"].shape == (32, 32)

    raw.diff3d = np.ones((3, 8, 8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="rank-4 pre-grouped diffraction"):
        canonicalize_flat_acquisition(raw, object_guess=truth)


def test_singleton_extraction_scope_does_not_rewrite_logical_gs2(monkeypatch):
    from ptycho import nongrid_simulation, params

    logical = TrainingConfig(
        model=ModelConfig(N=8, gridsize=2, object_big=True),
        data=DataConfig(nphotons=1e5),
        sampling=SamplingConfig(n_groups=2),
    )
    ambient = {"N": 17, "gridsize": 9, "marker": "ambient"}
    monkeypatch.setattr(params, "cfg", dict(ambient))

    def fake_from_simulation(xcoords, ycoords, probe, obj, scan_index):
        assert params.cfg["N"] == 8
        assert params.cfg["gridsize"] == 1
        assert params.cfg["nphotons"] == 1e5
        return SimpleNamespace(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=np.ones((2, 8, 8), dtype=np.float32),
            probeGuess=probe,
            scan_index=scan_index,
            Y=np.ones((2, 8, 8), dtype=np.complex64),
        )

    monkeypatch.setattr(
        nongrid_simulation.RawData, "from_simulation", fake_from_simulation
    )
    raw = nongrid_simulation._generate_simulated_data_legacy_params(
        logical,
        np.ones((32, 32), dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
        4,
        coordinate_rng=np.random.default_rng(3),
        detector_seed=5,
    )

    assert raw.diff3d.shape == (2, 8, 8)
    assert logical.model.gridsize == 2
    assert params.cfg == ambient


def test_detector_seed_is_independent_of_identical_coordinate_stream(monkeypatch):
    import tensorflow as tf

    from ptycho import nongrid_simulation

    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        data=DataConfig(nphotons=1e5),
        sampling=SamplingConfig(n_groups=2),
    )
    observed_coordinates: list[tuple[np.ndarray, np.ndarray]] = []
    numpy_seeds: list[int] = []
    tensorflow_seeds: list[int] = []

    monkeypatch.setattr(np.random, "seed", lambda seed: numpy_seeds.append(seed))
    monkeypatch.setattr(
        tf.random, "set_seed", lambda seed: tensorflow_seeds.append(seed)
    )

    def fake_from_simulation(xcoords, ycoords, probe, obj, scan_index):
        observed_coordinates.append((xcoords.copy(), ycoords.copy()))
        return SimpleNamespace(diff3d=np.ones((2, 8, 8), dtype=np.float32))

    monkeypatch.setattr(
        nongrid_simulation.RawData, "from_simulation", fake_from_simulation
    )
    for detector_seed in (101, 202):
        nongrid_simulation._generate_simulated_data_legacy_params(
            config,
            np.ones((32, 32), dtype=np.complex64),
            np.ones((8, 8), dtype=np.complex64),
            4,
            coordinate_rng=np.random.default_rng(77),
            detector_seed=detector_seed,
        )

    assert np.array_equal(
        observed_coordinates[0][0], observed_coordinates[1][0]
    )
    assert np.array_equal(
        observed_coordinates[0][1], observed_coordinates[1][1]
    )
    assert numpy_seeds == [101, 202]
    assert tensorflow_seeds == [101, 202]


def test_worker_fresh_process_determinism_and_manifest_identity(tmp_path: Path):
    same_a = _run_worker(tmp_path, name="same-a", seed=41)
    same_b = _run_worker(tmp_path, name="same-b", seed=41)
    changed = _run_worker(tmp_path, name="changed", seed=42)

    same_a_source = _load_arrays(same_a / "source.npz")
    same_b_source = _load_arrays(same_b / "source.npz")
    changed_source = _load_arrays(changed / "source.npz")
    same_a_train = _load_arrays(same_a / "train.npz")
    same_b_train = _load_arrays(same_b / "train.npz")
    changed_train = _load_arrays(changed / "train.npz")
    same_a_test = _load_arrays(same_a / "test.npz")

    for name in ("objectGuess", "probeGuess"):
        assert np.array_equal(same_a_source[name], same_b_source[name])
    for name in (
        "diff3d",
        "xcoords",
        "ycoords",
        "xcoords_start",
        "ycoords_start",
        "probeGuess",
        "objectGuess",
        "scan_index",
    ):
        assert np.array_equal(same_a_train[name], same_b_train[name])

    assert not np.array_equal(
        same_a_source["objectGuess"], changed_source["objectGuess"]
    )
    assert not np.array_equal(same_a_train["xcoords"], changed_train["xcoords"])
    assert not np.array_equal(same_a_train["diff3d"], changed_train["diff3d"])
    assert not np.array_equal(same_a_train["xcoords"], same_a_test["xcoords"])
    assert not np.array_equal(same_a_train["diff3d"], same_a_test["diff3d"])

    manifest = json.loads((same_a / "manifest.json").read_text(encoding="utf-8"))
    from ptycho.simulation.identity import (
        array_sha256,
        canonical_sha256,
        file_sha256,
    )

    assert manifest["storage_layout"] == "flat_acquisition_v1"
    assert manifest["artifacts"] == {
        "source": "source.npz",
        "train": "train.npz",
        "test": "test.npz",
    }
    assert manifest["object"]["recipe"] == "lines-object-v1"
    assert manifest["object"]["producer_symbols"] == [
        "ptycho.diffsim.mk_lines_img",
        "ptycho.diffsim.dummy_phi",
    ]
    assert len(manifest["object"]["source_commit"]) >= 7
    assert len(manifest["object"]["array_sha256"]) == 64
    assert manifest["object"]["array_sha256"] == array_sha256(
        same_a_source["objectGuess"]
    )
    assert manifest["object"]["seed"] == manifest["seed_lineage"]["object"]
    assert manifest["runtime_environment"] == {
        "cuda_visible_devices": "",
        "tensorflow_visible_gpu_count": 0,
    }
    assert (
        manifest["seed_lineage"]["train_coordinates"]
        != manifest["seed_lineage"]["test_coordinates"]
    )
    assert (
        manifest["seed_lineage"]["train_noise"]
        != manifest["seed_lineage"]["test_noise"]
    )
    for split in ("train", "test"):
        record = manifest["splits"][split]
        assert record["artifact_path"] == f"{split}.npz"
        assert record["measurement_identity"] == {
            "measurement_domain": "normalized_amplitude",
            "scale_contract_version": "legacy_v1",
            "photons_per_pattern": 1e9,
        }
        assert len(record["split_recipe_sha256"]) == 64
        assert len(record["dataset_sha256"]) == 64
        assert len(record["npz_sha256"]) == 64
        assert record["split_recipe_sha256"] == canonical_sha256(
            record["split_recipe_identity"]
        )
        assert record["split_recipe_identity"]["object_identity"] == {
            key: manifest["object"][key]
            for key in (
                "recipe",
                "producer_symbols",
                "source_commit",
                "array_sha256",
            )
        }
        assert record["dataset_sha256"] == canonical_sha256(
            record["dataset_identity"]
        )
        assert record["npz_sha256"] == file_sha256(same_a / f"{split}.npz")
        assert record["seed_lineage"] == manifest["seed_lineage"]
        assert record["shapes"]["diff3d"] == [2, 64, 64]
        assert record["dtypes"]["diff3d"] == "float32"
        split_arrays = _load_arrays(same_a / f"{split}.npz")
        assert set(record["array_sha256"]) == set(split_arrays)
        assert record["array_sha256"]["diff3d"] == array_sha256(
            split_arrays["diff3d"]
        )


def test_generate_flat_acquisitions_preserves_resolved_logical_geometry(tmp_path: Path):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "gridsize": 2,
                "seed": 13,
                "train_patterns": 4,
                "test_patterns": 4,
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 4,
                "neighbor_pool_size": 4,
            },
        }
    )
    result = generate_flat_acquisitions(resolved, tmp_path / "datasets")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert resolved.simulation.train.scan.grid_size == (2, 2)
    assert manifest["simulation"]["train"]["scan"]["grid_size"] == [2, 2]
    assert _load_arrays(result.train_path)["diff3d"].ndim == 3


def test_flat_service_rejects_pathless_custom_probe_before_writing(tmp_path: Path):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions

    resolved = resolve_synthetic_workflow()
    pathless_probe = replace(
        resolved.simulation.train.probe,
        source="custom",
        source_path=None,
    )
    simulation = replace(
        resolved.simulation,
        train=replace(resolved.simulation.train, probe=pathless_probe),
        test=replace(resolved.simulation.test, probe=pathless_probe),
    )
    resolved = replace(resolved, simulation=simulation)
    output = tmp_path / "datasets"

    with pytest.raises(ValueError, match="source_path.*custom probe"):
        generate_flat_acquisitions(resolved, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("field_path", "message"),
    [
        (
            "N",
            r"simulation\.test\.N must match simulation\.train\.N",
        ),
        (
            "probe.mask_diameter",
            (
                r"simulation\.test\.probe\.mask_diameter must match "
                r"simulation\.train\.probe\.mask_diameter"
            ),
        ),
    ],
)
def test_flat_service_rejects_mismatched_shared_split_recipe_before_writing(
    tmp_path: Path,
    field_path: str,
    message: str,
):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions

    resolved = resolve_synthetic_workflow()
    test_simulation = resolved.simulation.test
    if field_path == "N":
        test_simulation = replace(
            test_simulation,
            N=64,
            probe=replace(
                test_simulation.probe,
                transform_pipeline="smooth:0.5|pad_preserve:64",
            ),
        )
    else:
        test_simulation = replace(
            test_simulation,
            probe=replace(test_simulation.probe, mask_diameter=32.0),
        )
    resolved = replace(
        resolved,
        simulation=replace(resolved.simulation, test=test_simulation),
    )
    output = tmp_path / "datasets"

    with pytest.raises(ValueError, match=message):
        generate_flat_acquisitions(resolved, output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("control", "message"),
    [
        ("objects_per_probe", "objects_per_probe.*exactly 1"),
        ("scan_kind", "scan.kind.*nongrid"),
        ("beamstop", "beamstop_diameter.*unsupported"),
        ("offset", "scan.offset.*exactly 4"),
        ("outer_offset_train", "outer_offset_train.*exactly 8"),
        ("outer_offset_test", "outer_offset_test.*exactly 20"),
        ("scale_contract", "scale_contract_version.*legacy_v1"),
        ("measurement_domain", "measurement_domain.*normalized_amplitude"),
    ],
)
def test_flat_service_rejects_unimplemented_resolved_controls_before_writing(
    monkeypatch, tmp_path: Path, control: str, message: str
):
    from ptycho.simulation import flat_acquisition

    resolved = resolve_synthetic_workflow()
    simulation = resolved.simulation
    if control in {"scale_contract", "measurement_domain"}:
        field_name = (
            "scale_contract_version"
            if control == "scale_contract"
            else "measurement_domain"
        )
        value = "ci_intensity_v2" if control == "scale_contract" else "count_intensity"
        simulation = replace(simulation, **{field_name: value})
    else:
        def mutate(split):
            if control == "objects_per_probe":
                return replace(
                    split,
                    object=replace(split.object, objects_per_probe=2),
                )
            if control == "scan_kind":
                return replace(split, scan=replace(split.scan, kind="grid"))
            if control == "beamstop":
                return replace(
                    split,
                    detector=replace(split.detector, beamstop_diameter=8.0),
                )
            replacement = {
                "offset": 5,
                "outer_offset_train": 9,
                "outer_offset_test": 21,
            }[control]
            return replace(
                split,
                scan=replace(split.scan, **{control: replacement}),
            )

        simulation = replace(
            simulation,
            train=mutate(simulation.train),
            test=mutate(simulation.test),
        )
    resolved = replace(resolved, simulation=simulation)
    output = tmp_path / control
    monkeypatch.setattr(
        flat_acquisition,
        "build_lines_object",
        lambda rng: pytest.fail("generation started before control validation"),
    )

    with pytest.raises(ValueError, match=message):
        flat_acquisition.generate_flat_acquisitions(resolved, output)

    assert not output.exists()


def test_atomic_publication_never_clobbers_a_racing_destination(
    monkeypatch, tmp_path: Path
):
    from ptycho.simulation import flat_acquisition

    destination = tmp_path / "manifest.json"
    real_link = os.link

    def racing_link(source, target):
        Path(target).write_bytes(b"competitor")
        return real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)

    with pytest.raises(FileExistsError):
        flat_acquisition._write_json_atomic(destination, {"ours": True})

    assert destination.read_bytes() == b"competitor"
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
