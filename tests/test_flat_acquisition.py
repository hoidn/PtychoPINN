"""Deterministic production flat-acquisition generation contracts."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from ptycho.config.config import ModelConfig, TrainingConfig
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
            },
        },
    }


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _write_frozen_object_bank(
    path: Path,
    *,
    train_count: int = 2,
    test_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.stack(
        [
            np.full((392, 392), 1.0 + 0.1j * (index + 1), dtype=np.complex64)
            for index in range(train_count)
        ]
    )
    test = np.stack(
        [
            np.full((392, 392), 0.8 + 0.2j * (index + 1), dtype=np.complex64)
            for index in range(test_count)
        ]
    )
    np.savez(path, trainObjectGuess=train, testObjectGuess=test)
    return train, test


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


def test_frozen_object_bank_loader_binds_exact_arrays_and_source_identity(tmp_path):
    from ptycho.simulation.identity import array_sha256, file_sha256
    from ptycho.simulation.object_producers import load_frozen_object_banks

    source = tmp_path / "objects.npz"
    train, test = _write_frozen_object_bank(source)

    banks, source_identity = load_frozen_object_banks(
        "lines",
        source,
        train_count=2,
        test_count=1,
        image_size=(392, 392),
        shared_object=False,
    )

    assert np.array_equal(banks["train"][0].array, train[0])
    assert np.array_equal(banks["test"][0].array, test[0])
    assert source_identity == {
        "version": "frozen-object-bank-source-v1",
        "source_path": str(source),
        "source_file_sha256": file_sha256(source),
        "arrays": {
            "trainObjectGuess": {
                "array_sha256": array_sha256(train),
                "shape": [2, 392, 392],
                "dtype": "complex64",
            },
            "testObjectGuess": {
                "array_sha256": array_sha256(test),
                "shape": [1, 392, 392],
                "dtype": "complex64",
            },
        },
    }
    first_identity = banks["train"][0].source_identity
    assert first_identity["source_file_sha256"] == file_sha256(source)
    assert first_identity["source_key"] == "trainObjectGuess"
    assert first_identity["source_index"] == 0
    assert first_identity["source_array_sha256"] == array_sha256(train[0])
    assert banks["train"][0].rng_identity == {}
    assert banks["train"][0].phase_identity == {
        "version": "frozen-complex-source-v1"
    }


def test_frozen_object_bank_loader_hashes_the_exact_loaded_byte_snapshot(
    tmp_path,
    monkeypatch,
):
    from ptycho.simulation.object_producers import load_frozen_object_banks

    source = tmp_path / "objects.npz"
    original_train, original_test = _write_frozen_object_bank(source)
    original_bytes = source.read_bytes()
    replacement = io.BytesIO()
    np.savez(
        replacement,
        trainObjectGuess=np.full_like(original_train, 9 + 4j),
        testObjectGuess=np.full_like(original_test, 7 + 2j),
    )
    replacement_bytes = replacement.getvalue()
    read_bytes = Path.read_bytes

    def replace_after_read(path):
        payload = read_bytes(path)
        if path == source:
            source.write_bytes(replacement_bytes)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    banks, source_identity = load_frozen_object_banks(
        "lines",
        source,
        train_count=2,
        test_count=1,
        image_size=(392, 392),
        shared_object=False,
    )

    assert np.array_equal(banks["train"][0].array, original_train[0])
    assert np.array_equal(banks["test"][0].array, original_test[0])
    assert source_identity["source_file_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert source.read_bytes() == replacement_bytes


def test_frozen_object_bank_loader_rejects_duplicate_npz_members(tmp_path):
    from ptycho.simulation.object_producers import load_frozen_object_banks

    source = tmp_path / "objects.npz"
    train = np.ones((2, 392, 392), dtype=np.complex64)
    test = np.ones((1, 392, 392), dtype=np.complex64)

    def npy_bytes(array):
        payload = io.BytesIO()
        np.save(payload, array, allow_pickle=False)
        return payload.getvalue()

    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("trainObjectGuess.npy", npy_bytes(train))
        archive.writestr("trainObjectGuess.npy", npy_bytes(train))
        archive.writestr("testObjectGuess.npy", npy_bytes(test))

    with pytest.raises(ValueError, match="exactly"):
        load_frozen_object_banks(
            "lines",
            source,
            train_count=2,
            test_count=1,
            image_size=(392, 392),
            shared_object=False,
        )


def test_frozen_object_bank_recipe_cannot_be_misrepresented_as_seed_generated():
    from ptycho.simulation.object_producers import build_object_from_seed

    with pytest.raises(ValueError, match="source-backed"):
        build_object_from_seed("lines", "frozen-object-bank-v1", 3)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"trainObjectGuess": np.ones((2, 392, 392), dtype=np.complex64)},
            "exactly",
        ),
        (
            {
                "trainObjectGuess": np.ones((2, 392, 392), dtype=np.float32),
                "testObjectGuess": np.ones((1, 392, 392), dtype=np.complex64),
            },
            "complex64",
        ),
        (
            {
                "trainObjectGuess": np.ones((1, 392, 392), dtype=np.complex64),
                "testObjectGuess": np.ones((1, 392, 392), dtype=np.complex64),
            },
            "shape",
        ),
        (
            {
                "trainObjectGuess": np.full(
                    (2, 392, 392), np.complex64(np.nan + 0j)
                ),
                "testObjectGuess": np.ones((1, 392, 392), dtype=np.complex64),
            },
            "finite",
        ),
    ],
)
def test_frozen_object_bank_loader_rejects_ambiguous_or_invalid_inputs(
    tmp_path,
    payload,
    message,
):
    from ptycho.simulation.object_producers import load_frozen_object_banks

    source = tmp_path / "objects.npz"
    np.savez(source, **payload)

    with pytest.raises(ValueError, match=message):
        load_frozen_object_banks(
            "dead_leaves",
            source,
            train_count=2,
            test_count=1,
            image_size=(392, 392),
            shared_object=False,
        )


def test_frozen_object_bank_generation_records_and_revalidates_external_source(
    tmp_path,
):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions
    from ptycho.workflows import synthetic_pipeline

    source = tmp_path / "objects.npz"
    train, test = _write_frozen_object_bank(source)
    resolved = resolve_synthetic_workflow(
        profile="synthetic-lines",
        file_values={
            "simulation": {
                "N": 64,
                "seed": 5,
                "train_patterns": 8,
                "test_patterns": 4,
                "train_objects": 2,
                "test_objects": 1,
                "shared_object": False,
                "frame_order_recipe": "coordinate-major-interleaved-v1",
                "object_recipe": "frozen-object-bank-v1",
                "object": {"source_path": source},
                "scan": {
                    "position_layout": "fixed_pitch_raster",
                    "outer_offset_train": 8,
                    "outer_offset_test": 20,
                },
            },
            "training": {
                "train_raw_selection": 8,
                "training_groups": 8,
                "validation_groups": 4,
                "neighbor_count": 1,
            },
        },
    )

    result = generate_flat_acquisitions(resolved, tmp_path / "generated")

    generated_source = _load_arrays(result.source_path)
    generated_train = _load_arrays(result.train_path)
    assert np.array_equal(generated_source["trainObjectGuess"], train)
    assert np.array_equal(generated_source["testObjectGuess"], test)
    np.testing.assert_array_equal(generated_train["object_index"], [0, 1] * 4)
    np.testing.assert_array_equal(
        generated_train["xcoords"],
        [32.0, 32.0, 32.0, 32.0, 36.0, 36.0, 36.0, 36.0],
    )
    np.testing.assert_array_equal(
        generated_train["ycoords"],
        [32.0, 32.0, 36.0, 36.0, 32.0, 32.0, 36.0, 36.0],
    )
    assert result.manifest["object_source"]["source_path"] == str(source)
    assert result.manifest["schema_version"] == "flat-acquisition-manifest-v4"
    assert "seed" not in result.manifest["objects"]["train"][0]
    assert "rng_identity" not in result.manifest["objects"]["train"][0]
    assert result.manifest["objects"]["train"][0]["identity_mode"] == "source"
    assert result.manifest["objects"]["train"][0]["source_identity"][
        "source_key"
    ] == "trainObjectGuess"
    assert result.manifest["objects"]["test"][0]["source_identity"][
        "source_key"
    ] == "testObjectGuess"
    assert "object" not in result.manifest["seed_lineage"]
    for split in ("train", "test"):
        split_record = result.manifest["splits"][split]
        assert "object" not in split_record["seed_lineage"]
        assert "object_seed_records" not in split_record
        assert "object_seed_records" not in split_record["split_recipe_identity"]
        assert split_record["acquisition_seed_records"]
        assert split_record["split_recipe_identity"][
            "acquisition_seed_records"
        ] == split_record["acquisition_seed_records"]
        assert split_record["split_recipe_identity"][
            "frame_order_recipe"
        ] == "coordinate-major-interleaved-v1"
        assert all(
            "object_seed" not in record
            for record in split_record["acquisition_seed_records"]
        )
    synthetic_pipeline._load_matching_dataset_manifest(
        result.manifest_path,
        resolved,
    )

    original_manifest = result.manifest_path.read_text(encoding="utf-8")
    tampered = json.loads(original_manifest)
    tampered["seed_lineage"]["object"] = 123
    result.manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="seed_lineage"):
        synthetic_pipeline._load_matching_dataset_manifest(
            result.manifest_path,
            resolved,
        )
    result.manifest_path.write_text(original_manifest, encoding="utf-8")

    tampered = json.loads(original_manifest)
    tampered["objects"]["train"][0]["object_seed"] = 123
    result.manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity forbids"):
        synthetic_pipeline._load_matching_dataset_manifest(
            result.manifest_path,
            resolved,
        )
    result.manifest_path.write_text(original_manifest, encoding="utf-8")

    tampered = json.loads(original_manifest)
    tampered["splits"]["train"]["object_seed_records"] = [
        {"index": 0, "object_seed": 123}
    ]
    with pytest.raises(ValueError, match="object_seed_records.*forbidden"):
        synthetic_pipeline._verify_split_artifact(
            result.train_path,
            manifest=tampered,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split="train",
        )

    tampered = json.loads(original_manifest)
    tampered["splits"]["train"]["split_recipe_identity"][
        "frame_order_recipe"
    ] = "object-major-v1"
    with pytest.raises(ValueError, match="split_recipe_identity mismatch"):
        synthetic_pipeline._verify_split_artifact(
            result.train_path,
            manifest=tampered,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split="train",
        )

    _write_frozen_object_bank(source, train_count=2, test_count=1)
    with np.load(source, allow_pickle=False) as archive:
        mutated_train = np.array(archive["trainObjectGuess"], copy=True)
        unchanged_test = np.array(archive["testObjectGuess"], copy=True)
    mutated_train[0, 0, 0] += np.complex64(1.0)
    np.savez(
        source,
        trainObjectGuess=mutated_train,
        testObjectGuess=unchanged_test,
    )
    with pytest.raises(ValueError, match="object_source"):
        synthetic_pipeline._load_matching_dataset_manifest(
            result.manifest_path,
            resolved,
        )


def test_manifest_v4_is_exclusive_to_frozen_object_banks(tmp_path):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions
    from ptycho.workflows import synthetic_pipeline

    resolved = resolve_synthetic_workflow(**_small_request(seed=17))
    result = generate_flat_acquisitions(resolved, tmp_path / "generated")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "flat-acquisition-manifest-v3"
    manifest["schema_version"] = "flat-acquisition-manifest-v4"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exclusive to frozen-object-bank-v1"):
        synthetic_pipeline._load_matching_dataset_manifest(
            result.manifest_path,
            resolved,
        )


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
        model=ModelConfig(N=8, gridsize=1, object_big=False), training_groups=1
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
        model=ModelConfig(N=8, gridsize=1, object_big=False), training_groups=1
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


def test_dead_leaves_object_uses_the_locked_producers_recipe_and_complex64(
    monkeypatch,
):
    from ptycho import diffsim
    from ptycho.simulation.flat_acquisition import build_dead_leaves_object
    from ptycho.simulation.object_producers import fixed_dead_leaves_phase
    from ptycho_torch.datagen import objects

    rng = np.random.default_rng(31)
    calls: dict[str, object] = {}
    amplitude = np.linspace(0.6, 1.1, 392 * 392, dtype=np.float32).reshape(392, 392)
    raw_object = amplitude * np.exp(
        1j * np.linspace(-np.pi, np.pi, 392 * 392).reshape(392, 392)
    )

    shape_rng = np.random.default_rng(32)

    def fake_dead_leaves(shape, obj_arg, *, rng, shape_rng):
        calls.update(
            shape=shape,
            obj_arg=dict(obj_arg),
            rng=rng,
            shape_rng=shape_rng,
        )
        return raw_object

    monkeypatch.setattr(objects, "create_dead_leaves", fake_dead_leaves)
    monkeypatch.setattr(
        diffsim,
        "dummy_phi",
        lambda _amplitude: pytest.fail("v2 must not use the per-object phase law"),
    )

    result = build_dead_leaves_object(rng, shape_rng=shape_rng)

    expected_amplitude = np.abs(raw_object)
    expected = expected_amplitude * np.exp(
        1j * fixed_dead_leaves_phase(expected_amplitude)
    )
    assert calls["shape"] == (392, 392)
    assert calls["obj_arg"] == {
        "max_iters": 700,
        "r_min_frac": 0.02,
        "r_max_frac": 0.18,
        "r_sigma": 3,
    }
    assert calls["rng"] is rng
    assert calls["shape_rng"] is shape_rng
    assert result.recipe == "dead-leaves-object-v2"
    assert result.producer_symbols == (
        "ptycho_torch.datagen.objects.create_dead_leaves",
        "ptycho.simulation.object_producers.fixed_dead_leaves_phase",
    )
    assert result.phase_identity == {
        "version": "dead-leaves-fixed-phase-v1",
        "reference_max_amplitude": 1.1,
        "reference_mean_amplitude": 0.95,
    }
    assert result.array.dtype == np.complex64
    assert result.array.flags.c_contiguous
    assert np.allclose(result.array, expected.astype(np.complex64))


def test_frozen_dead_leaves_v1_retains_its_gpu_qualified_golden_bytes():
    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("the frozen v1 byte golden is qualified to TensorFlow GPU")

    from ptycho.simulation.identity import array_sha256
    from ptycho.simulation.object_producers import build_object_from_seed

    result = build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v1",
        123,
    )

    assert result.producer_symbols == (
        "ptycho_torch.datagen.objects.create_dead_leaves",
        "ptycho.diffsim.dummy_phi",
    )
    assert array_sha256(result.array) == (
        "42779775ac3bdde71a9f5535b1948eea0bcdcdbbd9f6e502d7caa5fa1b2be66e"
    )


def test_frozen_dead_leaves_v1_retains_its_backend_neutral_phase_semantics():
    from ptycho.simulation.object_producers import build_object_from_seed

    result = build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v1",
        123,
    )
    amplitude = np.asarray(np.abs(result.array), dtype=np.float32)
    expected_phase = np.asarray(
        np.float32(np.pi)
        * np.tanh(
            (amplitude - np.max(amplitude) / np.float32(2.0))
            / (np.float32(3.0) * np.mean(amplitude))
        ),
        dtype=np.float32,
    )

    assert result.producer_symbols == (
        "ptycho_torch.datagen.objects.create_dead_leaves",
        "ptycho.diffsim.dummy_phi",
    )
    assert result.array.dtype == np.complex64
    assert np.isfinite(result.array).all()
    np.testing.assert_allclose(
        np.angle(result.array),
        expected_phase,
        rtol=0.0,
        atol=3e-7,
    )


def test_registered_dead_leaves_object_is_seed_deterministic():
    from ptycho.simulation.object_producers import build_object_from_seed

    same_a = build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v2",
        123,
    )
    same_b = build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v2",
        123,
    )
    changed = build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v2",
        124,
    )

    assert np.array_equal(same_a.array, same_b.array)
    assert not np.array_equal(same_a.array, changed.array)
    assert same_a.rng_identity == {
        "version": "dead-leaves-rng-v2",
        "parent_seed": 123,
        "streams": {
            "shape": {
                "bit_generator": "PCG64",
                "seed_sequence": {"entropy": 123, "spawn_key": [0]},
            },
            "numeric": {
                "bit_generator": "PCG64",
                "seed_sequence": {"entropy": 123, "spawn_key": [1]},
            },
        },
    }


def test_dead_leaves_v2_instantiates_the_manifest_named_bit_generator(
    monkeypatch,
):
    from ptycho.simulation import object_producers
    from ptycho_torch.datagen import objects

    streams: dict[str, np.random.Generator] = {}

    def fake_dead_leaves(shape, _arguments, *, rng, shape_rng):
        streams.update(numeric=rng, shape=shape_rng)
        return np.ones(shape, dtype=np.complex64)

    default_rng = np.random.default_rng

    def changing_default(seed=None):
        if isinstance(seed, np.random.SeedSequence):
            return np.random.Generator(np.random.Philox(seed))
        return default_rng(seed)

    monkeypatch.setattr(objects, "create_dead_leaves", fake_dead_leaves)
    monkeypatch.setattr(np.random, "default_rng", changing_default)

    result = object_producers.build_object_from_seed(
        "dead_leaves",
        "dead-leaves-object-v2",
        123,
    )

    assert isinstance(streams["numeric"].bit_generator, np.random.PCG64)
    assert isinstance(streams["shape"].bit_generator, np.random.PCG64)
    assert {
        stream["bit_generator"] for stream in result.rng_identity["streams"].values()
    } == {"PCG64"}


def test_frozen_dead_leaves_v1_rejects_split_random_streams():
    from ptycho.simulation.object_producers import build_object

    with pytest.raises(ValueError, match="v1 requires one combined random stream"):
        build_object(
            "dead_leaves",
            "dead-leaves-object-v1",
            np.random.default_rng(1),
            shape_rng=np.random.default_rng(2),
        )


def test_seeded_dead_leaves_does_not_mutate_ambient_rng_state():
    import random

    from ptycho.simulation.object_producers import build_object_from_seed

    numpy_before = np.random.get_state()
    python_before = random.getstate()

    build_object_from_seed("dead_leaves", "dead-leaves-object-v2", 123)

    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()


def test_dead_leaves_v2_routes_geometry_and_material_draws_to_named_streams():
    from ptycho_torch.datagen.objects import dead_leaves_ptycho

    class RecordingGenerator(np.random.Generator):
        def __init__(self, seed):
            super().__init__(np.random.PCG64(seed))
            self.calls = []

        def choice(self, *args, **kwargs):
            self.calls.append("choice")
            return super().choice(*args, **kwargs)

        def integers(self, *args, **kwargs):
            self.calls.append("integers")
            return super().integers(*args, **kwargs)

        def uniform(self, *args, **kwargs):
            self.calls.append("uniform")
            return super().uniform(*args, **kwargs)

        def normal(self, *args, **kwargs):
            self.calls.append("normal")
            return super().normal(*args, **kwargs)

    numeric_rng = RecordingGenerator(11)
    shape_rng = RecordingGenerator(12)
    dead_leaves_ptycho(
        res=32,
        r_sigma_param=3,
        max_iters=10,
        r_min_frac=0.02,
        r_max_frac=0.18,
        beta_pareto_alpha=1.5,
        beta_scale=0.001,
        delta_beta_mean=100,
        delta_beta_std=10,
        thickness=3.0,
        min_phase=-np.pi,
        max_phase=np.pi,
        min_amp=0.6,
        max_amp=1.1,
        rng=numeric_rng,
        shape_rng=shape_rng,
    )

    assert numeric_rng.calls.count("uniform") == 10
    assert numeric_rng.calls.count("normal") == 10
    assert not {"choice", "integers"}.intersection(numeric_rng.calls)
    assert shape_rng.calls.count("choice") == 10
    assert "uniform" in shape_rng.calls
    assert "integers" in shape_rng.calls
    assert "normal" not in shape_rng.calls


def test_object_recipe_is_derived_from_kind_and_explicit_mismatches_fail_closed():
    dead_leaves = resolve_synthetic_workflow(
        file_values={"simulation": {"object": {"kind": "dead_leaves"}}}
    )

    assert dead_leaves.simulation.object_recipe == "dead-leaves-object-v2"

    frozen_v1 = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "object": {"kind": "dead_leaves"},
                "object_recipe": "dead-leaves-object-v1",
            }
        }
    )
    assert frozen_v1.simulation.object_recipe == "dead-leaves-object-v1"

    with pytest.raises(ValueError, match="object_recipe.*does not match.*dead_leaves"):
        resolve_synthetic_workflow(
            file_values={
                "simulation": {
                    "object": {"kind": "dead_leaves"},
                    "object_recipe": "lines-object-v1",
                }
            }
        )
    with pytest.raises(ValueError, match="unsupported simulation.object.kind"):
        resolve_synthetic_workflow(
            file_values={"simulation": {"object": {"kind": "natural_patch"}}}
        )


def test_flat_service_revalidates_object_kind_recipe_identity_before_writing(
    tmp_path: Path,
):
    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions

    resolved = resolve_synthetic_workflow(
        file_values={"simulation": {"object": {"kind": "dead_leaves"}}}
    )
    mismatched = replace(
        resolved,
        simulation=replace(resolved.simulation, object_recipe="lines-object-v1"),
    )
    output = tmp_path / "datasets"

    with pytest.raises(ValueError, match="object_recipe.*does not match.*dead_leaves"):
        generate_flat_acquisitions(mismatched, output)

    assert not output.exists()


def test_dead_leaves_manifest_records_locked_object_identity(
    monkeypatch,
    tmp_path: Path,
):
    from ptycho.simulation import flat_acquisition
    from ptycho.simulation.identity import array_sha256
    from ptycho_torch.datagen import objects

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "train_patterns": 2,
                "test_patterns": 2,
                "object": {"kind": "dead_leaves"},
            },
            "training": {
                "train_raw_selection": 2,
                "training_groups": 2,
                "validation_groups": 2,
                "neighbor_count": 1,
            },
        }
    )
    amplitude = np.full((392, 392), 0.75, dtype=np.float32)
    monkeypatch.setattr(
        objects,
        "create_dead_leaves",
        lambda shape, obj_arg, *, rng, shape_rng: amplitude,
    )
    monkeypatch.setattr(
        flat_acquisition,
        "_prepare_probe",
        lambda simulation: (
            np.ones((64, 64), dtype=np.complex64),
            {
                "probe_lineage": {
                    "raw_probe_sha256": "raw-probe",
                    "transformed_probe_sha256": "transformed-probe",
                }
            },
        ),
    )

    def fake_split(simulation, *, object_guess, probe_guess, **_kwargs):
        sample_count = simulation.object.diffractions_per_object
        return {
            "diff3d": np.ones((sample_count, 64, 64), dtype=np.float32),
            "xcoords": np.arange(sample_count, dtype=np.float64),
            "ycoords": np.arange(sample_count, dtype=np.float64),
            "probeGuess": probe_guess,
            "objectGuess": object_guess,
            "scan_index": np.arange(sample_count, dtype=np.int64),
        }

    monkeypatch.setattr(flat_acquisition, "_simulate_split", fake_split)
    monkeypatch.setattr(flat_acquisition, "_source_commit", lambda: "test-commit")
    monkeypatch.setattr(
        flat_acquisition,
        "_runtime_environment",
        lambda: {"cuda_visible_devices": "", "tensorflow_visible_gpu_count": 0},
    )
    monkeypatch.setattr(
        flat_acquisition,
        "_truth_forward_closure",
        lambda *_args, **_kwargs: {
            "version": "truth-forward-closure-v2",
            "sample_policy": "deterministic-random-per-object",
            "samples_per_object": 16,
            "measurement_domain": "normalized_amplitude",
            "relative_l2_limit_policy": (
                "max(0.005,3*N/(2*sqrt(photons_per_pattern)))"
            ),
            "truth_patch_max_relative_l2": 1e-6,
            "objects": [],
            "passed": True,
        },
    )

    result = flat_acquisition.generate_flat_acquisitions(
        resolved, tmp_path / "datasets"
    )

    source = _load_arrays(result.source_path)
    assert result.manifest["object"]["recipe"] == "dead-leaves-object-v2"
    assert result.manifest["object"]["rng_identity"]["version"] == (
        "dead-leaves-rng-v2"
    )
    assert result.manifest["object"]["producer_symbols"] == [
        "ptycho_torch.datagen.objects.create_dead_leaves",
        "ptycho.simulation.object_producers.fixed_dead_leaves_phase",
    ]
    assert result.manifest["object"]["array_sha256"] == array_sha256(
        source["objectGuess"]
    )


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
        "object_index": np.dtype("int64"),
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
        training_groups=2,
        nphotons=1e5,
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


def test_materialized_object_leaf_uses_generic_legacy_source_and_restores_kind(
    monkeypatch,
):
    from ptycho import nongrid_simulation, params

    logical = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        training_groups=2,
        nphotons=1e5,
    )
    ambient = {
        **params.DEFAULT_CFG,
        "data_source": "dead_leaves",
        "object_class": "dead_leaves",
        "marker": "modern-object-identity",
    }
    monkeypatch.setattr(params, "cfg", dict(ambient))

    def fake_from_simulation(xcoords, ycoords, probe, obj, scan_index):
        assert params.cfg["data_source"] == "generic"
        assert params.cfg["object_class"] == "dead_leaves"
        return SimpleNamespace(diff3d=np.ones((2, 8, 8), dtype=np.float32))

    monkeypatch.setattr(
        nongrid_simulation.RawData,
        "from_simulation",
        fake_from_simulation,
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
    assert params.cfg == ambient


def test_detector_seed_is_independent_of_identical_coordinate_stream(monkeypatch):
    import tensorflow as tf

    from ptycho import nongrid_simulation

    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        training_groups=2,
        nphotons=1e5,
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
                "rng_identity",
                "phase_identity",
                "producer_symbols",
                "source_commit",
                "array_sha256",
            )
        }
        assert "frame_order_recipe" not in record["split_recipe_identity"]
        assert record["dataset_sha256"] == canonical_sha256(
            record["dataset_identity"]
        )
        assert record["npz_sha256"] == file_sha256(same_a / f"{split}.npz")
        assert record["seed_lineage"] == manifest["seed_lineage"]
        assert record["object_seed_records"]
        assert "acquisition_seed_records" not in record
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
        ("objects_per_probe", "shared_object=True.*exactly one"),
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


# --- Count-intensity (CI) contract ------------------------------------------

# Cross-version probe baseline: the digest of the amplitude-path probe
# (``probeGuess`` / ``transformed_probe_sha256``). This value is
# device-independent (verified identical on CPU and GPU) and must not drift
# across refactors. Commit 1189c57ff ("refactor: canonicalize acquisition and
# grouping") changed simulation byte output for the diffraction arrays but left
# the probe transform pipeline unchanged; the CPU suite is authoritative for
# this value.
# The diffraction arrays (diff3d) are NOT pinned here: their bytes are
# device-dependent (TensorFlow Poisson sampling differs on GPU vs CPU) and were
# legitimately shifted by 1189c57ff. They are covered by the relative in-run
# comparison in test_amplitude_contract_arrays_are_unchanged_by_the_count_branch.
AMPLITUDE_BASELINE = {
    "probe": "c4f029239d37aaeb5ab97c921a95f19e3c7e01d1311dfc29ba759b5b5e92c0d8",
}


def _small_resolved(profile, seed=5, **overrides):
    file_values = {
        "simulation": {"N": 64, "seed": seed, "train_patterns": 2, "test_patterns": 2},
        "training": {
            "train_raw_selection": 2,
            "training_groups": 2,
            "validation_groups": 2,
            "neighbor_count": 1,
        },
    }
    for namespace, patch in overrides.items():
        file_values.setdefault(namespace, {}).update(patch)
    return resolve_synthetic_workflow(profile=profile, file_values=file_values)


def _small_dead_leaves_raster_workflow(
    *,
    object_recipe: str = "dead-leaves-object-v2",
):
    return _small_resolved(
        "cnn-lines-ci",
        simulation={
            "object": {"kind": "dead_leaves"},
            "object_recipe": object_recipe,
            "scan": {"position_layout": "fixed_pitch_raster"},
            "train_patterns": 4,
            "test_patterns": 4,
        },
        training={
            "train_raw_selection": 4,
            "training_groups": 4,
            "validation_groups": 4,
            "neighbor_count": 1,
        },
    )


def _generate(resolved, root):
    from ptycho.simulation import flat_acquisition

    return flat_acquisition.generate_flat_acquisitions(resolved, root)


def test_amplitude_contract_arrays_are_unchanged_by_the_count_branch(tmp_path: Path):
    from ptycho.simulation.identity import array_sha256

    # Amplitude path (count branch OFF) and count path (ON) share one seed
    # lineage; the count branch must be a pure post-transform of the amplitude
    # arrays it consumes, never a regeneration. Diffraction bytes are
    # device-dependent, so they are compared relatively in-run rather than
    # pinned absolutely (see the AMPLITUDE_BASELINE provenance note above).
    amplitude = _generate(_small_resolved("synthetic-lines"), tmp_path / "amp")
    counts = _generate(_small_resolved("cnn-lines-ci"), tmp_path / "ci")

    amp_train = _load_arrays(amplitude.train_path)
    amp_test = _load_arrays(amplitude.test_path)
    ci_train = _load_arrays(counts.train_path)
    ci_test = _load_arrays(counts.test_path)

    # The probe is device-independent and cross-version pinned.
    assert array_sha256(amp_train["probeGuess"]) == AMPLITUDE_BASELINE["probe"]
    assert amplitude.manifest["probe"]["transformed_probe_sha256"] == (
        AMPLITUDE_BASELINE["probe"]
    )
    # The count branch preserves the amplitude probe's identity.
    assert counts.manifest["probe"]["transformed_probe_sha256"] == array_sha256(
        amp_train["probeGuess"]
    )

    # The count branch consumed the amplitude path's exact arrays:
    # counts = (amplitude * S)**2 and physical probe = amplitude probe * S.
    scale = counts.manifest["probe"]["count_amplitude_scale"]["value"]
    np.testing.assert_array_equal(
        ci_train["diff3d"],
        np.square(amp_train["diff3d"].astype(np.float64) * scale).astype(np.float32),
    )
    np.testing.assert_array_equal(
        ci_test["diff3d"],
        np.square(amp_test["diff3d"].astype(np.float64) * scale).astype(np.float32),
    )
    np.testing.assert_array_equal(
        ci_train["probeGuess"],
        (amp_train["probeGuess"].astype(np.complex128) * scale).astype(np.complex64),
    )

    # The amplitude path must not emit count-branch artifacts.
    assert "physical_probe_sha256" not in amplitude.manifest["probe"]
    assert "count_amplitude_scale" not in amplitude.manifest["probe"]


def test_legacy_simulation_probe_is_distinct_from_persisted_training_probe(
    tmp_path: Path,
):
    from ptycho import probe as probe_module
    from ptycho.simulation.identity import array_sha256

    resolved = _small_resolved(
        "synthetic-lines",
        simulation={
            "probe": {"simulation_normalization_scale": 4.0},
        },
    )
    result = _generate(resolved, tmp_path / "legacy-probe")

    train = _load_arrays(result.train_path)
    source = _load_arrays(result.source_path)
    training_probe = train["probeGuess"]
    simulation_probe = train["probe_simulated"]
    mask = np.asarray(
        probe_module.get_probe_mask_real(training_probe.shape[0])
    ).squeeze()
    norm = 4.0 * np.mean(np.abs(mask * training_probe))

    np.testing.assert_allclose(simulation_probe, training_probe / norm, rtol=1e-6)
    np.testing.assert_array_equal(source["probeGuess"], training_probe)
    np.testing.assert_array_equal(source["probe_simulated"], simulation_probe)
    assert not np.array_equal(training_probe, simulation_probe)
    assert result.manifest["probe"]["training_probe_sha256"] == array_sha256(
        training_probe
    )
    assert result.manifest["probe"]["simulation_probe_sha256"] == array_sha256(
        simulation_probe
    )
    assert result.manifest["probe"]["simulation_normalization_scale"] == 4.0


def test_independent_object_banks_preserve_total_rows_truth_and_object_identity(
    tmp_path: Path,
):
    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "seed": 3,
                "train_patterns": 4,
                "test_patterns": 2,
                "train_objects": 2,
                "test_objects": 1,
                "shared_object": False,
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 2,
                "neighbor_count": 1,
            },
        }
    )

    result = _generate(resolved, tmp_path / "object-banks")
    train = _load_arrays(result.train_path)
    test = _load_arrays(result.test_path)
    source = _load_arrays(result.source_path)

    assert train["diff3d"].shape[0] == 4
    assert test["diff3d"].shape[0] == 2
    np.testing.assert_array_equal(train["object_index"], [0, 0, 1, 1])
    np.testing.assert_array_equal(test["object_index"], [0, 0])
    assert train["Y"].shape == (4, 64, 64)
    assert test["Y"].shape == (2, 64, 64)
    assert source["trainObjectGuess"].shape == (2, 392, 392)
    assert source["testObjectGuess"].shape == (1, 392, 392)
    np.testing.assert_array_equal(source["objectGuess"], source["testObjectGuess"][0])
    assert len(result.manifest["objects"]["train"]) == 2
    assert len(result.manifest["objects"]["test"]) == 1
    assert len({item["seed"] for item in result.manifest["objects"]["train"]}) == 2
    closure = result.manifest["truth_forward_closure"]
    assert closure["version"] == "truth-forward-closure-v2"
    assert closure["sample_policy"] == "deterministic-random-per-object"
    assert closure["passed"] is True
    assert {(item["split"], item["object_index"]) for item in closure["objects"]} == {
        ("train", 0),
        ("train", 1),
        ("test", 0),
    }
    assert all(
        item["relative_l2"] <= item["relative_l2_limit"]
        for item in closure["objects"]
    )
    assert all(
        item["truth_patch_relative_l2"]
        <= item["truth_patch_relative_l2_limit"]
        for item in closure["objects"]
    )


def test_coordinate_major_frame_recipe_interleaves_objects_and_transposes_raster():
    from ptycho.simulation.flat_acquisition import (
        _concatenate_object_payloads,
        _split_coordinates,
    )

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "seed": 3,
                "train_patterns": 8,
                "test_patterns": 4,
                "train_objects": 2,
                "test_objects": 1,
                "shared_object": False,
                "frame_order_recipe": "coordinate-major-interleaved-v1",
                "scan": {
                    "position_layout": "fixed_pitch_raster",
                    "outer_offset_train": 8,
                },
            },
            "training": {
                "train_raw_selection": 8,
                "training_groups": 8,
                "validation_groups": 4,
                "neighbor_count": 1,
            },
        }
    )
    xcoords, ycoords = _split_coordinates(
        resolved.simulation.train,
        split="train",
        frame_order_recipe=resolved.simulation.frame_order_recipe,
    )
    np.testing.assert_array_equal(xcoords, [32.0, 32.0, 36.0, 36.0])
    np.testing.assert_array_equal(ycoords, [32.0, 36.0, 32.0, 36.0])

    payloads = []
    for object_index in range(2):
        rows = np.arange(4, dtype=np.int64) + 10 * object_index
        payloads.append(
            {
                "diff3d": rows[:, None, None].astype(np.float32),
                "xcoords": xcoords,
                "ycoords": ycoords,
                "xcoords_start": xcoords,
                "ycoords_start": ycoords,
                "Y": rows[:, None, None].astype(np.complex64),
                "scan_index": np.zeros(4, dtype=np.int64),
                "object_index": np.full(4, object_index, dtype=np.int64),
            }
        )
    objects = [SimpleNamespace(array=np.ones((1, 1))) for _ in range(2)]

    combined = _concatenate_object_payloads(
        payloads,
        objects,
        frame_order_recipe=resolved.simulation.frame_order_recipe,
    )

    np.testing.assert_array_equal(combined["object_index"], [0, 1] * 4)
    np.testing.assert_array_equal(
        combined["diff3d"][:, 0, 0],
        [0, 10, 1, 11, 2, 12, 3, 13],
    )
    np.testing.assert_array_equal(
        combined["xcoords"],
        [32.0, 32.0, 32.0, 32.0, 36.0, 36.0, 36.0, 36.0],
    )
    np.testing.assert_array_equal(
        combined["ycoords"],
        [32.0, 32.0, 36.0, 36.0, 32.0, 32.0, 36.0, 36.0],
    )


def test_training_preflight_recomputes_truth_forward_closure_and_rejects_tampering(
    tmp_path: Path,
):
    from ptycho.workflows import synthetic_pipeline

    resolved = _small_resolved("synthetic-lines", seed=7)
    result = _generate(resolved, tmp_path / "closure")

    synthetic_pipeline._verify_truth_forward_closure(
        manifest=result.manifest,
        resolved=resolved,
        train_path=result.train_path,
        test_path=result.test_path,
    )
    tampered = json.loads(json.dumps(result.manifest))
    tampered["truth_forward_closure"]["objects"][0]["relative_l2"] = 0.25

    with pytest.raises(ValueError, match="truth_forward_closure mismatch"):
        synthetic_pipeline._verify_truth_forward_closure(
            manifest=tampered,
            resolved=resolved,
            train_path=result.train_path,
            test_path=result.test_path,
        )


def test_truth_forward_closure_replay_uses_generation_cpu_device(
    tmp_path: Path, monkeypatch
):
    import tensorflow as tf
    from ptycho.workflows import synthetic_pipeline

    resolved = _small_resolved("synthetic-lines", seed=7)
    result = _generate(resolved, tmp_path / "closure-device")
    selected_devices = []
    real_device = tf.device

    def record_device(name):
        selected_devices.append(name)
        return real_device(name)

    monkeypatch.setattr(tf, "device", record_device)

    synthetic_pipeline._verify_truth_forward_closure(
        manifest=result.manifest,
        resolved=resolved,
        train_path=result.train_path,
        test_path=result.test_path,
    )

    assert selected_devices == ["/CPU:0", "/CPU:0"]


def test_truth_forward_closure_is_dose_aware_without_fitting_probe_gain(
    tmp_path: Path,
):
    resolved = _small_resolved(
        "synthetic-lines",
        simulation={"detector": {"photons_per_pattern": 1.0e6}},
    )

    result = _generate(resolved, tmp_path / "low-dose-closure")

    closure = result.manifest["truth_forward_closure"]
    assert closure["passed"] is True
    assert all(
        item["relative_l2_limit"] > 0.005 for item in closure["objects"]
    )

    from ptycho.simulation.flat_acquisition import _truth_forward_closure

    payloads = {
        split: _load_arrays(path)
        for split, path in (("train", result.train_path), ("test", result.test_path))
    }
    payloads["train"]["probe_simulated"] *= 2.0
    source = _load_arrays(result.source_path)
    wrong_gain = _truth_forward_closure(
        payloads,
        base_seed=resolved.simulation.train.seed,
        measurement_domain=resolved.simulation.measurement_domain,
        photons_per_pattern={"train": 1.0e6, "test": 1.0e6},
        object_banks={
            "train": source["trainObjectGuess"],
            "test": source["testObjectGuess"],
        },
        diffractions_per_object={"train": 2, "test": 2},
        patch_amplitude_normalization={"train": "none", "test": "none"},
    )
    assert wrong_gain["passed"] is False


def test_truth_forward_closure_binds_frame_truth_to_source_object_bank(
    tmp_path: Path,
):
    from ptycho.workflows import synthetic_pipeline

    resolved = _small_resolved("synthetic-lines", seed=13)
    result = _generate(resolved, tmp_path / "truth-bank-closure")
    source = _load_arrays(result.source_path)
    source["trainObjectGuess"] = source["trainObjectGuess"] * np.complex64(1.1)
    np.savez(result.source_path, **source)

    with pytest.raises(ValueError, match="source_npz_sha256 mismatch"):
        synthetic_pipeline._verify_truth_forward_closure(
            manifest=result.manifest,
            resolved=resolved,
            train_path=result.train_path,
            test_path=result.test_path,
        )


def test_count_contract_emits_count_intensity_at_the_requested_dose(tmp_path: Path):
    resolved = _small_resolved("cnn-lines-ci")
    nphotons = resolved.simulation.train.detector.photons_per_pattern

    result = _generate(resolved, tmp_path / "ci")

    train = _load_arrays(result.train_path)
    counts = train["diff3d"]
    assert counts.dtype == np.float32
    assert np.all(counts >= 0)
    assert np.isfinite(counts).all()
    # S is derived so the TRAIN split averages exactly nphotons per pattern.
    mean_total = float(counts.astype(np.float64).sum(axis=(1, 2)).mean())
    assert mean_total == pytest.approx(nphotons, rel=1e-5)


def test_count_contract_records_measurement_identity_in_the_manifest(tmp_path: Path):
    result = _generate(_small_resolved("cnn-lines-ci"), tmp_path / "ci")

    assert result.manifest["measurement_identity"] == {
        "measurement_domain": "count_intensity",
        "scale_contract_version": "ci_intensity_v2",
    }
    for split in ("train", "test"):
        record = result.manifest["splits"][split]["measurement_identity"]
        assert record["measurement_domain"] == "count_intensity"
        assert record["scale_contract_version"] == "ci_intensity_v2"


def test_count_contract_stores_the_physical_probe_under_a_separate_digest(
    tmp_path: Path,
):
    from ptycho.simulation.identity import array_sha256

    result = _generate(_small_resolved("cnn-lines-ci"), tmp_path / "ci")

    probe_record = result.manifest["probe"]
    scale = probe_record["count_amplitude_scale"]["value"]
    assert scale > 1.0

    train = _load_arrays(result.train_path)
    stored = train["probeGuess"]

    # transformed_probe_sha256 keeps its meaning: the transform-pipeline output.
    assert probe_record["transformed_probe_sha256"] == AMPLITUDE_BASELINE["probe"]
    # The physical probe is published under its own field.
    assert probe_record["physical_probe_sha256"] == array_sha256(stored)
    assert probe_record["physical_probe_sha256"] != (
        probe_record["transformed_probe_sha256"]
    )


def test_count_contract_shares_one_train_derived_scale_across_both_splits(
    tmp_path: Path,
):
    result = _generate(_small_resolved("cnn-lines-ci"), tmp_path / "ci")

    train = _load_arrays(result.train_path)
    test = _load_arrays(result.test_path)
    source = _load_arrays(result.source_path)

    np.testing.assert_array_equal(train["probeGuess"], test["probeGuess"])
    np.testing.assert_array_equal(train["probeGuess"], source["probeGuess"])
    assert result.manifest["probe"]["count_amplitude_scale"]["split"] == "train"


def test_count_contract_generation_is_deterministic(tmp_path: Path):
    from ptycho.simulation.identity import array_sha256

    digests = []
    for attempt in ("first", "second"):
        result = _generate(
            _small_resolved("cnn-lines-ci"), tmp_path / attempt
        )
        arrays = _load_arrays(result.train_path)
        digests.append(
            (array_sha256(arrays["diff3d"]), array_sha256(arrays["probeGuess"]))
        )

    assert digests[0] == digests[1]


def test_count_amplitude_scale_matches_the_torch_reference_helper():
    """The numpy derivation must equal ptycho_torch's, which the CI stack uses."""

    from ptycho.simulation.flat_acquisition import derive_count_amplitude_scale
    from ptycho_torch.helper import derive_intensity_scale_from_amplitudes

    rng = np.random.default_rng(0)
    amplitudes = rng.random((7, 16, 16)).astype(np.float32)

    expected = float(
        derive_intensity_scale_from_amplitudes(amplitudes, 1e9).item()
    )
    assert derive_count_amplitude_scale(amplitudes, 1e9) == pytest.approx(
        expected, rel=1e-6
    )


def test_count_contract_datasets_pass_pipeline_manifest_verification(tmp_path: Path):
    """The training stage's manifest/probe lineage checks must accept counts."""

    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = _small_resolved("cnn-lines-ci")
    result = _generate(resolved, tmp_path / "ci")

    manifest = pipeline._load_matching_dataset_manifest(
        result.manifest_path, resolved
    )
    for split in ("train", "test"):
        pipeline._verify_split_artifact(
            getattr(result, f"{split}_path"),
            manifest=manifest,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split=split,
        )
    truth = pipeline._load_verified_source_truth(
        result.source_path,
        manifest=manifest,
        manifest_path=result.manifest_path,
        resolved=resolved,
    )
    assert truth.shape == (392, 392)


def test_dead_leaves_datasets_pass_pipeline_manifest_verification(tmp_path: Path):
    """Manifest validation must derive producer identity from the object registry."""

    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = _small_dead_leaves_raster_workflow()
    result = _generate(resolved, tmp_path / "dead-leaves")

    manifest = pipeline._load_matching_dataset_manifest(
        result.manifest_path, resolved
    )

    assert manifest["schema_version"] == "flat-acquisition-manifest-v3"
    assert manifest["object"]["producer_symbols"] == [
        "ptycho_torch.datagen.objects.create_dead_leaves",
        "ptycho.simulation.object_producers.fixed_dead_leaves_phase",
    ]
    assert manifest["morphology_attestation"]["applicable"] is True
    assert manifest["morphology_attestation"]["role"] == (
        "provenance_diagnostic_not_quality_gate"
    )
    for split in ("train", "test"):
        pipeline._verify_split_artifact(
            getattr(result, f"{split}_path"),
            manifest=manifest,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split=split,
        )
    truth = pipeline._load_verified_source_truth(
        result.source_path,
        manifest=manifest,
        manifest_path=result.manifest_path,
        resolved=resolved,
    )
    assert truth.shape == (392, 392)


def test_dead_leaves_v3_rejects_rng_and_morphology_tampering(tmp_path: Path):
    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = _small_dead_leaves_raster_workflow()
    result = _generate(resolved, tmp_path / "dead-leaves")

    rng_tampered = json.loads(json.dumps(result.manifest))
    rng_tampered["objects"]["train"][0]["rng_identity"]["streams"]["numeric"][
        "seed_sequence"
    ]["spawn_key"] = [99]
    tampered_manifest_path = result.manifest_path.with_name("rng-tampered.json")
    tampered_manifest_path.write_text(
        json.dumps(rng_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rng_identity mismatch"):
        pipeline._load_matching_dataset_manifest(tampered_manifest_path, resolved)

    morphology_tampered = json.loads(json.dumps(result.manifest))
    morphology_tampered["morphology_attestation"]["descriptors"]["train"][0][
        "coefficient_of_variation"
    ] += 0.1
    with pytest.raises(ValueError, match="morphology_attestation mismatch"):
        pipeline._load_verified_source_truth(
            result.source_path,
            manifest=morphology_tampered,
            manifest_path=result.manifest_path,
            resolved=resolved,
        )


def _downgrade_flat_manifest_v3_to_v2(
    manifest: dict[str, object],
) -> dict[str, object]:
    from ptycho.simulation.identity import canonical_sha256

    manifest = json.loads(json.dumps(manifest))
    manifest["schema_version"] = "flat-acquisition-manifest-v2"
    manifest.pop("morphology_attestation")

    def remove_v3_attestations(record):
        record.pop("rng_identity")
        record.pop("phase_identity")

    for record in [manifest["object"]] + [
        item for split in ("train", "test") for item in manifest["objects"][split]
    ]:
        remove_v3_attestations(record)
    for split in ("train", "test"):
        split_record = manifest["splits"][split]
        object_identity = split_record["split_recipe_identity"]["object_identity"]
        identity_objects = object_identity.get("objects")
        if identity_objects is None:
            remove_v3_attestations(object_identity)
        else:
            for record in identity_objects:
                remove_v3_attestations(record)
        recipe_digest = canonical_sha256(split_record["split_recipe_identity"])
        split_record["split_recipe_sha256"] = recipe_digest
        split_record["dataset_recipe_sha256"] = recipe_digest
        split_record["dataset_identity"]["split_recipe_sha256"] = recipe_digest
        split_record["dataset_sha256"] = canonical_sha256(
            split_record["dataset_identity"]
        )
    return manifest


def test_manifest_v2_rejects_dead_leaves_v2_without_attestations(tmp_path: Path):
    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = _small_dead_leaves_raster_workflow()
    result = _generate(resolved, tmp_path / "dead-leaves-v2")
    manifest = _downgrade_flat_manifest_v3_to_v2(result.manifest)
    v2_manifest_path = result.manifest_path.with_name("manifest-v2.json")
    v2_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="dead-leaves-object-v2 requires flat-acquisition-manifest-v3",
    ):
        pipeline._load_matching_dataset_manifest(v2_manifest_path, resolved)


@pytest.mark.parametrize(
    ("object_kind", "object_recipe"),
    [
        ("lines", "lines-object-v1"),
        ("dead_leaves", "dead-leaves-object-v1"),
    ],
)
def test_manifest_v2_remains_readable_for_legacy_object_recipes(
    tmp_path: Path,
    object_kind: str,
    object_recipe: str,
):
    from ptycho.workflows import synthetic_pipeline as pipeline

    if object_kind == "dead_leaves":
        resolved = _small_dead_leaves_raster_workflow(object_recipe=object_recipe)
    else:
        resolved = _small_resolved(
            "cnn-lines-ci",
            simulation={
                "object": {"kind": object_kind},
                "object_recipe": object_recipe,
            },
        )
    result = _generate(resolved, tmp_path / object_recipe)
    manifest = _downgrade_flat_manifest_v3_to_v2(result.manifest)
    v2_manifest_path = result.manifest_path.with_name("manifest-v2.json")
    v2_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = pipeline._load_matching_dataset_manifest(v2_manifest_path, resolved)
    for split in ("train", "test"):
        pipeline._verify_split_artifact(
            getattr(result, f"{split}_path"),
            manifest=loaded,
            manifest_path=v2_manifest_path,
            resolved=resolved,
            split=split,
        )


def test_validate_split_manifest_record_rejects_each_tampered_field(tmp_path: Path):
    from ptycho.simulation.flat_acquisition import validate_split_manifest_record
    from ptycho.simulation.identity import array_sha256, canonical_sha256, file_sha256

    path = tmp_path / "test.npz"
    arrays = {"diff3d": np.ones((2, 8, 8), dtype=np.float32)}
    np.savez(path, **arrays)
    hashes = {name: array_sha256(value) for name, value in arrays.items()}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    dtypes = {name: value.dtype.name for name, value in arrays.items()}
    split_recipe_sha256 = "a" * 64
    dataset_identity = {
        "split_recipe_sha256": split_recipe_sha256,
        "array_sha256": hashes,
        "shapes": shapes,
        "dtypes": dtypes,
    }

    def make_record():
        return {
            "npz_sha256": file_sha256(path),
            "array_sha256": hashes,
            "shapes": shapes,
            "dtypes": dtypes,
            "dataset_identity": dataset_identity,
            "dataset_sha256": canonical_sha256(dataset_identity),
        }

    returned = validate_split_manifest_record(
        path, make_record(), split="test", split_recipe_sha256=split_recipe_sha256
    )
    assert returned[0] == hashes

    for field, tamper in (
        ("npz_sha256", lambda r: r.update(npz_sha256="0" * 64)),
        ("array_sha256", lambda r: r.update(array_sha256={"diff3d": "0" * 64})),
        ("shapes", lambda r: r.update(shapes={"diff3d": [9, 9, 9]})),
        ("dtypes", lambda r: r.update(dtypes={"diff3d": "int64"})),
        ("dataset_identity", lambda r: r.update(dataset_identity={})),
        ("dataset_sha256", lambda r: r.update(dataset_sha256="0" * 64)),
    ):
        record = make_record()
        tamper(record)
        with pytest.raises(ValueError, match=field):
            validate_split_manifest_record(
                path, record, split="test", split_recipe_sha256=split_recipe_sha256
            )


def test_split_verification_binds_npz_object_scale_to_manifest(tmp_path: Path):
    from ptycho.simulation.identity import (
        array_sha256,
        canonical_sha256,
        file_sha256,
    )
    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "N": 64,
                "seed": 5,
                "train_patterns": 4,
                "test_patterns": 4,
                "object": {
                    "patch_amplitude_normalization": "mean_patch_max"
                },
                "scan": {"position_layout": "fixed_pitch_raster"},
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 1,
            },
            "inference": {
                "reconstruction_method": "tiled",
                "patch_weighting": "uniform",
                "varpro_scaling": False,
            },
        }
    )
    result = _generate(resolved, tmp_path / "normalized")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["scan_geometry"]["coordinate_frame"] == (
        "legacy_translation"
    )
    with np.load(result.test_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["object_amplitude_scale"] *= np.float64(2.0)
    np.savez(result.test_path, **arrays)
    record = manifest["splits"]["test"]
    hashes = {name: array_sha256(value) for name, value in arrays.items()}
    record["array_sha256"] = hashes
    record["npz_sha256"] = file_sha256(result.test_path)
    record["dataset_identity"] = {
        "split_recipe_sha256": record["split_recipe_sha256"],
        "array_sha256": hashes,
        "shapes": record["shapes"],
        "dtypes": record["dtypes"],
    }
    record["dataset_sha256"] = canonical_sha256(record["dataset_identity"])

    with pytest.raises(ValueError, match="disagrees with the manifest"):
        pipeline._verify_split_artifact(
            result.test_path,
            manifest=manifest,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split="test",
        )


def test_amplitude_datasets_still_pass_pipeline_manifest_verification(tmp_path: Path):
    from ptycho.workflows import synthetic_pipeline as pipeline

    resolved = _small_resolved("synthetic-lines")
    result = _generate(resolved, tmp_path / "amp")

    manifest = pipeline._load_matching_dataset_manifest(
        result.manifest_path, resolved
    )
    pipeline._verify_split_artifact(
        result.train_path,
        manifest=manifest,
        manifest_path=result.manifest_path,
        resolved=resolved,
        split="train",
    )
    assert pipeline._load_verified_source_truth(
        result.source_path,
        manifest=manifest,
        manifest_path=result.manifest_path,
        resolved=resolved,
    ).shape == (392, 392)


def test_count_dataset_feeds_the_reconstruct_stage_ci_loader(tmp_path: Path):
    """The reconstruct stage reads the NPZ through PtychoDataset, which treats
    the stored diffraction array directly as measured_intensity under CI."""

    from ptycho.workflows.synthetic_config import materialize_data_config
    from ptycho_torch.config_params import ModelConfig as TorchModelConfig
    from ptycho_torch.dataloader import PtychoDataset

    resolved = _small_resolved("cnn-lines-ci")
    result = _generate(resolved, tmp_path / "ci")

    data_config = materialize_data_config(resolved)
    model_config = TorchModelConfig(
        **{
            item.name: getattr(resolved.model, item.name)
            for item in __import__("dataclasses").fields(TorchModelConfig)
        }
    )

    import shutil

    split_dir = tmp_path / "held_out"
    split_dir.mkdir()
    shutil.copy2(result.test_path, split_dir / "test.npz")

    dataset = PtychoDataset(
        str(split_dir),
        model_config,
        data_config,
        data_dir=str(tmp_path / "memmap"),
        remake_map=True,
    )

    assert dataset.ci_contract_active is True
    statistics = dataset.get_ci_statistics()
    assert set(statistics) == {"rms_input_scale", "mean_measured_intensity"}

    stored = _load_arrays(result.test_path)["diff3d"]
    measured = np.asarray(dataset.mmap_ptycho["measured_intensity"][: stored.shape[0]])
    np.testing.assert_allclose(
        np.sort(measured.reshape(-1))[-64:],
        np.sort(stored.reshape(-1))[-64:],
        rtol=1e-5,
    )
    assert float(measured.mean()) > 1.0


# --- Raster scan position layout --------------------------------------------


def test_raster_positions_span_the_buffered_extent_in_row_major_order():
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    x, y = raster_scan_positions(n_positions=9, height=100, width=200, buffer=10)

    assert x.dtype == np.float64 and y.dtype == np.float64
    assert x.shape == y.shape == (9,)
    # Span-filling: first and last sample sit exactly on the buffered edges.
    assert x.min() == pytest.approx(10.0)
    assert x.max() == pytest.approx(190.0)
    assert y.min() == pytest.approx(10.0)
    assert y.max() == pytest.approx(90.0)
    # Row-major: y is the slow axis, x the fast axis.
    np.testing.assert_allclose(x, [10.0, 100.0, 190.0] * 3)
    np.testing.assert_allclose(
        y, [10.0] * 3 + [50.0] * 3 + [90.0] * 3
    )


def test_raster_positions_are_deterministic_and_consume_no_rng():
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    first = raster_scan_positions(n_positions=16, height=64, width=64, buffer=4)
    second = raster_scan_positions(n_positions=16, height=64, width=64, buffer=4)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_raster_positions_apply_the_same_buffer_clamp_as_the_legacy_leaf():
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    # buffer 40 exceeds min(h, w)/2 - 1 = 9 for a 20x20 canvas.
    x, _ = raster_scan_positions(n_positions=4, height=20, width=20, buffer=40)

    assert x.min() == pytest.approx(9.0)
    assert x.max() == pytest.approx(11.0)


@pytest.mark.parametrize("n_positions", [2, 5, 10, 4488, 4490])
def test_raster_positions_reject_non_square_counts(n_positions):
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    with pytest.raises(ValueError, match="perfect-square"):
        raster_scan_positions(
            n_positions=n_positions, height=64, width=64, buffer=4
        )


def test_raster_positions_reject_a_degenerate_single_point_grid():
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    with pytest.raises(ValueError, match="at least 2"):
        raster_scan_positions(n_positions=1, height=64, width=64, buffer=4)


def test_fixed_pitch_raster_reproduces_historical_train_and_test_geometry():
    from ptycho.simulation.flat_acquisition import fixed_pitch_raster_positions

    train_x, train_y = fixed_pitch_raster_positions(
        n_positions=4489,
        height=392,
        width=392,
        patch_size=128,
        pitch=4.0,
    )
    test_x, test_y = fixed_pitch_raster_positions(
        n_positions=729,
        height=392,
        width=392,
        patch_size=128,
        pitch=10.0,
    )

    # RawData's translation coordinates are measured from the padded-array
    # origin.  For even N, N / 2 maps the first scan to source slice [0:N];
    # the corresponding geometric pixel-center coordinate is N / 2 - 0.5.
    assert train_x[0] == train_y[0] == pytest.approx(64.0)
    assert train_x[-1] == train_y[-1] == pytest.approx(328.0)
    assert test_x[0] == test_y[0] == pytest.approx(64.0)
    assert test_x[-1] == test_y[-1] == pytest.approx(324.0)
    np.testing.assert_allclose(np.diff(train_x[:67]), 4.0)
    np.testing.assert_allclose(np.diff(test_x[:27]), 10.0)
    np.testing.assert_allclose(train_y[:67], 64.0)
    np.testing.assert_allclose(test_y[:27], 64.0)


def test_fixed_pitch_raster_coordinates_extract_exact_source_patches():
    from ptycho.raw_data import get_image_patches, get_relative_coords
    from ptycho.simulation.flat_acquisition import fixed_pitch_raster_positions

    N = 64
    source = (
        np.arange(80 * 80, dtype=np.float32).reshape(80, 80)
        + 1j * np.arange(80 * 80, dtype=np.float32).reshape(80, 80)[::-1]
    ).astype(np.complex64)
    xcoords, ycoords = fixed_pitch_raster_positions(
        n_positions=4,
        height=80,
        width=80,
        patch_size=N,
        pitch=4.0,
    )
    coordinates = np.zeros((4, 1, 2, 1), dtype=np.float64)
    coordinates[:, 0, 0, 0] = xcoords
    coordinates[:, 0, 1, 0] = ycoords
    global_offsets, local_offsets = get_relative_coords(coordinates)

    actual = np.asarray(
        get_image_patches(
            source,
            global_offsets,
            local_offsets,
            N=N,
            gridsize=1,
        )
    )[..., 0]
    expected = np.stack(
        [
            source[0:N, 0:N],
            source[0:N, 4 : 4 + N],
            source[4 : 4 + N, 0:N],
            source[4 : 4 + N, 4 : 4 + N],
        ]
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_positions": 8}, "perfect-square"),
        ({"pitch": 0.0}, "positive"),
        ({"n_positions": 16, "pitch": 20.0}, "does not fit"),
    ],
)
def test_fixed_pitch_raster_rejects_invalid_geometry(kwargs, message):
    from ptycho.simulation.flat_acquisition import fixed_pitch_raster_positions

    values = {
        "n_positions": 9,
        "height": 20,
        "width": 20,
        "patch_size": 8,
        "pitch": 2.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        fixed_pitch_raster_positions(**values)


def test_mean_patch_max_normalization_pools_all_objects_and_frames():
    from ptycho.simulation.flat_acquisition import (
        _apply_patch_amplitude_normalization,
    )

    maxima = np.asarray([1, 2, 3, 4, 2, 4, 6, 8], dtype=np.float32)
    truth = np.stack(
        [np.full((2, 2), value * 1j, dtype=np.complex64) for value in maxima]
    )
    diffraction = np.stack(
        [np.full((2, 2), 2 * value, dtype=np.float32) for value in maxima]
    )
    payload = {
        "Y": truth.copy(),
        "diff3d": diffraction.copy(),
        "object_index": np.repeat(np.arange(2), 4),
    }

    normalization = _apply_patch_amplitude_normalization(
        payload,
        method="mean_patch_max",
    )

    # Per-frame maxima are [1, 2, 3, 4] and [2, 4, 6, 8].
    assert normalization == pytest.approx(3.75)
    assert np.asarray(payload["object_amplitude_scale"]).shape == ()
    assert float(payload["object_amplitude_scale"]) == pytest.approx(3.75)
    np.testing.assert_allclose(payload["Y"], truth / 3.75)
    np.testing.assert_allclose(payload["diff3d"], diffraction / 3.75)


def test_nongrid_accepts_explicit_coordinates_and_skips_the_rng():
    from ptycho import nongrid_simulation

    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False), training_groups=4
    )
    xs = np.array([4.0, 6.0, 4.0, 6.0])
    ys = np.array([4.0, 4.0, 6.0, 6.0])

    raw = nongrid_simulation._generate_simulated_data_legacy_params(
        config,
        np.ones((16, 16), dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
        4,
        detector_seed=11,
        coordinates=(xs, ys),
    )

    np.testing.assert_array_equal(np.asarray(raw.xcoords), xs)
    np.testing.assert_array_equal(np.asarray(raw.ycoords), ys)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"random_seed": 3},
        {"coordinate_rng": np.random.default_rng(4)},
    ],
    ids=("random-seed", "coordinate-rng"),
)
def test_nongrid_rejects_coordinates_combined_with_a_position_rng(kwargs):
    from ptycho import nongrid_simulation

    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False), training_groups=4
    )

    with pytest.raises(ValueError, match="coordinates"):
        nongrid_simulation._generate_simulated_data_legacy_params(
            config,
            np.ones((16, 16), dtype=np.complex64),
            np.ones((8, 8), dtype=np.complex64),
            4,
            coordinates=(np.zeros(4), np.zeros(4)),
            **kwargs,
        )


def test_nongrid_public_adapter_forwards_coordinates(monkeypatch):
    from ptycho import nongrid_simulation

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        nongrid_simulation,
        "_generate_simulated_data_legacy_params",
        lambda **kwargs: captured.update(kwargs),
    )
    config = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False), training_groups=4
    )
    coordinates = (np.zeros(4), np.ones(4))

    nongrid_simulation.generate_simulated_data(
        config,
        np.ones((16, 16), dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
        4,
        return_patches=False,
        coordinates=coordinates,
    )

    assert captured["coordinates"] is coordinates


def _raster_resolved(profile="synthetic-lines", patterns=9, test_patterns=4):
    return resolve_synthetic_workflow(
        profile=profile,
        file_values={
            "simulation": {
                "N": 64,
                "seed": 5,
                "train_patterns": patterns,
                "test_patterns": test_patterns,
                "scan": {"position_layout": "raster"},
            },
            "training": {
                "train_raw_selection": min(patterns, test_patterns),
                "training_groups": min(patterns, test_patterns),
                "validation_groups": min(patterns, test_patterns),
                "neighbor_count": 1,
            },
        },
    )


def test_raster_flat_acquisition_writes_span_filling_coordinates(tmp_path: Path):
    from ptycho.simulation.flat_acquisition import raster_scan_positions

    resolved = _raster_resolved()
    result = _generate(resolved, tmp_path / "raster")

    train = _load_arrays(result.train_path)
    expected_x, expected_y = raster_scan_positions(
        n_positions=9,
        height=392,
        width=392,
        buffer=resolved.simulation.train.scan.buffer,
    )
    np.testing.assert_allclose(train["xcoords"], expected_x)
    np.testing.assert_allclose(train["ycoords"], expected_y)


def test_raster_manifest_records_layout_and_realized_pitch(tmp_path: Path):
    resolved = _raster_resolved()
    result = _generate(resolved, tmp_path / "raster")

    geometry = result.manifest["scan_geometry"]
    assert geometry["position_layout"] == "raster"
    assert geometry["train"]["side"] == 3
    assert geometry["test"]["side"] == 2
    # Span-filling pitch over the buffered 392-wide extent with buffer 64.
    assert geometry["train"]["pitch_x"] == pytest.approx((392 - 128) / 2)
    assert geometry["train"]["pitch_y"] == pytest.approx((392 - 128) / 2)


def test_uniform_random_manifest_records_the_default_layout(tmp_path: Path):
    result = _generate(_small_resolved("synthetic-lines"), tmp_path / "amp")

    assert result.manifest["scan_geometry"] == {
        "position_layout": "uniform_random"
    }


def test_pre_frame_order_default_dataset_manifest_remains_reusable(tmp_path: Path):
    from ptycho.workflows import synthetic_pipeline

    resolved = _small_resolved("synthetic-lines")
    result = _generate(resolved, tmp_path / "amp")
    historical = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    historical["simulation"].pop("frame_order_recipe")
    result.manifest_path.write_text(json.dumps(historical), encoding="utf-8")

    loaded = synthetic_pipeline._load_matching_dataset_manifest(
        result.manifest_path,
        resolved,
    )
    for split, path in (("train", result.train_path), ("test", result.test_path)):
        synthetic_pipeline._verify_split_artifact(
            path,
            manifest=loaded,
            manifest_path=result.manifest_path,
            resolved=resolved,
            split=split,
        )

    assert "frame_order_recipe" not in loaded["simulation"]


def test_raster_and_uniform_random_datasets_have_distinct_identities(
    tmp_path: Path,
):
    raster = _generate(_raster_resolved(), tmp_path / "raster")
    uniform = _generate(
        _small_resolved("synthetic-lines", seed=5), tmp_path / "uniform"
    )

    assert (
        raster.manifest["splits"]["train"]["simulation_config_sha256"]
        != uniform.manifest["splits"]["train"]["simulation_config_sha256"]
    )


def test_flat_service_rejects_a_split_position_layout_mismatch(tmp_path: Path):
    from ptycho.simulation import flat_acquisition

    resolved = _raster_resolved()
    simulation = resolved.simulation
    mismatched = replace(
        simulation,
        test=replace(
            simulation.test,
            scan=replace(simulation.test.scan, position_layout="uniform_random"),
        ),
    )
    resolved = replace(resolved, simulation=mismatched)

    with pytest.raises(ValueError, match="scan.position_layout"):
        flat_acquisition.generate_flat_acquisitions(resolved, tmp_path / "bad")


def test_coordinate_major_frame_recipe_requires_a_raster_layout():
    from ptycho.simulation.flat_acquisition import (
        validate_flat_acquisition_workflow,
    )

    resolved = resolve_synthetic_workflow(
        profile="cnn-lines-ci",
        file_values={
            "simulation": {
                "frame_order_recipe": "coordinate-major-interleaved-v1",
                "train_patterns": 4,
                "test_patterns": 4,
                "scan": {"position_layout": "fixed_pitch_raster"},
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 1,
            },
        }
    )
    simulation = replace(
        resolved.simulation,
        train=replace(
            resolved.simulation.train,
            scan=replace(
                resolved.simulation.train.scan,
                position_layout="uniform_random",
            ),
        ),
        test=replace(
            resolved.simulation.test,
            scan=replace(
                resolved.simulation.test.scan,
                position_layout="uniform_random",
            ),
        ),
    )
    resolved = replace(resolved, simulation=simulation)

    with pytest.raises(
        ValueError,
        match="coordinate-major-interleaved-v1.*raster",
    ):
        validate_flat_acquisition_workflow(resolved)


def test_raster_coordinate_order_rejects_unknown_recipe():
    from ptycho.simulation.flat_acquisition import ordered_raster_coordinates

    coordinates = (
        np.asarray([1.0, 2.0, 1.0, 2.0]),
        np.asarray([3.0, 3.0, 4.0, 4.0]),
    )

    with pytest.raises(ValueError, match="unsupported frame_order_recipe"):
        ordered_raster_coordinates(
            coordinates,
            frame_order_recipe="coordinate-major-v2",
        )
