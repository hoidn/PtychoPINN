"""Focused tests for the natural-patch fixed-probe CDI dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.studies.cdi_natural_patch_dataset import (
    DEFAULT_PATCH_SIZE,
    NaturalImageRecord,
    ObjectEncodingContract,
    ProbeBundle,
    SimulationContract,
    assign_source_splits,
    build_dataset,
    encode_object_patch,
    forward_amplitude,
    post_audit,
)


PATCH_SIZE = 64  # smaller patch for tractable tests


def _make_test_records(n_images: int, height: int = 96, width: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    records = []
    for idx in range(n_images):
        pixels = rng.uniform(0.0, 1.0, size=(height, width)).astype(np.float32)
        records.append(
            NaturalImageRecord(
                image_id=f"img_{idx:02d}",
                pixels=pixels,
                height=height,
                width=width,
            )
        )
    return records


def _make_test_probe(n: int = PATCH_SIZE) -> ProbeBundle:
    yy, xx = np.indices((n, n))
    cy = (n - 1) / 2.0
    cx = (n - 1) / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    amp = np.exp(-r2 / (2.0 * (n / 4.0) ** 2))
    phase = 0.05 * r2 / (n * n)
    probe = (amp * np.exp(1j * phase)).astype(np.complex64)
    return ProbeBundle(
        probe=probe,
        source_path="test/synthetic_probe",
        source_shape=(n, n),
        target_N=n,
        smoothing_sigma=0.0,
        scale_mode="pad_extrapolate",
        pipeline_spec="pad_extrapolate:" + str(n),
    )


def test_encode_object_patch_amplitude_phase_ranges():
    patch = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    obj = encode_object_patch(patch)
    assert obj.dtype == np.complex64
    amplitudes = np.abs(obj)
    phases = np.angle(obj)
    assert pytest.approx(amplitudes[0, 0], rel=1e-5) == 0.5
    assert pytest.approx(amplitudes[0, 1], rel=1e-5) == 1.0
    assert pytest.approx(phases[0, 0], rel=1e-5) == -np.pi / 2.0
    assert pytest.approx(phases[0, 1], rel=1e-5) == np.pi / 2.0


def test_forward_amplitude_basic_shape_and_dtype():
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((PATCH_SIZE, PATCH_SIZE)) + 1j * rng.standard_normal(
        (PATCH_SIZE, PATCH_SIZE)
    )
    diff = forward_amplitude(psi.astype(np.complex64))
    assert diff.shape == (PATCH_SIZE, PATCH_SIZE)
    assert diff.dtype == np.float32
    # Parseval: ||X||^2 == sum |x|^2 (with normalization sqrt(N) in our convention)
    parseval_lhs = float(np.sum(diff.astype(np.float64) ** 2))
    parseval_rhs = float(np.sum(np.abs(psi) ** 2))
    assert parseval_lhs == pytest.approx(parseval_rhs, rel=1e-4)


def test_assign_source_splits_deterministic_and_no_overlap():
    records = _make_test_records(8)
    splits_a = assign_source_splits(records, n_train=4, n_val=2, n_test=2, seed=11)
    splits_b = assign_source_splits(records, n_train=4, n_val=2, n_test=2, seed=11)
    for split_name in ("train", "val", "test"):
        ids_a = [r.image_id for r in splits_a[split_name]]
        ids_b = [r.image_id for r in splits_b[split_name]]
        assert ids_a == ids_b
    seen = set()
    for ids in splits_a.values():
        for record in ids:
            assert record.image_id not in seen
            seen.add(record.image_id)


def test_assign_source_splits_rejects_too_few_records():
    records = _make_test_records(3)
    with pytest.raises(ValueError):
        assign_source_splits(records, n_train=2, n_val=1, n_test=1, seed=0)


def test_build_dataset_produces_expected_artifacts(tmp_path: Path):
    records = _make_test_records(8)
    probe = _make_test_probe(PATCH_SIZE)
    split_counts = {"train": 12, "val": 4, "test": 4}
    split_source_counts = {"train": 4, "val": 2, "test": 2}
    result = build_dataset(
        dataset_root=tmp_path,
        records=records,
        probe_bundle=probe,
        split_counts=split_counts,
        split_source_counts=split_source_counts,
        patch_size=PATCH_SIZE,
        total_cap=64,
        split_seed=7,
        crop_seed=13,
        dataset_id="test_dataset_v1",
    )
    for split_name, expected in split_counts.items():
        path = tmp_path / f"{split_name}.npz"
        assert path.exists()
        with np.load(path, allow_pickle=True) as data:
            assert int(data["objects"].shape[0]) == expected
            assert data["objects"].shape[1:] == (PATCH_SIZE, PATCH_SIZE)
            assert data["objects"].dtype == np.complex64
            assert data["diffraction"].shape == (expected, PATCH_SIZE, PATCH_SIZE)
            assert data["diffraction"].dtype == np.float32
            assert data["crop_coords"].shape == (expected, 4)
            assert data["source_ids"].shape == (expected,)
            assert data["patch_ids"].shape == (expected,)
    assert result.split_counts == split_counts
    # No source overlap in the recorded membership
    seen = set()
    for ids in result.source_split_membership.values():
        for image_id in ids:
            assert image_id not in seen
            seen.add(image_id)


def test_build_dataset_emits_required_manifests(tmp_path: Path):
    records = _make_test_records(8)
    probe = _make_test_probe(PATCH_SIZE)
    build_dataset(
        dataset_root=tmp_path,
        records=records,
        probe_bundle=probe,
        split_counts={"train": 8, "val": 2, "test": 2},
        split_source_counts={"train": 4, "val": 2, "test": 2},
        patch_size=PATCH_SIZE,
        total_cap=32,
        split_seed=5,
        crop_seed=7,
        dataset_id="manifest_check_v1",
    )
    for name in (
        "dataset_manifest.json",
        "source_manifest.json",
        "split_manifest.json",
        "probe_manifest.json",
        "simulation_manifest.json",
        "adapter_contract.json",
        "probe.npz",
    ):
        assert (tmp_path / name).exists(), f"missing artifact {name}"

    dataset_manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    assert dataset_manifest["dataset_id"] == "manifest_check_v1"
    assert dataset_manifest["patch_size"] == PATCH_SIZE
    assert dataset_manifest["total_patches"] == 12
    probe_manifest = json.loads((tmp_path / "probe_manifest.json").read_text())
    assert probe_manifest["target_N"] == PATCH_SIZE
    assert probe_manifest["scale_mode"] == "pad_extrapolate"
    sim_manifest = json.loads((tmp_path / "simulation_manifest.json").read_text())
    assert sim_manifest["object_encoding"]["amplitude_min"] == pytest.approx(0.5)
    assert sim_manifest["object_encoding"]["phase_min_rad"] == pytest.approx(-np.pi / 2.0)
    adapter_contract = json.loads((tmp_path / "adapter_contract.json").read_text())
    assert adapter_contract["consumer_keys"]["objects"]["dtype"] == "complex64"
    assert adapter_contract["consumer_keys"]["diffraction"]["dtype"] == "float32"


def test_build_dataset_enforces_total_cap(tmp_path: Path):
    records = _make_test_records(6)
    probe = _make_test_probe(PATCH_SIZE)
    with pytest.raises(ValueError):
        build_dataset(
            dataset_root=tmp_path,
            records=records,
            probe_bundle=probe,
            split_counts={"train": 12, "val": 6, "test": 6},
            split_source_counts={"train": 3, "val": 2, "test": 1},
            patch_size=PATCH_SIZE,
            total_cap=20,  # 24 requested > 20 cap
            split_seed=0,
            crop_seed=0,
        )


def test_build_dataset_rejects_split_source_overlap(tmp_path: Path):
    records = _make_test_records(3)
    probe = _make_test_probe(PATCH_SIZE)
    with pytest.raises(ValueError):
        build_dataset(
            dataset_root=tmp_path,
            records=records,
            probe_bundle=probe,
            split_counts={"train": 4, "val": 2, "test": 2},
            split_source_counts={"train": 2, "val": 2, "test": 2},  # 6 > 3 records
            patch_size=PATCH_SIZE,
            total_cap=20,
            split_seed=0,
            crop_seed=0,
        )


def test_post_audit_passes_after_build(tmp_path: Path):
    records = _make_test_records(8)
    probe = _make_test_probe(PATCH_SIZE)
    split_counts = {"train": 8, "val": 4, "test": 4}
    build_dataset(
        dataset_root=tmp_path,
        records=records,
        probe_bundle=probe,
        split_counts=split_counts,
        split_source_counts={"train": 4, "val": 2, "test": 2},
        patch_size=PATCH_SIZE,
        total_cap=32,
        split_seed=3,
        crop_seed=9,
    )
    audit = post_audit(
        dataset_root=tmp_path,
        expected_split_counts=split_counts,
        total_cap=32,
    )
    assert audit["total_objects"] == 16
    assert audit["split_counts"] == split_counts
    assert audit["no_source_overlap"] is True
    assert audit["manifests_present"] is True


def test_adapter_contract_payload_consumable_for_grouped_npz(tmp_path: Path):
    """Smoke-prove the adapter contract: a consumer can construct a grouped
    CDI payload from the emitted dataset without changing model semantics."""
    records = _make_test_records(8)
    probe = _make_test_probe(PATCH_SIZE)
    build_dataset(
        dataset_root=tmp_path,
        records=records,
        probe_bundle=probe,
        split_counts={"train": 4, "val": 2, "test": 2},
        split_source_counts={"train": 4, "val": 2, "test": 2},
        patch_size=PATCH_SIZE,
        total_cap=16,
        split_seed=21,
        crop_seed=33,
    )
    with np.load(tmp_path / "train.npz", allow_pickle=True) as data:
        objects = np.asarray(data["objects"])
        diffractions = np.asarray(data["diffraction"])
    with np.load(tmp_path / "probe.npz", allow_pickle=True) as data:
        loaded_probe = np.asarray(data["probeGuess"])
    n_groups = int(objects.shape[0])
    grouped = {
        "X_full": diffractions[:, None, :, :],  # (N, 1, H, W)
        "Y": objects[:, None, :, :],            # (N, 1, H, W)
        "coords_relative": np.zeros((n_groups, 1, 2), dtype=np.float32),
        "coords_offsets": np.zeros((n_groups, 1, 2), dtype=np.float32),
        "probeGuess": loaded_probe,
    }
    assert grouped["X_full"].shape == (n_groups, 1, PATCH_SIZE, PATCH_SIZE)
    assert grouped["Y"].shape == (n_groups, 1, PATCH_SIZE, PATCH_SIZE)
    assert grouped["probeGuess"].shape == (PATCH_SIZE, PATCH_SIZE)
    # the adapter must NOT regenerate diffraction with a different operator;
    # confirm the emitted diffraction matches the documented forward model
    expected = forward_amplitude(loaded_probe.astype(np.complex64) * objects[0])
    np.testing.assert_allclose(diffractions[0], expected, rtol=1e-4, atol=1e-4)


def test_simulation_contract_dataclass_defaults():
    """Document the contract dataclass defaults so the schema cannot drift silently."""
    sim = SimulationContract()
    assert sim.forward_model == "single_shot_cdi_fraunhofer"
    assert sim.dtype_object == "complex64"
    assert sim.dtype_diffraction == "float32"
    enc = ObjectEncodingContract()
    assert enc.amplitude_min == pytest.approx(0.5)
    assert enc.amplitude_max == pytest.approx(1.0)
    assert enc.phase_min_rad == pytest.approx(-np.pi / 2.0)
    assert enc.phase_max_rad == pytest.approx(np.pi / 2.0)


def test_default_patch_size_is_128():
    assert DEFAULT_PATCH_SIZE == 128
