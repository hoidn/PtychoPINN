"""Dataset-contract tests for the BRDT smoke/preflight dataset.

Covers manifest-key stability, train-only normalization statistics,
normalize/unnormalize round-trip, deterministic split with disjoint
object seeds, geometry-validation against the locked operator authority,
the hard guard against feeding normalized q to the operator, and the
phantom-family roster.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest

from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.phantoms import (
    generate_refractive_index,
    overlapping_ellipses,
    sparse_inclusions,
    soft_blobs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_MODULE = "scripts.studies.born_rytov_dt.generate_brdt_dataset"


def _run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = ["python", "-m", GENERATOR_MODULE, *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# Manifest contract
# ----------------------------------------------------------------------
def _build_minimal_manifest(tmp_path: Path) -> dict:
    counts = dc.SplitCounts()
    seeds = dc.deterministic_object_seeds(counts, split_seed=42)
    families = dc.assign_phantom_families(counts, split_seed=42)
    q_train = np.random.default_rng(0).standard_normal((counts.train, 8, 8)) * 0.01
    stats = dc.compute_train_normalization(q_train)
    return dc.build_manifest(
        output_root=str(tmp_path),
        operator_validation_path=str(tmp_path / "operator_validation.json"),
        counts=counts,
        split_seed=42,
        object_seeds=seeds,
        families=families,
        normalization=stats,
        noise_sigma=0.001,
        measured_snr={"train_mean_db": 30.0},
        git_sha="deadbeef",
        git_dirty=False,
        generation_command="python -m scripts.studies.born_rytov_dt.generate_brdt_dataset",
        environment={"python": "3.11"},
        artifact_paths={"train": "train.h5", "val": "val.h5", "test": "test.h5"},
    )


def test_manifest_required_keys_present(tmp_path):
    manifest = _build_minimal_manifest(tmp_path)
    for key in dc.manifest_required_keys():
        assert key in manifest, f"missing top-level key {key}"


def test_manifest_locked_geometry_fields(tmp_path):
    manifest = _build_minimal_manifest(tmp_path)
    op = manifest["operator"]
    assert op["mode"] == "born"
    assert op["grid_size"] == 128
    assert op["detector_size"] == 128
    assert op["angle_count"] == 64
    assert op["wavelength_px"] == 8.0
    assert op["medium_ri"] == 1.333
    assert manifest["physical_target"]["forward_input_is_physical_q"] is True
    assert manifest["physical_target"]["model_output_space"] == "normalized_q"
    assert "unnormalize" in manifest["physical_target"]["physics_loss_rule"]


def test_manifest_serializable(tmp_path):
    manifest = _build_minimal_manifest(tmp_path)
    out_path = tmp_path / "manifest.json"
    dc.write_manifest(manifest, str(out_path))
    reloaded = json.loads(out_path.read_text())
    # Sorted keys, stable schema, lossless round-trip.
    assert reloaded["dataset_identity"]["name"] == dc.DATASET_NAME
    assert reloaded["physical_target"]["forward_input_is_physical_q"] is True


def test_dry_run_writes_manifest_skeleton_and_exact_command(tmp_path):
    output_root = tmp_path / "dryrun"
    cmd = [
        "--dry-run-manifest",
        "--output-root",
        str(output_root),
        "--split-seed",
        "123",
        "--noise-sigma",
        "0.002",
    ]
    _run_generator(*cmd)

    summary_path = output_root / "dry_run_summary.json"
    manifest_path = output_root / "dry_run_manifest.json"
    assert summary_path.exists()
    assert manifest_path.exists()

    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    expected_command = shlex.join(["python", "-m", GENERATOR_MODULE, *cmd])

    assert summary["generation_command"] == expected_command
    assert summary["manifest_skeleton_path"] == str(manifest_path)
    assert manifest["dataset_identity"]["generation_command"] == expected_command
    assert manifest["normalization"] is None
    assert manifest["noise"]["measured_snr"] is None
    assert manifest["extra"]["generation_mode"] == "dry_run_manifest"


def test_live_generation_reproducible_across_fresh_processes(tmp_path):
    output_root = tmp_path / "live"
    cmd = [
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--split-seed",
        "123",
        "--train-count",
        "2",
        "--val-count",
        "1",
        "--test-count",
        "1",
        "--noise-sigma",
        "0.002",
    ]
    expected_command = shlex.join(["python", "-m", GENERATOR_MODULE, *cmd])

    def read_state() -> tuple[dict, dict]:
        manifest = json.loads((output_root / "dataset_manifest.json").read_text())
        hashes = {
            split: _sha256(output_root / "dataset" / f"{dc.DATASET_NAME}_{split}.h5")
            for split in ("train", "val", "test")
        }
        return manifest, hashes

    _run_generator(*cmd)
    manifest_a, hashes_a = read_state()
    _run_generator(*cmd)
    manifest_b, hashes_b = read_state()

    assert manifest_a["noise"]["measured_snr"] == manifest_b["noise"]["measured_snr"]
    assert hashes_a == hashes_b
    assert manifest_a["dataset_identity"]["generation_command"] == expected_command
    assert manifest_b["dataset_identity"]["generation_command"] == expected_command


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def test_compute_train_normalization_uses_train_only():
    rng = np.random.default_rng(123)
    q_train = rng.standard_normal((16, 8, 8)) * 0.05 + 0.1
    q_val = rng.standard_normal((4, 8, 8)) * 100.0  # large outlier on val
    stats = dc.compute_train_normalization(q_train)
    # train-only stats must not be perturbed by val numbers
    assert math.isclose(stats.mean, float(q_train.mean()), abs_tol=1e-12)
    assert math.isclose(stats.std, float(q_train.std()), abs_tol=1e-12)
    assert stats.qmin <= q_train.min() + 1e-12
    assert stats.qmax >= q_train.max() - 1e-12
    # Confirm val stats are NOT what was recorded.
    assert not math.isclose(stats.std, float(q_val.std()), rel_tol=1e-3)


def test_normalize_unnormalize_round_trip():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((4, 8, 8)) * 0.2 - 0.05
    stats = dc.compute_train_normalization(q)
    qn = dc.normalize_q(q, stats)
    q_back = dc.unnormalize_q(qn, stats)
    np.testing.assert_allclose(q_back, q, atol=1e-12, rtol=0.0)


def test_normalize_handles_constant_field():
    q = np.ones((2, 4, 4))
    stats = dc.compute_train_normalization(q)
    # std is zero; safe_std falls back to 1.0 so unnormalize reproduces input.
    qn = dc.normalize_q(q, stats)
    q_back = dc.unnormalize_q(qn, stats)
    np.testing.assert_allclose(q_back, q, atol=1e-12)


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def test_split_counts_lock_to_16_4_4():
    counts = dc.SplitCounts()
    assert counts.train == 16
    assert counts.val == 4
    assert counts.test == 4
    assert counts.total == 24


def test_object_seeds_are_disjoint_and_deterministic():
    counts = dc.SplitCounts()
    seeds_a = dc.deterministic_object_seeds(counts, split_seed=42)
    seeds_b = dc.deterministic_object_seeds(counts, split_seed=42)
    assert seeds_a == seeds_b
    train, val, test = set(seeds_a["train"]), set(seeds_a["val"]), set(seeds_a["test"])
    assert len(train) == counts.train
    assert len(val) == counts.val
    assert len(test) == counts.test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_object_seeds_change_with_split_seed():
    counts = dc.SplitCounts()
    a = dc.deterministic_object_seeds(counts, split_seed=42)
    b = dc.deterministic_object_seeds(counts, split_seed=43)
    assert a != b


def test_phantom_family_assignment_balanced_and_deterministic():
    counts = dc.SplitCounts()
    a = dc.assign_phantom_families(counts, split_seed=42)
    b = dc.assign_phantom_families(counts, split_seed=42)
    assert a == b
    for split, n in (("train", counts.train), ("val", counts.val), ("test", counts.test)):
        assert len(a[split]) == n
        assert set(a[split]).issubset(set(dc.PHANTOM_FAMILIES))


# ----------------------------------------------------------------------
# Geometry validation
# ----------------------------------------------------------------------
def test_geometry_validation_passes_for_locked_authority():
    locked = {
        "mode": "born",
        "normalize": "unitary_fft",
        "grid_size": 128,
        "detector_size": 128,
        "wavelength_px": 8.0,
        "medium_ri": 1.333,
        "angle_count": 64,
    }
    assert dc.validate_geometry_against_operator_authority(locked) == []


def test_geometry_validation_flags_mismatch():
    bad = {
        "mode": "born",
        "normalize": "odtbrain_compatible",
        "grid_size": 64,
        "detector_size": 128,
        "wavelength_px": 8.0,
        "medium_ri": 1.333,
        "angle_count": 32,
    }
    diffs = dc.validate_geometry_against_operator_authority(bad)
    keys = " ".join(diffs)
    assert "normalize" in keys
    assert "grid_size" in keys
    assert "angle_count" in keys


# ----------------------------------------------------------------------
# Operator-routing guard
# ----------------------------------------------------------------------
def test_reject_normalized_q_routing():
    dc.reject_normalized_q_to_operator("physical_q")
    with pytest.raises(ValueError, match="physical q"):
        dc.reject_normalized_q_to_operator("normalized_q")
    with pytest.raises(ValueError):
        dc.reject_normalized_q_to_operator("anything_else")


# ----------------------------------------------------------------------
# Phantom roster
# ----------------------------------------------------------------------
def test_phantom_families_exposed():
    assert "overlapping_ellipses" in dc.PHANTOM_FAMILIES
    assert "soft_blobs" in dc.PHANTOM_FAMILIES
    assert "sparse_inclusions" in dc.PHANTOM_FAMILIES
    assert len(dc.PHANTOM_FAMILIES) >= 2


@pytest.mark.parametrize(
    "family,fn",
    [
        ("overlapping_ellipses", overlapping_ellipses),
        ("soft_blobs", soft_blobs),
        ("sparse_inclusions", sparse_inclusions),
    ],
)
def test_phantom_generators_in_weak_scattering_envelope(family, fn):
    grid = 32
    n_field = fn(seed=0, grid=grid, n_m=dc.LOCKED_MEDIUM_RI)
    assert n_field.shape == (grid, grid)
    delta = n_field - dc.LOCKED_MEDIUM_RI
    # Allow some accumulation from overlapping objects but cap at small
    # multiples of delta_n_max so we stay in the weak-scattering regime.
    assert np.max(np.abs(delta)) <= 8.0 * dc.DELTA_N_MAX, (
        f"{family} produced |delta_n|={np.max(np.abs(delta))}"
    )


def test_phantom_dispatch_seed_determinism():
    a = generate_refractive_index("overlapping_ellipses", seed=11, grid=32)
    b = generate_refractive_index("overlapping_ellipses", seed=11, grid=32)
    np.testing.assert_array_equal(a, b)
    c = generate_refractive_index("overlapping_ellipses", seed=12, grid=32)
    assert not np.array_equal(a, c)


def test_q_formula_round_trip_on_constant_medium():
    # n == n_m => q == 0
    n = np.full((8, 8), dc.LOCKED_MEDIUM_RI)
    q = dc.refractive_index_to_q(n)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)


def test_locked_angles_full_view():
    angles = dc.locked_angles()
    assert angles.shape == (64,)
    assert math.isclose(angles[0], 0.0)
    # Last angle just below 2*pi
    assert angles[-1] < 2.0 * math.pi
    assert math.isclose(angles[1] - angles[0], 2.0 * math.pi / 64.0, rel_tol=1e-12)
