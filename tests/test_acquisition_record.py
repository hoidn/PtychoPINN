"""Focused contracts for the framework-neutral acquisition handoff."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
import json

import numpy as np
import pytest


def _valid_acquisition_arrays():
    return {
        "xcoords": np.arange(3, dtype=np.float64),
        "ycoords": np.arange(3, dtype=np.float64),
        "diff3d": np.ones((3, 4, 4), dtype=np.float32),
        "probeGuess": np.ones((4, 4), dtype=np.complex64),
    }


def test_decode_acquisition_reads_basic_canonical_npz(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "basic.npz"
    diffraction = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    np.savez(
        path,
        xcoords=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ycoords=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        diff3d=diffraction,
        probeGuess=np.ones((4, 4), dtype=np.complex64),
    )

    record = decode_acquisition(path)

    np.testing.assert_array_equal(record.diff3d, diffraction)
    np.testing.assert_array_equal(record.xcoords, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(record.ycoords, [4.0, 5.0, 6.0])


@pytest.mark.parametrize("missing_key", ["xcoords", "ycoords", "probeGuess"])
def test_decode_acquisition_reports_missing_required_key(tmp_path, missing_key):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / f"missing_{missing_key}.npz"
    arrays = {
        "xcoords": np.arange(3, dtype=np.float64),
        "ycoords": np.arange(3, dtype=np.float64),
        "diff3d": np.ones((3, 4, 4), dtype=np.float32),
        "probeGuess": np.ones((4, 4), dtype=np.complex64),
    }
    del arrays[missing_key]
    np.savez(path, **arrays)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert missing_key in str(excinfo.value)


def test_decode_acquisition_canonicalizes_alias_legacy_and_singleton_layouts(tmp_path):
    from ptycho.acquisition import decode_acquisition

    canonical = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    coordinates = {
        "xcoords": np.arange(3, dtype=np.float64),
        "ycoords": np.arange(3, dtype=np.float64),
        "probeGuess": np.ones((4, 4), dtype=np.complex64),
    }
    sources = {
        "alias": {"diffraction": canonical},
        "legacy": {"diff3d": np.transpose(canonical, (1, 2, 0))},
        "singleton": {"diff3d": canonical[..., None]},
    }

    for name, diffraction in sources.items():
        path = tmp_path / f"{name}.npz"
        np.savez(path, **coordinates, **diffraction)
        np.testing.assert_array_equal(decode_acquisition(path).diff3d, canonical)


def test_decode_acquisition_requires_equal_canonical_dual_keys(tmp_path):
    from ptycho.acquisition import decode_acquisition

    canonical = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    common = {
        "xcoords": np.arange(3, dtype=np.float64),
        "ycoords": np.arange(3, dtype=np.float64),
        "probeGuess": np.ones((4, 4), dtype=np.complex64),
        "diff3d": canonical,
    }
    equal = tmp_path / "equal.npz"
    np.savez(equal, **common, diffraction=np.transpose(canonical, (1, 2, 0)))
    np.testing.assert_array_equal(decode_acquisition(equal).diff3d, canonical)

    conflict = tmp_path / "conflict.npz"
    np.savez(conflict, **common, diffraction=canonical + np.float32(1))
    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(conflict)
    assert str(conflict) in str(excinfo.value)
    assert "diff3d" in str(excinfo.value) and "diffraction" in str(excinfo.value)


def test_decode_acquisition_applies_explicit_strict_or_trailing_coordinate_policy(
    tmp_path,
):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "trailing.npz"
    xcoords = np.arange(5, dtype=np.float64)
    ycoords = xcoords + 10
    np.savez(
        path,
        xcoords=xcoords,
        ycoords=ycoords,
        xcoords_start=xcoords + 0.25,
        ycoords_start=ycoords - 0.5,
        diff3d=np.ones((3, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        scan_index=np.arange(5, dtype=np.int16),
        object_index=np.array([0, 0, 1, 2, 2], dtype=np.int32),
    )

    with pytest.raises(ValueError, match=r"trailing\.npz.*5 scan positions.*3 diffraction"):
        decode_acquisition(path, coordinate_policy="strict")
    with pytest.warns(RuntimeWarning, match="dropping the trailing 2 positions"):
        record = decode_acquisition(path, coordinate_policy="trailing")

    np.testing.assert_array_equal(record.xcoords, xcoords[:3])
    np.testing.assert_array_equal(record.xcoords_start, (xcoords + 0.25)[:3])
    np.testing.assert_array_equal(record.scan_index, np.arange(3, dtype=np.int64))
    np.testing.assert_array_equal(record.object_index, [0, 0, 1])

    short = tmp_path / "short.npz"
    np.savez(
        short,
        xcoords=xcoords[:2],
        ycoords=ycoords[:2],
        diff3d=np.ones((3, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
    )
    with pytest.raises(ValueError, match=r"2 scan positions.*3 diffraction.*Every pattern"):
        decode_acquisition(short, coordinate_policy="trailing")


def test_decode_acquisition_retains_optional_contract_fields(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "optional.npz"
    truth = np.ones((3, 4, 4), dtype=np.complex64) * (2 + 3j)
    label = truth * np.complex64(2)
    simulated_probe = np.ones((4, 4), dtype=np.complex64) * (4 - 1j)
    np.savez(
        path,
        xcoords=np.arange(3, dtype=np.float64),
        ycoords=np.arange(3, dtype=np.float64),
        diff3d=np.ones((3, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        scan_index=np.array([3, 4, 5], dtype=np.int32),
        object_index=np.array([7, 7, 8], dtype=np.int16),
        probe_simulated=simulated_probe,
        object_amplitude_scale=np.array(2.5, dtype=np.float64),
        Y=truth,
        label=label,
        scale_contract_version=np.array("ci_intensity_v2"),
        measurement_domain=np.array("count_intensity"),
    )

    record = decode_acquisition(path, experiment_id=11)

    assert record.scan_index.dtype == np.int64
    assert record.object_index.dtype == np.int64
    np.testing.assert_array_equal(record.probe_simulated, simulated_probe)
    np.testing.assert_array_equal(record.Y, truth)
    np.testing.assert_array_equal(record.label, label)
    assert record.object_amplitude_scale == np.float64(2.5)
    assert record.scale_contract_version == "ci_intensity_v2"
    assert record.measurement_domain == "count_intensity"
    assert record.experiment_id == 11


@pytest.mark.parametrize(
    ("version", "domain"),
    [
        ("unknown", "count_intensity"),
        ("legacy_v1", "count_intensity"),
        ("legacy_v1", None),
        (None, "normalized_amplitude"),
    ],
)
def test_decode_acquisition_rejects_invalid_measurement_pair(
    tmp_path, version, domain
):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "invalid_measurement_pair.npz"
    arrays = _valid_acquisition_arrays()
    if version is not None:
        arrays["scale_contract_version"] = np.array(version)
    if domain is not None:
        arrays["measurement_domain"] = np.array(domain)
    np.savez(path, **arrays)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "scale_contract_version" in str(excinfo.value)
    assert "measurement_domain" in str(excinfo.value)


@pytest.mark.parametrize(
    ("version", "domain"),
    [
        ("legacy_v1", "normalized_amplitude"),
        ("ci_intensity_v2", "count_intensity"),
    ],
)
def test_decode_acquisition_accepts_exact_measurement_pairs(
    tmp_path, version, domain
):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / f"{version}.npz"
    np.savez(
        path,
        **_valid_acquisition_arrays(),
        scale_contract_version=np.array(version),
        measurement_domain=np.array(domain),
    )

    record = decode_acquisition(path)

    assert (record.scale_contract_version, record.measurement_domain) == (
        version,
        domain,
    )


@pytest.mark.parametrize(
    "scale",
    [
        np.array([1.0], dtype=np.float64),
        np.array(0.0, dtype=np.float64),
        np.array(-1.0, dtype=np.float64),
        np.array(np.nan, dtype=np.float64),
        np.array(np.inf, dtype=np.float64),
        np.array(1.0, dtype=np.float32),
    ],
    ids=["nonscalar", "zero", "negative", "nan", "infinite", "float32"],
)
def test_decode_acquisition_rejects_invalid_object_amplitude_scale(tmp_path, scale):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "invalid_scale.npz"
    np.savez(path, **_valid_acquisition_arrays(), object_amplitude_scale=scale)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "object_amplitude_scale" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("xcoords_start", np.ones((3, 1), dtype=np.float64)),
        ("ycoords_start", np.ones((3, 1), dtype=np.float64)),
        ("diff3d", np.ones((3, 4, 5), dtype=np.float32)),
        ("probeGuess", np.ones((4,), dtype=np.complex64)),
        ("probeGuess", np.ones((5, 5), dtype=np.complex64)),
        ("objectGuess", np.ones((2, 4, 4), dtype=np.complex64)),
        ("Y", np.ones((2, 4, 4), dtype=np.complex64)),
        ("Y", np.ones((3, 4, 5), dtype=np.complex64)),
        ("label", np.ones((2, 4, 4), dtype=np.complex64)),
        ("probe_simulated", np.ones((2, 4, 4), dtype=np.complex64)),
    ],
    ids=[
        "x-start-rank",
        "y-start-rank",
        "rectangular-diffraction",
        "probe-rank",
        "probe-spatial-mismatch",
        "object-rank",
        "truth-count",
        "truth-spatial",
        "label-shape",
        "simulated-probe-rank",
    ],
)
def test_decode_acquisition_rejects_invalid_raw_array_shape(tmp_path, field, value):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / f"invalid_{field}.npz"
    arrays = _valid_acquisition_arrays()
    arrays[field] = value
    np.savez(path, **arrays)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert field in str(excinfo.value) or (
        field == "diff3d" and "diffraction" in str(excinfo.value)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("xcoords", np.array([0.0, np.nan, 2.0])),
        ("ycoords", np.array(["0", "1", "2"])),
        ("xcoords_start", np.array([0.0, 1.0, np.inf])),
        ("ycoords_start", np.array(["0", "1", "2"])),
    ],
)
def test_decode_acquisition_rejects_nonfinite_or_nonnumeric_coordinates(
    tmp_path, field, value
):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / f"invalid_{field}_values.npz"
    arrays = _valid_acquisition_arrays()
    arrays[field] = value
    np.savez(path, **arrays)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert field in str(excinfo.value)
    assert "finite numeric" in str(excinfo.value)


@pytest.mark.parametrize(
    "probe",
    [
        np.ones((4, 4), dtype=np.complex64),
        np.ones((2, 4, 4), dtype=np.complex64),
        np.ones((4, 4, 1), dtype=np.complex64),
    ],
    ids=["two-dimensional", "mode-first", "legacy-singleton"],
)
def test_decode_acquisition_accepts_supported_probe_layouts(tmp_path, probe):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "supported_probe.npz"
    np.savez(path, **_valid_acquisition_arrays() | {"probeGuess": probe})

    np.testing.assert_array_equal(decode_acquisition(path).probeGuess, probe)


def test_decode_acquisition_preserves_legacy_numeric_dtypes(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "legacy_dtypes.npz"
    arrays = _valid_acquisition_arrays()
    arrays["diff3d"] = arrays["diff3d"].astype(np.float64)
    arrays["probeGuess"] = arrays["probeGuess"].astype(np.complex128)
    arrays["objectGuess"] = np.ones((8, 8), dtype=np.complex128)
    np.savez(path, **arrays)

    record = decode_acquisition(path)

    assert record.diff3d.dtype == np.float64
    assert record.probeGuess.dtype == np.complex128
    assert record.objectGuess.dtype == np.complex128


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "diff3d",
            np.full((3, 4, 4), np.nan, dtype=np.float32),
            "finite numeric",
        ),
        (
            "diff3d",
            -np.ones((3, 4, 4), dtype=np.float32),
            "nonnegative",
        ),
        (
            "probeGuess",
            np.full((4, 4), np.inf, dtype=np.complex64),
            "finite numeric",
        ),
    ],
    ids=["nonfinite-diffraction", "negative-diffraction", "nonfinite-probe"],
)
def test_decode_acquisition_rejects_invalid_measurement_values(
    tmp_path, field, value, message
):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / f"invalid_{field}_values.npz"
    np.savez(path, **_valid_acquisition_arrays() | {field: value})

    with pytest.raises(ValueError, match=message):
        decode_acquisition(path)


def test_decode_acquisition_preserves_ambiguous_cubic_layout(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "cubic.npz"
    diffraction = np.arange(4**3, dtype=np.float32).reshape(4, 4, 4)
    np.savez(
        path,
        xcoords=np.arange(4, dtype=np.float64),
        ycoords=np.arange(4, dtype=np.float64),
        diff3d=diffraction,
        probeGuess=np.ones((4, 4), dtype=np.complex64),
    )

    np.testing.assert_array_equal(decode_acquisition(path).diff3d, diffraction)


@pytest.mark.parametrize(
    "experiment_id",
    [np.array(-1), np.array(1.5), np.array([1]), np.array(True)],
    ids=["negative", "float", "nonscalar", "boolean"],
)
def test_decode_acquisition_rejects_invalid_experiment_id(tmp_path, experiment_id):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "invalid_experiment.npz"
    np.savez(path, **_valid_acquisition_arrays(), experiment_id=experiment_id)

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "experiment_id" in str(excinfo.value)


def test_inspect_acquisition_reports_missing_probe(tmp_path):
    from ptycho.acquisition import inspect_acquisition

    path = tmp_path / "missing_probe_header.npz"
    arrays = _valid_acquisition_arrays()
    del arrays["probeGuess"]
    np.savez(path, **arrays)

    with pytest.raises(ValueError) as excinfo:
        inspect_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "probeGuess" in str(excinfo.value)


@pytest.mark.parametrize(
    "probe_shape",
    [(4, 4), (2, 4, 4), (4, 4, 1)],
    ids=["two-dimensional", "mode-first", "legacy-singleton"],
)
def test_inspect_acquisition_validates_supported_probe_layouts(tmp_path, probe_shape):
    from ptycho.acquisition import inspect_acquisition

    path = tmp_path / "probe_header.npz"
    np.savez(
        path,
        **_valid_acquisition_arrays()
        | {"probeGuess": np.ones(probe_shape, dtype=np.complex64)},
    )

    assert inspect_acquisition(path).probe_shape == probe_shape


def test_inspect_acquisition_rejects_probe_spatial_mismatch(tmp_path):
    from ptycho.acquisition import inspect_acquisition

    path = tmp_path / "bad_probe_header.npz"
    np.savez(
        path,
        **_valid_acquisition_arrays()
        | {"probeGuess": np.ones((2, 5, 5), dtype=np.complex64)},
    )

    with pytest.raises(ValueError, match=r"bad_probe_header\.npz.*probeGuess.*shape"):
        inspect_acquisition(path)


def test_inspect_acquisition_aligns_object_index_with_trailing_coordinates(tmp_path):
    from ptycho.acquisition import inspect_acquisition

    path = tmp_path / "trailing_header_identity.npz"
    np.savez(
        path,
        xcoords=np.arange(5, dtype=np.float64),
        ycoords=np.arange(5, dtype=np.float64),
        diff3d=np.ones((3, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        object_index=np.array([0, 0, 1, 2, 2], dtype=np.int16),
    )

    with pytest.warns(RuntimeWarning, match="dropping the trailing 2 positions"):
        header = inspect_acquisition(path, coordinate_policy="trailing")

    np.testing.assert_array_equal(header.object_index, [0, 0, 1])
    assert header.object_index.dtype == np.int64


def test_inspect_acquisition_rejects_invalid_object_index(tmp_path):
    from ptycho.acquisition import inspect_acquisition

    path = tmp_path / "invalid_header_identity.npz"
    np.savez(
        path,
        **_valid_acquisition_arrays(),
        object_index=np.array([0, -1, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError) as excinfo:
        inspect_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "object_index" in str(excinfo.value)


def test_decode_acquisition_preserves_json_metadata(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "metadata.npz"
    metadata = {"source": "fixture", "nested": {"count": 3}}
    np.savez(
        path,
        **_valid_acquisition_arrays(),
        _metadata=np.array(json.dumps(metadata)),
    )

    assert decode_acquisition(path).metadata == metadata


def test_decode_acquisition_rejects_object_metadata_without_pickle(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "unsafe_metadata.npz"
    np.savez(
        path,
        **_valid_acquisition_arrays(),
        _metadata=np.array(json.dumps({"source": "unsafe"}), dtype=object),
    )

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "_metadata" in str(excinfo.value)
    assert "object" in str(excinfo.value)


def test_decode_acquisition_reports_invalid_metadata_json(tmp_path):
    from ptycho.acquisition import decode_acquisition

    path = tmp_path / "bad_metadata_json.npz"
    np.savez(path, **_valid_acquisition_arrays(), _metadata=np.array("not-json"))

    with pytest.raises(ValueError) as excinfo:
        decode_acquisition(path)

    assert str(path) in str(excinfo.value)
    assert "_metadata" in str(excinfo.value)
    assert "JSON" in str(excinfo.value)


def test_transform_coordinates_is_pure_and_preserves_operation_order(tmp_path):
    from ptycho.acquisition import decode_acquisition, transform_coordinates

    path = tmp_path / "transform.npz"
    np.savez(
        path,
        xcoords=np.array([1.0, 2.0, 3.0]),
        ycoords=np.array([10.0, 20.0, 30.0]),
        xcoords_start=np.array([1.5, 2.5, 3.5]),
        ycoords_start=np.array([9.0, 19.0, 29.0]),
        diff3d=np.ones((3, 4, 4), dtype=np.float32),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
    )
    original = decode_acquisition(path)

    transformed = transform_coordinates(
        original, flip_x=True, swap_xy=True, scale=2.0
    )

    np.testing.assert_array_equal(transformed.xcoords, [20.0, 40.0, 60.0])
    np.testing.assert_array_equal(transformed.ycoords, [-2.0, -4.0, -6.0])
    np.testing.assert_array_equal(transformed.xcoords_start, [18.0, 38.0, 58.0])
    np.testing.assert_array_equal(transformed.ycoords_start, [-3.0, -5.0, -7.0])
    np.testing.assert_array_equal(original.xcoords, [1.0, 2.0, 3.0])


def test_transform_coordinates_preserves_missing_start_coordinates():
    from ptycho.acquisition import AcquisitionRecord, transform_coordinates

    record = AcquisitionRecord(
        xcoords=np.array([1.0, 2.0]),
        ycoords=np.array([3.0, 4.0]),
        xcoords_start=None,
        ycoords_start=None,
        diff3d=np.ones((2, 1, 1), dtype=np.float32),
        probeGuess=np.ones((1, 1), dtype=np.complex64),
        scan_index=np.zeros(2, dtype=np.int64),
    )

    transformed = transform_coordinates(
        record, flip_x=True, flip_y=True, swap_xy=True, scale=2.0
    )

    assert transformed.xcoords_start is None
    assert transformed.ycoords_start is None


def test_seeded_selection_returns_provenance_without_mutating_global_rng():
    from ptycho.acquisition import select_acquisition

    np.random.seed(20260813)
    state_before = np.random.get_state()
    expected = np.array([1, 2, 8, 11, 13, 19], dtype=np.int64)

    selection = select_acquisition(20, count=6, seed=17)

    np.testing.assert_array_equal(selection.source_indices, expected)
    assert selection.seed == 17
    assert selection.mode == "random_without_replacement"
    with pytest.raises(ValueError, match="read-only"):
        selection.source_indices[0] = 0
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]

    all_rows = select_acquisition(4)
    np.testing.assert_array_equal(all_rows.source_indices, np.arange(4))
    assert all_rows.mode == "all"
    assert not all_rows.source_indices.flags.writeable


def test_header_inspection_canonicalizes_layout_and_reads_only_shapes(tmp_path):
    from ptycho.acquisition import inspect_acquisition, read_npz_array_shape

    path = tmp_path / "header.npz"
    canonical = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    np.savez(
        path,
        xcoords=np.arange(3, dtype=np.float64),
        ycoords=np.arange(3, dtype=np.float64),
        diffraction=np.transpose(canonical, (1, 2, 0)),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
        label=np.ones_like(canonical, dtype=np.complex64),
    )

    header = inspect_acquisition(path)

    assert header.diffraction_shape == (3, 4, 4)
    assert header.probe_shape == (4, 4)
    assert header.label_shape == (3, 4, 4)
    np.testing.assert_array_equal(header.xcoords, [0.0, 1.0, 2.0])
    assert read_npz_array_shape(path, "probeGuess") == (4, 4)
    assert read_npz_array_shape(path, "missing") is None


def test_acquisition_record_snapshot_includes_new_optional_fields():
    from ptycho.acquisition import AcquisitionRecord

    raw = SimpleNamespace(
        xcoords=np.arange(2),
        ycoords=np.arange(2),
        xcoords_start=np.arange(2),
        ycoords_start=np.arange(2),
        diff3d=np.ones((2, 1, 1)),
        probeGuess=np.ones((1, 1)),
        scan_index=np.arange(2),
        object_index=np.array([4, 5]),
        objectGuess=None,
        Y=None,
        probe_simulated=np.ones((1, 1)) * 2,
        object_amplitude_scale=np.float64(3),
        label=np.ones((2, 1, 1)),
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        experiment_id=9,
    )

    record = AcquisitionRecord.from_raw_data(raw)

    np.testing.assert_array_equal(record.object_index, [4, 5])
    assert record.object_amplitude_scale == 3
    assert record.scale_contract_version == "legacy_v1"
    assert record.measurement_domain == "normalized_amplitude"
    assert record.experiment_id == 9


def test_acquisition_record_imports_without_tensorflow_or_torch():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_frameworks(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tensorflow" or name.startswith("tensorflow."):
        raise AssertionError(f"acquisition record imported TensorFlow: {name}")
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"acquisition record imported Torch: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_frameworks

from ptycho.acquisition import AcquisitionRecord

assert AcquisitionRecord.__module__ == "ptycho.acquisition"
assert not any(
    name == "tensorflow" or name.startswith("tensorflow.")
    or name == "torch" or name.startswith("torch.")
    for name in sys.modules
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_raw_data_passes_selection_source_mapping_to_grouping_owner(monkeypatch):
    from ptycho.acquisition import select_acquisition
    from ptycho import grouping
    from ptycho.raw_data import RawData

    selection = select_acquisition(15, count=4, seed=29)
    source_indices = selection.source_indices
    raw = RawData(
        xcoords=np.arange(4, dtype=np.float64),
        ycoords=np.zeros(4, dtype=np.float64),
        xcoords_start=np.arange(4, dtype=np.float64),
        ycoords_start=np.zeros(4, dtype=np.float64),
        diff3d=np.ones((4, 2, 2), dtype=np.float32),
        probeGuess=np.ones((2, 2), dtype=np.complex64),
        scan_index=np.arange(4, dtype=np.int64),
    )
    raw.sample_indices = source_indices
    real_planner = grouping.plan_sample_then_group
    captured = {}

    def capture_mapping(*args, **kwargs):
        captured["source_indices"] = kwargs.get("source_indices")
        return real_planner(*args, **kwargs)

    monkeypatch.setattr(grouping, "plan_sample_then_group", capture_mapping)

    grouped = raw.generate_grouped_data(
        N=2,
        K=1,
        nsamples=4,
        sequential_sampling=True,
        gridsize=1,
    )

    np.testing.assert_array_equal(captured["source_indices"], source_indices)
    np.testing.assert_array_equal(
        grouped["sample_indices"][grouped["nn_indices"]],
        source_indices[grouped["nn_indices"]],
    )
