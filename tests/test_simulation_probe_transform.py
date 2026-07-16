"""Numerical contract for reusable probe transforms."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from skimage.restoration import unwrap_phase


def _api():
    import importlib

    try:
        module = importlib.import_module("ptycho.simulation.probe_transform")
    except ModuleNotFoundError:
        pytest.fail("ptycho.simulation.probe_transform has not been implemented")
    required = (
        "BoundaryMatchedProbeResult",
        "extend_probe_boundary_matched",
        "parse_probe_transform_pipeline",
        "normalize_probe_transform_pipeline",
        "apply_probe_transform_pipeline",
        "apply_probe_transform_pipeline_with_metadata",
    )
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, f"probe transform API is missing {missing}"
    return module


def _fixture(size: int, *, quadratic: bool = False, constant_amp: bool = False):
    yy, xx = np.indices((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    r2 = (yy - center) ** 2 + (xx - center) ** 2
    phase = 0.017 * r2 + 0.13
    if not quadratic:
        phase = phase + 0.1 * np.sin(xx / 2.0) - 0.07 * np.cos(yy / 3.0)
    amplitude = np.ones((size, size), dtype=np.float64)
    if not constant_amp:
        amplitude = amplitude + 0.05 * yy + 0.03 * xx
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _perimeter_mask(shape, row_slice, column_slice):
    mask = np.zeros(shape, dtype=bool)
    y0, y1 = row_slice
    x0, x1 = column_slice
    mask[y0, x0:x1] = True
    mask[y1 - 1, x0:x1] = True
    mask[y0:y1, x0] = True
    mask[y0:y1, x1 - 1] = True
    return mask


def test_boundary_matched_solver_feasibility_gate_is_deterministic():
    api = _api()
    source = _fixture(8)

    first = api.extend_probe_boundary_matched(source, 16)
    second = api.extend_probe_boundary_matched(source, 16)

    assert first.source_rows == (4, 12)
    assert first.source_columns == (4, 12)
    assert np.array_equal(first.probe[4:12, 4:12], source)
    inner = _perimeter_mask(
        first.probe.shape, first.source_rows, first.source_columns
    )
    source_phase = unwrap_phase(np.angle(source))
    expected_inner = source_phase - first.quadratic_phase[4:12, 4:12]
    np.testing.assert_allclose(
        first.correction[inner],
        expected_inner[
            _perimeter_mask(source.shape, (0, 8), (0, 8))
        ],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(first.correction[first.outer_boundary_mask], 0.0)
    assert first.laplacian_residual <= first.solver_tolerance == 1e-10
    assert first.seam_residual <= 1e-10
    assert np.isfinite(first.probe).all()
    assert np.array_equal(first.probe, second.probe)
    assert first.laplacian_residual == second.laplacian_residual


@pytest.mark.parametrize(("source_size", "target_size"), [(8, 16), (7, 12)])
def test_boundary_matched_extension_handles_even_and_odd_padding(
    source_size,
    target_size,
):
    api = _api()
    source = _fixture(source_size)
    result = api.extend_probe_boundary_matched(source, target_size)
    before = (target_size - source_size) // 2
    after = before + source_size

    assert result.source_rows == (before, after)
    assert result.source_columns == (before, after)
    assert np.array_equal(result.probe[before:after, before:after], source)
    assert np.isfinite(result.probe).all()


def test_odd_padding_aligns_quadratic_to_the_copied_source_center():
    api = _api()
    source = _fixture(7, quadratic=True, constant_amp=True)

    result = api.extend_probe_boundary_matched(source, 12)

    outside = ~result.source_footprint_mask
    assert np.max(np.abs(result.correction[outside])) < 2e-8


def test_boundary_matched_extension_places_64_probe_at_32_to_96():
    api = _api()
    source = _fixture(64)
    result = api.extend_probe_boundary_matched(source, 128)

    assert result.source_rows == (32, 96)
    assert result.source_columns == (32, 96)
    assert np.array_equal(result.probe[32:96, 32:96], source)


@pytest.mark.parametrize("constant_amp", [True, False])
def test_boundary_matched_extension_edge_pads_amplitude(constant_amp):
    api = _api()
    source = _fixture(8, constant_amp=constant_amp)
    result = api.extend_probe_boundary_matched(source, 16)
    expected = np.pad(np.abs(source), 4, mode="edge")

    np.testing.assert_allclose(np.abs(result.probe), expected, rtol=2e-6, atol=2e-6)
    assert np.array_equal(result.probe[4:12, 4:12], source)


def test_exact_quadratic_source_needs_no_harmonic_correction_outside():
    api = _api()
    source = _fixture(8, quadratic=True, constant_amp=True)
    result = api.extend_probe_boundary_matched(source, 16)
    outside = ~result.source_footprint_mask

    # The source is complex64, so angle/unwrap quantization leaves an O(1e-8)
    # fit remainder even when the generating phase is analytically quadratic.
    assert np.max(np.abs(result.correction[outside])) < 2e-8
    assert result.seam_residual < 1e-10


def test_pipeline_supports_boundary_matched_and_pad_preserve_spellings():
    api = _api()
    assert api.parse_probe_transform_pipeline(
        "smooth:0.5|pad_extrapolate_boundary_matched:128"
    ) == [
        {"op": "smooth_complex", "sigma": 0.5},
        {
            "op": "pad_extrapolate_boundary_matched_complex",
            "target_N": 128,
        },
    ]
    normalized, steps = api.normalize_probe_transform_pipeline(
        target_N=128,
        probe_shape=(64, 64),
        probe_scale_mode="pipeline",
        probe_smoothing_sigma=0,
        probe_transform_pipeline="pad_preserve:128",
    )
    assert normalized == "pad_preserve:128"
    assert steps == [{"op": "pad_complex", "target_N": 128}]


@pytest.mark.parametrize(
    "pipeline",
    [
        "pad_extrapolate_boundary_matched:128|smooth:0.5",
        "pad_extrapolate_boundary_matched:128|pad:256",
        "pad_extrapolate_boundary_matched:128|interp:256",
    ],
)
def test_pipeline_rejects_operations_after_boundary_matched_extension(pipeline):
    api = _api()
    with pytest.raises(ValueError, match="must be the final operation"):
        api.normalize_probe_transform_pipeline(
            target_N=256,
            probe_shape=(64, 64),
            probe_scale_mode="pipeline",
            probe_smoothing_sigma=0,
            probe_transform_pipeline=pipeline,
        )


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        (
            "pad_extrapolate_boundary_matched:12|pad_extrapolate_boundary_matched:16",
            "exactly once",
        ),
        ("pad_extrapolate_boundary_matched:8", "outer pixel"),
        ("pad_extrapolate_boundary_matched:9", "outer pixel"),
    ],
)
def test_pipeline_rejects_ambiguous_or_nonextending_boundary_steps(
    pipeline, message
):
    api = _api()
    with pytest.raises(ValueError, match=message):
        api.normalize_probe_transform_pipeline(
            target_N=int(pipeline.rsplit(":", 1)[1]),
            probe_shape=(8, 8),
            probe_scale_mode="pipeline",
            probe_smoothing_sigma=0,
            probe_transform_pipeline=pipeline,
        )


def test_legacy_global_extrapolation_hash_is_frozen_across_extraction():
    api = _api()
    source = _fixture(8)
    normalized, steps = api.normalize_probe_transform_pipeline(
        target_N=16,
        probe_shape=source.shape,
        probe_scale_mode="pipeline",
        probe_smoothing_sigma=0,
        probe_transform_pipeline="pad_extrapolate:16|smooth:0.5",
    )
    result = api.apply_probe_transform_pipeline(source, steps)
    digest = hashlib.sha256(
        np.ascontiguousarray(result).view(np.uint8)
    ).hexdigest()

    assert normalized == "pad_extrapolate:16|smooth:0.5"
    assert digest == "333ed47d4e586e5015f0954ea1f561d91c3d02cd1fb764711c2f2afbd5868818"


def test_pipeline_metadata_records_boundary_solver_contract():
    api = _api()
    source = _fixture(8)
    steps = api.parse_probe_transform_pipeline(
        "smooth:0.5|pad_extrapolate_boundary_matched:16"
    )
    result = api.apply_probe_transform_pipeline_with_metadata(source, steps)

    assert result.probe.shape == (16, 16)
    assert result.metadata["boundary_method"] == "harmonic_dirichlet_c0"
    assert result.metadata["solver"] == "scipy.sparse.linalg.spsolve"
    assert result.metadata["solver_tolerance"] == 1e-10
    assert result.metadata["laplacian_residual"] <= 1e-10
    assert result.metadata["seam_residual"] <= 1e-10
