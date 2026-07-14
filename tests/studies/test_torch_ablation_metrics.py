import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.studies.ablation import metrics
from scripts.studies.ablation import manifest


def _complex_ramp(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    return (x + 10 * y + 1j * (100 + x + 10 * y)).astype(np.complex64)


def _anchor(scan_com: tuple[float, float], shape: tuple[int, int]) -> dict[str, object]:
    return {
        "scan_com": np.asarray(scan_com, dtype=np.float64),
        "canvas_shape": shape,
        "canvas_origin_offset": (
            shape[1] // 2 - scan_com[0],
            shape[0] // 2 - scan_com[1],
        ),
    }


def test_prepare_anchor_aligned_maps_integer_nonzero_scan_com_exactly():
    truth = _complex_ramp(5, 6)
    reconstruction = np.ones((2, 3), dtype=np.complex64)

    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones_like(reconstruction.real),
        _anchor((1.0, 1.0), reconstruction.shape),
        truth,
    )

    np.testing.assert_array_equal(prepared.target, truth[:2, :3])
    np.testing.assert_array_equal(prepared.common_mask, np.ones((2, 3), dtype=bool))
    assert prepared.target.dtype == truth.dtype


def test_prepare_anchor_aligned_bilinearly_samples_complex_half_pixel_com():
    truth = _complex_ramp(5, 6)
    reconstruction = np.ones((2, 2), dtype=np.complex64)

    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((2, 2), dtype=np.float32),
        _anchor((1.5, 1.5), reconstruction.shape),
        truth,
    )

    expected = np.asarray(
        [[5.5 + 105.5j, 6.5 + 106.5j], [15.5 + 115.5j, 16.5 + 116.5j]],
        dtype=np.complex64,
    )
    np.testing.assert_allclose(prepared.target, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(prepared.common_mask, True)


def test_prepare_anchor_aligned_intersects_only_oob_coordinates_and_positive_weights():
    truth = _complex_ramp(3, 3)
    reconstruction = np.ones((3, 3), dtype=np.complex64)
    reconstruction[2, 2] = np.nan + 1j
    weights = np.ones((3, 3), dtype=np.float32)
    weights[1, 2] = 0.0

    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        weights,
        _anchor((0.0, 0.0), reconstruction.shape),
        truth,
    )

    expected = np.asarray(
        [[False, False, False], [False, True, False], [False, True, True]]
    )
    np.testing.assert_array_equal(prepared.common_mask, expected)


def test_prepare_anchor_aligned_includes_exact_last_pixel_and_rejects_beyond_boundary():
    truth = _complex_ramp(3, 3)
    reconstruction = np.ones((3, 3), dtype=np.complex64)
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((3, 3), dtype=np.float32),
        _anchor((2.0, 2.0), reconstruction.shape),
        truth,
    )

    np.testing.assert_array_equal(prepared.target[:2, :2], truth[1:3, 1:3])
    np.testing.assert_array_equal(
        prepared.common_mask,
        [[True, True, False], [True, True, False], [False, False, False]],
    )


def test_nonfinite_reconstruction_does_not_shrink_mask_or_rectangle_but_metrics_fail():
    truth = _complex_ramp(3, 3)
    reconstruction = np.ones((3, 3), dtype=np.complex64)
    reconstruction[1, 1] = np.nan + 1j
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((3, 3), dtype=np.float32),
        _anchor((1.0, 1.0), reconstruction.shape),
        truth,
    )

    np.testing.assert_array_equal(prepared.common_mask, np.ones((3, 3), dtype=bool))
    assert prepared.ssim_bounds == metrics.Bounds(0, 0, 3, 3)
    with pytest.raises(metrics.MetricError, match="finite"):
        metrics.amplitude_pearson(prepared)


def test_nonfinite_truth_does_not_shrink_geometric_mask_or_rectangle_but_metrics_fail():
    truth = _complex_ramp(3, 3)
    truth[1, 1] = np.nan + 1j
    reconstruction = np.ones((3, 3), dtype=np.complex64)
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((3, 3), dtype=np.float32),
        _anchor((1.0, 1.0), reconstruction.shape),
        truth,
    )

    np.testing.assert_array_equal(prepared.common_mask, np.ones((3, 3), dtype=bool))
    assert prepared.ssim_bounds == metrics.Bounds(0, 0, 3, 3)
    with pytest.raises(metrics.MetricError, match="finite"):
        metrics.amplitude_pearson(prepared)


def test_prepare_anchor_aligned_does_not_object_center_crop_truth():
    truth = _complex_ramp(7, 7)
    reconstruction = np.ones((3, 3), dtype=np.complex64)

    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((3, 3), dtype=np.float32),
        _anchor((1.0, 1.0), reconstruction.shape),
        truth,
    )
    object_center_crop = truth[2:5, 2:5]

    np.testing.assert_array_equal(prepared.target, truth[:3, :3])
    assert not np.array_equal(prepared.target, object_center_crop)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([[1, 1, 1], [1, 1, 0]], (0, 0, 2, 2)),  # greatest area
        ([[0, 0, 1, 1], [1, 1, 0, 0]], (0, 2, 1, 4)),  # smallest top
        ([[1, 1, 0, 1, 1]], (0, 0, 1, 2)),  # smallest left
        ([[1, 1, 1, 1], [1, 1, 0, 0]], (0, 0, 1, 4)),  # smallest bottom
    ],
)
def test_largest_true_rectangle_uses_area_then_lexicographic_ties(mask, expected):
    bounds = metrics.largest_true_rectangle(np.asarray(mask, dtype=bool))
    assert (bounds.top, bounds.left, bounds.bottom, bounds.right) == expected
    assert np.asarray(mask, dtype=bool)[bounds.row_slice, bounds.col_slice].all()


def test_largest_true_rectangle_rejects_empty_mask_with_typed_error():
    with pytest.raises(metrics.AlignmentError, match="common mask is empty"):
        metrics.largest_true_rectangle(np.zeros((3, 4), dtype=bool))


@pytest.mark.parametrize("coordinate", [0.5, True, "0"])
def test_bounds_reject_noninteger_coordinates(coordinate):
    with pytest.raises(metrics.AlignmentError, match="integer"):
        metrics.Bounds(coordinate, 0, 2, 2)


@pytest.mark.parametrize(
    ("rectangle", "expected"),
    [
        ((1, 2, 6, 10), (1, 3, 6, 8)),
        ((2, 1, 10, 6), (3, 1, 8, 6)),
    ],
)
def test_centered_square_removes_odd_excess_from_bottom_or_right(rectangle, expected):
    square = metrics.centered_square_bounds(metrics.Bounds(*rectangle))
    assert (square.top, square.left, square.bottom, square.right) == expected


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda: metrics.prepare_anchor_aligned(
            np.ones((2, 2), dtype=np.complex64),
            np.ones((2, 3)),
            _anchor((1.0, 1.0), (2, 2)),
            _complex_ramp(3, 3),
        ),
        lambda: metrics.prepare_anchor_aligned(
            np.ones((2, 2), dtype=np.complex64),
            np.ones((2, 2)),
            {
                **_anchor((1.0, 1.0), (2, 2)),
                "canvas_origin_offset": (99.0, 99.0),
            },
            _complex_ramp(3, 3),
        ),
        lambda: metrics.prepare_anchor_aligned(
            np.ones((2, 2), dtype=np.complex64),
            np.asarray([[1.0, -1.0], [1.0, 1.0]]),
            _anchor((1.0, 1.0), (2, 2)),
            _complex_ramp(3, 3),
        ),
        lambda: metrics.prepare_anchor_aligned(
            np.ones((2, 2), dtype=np.complex64),
            np.ones((2, 2)),
            _anchor((1.0, 1.0), (3, 2)),
            _complex_ramp(3, 3),
        ),
        lambda: metrics.prepare_anchor_aligned(
            np.ones((2, 2), dtype=np.complex64),
            np.ones((2, 2)),
            _anchor((np.nan, 1.0), (2, 2)),
            _complex_ramp(3, 3),
        ),
    ],
)
def test_prepare_anchor_aligned_rejects_invalid_shapes_weights_and_anchor(bad_call):
    with pytest.raises(metrics.AlignmentError):
        bad_call()


def test_prepared_arrays_are_copies_and_read_only():
    reconstruction = np.ones((2, 2), dtype=np.complex64)
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones((2, 2), dtype=np.float32),
        _anchor((1.0, 1.0), (2, 2)),
        _complex_ramp(3, 3),
    )
    reconstruction[:] = 9

    assert np.all(prepared.reconstruction == 1)
    with pytest.raises(ValueError):
        prepared.reconstruction[0, 0] = 2


def test_direct_prepared_comparison_defensively_copies_all_arrays():
    reconstruction = np.ones((3, 3), dtype=np.complex64)
    target = _complex_ramp(3, 3)
    mask = np.ones((3, 3), dtype=bool)
    prepared = metrics.PreparedComparison(
        reconstruction,
        target,
        mask,
        metrics.Bounds(0, 0, 3, 3),
        metrics.Bounds(0, 0, 3, 3),
    )
    reconstruction[:] = 7
    target[:] = 8
    mask[:] = False

    np.testing.assert_array_equal(prepared.reconstruction, 1)
    np.testing.assert_array_equal(prepared.target, _complex_ramp(3, 3))
    np.testing.assert_array_equal(prepared.common_mask, True)
    for array in (
        prepared.reconstruction,
        prepared.target,
        prepared.common_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: metrics.PreparedComparison(
            np.ones((3, 3), dtype=np.complex64),
            np.ones((2, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=bool),
            metrics.Bounds(0, 0, 3, 3),
            metrics.Bounds(0, 0, 3, 3),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((3, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=bool),
            metrics.Bounds(0, 0, 3, 3),
            metrics.Bounds(0, 0, 3, 3),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=np.int8),
            metrics.Bounds(0, 0, 3, 3),
            metrics.Bounds(0, 0, 3, 3),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=bool),
            metrics.Bounds(0, 0, 4, 3),
            metrics.Bounds(0, 0, 3, 3),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((3, 3), dtype=np.complex64),
            np.ones((3, 3), dtype=np.complex64),
            np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
            metrics.Bounds(0, 0, 3, 3),
            metrics.Bounds(0, 0, 3, 3),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((4, 5), dtype=np.complex64),
            np.ones((4, 5), dtype=np.complex64),
            np.ones((4, 5), dtype=bool),
            metrics.Bounds(0, 0, 4, 5),
            metrics.Bounds(0, 0, 3, 2),
        ),
        lambda: metrics.PreparedComparison(
            np.ones((5, 5), dtype=np.complex64),
            np.ones((5, 5), dtype=np.complex64),
            np.ones((5, 5), dtype=bool),
            metrics.Bounds(1, 1, 4, 4),
            metrics.Bounds(0, 0, 3, 3),
        ),
    ],
)
def test_direct_prepared_comparison_validates_shapes_dtypes_and_bounds(factory):
    with pytest.raises(metrics.AlignmentError):
        factory()


def _prepare_same_frame(
    reconstruction: np.ndarray, target: np.ndarray
) -> metrics.PreparedComparison:
    assert reconstruction.shape == target.shape
    height, width = reconstruction.shape
    return metrics.prepare_anchor_aligned(
        reconstruction,
        np.ones(reconstruction.shape, dtype=np.float32),
        _anchor((width // 2, height // 2), (height, width)),
        target,
    )


def test_absolute_metrics_are_pinned_without_amplitude_gauge():
    reconstruction = np.asarray([[1j, 2j]], dtype=np.complex64)
    target = np.asarray([[2 + 0j, 4 + 0j]], dtype=np.complex64)

    values = metrics.absolute_scale_metrics(_prepare_same_frame(reconstruction, target))

    assert values["absolute_amp_mae"] == pytest.approx(1.5)
    assert values["absolute_amp_nrmse"] == pytest.approx(0.5)
    assert values["absolute_complex_nrmse"] == pytest.approx(0.5)
    assert values["amp_mean_ratio"] == pytest.approx(0.5)
    assert values["amp_quantile_ratio_p05"] == pytest.approx(0.5)
    assert values["amp_quantile_ratio_p50"] == pytest.approx(0.5)
    assert values["amp_quantile_ratio_p95"] == pytest.approx(0.5)
    assert values["absolute_amp_mae"] != pytest.approx(0.0)


def test_low_amplitude_metrics_are_scale_invariant_and_finite():
    y, x = np.mgrid[:9, :9]
    amplitude = (1e-30 * (1.0 + x + y)).astype(np.float64)
    target = (amplitude * np.exp(1j * 0.2)).astype(np.complex128)
    reconstruction = target.copy()
    prepared = _prepare_same_frame(reconstruction, target)

    factor = metrics.global_phase_factor(reconstruction, target)
    absolute = metrics.absolute_scale_metrics(prepared)

    assert factor == pytest.approx(1.0 + 0j)
    assert absolute["absolute_amp_mae"] == pytest.approx(0.0)
    assert absolute["absolute_amp_nrmse"] == pytest.approx(0.0)
    assert absolute["absolute_complex_nrmse"] == pytest.approx(0.0)
    assert absolute["amp_mean_ratio"] == pytest.approx(1.0)
    assert metrics.amplitude_pearson(prepared) == pytest.approx(1.0)
    assert metrics.amplitude_ssim(prepared) == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in absolute.values())


def test_high_finite_complex64_metrics_do_not_overflow():
    amplitude = np.asarray([[1e30, 2e30, 3e30, 4e30]], dtype=np.float32)
    target = (amplitude.astype(np.complex64) * np.complex64(1 + 0.5j)).astype(
        np.complex64
    )
    reconstruction = target.copy()
    prepared = _prepare_same_frame(reconstruction, target)

    factor = metrics.global_phase_factor(reconstruction, target)
    absolute = metrics.absolute_scale_metrics(prepared)

    assert abs(factor) == pytest.approx(1.0)
    assert absolute["absolute_amp_nrmse"] == pytest.approx(0.0)
    assert absolute["absolute_complex_nrmse"] == pytest.approx(0.0)
    assert absolute["amp_mean_ratio"] == pytest.approx(1.0)
    assert metrics.amplitude_pearson(prepared) == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in absolute.values())


def test_near_float64_max_identical_arrays_have_finite_absolute_metrics():
    value = 0.51 * np.finfo(np.float64).max
    target = np.full((1, 8), value + 0j, dtype=np.complex128)
    reconstruction = target.copy()
    prepared = _prepare_same_frame(reconstruction, target)

    factor = metrics.global_phase_factor(reconstruction, target)
    absolute = metrics.absolute_scale_metrics(prepared)

    assert factor == pytest.approx(1.0 + 0j)
    assert absolute["absolute_amp_mae"] == pytest.approx(0.0)
    assert absolute["absolute_amp_nrmse"] == pytest.approx(0.0)
    assert absolute["absolute_complex_nrmse"] == pytest.approx(0.0)
    assert absolute["amp_mean_ratio"] == pytest.approx(1.0)
    assert absolute["amp_quantile_ratio_p05"] == pytest.approx(1.0)
    assert absolute["amp_quantile_ratio_p50"] == pytest.approx(1.0)
    assert absolute["amp_quantile_ratio_p95"] == pytest.approx(1.0)
    assert all(np.isfinite(metric_value) for metric_value in absolute.values())


def test_scaled_rms_avoids_multi_element_true_norm_overflow():
    value = 0.51 * np.finfo(np.float64).max
    values = np.full(32, value, dtype=np.float64)

    result = metrics._rms(values)

    assert np.isfinite(result)
    assert result == pytest.approx(value)


def test_global_phase_factor_is_unit_magnitude_and_aligns_reconstruction():
    target = np.asarray([[1 + 2j, 3 - 1j]], dtype=np.complex128)
    theta = 0.73
    reconstruction = target * np.exp(1j * theta)

    factor = metrics.global_phase_factor(reconstruction, target)

    assert abs(factor) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(factor * reconstruction, target, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "reconstruction,target",
    [
        (
            np.asarray([[1 + 0j, 1 + 0j]]),
            np.asarray([[1 + 0j, -1 + 0j]]),
        ),
        (
            np.asarray([[np.inf + 0j]]),
            np.asarray([[1 + 0j]]),
        ),
    ],
)
def test_global_phase_factor_rejects_near_zero_or_nonfinite_correlation(
    reconstruction, target
):
    with pytest.raises(metrics.MetricError, match="correlation"):
        metrics.global_phase_factor(reconstruction, target)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros((1, 4), dtype=np.complex64),
        np.asarray([[0, 0, 2, 3]], dtype=np.complex64),
    ],
)
def test_absolute_metrics_reject_near_zero_norm_or_quantile_denominator(target):
    prepared = _prepare_same_frame(np.ones_like(target), target)
    with pytest.raises(metrics.MetricError, match="denominator"):
        metrics.absolute_scale_metrics(prepared)


def test_amplitude_pearson_uses_raw_common_mask_points():
    reconstruction = np.asarray([[1, 2, 100, 4]], dtype=np.complex64)
    target = np.asarray([[2, 4, 999, 8]], dtype=np.complex64)
    weights = np.asarray([[1, 1, 0, 1]], dtype=np.float32)
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        weights,
        _anchor((2.0, 0.0), reconstruction.shape),
        target,
    )

    assert metrics.amplitude_pearson(prepared) == pytest.approx(1.0)
    assert metrics.patch_amplitude_pearson(
        reconstruction[:, :2], target[:, :2]
    ) == pytest.approx(1.0)


def test_amplitude_pearson_rejects_zero_variance():
    prepared = _prepare_same_frame(
        np.ones((1, 3), dtype=np.complex64),
        np.asarray([[1, 2, 3]], dtype=np.complex64),
    )
    with pytest.raises(metrics.MetricError, match="variance"):
        metrics.amplitude_pearson(prepared)


def test_amplitude_ssim_uses_only_mean_scaled_prediction_and_target_range(monkeypatch):
    y, x = np.mgrid[:9, :9]
    target = (1 + x + 2 * y).astype(np.complex64)
    reconstruction = (0.25 * np.abs(target)).astype(np.complex64)
    prepared = _prepare_same_frame(reconstruction, target)
    seen = {}

    def fake_ssim(prediction, expected, *, data_range, win_size):
        seen.update(
            prediction=np.array(prediction),
            target=np.array(expected),
            data_range=data_range,
            win_size=win_size,
        )
        return 0.625

    monkeypatch.setattr(metrics, "_structural_similarity", fake_ssim)
    value = metrics.amplitude_ssim(prepared)

    assert value == pytest.approx(0.625)
    np.testing.assert_allclose(seen["prediction"], np.abs(target))
    np.testing.assert_allclose(seen["target"], np.abs(target))
    assert seen["data_range"] == pytest.approx(24.0)
    assert seen["win_size"] == 7


def test_amplitude_ms_ssim_pins_sigma_one_and_explicit_range(monkeypatch):
    y, x = np.mgrid[:9, :9]
    target = (2 + x + y).astype(np.complex64)
    reconstruction = (0.5 * np.abs(target)).astype(np.complex64)
    seen = {}

    def fake_ms(prediction, expected, *, data_range, sigma, levels=5, win_size=7):
        seen.update(
            prediction=np.array(prediction),
            target=np.array(expected),
            data_range=data_range,
            sigma=sigma,
            levels=levels,
            win_size=win_size,
        )
        return 0.75

    monkeypatch.setattr(metrics, "multiscale_ssim", fake_ms)
    value = metrics.amplitude_ms_ssim(_prepare_same_frame(reconstruction, target))

    assert value == pytest.approx(0.75)
    np.testing.assert_allclose(seen["prediction"], np.abs(target))
    assert seen["data_range"] == pytest.approx(16.0)
    assert seen["sigma"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reconstruction", "target", "message"),
    [
        (
            np.zeros((9, 9), dtype=np.complex64),
            np.ones((9, 9), dtype=np.complex64),
            "prediction mean",
        ),
        (
            np.ones((9, 9), dtype=np.complex64),
            np.ones((9, 9), dtype=np.complex64),
            "target data range",
        ),
    ],
)
def test_amplitude_similarity_rejects_near_zero_mean_or_range(
    reconstruction, target, message
):
    with pytest.raises(metrics.MetricError, match=message):
        metrics.amplitude_ssim(_prepare_same_frame(reconstruction, target))


def test_phase_metrics_use_global_alignment_mapping_and_wrapped_residual(monkeypatch):
    phase = np.linspace(-2.8, 2.8, 81, dtype=np.float64).reshape(9, 9)
    target = np.exp(1j * phase)
    reconstruction = np.exp(1j * (phase + 0.4))
    prepared = _prepare_same_frame(reconstruction, target)
    seen = {}

    def fake_ssim(prediction, expected, *, data_range, win_size):
        seen.update(
            prediction=np.array(prediction),
            target=np.array(expected),
            data_range=data_range,
        )
        return 0.9

    monkeypatch.setattr(metrics, "_structural_similarity", fake_ssim)

    assert metrics.phase_wrapped_mae(prepared) == pytest.approx(0.0, abs=1e-12)
    assert metrics.phase_ssim(prepared) == pytest.approx(0.9)
    np.testing.assert_allclose(seen["prediction"], seen["target"], atol=1e-12)
    assert np.all((seen["prediction"] >= 0) & (seen["prediction"] <= 1))
    assert seen["data_range"] == pytest.approx(1.0)


def test_phase_wrapped_mae_uses_every_mask_point_across_phase_boundaries():
    target_phase = np.asarray([[3.0, -3.0, 0.0]])
    residual = np.asarray([[0.2, -0.2, 0.0]])
    prepared = _prepare_same_frame(
        np.exp(1j * (target_phase + residual)), np.exp(1j * target_phase)
    )
    assert metrics.phase_wrapped_mae(prepared) == pytest.approx(2.0 / 15.0)


def test_phase_ms_ssim_has_no_gaussian_smoothing(monkeypatch):
    phase = np.linspace(-1.0, 1.0, 81).reshape(9, 9)
    prepared = _prepare_same_frame(np.exp(1j * (phase + 0.2)), np.exp(1j * phase))
    seen = {}

    def fake_ms(prediction, expected, *, data_range, sigma, levels=5, win_size=7):
        seen.update(data_range=data_range, sigma=sigma)
        return 0.8

    monkeypatch.setattr(metrics, "multiscale_ssim", fake_ms)
    assert metrics.phase_ms_ssim(prepared) == pytest.approx(0.8)
    assert seen == {"data_range": 1.0, "sigma": 0.0}


def test_multiscale_ssim_is_explicit_and_rejects_too_small_footprints():
    image = np.arange(31 * 31, dtype=np.float64).reshape(31, 31)
    assert metrics.multiscale_ssim(
        image, image, data_range=float(np.ptp(image)), sigma=1.0
    ) == pytest.approx(1.0)
    with pytest.raises(metrics.MetricError, match="too small"):
        metrics.multiscale_ssim(
            np.ones((6, 6)), np.ones((6, 6)), data_range=1.0, sigma=0.0
        )


def test_ms_ssim_uses_component_formula_not_full_ssim_at_every_scale():
    prediction = np.full((15, 15), 2.0, dtype=np.float64)
    target = np.ones((15, 15), dtype=np.float64)
    data_range = 2.0
    c1 = (0.01 * data_range) ** 2
    luminance = (2.0 * 2.0 * 1.0 + c1) / (2.0**2 + 1.0**2 + c1)
    final_weight = 0.2856 / (0.0448 + 0.2856)
    expected = luminance**final_weight

    actual = metrics.multiscale_ssim(
        prediction,
        target,
        data_range=data_range,
        sigma=0.0,
        levels=2,
        win_size=7,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert actual != pytest.approx(luminance, rel=1e-6)


def test_ms_ssim_average_pool_antialiases_checkerboard_and_drops_odd_edges():
    checkerboard = (np.indices((5, 7)).sum(axis=0) % 2).astype(np.float64)

    pooled = metrics._average_pool_2x2(checkerboard)

    np.testing.assert_array_equal(pooled, np.full((2, 3), 0.5))
    assert not np.array_equal(pooled, checkerboard[::2, ::2][:2, :3])


def test_ms_ssim_identical_odd_shape_is_deterministic():
    image = np.arange(15 * 17, dtype=np.float64).reshape(15, 17)
    first = metrics.multiscale_ssim(
        image, image, data_range=float(np.ptp(image)), sigma=0.0
    )
    second = metrics.multiscale_ssim(
        image, image, data_range=float(np.ptp(image)), sigma=0.0
    )
    assert first == pytest.approx(1.0)
    assert second == first


def test_ms_ssim_prefilter_is_applied_once_per_input(monkeypatch):
    image = np.arange(15 * 15, dtype=np.float64).reshape(15, 15)
    real_filter = metrics.gaussian_filter
    calls = []

    def recording_filter(value, *, sigma, **kwargs):
        calls.append((np.asarray(value).shape, sigma))
        return real_filter(value, sigma=sigma, **kwargs)

    monkeypatch.setattr(metrics, "gaussian_filter", recording_filter)
    assert metrics.multiscale_ssim(
        image, image, data_range=float(np.ptp(image)), sigma=1.0, levels=2
    ) == pytest.approx(1.0)
    assert calls == [((15, 15), 1.0), ((15, 15), 1.0)]


def test_frc_receives_exact_square_without_recrop_or_smoothing(monkeypatch):
    y, x = np.mgrid[:7, :9]
    target = (1 + x + y).astype(np.complex64)
    reconstruction = (2 + 2 * x + y).astype(np.complex64)
    prepared = _prepare_same_frame(reconstruction, target)
    seen = {}
    curve = np.asarray([0.8, 0.4, 0.2, 0.1], dtype=np.float64)

    def fake_fsc(first, second):
        seen.update(first=np.array(first), second=np.array(second))
        return curve.copy()

    monkeypatch.setattr(metrics, "_run_fsc", fake_fsc)
    result = metrics.amplitude_frc(prepared)

    assert seen["first"].shape == seen["second"].shape == (7, 7)
    np.testing.assert_array_equal(seen["first"], np.abs(target[:, 1:8]))
    np.testing.assert_array_equal(seen["second"], np.abs(reconstruction[:, 1:8]))
    assert result.to_json() == {
        "curve": [0.8, 0.4, 0.2, 0.1],
        "frc50": pytest.approx(0.75),
        "frc1over7": pytest.approx(2.5714285714285716),
    }


def test_phase_frc_passes_aligned_phase_square(monkeypatch):
    phase = np.linspace(-2.0, 2.0, 63).reshape(7, 9)
    prepared = _prepare_same_frame(np.exp(1j * (phase + 0.5)), np.exp(1j * phase))
    seen = {}

    def fake_fsc(first, second):
        seen.update(first=np.array(first), second=np.array(second))
        return np.asarray([1.0, 0.6, 0.4])

    monkeypatch.setattr(metrics, "_run_fsc", fake_fsc)
    metrics.phase_frc(prepared)
    np.testing.assert_allclose(seen["first"], seen["second"], atol=1e-12)
    assert seen["first"].shape == (7, 7)


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (np.ones((3, 4)), np.ones((3, 4)), "square"),
        (np.ones((3, 3)), np.ones((4, 4)), "equal"),
        (np.ones((0, 0)), np.ones((0, 0)), "nonempty"),
    ],
)
def test_frc_rejects_bad_inputs_before_low_level_call(prediction, target, message):
    with pytest.raises(metrics.MetricError, match=message):
        metrics.frc_metrics(prediction, target)


@pytest.mark.parametrize(
    "curve",
    [
        np.asarray([]),
        np.asarray([0.8, np.nan]),
        np.asarray([[0.8, 0.4]]),
        np.asarray([1.2, 0.4]),
    ],
)
def test_frc_rejects_invalid_curves(monkeypatch, curve):
    monkeypatch.setattr(metrics, "_run_fsc", lambda _a, _b: curve)
    with pytest.raises(metrics.MetricError, match="curve"):
        metrics.frc_metrics(np.ones((8, 8)), np.ones((8, 8)))


def test_frc_preserves_tolerated_raw_curve_excursion_in_json(monkeypatch):
    excursion = 1.0 + 5e-13
    raw_curve = np.asarray([excursion, 0.6, 0.4], dtype=np.float64)
    monkeypatch.setattr(metrics, "_run_fsc", lambda _a, _b: raw_curve)

    result = metrics.frc_metrics(np.ones((8, 8)), np.ones((8, 8)))

    assert result.curve[0] == excursion
    payload = result.to_json()
    assert payload["curve"] == [excursion, 0.6, 0.4]
    assert json.loads(json.dumps(payload)) == payload


def test_metric_registry_is_the_exact_manifest_registry():
    assert metrics.METRIC_PATHS is manifest.METRIC_PATHS
    assert metrics.metric_paths() == manifest.METRIC_PATHS
    assert manifest._METRIC_PATHS is manifest.METRIC_PATHS


@pytest.mark.parametrize(
    ("truth_role", "namespace"),
    [
        ("object_truth", "truth_quality"),
        ("reference_reconstruction", "reference_agreement"),
    ],
)
def test_same_image_formula_uses_role_appropriate_namespace(truth_role, namespace):
    record = metrics.build_image_metric_record(
        "amp_pearson",
        0.75,
        truth_role=truth_role,
        basis="raw_amplitude",
        alignment="pearson_centering_only",
    )
    assert record.path == f"{namespace}.amp_pearson"
    assert record.to_json() == {
        "value": 0.75,
        "metadata": {
            "basis": "raw_amplitude",
            "alignment": "pearson_centering_only",
        },
    }


def test_reference_role_cannot_emit_absolute_correctness_metric():
    with pytest.raises(metrics.RegistryError, match="absolute"):
        metrics.build_image_metric_record(
            "absolute_amp_mae",
            0.2,
            truth_role="reference_reconstruction",
            basis="absolute_amplitude",
            alignment="none",
        )


@pytest.mark.parametrize(
    "name",
    [
        "relative_l2_intensity_error",
        "mean_raw_poisson_nll",
        "varpro.s1",
        "varpro.s2",
        "varpro.condition",
        "varpro.unit_objective",
        "varpro.fitted_objective",
        "dose.object_scale",
        "dose.object_scale_cv",
    ],
)
def test_measurement_metrics_can_only_emit_measurement_consistency(name):
    record = metrics.build_measurement_metric_record(
        name, 1.25, basis="physical_counts", alignment="none"
    )
    assert record.path == f"measurement_consistency.{name}"
    with pytest.raises(metrics.RegistryError):
        metrics.build_metric_record(
            record.path,
            1.25,
            basis="physical_counts",
            alignment="none",
            truth_role="object_truth",
        )


@pytest.mark.parametrize(
    "path",
    [
        "truth_quality.not_registered",
        "reference_agreement.absolute_amp_mae",
        "other.value",
    ],
)
def test_unknown_metric_paths_fail(path):
    with pytest.raises(metrics.RegistryError, match="closed metric registry"):
        metrics.build_metric_record(path, 1.0, basis="raw", alignment="none")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "1.0", None])
def test_metric_records_reject_nonfinite_or_nonnumeric_scalars(value):
    with pytest.raises(metrics.RegistryError, match="finite scalar"):
        metrics.build_metric_record(
            "stability.gradient_norm_mean",
            value,
            basis="training_history",
            alignment="none",
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("truth_quality.amp_frc_curve", []),
        ("truth_quality.amp_frc_curve", [0.8, np.nan]),
        ("truth_quality.amp_frc_curve", [[0.8, 0.4]]),
        ("truth_quality.amp_frc_curve", 0.8),
        ("truth_quality.amp_pearson", [0.8]),
    ],
)
def test_metric_records_reject_malformed_curves_and_scalar_shapes(path, value):
    with pytest.raises(metrics.RegistryError):
        metrics.build_metric_record(path, value, basis="frequency", alignment="none")


@pytest.mark.parametrize(
    "curve",
    [
        [0.8 + 0j, 0.4],
        [0.8, True],
        [-1e-6, 0.4],
        [1.0 + 1e-6, 0.4],
    ],
)
def test_metric_registry_rejects_nonreal_bool_or_out_of_domain_curve_entries(curve):
    with pytest.raises(metrics.RegistryError, match="curve"):
        metrics.build_image_metric_record(
            "amp_frc_curve",
            curve,
            truth_role="object_truth",
            basis="raw_amplitude_frequency",
            alignment="none",
        )


def test_metric_registry_preserves_tolerated_curve_excursions_as_json_native():
    low_excursion = -5e-13
    high_excursion = 1.0 + 5e-13
    record = metrics.build_image_metric_record(
        "amp_frc_curve",
        [low_excursion, np.float64(0.4), high_excursion],
        truth_role="object_truth",
        basis="raw_amplitude_frequency",
        alignment="none",
    )

    payload = record.to_json()
    assert payload["value"] == [low_excursion, 0.4, high_excursion]
    assert all(type(value) is float for value in payload["value"])
    assert json.loads(json.dumps(payload)) == payload


def test_metric_bundle_is_nested_json_ready_and_preserves_full_curves_as_lists():
    curve = [0.9, 0.6, 0.4]
    curve_record = metrics.build_image_metric_record(
        "amp_frc_curve",
        curve,
        truth_role="object_truth",
        basis="raw_amplitude_frequency",
        alignment="none",
    )
    scalar_record = metrics.build_measurement_metric_record(
        "varpro.s1", 1.2, basis="physical_counts", alignment="none"
    )
    bundle = metrics.MetricBundle((curve_record, scalar_record))
    curve[0] = 0.0

    payload = bundle.to_json()
    assert payload["truth_quality"]["amp_frc_curve"]["value"] == [0.9, 0.6, 0.4]
    assert payload["measurement_consistency"]["varpro"]["s1"]["value"] == 1.2
    assert json.loads(json.dumps(payload)) == payload


def test_metric_bundle_rejects_duplicate_paths():
    record = metrics.build_metric_record(
        "runtime.train_seconds", 2.0, basis="wall_clock", alignment="none"
    )
    with pytest.raises(metrics.RegistryError, match="duplicate"):
        metrics.MetricBundle((record, record))


def test_metric_bundle_safely_converts_iterables_and_validates_members():
    record = metrics.build_metric_record(
        "runtime.train_seconds", 2.0, basis="wall_clock", alignment="none"
    )
    bundle = metrics.MetricBundle([record])
    assert bundle.records == (record,)

    with pytest.raises(metrics.RegistryError, match="iterable"):
        metrics.MetricBundle(None)
    with pytest.raises(metrics.RegistryError, match="MetricRecord"):
        metrics.MetricBundle((object(),))


def test_importing_metrics_does_not_load_matplotlib_or_tensorflow():
    code = """
import sys
import scripts.studies.ablation.metrics
assert "matplotlib.pyplot" not in sys.modules
assert "tensorflow" not in sys.modules
print("isolated")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated"


def test_real_frc_backend_is_finite_and_silent_for_structured_pair(capsys):
    y, x = np.mgrid[:16, :16]
    target = np.sin(x / 2.0) + np.cos(y / 3.0)
    prediction = np.roll(target, 1, axis=1)

    result = metrics.frc_metrics(prediction, target)

    assert result.curve
    assert np.isfinite(result.curve).all()
    assert capsys.readouterr().out == ""


def test_real_frc_backend_nonfinite_curve_becomes_typed_error_and_is_silent(capsys):
    constant = np.ones((16, 16), dtype=np.float64)
    with pytest.raises(metrics.MetricError, match="finite"):
        metrics.frc_metrics(constant, constant)
    assert capsys.readouterr().out == ""


def test_patch_pearson_record_is_separate_and_role_validated():
    record = metrics.build_patch_amplitude_pearson_record(
        np.asarray([[1, 2, 3]], dtype=np.complex64),
        np.asarray([[2, 4, 6]], dtype=np.complex64),
        truth_role="reference_reconstruction",
    )
    assert record.path == "reference_agreement.patch_amp_pearson"
    assert record.to_json()["metadata"] == {
        "basis": "object_frame_raw_amplitude",
        "alignment": "pearson_centering_only",
    }


def test_varpro_quality_metrics_preserve_typed_before_and_after_stages():
    coordinate = np.linspace(-0.6, 0.6, 64).reshape(8, 8)
    target = (1.0 + 0.2 * coordinate) * np.exp(1j * coordinate)
    before = _prepare_same_frame(0.8 * target, target)
    after = _prepare_same_frame(np.full((8, 8), 0.8 + 0.0j), target)

    result = metrics.varpro_quality_metrics(before, after)

    assert isinstance(result.before, metrics.ImageQualityMetrics)
    assert isinstance(result.after, metrics.ImageQualityMetrics)
    assert result.before.amplitude_ssim == pytest.approx(1.0)
    assert result.before.phase_wrapped_mae == pytest.approx(0.0, abs=1e-12)
    assert result.after.phase_wrapped_mae > result.before.phase_wrapped_mae


def test_valid_mask_diagnostics_detect_collapse_and_decoder_head_saturation():
    reconstruction = np.ones((6, 6), dtype=np.complex128) * (1.2 + 1.2j)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True

    result = metrics.valid_mask_diagnostics(
        reconstruction,
        mask,
        decoder_output=reconstruction,
    )

    assert result.amplitude_variance == pytest.approx(0.0)
    assert result.phase_variance == pytest.approx(0.0)
    assert result.amplitude_dynamic_range == pytest.approx(0.0)
    assert result.phase_dynamic_range == pytest.approx(0.0)
    assert result.real_head_saturation_fraction == pytest.approx(1.0)
    assert result.imag_head_saturation_fraction == pytest.approx(1.0)
    assert result.real_head_lower_saturation_fraction == pytest.approx(0.0)
    assert result.real_head_upper_saturation_fraction == pytest.approx(1.0)
    assert result.imag_head_lower_saturation_fraction == pytest.approx(0.0)
    assert result.imag_head_upper_saturation_fraction == pytest.approx(1.0)


def test_valid_mask_diagnostics_respects_asymmetric_cnn_output_rails():
    decoder = np.asarray(
        [[-0.8 - 1.2j, 1.2 + 1.2j], [0.0 + 0.0j, 0.2 + 0.1j]],
        dtype=np.complex128,
    )

    result = metrics.valid_mask_diagnostics(
        decoder,
        np.ones((2, 2), dtype=bool),
        decoder_output=decoder,
    )

    assert result.real_head_lower_saturation_fraction == pytest.approx(0.25)
    assert result.real_head_upper_saturation_fraction == pytest.approx(0.25)
    assert result.imag_head_lower_saturation_fraction == pytest.approx(0.25)
    assert result.imag_head_upper_saturation_fraction == pytest.approx(0.25)
    assert result.real_head_saturation_fraction == pytest.approx(0.5)
    assert result.imag_head_saturation_fraction == pytest.approx(0.5)


def test_convergence_metrics_reject_a_run_still_improving_at_budget_boundary():
    improving = metrics.convergence_metrics(np.linspace(2.0, 1.0, 20))
    settled = metrics.convergence_metrics([2.0, 1.5, 1.2, *([1.0] * 17)])

    assert improving.tail_relative_improvement > 0.05
    assert improving.budget_boundary_improving == 1.0
    assert settled.tail_relative_improvement == pytest.approx(0.0)
    assert settled.budget_boundary_improving == 0.0


def test_scan_utilization_metrics_expose_historical_grouping_regression():
    result = metrics.scan_utilization_metrics(
        used_scan_ids=range(756),
        used_center_scan_ids=range(700),
        expected_scan_ids=range(1250),
        filtered_eligible_scan_ids=range(900),
        canvas_weights=np.r_[np.ones(71), np.zeros(29)].reshape(10, 10),
    )

    assert result.unique_scans_used == 756
    assert result.unique_centers_used == 700
    assert result.unique_scans_expected == 1250
    assert result.scan_utilization_fraction == pytest.approx(756 / 1250)
    assert result.unique_scans_filtered_eligible == 900
    assert result.filtered_scan_utilization_fraction == pytest.approx(700 / 900)
    assert result.canvas_coverage_fraction == pytest.approx(0.71)
    assert result.scan_utilization_fraction < 0.99
    assert result.canvas_coverage_fraction < 0.95


def test_poisson_noise_oracle_bounds_model_count_error_relative_to_noise_floor():
    expected = np.asarray([[100.0, 64.0], [36.0, 25.0]])
    observed = np.asarray([[110.0, 60.0], [32.0, 27.0]])
    model = np.asarray([[130.0, 45.0], [20.0, 35.0]])

    result = metrics.poisson_noise_oracle_metrics(observed, expected, model)

    assert result.oracle_relative_l2_error > 0.0
    assert result.model_relative_l2_error > result.oracle_relative_l2_error
    assert result.model_to_oracle_error_ratio == pytest.approx(
        result.model_relative_l2_error / result.oracle_relative_l2_error
    )


def test_task19_metric_registry_is_closed_over_new_gate_operands():
    required = {
        "truth_quality.pre_varpro.amp_ssim",
        "truth_quality.pre_varpro.phase_ssim",
        "truth_quality.post_varpro.amp_ssim",
        "truth_quality.post_varpro.phase_ssim",
        "stability.amp_dynamic_range",
        "stability.phase_dynamic_range",
        "stability.real_head_saturation_fraction",
        "stability.imag_head_saturation_fraction",
        "stability.loss_tail_relative_improvement",
        "stability.budget_boundary_improving",
        "stability.scan_utilization_fraction",
        "stability.unique_scans_used",
        "stability.unique_scans_expected",
        "measurement_consistency.poisson_oracle_relative_l2_error",
        "measurement_consistency.model_to_poisson_oracle_error_ratio",
    }

    assert required <= metrics.metric_paths()
