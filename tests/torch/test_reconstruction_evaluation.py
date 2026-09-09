"""Raw-array reconstruction quality and visual-diagnostic contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


def _anchor(scan_com: tuple[float, float], shape: tuple[int, int]):
    return {
        "scan_com": list(scan_com),
        "canvas_shape": list(shape),
        "canvas_origin_offset": [
            shape[1] // 2 - scan_com[0],
            shape[0] // 2 - scan_com[1],
        ],
    }


def _truth(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    amplitude = 1.0 + 0.03 * x + 0.05 * y
    phase = -0.8 + 0.04 * x - 0.02 * y
    return np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex64)


def _center_block_truth(shape: tuple[int, int] = (96, 96)) -> np.ndarray:
    """Object-centered truth with a distinctive bright center block."""
    rows, cols = shape
    y, x = np.indices(shape, dtype=np.float64)
    amplitude = 1.0 + 0.01 * (x / rows) + 0.02 * (y / cols)
    block = np.zeros(shape, dtype=bool)
    block[rows // 2 - 4 : rows // 2 + 4, cols // 2 - 4 : cols // 2 + 4] = True
    amplitude = amplitude + np.where(block, 9.0, 0.0)
    phase = 0.3 * (x + y) / float(rows + cols)
    return np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex64)


def _reassembly(
    *,
    patches: int = 8,
    accepted_patches: int | None = None,
    precision: str = "32-true",
    count_metrics=None,
    **overrides,
):
    values = {
        "accepted_patches": patches,
        "total_patches": patches,
        "used_scan_ids": tuple(range(8)),
        "used_center_scan_ids": (0, 4),
        "expected_scan_ids": tuple(range(8)),
        "filtered_eligible_scan_ids": (0, 4),
        "s1": 1.2,
        "s2": 0.8,
        "mask_digest": "a" * 64,
        "effective_precision": precision,
        "count_metrics": count_metrics
        or SimpleNamespace(
            status="not_applicable",
            reason="legacy_normalized_amplitude",
        ),
    }
    if accepted_patches is not None:
        values["accepted_patches"] = accepted_patches
    values.update(overrides)
    return SimpleNamespace(**values)


def _evaluation_inputs():
    truth = _truth()
    target = truth[:9, :9]
    residual = np.linspace(-0.2, 0.2, target.size).reshape(target.shape)
    canvas = np.asarray(2.0 * target * np.exp(1j * (0.35 + residual)))
    prescale = np.asarray(0.5 * target * np.exp(-0.2j))
    return {
        "complex_canvas": canvas,
        "prescale_canvas": prescale,
        "canvas_weights": np.ones(canvas.shape, dtype=np.float32),
        "canvas_anchor": _anchor((4.0, 4.0), canvas.shape),
        "truth": truth,
        "reassembly": _reassembly(),
        "channel_indices": np.arange(8, dtype=np.int64).reshape(2, 4),
        "groups_per_center": 1,
    }


def test_prepare_anchor_aligned_supports_integer_and_half_pixel_sampling():
    from ptycho_torch.reconstruction_evaluation import prepare_anchor_aligned

    truth = _truth((12, 12))
    reconstruction = np.ones((8, 8), dtype=np.complex64)
    weights = np.ones((8, 8), dtype=np.float32)
    integer = prepare_anchor_aligned(
        reconstruction,
        weights,
        _anchor((4.0, 4.0), reconstruction.shape),
        truth,
    )
    np.testing.assert_array_equal(integer.target, truth[:8, :8])

    half = prepare_anchor_aligned(
        reconstruction,
        weights,
        _anchor((4.5, 4.5), reconstruction.shape),
        truth,
    )
    expected = 0.25 * (
        truth[:8, :8] + truth[1:9, :8] + truth[:8, 1:9] + truth[1:9, 1:9]
    )
    np.testing.assert_allclose(half.target, expected, rtol=1e-6, atol=1e-6)
    assert half.common_mask.all()


def test_prepare_anchor_aligned_applies_the_global_gauge():
    from ptycho_torch.reconstruction_evaluation import (
        absolute_scale_metrics,
        phase_wrapped_mae,
        prepare_anchor_aligned,
    )

    truth = _truth((14, 14))
    rows, cols = np.indices((10, 10), dtype=np.float64)
    reconstruction = 3.0 * truth[:10, :10] * np.exp(1j * (0.7 + 0.03 * cols - 0.02 * rows))
    weights = np.ones((10, 10), dtype=np.float32)
    weights[0, :] = 0.0
    anchor = _anchor((5.0, 5.0), reconstruction.shape)

    prepared = prepare_anchor_aligned(reconstruction, weights, anchor, truth)
    assert prepared.gauge is not None
    assert prepared.gauge.scale == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert prepared.gauge.ramp_x == pytest.approx(-0.03, abs=1e-6)
    assert prepared.gauge.ramp_y == pytest.approx(0.02, abs=1e-6)
    mask = prepared.common_mask
    np.testing.assert_allclose(prepared.reconstruction[mask], truth[:10, :10][mask], atol=1e-6)
    np.testing.assert_allclose(prepared.raw_reconstruction, reconstruction)
    assert phase_wrapped_mae(prepared) == pytest.approx(0.0, abs=1e-6)
    absolute = absolute_scale_metrics(prepared)
    assert absolute["amp_mae"] == pytest.approx(0.0, abs=1e-6)
    assert absolute["absolute_amp_mae"] == pytest.approx(
        float(np.mean(2.0 * np.abs(truth[:10, :10])[mask]))
    )
    assert absolute["amp_mean_ratio"] == pytest.approx(3.0, rel=1e-6)

    ungauged = prepare_anchor_aligned(reconstruction, weights, anchor, truth, gauge="none")
    assert ungauged.gauge is None
    np.testing.assert_allclose(ungauged.reconstruction, reconstruction)
    assert phase_wrapped_mae(ungauged) > 0.05


def test_prepare_anchor_aligned_supports_exact_truth_origin_and_metric_crop():
    from ptycho_torch.reconstruction_evaluation import prepare_anchor_aligned

    truth = _truth((14, 15))
    reconstruction = np.array(truth[3:11, 4:12], copy=True)
    weights = np.ones(reconstruction.shape, dtype=np.float32)
    anchor = _anchor((7.5, 6.5), reconstruction.shape)
    anchor["truth_origin"] = [4, 3]

    prepared = prepare_anchor_aligned(
        reconstruction,
        weights,
        anchor,
        truth,
        metric_crop_border=2,
    )

    np.testing.assert_array_equal(prepared.target, reconstruction)
    assert prepared.ssim_bounds.to_jsonable() == {
        "top": 2,
        "left": 2,
        "bottom": 6,
        "right": 6,
    }
    assert int(np.count_nonzero(prepared.common_mask)) == 16


def test_canvas_positions_follow_reassembly_placement():
    from ptycho_torch.reconstruction_evaluation import canvas_positions

    anchor = _anchor((10.5, 20.25), (40, 60))
    actual = canvas_positions(
        np.asarray([10.5, 12.0]),
        np.asarray([20.25, 19.0]),
        anchor,
        (40, 60),
    )
    np.testing.assert_allclose(actual, [[30.0, 20.0], [31.5, 18.75]])


def test_prepare_anchor_aligned_intersects_acquisition_metric_support():
    from ptycho_torch.reconstruction_evaluation import prepare_anchor_aligned

    truth = _truth((12, 12))
    reconstruction = np.array(truth[:9, :9], copy=True)
    weights = np.ones(reconstruction.shape, dtype=np.float32)
    support = np.zeros(reconstruction.shape, dtype=bool)
    support[1:8, 2:9] = True
    prepared = prepare_anchor_aligned(
        reconstruction,
        weights,
        _anchor((4.5, 4.5), reconstruction.shape),
        truth,
        metric_support=support,
    )
    np.testing.assert_array_equal(prepared.common_mask, support)


def test_probe_exposure_support_uses_stitch_denominator_and_single_probe_peak():
    from ptycho_torch.reconstruction_evaluation import probe_exposure_support

    weights = np.asarray([[0.0, 0.09, 0.1, 0.4]], dtype=np.float32)
    probe = np.ones((2, 2), dtype=np.complex64)
    np.testing.assert_array_equal(
        probe_exposure_support(weights, probe),
        np.asarray([[False, False, True, True]]),
    )


def test_frc50_uses_last_ring_above_threshold_and_reports_resolution():
    from ptycho_torch.reconstruction_evaluation import (
        _last_frc50_frequency,
        frc50_metrics,
        prepare_anchor_aligned,
    )

    assert _last_frc50_frequency(
        [0.05, 0.15, 0.25, 0.35], [0.8, 0.4, 0.7, 0.3]
    ) == pytest.approx(0.25)
    truth = _truth((128, 128))
    prepared = prepare_anchor_aligned(
        truth,
        np.ones(truth.shape, dtype=np.float32),
        _anchor((64.0, 64.0), truth.shape),
        truth,
    )
    metrics = frc50_metrics(prepared)
    assert metrics["amplitude_frc50_frequency"] == pytest.approx(0.495)
    assert metrics["phase_frc50_frequency"] == pytest.approx(0.495)
    assert metrics["amplitude_frc50_pixels"] == pytest.approx(1.0 / 0.495)
    assert metrics["phase_frc50_pixels"] == pytest.approx(1.0 / 0.495)


def test_truth_origin_selects_scanned_object_region_for_metrics():
    """Object-centered scan_com must sample the scanned truth region.

    Without ``truth_origin`` the array-relative mapping evaluates the mostly
    unscanned top-left truth region (the cameraman metric bug); threading the
    source-object center offset must select the scanned region instead.
    """
    from ptycho_torch.reconstruction_evaluation import (
        image_quality_metrics,
        prepare_anchor_aligned,
    )

    truth = _center_block_truth((96, 96))
    canvas = np.array(truth[16:80, 16:80], copy=True)
    weights = np.ones(canvas.shape, dtype=np.float32)
    anchor = _anchor((0.0, 0.0), canvas.shape)
    anchored = dict(anchor)
    anchored["truth_origin"] = [48, 48]

    prepared = prepare_anchor_aligned(canvas, weights, anchored, truth)
    quality = image_quality_metrics(prepared)
    assert quality.amplitude_pearson == pytest.approx(1.0, abs=1e-6)

    buggy = image_quality_metrics(
        prepare_anchor_aligned(canvas, weights, anchor, truth)
    )
    assert buggy.amplitude_pearson < 0.5


def test_global_phase_factor_is_unit_and_wrapped_mae_crosses_phase_boundary():
    from ptycho_torch.reconstruction_evaluation import (
        PreparedComparison,
        Bounds,
        global_phase_factor,
        phase_wrapped_mae,
    )

    target_phase = np.asarray([[3.0, -3.0, 0.0, 0.2, -0.4, 1.0, -1.0]])
    residual = np.asarray([[0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = np.exp(1j * target_phase)
    reconstruction = np.exp(1j * (target_phase + residual))
    factor = global_phase_factor(reconstruction, target)
    assert abs(factor) == pytest.approx(1.0)
    prepared = PreparedComparison(
        reconstruction=reconstruction,
        target=target,
        common_mask=np.ones(reconstruction.shape, dtype=bool),
        ssim_bounds=Bounds(0, 0, 1, 7),
        frc_bounds=Bounds(0, 3, 1, 4),
    )
    assert phase_wrapped_mae(prepared) == pytest.approx(0.4 / 7.0, rel=0.02)


def test_evaluate_quality_writes_raw_metrics_and_fixed_six_panel_png(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        METRIC_CONTRACT_VERSION,
        evaluate_reconstruction_quality,
    )
    from skimage.metrics import structural_similarity

    values = _evaluation_inputs()
    result = evaluate_reconstruction_quality(
        **values,
        output_dir=tmp_path / "reconstruction",
    )

    assert result.metrics_path == tmp_path / "reconstruction" / "metrics.json"
    assert result.comparison_path == tmp_path / "reconstruction" / "comparison.png"
    assert not (tmp_path / "reconstruction" / "diagnostics.json").exists()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload == result.metrics
    assert payload["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert set(payload) == {
        "metric_contract_version",
        "amplitude_ssim",
        "phase_ssim",
        "absolute_amp_mae",
        "phase_wrapped_mae",
        "amplitude_frc50_frequency",
        "phase_frc50_frequency",
        "amplitude_frc50_pixels",
        "phase_frc50_pixels",
        "frc_curves",
        "valid_pixel_count",
        "alignment",
        "amp_mae",
        "gauge",
    }
    target = values["truth"][:9, :9]
    # The canvas is 2 * truth * exp(i * (0.35 + ramp)); the global gauge
    # removes the scale, the constant, and the ramp, so the gauged metrics are
    # exact and the absolute amplitude diagnostic keeps the scale error.
    assert payload["absolute_amp_mae"] == pytest.approx(float(np.mean(np.abs(target))))
    assert payload["amp_mae"] == pytest.approx(0.0, abs=1e-6)
    assert payload["phase_wrapped_mae"] == pytest.approx(0.0, abs=1e-6)
    assert payload["amplitude_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert payload["phase_ssim"] == pytest.approx(1.0, abs=1e-6)
    gauge = payload["gauge"]
    assert gauge["method"] == "global_scale_affine_phase_v1"
    assert gauge["scale"] == pytest.approx(0.5, rel=1e-6)
    assert gauge["phase_offset"] == pytest.approx(-0.15, abs=1e-6)  # 0.35 + ramp origin -0.2
    assert gauge["ramp_x_rad_per_px"] == pytest.approx(-0.005, abs=1e-6)
    assert gauge["ramp_y_rad_per_px"] == pytest.approx(-0.045, abs=1e-6)
    assert gauge["ramp_span_rad"] == pytest.approx(math.hypot(8 * 0.005, 8 * 0.045), rel=1e-5)

    prescale = result.metric_validity["prescale_metrics"]
    assert set(prescale) == {
        "amplitude_ssim",
        "phase_ssim",
        "absolute_amp_mae",
        "phase_wrapped_mae",
    }
    assert result.metric_validity["post_varpro_metrics"] == {
        name: payload[name]
        for name in (
            "amplitude_ssim",
            "phase_ssim",
            "absolute_amp_mae",
            "amp_mae",
            "phase_wrapped_mae",
            "amplitude_frc50_frequency",
            "phase_frc50_frequency",
            "amplitude_frc50_pixels",
            "phase_frc50_pixels",
            "frc_curves",
        )
    }
    assert result.metric_validity["valid"] is True
    assert result.metric_validity["quality_gate_canvas"] == "post_varpro"
    assert result.metric_validity["prescale_role"] == "diagnostic_only"
    assert result.metric_validity["prescale_metrics_status"]["status"] == "complete"
    assert prescale["absolute_amp_mae"] > 0.0
    json.dumps(result.metric_validity, allow_nan=False)
    assert payload["alignment"]["translation_registration"] == "none"
    assert payload["alignment"]["object_center_crop"] is False

    expected_panels = [
        "amplitude_truth",
        "amplitude_reconstruction",
        "amplitude_absolute_error",
        "phase_truth",
        "phase_reconstruction",
        "phase_wrapped_error",
    ]
    assert result.render["panels"] == expected_panels
    assert result.render["aligned_shape"] == [9, 9]
    assert result.render["valid_pixel_count"] == 81
    assert result.render["png_width"] == 2100
    assert result.render["png_height"] == 1350
    with Image.open(result.comparison_path) as image:
        assert image.size == (2100, 1350)
        pixels = np.asarray(image)
    assert np.isfinite(pixels).all()
    assert float(np.var(pixels)) > 0.0
    assert set(result.render["panel_bounds"]) == set(expected_panels)
    for panel_name in expected_panels:
        bounds = result.render["panel_bounds"][panel_name]
        region = pixels[
            bounds["top"] : bounds["bottom"],
            bounds["left"] : bounds["right"],
        ]
        assert region.size > 0
        assert np.any(region[..., :3] < 250), panel_name
    assert sorted(path.name for path in result.metrics_path.parent.iterdir()) == [
        "comparison.png",
        "metrics.json",
    ]


def test_evaluate_quality_accepts_declared_single_channel_groups(tmp_path):
    from ptycho_torch.reconstruction_evaluation import evaluate_reconstruction_quality

    values = _evaluation_inputs()
    values["channel_indices"] = np.arange(8, dtype=np.int64).reshape(8, 1)

    result = evaluate_reconstruction_quality(
        **values,
        expected_channels=1,
        output_dir=tmp_path / "reconstruction",
    )

    assert result.metric_validity["channel_groups"] == {
        "group_count": 8,
        "channel_count": 1,
        "all_groups_distinct": True,
    }


def test_evaluate_quality_rejects_incomplete_source_participant_union(tmp_path):
    """Complete center coverage cannot hide an unused source measurement."""
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    rows = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 0],
            [6, 7, 0, 1],
            [7, 0, 1, 2],
            [0, 2, 4, 6],
        ],
        dtype=np.int64,
    )
    values["channel_indices"] = rows
    values["reassembly"] = _reassembly(
        patches=int(rows.size),
        used_scan_ids=tuple(range(8)),
        used_center_scan_ids=tuple(range(9)),
        expected_scan_ids=tuple(range(9)),
        filtered_eligible_scan_ids=tuple(range(9)),
    )

    with pytest.raises(MetricError, match="complete scan and center utilization"):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path / "reconstruction",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda values: values.update(complex_canvas=np.ones((9, 9), complex)),
            "variance",
        ),
        (
            lambda values: values.update(
                channel_indices=np.asarray([[0, 1, 1, 3]], dtype=np.int64)
            ),
            "distinct",
        ),
        (lambda values: values.update(reassembly=_reassembly(patches=0)), "patch"),
        (
            lambda values: values.update(reassembly=_reassembly(precision="16-mixed")),
            "32-true",
        ),
        (lambda values: values.update(canvas_weights=np.zeros((9, 9))), "positive"),
        (
            lambda values: values["complex_canvas"].__setitem__((0, 0), np.nan),
            "finite",
        ),
    ],
)
def test_evaluate_quality_rejects_invalid_scoring_evidence(tmp_path, mutation, message):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    mutation(values)
    with pytest.raises(MetricError, match=message):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path / "reconstruction",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda values: values.update(reassembly=_reassembly(accepted_patches=7)),
            "accepted == total",
        ),
        (
            lambda values: values.update(reassembly=_reassembly(s1=np.inf)),
            "s1/s2",
        ),
        (
            lambda values: values.update(
                reassembly=_reassembly(
                    count_metrics=SimpleNamespace(status="not_evaluated")
                )
            ),
            "not_applicable",
        ),
        (
            lambda values: values.update(
                reassembly=_reassembly(used_scan_ids=tuple(range(7)))
            ),
            "match every accepted",
        ),
        (
            lambda values: values.update(
                reassembly=_reassembly(used_center_scan_ids=(0,))
            ),
            "complete scan",
        ),
        (
            lambda values: values.update(
                channel_indices=np.asarray([[0, 1, 2]], dtype=np.int64)
            ),
            "C4",
        ),
        (
            lambda values: values.update(
                channel_indices=np.asarray([[0.0, 1.0, 2.5, 3.0]])
            ),
            "integers",
        ),
        (
            lambda values: values.update(
                channel_indices=np.arange(20, 28, dtype=np.int64).reshape(2, 4)
            ),
            "match every accepted",
        ),
        (
            lambda values: values.update(
                canvas_weights=np.pad(
                    np.ones((6, 9), dtype=np.float32),
                    ((0, 3), (0, 0)),
                )
            ),
            "7-by-7",
        ),
        (
            lambda values: values.update(
                canvas_weights=np.full((9, 9), -1.0, dtype=np.float32)
            ),
            "nonnegative",
        ),
        (
            lambda values: values.update(
                prescale_canvas=np.ones((8, 8), dtype=np.complex64)
            ),
            "shape",
        ),
        (
            lambda values: values["prescale_canvas"].__setitem__((0, 0), np.inf),
            "finite",
        ),
    ],
)
def test_evaluate_quality_enforces_every_runtime_validity_gate(
    tmp_path, mutation, message
):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    mutation(values)
    with pytest.raises(MetricError, match=message):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path / "reconstruction",
        )
    assert not (tmp_path / "reconstruction" / "metrics.json").exists()
    assert not (tmp_path / "reconstruction" / "comparison.png").exists()


def test_renderer_uses_the_six_aligned_raw_panels(monkeypatch, tmp_path):
    import matplotlib.axes

    from ptycho_torch.reconstruction_evaluation import (
        global_phase_factor,
        prepare_anchor_aligned,
        render_comparison,
    )

    values = _evaluation_inputs()
    prepared = prepare_anchor_aligned(
        values["complex_canvas"],
        values["canvas_weights"],
        values["canvas_anchor"],
        values["truth"],
    )
    factor = global_phase_factor(
        prepared.reconstruction,
        prepared.target,
        prepared.common_mask,
    )
    captured = []
    original = matplotlib.axes.Axes.imshow

    def recording_imshow(axis, image, *args, **kwargs):
        captured.append(
            (
                np.asarray(np.ma.filled(image, np.nan)),
                kwargs.get("vmin"),
                kwargs.get("vmax"),
            )
        )
        return original(axis, image, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", recording_imshow)
    render = render_comparison(
        prepared,
        tmp_path / "comparison.png",
        gauge_factor=factor,
    )

    assert len(captured) == 6
    aligned = factor * prepared.reconstruction
    expected = (
        np.abs(prepared.target),
        np.abs(aligned),
        np.abs(np.abs(aligned) - np.abs(prepared.target)),
        np.angle(prepared.target),
        np.angle(aligned),
        np.angle(np.exp(1j * (np.angle(aligned) - np.angle(prepared.target)))),
    )
    for (observed, _vmin, _vmax), wanted in zip(captured, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=0, atol=1e-12)
    assert captured[0][1:] == captured[1][1:]
    assert captured[3][1:] == (-math.pi, math.pi)
    assert captured[4][1:] == (-math.pi, math.pi)
    assert captured[5][1:] == (-math.pi, math.pi)
    assert render["panels"] == [
        "amplitude_truth",
        "amplitude_reconstruction",
        "amplitude_absolute_error",
        "phase_truth",
        "phase_reconstruction",
        "phase_wrapped_error",
    ]


@pytest.mark.parametrize("serialized", [False, True])
def test_evaluator_accepts_live_and_serialized_reassembly_diagnostics(
    tmp_path, monkeypatch, serialized
):
    import torch

    from ptycho_torch import reconstruction_evaluation as evaluation
    from ptycho_torch.reassembly_diagnostics import (
        ReassemblyDiagnostics,
        not_applicable,
    )

    values = _evaluation_inputs()
    diagnostics = ReassemblyDiagnostics.legacy_not_applicable(
        effective_probe_mask=torch.ones((9, 9)),
        inference_time=0.1,
        assembly_time=0.2,
        solve_time=0.3,
        s1=1.2,
        s2=0.8,
        scale_profile="legacy_v1",
        canvas_anchor=values["canvas_anchor"],
        canvas_weights=torch.as_tensor(values["canvas_weights"]),
        accepted_patches=8,
        total_patches=8,
        count_metrics=not_applicable(),
        used_scan_ids=tuple(range(8)),
        used_center_scan_ids=(0, 4),
        expected_scan_ids=tuple(range(8)),
        filtered_eligible_scan_ids=(0, 4),
        effective_precision="32-true",
    )
    values["reassembly"] = diagnostics.to_jsonable() if serialized else diagnostics

    def fake_render(_prepared, output_path, *, gauge_factor):
        assert abs(gauge_factor) == pytest.approx(1.0)
        Path(output_path).write_bytes(b"png")
        return {"renderer_version": "test-renderer"}

    from pathlib import Path

    monkeypatch.setattr(evaluation, "render_comparison", fake_render)
    result = evaluation.evaluate_reconstruction_quality(
        **values,
        output_dir=tmp_path / "reconstruction",
    )

    assert result.metrics_path.is_file()
    assert result.comparison_path.read_bytes() == b"png"


def test_degenerate_prescale_is_recorded_with_finite_sentinels_not_gated(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    values["prescale_canvas"] = np.zeros((9, 9), dtype=np.complex64)
    result = evaluate_reconstruction_quality(
        **values,
        output_dir=tmp_path / "reconstruction",
    )

    metrics = result.metric_validity["prescale_metrics"]
    status = result.metric_validity["prescale_metrics_status"]
    assert status["status"] == "partial_sentinel"
    assert set(status["undefined"]) == {
        "amplitude_ssim",
        "phase_ssim",
        "phase_wrapped_mae",
    }
    assert metrics["amplitude_ssim"] == -1.0
    assert metrics["phase_ssim"] == -1.0
    assert metrics["phase_wrapped_mae"] == pytest.approx(math.pi)
    assert metrics["absolute_amp_mae"] > 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_render_failure_preserves_prior_evaluation_artifacts(tmp_path, monkeypatch):
    from ptycho_torch import reconstruction_evaluation as evaluation

    output_dir = tmp_path / "reconstruction"
    output_dir.mkdir()
    metrics_path = output_dir / "metrics.json"
    comparison_path = output_dir / "comparison.png"
    metrics_path.write_bytes(b"old metrics")
    comparison_path.write_bytes(b"old comparison")

    def fail_render(_prepared, output_path, *, gauge_factor):
        assert Path(output_path).parent != output_dir
        assert abs(gauge_factor) == pytest.approx(1.0)
        raise RuntimeError("render failed")

    from pathlib import Path

    monkeypatch.setattr(evaluation, "render_comparison", fail_render)
    with pytest.raises(RuntimeError, match="render failed"):
        evaluation.evaluate_reconstruction_quality(
            **_evaluation_inputs(),
            output_dir=output_dir,
        )

    assert metrics_path.read_bytes() == b"old metrics"
    assert comparison_path.read_bytes() == b"old comparison"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "comparison.png",
        "metrics.json",
    ]


def test_renderer_closes_figure_when_panel_rendering_raises(tmp_path, monkeypatch):
    import matplotlib.axes
    from matplotlib import pyplot as plt

    from ptycho_torch.reconstruction_evaluation import (
        prepare_anchor_aligned,
        render_comparison,
    )

    values = _evaluation_inputs()
    prepared = prepare_anchor_aligned(
        values["complex_canvas"],
        values["canvas_weights"],
        values["canvas_anchor"],
        values["truth"],
    )
    existing_figures = set(plt.get_fignums())
    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "imshow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("imshow failed")),
    )

    with pytest.raises(RuntimeError, match="imshow failed"):
        render_comparison(prepared, tmp_path / "comparison.png")

    assert set(plt.get_fignums()) == existing_figures
    assert not (tmp_path / "comparison.png").exists()


def test_production_metric_import_is_plotting_and_study_independent():
    code = """
import sys
import ptycho_torch.reconstruction_evaluation
for prefix in ('scripts.studies', 'matplotlib', 'tensorflow'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


# --- Count-intensity (CI) contract ------------------------------------------


def _fitted_count_metrics(**overrides):
    from ptycho_torch.reassembly_diagnostics import FittedCountMetrics

    values = {
        "relative_l2_intensity_error": 0.031,
        "mean_raw_poisson_nll": 1234.5,
        "n_samples": 8,
        "n_pixels": 8 * 16384,
        "effective_mask_digest": "a" * 64,
        "sample_ids": tuple(range(8)),
    }
    values.update(overrides)
    return FittedCountMetrics(**values)


def test_count_contract_evaluation_accepts_fitted_count_diagnostics(tmp_path):
    from ptycho_torch.reconstruction_evaluation import evaluate_reconstruction_quality

    values = _evaluation_inputs()
    values["reassembly"] = _reassembly(count_metrics=_fitted_count_metrics())

    result = evaluate_reconstruction_quality(
        **values,
        output_dir=tmp_path,
        measurement_domain="count_intensity",
    )

    assert Path(result.metrics_path).is_file()
    assert Path(result.comparison_path).is_file()


def test_count_contract_tiled_assembly_still_requires_fitted_diagnostics(tmp_path):
    from ptycho_torch.reconstruction_evaluation import evaluate_reconstruction_quality

    values = _evaluation_inputs()
    values["canvas_anchor"] = {
        **values["canvas_anchor"],
        "truth_origin": [0, 0],
        "assembly_method": "tiled_raster_v1",
    }
    values["reassembly"] = _reassembly(
        count_metrics=_fitted_count_metrics()
    )

    result = evaluate_reconstruction_quality(
        **values,
        output_dir=tmp_path,
        measurement_domain="count_intensity",
        metric_crop_border=1,
    )

    assert result.metric_validity["count_diagnostics"][
        "relative_l2_intensity_error"
    ] == pytest.approx(0.031)
    assert result.metric_validity["quality_gate_canvas"] == (
        "tiled_source_object_gauge"
    )
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["alignment"]["truth_origin"] == [0, 0]
    assert metrics["alignment"]["metric_crop_border"] == 1
    assert metrics["alignment"]["aligned_shape"] == [9, 9]
    bounds = metrics["alignment"]["ssim_bounds"]
    assert bounds["bottom"] - bounds["top"] == 7
    assert bounds["right"] - bounds["left"] == 7


def test_count_contract_rejects_tiled_not_evaluated_marker(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    values["canvas_anchor"] = {
        **values["canvas_anchor"],
        "assembly_method": "tiled_raster_v1",
    }
    values["reassembly"] = _reassembly(
        count_metrics={
            "status": "not_evaluated",
            "reason": "legacy_tiled_reconstruction",
        }
    )

    with pytest.raises(MetricError, match="count"):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path,
            measurement_domain="count_intensity",
        )


def test_count_contract_evaluation_rejects_legacy_not_applicable(tmp_path):
    """A CI run whose count diagnostics never ran must fail closed."""

    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()  # default is the legacy not_applicable marker

    with pytest.raises(MetricError, match="count"):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path,
            measurement_domain="count_intensity",
        )


def test_count_contract_evaluation_rejects_deferred_count_diagnostics(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    values["reassembly"] = _reassembly(
        count_metrics=SimpleNamespace(status="not_evaluated")
    )

    with pytest.raises(MetricError, match="count"):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path,
            measurement_domain="count_intensity",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(sample_ids=list(range(1, 9))), "sample_ids"),
        (lambda record: record.update(effective_mask_digest="not-a-digest"), "digest"),
        (lambda record: record.update(relative_l2_intensity_error=-0.1), "nonnegative"),
    ],
)
def test_count_contract_rejects_unbound_fitted_evidence(
    tmp_path,
    mutation,
    message,
):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    record = _fitted_count_metrics().to_jsonable()
    mutation(record)
    values["reassembly"] = _reassembly(count_metrics=record)

    with pytest.raises(MetricError, match=message):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path,
            measurement_domain="count_intensity",
        )


def test_count_contract_rejects_mask_digest_mismatch(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    values["reassembly"] = _reassembly(
        count_metrics=_fitted_count_metrics(effective_mask_digest="a" * 64),
        mask_digest="b" * 64,
    )

    with pytest.raises(MetricError, match="effective_mask_digest.*mask_digest"):
        evaluate_reconstruction_quality(
            **values,
            output_dir=tmp_path,
            measurement_domain="count_intensity",
        )


def test_amplitude_contract_still_rejects_fitted_count_diagnostics(tmp_path):
    """The legacy gate is unchanged: amplitude runs must not carry CI metrics."""

    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    values = _evaluation_inputs()
    values["reassembly"] = _reassembly(count_metrics=_fitted_count_metrics())

    with pytest.raises(MetricError, match="not_applicable"):
        evaluate_reconstruction_quality(**values, output_dir=tmp_path)


def test_evaluation_rejects_an_unknown_measurement_domain(tmp_path):
    from ptycho_torch.reconstruction_evaluation import (
        MetricError,
        evaluate_reconstruction_quality,
    )

    with pytest.raises(MetricError, match="measurement_domain"):
        evaluate_reconstruction_quality(
            **_evaluation_inputs(),
            output_dir=tmp_path,
            measurement_domain="photons",
        )
