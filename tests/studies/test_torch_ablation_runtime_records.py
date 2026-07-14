"""Host-transfer regression coverage for canonical run array artifacts."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.studies.ablation import metrics
from scripts.studies.ablation import runtime_records as records_api
from scripts.studies.ablation.manifest import (
    FrozenDict,
    Gate,
    ResolvedGate,
    RuleApplicability,
)
from scripts.studies.ablation.runtime_records import (
    _image_records,
    _runtime_records,
    _stability_records,
    flat_metrics,
    npy_bytes,
    training_history_records,
)
from scripts.studies.ablation.verdicts import (
    AttemptRow,
    AttemptStatus,
    CompletionState,
    GateResult,
    Verdict,
    evaluate_gate,
)


class _DeviceTensorLike:
    """Tensor-like value that must transfer to host before NumPy conversion."""

    def __init__(self) -> None:
        self.detached = False
        self.host_transferred = False

    def detach(self) -> _DeviceTensorLike:
        self.detached = True
        return self

    def cpu(self) -> _DeviceTensorLike:
        self.host_transferred = True
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray([[1.0, 2.0]], dtype=np.float32)

    def __array__(self, dtype: object = None) -> np.ndarray:
        raise TypeError("device data must transfer to host before NumPy conversion")


def test_npy_bytes_transfers_tensor_like_values_to_host() -> None:
    value = _DeviceTensorLike()

    payload = npy_bytes(value)

    assert value.detached is True
    assert value.host_transferred is True
    assert np.array_equal(
        np.load(io.BytesIO(payload), allow_pickle=False), [[1.0, 2.0]]
    )


def _runtime_result(peak_memory_bytes: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        train_seconds=4.0,
        peak_memory_bytes=peak_memory_bytes,
        reloaded_diagnostics=SimpleNamespace(
            inference_time=2.0,
            assembly_time=1.0,
        ),
    )


def test_runtime_records_route_framework_cuda_peak_memory_exactly() -> None:
    records = _runtime_records(_runtime_result(987_654_321))

    peak = next(
        record for record in records if record.path == "runtime.peak_memory_bytes"
    )
    assert peak.value == 987_654_321.0
    assert peak.basis == "framework_cuda_allocator"
    assert peak.alignment == "none"
    # Canonical ablation preflight permits one effective device, so this is the
    # selected device allocator peak rather than an unaggregated rank-local value.
    assert peak.basis != "process_gpu_memory"


def test_runtime_records_omit_peak_memory_for_cpu_not_applicable() -> None:
    records = _runtime_records(_runtime_result(None))

    assert "runtime.peak_memory_bytes" not in {record.path for record in records}


def test_stability_records_publish_reload_allclose_for_texture_and_canvas() -> None:
    reference_texture = np.asarray([1.0, 2.0])
    reference_canvas = np.asarray([[3.0, 4.0]])
    result = SimpleNamespace(
        reference_texture=reference_texture,
        reloaded_texture=reference_texture + 5e-7,
        reference_canvas=reference_canvas,
        reloaded_canvas=reference_canvas + 5e-7,
        reload_max_abs_error=5e-7,
        reload_allclose=True,
        reloaded_diagnostics=SimpleNamespace(
            canvas_weights=np.ones((1, 2)), patches_accepted=2, patches_total=2
        ),
    )

    records = flat_metrics(tuple(_stability_records(result)))

    assert records["stability.reload_allclose"] == 1.0


def test_image_records_emit_anchor_aligned_mean_scaled_amp_ssim_for_truth_gate() -> (
    None
):
    y, x = np.mgrid[:9, :10]
    truth = (2.0 + x + 2.0 * y).astype(np.complex64)
    reconstruction = (0.7 * truth + 0.15 * y).astype(np.complex64)
    result = SimpleNamespace(
        reloaded_canvas=reconstruction,
        reloaded_texture=0.8 * reconstruction,
        reloaded_diagnostics=SimpleNamespace(
            canvas_weights=np.ones(reconstruction.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": np.asarray((5.0, 4.0)),
                "canvas_shape": reconstruction.shape,
                "canvas_origin_offset": (0.0, 0.0),
            },
        ),
    )

    records, _ = _image_records(result, "object_truth", truth)

    amp_ssim = next(record for record in records if record.path.endswith(".amp_ssim"))
    prepared = metrics.prepare_anchor_aligned(
        reconstruction,
        result.reloaded_diagnostics.canvas_weights,
        result.reloaded_diagnostics.canvas_anchor,
        truth,
    )
    assert amp_ssim.path == "truth_quality.amp_ssim"
    assert amp_ssim.value == pytest.approx(metrics.amplitude_ssim(prepared))
    assert amp_ssim.basis == "mean_scaled_amplitude"
    assert amp_ssim.alignment == "anchor_common_mask_largest_valid_rectangle"

    gate = ResolvedGate(
        Gate(
            "truth_ssim",
            FrozenDict(),
            "ge",
            "truth_quality.amp_ssim",
            "median",
            -1.0,
            1,
        ),
        "synthetic",
        "arm",
        RuleApplicability.ACTIVE,
    )
    result = evaluate_gate(
        gate,
        (
            AttemptRow(
                "run-1",
                "arm",
                "synthetic",
                1,
                AttemptStatus.SUCCESS,
                CompletionState.TERMINAL,
                flat_metrics(tuple(records)),
            ),
        ),
        requested_seeds=(1,),
        status_result=GateResult.active("seed_success", Verdict.PASS),
    )
    assert result.verdict is Verdict.PASS
    assert result.reason != "missing_or_invalid_operand"


def test_image_records_emit_quality_before_and_after_varpro() -> None:
    y, x = np.mgrid[:9, :10]
    truth = (1.0 + 0.02 * x + 0.03 * y) * np.exp(1j * (x - y) / 20.0)
    texture = 0.8 * truth
    scaled = np.full_like(truth, np.mean(np.abs(truth)))
    result = SimpleNamespace(
        reloaded_texture=texture,
        reloaded_canvas=scaled,
        reloaded_diagnostics=SimpleNamespace(
            canvas_weights=np.ones(truth.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": np.asarray((5.0, 4.0)),
                "canvas_shape": truth.shape,
                "canvas_origin_offset": (0.0, 0.0),
            },
        ),
    )

    records, _ = _image_records(result, "object_truth", truth)
    values = flat_metrics(tuple(records))

    assert values["truth_quality.pre_varpro.amp_ssim"] > values[
        "truth_quality.post_varpro.amp_ssim"
    ]
    assert values["truth_quality.pre_varpro.phase_wrapped_mae"] < values[
        "truth_quality.post_varpro.phase_wrapped_mae"
    ]


def test_stability_records_emit_valid_mask_collapse_and_saturation_metrics() -> None:
    texture = np.full((6, 6), 1.2 + 1.2j)
    result = SimpleNamespace(
        reference_texture=texture,
        reloaded_texture=texture,
        reference_canvas=texture,
        reloaded_canvas=texture,
        reload_max_abs_error=0.0,
        reload_allclose=True,
        reloaded_diagnostics=SimpleNamespace(
            canvas_weights=np.ones(texture.shape),
            patches_accepted=36,
            patches_total=36,
            used_scan_ids=(0, 1, 2),
            used_center_scan_ids=(0, 1, 2),
            expected_scan_ids=tuple(range(10)),
            filtered_eligible_scan_ids=(0, 1, 2, 3),
            decoder_real_saturation_fraction=0.75,
            decoder_imag_saturation_fraction=0.5,
            decoder_real_lower_saturation_fraction=0.25,
            decoder_real_upper_saturation_fraction=0.5,
            decoder_imag_lower_saturation_fraction=0.125,
            decoder_imag_upper_saturation_fraction=0.375,
        ),
    )

    values = flat_metrics(tuple(_stability_records(result)))

    assert values["stability.amp_variance"] == pytest.approx(0.0)
    assert values["stability.phase_variance"] == pytest.approx(0.0)
    assert values["stability.real_head_saturation_fraction"] == pytest.approx(0.75)
    assert values["stability.imag_head_saturation_fraction"] == pytest.approx(0.5)
    assert values["stability.real_head_lower_saturation_fraction"] == pytest.approx(0.25)
    assert values["stability.real_head_upper_saturation_fraction"] == pytest.approx(0.5)
    assert values["stability.imag_head_lower_saturation_fraction"] == pytest.approx(0.125)
    assert values["stability.imag_head_upper_saturation_fraction"] == pytest.approx(0.375)
    assert values["stability.unique_scans_used"] == 3
    assert values["stability.unique_centers_used"] == 3
    assert values["stability.unique_scans_expected"] == 10
    assert values["stability.scan_utilization_fraction"] == pytest.approx(0.3)
    assert values["stability.unique_scans_filtered_eligible"] == 4
    assert values["stability.filtered_scan_utilization_fraction"] == pytest.approx(0.75)


def test_stability_records_suppress_center_utilization_when_identity_unavailable():
    result = SimpleNamespace(
        reloaded_diagnostics=SimpleNamespace(
            canvas_weights=np.ones((2, 2)),
            patches_accepted=2,
            patches_total=2,
            used_scan_ids=(0, 2),
            used_center_scan_ids=(),
            center_identity_available=False,
            expected_scan_ids=(0, 1, 2, 3),
            filtered_eligible_scan_ids=(0, 1, 2),
        ),
        reference_canvas=np.ones((2, 2), dtype=np.complex64),
        reference_texture=np.ones((2, 2), dtype=np.complex64),
        reloaded_canvas=np.ones((2, 2), dtype=np.complex64),
        reloaded_texture=np.ones((2, 2), dtype=np.complex64),
        reload_max_abs_error=0.0,
        reload_allclose=True,
    )

    values = flat_metrics(tuple(records_api._stability_records(result)))

    assert values["stability.unique_scans_used"] == 2
    assert values["stability.unique_scans_expected"] == 4
    assert values["stability.scan_utilization_fraction"] == pytest.approx(0.5)
    assert "stability.unique_centers_used" not in values
    assert "stability.filtered_scan_utilization_fraction" not in values


def test_training_history_records_emit_validation_lr_and_checkpoint_evidence() -> None:
    records = training_history_records(
        {
            "train_loss_name": "train_loss",
            "val_loss_name": "val_loss",
            "series": {
                "train_loss_epoch": {"value": [2.0, 1.5, 1.0, 0.8]},
                "val_loss": {"value": [2.0, 1.4, 1.1, 1.0]},
                "lr-Adam": {"value": [2e-4, 2e-4, 1e-4, 1e-4]},
                "grad_norm_preclip_step": {"value": [2.0, 1.0]},
            },
        },
        checkpoint_sha256="a" * 64,
        checkpoint_epoch=3,
    )
    values = flat_metrics(records)

    assert values["stability.validation_loss_final"] == pytest.approx(1.0)
    assert values["stability.validation_loss_tail_normalized_slope"] < 0.0
    assert values["stability.learning_rate_initial"] == pytest.approx(2e-4)
    assert values["stability.learning_rate_final"] == pytest.approx(1e-4)
    assert values["stability.learning_rate_reduction_count"] == 1
    assert values["stability.checkpoint_identity_present"] == 1.0
    assert values["stability.checkpoint_epoch"] == 3


def test_milestone_records_reuse_checkpoint_metric_record_machinery(monkeypatch) -> None:
    result = SimpleNamespace(
        checkpoint_sha256="b" * 64,
        checkpoint_epoch=19,
        training_history={"series": {}},
        train_seconds=4.0,
        peak_memory_bytes=None,
    )
    resolved = SimpleNamespace(ci_scaling_active=False)
    descriptor = SimpleNamespace(truth_location="none")
    calls = []

    def checkpoint_records(resolved_arg, result_arg, descriptor_arg, *, runtime):
        calls.append((resolved_arg, result_arg, descriptor_arg, runtime))
        return ((), {})

    monkeypatch.setattr(records_api, "_checkpoint_metric_records", checkpoint_records)

    records, arrays = records_api.build_milestone_metric_records(
        resolved, result, descriptor
    )

    assert (records, arrays) == ((), {})
    assert calls == [(resolved, result, descriptor, False)]


def test_truth_forward_poisson_oracle_matches_repeated_loader_sample_stream_and_caches(
    tmp_path, monkeypatch
) -> None:
    patches = np.asarray(
        [
            [[1.0 + 0.0j, 0.8 + 0.1j], [0.7 - 0.2j, 1.1 + 0.0j]],
            [[0.9 + 0.2j, 1.0 + 0.0j], [0.6 + 0.1j, 0.8 - 0.1j]],
        ],
        dtype=np.complex64,
    )
    probe = np.asarray([[2.0 + 0.0j, 1.0 + 0.0j], [1.5 + 0.0j, 0.5 + 0.0j]])
    exit_waves = patches[:, None] * probe[None, None]
    expected = np.fft.fftshift(
        np.abs(np.fft.fft2(exit_waves, axes=(-2, -1), norm="ortho")) ** 2,
        axes=(-2, -1),
    ).sum(axis=1)
    observed = np.rint(expected + np.asarray([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])).astype(np.uint16)
    source = tmp_path / "synthetic.npz"
    np.savez(
        source,
        diff3d=observed,
        ground_truth_patches=patches[..., None],
        probeGuess=probe[None].astype(np.complex64),
    )
    descriptor = SimpleNamespace(
        test=source,
        measurement_key="diff3d",
        truth="object_truth",
    )
    result = SimpleNamespace(
        count_metrics=SimpleNamespace(
            relative_l2_intensity_error=0.25,
            sample_ids=(1, 0, 1),
            sample_identity_digest="b" * 64,
        )
    )
    records_api._cached_poisson_oracle.cache_clear()
    fft_calls = 0
    fft2 = np.fft.fft2

    def counted_fft2(*args, **kwargs):
        nonlocal fft_calls
        fft_calls += 1
        return fft2(*args, **kwargs)

    monkeypatch.setattr(np.fft, "fft2", counted_fft2)

    first = records_api._truth_forward_poisson_oracle_records(descriptor, result)
    second = records_api._truth_forward_poisson_oracle_records(descriptor, result)
    values = flat_metrics(tuple(first))

    assert second == first
    assert fft_calls == 1
    assert values["measurement_consistency.poisson_oracle_relative_l2_error"] > 0
    assert values[
        "measurement_consistency.model_to_poisson_oracle_error_ratio"
    ] == pytest.approx(
        0.25 / values["measurement_consistency.poisson_oracle_relative_l2_error"]
    )
