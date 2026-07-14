import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from ptycho_torch import reassembly
from ptycho_torch import train_lightning_only as train_module
from ptycho.config.config import PyTorchExecutionConfig
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.reassembly_diagnostics import (
    ConditionStatus,
    FittedCountMetrics,
    NotApplicable,
    ReassemblyDiagnostics,
    VarProSufficientStatistics,
    array_digest,
)


def test_scan_identity_evidence_preserves_grouped_source_ids():
    source = SimpleNamespace(
        model_config=SimpleNamespace(object_big=True),
        valid_indices_per_file=(np.asarray([4, 7, 9]),),
        source_indices_per_file=(np.arange(12),),
    )
    subset = SimpleNamespace(mmap_ptycho={
        "nn_indices": torch.tensor([[9, 4], [4, 9]]),
        "center_scan_id": torch.tensor([7, 7]),
        "center_scan_id_available": torch.tensor([True, True]),
    })

    used, centers, available, filtered, source_ids = reassembly._scan_identity_evidence(
        source, subset, 0
    )

    assert used == (4, 9)
    assert centers == (7,)
    assert available is True
    assert filtered == (4, 7, 9)
    assert source_ids == tuple(range(12))


def test_scan_identity_allows_group_neighbor_outside_bounds_eligible_centers():
    source = SimpleNamespace(
        model_config=SimpleNamespace(object_big=True),
        valid_indices_per_file=(np.asarray([4, 7, 9]),),
        source_indices_per_file=(np.arange(12),),
    )
    subset = SimpleNamespace(mmap_ptycho={
        "nn_indices": torch.tensor([[4, 11]]),
        "center_scan_id": torch.tensor([7]),
        "center_scan_id_available": torch.tensor([True]),
    })

    participating, centers, available, filtered, source_ids = (
        reassembly._scan_identity_evidence(source, subset, 0)
    )

    assert participating == (4, 11)
    assert centers == (7,)
    assert available is True
    assert filtered == (4, 7, 9)
    assert source_ids == tuple(range(12))


def test_scan_identity_evidence_maps_ungrouped_local_ids_to_source_ids():
    source = SimpleNamespace(
        model_config=SimpleNamespace(object_big=False),
        valid_indices_per_file=(np.asarray([4, 7, 9]),),
        source_indices_per_file=(np.arange(12),),
    )
    subset = SimpleNamespace(
        mmap_ptycho={"nn_indices": torch.tensor([[0], [1], [2]])}
    )

    used, centers, available, filtered, source_ids = reassembly._scan_identity_evidence(
        source, subset, 0
    )

    assert used == (4, 7, 9)
    assert centers == (4, 7, 9)
    assert available is True
    assert filtered == (4, 7, 9)
    assert source_ids == tuple(range(12))


def test_decoder_saturation_loop_has_no_scalar_device_transfers():
    import inspect

    source = inspect.getsource(reassembly.reconstruct_image_barycentric)
    timed_loop = source.split("#Actual loop", 1)[1].split(
        "# 2. Finalize texture canvas", 1
    )[0]

    assert ".item()" not in timed_loop


def _known_statistics():
    fitted_z = torch.tensor([4.0, 0.25, 1.0], dtype=torch.float64)
    ata = torch.eye(3, dtype=torch.float64)
    atb = fitted_z[:, None].clone()
    sum_i2 = float(fitted_z @ fitted_z)
    return ata, atb, sum_i2


def test_sufficient_statistics_clone_inputs_and_compute_closed_form_objective():
    ata, atb, sum_i2 = _known_statistics()
    stats = VarProSufficientStatistics(
        ATA=ata,
        ATb=atb,
        sum_i2=sum_i2,
        n_pixels=5,
    )
    ata.zero_()
    atb.zero_()

    expected_z = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    expected = (
        expected_z @ torch.eye(3, dtype=torch.float64) @ expected_z
        - 2 * expected_z @ torch.tensor([4.0, 0.25, 1.0], dtype=torch.float64)
        + sum_i2
    ) / 5

    assert stats.objective(1.0, 1.0) == pytest.approx(float(expected))
    assert stats.objective(2.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert stats.ATA.dtype == torch.float64
    assert stats.ATb.dtype == torch.float64


def test_reassembly_diagnostics_condition_contract_and_json_serialization():
    ata, atb, sum_i2 = _known_statistics()
    mask = torch.tensor([[1.0, 0.5], [0.25, 0.0]], dtype=torch.float32)
    weights = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    stats = VarProSufficientStatistics(
        ata=ata,
        atb=atb,
        sum_i2=sum_i2,
        n_pixels=5,
    )
    metrics = FittedCountMetrics(
        relative_l2_intensity_error=0.0,
        mean_raw_poisson_nll=-1.25,
        n_samples=2,
        n_pixels=8,
        effective_mask_digest=array_digest(mask),
    )
    canvas_anchor = {
        "scan_com": torch.tensor([4.0, 6.0]),
        "canvas_shape": (2, 3),
        "canvas_origin_offset": (0.0, -1.0),
    }

    diagnostics = ReassemblyDiagnostics.from_statistics(
        stats,
        inference_time=1.25,
        assembly_time=0.5,
        solve_time=0.125,
        s1=2.0,
        s2=0.5,
        profile="ci_intensity_v2",
        effective_probe_mask=mask,
        canvas_anchor=canvas_anchor,
        canvas_weights=weights,
        accepted_patches=3,
        total_patches=4,
        count_metrics=metrics,
        effective_precision="bf16-mixed",
        used_scan_ids=(0, 2, 3),
        expected_scan_ids=(0, 1, 2, 3),
        filtered_eligible_scan_ids=(0, 2, 3),
        decoder_real_saturation_fraction=0.25,
        decoder_imag_saturation_fraction=0.5,
        decoder_real_lower_saturation_fraction=0.1,
        decoder_real_upper_saturation_fraction=0.15,
        decoder_imag_lower_saturation_fraction=0.2,
        decoder_imag_upper_saturation_fraction=0.3,
    )
    weights.fill_(-100.0)
    canvas_anchor["scan_com"].zero_()
    canvas_anchor["canvas_shape"] = (99, 99)

    assert diagnostics.schema_version == 1
    assert diagnostics.effective_precision == "bf16-mixed"
    assert diagnostics.profile == "ci_intensity_v2"
    assert diagnostics.condition == pytest.approx(1.0)
    assert diagnostics.condition_number == pytest.approx(1.0)
    assert diagnostics.timing == {
        "inference_time": 1.25,
        "assembly_time": 0.5,
        "solve_time": 0.125,
    }
    assert diagnostics.canvas_weights_metadata["shape"] == [2, 3]
    assert diagnostics.patches_accepted == 3
    assert diagnostics.filtered_eligible_scan_ids == (0, 2, 3)
    assert diagnostics.patches_total == 4
    assert diagnostics.used_scan_ids == (0, 2, 3)
    assert diagnostics.expected_scan_ids == (0, 1, 2, 3)
    assert diagnostics.decoder_real_saturation_fraction == pytest.approx(0.25)
    assert diagnostics.decoder_imag_saturation_fraction == pytest.approx(0.5)
    assert diagnostics.decoder_real_lower_saturation_fraction == pytest.approx(0.1)
    assert diagnostics.decoder_real_upper_saturation_fraction == pytest.approx(0.15)
    assert diagnostics.fitted_objective == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.fitted_objective <= (
        diagnostics.unit_objective
        + 1e-12
        + 1e-10 * abs(diagnostics.unit_objective)
    )
    assert torch.all(diagnostics.canvas_weights >= 0)

    payload = diagnostics.to_jsonable()
    assert payload["schema_version"] == 1
    assert payload["effective_precision"] == "bf16-mixed"
    assert payload["canvas_weights"]["shape"] == [2, 3]
    assert payload["canvas_weights"]["dtype"] == "float32"
    assert len(payload["canvas_weights"]["sha256"]) == 64
    assert len(payload["mask_digest"]) == 64
    assert payload["canvas_anchor"]["scan_com"] == [4.0, 6.0]
    assert payload["count_metrics"]["n_samples"] == 2
    json.dumps(payload)


def test_diagnostic_backing_state_is_immutable_and_serialization_isolated():
    ata, atb, sum_i2 = _known_statistics()
    stats = VarProSufficientStatistics(
        ATA=ata,
        ATb=atb,
        sum_i2=sum_i2,
        n_pixels=5,
    )
    baseline_objective = stats.objective(1.0, 1.0)

    stats_backing = [
        getattr(stats, field.name) for field in dataclasses.fields(stats)
    ]
    for value in stats_backing:
        if isinstance(value, torch.Tensor):
            value.zero_()
    assert not hasattr(stats, "__dict__")
    assert not any(isinstance(value, torch.Tensor) for value in stats_backing)
    assert stats.objective(1.0, 1.0) == baseline_objective

    mask = torch.ones(2, 2)
    diagnostics = ReassemblyDiagnostics.from_statistics(
        stats,
        inference_time=0.0,
        assembly_time=0.0,
        solve_time=0.0,
        s1=2.0,
        s2=0.5,
        profile="ci_intensity_v2",
        effective_probe_mask=mask,
        canvas_anchor={"nested": {"origin": [1.0, 2.0]}},
        canvas_weights=torch.arange(4, dtype=torch.float32).reshape(2, 2),
        accepted_patches=1,
        total_patches=1,
        count_metrics=FittedCountMetrics(
            relative_l2_intensity_error=0.0,
            mean_raw_poisson_nll=0.0,
            n_samples=1,
            n_pixels=4,
            effective_mask_digest=array_digest(mask),
        ),
    )
    baseline_payload = diagnostics.to_jsonable()

    diagnostics_backing = [
        getattr(diagnostics, field.name)
        for field in dataclasses.fields(diagnostics)
    ]
    for value in diagnostics_backing:
        if isinstance(value, torch.Tensor):
            value.zero_()
        elif isinstance(value, dict):
            value.clear()
    leaked_anchor = diagnostics.canvas_anchor
    leaked_anchor["nested"]["origin"][0] = 99.0
    leaked_weights = diagnostics.canvas_weights
    leaked_weights.zero_()
    leaked_payload = diagnostics.to_jsonable()
    leaked_payload["canvas_anchor"]["nested"]["origin"][1] = 99.0

    assert not hasattr(diagnostics, "__dict__")
    assert not any(
        isinstance(value, (torch.Tensor, dict, list))
        for value in diagnostics_backing
    )
    assert diagnostics.to_jsonable() == baseline_payload


def test_reassembly_diagnostics_rejects_a_worse_fitted_objective():
    stats = VarProSufficientStatistics(
        ATA=torch.eye(3, dtype=torch.float64),
        ATb=torch.ones(3, 1, dtype=torch.float64),
        sum_i2=3.0,
        n_pixels=3,
    )

    with pytest.raises(ValueError, match="fitted objective"):
        ReassemblyDiagnostics.from_statistics(
            stats,
            inference_time=0.0,
            assembly_time=0.0,
            solve_time=0.0,
            s1=2.0,
            s2=2.0,
            scale_profile="ci_intensity_v2",
            effective_probe_mask=torch.ones(1, 1),
            canvas_anchor={},
            canvas_weights=torch.ones(1, 1),
            accepted_patches=1,
            total_patches=1,
            count_metrics=None,
        )


@pytest.mark.parametrize(
    "ata",
    [
        torch.zeros(3, 3, dtype=torch.float64),
        torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)),
    ],
    ids=["zero", "rank-deficient"],
)
def test_rank_deficient_condition_is_typed_and_strict_json_safe(ata):
    stats = VarProSufficientStatistics(
        ATA=ata,
        ATb=torch.zeros(3, 1, dtype=torch.float64),
        sum_i2=4.0,
        n_pixels=4,
    )
    mask = torch.ones(1, 1)
    diagnostics = ReassemblyDiagnostics.from_statistics(
        stats,
        inference_time=0.0,
        assembly_time=0.0,
        solve_time=0.0,
        s1=1.0,
        s2=1.0,
        profile="ci_intensity_v2",
        effective_probe_mask=mask,
        canvas_anchor={"scan_com": [0.0, 0.0]},
        canvas_weights=torch.ones(1, 1),
        accepted_patches=1,
        total_patches=1,
        count_metrics=FittedCountMetrics(
            relative_l2_intensity_error=0.0,
            mean_raw_poisson_nll=0.0,
            n_samples=1,
            n_pixels=1,
            effective_mask_digest=array_digest(mask),
        ),
    )

    assert isinstance(diagnostics.condition, ConditionStatus)
    assert diagnostics.condition.status == "rank_deficient"
    assert diagnostics.condition.value is None
    payload = diagnostics.to_jsonable()
    assert payload["condition"] == {
        "status": "rank_deficient",
        "value": None,
        "reason": "nonfinite_condition_number",
    }
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("ATA", torch.full((3, 3), float("nan"), dtype=torch.float64)),
        ("ATb", torch.full((3, 1), float("inf"), dtype=torch.float64)),
        ("sum_i2", float("nan")),
        ("n_pixels", float("inf")),
    ],
)
def test_sufficient_statistics_reject_nonfinite_inputs(field_name, value):
    kwargs = {
        "ATA": torch.eye(3, dtype=torch.float64),
        "ATb": torch.zeros(3, 1, dtype=torch.float64),
        "sum_i2": 1.0,
        "n_pixels": 1,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match="finite"):
        VarProSufficientStatistics(**kwargs)


def test_nonfinite_fitted_objective_and_diagnostic_fields_are_rejected():
    stats = VarProSufficientStatistics(
        ATA=torch.eye(3, dtype=torch.float64),
        ATb=torch.zeros(3, 1, dtype=torch.float64),
        sum_i2=1.0,
        n_pixels=1,
    )
    common = {
        "statistics": stats,
        "inference_time": 0.0,
        "assembly_time": 0.0,
        "solve_time": 0.0,
        "s1": 1.0,
        "s2": 1.0,
        "profile": "ci_intensity_v2",
        "effective_probe_mask": torch.ones(1, 1),
        "canvas_anchor": {"scan_com": [0.0, 0.0]},
        "canvas_weights": torch.ones(1, 1),
        "accepted_patches": 1,
        "total_patches": 1,
        "count_metrics": None,
    }

    with pytest.raises(ValueError, match="finite"):
        ReassemblyDiagnostics.from_statistics(**{**common, "s1": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        ReassemblyDiagnostics.from_statistics(
            **{**common, "inference_time": float("inf")}
        )
    with pytest.raises(ValueError, match="finite"):
        ReassemblyDiagnostics.from_statistics(
            **{
                **common,
                "canvas_anchor": {"scan_com": [float("nan"), 0.0]},
            }
        )
    with pytest.raises(ValueError, match="finite"):
        FittedCountMetrics(
            relative_l2_intensity_error=float("inf"),
            mean_raw_poisson_nll=0.0,
            n_samples=1,
            n_pixels=1,
            effective_mask_digest="0" * 64,
        )


def test_varpro_scaler_sufficient_statistics_are_aggregate_and_isolated():
    scaler = reassembly.VarProScaler("cpu")
    intensity_1 = torch.tensor([[[[3.0, 5.0]]]])
    intensity_2 = torch.tensor([[[[7.0, 11.0]]]])
    basis_1 = (
        torch.tensor([[[[1.0, 2.0]]]]),
        torch.tensor([[[[0.5, 1.5]]]]),
        torch.tensor([[[[-1.0, 0.25]]]]),
    )
    basis_2 = tuple(value * 2 for value in basis_1)

    scaler.accumulate_batch_from_basis(intensity_1, *basis_1)
    scaler.accumulate_batch_from_basis(intensity_2, *basis_2)
    snapshot = scaler.sufficient_statistics()
    expected_atb = torch.stack(
        [
            (basis_1[index] * intensity_1).sum()
            + (basis_2[index] * intensity_2).sum()
            for index in range(3)
        ]
    ).to(torch.float64)

    torch.testing.assert_close(snapshot.ATb.flatten(), expected_atb)
    assert snapshot.sum_i2 == pytest.approx(
        float(intensity_1.square().sum() + intensity_2.square().sum())
    )
    assert snapshot.n_pixels == 4

    leaked = snapshot.ATA
    leaked.zero_()
    assert torch.count_nonzero(scaler.sufficient_statistics().ATA) > 0


def test_varpro_scaler_accumulate_batch_sums_multimode_basis():
    generator = torch.Generator().manual_seed(73)
    psi_a = torch.complex(
        torch.rand(2, 1, 3, 4, 4, generator=generator),
        torch.rand(2, 1, 3, 4, 4, generator=generator),
    )
    psi_b = torch.complex(
        torch.rand(2, 1, 3, 4, 4, generator=generator),
        torch.rand(2, 1, 3, 4, 4, generator=generator),
    )
    intensity = torch.rand(2, 1, 4, 4, generator=generator)
    direct = reassembly.VarProScaler("cpu")
    aggregate = reassembly.VarProScaler("cpu")

    direct.accumulate_batch(intensity, psi_a, psi_b)
    aggregate.accumulate_batch_from_basis(
        intensity,
        psi_a.abs().square().sum(dim=2),
        psi_b.abs().square().sum(dim=2),
        (2 * torch.real(psi_a * torch.conj(psi_b))).sum(dim=2),
    )

    direct_stats = direct.sufficient_statistics()
    aggregate_stats = aggregate.sufficient_statistics()
    torch.testing.assert_close(direct_stats.ATA, aggregate_stats.ATA)
    torch.testing.assert_close(direct_stats.ATb, aggregate_stats.ATb)
    assert direct_stats.sum_i2 == pytest.approx(aggregate_stats.sum_i2)
    assert direct_stats.n_pixels == aggregate_stats.n_pixels


def test_varpro_scaler_reduces_products_in_float64():
    n_values = 10_000
    x1 = torch.linspace(1.0, 1.001, n_values, dtype=torch.float32).reshape(
        1, 1, 100, 100
    )
    x2 = 0.75 * x1
    x3 = -0.25 * x1
    intensity = torch.linspace(
        0.999, 1.0, n_values, dtype=torch.float32
    ).reshape_as(x1)
    scaler = reassembly.VarProScaler("cpu")

    scaler.accumulate_batch_from_basis(intensity, x1, x2, x3)
    stats = scaler.sufficient_statistics()
    bases64 = [value.to(torch.float64) for value in (x1, x2, x3)]
    intensity64 = intensity.to(torch.float64)
    expected_ata = torch.stack(
        [
            torch.stack([(left * right).sum() for right in bases64])
            for left in bases64
        ]
    )
    expected_atb = torch.stack(
        [(basis * intensity64).sum() for basis in bases64]
    )

    torch.testing.assert_close(stats.ATA, expected_ata, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        stats.ATb.flatten(), expected_atb, rtol=0.0, atol=0.0
    )
    assert stats.sum_i2 == float((intensity64 * intensity64).sum())


@pytest.mark.parametrize(
    ("legacy", "precision", "expected"),
    [
        (None, None, "32-true"),
        (False, None, "32-true"),
        (True, None, "16-mixed"),
        (None, "16-mixed", "16-mixed"),
        (None, "bf16-mixed", "bf16-mixed"),
        (False, "32-true", "32-true"),
        (True, "16-mixed", "16-mixed"),
    ],
)
def test_effective_inference_precision_resolution(legacy, precision, expected):
    assert reassembly.resolve_inference_precision(legacy, precision) == expected


@pytest.mark.parametrize(
    ("legacy", "precision"),
    [(True, "32-true"), (False, "16-mixed"), (True, "bf16-mixed")],
)
def test_effective_inference_precision_rejects_conflicts(legacy, precision):
    with pytest.raises(ValueError, match="Conflicting inference precision"):
        reassembly.resolve_inference_precision(legacy, precision)


def _probe(n, n_modes):
    axis = torch.arange(n, dtype=torch.float32) - n // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    modes = []
    for mode in range(n_modes):
        envelope = torch.exp(
            -((xx - mode).square() + (yy + mode).square()) / (2 * (n / 3) ** 2)
        )
        phase = torch.exp(1j * (0.1 + 0.05 * mode) * xx)
        modes.append((envelope * phase / math.sqrt(mode + 1)).to(torch.complex64))
    return torch.stack(modes).view(1, 1, n_modes, n, n)


def _texture(n):
    axis = torch.linspace(-1.0, 1.0, n)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.complex(0.8 + 0.1 * xx, 0.2 * torch.sin(2.0 * yy)).view(
        1, 1, n, n
    )


def _oracle(probe, texture, mask, s1, s2):
    exit_wave = probe * mask.view(1, 1, 1, *mask.shape) * (
        s1 * texture.real.unsqueeze(2) + 1j * s2 * texture.imag.unsqueeze(2)
    )
    wave = torch.fft.fftshift(
        torch.fft.fft2(exit_wave, norm="ortho"), dim=(-2, -1)
    )
    return wave.abs().square().sum(dim=2)


class _AutocastRecordingModel(torch.nn.Module):
    def __init__(self, texture):
        super().__init__()
        self.register_buffer("texture", texture)
        self.autocast_calls = []

    def forward_predict(self, intensity, positions, probe, input_scale):
        device_type = intensity.device.type
        self.autocast_calls.append(
            (
                torch.is_autocast_enabled(device_type),
                torch.get_autocast_dtype(device_type),
            )
        )
        return self.texture.expand(intensity.shape[0], -1, -1, -1)


def _count_batches(n_modes=1, *, s1=1.2, s2=0.7):
    n = 8
    texture = _texture(n)
    probe = _probe(n, n_modes)
    mask = torch.linspace(0.25, 1.0, n * n, dtype=torch.float32).reshape(n, n)
    measured = _oracle(probe, texture, mask, s1, s2)
    batches = []
    for x_coordinate in (10.0, 12.0):
        batch_data = {
            "measured_intensity": measured.clone(),
            "coords_relative": torch.zeros(1, 1, 1, 2),
            "coords_global": torch.tensor([[[[x_coordinate, 10.0]]]]),
            "rms_input_scale": torch.ones(1, 1, 1, 1),
            "probe_physical": probe.clone(),
        }
        batches.append((batch_data, 0.01 * probe, torch.ones(1, 1, 1, 1)))
    data_config = DataConfig(
        N=n,
        C=1,
        scale_contract_version="ci_intensity_v2",
        measurement_domain="count_intensity",
    )
    model_config = ModelConfig(
        physics_forward_mode="rectangular_scaled",
        probe_mask_tensor=mask,
    )
    return batches, texture, data_config, model_config, measured


@pytest.mark.parametrize("n_modes", [1, 2], ids=["single-mode", "multimode"])
def test_fitted_count_metrics_stream_two_batches_and_sum_modes(n_modes):
    s1, s2 = 1.2, 0.7
    batches, texture, data_config, model_config, measured = _count_batches(
        n_modes, s1=s1, s2=s2
    )
    model = _AutocastRecordingModel(texture)

    metrics = reassembly.evaluate_fitted_count_metrics(
        model,
        batches,
        data_config,
        model_config,
        s1=s1,
        s2=s2,
        device=torch.device("cpu"),
        scale_profile="ci_intensity_v2",
        precision="32-true",
    )
    expected_nll = (
        measured - measured * torch.log(torch.clamp(measured, min=1e-8))
    ).mean()

    assert metrics.relative_l2_intensity_error == pytest.approx(0.0, abs=1e-6)
    assert metrics.mean_raw_poisson_nll == pytest.approx(float(expected_nll), rel=1e-6)
    assert metrics.n_samples == 2
    assert metrics.n_pixels == 2 * measured.numel()
    assert metrics.effective_mask_digest == array_digest(
        model_config.probe_mask_tensor
    )
    assert dataclasses.asdict(metrics)["n_samples"] == 2


def test_fitted_count_metrics_preserves_loader_sample_order_and_multiplicity():
    batches, texture, data_config, model_config, _measured = _count_batches()
    batches[0][0]["nn_indices"] = torch.tensor([[1]])
    batches[1][0]["nn_indices"] = torch.tensor([[1]])

    result = reassembly.evaluate_fitted_count_metrics(
        _AutocastRecordingModel(texture),
        batches,
        data_config,
        model_config,
        s1=1.2,
        s2=0.7,
        device=torch.device("cpu"),
        scale_profile="ci_intensity_v2",
        precision="32-true",
    )

    assert result.sample_ids == (1, 1)
    assert result.sample_identity_digest == array_digest(torch.tensor([1, 1]))


def test_fitted_count_metrics_applies_reconstruction_channel_swap():
    s1, s2 = 1.2, 0.7
    batches, texture, data_config, model_config, _measured = _count_batches(
        n_modes=2, s1=s1, s2=s2
    )
    swapped_texture = torch.complex(texture.imag, texture.real)
    model = _AutocastRecordingModel(swapped_texture)

    metrics = reassembly.evaluate_fitted_count_metrics(
        model,
        batches,
        data_config,
        model_config,
        s1=s1,
        s2=s2,
        device=torch.device("cpu"),
        scale_profile="ci_intensity_v2",
        precision="32-true",
        channels_swapped=True,
    )

    assert metrics.relative_l2_intensity_error == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("16-mixed", torch.bfloat16), ("bf16-mixed", torch.bfloat16)],
)
def test_fitted_count_metrics_enters_selected_autocast(precision, expected_dtype):
    batches, texture, data_config, model_config, _measured = _count_batches()
    model = _AutocastRecordingModel(texture)

    reassembly.evaluate_fitted_count_metrics(
        model,
        batches[:1],
        data_config,
        model_config,
        s1=1.2,
        s2=0.7,
        device=torch.device("cpu"),
        scale_profile="ci_intensity_v2",
        precision=precision,
    )

    assert model.autocast_calls == [(True, expected_dtype)]


def test_fitted_count_metrics_marks_legacy_profile_not_applicable():
    result = reassembly.evaluate_fitted_count_metrics(
        object(),
        [],
        DataConfig(N=8, C=1),
        ModelConfig(),
        s1=1.0,
        s2=1.0,
        device=torch.device("cpu"),
        scale_profile="legacy_v1",
    )

    assert isinstance(result, NotApplicable)
    assert result.to_jsonable() == {
        "status": "not_applicable",
        "reason": "legacy_normalized_amplitude",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reason = "changed"
    with pytest.raises(TypeError):
        NotApplicable(reason=[])


def test_weighted_accumulator_counts_accepted_and_dropped_patches():
    accumulator = reassembly.VectorizedWeightedAccumulator((10, 10), torch.device("cpu"))
    canvas = torch.zeros(10, 10, dtype=torch.complex64)
    weights = torch.zeros(10, 10)
    patches = torch.ones(2, 2, 2, dtype=torch.complex64)
    positions = torch.tensor([[5.0, 5.0], [100.0, 100.0]])

    with pytest.warns(UserWarning, match="dropping 1/2"):
        accumulator.accumulate_batch(
            canvas,
            weights,
            patches,
            positions,
            torch.ones(2, 2),
            patch_size=2,
        )

    assert accumulator.accepted_patches == 1
    assert accumulator.total_patches == 2
    assert accumulator.patches_accepted == 1
    assert accumulator.patches_total == 2


class _TinyCIDataset:
    def __init__(self, batches, texture):
        measured = torch.cat([batch[0]["measured_intensity"] for batch in batches])
        probe = torch.cat([batch[0]["probe_physical"] for batch in batches])
        coords = torch.cat([batch[0]["coords_global"] for batch in batches])
        n_samples = measured.shape[0]
        self.mmap_ptycho = TensorDict(
            {
                "images": measured,
                "measured_intensity": measured,
                "coords_relative": torch.zeros(n_samples, 1, 1, 2),
                "coords_global": coords,
                "experiment_id": torch.zeros(n_samples, dtype=torch.long),
                "rms_input_scale": torch.ones(n_samples, 1, 1, 1),
                "mean_measured_intensity": measured.mean().expand(
                    n_samples, 1, 1, 1
                ),
                "probe_physical": probe,
                "rms_scaling_constant": torch.ones(n_samples, 1, 1, 1),
                "physics_scaling_constant": torch.ones(n_samples, 1, 1, 1),
                "nn_indices": torch.arange(n_samples, dtype=torch.long)[:, None],
            },
            batch_size=[n_samples],
        )
        self.n_files = 1
        self.data_dict = {}
        self.valid_indices_per_file = [np.arange(n_samples, dtype=np.int64)]
        self.source_indices_per_file = [np.arange(n_samples, dtype=np.int64)]
        self._tuple_probe = 0.01 * probe[0]
        self.texture = texture

    def __len__(self):
        return self.mmap_ptycho.batch_size[0]

    def __getitem__(self, index):
        batch = self.mmap_ptycho[index]
        batch_size = batch.batch_size[0]
        tuple_probe = self._tuple_probe.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )
        return batch, tuple_probe, torch.ones(batch_size, 1, 1, 1)


def _run_tiny_reconstruction(
    *,
    return_diagnostics=False,
    structured_diagnostics=False,
    legacy=False,
    precision=None,
    swap_detection="None",
    return_model=False,
    swapped_model=False,
    oracle_s1=1.2,
    oracle_s2=0.7,
    compute_count_metrics=True,
    model_texture_override=None,
):
    batches, texture, data_config, model_config, _measured = _count_batches(
        s1=oracle_s1,
        s2=oracle_s2,
    )
    if legacy:
        data_config = DataConfig(
            N=8,
            C=1,
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        )
        model_config = ModelConfig(physics_forward_mode="amplitude")
    dataset = _TinyCIDataset(batches, texture)
    model_texture = texture if model_texture_override is None else model_texture_override
    if swapped_model:
        model_texture = torch.complex(model_texture.imag, model_texture.real)
    model = _AutocastRecordingModel(model_texture)
    training_config = dataclasses.replace(
        TrainingConfig(), device="cpu", num_workers=0
    )
    inference_config = InferenceConfig(
        middle_trim=4,
        batch_size=1,
        patch_weighting="uniform",
        varpro_scaling=False,
    )
    result = reassembly.reconstruct_image_barycentric(
        model,
        dataset,
        training_config,
        data_config,
        model_config,
        inference_config,
        gpu_ids=None,
        use_mixed_precision=None,
        verbose=False,
        swap_detection=swap_detection,
        return_diagnostics=return_diagnostics,
        structured_diagnostics=structured_diagnostics,
        precision=precision,
        compute_count_metrics=compute_count_metrics,
    )
    if return_model:
        return result, model
    return result


def test_reconstruct_preserves_old_tuple_returns_when_structured_is_false():
    default_result = _run_tiny_reconstruction()
    legacy_diagnostics_result = _run_tiny_reconstruction(return_diagnostics=True)

    assert len(default_result) == 3
    assert len(default_result[2]) == 3
    assert len(legacy_diagnostics_result) == 4
    assert len(legacy_diagnostics_result[2]) == 7
    assert torch.is_tensor(legacy_diagnostics_result[2][2])
    assert torch.is_tensor(legacy_diagnostics_result[2][3])


def test_structured_return_contains_dataset_aggregate_not_final_batch_basis():
    canvas, subset, diagnostics, prescale = _run_tiny_reconstruction(
        structured_diagnostics=True
    )

    assert isinstance(diagnostics, ReassemblyDiagnostics)
    assert diagnostics.sufficient_statistics.n_pixels == 2 * 8 * 8
    assert diagnostics.accepted_patches == 2
    assert diagnostics.total_patches == 2
    assert diagnostics.used_scan_ids == (0, 1)
    assert diagnostics.expected_scan_ids == (0, 1)
    assert diagnostics.decoder_real_saturation_fraction == pytest.approx(0.0)
    assert diagnostics.decoder_imag_saturation_fraction == pytest.approx(0.0)
    assert diagnostics.count_metrics.n_samples == 2
    assert diagnostics.mask_digest == diagnostics.count_metrics.effective_mask_digest
    assert not hasattr(diagnostics, "Psi_a")
    assert not hasattr(diagnostics, "Psi_b")
    assert "Psi_a" not in diagnostics.to_jsonable()
    assert canvas.shape == prescale.shape
    assert subset.n_files == 1


def test_decoder_saturation_uses_cropped_active_support():
    texture = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
    texture[:, :, 2:6, 2:6] = 1.2 + 1.2j

    _canvas, _subset, diagnostics, _prescale = _run_tiny_reconstruction(
        structured_diagnostics=True,
        compute_count_metrics=False,
        model_texture_override=texture,
    )

    assert diagnostics.decoder_real_lower_saturation_fraction == pytest.approx(0.0)
    assert diagnostics.decoder_real_upper_saturation_fraction == pytest.approx(1.0)
    assert diagnostics.decoder_imag_lower_saturation_fraction == pytest.approx(0.0)
    assert diagnostics.decoder_imag_upper_saturation_fraction == pytest.approx(1.0)
    assert diagnostics.decoder_real_saturation_fraction == pytest.approx(1.0)
    assert diagnostics.decoder_imag_saturation_fraction == pytest.approx(1.0)


def test_structured_reconstruction_swapped_oracle_has_zero_count_error():
    _canvas, _subset, diagnostics, _prescale = _run_tiny_reconstruction(
        structured_diagnostics=True,
        swapped_model=True,
        oracle_s1=1.0,
        oracle_s2=1.0,
        swap_detection="mean",
    )
    assert diagnostics.count_metrics.relative_l2_intensity_error == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_structured_reconstruction_uses_shared_ci_preparation_for_both_passes(
    monkeypatch,
):
    calls = []
    original = reassembly._prepare_ci_varpro_batch

    def record_call(*args, **kwargs):
        calls.append(kwargs["channels_swapped"])
        return original(*args, **kwargs)

    monkeypatch.setattr(reassembly, "_prepare_ci_varpro_batch", record_call)

    _run_tiny_reconstruction(structured_diagnostics=True)

    assert calls == [False, False, False, False]


def test_structured_ci_reconstruction_can_defer_count_metrics_to_runtime(
    monkeypatch,
):
    calls = []
    original = reassembly.evaluate_fitted_count_metrics

    def record_call(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(reassembly, "evaluate_fitted_count_metrics", record_call)

    _canvas, _subset, diagnostics, _prescale = _run_tiny_reconstruction(
        structured_diagnostics=True,
        compute_count_metrics=False,
    )

    assert calls == []
    assert diagnostics.count_metrics.to_jsonable() == {
        "status": "not_evaluated",
        "reason": "deferred_to_canonical_runtime",
    }


def test_legacy_structured_return_uses_typed_not_applicable_records():
    canvas, subset, diagnostics, prescale = _run_tiny_reconstruction(
        structured_diagnostics=True,
        legacy=True,
    )

    assert isinstance(diagnostics, ReassemblyDiagnostics)
    assert isinstance(diagnostics.sufficient_statistics, NotApplicable)
    assert isinstance(diagnostics.condition, NotApplicable)
    assert isinstance(diagnostics.unit_objective, NotApplicable)
    assert isinstance(diagnostics.fitted_objective, NotApplicable)
    assert isinstance(diagnostics.count_metrics, NotApplicable)
    expected = {
        "status": "not_applicable",
        "reason": "legacy_normalized_amplitude",
    }
    payload = diagnostics.to_jsonable()
    assert payload["sufficient_statistics"] == expected
    assert payload["count_metrics"] == expected
    assert canvas.shape == prescale.shape
    assert subset.n_files == 1


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [("16-mixed", torch.bfloat16), ("bf16-mixed", torch.bfloat16)],
)
def test_reconstruction_applies_autocast_to_every_model_forward(
    precision, expected_dtype
):
    (result, model) = _run_tiny_reconstruction(
        structured_diagnostics=True,
        precision=precision,
        swap_detection="probe",
        return_model=True,
    )

    assert len(model.autocast_calls) == 5
    assert model.autocast_calls == [(True, expected_dtype)] * 5
    assert result[2].effective_precision == (
        "bf16-mixed" if precision == "16-mixed" else precision
    )


def test_reconstruction_precision_agrees_with_effective_runtime_json(tmp_path):
    execution_config = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
    )
    trainer = SimpleNamespace(
        precision_plugin=SimpleNamespace(precision="bf16-mixed"),
        strategy=SimpleNamespace(
            root_device=torch.device("cpu"),
            parallel_devices=[torch.device("cpu")],
            process_group_backend=None,
            launcher=None,
        ),
        accelerator=SimpleNamespace(),
        num_devices=1,
        device_ids=[0],
        callbacks=[],
        loggers=[],
    )
    runtime = train_module._build_effective_runtime(
        11,
        {"precision": "bf16-mixed", "callbacks": [], "logger": []},
        execution_config,
        dataloader_settings={
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        },
        trainer=trainer,
    )
    runtime_path = tmp_path / "effective_runtime.json"
    with runtime_path.open("w") as stream:
        json.dump(runtime, stream, indent=2, sort_keys=True)
    serialized_runtime = json.loads(runtime_path.read_text())

    (_result, model) = _run_tiny_reconstruction(
        structured_diagnostics=True,
        precision=serialized_runtime["precision"],
        swap_detection="probe",
        return_model=True,
    )

    assert serialized_runtime == runtime
    assert serialized_runtime["precision"] == runtime["effective"]["precision"]["value"]
    assert model.autocast_calls == [(True, torch.bfloat16)] * 5
