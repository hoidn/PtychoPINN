import numpy as np
import pytest
import torch

from ptycho_torch.scaling_contract import (
    derive_ci_experiment_statistics,
    normalize_ci_poisson_per_sample,
    rescale_amplitude_to_nphotons,
)


def test_ci_statistics_match_closed_form_for_multichannel_experiment():
    measured = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 2.0]], [[1.0, 2.0]]],
            [[[2.0, 0.0]], [[0.0, 1.0]], [[2.0, 1.0]]],
        ],
        dtype=torch.float64,
    )

    statistics = derive_ci_experiment_statistics(measured, N=4)

    assert statistics.rms_input_scale.shape == torch.Size([])
    assert statistics.mean_measured_intensity.shape == torch.Size([])
    assert statistics.rms_input_scale.item() == pytest.approx(1.0954451150103321)
    assert statistics.mean_measured_intensity.item() == pytest.approx(1.0)


def test_ci_statistics_preserve_dtype_device_and_gradients():
    measured = torch.linspace(
        0.25,
        3.0,
        steps=2 * 3 * 2 * 5,
        dtype=torch.float64,
    ).reshape(2, 3, 2, 5).requires_grad_()

    statistics = derive_ci_experiment_statistics(measured, torch.tensor(8.0))

    assert statistics.rms_input_scale.dtype == measured.dtype
    assert statistics.mean_measured_intensity.dtype == measured.dtype
    assert statistics.rms_input_scale.device == measured.device
    assert statistics.mean_measured_intensity.device == measured.device
    assert statistics.rms_input_scale.requires_grad
    assert statistics.mean_measured_intensity.requires_grad

    (statistics.rms_input_scale + statistics.mean_measured_intensity).backward()
    assert measured.grad is not None
    assert torch.isfinite(measured.grad).all()


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 5, 1)])
def test_ci_statistics_reject_invalid_intensity_shape(shape):
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        derive_ci_experiment_statistics(torch.ones(shape), 8)


@pytest.mark.parametrize(
    "measured",
    [
        torch.ones((2, 3, 4, 5), dtype=torch.int64),
        torch.ones((2, 3, 4, 5), dtype=torch.complex64),
    ],
)
def test_ci_statistics_reject_non_floating_or_complex_intensity(measured):
    with pytest.raises(TypeError, match="real floating"):
        derive_ci_experiment_statistics(measured, 8)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.25])
def test_ci_statistics_reject_nonfinite_and_negative_counts(bad_value):
    measured = torch.ones((2, 3, 2, 4))
    measured[1, 2, 1, 3] = bad_value

    with pytest.raises(ValueError, match="finite|nonnegative"):
        derive_ci_experiment_statistics(measured, 8)


def test_ci_statistics_reject_all_zero_counts():
    with pytest.raises(ValueError, match="positive|degenerate"):
        derive_ci_experiment_statistics(torch.zeros((3, 2, 2, 4)), 8)


@pytest.mark.parametrize(
    "bad_n",
    [0, -2, float("nan"), float("inf"), torch.tensor([8.0, 9.0])],
)
def test_ci_statistics_reject_invalid_n(bad_n):
    with pytest.raises((TypeError, ValueError), match="N"):
        derive_ci_experiment_statistics(torch.ones((3, 2, 2, 4)), bad_n)


def test_ci_statistics_reject_float32_target_energy_overflow():
    measured = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    huge_finite_n = torch.tensor(1.0e30, dtype=torch.float32)

    assert torch.isfinite(huge_finite_n)
    with pytest.raises(ValueError, match="target energy.*finite"):
        derive_ci_experiment_statistics(measured, huge_finite_n)


def test_ci_statistics_reject_nonfinite_final_rms_scale():
    measured = torch.full((1, 1, 1, 1), 1.0e-20, dtype=torch.float32)

    with pytest.raises(ValueError, match="rms_input_scale.*finite"):
        derive_ci_experiment_statistics(measured, N=2)


@pytest.mark.parametrize("multimode", [False, True])
def test_rescale_amplitude_to_nphotons_matches_complete_acquisition_formula(
    multimode,
):
    amplitude = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[1.5, 2.5], [3.5, 4.5]],
        ],
        dtype=np.float32,
    )
    single_probe = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0 + 0.5j, 0.25 - 4.0j]],
        dtype=np.complex64,
    )
    probe = (
        np.stack((single_probe, single_probe * np.complex64(0.25 - 0.5j)))
        if multimode
        else single_probe
    )
    probe_simulated = probe * np.complex64(-0.75 + 0.25j)
    nphotons = 1.0e9

    intensity, scaled_probe, scaled_simulated = rescale_amplitude_to_nphotons(
        amplitude,
        probe,
        nphotons,
        probe_simulated,
    )

    expected_scale = np.sqrt(
        nphotons
        / np.mean(
            np.sum(
                np.square(amplitude, dtype=np.float64),
                axis=(-2, -1),
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(intensity, (expected_scale * amplitude) ** 2)
    np.testing.assert_array_equal(scaled_probe, expected_scale * probe)
    np.testing.assert_array_equal(
        scaled_simulated,
        expected_scale * probe_simulated,
    )
    np.testing.assert_allclose(
        np.mean(np.sum(intensity, axis=(-2, -1), dtype=np.float64)),
        nphotons,
        rtol=2e-7,
    )
    assert not np.isclose(np.sum(np.abs(scaled_probe) ** 2), nphotons)
    assert intensity.dtype == amplitude.dtype
    assert scaled_probe.dtype == probe.dtype
    assert scaled_simulated.dtype == probe_simulated.dtype


def test_rescale_amplitude_to_nphotons_returns_none_for_omitted_simulated_probe():
    result = rescale_amplitude_to_nphotons(
        np.ones((2, 2, 2), dtype=np.float64),
        np.ones((2, 2), dtype=np.complex128),
        8.0,
    )

    assert len(result) == 3
    assert result[2] is None


@pytest.mark.parametrize(
    ("amplitude", "message"),
    [
        (np.array([[[1.0, -1.0]]]), "nonnegative"),
        (np.array([[[1.0, np.nan]]]), "finite"),
        (np.zeros((2, 2, 2)), "nonzero"),
    ],
)
def test_rescale_amplitude_to_nphotons_rejects_invalid_amplitude(
    amplitude,
    message,
):
    with pytest.raises(ValueError, match=f"amplitude.*{message}"):
        rescale_amplitude_to_nphotons(
            amplitude,
            np.ones((2, 2), dtype=np.complex64),
            10.0,
        )


@pytest.mark.parametrize(
    ("probe", "message"),
    [
        (np.array([[1.0 + 0.0j, np.nan + 1.0j]]), "finite"),
        (np.zeros((2, 2), dtype=np.complex64), "nonzero"),
    ],
)
def test_rescale_amplitude_to_nphotons_rejects_invalid_probe(probe, message):
    with pytest.raises(ValueError, match=f"probe.*{message}"):
        rescale_amplitude_to_nphotons(np.ones((2, 2, 2)), probe, 10.0)


@pytest.mark.parametrize("nphotons", [0.0, -1.0, np.nan, np.inf])
def test_rescale_amplitude_to_nphotons_rejects_invalid_nphotons(nphotons):
    with pytest.raises(ValueError, match="nphotons.*positive.*finite"):
        rescale_amplitude_to_nphotons(
            np.ones((2, 2, 2)),
            np.ones((2, 2), dtype=np.complex64),
            nphotons,
        )


def test_rescale_amplitude_to_nphotons_rejects_nonfinite_derived_scale():
    amplitude = np.full((1, 1, 1), np.nextafter(0.0, 1.0), dtype=np.float64)

    with pytest.raises(ValueError, match="derived scale.*finite"):
        rescale_amplitude_to_nphotons(
            amplitude,
            np.ones((1, 1), dtype=np.complex128),
            1.0,
        )


@pytest.mark.parametrize("overflow_target", ["intensity", "probe", "simulated"])
def test_rescale_amplitude_to_nphotons_rejects_numeric_overflow(overflow_target):
    amplitude = np.ones((1, 1, 1), dtype=np.float32)
    probe = np.ones((1, 1), dtype=np.complex64)
    probe_simulated = np.ones((1, 1), dtype=np.complex64)
    nphotons = 1.0
    if overflow_target == "intensity":
        amplitude = np.array([[[10.0]], [[0.0]]], dtype=np.float64)
        nphotons = 1.0e308
    elif overflow_target == "probe":
        probe[0, 0] = np.finfo(np.float32).max + 0j
        nphotons = 4.0
    else:
        probe_simulated[0, 0] = np.finfo(np.float32).max + 0j
        nphotons = 4.0

    with pytest.raises(ValueError, match="numeric overflow"):
        rescale_amplitude_to_nphotons(
            amplitude,
            probe,
            nphotons,
            probe_simulated,
        )


def test_rescale_amplitude_to_nphotons_rejects_numeric_underflow():
    amplitude = np.ones((1, 1, 1), dtype=np.float32)
    probe = np.ones((1, 1), dtype=np.complex64)

    with pytest.raises(ValueError, match="numeric underflow"):
        rescale_amplitude_to_nphotons(
            amplitude,
            probe,
            1.0e-300,
            probe.copy(),
        )


def test_poisson_normalizer_accepts_scalar_and_collated_experiment_means():
    raw_nll = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)

    scalar_result = normalize_ci_poisson_per_sample(raw_nll, raw_nll.new_tensor(5.0))
    collated_result = normalize_ci_poisson_per_sample(
        raw_nll,
        raw_nll.new_tensor([5.0, 10.0, 15.0]).reshape(3, 1, 1, 1),
    )

    torch.testing.assert_close(scalar_result, raw_nll.new_tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(collated_result, raw_nll.new_tensor([2.0, 2.0, 2.0]))
    assert scalar_result.shape == raw_nll.shape
    assert collated_result.shape == raw_nll.shape


def test_poisson_normalizer_detaches_physical_mean_only():
    raw_nll = torch.tensor([4.0, 8.0], requires_grad=True)
    physical_mean = torch.tensor(2.0, requires_grad=True)

    normalize_ci_poisson_per_sample(raw_nll, physical_mean).sum().backward()

    torch.testing.assert_close(raw_nll.grad, torch.tensor([0.5, 0.5]))
    assert physical_mean.grad is None


class _NoScalarExtractionTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)

    def __bool__(self):
        raise AssertionError("tensor truth-value extraction is forbidden")

    def item(self, *args, **kwargs):
        raise AssertionError("tensor item extraction is forbidden")


def test_poisson_normalizer_has_no_data_dependent_scalar_extraction():
    raw_nll = _NoScalarExtractionTensor(torch.tensor([4.0, 8.0]))
    physical_mean = _NoScalarExtractionTensor(torch.tensor(2.0))

    result = normalize_ci_poisson_per_sample(raw_nll, physical_mean)

    torch.testing.assert_close(
        result.as_subclass(torch.Tensor),
        torch.tensor([2.0, 4.0]),
    )


def test_poisson_normalizer_requires_matching_dtypes():
    raw_nll = torch.ones(2, dtype=torch.float32)
    physical_mean = torch.tensor(2.0, dtype=torch.float64)

    with pytest.raises(ValueError, match="dtype"):
        normalize_ci_poisson_per_sample(raw_nll, physical_mean)


def test_poisson_normalizer_propagates_nonfinite_raw_loss_without_validation():
    raw_result = normalize_ci_poisson_per_sample(
        torch.tensor([float("nan"), float("inf")]),
        torch.tensor(2.0),
    )

    assert torch.isnan(raw_result[0])
    assert torch.isinf(raw_result[1])


@pytest.mark.parametrize(
    ("bad_denominator", "message"),
    [
        (0.0, "positive"),
        (-1.0, "positive"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-float("inf"), "finite"),
    ],
)
def test_poisson_normalizer_rejects_invalid_denominator_asynchronously(
    bad_denominator,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        normalize_ci_poisson_per_sample(
            torch.tensor([1.0, 2.0]),
            torch.tensor(bad_denominator),
        )


@pytest.mark.parametrize(
    "raw_nll, physical_mean",
    [
        (torch.tensor(1.0), torch.tensor(2.0)),
        (torch.ones(3), torch.ones(2)),
        (torch.ones(3), torch.ones(3, 2)),
        (torch.ones(3, dtype=torch.int64), torch.tensor(2.0)),
        (torch.ones(3), torch.tensor(2, dtype=torch.int64)),
    ],
)
def test_poisson_normalizer_rejects_invalid_static_shapes_and_types(
    raw_nll,
    physical_mean,
):
    with pytest.raises((TypeError, ValueError), match="raw_nll|mean_measured_intensity"):
        normalize_ci_poisson_per_sample(raw_nll, physical_mean)


def _poisson_gradient_at_scale(scale, *, normalized):
    prediction_amplitude = torch.tensor(
        [
            [[[0.8, 1.2], [1.5, 0.7]]],
            [[[1.1, 0.9], [0.6, 1.4]]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    measured_amplitude = torch.tensor(
        [
            [[[1.0, 0.7], [1.3, 0.9]]],
            [[[0.8, 1.2], [1.1, 0.6]]],
        ],
        dtype=torch.float64,
    )
    predicted_intensity = (scale * prediction_amplitude).square()
    measured_intensity = (scale * measured_amplitude).square()
    raw_nll = (
        predicted_intensity - measured_intensity * predicted_intensity.log()
    ).sum(dim=(1, 2, 3))
    if normalized:
        per_sample_loss = normalize_ci_poisson_per_sample(
            raw_nll,
            measured_intensity.mean(),
        )
    else:
        per_sample_loss = raw_nll
    per_sample_loss.mean().backward()
    return prediction_amplitude.grad


def test_physical_mean_normalization_makes_poisson_gradients_scale_invariant():
    scale = 329.0

    normalized_at_one = _poisson_gradient_at_scale(1.0, normalized=True)
    normalized_at_scale = _poisson_gradient_at_scale(scale, normalized=True)
    torch.testing.assert_close(
        normalized_at_scale,
        normalized_at_one,
        rtol=1e-5,
        atol=1e-8,
    )

    raw_at_one = _poisson_gradient_at_scale(1.0, normalized=False)
    raw_at_scale = _poisson_gradient_at_scale(scale, normalized=False)
    raw_gradient_ratio = raw_at_scale.norm() / raw_at_one.norm()
    assert raw_gradient_ratio.item() == pytest.approx(scale**2, rel=1e-5)
