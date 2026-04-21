import torch


def test_fourier_rmse_bands_isolates_low_frequency_constant_error():
    from scripts.studies.pdebench_image128.metrics import fourier_rmse_bands

    prediction = torch.ones(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)

    payload = fourier_rmse_bands(prediction, target)

    assert payload["band_definition"] == "fftshifted_radial_frequency_thirds"
    assert payload["fRMSE_low"] > 0.0
    assert payload["fRMSE_mid"] == 0.0
    assert payload["fRMSE_high"] == 0.0
    assert payload["per_channel"]["fRMSE_high"] == [0.0]


def test_dynamic_state_metric_payload_reports_cns_fourier_rmse_bands_after_denormalization():
    from scripts.studies.pdebench_image128.metrics import dynamic_state_metric_payload

    state_stats = {"mean": [10.0], "std": [2.0]}
    predictions_normalized = [torch.ones(1, 1, 8, 8)]
    targets_normalized = [torch.zeros(1, 1, 8, 8)]

    payload = dynamic_state_metric_payload(
        predictions_normalized,
        targets_normalized,
        normalized=True,
        state_stats=state_stats,
    )

    assert payload["metric_units"] == "denormalized_state_units"
    assert payload["fourier_metric_units"] == "denormalized_state_units_fft_ortho"
    assert payload["fRMSE_low"] > 0.0
    assert payload["fRMSE_mid"] == 0.0
    assert payload["fRMSE_high"] == 0.0
