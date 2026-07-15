"""Deterministic end-to-end regression for the grid-lines CI probe gauge.

This test crosses the boundaries that allowed ``probeGuess`` (the probe supplied
to the simulator) to be confused with the normalized illumination that actually
generated the diffraction data.  It intentionally exercises the real workflow
writer, NPZ loader, CI probe selector, CI normalization adapter, and rectangular
count-intensity forward in one CPU test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptycho import params
from ptycho.config.config import ModelConfig as TFModelConfig
from ptycho.config.config import TrainingConfig as TFTrainingConfig
from ptycho.config.config import update_legacy_dict
from ptycho.workflows.grid_lines_workflow import (
    GridLinesConfig,
    configure_legacy_params,
    save_split_npz,
    simulate_grid_data,
)
from ptycho_torch.config_params import ModelConfig
from ptycho_torch.model import RectangularScaledDiffraction
from ptycho_torch.workflows.components import NormalizedAmplitudeCIDictAdapter
from scripts.studies import grid_lines_torch_runner as runner


pytestmark = [
    pytest.mark.integration,
    pytest.mark.torch,
    pytest.mark.deterministic,
]


@pytest.fixture
def _params_cfg_guard():
    original = params.cfg.copy()
    yield
    params.cfg.clear()
    params.cfg.update(original)


def _complex_object(N: int) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        indexing="ij",
    )
    amplitude = 0.45 + 0.35 * np.exp(-2.0 * (xx**2 + yy**2))
    phase = 0.35 * xx - 0.2 * yy + 0.15 * xx * yy
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _raw_probe(N: int) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        np.linspace(-1.0, 1.0, N, dtype=np.float32),
        indexing="ij",
    )
    # The positive pedestal keeps every element nonzero, making the scalar
    # simulation-probe/guess ratio directly observable.
    amplitude = 0.08 + 0.4 * np.exp(-2.5 * (xx**2 + yy**2))
    phase = 0.25 * xx + 0.1 * yy**2
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def _forward_amplitude(object_field: np.ndarray, probe: np.ndarray) -> np.ndarray:
    exit_wave = object_field * probe
    detector_field = np.fft.fftshift(
        np.fft.fft2(exit_wave, norm="ortho"),
        axes=(-2, -1),
    )
    return np.abs(detector_field).astype(np.float32)[None, ..., None]


def _fake_dataset(
    object_field: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[object, ...]:
    offsets = np.zeros((1, 1, 2, 1), dtype=np.float32)
    coords = np.zeros_like(offsets)
    container = SimpleNamespace(
        coords_nominal=coords,
        coords_true=coords,
        global_offsets=offsets,
        YY_full=object_field,
    )
    dataset = SimpleNamespace(train_data=container, test_data=container)
    label_amp = np.abs(object_field).astype(np.float32)[None, ..., None]
    label_phase = np.angle(object_field).astype(np.float32)[None, ..., None]
    return (
        amplitude,
        label_amp,
        label_phase,
        amplitude.copy(),
        label_amp.copy(),
        label_phase.copy(),
        object_field,
        dataset,
        object_field,
        np.array([1.0], dtype=np.float32),
    )


def _adapt_and_predict(
    *,
    amplitude: np.ndarray,
    probe: np.ndarray,
    object_field: np.ndarray,
    count_amplitude_scale: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    N = object_field.shape[0]
    container: dict[str, object] = {
        "observed_images": np.array(amplitude, copy=True),
        "probe": np.array(probe, copy=True),
    }
    NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=count_amplitude_scale,
        N=N,
        probe_scale=4.0,
        probe_mask=False,
    ).adapt(container)

    probe_training = torch.as_tensor(container["probe_training"])
    probe_batch = probe_training.unsqueeze(0).unsqueeze(0)
    probe_normalization = torch.as_tensor(container["probe_normalization"])
    output_scale = probe_normalization.reciprocal().reshape(1, 1, 1, 1)
    object_batch = torch.from_numpy(object_field).reshape(1, 1, N, N)
    measured = torch.as_tensor(container["measured_intensity"]).permute(0, 3, 1, 2)

    forward = RectangularScaledDiffraction(
        ModelConfig(num_datasets=1, rect_s1s2_trainable=False)
    )
    prediction = forward(
        object_batch,
        measured,
        probe_batch,
        output_scale,
        torch.zeros(1, dtype=torch.long),
    )
    return prediction, container


def test_grid_lines_ci_roundtrip_uses_realized_probe_and_preserves_count_rate(
    tmp_path,
    monkeypatch,
    _params_cfg_guard,
) -> None:
    N = 16
    count_amplitude_scale = 7.0
    raw_probe = _raw_probe(N)
    object_field = _complex_object(N)
    realized_during_simulation: dict[str, np.ndarray] = {}

    update_legacy_dict(
        params.cfg,
        TFTrainingConfig(model=TFModelConfig(N=N, gridsize=1, object_big=False)),
    )
    cfg = GridLinesConfig(
        N=N,
        gridsize=1,
        output_dir=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        size=N,
        nimgs_train=1,
        nimgs_test=1,
        nphotons=1e9,
        probe_source="custom",
        probe_smoothing_sigma=0.0,
        probe_scale_mode="pad_preserve",
        set_phi=True,
        seed=3,
    )

    # ``simulate_grid_data`` first calls the real probe setup.  The fake scan
    # producer then reads that realized probe and generates an exact diffraction
    # amplitude from it, avoiding random sampling or a training dependency.
    def fake_generate_data():
        realized = np.asarray(params.get("probe"))
        if realized.ndim == 3 and realized.shape[-1] == 1:
            realized = realized[..., 0]
        realized_during_simulation["probe"] = realized.copy()
        return _fake_dataset(
            object_field,
            _forward_amplitude(object_field, realized),
        )

    monkeypatch.setattr("ptycho.data_preprocessing.generate_data", fake_generate_data)
    params.set("intensity_scale", 1.0)
    simulation = simulate_grid_data(cfg, raw_probe)
    realized = realized_during_simulation["probe"]
    for split in ("train", "test"):
        assert "probe_simulated" in simulation[split], (
            "simulate_grid_data must bind the realized illumination to each split"
        )
        np.testing.assert_array_equal(
            simulation[split]["probe_simulated"],
            realized,
        )

    config = configure_legacy_params(cfg, raw_probe)
    for split in ("train", "test"):
        simulation[split]["probeGuess"] = raw_probe

    # Prove persistence comes from the split-bound value, not mutable legacy
    # global state consulted later by the writer.
    ambient_sentinel = np.full_like(realized, 17.0 - 9.0j)
    params.set("probe", ambient_sentinel)

    train_path = save_split_npz(cfg, "train", simulation["train"], config)
    test_path = save_split_npz(cfg, "test", simulation["test"], config)
    train, _ = runner.load_cached_dataset_with_metadata(train_path)
    test, _ = runner.load_cached_dataset_with_metadata(test_path)

    assert "probe_simulated" in train, (
        "the NPZ must retain the realized illumination, not only probeGuess"
    )
    assert "probe_simulated" in test
    assert train["probe_simulated"].shape == (N, N)
    assert train["probe_simulated"].dtype == np.complex64
    np.testing.assert_array_equal(train["probe_simulated"], realized)
    assert not np.array_equal(train["probe_simulated"], ambient_sentinel)

    select_ci_probe = getattr(runner, "_select_ci_probe", None)
    assert select_ci_probe is not None, (
        "the CI runner needs one explicit probe-selection boundary"
    )
    selected_train, selected_test, provenance = select_ci_probe(
        train,
        test,
        ci_dict_active=True,
    )
    assert provenance == "simulated"
    np.testing.assert_array_equal(selected_train, train["probe_simulated"])
    np.testing.assert_array_equal(selected_test, test["probe_simulated"])

    ratio = selected_train / train["probeGuess"]
    k = float(np.median(ratio.real))
    np.testing.assert_allclose(ratio.imag, 0.0, atol=2e-6)
    np.testing.assert_allclose(ratio.real, k, rtol=2e-6, atol=2e-6)
    assert not np.isclose(k, 1.0)

    prediction, selected_container = _adapt_and_predict(
        amplitude=train["diffraction"],
        probe=selected_train,
        object_field=object_field,
        count_amplitude_scale=count_amplitude_scale,
    )
    measured = torch.as_tensor(selected_container["measured_intensity"]).permute(
        0, 3, 1, 2
    )
    torch.testing.assert_close(prediction, measured, rtol=2e-5, atol=2e-5)

    # The training probe is normalized, but its reciprocal output scale must
    # restore the selected physical probe exactly at the forward boundary.
    restored_probe = torch.as_tensor(
        selected_container["probe_training"]
    ) / torch.as_tensor(selected_container["probe_normalization"])
    torch.testing.assert_close(
        restored_probe,
        torch.as_tensor(selected_container["probe_physical"]),
        rtol=2e-6,
        atol=2e-6,
    )

    stale_prediction, _ = _adapt_and_predict(
        amplitude=train["diffraction"],
        probe=train["probeGuess"],
        object_field=object_field,
        count_amplitude_scale=count_amplitude_scale,
    )
    observed_rate_ratio = float(measured.sum() / stale_prediction.sum())
    assert observed_rate_ratio == pytest.approx(k**2, rel=3e-5)
    assert observed_rate_ratio > 2.0
