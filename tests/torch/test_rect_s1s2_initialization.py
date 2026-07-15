"""Focused initialization calibration tests for rectangular CI physics."""

import pytest
import torch

def _tiny_rect_scaled_module():
    """Smallest real PtychoPINN_Lightning under the CI contract (2026-07-14 RCA
    arm: N=64, gridsize=1, architecture='cnn', cnn_output_mode='real_imag',
    physics_forward_mode='rectangular_scaled', count_intensity/ci_intensity_v2,
    amplitude_physics_gain=1.0), plus one real training batch built through the
    same factory (create_training_payload), CI dict adapter, and workflow
    dataloader path that _train_with_lightning / run_torch_training use."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from ptycho_torch import helper as hh
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.model import PtychoPINN_Lightning
    from ptycho_torch.workflows.components import (
        NormalizedAmplitudeCIDictAdapter,
        _build_lightning_dataloaders,
    )

    torch.manual_seed(20260714)
    rng = np.random.default_rng(20260714)
    B, N = 4, 64
    amplitudes = rng.uniform(0.1, 1.0, size=(B, N, N, 1)).astype(np.float32)
    probe = np.ones((N, N), dtype=np.complex64)

    tmpdir = Path(tempfile.mkdtemp(prefix="rect_s1s2_calib_"))
    npz_path = tmpdir / "tiny_train.npz"
    np.savez(
        npz_path,
        diffraction=amplitudes[..., 0],
        xcoords=np.linspace(0.0, 3.0, B),
        ycoords=np.linspace(0.0, 3.0, B),
        probeGuess=probe,
        objectGuess=np.ones((2 * N, 2 * N), dtype=np.complex64),
    )
    payload = create_training_payload(
        train_data_file=npz_path,
        output_dir=tmpdir / "out",
        overrides={
            "n_groups": B,
            "gridsize": 1,
            "architecture": "cnn",
            "model_type": "Unsupervised",
            "cnn_output_mode": "real_imag",
            "physics_forward_mode": "rectangular_scaled",
            "torch_loss_mode": "poisson",
            "scale_contract_version": "ci_intensity_v2",
            "measurement_domain": "count_intensity",
            "amplitude_physics_gain": 1.0,
            "object_big": False,
            "batch_size": B,
        },
    )
    model = PtychoPINN_Lightning(
        payload.pt_model_config,
        payload.pt_data_config,
        payload.pt_training_config,
        InferenceConfig(),
    )

    # Same CI count-domain container preparation as run_torch_training's CI arm
    # (high per-pixel counts reproduce the RCA's init-scale mismatch).
    container = {"observed_images": amplitudes, "probe": probe}
    count_amplitude_scale = hh.derive_intensity_scale_from_amplitudes(
        torch.as_tensor(amplitudes), 1e9
    )
    NormalizedAmplitudeCIDictAdapter(
        count_amplitude_scale=count_amplitude_scale,
        N=N,
    ).adapt(container)
    container["X"] = container["measured_intensity"]

    train_loader, _ = _build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=None,
        payload=payload,
    )
    batch = next(iter(train_loader))
    return model, batch


def test_calibrate_rect_s1s2_equalizes_loss_domain_means():
    model, batch = _tiny_rect_scaled_module()
    s_init = model.calibrate_rect_s1s2(batch)
    assert s_init is not None and s_init > 0
    scaler = [m for m in model.modules()
              if type(m).__name__ == "RectangularScaledDiffraction"][0]
    assert torch.allclose(scaler.s1.detach(), torch.full_like(scaler.s1, s_init))
    assert torch.allclose(scaler.s2.detach(), torch.full_like(scaler.s2, s_init))
    # after calibration the compared means agree within 2x (perfect equality is
    # not required: the object prediction is random-init; the point is removing
    # the orders-of-magnitude mismatch that drove head saturation)
    pred_mean, target_mean = model._last_calibration_means
    assert 0.5 < float(pred_mean / target_mean) < 2.0


def test_calibrate_rect_s1s2_is_noop_for_amplitude_mode():
    model, batch = _tiny_rect_scaled_module()
    model.model_config.physics_forward_mode = "amplitude"
    assert model.calibrate_rect_s1s2(batch) is None
