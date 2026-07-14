"""Amplitude-mode inline-dataset emission regression (scaling passthrough +
documented probe layout).

History: commit ``82da7796`` reshaped probe AND scaling unconditionally and
degraded trained amp MAE 0.0846 -> 0.233; ``8b3d7a011`` (Task R1-fix)
restored the flat "pre-82da7796 raw convention" for amplitude mode without
root-causing WHY the reshape degraded quality. The 2026-07-12 probe-rank
root-cause analysis (docs/findings.md PROBE-RANK-001) established the
mechanism: the flat (B, H, W) probe broadcast into ProbeIllumination's mode
slot and the coherent mode sum multiplied the predicted field by the batch
size — an accidental, batch-size-dependent physics gain. The documented
per-sample probe layout is now UNCONDITIONAL (design 2026-07-12 §3.4,
docs/specs/spec-ptycho-torch-probe-layout.md); the conditioning benefit of
the accidental gain is preserved by the explicit
``ModelConfig.amplitude_physics_gain`` field instead.

This test drives the real inline-dataset -> DataLoader-collate path in
amplitude mode (the default; no ``physics_forward_mode`` override) with
``batch_size > 1`` and asserts:
  - probe: documented per-sample (C, P, H, W) layout, collated to
    (B, C, P, N, N) -- the flat (B, H, W) emission is banned
    (ProbeIllumination raises ProbeLayoutError on it).
  - scaling: the raw, un-indexed ``scaling_constant`` tensor, collated across
    the batch -- NOT per-sample-selected and collapsed via
    ``_select_scale`` + ``view(-1, 1, 1)[:1]`` (batch[2] is never read by
    the amplitude-mode ``compute_loss``; the 82da7796 scaling collapse
    remains rectangular_scaled-only).
"""
import pytest
import torch

from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    update_legacy_dict,
)
from ptycho import params
from ptycho_torch.workflows import components as torch_components
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_torch_training


@pytest.fixture
def params_cfg_snapshot():
    """Snapshot and restore params.cfg state across tests (POLICY: CONFIG-001)."""
    original = params.cfg.copy()
    yield params.cfg
    params.cfg.clear()
    params.cfg.update(original)


def _build_container(n_samples: int, N: int) -> dict:
    """Dict-based container fixture (duck-typed; mirrors PtychoDataContainerTorch/
    grid_lines_torch_runner.run_torch_training's container shape). ``probe`` is
    shared (H, W) and ``scaling_constant`` carries distinct values so a collapse
    to a single (first) element is observable.
    """
    return {
        "X": torch.rand(n_samples, N, N, 1, dtype=torch.float32),
        "coords_nominal": torch.zeros(n_samples, 1, 2, 1, dtype=torch.float32),
        "coords_relative": torch.zeros(n_samples, 1, 2, 1, dtype=torch.float32),
        "rms_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
        "physics_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
        "probe": (torch.randn(N, N) + 1j * torch.randn(N, N)).to(torch.complex64),
        "scaling_constant": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    }


@pytest.mark.torch
def test_inline_dataset_amplitude_mode_documented_probe_raw_scaling(
    tmp_path, params_cfg_snapshot
):
    """Amplitude default (no ``physics_forward_mode`` override): the collated
    ``probe`` must follow the documented per-sample layout (PROBE-RANK-001)
    while ``scaling`` keeps the raw pre-82da7796 passthrough (unread by the
    amplitude-mode loss)."""
    N = 32
    batch_size = 4
    n_samples = batch_size * 2

    tf_model_cfg = TFModelConfig(N=N, gridsize=1, object_big=False)
    tf_training_cfg = TFTrainingConfig(
        model=tf_model_cfg,
        train_data_file=None,
        output_dir=tmp_path,
        batch_size=batch_size,
        n_groups=n_samples,
    )
    update_legacy_dict(params.cfg, tf_training_cfg)

    train_container = _build_container(n_samples, N)
    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=train_container,
        test_container=None,
        config=tf_training_cfg,
        payload=None,
    )

    tensor_dict, probe, scaling = next(iter(train_loader))

    # Documented probe layout (design 2026-07-12 §3.1/§3.4): per-sample
    # (C=1, P=1, H, W) -> collated (B, 1, 1, N, N). The legacy flat (B, N, N)
    # emission is banned at the model boundary (ProbeLayoutError).
    assert probe.shape == (batch_size, 1, 1, N, N), (
        f"probe shape {tuple(probe.shape)} != documented amplitude-mode "
        f"layout {(batch_size, 1, 1, N, N)} (PROBE-RANK-001)"
    )

    # Pre-82da7796 scaling convention: raw, un-indexed scaling_constant tensor,
    # collated across the batch -- i.e. every batch row is the FULL distinct-value
    # array, not a single value selected+collapsed by `_select_scale` + view.
    n_scaling_values = train_container["scaling_constant"].numel()
    assert scaling.shape == (batch_size, n_scaling_values), (
        f"scaling shape {tuple(scaling.shape)} != expected raw amplitude-mode "
        f"convention {(batch_size, n_scaling_values)}"
    )
    for row in scaling:
        assert torch.equal(row, train_container["scaling_constant"]), (
            "scaling was per-sample-selected/collapsed instead of the raw "
            "pre-82da7796 full-array passthrough"
        )


@pytest.mark.torch
def test_amplitude_runner_does_not_invoke_ci_conversion(tmp_path, monkeypatch):
    N = 32
    n_samples = 4
    amplitude = torch.linspace(
        0.1,
        1.0,
        n_samples * N * N,
        dtype=torch.float32,
    ).reshape(n_samples, N, N, 1).numpy()
    probe = torch.ones(N, N, dtype=torch.complex64).numpy()
    data = {
        "diffraction": amplitude,
        "probeGuess": probe,
        "coords_nominal": torch.zeros(n_samples, 2).numpy(),
    }
    captured = {}

    def fail_ci_conversion(*args, **kwargs):
        raise AssertionError("amplitude mode must not invoke CI conversion")

    def fake_train(
        train_container,
        test_container,
        config,
        execution_config=None,
        overrides=None,
    ):
        captured["train_container"] = train_container
        captured["test_container"] = test_container
        return {"history": {}, "models": {}}

    monkeypatch.setattr(
        torch_components.NormalizedAmplitudeCIDictAdapter,
        "adapt",
        fail_ci_conversion,
    )
    monkeypatch.setattr(torch_components, "_train_with_lightning", fake_train)

    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "output",
        architecture="cnn",
        physics_forward_mode="amplitude",
        torch_loss_mode="mae",
        N=N,
    )
    run_torch_training(cfg, data, data)

    assert "measured_intensity" not in captured["train_container"]
    assert "probe_physical" not in captured["train_container"]
    assert torch.equal(
        torch.from_numpy(captured["train_container"]["observed_images"]),
        torch.from_numpy(amplitude),
    )
