"""Task 2.8/2.9 fix: B>1 batched regression test for the inline-dataset -> collate ->
RectangularScaledDiffraction.forward crash.

Bug (root-caused in the Task 2.9 brief): ``ptycho_torch/workflows/components.py``'s
inline training dataset (``PtychoLightningDataset``, built inside
``_build_lightning_dataloaders``) per-sample selected + reshaped ``rms_scale``/
``phys_scale`` to ``(1, 1, 1)`` but returned ``scaling`` (batch[2], =
``self.scaling_constant``) and ``probe`` (``self.probe``) RAW / un-indexed /
un-reshaped. PyTorch's default ``collate_fn`` then stacked a spurious leading
batch axis onto both (``scale`` -> ``(B, 1, B, 1)`` once combined with
``physics_scaling_constant`` in ``compute_loss``; ``probe`` -> ``(1, 1, B, N, N)``
after broadcasting against the probe mask). ``RectangularScaledDiffraction.forward``
(``ptycho_torch/model.py``) does ``scale.unsqueeze(dim=2)``, which pushed that
stray ``B`` axis into dim 3 where it collided with ``H`` -- but only when
``batch_size > 1``; every existing fixture (``test_rectangular_scaled_forward.py``,
``test_cross_branch_rectangular_parity.py``, the Task 2.1 pin) hand-builds
already-correctly-shaped B=1 or pre-shaped batches, so this bug class was
uncovered.

This test drives the REAL inline-dataset -> DataLoader-collate path via
``_build_lightning_dataloaders`` with ``batch_size > 1``, asserts the collated
``probe``/``scaling`` shapes match the native ``PtychoDataset``/``Collate_Lightning``
contract (``probe=(B, C, 1, N, N)``, ``scale=(B, 1, 1, 1)``), and then feeds that
real collated batch into ``RectangularScaledDiffraction.forward`` -- the exact
crash site -- asserting no shape crash and a finite output.
"""
from types import SimpleNamespace

import pytest
import torch

from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    update_legacy_dict,
)
from ptycho import params
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
)
from ptycho_torch.model import RectangularScaledDiffraction
from ptycho_torch.workflows import components as torch_components


def _rectangular_scaled_payload() -> SimpleNamespace:
    """Task R1-fix: the (C, P, H, W) probe reshape / scaling select+collapse in
    ``PtychoLightningDataset.__getitem__`` is now conditioned on
    ``model_config.physics_forward_mode == 'rectangular_scaled'`` (bisect-report.md
    #4: applying it unconditionally also degraded the amplitude default). The TF
    ``ModelConfig`` used elsewhere in this file has no ``physics_forward_mode``
    field, so the payload carries the PyTorch model config. This fixture stores
    normalized amplitudes and exercises the historical MAE routing, so it also
    declares the complete legacy profile instead of inheriting the CI defaults.
    """
    return SimpleNamespace(
        pt_data_config=PTDataConfig(
            scale_contract_version="legacy_v1",
            measurement_domain="normalized_amplitude",
        ),
        pt_model_config=PTModelConfig(
            physics_forward_mode="rectangular_scaled",
            loss_function="MAE",
        ),
        pt_training_config=PTTrainingConfig(torch_loss_mode="mae"),
    )


@pytest.fixture
def params_cfg_snapshot():
    """Snapshot and restore params.cfg state across tests (POLICY: CONFIG-001)."""
    original = params.cfg.copy()
    yield params.cfg
    params.cfg.clear()
    params.cfg.update(original)


def _build_container(n_samples: int, n_channels: int, N: int) -> dict:
    """Dict-based container fixture (duck-typed; mirrors PtychoDataContainerTorch/
    grid_lines_torch_runner.run_torch_training's container shape). ``X``/coords use
    TensorFlow's channel-LAST convention (``(n, H, W, C)`` / ``(n, 1, 2, C)``) per
    the inline dataset's documented contract (components.py ~L479-506) -- the
    inline ``__getitem__`` permutes a per-sample 3D slice assuming (H, W, C).
    ``probe`` is shared (H, W) -- the same shape every real container/dict source
    in this repo produces -- deliberately NOT pre-indexed/pre-reshaped, so this
    test exercises the inline dataset's own per-sample selection + reshaping.
    """
    return {
        "X": torch.rand(n_samples, N, N, n_channels, dtype=torch.float32),
        "coords_nominal": torch.zeros(n_samples, 1, 2, n_channels, dtype=torch.float32),
        "coords_relative": torch.zeros(n_samples, 1, 2, n_channels, dtype=torch.float32),
        "rms_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
        "physics_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
        "probe": (torch.randn(N, N) + 1j * torch.randn(N, N)).to(torch.complex64),
        "scaling_constant": torch.ones(1, dtype=torch.float32),
    }


@pytest.mark.torch
@pytest.mark.parametrize("batch_size", [2, 4])
def test_inline_dataset_collate_shapes_match_native_contract(
    batch_size, tmp_path, params_cfg_snapshot
):
    """The collated batch[1] (probe) / batch[2] (scaling) shapes produced by the
    inline dataset must match the native PtychoDataset/Collate_Lightning contract:
    probe=(B, C, 1, N, N), scale=(B, 1, 1, 1) -- NOT the pre-fix (B, N, N) /
    (B, 1) shapes that collided inside RectangularScaledDiffraction.forward.
    """
    N, C = 64, 1
    n_samples = batch_size * 2  # multiple full batches so B>1 collation is exercised

    tf_model_cfg = TFModelConfig(N=N, gridsize=1, object_big=False)
    tf_training_cfg = TFTrainingConfig(
        model=tf_model_cfg,
        train_data_file=None,
        output_dir=tmp_path,
        batch_size=batch_size,
        n_groups=n_samples,
    )
    update_legacy_dict(params.cfg, tf_training_cfg)

    train_container = _build_container(n_samples, C, N)
    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=train_container,
        test_container=None,
        config=tf_training_cfg,
        payload=_rectangular_scaled_payload(),
    )

    tensor_dict, probe, scaling = next(iter(train_loader))

    assert probe.shape == (batch_size, C, 1, N, N), (
        f"probe shape {tuple(probe.shape)} != expected {(batch_size, C, 1, N, N)} "
        "(native PtychoDataset/Collate_Lightning contract)"
    )
    assert scaling.shape == (batch_size, 1, 1, 1), (
        f"scaling shape {tuple(scaling.shape)} != expected {(batch_size, 1, 1, 1)}"
    )
    assert tensor_dict["physics_scaling_constant"].shape == (batch_size, 1, 1, 1)


@pytest.mark.torch
@pytest.mark.parametrize("batch_size", [2, 4])
def test_inline_dataset_collate_rectangular_scaled_forward_no_crash(
    batch_size, tmp_path, params_cfg_snapshot
):
    """Smallest faithful reproduction of the reported crash: drive the REAL
    inline-dataset -> DataLoader-collate batch straight into
    RectangularScaledDiffraction.forward (the exact site of the reported
    "tensor a (B) must match tensor b (N) at dim 3" crash) with batch_size > 1,
    and assert it no longer crashes and produces a finite output.
    """
    N, C = 64, 1
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

    train_container = _build_container(n_samples, C, N)
    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=train_container,
        test_container=None,
        config=tf_training_cfg,
        payload=_rectangular_scaled_payload(),
    )

    tensor_dict, probe, scaling = next(iter(train_loader))
    physics_scale = tensor_dict["physics_scaling_constant"]
    experiment_ids = tensor_dict["experiment_id"]

    # compute_loss's rectangular_mode output_scale (ptycho_torch/model.py ~L2267).
    output_scale = torch.sqrt(1.0 / (scaling ** 2 * physics_scale + 1e-9))

    # A real/imag-derived complex object standing in for the generator's output
    # (compute_loss feeds the generator's combined complex output here, not the
    # raw diffraction images; this is the "smallest faithful reproduction" of
    # that shape contract).
    x = (torch.randn(batch_size, C, N, N) + 1j * torch.randn(batch_size, C, N, N)).to(torch.complex64)

    pt_model_cfg = PTModelConfig()
    rect_scaler = RectangularScaledDiffraction(pt_model_cfg)

    out = rect_scaler(
        x=x,
        I_raw=None,
        probe=probe,
        scale=output_scale,
        experiment_ids=experiment_ids,
        autograd=True,
    )

    assert out.shape == (batch_size, C, N, N)
    assert torch.isfinite(out).all(), "RectangularScaledDiffraction.forward produced non-finite output"
