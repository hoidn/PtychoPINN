"""TDD tests for Task 8 (POISSON-SCALE-001 fix): attach an absolute
photon-count physics scale to the grid-lines dict-container path via
``ptycho_torch.workflows.components.derive_dict_physics_scale``.

Exercises the helper and the existing ``_get_tensor``/``_select_scale``
dataloader wiring directly (CPU, tiny synthetic containers) -- no training
run, no GPU, no mocks of the components under test.
"""
import math

import numpy as np
import pytest
import torch
import torch.distributions as dist

from ptycho import params
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    update_legacy_dict,
)
from ptycho_torch.model import PoissonLoss
from ptycho_torch.workflows import components as torch_components
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

NPHOTONS = 1e9


@pytest.fixture
def params_cfg_snapshot():
    """Snapshot and restore params.cfg state across tests (POLICY: CONFIG-001)."""
    original = params.cfg.copy()
    yield params.cfg
    params.cfg.clear()
    params.cfg.update(original)


def _closed_form_scale(x: np.ndarray, nphotons: float) -> float:
    """Reference S = sqrt(nphotons / mean_patterns(sum_HW(x**2))) for the toy fixture."""
    per_sample = np.sum(x.astype(np.float64) ** 2, axis=(1, 2, 3))
    return math.sqrt(nphotons / per_sample.mean())


def _build_container(n_samples: int, N: int, seed: int = 0) -> dict:
    """Dict-based container fixture mirroring run_torch_training's container
    shape: small amplitude X with a known mean_patterns(sum_HW(X**2)), a
    shared probe, zero coords, observed_images=X."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.1, 1.0, size=(n_samples, N, N, 1)).astype(np.float32)
    probe = np.ones((N, N), dtype=np.complex64)
    return {
        "X": X,
        "observed_images": X.copy(),
        "coords_nominal": np.zeros((n_samples, 1, 2, 1), dtype=np.float32),
        "coords_relative": np.zeros((n_samples, 1, 2, 1), dtype=np.float32),
        "probe": probe,
    }


def _training_config(tmp_path, N: int, batch_size: int, n_samples: int) -> TFTrainingConfig:
    tf_model_cfg = TFModelConfig(N=N, gridsize=1, object_big=False)
    return TFTrainingConfig(
        model=tf_model_cfg,
        train_data_file=None,
        output_dir=tmp_path,
        batch_size=batch_size,
        n_groups=n_samples,
    )


@pytest.mark.torch
def test_auto_mode_exposes_physics_scale_matching_closed_form(tmp_path, params_cfg_snapshot):
    """(a) auto exposes physics_scale=S, matching the closed form both from
    the helper's return value and from the collated dataloader batch."""
    N = 16
    batch_size = 4
    n_samples = batch_size * 2

    container = _build_container(n_samples, N)
    expected_S = _closed_form_scale(container["X"], NPHOTONS)

    returned_scale = torch_components.derive_dict_physics_scale(container, NPHOTONS, "auto")

    assert returned_scale.item() == pytest.approx(expected_S, rel=1e-5)
    assert container["physics_scaling_constant"].shape == (1, 1, 1)
    assert container["physics_scaling_constant"].dtype == torch.float32

    tf_training_cfg = _training_config(tmp_path, N, batch_size, n_samples)
    update_legacy_dict(params.cfg, tf_training_cfg)

    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=tf_training_cfg,
        payload=None,
    )
    tensor_dict, _, _ = next(iter(train_loader))
    physics_scale = tensor_dict["physics_scaling_constant"]

    assert physics_scale.shape == (batch_size, 1, 1, 1)
    assert torch.allclose(physics_scale, torch.full_like(physics_scale, expected_S), rtol=1e-5)


@pytest.mark.torch
def test_off_mode_yields_exactly_one(tmp_path, params_cfg_snapshot):
    """(b) off leaves physics_scaling_constant absent; collation defaults it
    to 1.0, byte-identical to today's behavior."""
    N = 16
    batch_size = 4
    n_samples = batch_size * 2

    container = _build_container(n_samples, N)

    returned_scale = torch_components.derive_dict_physics_scale(container, NPHOTONS, "off")

    assert returned_scale is None
    assert "physics_scaling_constant" not in container

    tf_training_cfg = _training_config(tmp_path, N, batch_size, n_samples)
    update_legacy_dict(params.cfg, tf_training_cfg)

    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=tf_training_cfg,
        payload=None,
    )
    tensor_dict, _, _ = next(iter(train_loader))
    physics_scale = tensor_dict["physics_scaling_constant"]

    assert physics_scale.shape == (batch_size, 1, 1, 1)
    assert torch.equal(physics_scale, torch.ones_like(physics_scale))


@pytest.mark.torch
def test_auto_mode_derives_scale_from_observed_images_not_conditioned_x(params_cfg_snapshot):
    """auto must derive S from observed_images (the loss-side raw diffraction,
    unconditioned in every input_conditioning_mode), not from X, which can
    carry appended non-physical conditioning channels that would corrupt S."""
    container = _build_container(4, 16)
    # Mimic coordinate_grid conditioning: X gains 2 non-physical channels
    # while observed_images stays pure diffraction.
    n, h, w, _ = container["X"].shape
    coords = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(1, h, w, 1)
    extra = np.broadcast_to(coords, (n, h, w, 1))
    container["X"] = np.concatenate([container["X"], extra, extra], axis=-1)

    expected_S = _closed_form_scale(container["observed_images"], NPHOTONS)
    corrupted_S = _closed_form_scale(container["X"], NPHOTONS)
    assert corrupted_S != pytest.approx(expected_S, rel=1e-3)

    returned_scale = torch_components.derive_dict_physics_scale(container, NPHOTONS, "auto")

    assert returned_scale.item() == pytest.approx(expected_S, rel=1e-5)


@pytest.mark.torch
def test_both_sides_lift_identity(tmp_path, params_cfg_snapshot):
    """(c) both-sides-lift identity: the loss with physics=S (using the real
    production PoissonLoss class on the wiring-derived, collated
    physics_scale) equals the exact hand-computed
    mean(-Independent(Poisson((pred*S)**2), 3).log_prob((obs*S)**2)) / mean(obs),
    and the batch-wired physics_scale equals the closed-form S. Does NOT
    assert L(auto)/L(off) == S**2 (Task 7 §4b: operating-point dependent).
    """
    N = 16
    batch_size = 4
    n_samples = batch_size * 2

    container = _build_container(n_samples, N, seed=1)
    expected_S = _closed_form_scale(container["X"], NPHOTONS)
    torch_components.derive_dict_physics_scale(container, NPHOTONS, "auto")

    tf_training_cfg = _training_config(tmp_path, N, batch_size, n_samples)
    update_legacy_dict(params.cfg, tf_training_cfg)

    train_loader, _ = torch_components._build_lightning_dataloaders(
        train_container=container,
        test_container=None,
        config=tf_training_cfg,
        payload=None,
    )
    tensor_dict, _, _ = next(iter(train_loader))
    physics_scale = tensor_dict["physics_scaling_constant"]
    obs = tensor_dict["observed_images"]

    assert torch.allclose(physics_scale, torch.full_like(physics_scale, expected_S), rtol=1e-5)

    torch.manual_seed(0)
    pred = torch.rand_like(obs) + 0.1

    pred_physics = pred * physics_scale
    obs_physics = obs * physics_scale

    actual_loss = PoissonLoss()(pred_physics, obs_physics).mean() / obs.mean()

    hand_computed = -dist.Independent(
        dist.Poisson(pred_physics ** 2, validate_args=False), 3
    ).log_prob(obs_physics ** 2)
    hand_computed = hand_computed.mean() / obs.mean()

    assert torch.allclose(actual_loss, hand_computed, rtol=1e-6)


def test_count_scale_mode_defaults_to_off(tmp_path):
    """Red-gate: the Task 9 GPU A/B falsified auto-mode's outcome invariance,
    so the runner's count_scale_mode must default to 'off' (opt-in only)."""
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path / "output",
        architecture="fno",
    )
    assert cfg.count_scale_mode == "off"
