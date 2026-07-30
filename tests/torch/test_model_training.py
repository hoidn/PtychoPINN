"""Tests for PyTorch model training functionality."""
import pytest
import torch


def _build_lightning_module(training_config):
    """Build the real Lightning module with a supplied Torch training config."""
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho_torch.config_params import DataConfig, InferenceConfig, ModelConfig
    from ptycho_torch.model import PtychoPINN_Lightning

    update_legacy_dict(params.cfg, training_config)
    return PtychoPINN_Lightning(
        model_config=ModelConfig(),
        data_config=DataConfig(N=64, C=1, grid_size=(1, 1)),
        training_config=training_config,
        inference_config=InferenceConfig(),
    )


def test_configure_optimizers_scheduler_plateau_uses_resolved_learning_rate():
    """Test that ReduceLROnPlateau uses TrainingConfig params and monitor."""
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho_torch.config_params import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
    from ptycho_torch.model import PtychoPINN_Lightning

    model_cfg = ModelConfig()
    data_cfg = DataConfig(N=64, C=1, grid_size=(1, 1))
    train_cfg = TrainingConfig(
        train_data_file="train.npz",
        test_data_file="test.npz",
        output_dir="training_outputs",
        learning_rate=4e-4,
        scheduler="ReduceLROnPlateau",
        plateau_factor=0.25,
        plateau_patience=5,
        plateau_min_lr=1e-5,
        plateau_threshold=1e-3,
    )
    infer_cfg = InferenceConfig()

    update_legacy_dict(params.cfg, train_cfg)

    module = PtychoPINN_Lightning(
        model_config=model_cfg,
        data_config=data_cfg,
        training_config=train_cfg,
        inference_config=infer_cfg,
    )
    result = module.configure_optimizers()
    assert result["optimizer"].param_groups[0]["lr"] == pytest.approx(4e-4)
    sched_dict = result["lr_scheduler"]

    assert sched_dict["monitor"] == module.val_loss_name
    scheduler = sched_dict["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.factor == 0.25
    assert scheduler.patience == 5
    assert scheduler.min_lrs == [1e-5]
    assert scheduler.threshold == 1e-3


def test_configure_optimizers_uses_sgd_training_config_values():
    """The module must not reinterpret a flat Torch optimizer string."""
    from ptycho_torch.config_params import TrainingConfig

    training_config = TrainingConfig(
        optimizer="sgd",
        momentum=0.37,
        weight_decay=0.0123,
    )

    result = _build_lightning_module(training_config).configure_optimizers()
    optimizer = result["optimizer"]

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == pytest.approx(0.37)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.0123)


def test_configure_optimizers_uses_adamw_training_config_values():
    """AdamW selection and non-default hyperparameters must reach PyTorch."""
    from ptycho_torch.config_params import TrainingConfig

    training_config = TrainingConfig(
        optimizer="adamw",
        weight_decay=0.045,
        adam_beta1=0.81,
        adam_beta2=0.93,
    )

    result = _build_lightning_module(training_config).configure_optimizers()
    optimizer = result["optimizer"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.045)
    assert optimizer.defaults["betas"] == pytest.approx((0.81, 0.93))


@pytest.mark.parametrize("clip_algorithm", ["norm", "value", "agc"])
def test_manual_gradient_clip_uses_resolved_training_config(
    clip_algorithm,
    monkeypatch,
):
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho_torch import train_utils
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model import PtychoPINN_Lightning

    training_config = TrainingConfig(
        gradient_clip_val=0.25,
        gradient_clip_algorithm=clip_algorithm,
        accum_steps=1,
    )
    update_legacy_dict(params.cfg, training_config)
    module = PtychoPINN_Lightning(
        model_config=ModelConfig(),
        data_config=DataConfig(N=64, C=1, grid_size=(1, 1)),
        training_config=training_config,
        inference_config=InferenceConfig(),
    )
    calls = []

    class FakeOptimizer:
        def step(self):
            calls.append(("step", None))

        def zero_grad(self):
            calls.append(("zero_grad", None))

    module.optimizers = lambda: FakeOptimizer()
    module.compute_loss = lambda _batch: torch.tensor(2.0, requires_grad=True)

    def fake_backward(_loss):
        parameter = next(module.parameters())
        parameter.grad = torch.ones_like(parameter)

    module.manual_backward = fake_backward
    module.log = lambda *_args, **_kwargs: None
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *_args, **kwargs: calls.append(("norm", kwargs["max_norm"])),
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_value_",
        lambda _parameters, value: calls.append(("value", value)),
    )
    monkeypatch.setattr(
        train_utils,
        "adaptive_gradient_clip_",
        lambda _parameters, *, clip_factor: calls.append(
            ("agc", clip_factor)
        ),
    )

    module.training_step({}, batch_idx=0)

    assert module.training_config is training_config
    assert module.gradient_clip_val == 0.25
    assert (clip_algorithm, 0.25) in calls
    assert {
        name for name, _value in calls if name in {"norm", "value", "agc"}
    } == {clip_algorithm}


def test_configure_optimizers_selects_warmup_scheduler():
    """The configured scheduler must survive the TrainingConfig-to-model path."""
    from ptycho_torch.config_params import TrainingConfig

    training_config = TrainingConfig(
        epochs=12,
        scheduler="WarmupCosine",
        lr_warmup_epochs=3,
        lr_min_ratio=0.07,
    )

    result = _build_lightning_module(training_config).configure_optimizers()
    scheduler = result["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert scheduler._milestones == [3]
    assert scheduler._schedulers[1].eta_min == pytest.approx(7e-5)


class TestOptimizerSelection:
    """Tests for optimizer selection in PtychoPINN_Lightning.

    Task ID: FNO-STABILITY-OVERHAUL-001 Phase 8 Task 1
    """

    def _build_optimizer(self, optimizer_name, **kwargs):
        """Helper: build optimizer via the same logic as configure_optimizers."""
        from ptycho_torch.model import _build_optimizer
        model = torch.nn.Linear(4, 4)
        return _build_optimizer(model.parameters(), lr=1e-3, optimizer=optimizer_name, **kwargs)

    def test_configures_sgd(self):
        """Test that optimizer='sgd' returns SGD with momentum."""
        opt = self._build_optimizer('sgd', momentum=0.9, weight_decay=0.0,
                                    adam_beta1=0.9, adam_beta2=0.999)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults['momentum'] == 0.9
        assert opt.defaults['weight_decay'] == 0.0

    def test_configures_adamw(self):
        """Test that optimizer='adamw' returns AdamW with weight_decay."""
        opt = self._build_optimizer('adamw', momentum=0.9, weight_decay=0.01,
                                    adam_beta1=0.9, adam_beta2=0.999)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults['weight_decay'] == 0.01
        assert opt.defaults['betas'] == (0.9, 0.999)

    def test_configures_adam(self):
        """Test that optimizer='adam' returns Adam (default)."""
        opt = self._build_optimizer('adam', momentum=0.9, weight_decay=0.0,
                                    adam_beta1=0.9, adam_beta2=0.999)
        assert isinstance(opt, torch.optim.Adam)

    def test_invalid_optimizer_raises(self):
        """Test that unsupported optimizer string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            self._build_optimizer('rmsprop', momentum=0.9, weight_decay=0.0,
                                  adam_beta1=0.9, adam_beta2=0.999)
