"""Tests for PtychoPINN_Lightning's default-off TF-parity intensity-scale
mechanism (``parity_scale_mode``/``parity_fixed_delta``/``parity_init_scheme``).

See docs/plans/2026-07-08-cnn-n128-tf-parity.md Task 1. TF tie direction
(ptycho/model.py:291-298, read-only): input is DIVIDED by exp(w), output is
MULTIPLIED by exp(w).

CPU-only, no dataloader -- follows the hand-built-tensor fixture convention
established by tests/torch/test_physics_scale_loss.py (N=64, C=1,
object_big=False; N must be 64/128/256, see ptycho_torch/model.py Encoder).
"""
import json

import pytest
import torch
from torch import nn

try:
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
except ImportError:  # pragma: no cover - torch backend is mandatory (POLICY-001)
    pl = None
    Trainer = None

from ptycho_torch.config_params import (
    DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig,
)
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import (
    load_checkpoint_with_configs,
    load_configs_from_checkpoint,
)
from ptycho_torch.utils import config_to_json_serializable_dict


def _tiny_configs(**model_overrides):
    model_cfg = ModelConfig(
        C_model=1, C_forward=1, object_big=False, probe_big=False,
        n_filters_scale=1, **model_overrides,
    )
    data_cfg = DataConfig(N=64, C=1, grid_size=(1, 1))
    train_cfg = TrainingConfig(device="cpu")
    infer_cfg = InferenceConfig()
    return model_cfg, data_cfg, train_cfg, infer_cfg


def _build_model(parity_scale_mode="off", parity_fixed_delta=0.0, parity_init_scheme="default"):
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    return PtychoPINN_Lightning(
        model_cfg, data_cfg, train_cfg, infer_cfg,
        parity_scale_mode=parity_scale_mode,
        parity_fixed_delta=parity_fixed_delta,
        parity_init_scheme=parity_init_scheme,
    )


def _save_checkpoint_with_configs(model, model_cfg, data_cfg, train_cfg, infer_cfg, tmp_path, name="run"):
    """Mimic ptycho_torch.lightning_utils.ConfigLogger's on-disk layout
    (``run_dir/configs/*.json`` + ``run_dir/checkpoints/*.ckpt``) so
    ``load_checkpoint_with_configs`` -- which reads that layout -- can be
    exercised without a full train_lightning_only training loop.
    """
    run_dir = tmp_path / name
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "test.ckpt"

    configs = {
        "data_config": data_cfg,
        "model_config": model_cfg,
        "training_config": train_cfg,
        "inference_config": infer_cfg,
        "datagen_config": DatagenConfig(),
    }
    for cfg_name, cfg in configs.items():
        (config_dir / f"{cfg_name}.json").write_text(
            json.dumps(config_to_json_serializable_dict(cfg))
        )

    trainer = Trainer(
        max_epochs=0, enable_checkpointing=True, default_root_dir=str(run_dir),
        logger=False, enable_progress_bar=False, accelerator="cpu",
    )
    trainer.strategy._lightning_module = model
    trainer.save_checkpoint(str(ckpt_path))
    return ckpt_path


# ---------------------------------------------------------------------------
# off mode: no delta parameter is created; hparams record the mode
# ---------------------------------------------------------------------------

def test_off_mode_creates_no_delta_parameter():
    model = _build_model(parity_scale_mode="off")

    assert "log_scale_delta" not in dict(model.named_parameters())
    assert not hasattr(model, "log_scale_delta")
    assert model.hparams["parity_scale_mode"] == "off"


def test_old_style_checkpoint_loads_via_load_checkpoint_with_configs(tmp_path):
    """A checkpoint from a model constructed WITHOUT any parity kwargs (i.e.
    the pre-existing call signature) must still load cleanly."""
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)

    ckpt_path = _save_checkpoint_with_configs(model, model_cfg, data_cfg, train_cfg, infer_cfg, tmp_path)

    loaded_model, _configs = load_checkpoint_with_configs(str(ckpt_path), PtychoPINN_Lightning, device="cpu")

    assert loaded_model.parity_scale_mode == "off"
    assert "log_scale_delta" not in dict(loaded_model.named_parameters())


def test_milestone_checkpoint_loads_configs_from_run_directory(tmp_path):
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)
    checkpoint_path = _save_checkpoint_with_configs(
        model,
        model_cfg,
        data_cfg,
        train_cfg,
        infer_cfg,
        tmp_path,
    )
    milestone_path = checkpoint_path.parent / "milestones" / "epoch-0005.ckpt"
    milestone_path.parent.mkdir()
    checkpoint_path.replace(milestone_path)

    loaded = load_configs_from_checkpoint(str(milestone_path))

    expected = (data_cfg, model_cfg, train_cfg, infer_cfg, DatagenConfig())
    expected_payloads = tuple(
        json.loads(json.dumps(config_to_json_serializable_dict(config)))
        for config in expected
    )
    actual_payloads = tuple(
        json.loads(json.dumps(config_to_json_serializable_dict(config)))
        for config in loaded
    )
    assert actual_payloads == expected_payloads


def test_checkpoint_in_unsupported_nested_layout_is_rejected(tmp_path):
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    model = PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg)
    checkpoint_path = _save_checkpoint_with_configs(
        model,
        model_cfg,
        data_cfg,
        train_cfg,
        infer_cfg,
        tmp_path,
    )
    unsupported_path = (
        checkpoint_path.parent.parent / "other" / "nested" / "epoch-0005.ckpt"
    )
    unsupported_path.parent.mkdir(parents=True)
    checkpoint_path.replace(unsupported_path)

    with pytest.raises(FileNotFoundError, match="Unsupported checkpoint layout"):
        load_configs_from_checkpoint(str(unsupported_path))


# ---------------------------------------------------------------------------
# trainable/fixed delta semantics
# ---------------------------------------------------------------------------

def test_delta_param_registered_in_optimizer():
    model = _build_model(parity_scale_mode="tied")

    result = model.configure_optimizers()
    optimizer = result["optimizer"]
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    assert any(p is model.log_scale_delta for p in all_params)


def test_fixed_mode_frozen():
    model = _build_model(parity_scale_mode="fixed", parity_fixed_delta=0.7)

    assert model.log_scale_delta.requires_grad is False
    assert float(model.log_scale_delta) == pytest.approx(0.7)


def test_tied_factors_are_inverse():
    tied = _build_model(parity_scale_mode="tied", parity_fixed_delta=0.3)
    f_in, f_out = tied._parity_scale_factors()
    torch.testing.assert_close(f_in * f_out, torch.tensor(1.0))

    input_only = _build_model(parity_scale_mode="input", parity_fixed_delta=0.3)
    f_in2, f_out2 = input_only._parity_scale_factors()
    assert f_out2 is None  # output side left untouched -- effective factor 1

    output_only = _build_model(parity_scale_mode="output", parity_fixed_delta=0.3)
    f_in3, f_out3 = output_only._parity_scale_factors()
    assert f_in3 is None  # input side left untouched -- effective factor 1


# ---------------------------------------------------------------------------
# checkpoint roundtrip preserves parity kwargs + delta value
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_preserves_parity(tmp_path):
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    model = PtychoPINN_Lightning(
        model_cfg, data_cfg, train_cfg, infer_cfg,
        parity_scale_mode="tied", parity_fixed_delta=0.37, parity_init_scheme="default",
    )
    with torch.no_grad():
        model.log_scale_delta.add_(0.05)  # simulate the optimizer having moved it

    ckpt_path = _save_checkpoint_with_configs(model, model_cfg, data_cfg, train_cfg, infer_cfg, tmp_path)

    loaded_model, _configs = load_checkpoint_with_configs(str(ckpt_path), PtychoPINN_Lightning, device="cpu")

    assert loaded_model.parity_scale_mode == "tied"
    assert loaded_model.hparams["parity_scale_mode"] == "tied"
    assert loaded_model.log_scale_delta.detach().item() == pytest.approx(0.42, abs=1e-6)


# ---------------------------------------------------------------------------
# tf_glorot weight-init preset
# ---------------------------------------------------------------------------

def test_tf_glorot_init_applied():
    torch.manual_seed(20260708)
    default_model = _build_model(parity_scale_mode="off", parity_init_scheme="default")
    torch.manual_seed(20260708)
    glorot_model = _build_model(parity_scale_mode="off", parity_init_scheme="tf_glorot")

    default_convs = [m for m in default_model.model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    glorot_convs = [m for m in glorot_model.model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    assert len(default_convs) == len(glorot_convs) > 0

    any_weight_changed = False
    for dconv, gconv in zip(default_convs, glorot_convs):
        if gconv.bias is not None:
            assert torch.equal(gconv.bias, torch.zeros_like(gconv.bias))
        if not torch.equal(dconv.weight, gconv.weight):
            any_weight_changed = True
    assert any_weight_changed


# ---------------------------------------------------------------------------
# fail-fast validation
# ---------------------------------------------------------------------------

def test_invalid_parity_scale_mode_raises():
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    with pytest.raises(ValueError, match="parity_scale_mode"):
        PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg, parity_scale_mode="garbage")


def test_invalid_parity_init_scheme_raises():
    model_cfg, data_cfg, train_cfg, infer_cfg = _tiny_configs()
    with pytest.raises(ValueError, match="parity_init_scheme"):
        PtychoPINN_Lightning(model_cfg, data_cfg, train_cfg, infer_cfg, parity_init_scheme="garbage")
