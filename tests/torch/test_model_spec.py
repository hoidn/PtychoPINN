from dataclasses import fields, replace

import pytest
import torch

from ptycho.config.config import ModelConfig as CanonicalModelConfig
from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)


def _coherent_configs(**model_overrides):
    data = DataConfig(N=64, C=4, grid_size=(2, 2), probe_scale=4.0)
    model = ModelConfig(C_model=4, C_forward=4, **model_overrides)
    canonical = to_model_config(data, model)
    return canonical, data, model


def test_model_spec_declares_every_torch_model_field_exactly_once():
    from ptycho_torch.model_spec import (
        CANONICAL_MODEL_FIELDS,
        TORCH_EXTENSION_FIELDS,
    )

    owned = set(CANONICAL_MODEL_FIELDS) | set(TORCH_EXTENSION_FIELDS)
    assert owned == {item.name for item in fields(ModelConfig)}
    assert set(CANONICAL_MODEL_FIELDS).isdisjoint(TORCH_EXTENSION_FIELDS)


def test_model_spec_is_versioned_and_materializes_fresh_model_configs():
    from ptycho_torch.model_spec import CURRENT_MODEL_SPEC_VERSION, derive_model_spec

    canonical, data, model = _coherent_configs(fno_width=48)
    spec = derive_model_spec(canonical, model, data)

    assert spec.schema_version == CURRENT_MODEL_SPEC_VERSION
    first = spec.to_model_config()
    second = spec.to_model_config()
    assert first == model
    assert second == model
    assert first is not second


def test_model_spec_rejects_shared_and_data_join_mismatches():
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs(fno_width=48)
    with pytest.raises(ValueError, match=r"fno_width.*canonical.*48|fno_width.*48.*canonical"):
        derive_model_spec(replace(canonical, fno_width=32), model, data)

    with pytest.raises(ValueError, match=r"C_model.*data.*4"):
        derive_model_spec(canonical, replace(model, C_model=1), data)

    with pytest.raises(ValueError, match=r"gridsize.*grid_size"):
        derive_model_spec(replace(canonical, gridsize=1), model, data)


def test_model_spec_preserves_tensor_mask_without_aliasing():
    from ptycho_torch.model_spec import derive_model_spec

    mask = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    data = DataConfig(N=64, C=4, grid_size=(2, 2), probe_scale=4.0)
    model = ModelConfig(
        C_model=4,
        C_forward=4,
        probe_mask=True,
        probe_mask_tensor=mask,
    )
    canonical = to_model_config(data, model)
    spec = derive_model_spec(canonical, model, data)

    mask.add_(100)
    first = spec.to_model_config()
    second = spec.to_model_config()
    assert torch.equal(first.probe_mask_tensor, torch.arange(16).reshape(4, 4))
    assert first.probe_mask_tensor is not second.probe_mask_tensor


def test_model_spec_rejects_canonical_probe_mask_contradiction():
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs(probe_mask=True)
    with pytest.raises(ValueError, match=r"probe_mask.*canonical"):
        derive_model_spec(replace(canonical, probe_mask=False), model, data)


def test_model_spec_parity_settings_are_rebuild_identity():
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs()
    spec = derive_model_spec(
        canonical,
        model,
        data,
        parity_scale_mode="fixed",
        parity_fixed_delta=1.25,
        parity_init_scheme="tf_glorot",
    )

    assert spec.parity_scale_mode == "fixed"
    assert spec.parity_fixed_delta == 1.25
    assert spec.parity_init_scheme == "tf_glorot"
    assert spec.to_payload()["schema_version"] == spec.schema_version


def test_application_factory_composes_sections_without_runtime_config():
    from inspect import signature

    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model import PtychoPINN_Lightning
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs()
    spec = derive_model_spec(canonical, model, data)
    training = TrainingConfig(torch_loss_mode="poisson")
    inference = InferenceConfig()

    module = build_ptychopinn_application(spec, data, training, inference)

    assert isinstance(module, PtychoPINN_Lightning)
    assert module.model_config == model
    assert "execution_config" not in signature(build_ptychopinn_application).parameters


def test_application_factory_rejects_loss_identity_contradiction():
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs(loss_function="Poisson")
    spec = derive_model_spec(canonical, model, data)

    with pytest.raises(ValueError, match=r"torch_loss_mode.*model.*loss_function"):
        build_ptychopinn_application(
            spec,
            data,
            TrainingConfig(torch_loss_mode="mae"),
            InferenceConfig(),
        )


def test_training_payload_carries_current_model_spec(tmp_path):
    import numpy as np

    from ptycho_torch.config_factory import create_training_payload

    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        diffraction=np.zeros((1, 1, 64, 64), dtype=np.float32),
        probeGuess=np.ones((64, 64), dtype=np.complex64),
    )
    payload = create_training_payload(
        train_data_file=data_path,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 1, "gridsize": 1},
    )

    assert payload.model_spec.to_model_config() == payload.pt_model_config
    assert payload.model_spec.schema_version == "torch-model-spec-v1"


@pytest.mark.parametrize(
    "architecture",
    [
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
    ],
)
def test_application_factory_rebuilds_registered_missing_architectures(architecture):
    from ptycho_torch.application_factory import build_ptychopinn_application
    from ptycho_torch.model_spec import derive_model_spec

    canonical, data, model = _coherent_configs(
        architecture=architecture,
        fno_width=4,
        fno_modes=2,
        fno_blocks=3,
        hybrid_resnet_blocks=1,
        hybrid_downsample_steps=1,
        spectral_bottleneck_blocks=1,
        spectral_bottleneck_modes=2,
    )
    spec = derive_model_spec(canonical, model, data)
    training = TrainingConfig(torch_loss_mode="poisson")

    first = build_ptychopinn_application(spec, data, training, InferenceConfig())
    second = build_ptychopinn_application(spec, data, training, InferenceConfig())
    assert {
        key: tuple(value.shape) for key, value in first.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in second.state_dict().items()
    }
    second.load_state_dict(first.state_dict(), strict=True)
