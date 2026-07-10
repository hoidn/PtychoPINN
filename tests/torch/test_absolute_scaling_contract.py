from types import SimpleNamespace

import pytest

from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.scaling_contract import (
    CI_SCALE_CONTRACT,
    COUNT_INTENSITY,
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
    ResolvedScaleContract,
    ci_scaling_active,
    resolve_scale_contract,
    validate_scale_contract,
)


CI_PROFILE = ResolvedScaleContract(CI_SCALE_CONTRACT, COUNT_INTENSITY)
LEGACY_PROFILE = ResolvedScaleContract(
    LEGACY_SCALE_CONTRACT,
    NORMALIZED_AMPLITUDE,
)


@pytest.mark.parametrize(
    ("version", "measurement_domain", "expected"),
    [
        (None, None, CI_PROFILE),
        (CI_SCALE_CONTRACT, None, CI_PROFILE),
        (None, COUNT_INTENSITY, CI_PROFILE),
        (CI_SCALE_CONTRACT, COUNT_INTENSITY, CI_PROFILE),
        (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE, LEGACY_PROFILE),
    ],
)
def test_resolve_scale_contract_accepts_only_supported_profiles(
    version,
    measurement_domain,
    expected,
):
    assert resolve_scale_contract(version, measurement_domain) == expected


@pytest.mark.parametrize(
    ("version", "measurement_domain"),
    [
        (None, NORMALIZED_AMPLITUDE),
        (None, "unknown_domain"),
        (CI_SCALE_CONTRACT, NORMALIZED_AMPLITUDE),
        (CI_SCALE_CONTRACT, "unknown_domain"),
        (LEGACY_SCALE_CONTRACT, None),
        (LEGACY_SCALE_CONTRACT, COUNT_INTENSITY),
        (LEGACY_SCALE_CONTRACT, "unknown_domain"),
        ("unknown_version", None),
        ("unknown_version", COUNT_INTENSITY),
        ("unknown_version", NORMALIZED_AMPLITUDE),
        ("unknown_version", "unknown_domain"),
    ],
)
def test_resolve_scale_contract_rejects_partial_contradictory_and_unknown_profiles(
    version,
    measurement_domain,
):
    with pytest.raises(ValueError, match="scale contract"):
        resolve_scale_contract(version, measurement_domain)


def test_data_config_defaults_to_ci_profile():
    config = DataConfig()

    assert config.scale_contract_version == CI_SCALE_CONTRACT
    assert config.measurement_domain == COUNT_INTENSITY


def test_amplitude_mode_does_not_activate_or_validate_ci_contract():
    data_config = SimpleNamespace(
        scale_contract_version="invalid",
        measurement_domain="invalid",
    )
    model_config = SimpleNamespace(
        physics_forward_mode="amplitude",
        mode="Supervised",
        loss_function="MAE",
    )
    training_config = SimpleNamespace(torch_loss_mode="mae")

    assert ci_scaling_active(model_config) is False
    assert validate_scale_contract(data_config, model_config, training_config) is None


def test_rectangular_ci_accepts_unsupervised_poisson():
    model_config = ModelConfig(
        mode="Unsupervised",
        physics_forward_mode="rectangular_scaled",
    )
    training_config = TrainingConfig(torch_loss_mode="poisson")

    assert ci_scaling_active(model_config) is True
    assert validate_scale_contract(DataConfig(), model_config, training_config) == CI_PROFILE


@pytest.mark.parametrize(
    ("mode", "torch_loss_mode"),
    [
        ("Unsupervised", "mae"),
        ("Supervised", "poisson"),
        ("Unsupervised", "mse"),
    ],
)
def test_rectangular_ci_rejects_non_ci_training_modes(mode, torch_loss_mode):
    model_config = ModelConfig(
        mode=mode,
        physics_forward_mode="rectangular_scaled",
    )
    training_config = TrainingConfig()
    training_config.torch_loss_mode = torch_loss_mode

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        validate_scale_contract(DataConfig(), model_config, training_config)


def test_torch_loss_mode_is_authoritative_over_model_loss_function():
    model_config = ModelConfig(
        mode="Unsupervised",
        physics_forward_mode="rectangular_scaled",
        loss_function="MAE",
    )

    assert validate_scale_contract(
        DataConfig(),
        model_config,
        TrainingConfig(torch_loss_mode="poisson"),
    ) == CI_PROFILE


def test_rectangular_ci_accepts_auxiliary_object_regularizers():
    model_config = ModelConfig(
        mode="Unsupervised",
        physics_forward_mode="rectangular_scaled",
        amp_loss="Total_Variation",
        phase_loss="Mean_Deviation",
    )

    assert validate_scale_contract(
        DataConfig(),
        model_config,
        TrainingConfig(torch_loss_mode="poisson"),
    ) == CI_PROFILE


def test_explicit_legacy_rectangular_profile_keeps_legacy_loss_compatibility():
    data_config = DataConfig(
        scale_contract_version=LEGACY_SCALE_CONTRACT,
        measurement_domain=NORMALIZED_AMPLITUDE,
    )
    model_config = ModelConfig(
        mode="Unsupervised",
        physics_forward_mode="rectangular_scaled",
    )

    assert validate_scale_contract(
        data_config,
        model_config,
        TrainingConfig(torch_loss_mode="mae"),
    ) == LEGACY_PROFILE


def test_lightning_rejects_invalid_ci_before_generator_resolution(monkeypatch):
    import ptycho_torch.model as model_module

    generator_resolved = False

    def fail_if_generator_is_resolved(*args, **kwargs):
        nonlocal generator_resolved
        generator_resolved = True
        raise AssertionError("generator resolution must not run")

    monkeypatch.setattr(
        model_module,
        "_resolve_generator_from_config",
        fail_if_generator_is_resolved,
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        model_module.PtychoPINN_Lightning(
            ModelConfig(
                mode="Unsupervised",
                physics_forward_mode="rectangular_scaled",
            ),
            DataConfig(N=64, grid_size=(1, 1)),
            TrainingConfig(torch_loss_mode="mae"),
            InferenceConfig(),
        )

    assert generator_resolved is False


def test_lightning_dataloader_gate_rejects_before_reading_container():
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    class UnreadableContainer:
        def __getattribute__(self, name):
            raise AssertionError("container must not be read")

    payload = SimpleNamespace(
        pt_data_config=DataConfig(),
        pt_model_config=ModelConfig(
            mode="Unsupervised",
            physics_forward_mode="rectangular_scaled",
        ),
        pt_training_config=TrainingConfig(torch_loss_mode="mae"),
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        _build_lightning_dataloaders(
            UnreadableContainer(),
            None,
            SimpleNamespace(subsample_seed=None),
            payload=payload,
        )


def test_lightning_dataloader_partial_payload_defaults_ci_before_reading_container():
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    class UnreadableContainer:
        def __getattribute__(self, name):
            raise AssertionError("container must not be read")

    payload = SimpleNamespace(
        pt_model_config=ModelConfig(
            mode="Unsupervised",
            physics_forward_mode="rectangular_scaled",
        ),
        pt_training_config=TrainingConfig(torch_loss_mode="mae"),
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        _build_lightning_dataloaders(
            UnreadableContainer(),
            None,
            config=None,
            payload=payload,
        )


def test_lightning_dataloader_gate_without_payload_rejects_before_reading_container():
    from ptycho_torch.workflows.components import _build_lightning_dataloaders

    class UnreadableContainer:
        def __getattribute__(self, name):
            raise AssertionError("container must not be read")

    config = SimpleNamespace(
        model=ModelConfig(
            mode="Unsupervised",
            physics_forward_mode="rectangular_scaled",
        ),
        torch_loss_mode="mae",
        subsample_seed=None,
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        _build_lightning_dataloaders(
            UnreadableContainer(),
            None,
            config,
            payload=None,
        )


def _invalid_ci_entrypoint_configs():
    return (
        DataConfig(),
        ModelConfig(
            mode="Unsupervised",
            physics_forward_mode="rectangular_scaled",
        ),
        TrainingConfig(torch_loss_mode="mae"),
        InferenceConfig(),
        DatagenConfig(),
    )


def test_train_main_rejects_before_data_module_construction(monkeypatch):
    import ptycho_torch.train as train_module

    data_module_constructed = False

    def fail_if_data_module_is_constructed(*args, **kwargs):
        nonlocal data_module_constructed
        data_module_constructed = True
        raise AssertionError("data module construction must not run")

    monkeypatch.setattr(
        train_module,
        "PtychoDataModule",
        fail_if_data_module_is_constructed,
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        train_module.main(
            "unused",
            existing_config=_invalid_ci_entrypoint_configs(),
            disable_mlflow=True,
        )

    assert data_module_constructed is False


def test_train_main_lightning_rejects_before_data_module_construction(
    monkeypatch,
    tmp_path,
):
    import ptycho_torch.train as train_module
    import ptycho_torch.utils as utils_module

    data_module_constructed = False

    def fail_if_data_module_is_constructed(*args, **kwargs):
        nonlocal data_module_constructed
        data_module_constructed = True
        raise AssertionError("data module construction must not run")

    monkeypatch.setattr(utils_module, "load_config_from_json", lambda path: {})
    monkeypatch.setattr(
        utils_module,
        "validate_and_process_config",
        lambda config: (
            None,
            {"physics_forward_mode": "rectangular_scaled"},
            {"torch_loss_mode": "mae"},
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "PtychoDataModule",
        fail_if_data_module_is_constructed,
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        train_module.main_lightning(
            "unused",
            config_path="unused.json",
            output_dir=tmp_path,
        )

    assert data_module_constructed is False


def test_train_lightning_only_rejects_before_data_module_construction(
    monkeypatch,
    tmp_path,
):
    import ptycho_torch.train_lightning_only as train_module

    data_module_constructed = False

    def fail_if_data_module_is_constructed(*args, **kwargs):
        nonlocal data_module_constructed
        data_module_constructed = True
        raise AssertionError("data module construction must not run")

    monkeypatch.setattr(
        train_module,
        "PtychoDataModuleLightning",
        fail_if_data_module_is_constructed,
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        train_module.main(
            "unused",
            existing_config=_invalid_ci_entrypoint_configs(),
            output_dir=tmp_path,
        )

    assert data_module_constructed is False


def test_grid_lines_gate_rejects_before_reading_training_dict(tmp_path):
    from scripts.studies.grid_lines_torch_runner import (
        TorchRunnerConfig,
        run_torch_training,
    )

    class UnreadableDict(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("training dictionary must not be read")

        def __getitem__(self, key):
            raise AssertionError("training dictionary must not be read")

    config = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="fno",
        physics_forward_mode="rectangular_scaled",
        torch_loss_mode="mae",
    )

    with pytest.raises(ValueError, match="ci_intensity_v2"):
        run_torch_training(config, UnreadableDict(), {})
