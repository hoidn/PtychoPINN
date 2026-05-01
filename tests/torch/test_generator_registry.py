# tests/torch/test_generator_registry.py
import pytest
from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho_torch.generators.registry import resolve_generator


def test_resolve_generator_cnn():
    cfg = TrainingConfig(model=ModelConfig(architecture='cnn'))
    gen = resolve_generator(cfg)
    assert gen.name == 'cnn'


def test_resolve_generator_unknown_raises():
    cfg = TrainingConfig(model=ModelConfig(architecture='unknown'))
    with pytest.raises(ValueError):
        resolve_generator(cfg)


def test_resolve_generator_fno_vanilla():
    cfg = TrainingConfig(model=ModelConfig(architecture='fno_vanilla'))
    gen = resolve_generator(cfg)
    assert gen.name == 'fno_vanilla'


def test_resolve_generator_hybrid_resnet():
    cfg = TrainingConfig(model=ModelConfig(architecture='hybrid_resnet'))
    gen = resolve_generator(cfg)
    assert gen.name == 'hybrid_resnet'


def test_resolve_generator_hybrid_resnet_ffno_bottleneck():
    cfg = TrainingConfig(model=ModelConfig(architecture='hybrid_resnet_ffno_bottleneck'))
    gen = resolve_generator(cfg)
    assert gen.name == 'hybrid_resnet_ffno_bottleneck'


def test_resolve_generator_ffno():
    cfg = TrainingConfig(model=ModelConfig(architecture='ffno'))
    gen = resolve_generator(cfg)
    assert gen.name == 'ffno'


def test_resolve_generator_neuralop_uno():
    cfg = TrainingConfig(model=ModelConfig(architecture='neuralop_uno'))
    gen = resolve_generator(cfg)
    assert gen.name == 'neuralop_uno'


def test_resolve_generator_spectral_resnet_bottleneck_linear_decoder():
    cfg = TrainingConfig(model=ModelConfig(architecture='spectral_resnet_bottleneck_linear_decoder'))
    gen = resolve_generator(cfg)
    assert gen.name == 'spectral_resnet_bottleneck_linear_decoder'


def test_ffno_generator_builds_lightning_model():
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig as PTInferenceConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )
    from ptycho_torch.model import PtychoPINN_Lightning

    cfg = TrainingConfig(model=ModelConfig(architecture='ffno', N=64, gridsize=1))
    gen = resolve_generator(cfg)

    pt_configs = {
        "data_config": DataConfig(N=64, C=1),
        "model_config": PTModelConfig(
            architecture='ffno',
            fno_width=32,
            fno_blocks=4,
            fno_cnn_blocks=2,
        ),
        "training_config": PTTrainingConfig(),
        "inference_config": PTInferenceConfig(),
    }

    model = gen.build_model(pt_configs)
    assert isinstance(model, PtychoPINN_Lightning)
    assert model.model.generator.__class__.__name__ == "FfnoGeneratorModule"


def test_ffno_generator_builds_supervised_lightning_model():
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig as PTInferenceConfig,
        ModelConfig as PTModelConfig,
        TrainingConfig as PTTrainingConfig,
    )
    from ptycho_torch.model import PtychoPINN_Lightning, Ptycho_Supervised

    cfg = TrainingConfig(model=ModelConfig(architecture='ffno', model_type='supervised', N=64, gridsize=1))
    gen = resolve_generator(cfg)

    pt_configs = {
        "data_config": DataConfig(N=64, C=1, grid_size=(1, 1)),
        "model_config": PTModelConfig(
            mode='Supervised',
            architecture='ffno',
            fno_width=32,
            fno_blocks=4,
            fno_cnn_blocks=2,
            generator_output_mode='real_imag',
            loss_function='MAE',
        ),
        "training_config": PTTrainingConfig(torch_loss_mode='mae'),
        "inference_config": PTInferenceConfig(),
    }

    model = gen.build_model(pt_configs)
    assert isinstance(model, PtychoPINN_Lightning)
    assert isinstance(model.model, Ptycho_Supervised)
    assert model.model.autoencoder.__class__.__name__ == "FfnoGeneratorModule"
    assert model.model.generator_output == "real_imag"
