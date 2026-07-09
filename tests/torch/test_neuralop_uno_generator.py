import importlib

import pytest
import torch

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig as PTInferenceConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
)


def _pt_configs(*, mode: str = "Unsupervised"):
    return {
        "data_config": DataConfig(N=128, C=1, grid_size=(1, 1)),
        "model_config": PTModelConfig(
            mode=mode,
            architecture="neuralop_uno",
            generator_output_mode="real_imag",
            loss_function="MAE" if mode == "Supervised" else "Poisson",
            probe_mask=False,
            object_big=False,
            probe_big=False,
        ),
        "training_config": PTTrainingConfig(
            torch_loss_mode="mae" if mode == "Supervised" else "poisson",
        ),
        "inference_config": PTInferenceConfig(),
    }


def test_neuralop_uno_adapter_matches_locked_real_imag_shape():
    from ptycho_torch.generators.neuralop_uno import NeuralopUnoGeneratorModule

    module = NeuralopUnoGeneratorModule(C=1, output_mode="real_imag")
    x = torch.randn(2, 1, 128, 128)

    out = module(x)

    assert out.shape == (2, 128, 128, 1, 2)
    assert out.dtype == x.dtype


def test_neuralop_uno_rejects_multi_channel_grouping():
    from ptycho_torch.generators.neuralop_uno import NeuralopUnoGeneratorModule

    with pytest.raises(ValueError, match="C=1"):
        NeuralopUnoGeneratorModule(C=2, output_mode="real_imag")


def test_neuralop_uno_rejects_non_real_imag_output_modes():
    from ptycho_torch.generators.neuralop_uno import NeuralopUnoGeneratorModule

    with pytest.raises(ValueError, match="real_imag"):
        NeuralopUnoGeneratorModule(C=1, output_mode="amp_phase")


def test_neuralop_uno_missing_dependency_raises_actionable_error(monkeypatch):
    from ptycho_torch.generators import neuralop_uno

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "neuralop.models":
            raise ImportError("missing neuralop")
        return real_import_module(name, package)

    monkeypatch.setattr(neuralop_uno.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="neuraloperator==2.0.0"):
        neuralop_uno.NeuralopUnoGeneratorModule(C=1, output_mode="real_imag")


def test_neuralop_uno_incompatible_dependency_raises_actionable_error(monkeypatch):
    from ptycho_torch.generators import neuralop_uno

    class MissingUnoModule:
        pass

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "neuralop.models":
            return MissingUnoModule()
        return real_import_module(name, package)

    monkeypatch.setattr(neuralop_uno.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="neuralop.models.UNO"):
        neuralop_uno.NeuralopUnoGeneratorModule(C=1, output_mode="real_imag")


def test_neuralop_uno_builds_unsupervised_lightning_with_uno_body():
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.generators.neuralop_uno import NeuralopUnoGenerator
    from ptycho_torch.model import PtychoPINN, PtychoPINN_Lightning

    cfg = TrainingConfig(model=ModelConfig(architecture="neuralop_uno", N=128, gridsize=1))
    generator = NeuralopUnoGenerator(cfg)

    model = generator.build_model(_pt_configs(mode="Unsupervised"))

    assert isinstance(model, PtychoPINN_Lightning)
    assert isinstance(model.model, PtychoPINN)
    assert type(model.model.autoencoder).__name__ == "NeuralopUnoGeneratorModule"
    assert model.model.generator_output == "real_imag"


def test_neuralop_uno_builds_supervised_lightning_with_same_uno_body():
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.generators.neuralop_uno import NeuralopUnoGenerator
    from ptycho_torch.model import PtychoPINN_Lightning, Ptycho_Supervised

    cfg = TrainingConfig(
        model=ModelConfig(
            architecture="neuralop_uno",
            model_type="supervised",
            N=128,
            gridsize=1,
        )
    )
    generator = NeuralopUnoGenerator(cfg)

    model = generator.build_model(_pt_configs(mode="Supervised"))

    assert isinstance(model, PtychoPINN_Lightning)
    assert isinstance(model.model, Ptycho_Supervised)
    assert type(model.model.autoencoder).__name__ == "NeuralopUnoGeneratorModule"
    assert model.model.generator_output == "real_imag"
