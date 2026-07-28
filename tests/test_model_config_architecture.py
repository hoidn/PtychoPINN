# tests/test_model_config_architecture.py
import pytest
from ptycho.config import validate_model_config_structure
from ptycho.config.config import ModelConfig, validate_model_config


def test_model_config_architecture_default_ok():
    cfg = ModelConfig()
    validate_model_config(cfg)


def test_model_config_architecture_invalid_raises():
    cfg = ModelConfig(architecture="not-a-real-arch")
    with pytest.raises(ValueError):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    "architecture",
    ["cnn", "ffno", "fno", "fno_vanilla", "neuralop_uno"],
)
def test_structural_model_accepts_exact_five_architectures(architecture):
    validate_model_config_structure(ModelConfig(architecture=architecture))


@pytest.mark.parametrize(
    "amp_activation",
    ["sigmoid", "swish", "softplus", "relu"],
)
def test_structural_model_accepts_public_activation_spellings(
    amp_activation,
):
    validate_model_config_structure(ModelConfig(amp_activation=amp_activation))


@pytest.mark.parametrize(
    "updates",
    [
        {
            "object_layout": "single_patch",
            "training_canvas": "independent",
            "training_patch_weighting": "central_mask",
            "object_big": False,
        },
        {
            "object_layout": "grouped_patches",
            "training_canvas": "relative_overlap",
            "training_patch_weighting": "probe",
            "object_big": True,
        },
    ],
)
def test_structural_model_accepts_coherent_object_policy_joins(updates):
    validate_model_config_structure(ModelConfig(**updates))


def test_structural_model_rejects_incoherent_object_policy_join():
    with pytest.raises(ValueError, match="object_layout|object_big"):
        validate_model_config_structure(
            ModelConfig(
                object_big=False,
                object_layout="grouped_patches",
                training_canvas="relative_overlap",
                training_patch_weighting="probe",
            )
        )
