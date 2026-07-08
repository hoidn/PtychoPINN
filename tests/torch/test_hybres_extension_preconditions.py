# tests/torch/test_hybres_extension_preconditions.py
"""Preconditions gate for the representation/scaling comparison extension.

Pins the branch invariants that the E2/E3 comparison arms depend on: the
four ModelConfig knobs and their safe defaults, the hybrid_resnet generator
registry key, cnn_output_mode's accepted Literal values, and the presence
of the flux-sweep and harness scripts/fixtures. Pure config/registry checks
only -- no training, no GPU.
"""
import os

from ptycho.config.config import ModelConfig as PtychoModelConfig, TrainingConfig as PtychoTrainingConfig
from ptycho_torch.config_params import ModelConfig
from ptycho_torch.generators.registry import resolve_generator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_config_extension_knob_defaults():
    config = ModelConfig()
    assert config.training_patch_weighting == 'central_mask'
    assert config.physics_forward_mode == 'amplitude'
    assert config.rect_s1s2_trainable is True
    assert config.cnn_output_mode == 'amp_phase'


def test_model_config_extension_knob_literal_members_accepted():
    assert ModelConfig(training_patch_weighting='probe').training_patch_weighting == 'probe'
    assert ModelConfig(training_patch_weighting='uniform').training_patch_weighting == 'uniform'
    assert ModelConfig(physics_forward_mode='rectangular_scaled').physics_forward_mode == 'rectangular_scaled'
    assert ModelConfig(cnn_output_mode='real_imag').cnn_output_mode == 'real_imag'


def test_cnn_output_mode_accepts_both_values():
    assert ModelConfig(cnn_output_mode='amp_phase').cnn_output_mode == 'amp_phase'
    assert ModelConfig(cnn_output_mode='real_imag').cnn_output_mode == 'real_imag'


def test_hybrid_resnet_registry_key_resolves():
    cfg = PtychoTrainingConfig(model=PtychoModelConfig(architecture='hybrid_resnet'))
    generator = resolve_generator(cfg)
    assert generator.name == 'hybrid_resnet'


def test_flux_sweep_and_harness_scripts_exist():
    for relative_path in (
        'scripts/studies/make_flux_sweep.py',
        'scripts/studies/flux_sweep_eval.py',
        'scripts/studies/varpro_probe_ablation_runner.py',
    ):
        assert os.path.isfile(os.path.join(REPO_ROOT, relative_path))
    assert os.path.isdir(os.path.join(REPO_ROOT, 'tests/fixtures/varpro_parity'))
