# Modular Generator Implementation - Test Summary

## Date: 2026-01-26

## Test Results

### Core Tests (10 passed, 1 skipped)

| Test | Status |
|------|--------|
| test_model_config_architecture_default_ok | PASSED |
| test_model_config_architecture_invalid_raises | PASSED |
| test_resolve_generator_cnn (TF) | PASSED |
| test_resolve_generator_unknown_raises (TF) | PASSED |
| test_resolve_generator_cnn (Torch) | PASSED |
| test_resolve_generator_unknown_raises (Torch) | PASSED |
| test_train_cdi_model_calls_resolve_generator | PASSED |
| test_train_with_lightning_calls_resolve_generator | SKIPPED (lightning not installed) |
| test_config_bridge_architecture_override | PASSED |
| test_config_bridge_architecture_default | PASSED |
| test_config_bridge_architecture_from_pt_model | PASSED |

### Config Bridge Tests (51 passed)
All existing config bridge tests continue to pass.

## Implementation Summary

### Files Created
- `ptycho/generators/__init__.py`
- `ptycho/generators/registry.py`
- `ptycho/generators/cnn.py`
- `ptycho/generators/README.md`
- `ptycho_torch/generators/__init__.py`
- `ptycho_torch/generators/registry.py`
- `ptycho_torch/generators/cnn.py`
- `ptycho_torch/generators/README.md`
- `tests/test_model_config_architecture.py`
- `tests/test_generator_registry.py`
- `tests/torch/test_generator_registry.py`
- `tests/test_workflow_generator_integration.py`

### Files Modified
- `ptycho/config/config.py` - Added architecture field to ModelConfig
- `ptycho_torch/config_params.py` - Added architecture field to ModelConfig
- `ptycho_torch/config_bridge.py` - Pass architecture through bridge
- `ptycho_torch/config_factory.py` - Include architecture in factory
- `ptycho/workflows/components.py` - Wire generator registry
- `ptycho/train_pinn.py` - Accept model_instance parameter
- `ptycho_torch/workflows/components.py` - Wire generator registry
- `docs/CONFIGURATION.md` - Document architecture field
- `docs/workflows/pytorch.md` - Document architecture in config
- `docs/specs/spec-ptycho-config-bridge.md` - Add mapping rule

## Commits
1. feat: add model.architecture config field
2. feat: bridge model.architecture through torch config
3. feat: add generator registry with cnn implementation
4. feat: wire generator registry into workflow entry points
5. docs: add generator README guides for TF and PyTorch
