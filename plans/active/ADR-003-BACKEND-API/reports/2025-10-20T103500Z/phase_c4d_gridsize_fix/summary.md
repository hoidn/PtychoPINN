# Phase C4.D Gridsize Channel Parity Fix — Loop Summary

## Problem Statement

Quoted from `specs/ptychodus_api_spec.md` and `docs/findings.md#BUG-TF-001`:
> When `gridsize > 1`, the PyTorch backend must ensure channel count `C = gridsize**2` propagates
> correctly from configuration to model architecture. Failure to synchronize causes shape mismatches
> during forward pass when grouped patches expect N channels but model layers are configured for 1 channel.

The `_train_with_lightning` function manually constructed PyTorch config objects without computing
`C = gridsize**2` or setting `C_model` and `C_forward`, causing Lightning models to use default
`C=1` regardless of requested `gridsize`.

## Architecture Alignment (ADR-003)

Quoted from `docs/architecture.md` ADR-003-BACKEND-API:
> **Factory Pattern for Config Construction (§3.1)**: All PyTorch config objects MUST be created
> via `config_factory.create_training_payload()` to ensure consistent field derivation and
> channel count propagation per `grid_size → C = grid_size[0] * grid_size[1]`.

The fix refactors `_train_with_lightning` (ptycho_torch/workflows/components.py:610-658) to delegate
config construction to the factory, eliminating manual PTDataConfig/PTModelConfig instantiation.

## Search Summary (Existing vs. Missing)

**Existing**:
- `ptycho_torch/config_factory.py:196-223` — Factory computes `C = grid_size[0] * grid_size[1]` and
  sets `pt_model_config.C_model` and `pt_model_config.C_forward` correctly.
- `ptycho_torch/model.py:847-887` — `PtychoPINN` first conv layer input channels determined by
  `model_config.C_model` passed to `Autoencoder` constructor.

**Missing (before this loop)**:
- No regression test validating that `_train_with_lightning` respects `config.model.gridsize`
  when instantiating Lightning modules.
- `_train_with_lightning` manually built PyTorch configs instead of reusing factory logic.

## Implementation Changes

### 1. New Test (TDD Red Phase)
**File**: `tests/torch/test_workflows_components.py:568-696`
**Method**: `TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize`

Test design:
- Create `TrainingConfig` with `gridsize=2` (requires 4 input channels: 2×2 grouping).
- Monkeypatch `PtychoPINN_Lightning` to spy on instantiated module's first conv layer.
- Invoke `_train_with_lightning` and assert `first_conv.in_channels == 4` (not default 1).

RED baseline captured: `pytest_gridsize_red.log` shows `in_channels=1` when `gridsize=2` → assertion fails.

### 2. Production Fix (Green Phase)
**File**: `ptycho_torch/workflows/components.py:610-658`

Replaced manual config construction:
```python
# OLD (lines 620-631 before fix):
pt_data_config = PTDataConfig(
    N=config.model.N,
    grid_size=(config.model.gridsize, config.model.gridsize),
    # Missing: C computation
)
pt_model_config = PTModelConfig(
    mode='Unsupervised',
    # Missing: C_model, C_forward
)
```

With factory-driven construction:
```python
# NEW (lines 619-650 after fix):
from ptycho_torch.config_factory import create_training_payload

mode_map = {'pinn': 'Unsupervised', 'supervised': 'Supervised'}
factory_overrides = {
    'n_groups': config.n_groups,
    'gridsize': config.model.gridsize,
    'model_type': mode_map.get(config.model.model_type, 'Unsupervised'),
    # ... other fields from TrainingConfig
}

payload = create_training_payload(
    train_data_file=config.train_data_file,
    output_dir=getattr(config, 'output_dir', Path('./outputs')),
    execution_config=execution_config,
    overrides=factory_overrides
)

pt_data_config = payload.pt_data_config  # C = 4 when gridsize=2
pt_model_config = payload.pt_model_config  # C_model=4, C_forward=4
pt_training_config = payload.pt_training_config
pt_inference_config = PTInferenceConfig()  # Not included in TrainingPayload
```

**Critical**: Factory ensures `C = gridsize**2` propagation per config_factory.py:198-223.

### 3. Test Execution Results

**Targeted test (RED → GREEN)**:
```bash
CUDA_VISIBLE_DEVICES="" pytest \
  tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_training_respects_gridsize \
  -vv
```

- **RED log**: `pytest_gridsize_red.log` — AssertionError: `in_channels=1 != expected 4`
- **GREEN log**: `pytest_gridsize_green_final.log` — PASSED (1 passed, 4 warnings)

**Full suite**: Running in background (shell_id 544714), results will be in `pytest_full_suite.log`.

## Artifacts

All artifacts stored under:
`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103500Z/phase_c4d_gridsize_fix/`

- `pytest_gridsize_red.log` — RED phase baseline (in_channels=1 mismatch)
- `pytest_gridsize_green_final.log` — GREEN phase validation (PASSED)
- `pytest_full_suite.log` — Full regression test suite (running)
- `summary.md` — This document

## Next Actions

1. **Full suite validation**: Once `pytest_full_suite.log` completes, verify no regressions introduced.
2. **Manual CLI smoke test**: Execute Phase C4.D B3 manual smoke with `gridsize=2`:
   ```bash
   CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train \
     --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
     --output_dir /tmp/cli_smoke_gridsize2 \
     --n_images 64 --max_epochs 1 --accelerator cpu --deterministic \
     --num-workers 0 --gridsize 2
   ```
3. **Update `docs/fix_plan.md`**: Log Attempt #N with artifact paths and GREEN status.
4. **Commit changes**: Include test + production code + documentation updates.

## Metrics

- **Lines changed**: ~50 (test) + ~40 (production refactor)
- **Test runtime**: 5.00s (targeted), ~XXXs (full suite pending)
- **Channel count verified**: `gridsize=2 → C=4` correctly propagated to PyTorch model layers

## References

- Spec: `specs/ptychodus_api_spec.md` §4.6–§4.8 (config contract)
- Findings: `docs/findings.md#BUG-TF-001` (gridsize mismatch pattern)
- Architecture: `docs/architecture.md` ADR-003 §3.1 (factory pattern mandate)
- Plan: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md` Phase B
