# Phase C2 Factory Wiring Summary — PyTorchExecutionConfig Integration (2025-10-20T010900Z)

## Objective
Wire PyTorchExecutionConfig into factory payload dataclasses, implement default instantiation + override merging, and extend tests to verify execution knob propagation.

## Scope (Per Phase C Execution Plan)
- **C2.B1**: Update `TrainingPayload`/`InferencePayload` `execution_config` field type from `Any` to `PyTorchExecutionConfig`
- **C2.B2**: Merge overrides into execution config, record applied knobs in `overrides_applied` audit trail
- **C2.B3**: Extend factory tests with `TestExecutionConfigOverrides` class (RED→GREEN cycle)
- **C2.B4**: Document override precedence decisions + field coverage

## Implementation Summary

### C2.B1: Payload Dataclass Type Updates

**Files Modified:**
- `ptycho_torch/config_factory.py:83,100` — Updated type hints from `Any` to `PyTorchExecutionConfig`

**Changes:**
```python
# Before (Phase B3):
execution_config: Any  # PyTorchExecutionConfig (type: ignore during RED phase)

# After (Phase C2):
execution_config: PyTorchExecutionConfig  # Execution knobs (Phase C2)
```

**Impact:**
- Type safety enforced: factories now require PyTorchExecutionConfig instances
- IDE autocomplete + type checkers can validate execution knobs
- No breaking change: existing call sites with `execution_config=None` handled by default instantiation logic

### C2.B2: Execution Config Instantiation + Override Merging

**Files Modified:**
- `ptycho_torch/config_factory.py:258-269` (training payload)
- `ptycho_torch/config_factory.py:430-439` (inference payload)

**Implementation Logic:**
1. **Default Instantiation**: If `execution_config=None`, create default `PyTorchExecutionConfig()` with CPU-safe defaults
2. **Override Merging**: Record applied execution knobs in `overrides_applied` audit trail for transparency
3. **Priority Enforcement**: Explicit `execution_config` parameter (if provided) takes precedence over defaults

**Training Factory Knobs Recorded:**
- `accelerator` (default: 'cpu')
- `deterministic` (default: True)
- `num_workers` (default: 0)
- `enable_progress_bar` (default: False)
- `learning_rate` (default: 1e-3)

**Inference Factory Knobs Recorded:**
- `accelerator` (default: 'cpu')
- `num_workers` (default: 0)
- `inference_batch_size` (default: None)

**Design Rationale:**
- **Priority Level 2 Placement**: Execution config sits between explicit overrides (Priority 1) and CLI defaults (Priority 3) per `override_matrix.md` §1
- **Audit Trail Transparency**: Recording applied knobs in `overrides_applied` enables debugging and governance
- **CPU-Safe Defaults**: Default values ensure reproducibility and work on CPU-only environments (per POLICY-001)

### C2.B3: Factory Tests Extension (RED→GREEN Cycle)

**Files Modified:**
- `tests/torch/test_config_factory.py:425-542` — Added `TestExecutionConfigOverrides` class (6 new tests)

**Test Coverage:**

| Test | Assertion | Phase |
|------|-----------|-------|
| `test_training_payload_execution_config_not_none` | Returns `PyTorchExecutionConfig` instance (not None) | ✅ GREEN |
| `test_inference_payload_execution_config_not_none` | Returns `PyTorchExecutionConfig` instance | ✅ GREEN |
| `test_execution_config_defaults_applied` | Defaults match `design_delta.md` (cpu, deterministic=True, num_workers=0) | ✅ GREEN |
| `test_execution_config_explicit_instance_propagates` | User-provided execution_config propagates through payload | ✅ GREEN |
| `test_execution_config_fields_accessible` | Critical fields accessible (accelerator, deterministic, num_workers, etc.) | ✅ GREEN |
| `test_overrides_applied_records_execution_knobs` | Audit trail captures execution knobs | ✅ GREEN |

**RED Phase Evidence:**
- Selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv`
- Log: `pytest_factory_execution_red.log` (6 FAILED on ImportError + None assertion)
- Timestamp: 2025-10-20T010920Z

**GREEN Phase Evidence:**
- Selector: Same as RED
- Log: `pytest_factory_execution_green.log` (6 PASSED, 11 warnings)
- Timestamp: 2025-10-20T024540Z

### C2.B4: Override Precedence Documentation

**Override Precedence Rules (Per override_matrix.md §1):**

1. **Explicit Overrides** (highest priority)
   - Example: `overrides={'n_groups': 1024}` → always wins
   - Applies to canonical config fields (N, gridsize, batch_size, etc.)

2. **Execution Config Fields** (Priority Level 2 — Phase C2 scope)
   - Example: `execution_config=PyTorchExecutionConfig(accelerator='gpu')` → overrides default 'cpu'
   - Applies to execution-only knobs (accelerator, deterministic, num_workers, learning_rate, etc.)
   - Does NOT apply to canonical config fields (N, gridsize, etc.)

3. **CLI Defaults** (Priority Level 3)
   - Example: `--batch_size 4` (argparse default)
   - Overridden by Priority 1 (explicit overrides) or Priority 2 (execution config)

4. **PyTorch Config Defaults** (Priority Level 4)
   - Example: `DataConfig(nphotons=1e5)`, `TrainingConfig(epochs=50)`
   - Used when no higher-priority source specifies value

5. **TensorFlow Config Defaults** (lowest priority)
   - Example: `TrainingConfig(nphotons=1e9)`
   - Canonical source of truth; used for bridge translation

**Conflict Resolution Examples:**

**Example 1: execution_config wins over CLI default**
```python
# CLI: --batch_size not specified (default=4)
# Execution config: (no batch_size field — execution config is execution-only)
# Result: CLI default wins (batch_size=4)
# Note: batch_size is NOT an execution knob; it's a canonical training parameter
```

**Example 2: Explicit override wins over execution_config**
```python
# Overrides dict: {'n_groups': 1024}
# Execution config: PyTorchExecutionConfig(accelerator='gpu')
# Result: Both apply (different priority domains)
# - n_groups=1024 (Priority 1, canonical config)
# - accelerator='gpu' (Priority 2, execution config)
```

**Example 3: execution_config default instantiation**
```python
# Factory call: create_training_payload(..., execution_config=None)
# Result: PyTorchExecutionConfig() instantiated with defaults
# - accelerator='cpu' (CPU-safe)
# - deterministic=True (reproducibility)
# - num_workers=0 (main process only)
```

**Field Coverage (Phase C2):**

| Field Category | Coverage | Notes |
|----------------|----------|-------|
| Lightning Trainer | ✅ Partial | accelerator, deterministic (strategy, gradient_clip_val deferred to Phase C3) |
| DataLoader | ✅ Partial | num_workers (pin_memory, persistent_workers, prefetch_factor not yet consumed) |
| Optimization | ✅ Partial | learning_rate (scheduler, accum_steps deferred to Phase C3) |
| Checkpoint/Logging | ✅ Partial | enable_progress_bar (checkpoint_save_top_k, logger_backend deferred to Phase C3/D) |
| Inference | ✅ Partial | inference_batch_size (middle_trim, pad_eval not yet implemented) |

## Test Results

### Targeted ExecutionConfig Tests
- **Selector**: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestExecutionConfigOverrides -vv`
- **Result**: 6 passed, 11 warnings in 3.67s
- **Log**: `pytest_factory_execution_green.log`

### Full Test Suite
- **Selector**: `pytest tests/ -v`
- **Result**: 268 passed, 17 skipped, 1 xfailed, 57 warnings in 236.71s (0:03:56)
- **Log**: `pytest_full_suite.log`
- **Regression**: None — all existing tests remain GREEN

## Artifacts
- `pytest_factory_execution_red.log` — RED phase (6 FAILED on ImportError)
- `pytest_factory_execution_green.log` — GREEN phase (6 PASSED)
- `pytest_full_suite.log` — Full regression validation (268 PASSED)
- `summary.md` — This document

## Exit Criteria Validation (Per Phase C Execution Plan §C2)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Payload dataclasses use PyTorchExecutionConfig type | ✅ Complete | `config_factory.py:83,100` |
| Factory instantiates default execution config | ✅ Complete | `config_factory.py:260-261,432-433` |
| Override merging implemented | ✅ Complete | `config_factory.py:263-269,435-439` |
| overrides_applied audit trail captures execution knobs | ✅ Complete | Tests verify field presence |
| TestExecutionConfigOverrides class added with RED→GREEN cycle | ✅ Complete | 6/6 tests GREEN |
| Override precedence documented | ✅ Complete | This summary §C2.B4 |

## Phase C2 Completion Status
**✅ COMPLETE** — All exit criteria satisfied. Ready for Phase C3 workflow integration.

## Next Phase Handoff (Phase C3)

**Unfinished Business from C2:**
- Lightning Trainer knobs not yet consumed: strategy, gradient_clip_val, accum_steps
- DataLoader knobs not yet consumed: pin_memory, persistent_workers, prefetch_factor
- Checkpoint knobs not yet consumed: checkpoint_save_top_k, checkpoint_monitor_metric, early_stop_patience
- Logger knobs deferred to Phase D: logger_backend

**Phase C3 Entry Conditions:**
- Factories return PyTorchExecutionConfig instances ✅
- Tests assert execution config fields accessible ✅
- Override precedence documented ✅

**Phase C3 Scope (Per Plan §C3):**
- C3.C1: Update `_train_with_lightning` signature to accept `execution_config` parameter
- C3.C2: Thread execution knobs into Lightning Trainer (accelerator, strategy, deterministic, gradient_clip_val)
- C3.C3: Extend `tests/torch/test_workflows_components.py` to assert Trainer kwargs
- C3.C4: Document runtime evidence + integration gaps

**Open Questions for C3:**
- Should Lightning strategy='auto' be overridden in execution_config, or left as-is?
- How to validate accelerator='gpu' in CPU-only CI environments (skip tests or mock Trainer)?
- Defer MLflow logger_backend to Phase D or prototype in C3?

## References
- Phase C Execution Plan: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md`
- Override Matrix: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §5
- Factory Design: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md` §2.2
