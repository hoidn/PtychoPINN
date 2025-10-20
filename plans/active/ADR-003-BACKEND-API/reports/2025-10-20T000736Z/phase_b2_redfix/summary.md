# ADR-003 Phase B2 RED Baseline — Factory Tests Failing on NotImplementedError

**Loop Timestamp:** 2025-10-20T000736Z
**Phase:** B2.a+B2.b (TDD RED)
**Objective:** Establish genuine RED baseline for config factory tests by removing `pytest.raises` guards and demonstrating tests fail on NotImplementedError from factory stubs.
**Mode:** TDD RED Phase
**Plan Reference:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md` §B2

---

## Executive Summary

Successfully converted factory tests from supervisor-approved FALSE GREEN state (19 tests passing with `pytest.raises` guards) to genuine TDD RED state (19 tests failing with NotImplementedError). All test assertions are now active and will validate factory behavior once implementation lands in Phase B3.

**Key Outcomes:**
- ✓ Removed all `pytest.raises(NotImplementedError)` guards from `tests/torch/test_config_factory.py`
- ✓ Uncommented and activated 56 assertions across 19 tests
- ✓ Captured RED failure log with clear NotImplementedError tracebacks
- ✓ Verified factory stubs remain untouched (keep raising NotImplementedError)
- ✓ Established baseline for GREEN phase validation

---

## Baseline Test Results

**Selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv
```

**Result:** 19 failed, 0 passed
**Runtime:** ~2.1 seconds
**Log Size:** 1,794 lines
**Log Path:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/pytest_factory_redfix.log`

### Failure Signature

All 19 tests fail with the same NotImplementedError pattern:

```python
NotImplementedError: create_training_payload() is a Phase B2 RED scaffold.
Implementation pending in Phase B3.a per
plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md
```

**Representative Stack Trace (from test_training_payload_returns_dataclass):**
```
tests/torch/test_config_factory.py:156:
    payload = create_training_payload(
        train_data_file=mock_train_npz,
        output_dir=temp_output_dir,
        overrides={'n_groups': 512, 'batch_size': 4},
    )
ptycho_torch/config_factory.py:177: NotImplementedError
```

### Test Coverage Breakdown

**Test Category 1: Training Payload Structure (4 tests)**
- `test_training_payload_returns_dataclass` — FAILED
- `test_training_payload_contains_tf_config` — FAILED
- `test_training_payload_contains_pytorch_configs` — FAILED
- `test_training_payload_contains_overrides_dict` — FAILED

**Test Category 2: Inference Payload Structure (3 tests)**
- `test_inference_payload_returns_dataclass` — FAILED
- `test_inference_payload_contains_tf_config` — FAILED
- `test_inference_payload_contains_pytorch_configs` — FAILED

**Test Category 3: Config Bridge Integration (3 tests)**
- `test_grid_size_tuple_to_gridsize_int` — FAILED
- `test_epochs_to_nepochs_conversion` — FAILED
- `test_k_to_neighbor_count_conversion` — FAILED

**Test Category 4: params.cfg Population (2 tests)**
- `test_factory_populates_params_cfg` — FAILED
- `test_populate_legacy_params_helper` — FAILED

**Test Category 5: Override Precedence (2 tests)**
- `test_override_dict_wins_over_defaults` — FAILED
- `test_probe_size_override_wins_over_inference` — FAILED

**Test Category 6: Validation Errors (3 tests)**
- `test_missing_n_groups_raises_error` — FAILED
- `test_nonexistent_train_data_file_raises_error` — FAILED
- `test_missing_checkpoint_raises_error` — FAILED

**Test Category 7: Probe Size Inference (2 tests)**
- `test_infer_probe_size_from_npz` — FAILED
- `test_infer_probe_size_missing_file_fallback` — FAILED

---

## Implementation Changes

### Modified Files

**1. tests/torch/test_config_factory.py** (~463 lines)
- Removed all 19 `with pytest.raises(NotImplementedError, match="Phase B2 RED scaffold"):` guards
- Uncommented 56 GREEN phase assertions (dataclass validation, field checks, CONFIG-001 compliance)
- Fixed indentation for active assertions (8 spaces)
- Converted all commented assertions to active code ready for GREEN validation

**Representative Change Pattern:**
```diff
# BEFORE (FALSE GREEN with guard):
def test_training_payload_returns_dataclass(self, mock_train_npz, temp_output_dir):
    """Factory returns TrainingPayload dataclass instance."""
    with pytest.raises(NotImplementedError, match="Phase B2 RED scaffold"):
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'batch_size': 4},
        )
        # GREEN phase assertions (will run after implementation):
        # assert is_dataclass(payload)
        # assert isinstance(payload, TrainingPayload)

# AFTER (TRUE RED with active assertions):
def test_training_payload_returns_dataclass(self, mock_train_npz, temp_output_dir):
    """Factory returns TrainingPayload dataclass instance."""
    payload = create_training_payload(
        train_data_file=mock_train_npz,
        output_dir=temp_output_dir,
        overrides={'n_groups': 512, 'batch_size': 4},
    )
    # GREEN phase assertions (will run after implementation):
    assert is_dataclass(payload)
    assert isinstance(payload, TrainingPayload)
```

**2. ptycho_torch/config_factory.py**
- **NO CHANGES** (stubs remain as designed per plan.md §B2.a)
- All factory functions still raise NotImplementedError with clear plan reference
- Payload dataclasses and function signatures unchanged

---

## Validation Checklist

- [x] All `pytest.raises` guards removed from test file
- [x] All GREEN phase assertions uncommented and active
- [x] Factory stubs remain unchanged (still raising NotImplementedError)
- [x] Test selector runs without syntax/import errors
- [x] All 19 tests fail with NotImplementedError (not assertion errors)
- [x] RED log captured to timestamped artifact directory
- [x] Log includes full stack traces for debugging
- [x] Artifact directory follows hygiene policy (`plans/active/.../reports/<timestamp>/`)

---

## First Divergence Analysis

**Question:** Why did tests initially pass (Attempt #6 mentioned 19 passed)?

**Answer:** Tests were passing because `pytest.raises(NotImplementedError)` guards **caught and swallowed** the NotImplementedError exceptions from factory stubs. This created a FALSE GREEN state where tests appeared successful but assertions inside the guards never executed.

**Root Cause:** Supervisor directive in `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T234600Z/phase_b2_skeleton/summary.md` flagged this discrepancy and mandated guard removal to establish genuine RED baseline before proceeding to implementation.

**Resolution:** This loop removes guards and activates assertions, converting FALSE GREEN → TRUE RED per TDD methodology.

---

## Next Steps (Phase B3 GREEN)

**Immediate Next Loop (per input.md Do Now §3):**
1. Implement `create_training_payload()` factory logic (B3.a)
   - Validate train_data_file path existence
   - Infer probe size via `infer_probe_size()` helper
   - Construct PyTorch singleton configs (DataConfig, ModelConfig, TrainingConfig)
   - Apply override precedence rules
   - Translate to TF configs via `ptycho_torch.config_bridge`
   - Call `populate_legacy_params()` (CONFIG-001 checkpoint)
   - Return TrainingPayload with audit trail

2. Implement `create_inference_payload()` factory logic (B3.a)
   - Validate model_path contains wts.h5.zip
   - Load checkpoint config or infer from NPZ
   - Construct PyTorch inference configs
   - Apply overrides and translate to TF configs
   - Return InferencePayload

3. Implement helper functions (B3.a)
   - `infer_probe_size()`: Extract N from NPZ probeGuess array
   - `populate_legacy_params()`: Wrapper around `update_legacy_dict` with logging

4. Re-run tests and capture GREEN log (B3.c)
   ```bash
   CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -vv | \
       tee plans/active/ADR-003-BACKEND-API/reports/<next-timestamp>/phase_b3_green/pytest_factory_green.log
   ```
   Expected: 19 passed, 0 failed

5. Update plan and ledger with GREEN evidence

**Outstanding Design Questions:**
- PyTorchExecutionConfig placement confirmed as Option A (`ptycho/config/config.py`)
- Override precedence matrix validated against `override_matrix.md`
- MLflow/DDP knobs deferred to Phase C (CLI integration)

---

## Artifact Inventory

```
plans/active/ADR-003-BACKEND-API/reports/2025-10-20T000736Z/phase_b2_redfix/
├── summary.md (this file)
└── pytest_factory_redfix.log (1,794 lines, RED baseline)
```

**Artifact Cross-References:**
- Plan: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md` §B2
- Implementation Checklist: `plans/active/ADR-003-BACKEND-API/implementation.md` Phase B rows
- Ledger: `docs/fix_plan.md` ADR-003-BACKEND-API Attempt #7

---

## Exit Criteria (Phase B2.b)

**All criteria satisfied:**
- ✓ Factory module skeleton exists with NotImplementedError stubs
- ✓ Test suite captures expected factory behavior with active assertions
- ✓ RED failure log demonstrates clear NotImplementedError exceptions
- ✓ Artifact paths recorded in plan + ledger
- ✓ No production code changes (tests-only loop per TDD discipline)

**Phase B2 COMPLETE** — Ready for Phase B3 GREEN implementation.

---

**Loop Completion Time:** 2025-10-20T00:12:00Z
**Next Loop Focus:** Phase B3.a factory implementation + GREEN validation
