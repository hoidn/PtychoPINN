# Phase A Implementation Notes

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** B.B5 Phase A — Enable Parity Test Harness Without PyTorch
**Timestamp:** 2025-10-17T050930Z
**Loop:** Attempt #14

---

## Phase A.A1: Skip Mechanics Audit

### Current Behavior
`tests/conftest.py` applies auto-skip logic at lines 40-43:
```python
if "torch" in str(item.fspath).lower() or item.get_closest_marker("torch"):
    if not torch_available:
        item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

This causes **all tests in `tests/torch/`** to skip when PyTorch is unavailable, including configuration bridge parity tests that should not require actual torch tensors.

### Problem Analysis
1. `ptycho_torch/config_bridge.py:72-76` has hard import requiring PyTorch:
   ```python
   try:
       from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
   except ImportError as e:
       raise ImportError(...) from e
   ```

2. `ptycho_torch/config_params.py:3` imports torch at module level:
   ```python
   import torch
   ```

3. Config bridge tests try to import the bridge module, which triggers the hard dependency.

### Decision: Implement Optional Import Pattern

**Strategy:** Make `config_params.py` importable without torch by using lazy/guarded imports.

**Approach A (Chosen):** Add optional torch import guard in `config_params.py`
- Pro: Minimal changes, preserves existing API when torch available
- Pro: Allows config bridge to provide stub classes when torch missing
- Con: Requires careful handling of torch.Tensor type annotations

**Approach B (Rejected):** Create separate `_torch_optional.py` shim module
- Pro: Clean separation of concerns
- Con: More complex import chain, harder to maintain

**Approach C (Rejected):** Move torch imports into class methods
- Pro: Truly lazy evaluation
- Con: Breaks dataclass semantics, complicates type checking

---

## Phase A.A2: Implement Optional Torch Shim

### Implementation Plan

1. **Guard torch import in config_params.py:**
   ```python
   try:
       import torch
       TORCH_AVAILABLE = True
   except ImportError:
       torch = None
       TORCH_AVAILABLE = False
   ```

2. **Handle Tensor type annotation:**
   ```python
   if TYPE_CHECKING or TORCH_AVAILABLE:
       from typing import Optional as _Optional
       TensorType = torch.Tensor if TORCH_AVAILABLE else Any
   else:
       TensorType = Any
   ```

3. **Update probe_mask field:**
   ```python
   probe_mask: Optional[TensorType] = None
   ```

### Risk Assessment
- **Risk:** Tests may fail if they actually need torch operations
- **Mitigation:** Tests should focus on dataclass translation, not torch ops
- **Verification:** Run parity tests with `TORCH_AVAILABLE=False` assertion

---

## Phase A.A3: Adjust Pytest Gating

### Current Skip Logic Issues
1. Blanket skip for all `tests/torch/` paths prevents config bridge tests from running
2. No way to distinguish "config-only" tests from "torch-ops-required" tests

### Proposed Changes

**Option 1 (Minimal, Chosen):** Make config bridge importable without torch, tests execute with stub types
- Tests validate dataclass structure and field mappings only
- Actual torch operations (if any) deferred to tests marked with `@pytest.mark.torch`

**Option 2 (Surgical):** Update conftest.py to check file content, not just path
- Skip only if test file imports `torch` directly (not via config_params)
- Con: Fragile, requires parsing imports

**Option 3 (Explicit Markers):** Require explicit `@pytest.mark.torch` for hard dependencies
- Remove path-based skip, rely only on markers
- Con: Requires updating all existing torch tests

### Implementation
Use Option 1: no conftest changes needed, just make imports optional.

---

## Phase A.A4: Fallback Verification

### Test Strategy
After implementing shim:
```bash
# Verify tests run without torch (not skip)
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -k N-direct -v
```

Expected output:
- **Before:** `SKIPPED [1] (PyTorch not available)`
- **After:** Test executes, may FAIL if logic incomplete, but should NOT SKIP

### Validation Criteria
- [ ] Tests collect without ImportError
- [ ] Test execution reaches assertion stage (not skipped)
- [ ] Informative message if fallback mode active (optional log/warning)

---

## Open Questions
1. Should config bridge raise NotImplementedError when torch unavailable, or provide stub classes?
   - **Decision:** Provide stub classes - tests validate structure, not torch ops
2. Do parity tests need actual torch tensors, or just dataclass field validation?
   - **Decision:** Field validation only, no tensor ops needed in Phase A
3. Should we warn when running in fallback mode?
   - **Decision:** Defer to Phase B if warnings needed for diagnostics

---

---

## Phase A Results

### Deliverables Completed
1. **Optional Torch Import Guard** (`ptycho_torch/config_params.py:4-13`)
   - Implemented try/except block for torch import
   - Created `TORCH_AVAILABLE` flag and `TensorType` alias
   - probe_mask field updated to use `Optional[TensorType]` instead of `Optional[torch.Tensor]`

2. **Config Bridge Import Update** (`ptycho_torch/config_bridge.py:70-78`)
   - Removed hard ImportError on torch unavailability
   - Now imports TORCH_AVAILABLE flag from config_params
   - Module can be imported successfully without torch

3. **Conftest Exemption** (`tests/conftest.py:39-46`)
   - Added exception for test_config_bridge in skip logic
   - Tests in `tests/torch/test_config_bridge.py` now execute when torch unavailable
   - Other torch tests still appropriately skipped

4. **Pytest Log Captured** (`pytest_phaseA.log`)
   - Test execution confirmed: `TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg`
   - Status: FAILED (not SKIPPED) - validates Phase A goal achieved
   - Failure reason: Minor assertion (Path vs str comparison), not import/skip issue

### Exit Criteria Validation
- [x] Tests collect without ImportError
- [x] Test execution reaches assertion stage (not skipped)
- [x] Config bridge module importable without torch (`python3 -c "from ptycho_torch import config_bridge"` succeeds)
- [x] Pytest log showing execution (not skip) captured at `pytest_phaseA.log`

### Known Issues
1. **Parametrized Tests Incompatible with unittest.TestCase:**
   - `TestConfigBridgeParity` tests use `@pytest.mark.parametrize` with `unittest.TestCase`
   - This combination fails with TypeError (missing positional arguments)
   - **Impact:** Parity tests cannot execute in current form
   - **Resolution Required:** Convert `TestConfigBridgeParity` from `unittest.TestCase` to pure pytest style (remove inheritance, use plain functions)

2. **Minor Test Assertion Failure:**
   - `params.cfg['model_path']` returns `PosixPath('model_dir')` instead of string
   - Expected behavior unclear (may be acceptable per dataclass_to_legacy_dict logic)
   - Does not block Phase A completion (tests are runnable, just not passing)

---

## Next Steps (Phase B)
After Phase A complete:
1. **Fix parametrized test compatibility:** Convert `TestConfigBridgeParity` to pytest-style (remove `unittest.TestCase` inheritance)
2. Implement probe_mask Tensor→bool conversion (P0 blocker)
3. Enforce nphotons override validation (P0 blocker)
4. Run full parity suite and capture pytest_green.log
