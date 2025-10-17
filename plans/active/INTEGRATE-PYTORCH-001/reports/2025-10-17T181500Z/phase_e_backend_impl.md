# Phase E1.C Backend Selection Implementation Summary

## Context
- Initiative: INTEGRATE-PYTORCH-001 Phase E1.C
- Date: 2025-10-17
- Goal: Add `backend` field to configuration dataclasses to enable Ptychodus to select PyTorch vs TensorFlow workflows
- Tasks completed: E1.C1 (dataclass schema), E1.C2 (PyTorch adapter propagation)
- Tasks deferred: E1.C3 (dispatcher implementation), E1.C4 (logging/instrumentation) - require Ptychodus integration

## Implementation Summary

### E1.C1: Extended Dataclass Schema ✅
**File:** `ptycho/config/config.py`

Added `backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'` field to both:
- `TrainingConfig` (line 110)
- `InferenceConfig` (line 142)

**Design Decisions:**
1. Used `Literal` type annotation for type safety
2. Defaulted to `'tensorflow'` to preserve backward compatibility
3. Field automatically propagates through `dataclass_to_legacy_dict()` to `params.cfg['backend']`
4. YAML-compatible (no custom validators needed)

**Backward Compatibility:** All existing code continues to work unchanged. New `backend` field is optional and defaults to TensorFlow.

### E1.C2: PyTorch Config Bridge Propagation ✅
**File:** `ptycho_torch/config_bridge.py`

Updated translation functions to set `backend='pytorch'` on emitted TensorFlow dataclasses:
- `to_training_config()` line 239: Added `'backend': 'pytorch'` to kwargs
- `to_inference_config()` line 346: Added `'backend': 'pytorch'` to kwargs

**Design Decisions:**
1. Backend selection happens at translation layer, not via override mechanism
2. Maintains torch-optional imports (no new dependencies)
3. Allows explicit override if needed (via `overrides` dict)
4. Config bridge now serves as single point of backend selection for PyTorch workflows

**Integration:** When PyTorch configs translate to TensorFlow dataclasses, they automatically mark themselves with `backend='pytorch'`, enabling downstream dispatcher logic (Phase E1.C3).

## Test Results

### Targeted Tests (Backend Selection)
**Command:** `pytest tests/torch/test_backend_selection.py -vv`
**Results:** 4 passed, 2 xfailed (3.37s)

**Passing Tests:**
1. ✅ `test_defaults_to_tensorflow_backend` - Validates default behavior (backward compatibility)
2. ✅ `test_selects_pytorch_backend` - Validates explicit PyTorch selection via `backend='pytorch'`
3. ✅ `test_inference_config_supports_backend_selection` - Validates InferenceConfig parity
4. ✅ `test_backend_selection_preserves_api_parity` - Validates identical signatures between backends

**Deferred Tests (E1.C3 scope):**
1. ⏸️ `test_pytorch_backend_calls_update_legacy_dict` - Requires dispatcher implementation
2. ⏸️ `test_pytorch_unavailable_raises_error` - Requires import guard in dispatcher

### Full Regression Suite
**Command:** `pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -x --tb=no -q`
**Results:** 201 passed, 13 skipped, 3 xfailed (213.00s)

**Regression Check:** ✅ PASSED
- No new failures introduced
- All pre-existing tests continue to pass
- Skipped/xfailed tests are expected (pre-existing conditions)

## Spec Compliance

### specs/ptychodus_api_spec.md §5 Alignment
- ✅ §5.1: `ModelConfig` fields preserved (no changes required)
- ✅ §5.2: `TrainingConfig` extended with `backend` field
- ✅ §5.3: `InferenceConfig` extended with `backend` field

### CONFIG-001 Finding (docs/findings.md)
- ✅ Config bridge continues to enable `update_legacy_dict()` call
- ✅ `backend` field propagates to `params.cfg['backend']` for visibility
- ⏸️ Dispatcher-level CONFIG-001 gate deferred to E1.C3

## Code Changes

### Summary
1. **ptycho/config/config.py** (2 additions):
   - Line 110: `backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'` in TrainingConfig
   - Line 142: `backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'` in InferenceConfig

2. **ptycho_torch/config_bridge.py** (2 additions):
   - Line 239: `'backend': 'pytorch'` in `to_training_config()` kwargs
   - Line 346: `'backend': 'pytorch'` in `to_inference_config()` kwargs

3. **tests/torch/test_backend_selection.py** (4 edits):
   - Removed `@pytest.mark.xfail` from 4 tests (tests now pass)

### Files Modified (Total: 3)
- ptycho/config/config.py
- ptycho_torch/config_bridge.py
- tests/torch/test_backend_selection.py

## Outstanding Work

### E1.C3: Dispatcher Implementation (Not in Scope for This Loop)
**Requirement:** Implement backend selection dispatcher in Ptychodus reconstructor or ptycho workflows layer
**Blockers:** Requires Ptychodus integration planning (Phase E2)
**Scope:** Import guard, CONFIG-001 gate, workflow module selection logic

### E1.C4: Logging & Instrumentation (Not in Scope for This Loop)
**Requirement:** Add `logger.info("Using %s backend", config.backend)` and `results['backend']` metadata
**Blockers:** Requires E1.C3 dispatcher implementation first
**Scope:** Backend logging at entry points, results dict instrumentation

## Risks & Mitigation

### Risk 1: YAML Parsing
- **Risk:** YAML configs lacking `backend` key might fail parsing
- **Mitigation:** ✅ Default value `'tensorflow'` prevents KeyError
- **Validation:** Tested with minimal config (no `backend` key) - PASSED

### Risk 2: Legacy Module Compatibility
- **Risk:** Legacy modules accessing `params.cfg['backend']` might break
- **Mitigation:** ✅ Field propagates to `params.cfg` but is ignored by legacy code
- **Validation:** Full regression passed with no new failures

### Risk 3: Type Safety
- **Risk:** String typos like `backend='pytorh'` might bypass validation
- **Mitigation:** ✅ `Literal` type annotation provides IDE/mypy checking
- **Future:** Could add runtime validation in `__post_init__` if needed

## Next Steps

### Immediate (This Initiative)
1. ✅ E1.C1-C1.C2 complete (this loop)
2. ⏸️ E1.C3: Dispatcher implementation (requires Phase E2 coordination)
3. ⏸️ E1.C4: Logging instrumentation (follows E1.C3)

### Coordination (Cross-Initiative)
1. **TEST-PYTORCH-001:** Share backend selection test patterns
2. **Phase E2:** Integration regression harness (subprocess-style tests)
3. **Ptychodus:** Reconstructor library integration planning

## Exit Criteria Status

### Phase E1.C Completion Checklist
- [x] E1.C1: Backend field added to TrainingConfig and InferenceConfig
- [x] E1.C2: PyTorch config bridge propagates `backend='pytorch'`
- [ ] E1.C3: Dispatcher implemented in orchestration layer (DEFERRED - see notes)
- [ ] E1.C4: Logging and results metadata (DEFERRED - depends on E1.C3)

### Partial Completion Rationale
**Tasks E1.C1 and E1.C2 are self-contained and can be delivered independently.**
- Configuration layer changes are complete and tested
- No dispatcher implementation needed yet (Ptychodus integration pending)
- Allows downstream initiatives to begin planning against stable config schema
- Minimal risk: backend selection is passive until E1.C3 dispatcher activates it

## Artifacts

### Test Logs
- `pytest_backend_selection.log` - Targeted backend selection test output (4 passed, 2 xfailed)

### Documentation References
- Blueprint: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T180500Z/phase_e_backend_design.md`
- Callchain: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/`
- Red tests: `tests/torch/test_backend_selection.py` (6 tests total, 4 passing, 2 deferred)

### Commit Details
**Branch:** feature/torchapi
**Scope:** Phase E1.C1+E1.C2 backend configuration
**Files:** 3 modified (config.py, config_bridge.py, test_backend_selection.py)
**Tests:** 201 passed, 13 skipped, 3 xfailed (full regression ✅)

---

**Status:** Phase E1.C partially complete (E1.C1+E1.C2 delivered; E1.C3+E1.C4 deferred to Phase E2 coordination)
**Quality Gate:** ✅ All tests passing, no regressions, backward compatibility preserved
**Handoff:** Ready for Phase E2 planning (integration regression + Ptychodus coordination)
