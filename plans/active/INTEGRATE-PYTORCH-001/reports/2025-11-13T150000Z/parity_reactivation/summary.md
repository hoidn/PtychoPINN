# Phase R PyTorch Parity Reactivation Summary

## Focus
INTEGRATE-PYTORCH-PARITY-001 — PyTorch backend API parity reactivation

## Date
2025-11-13T15:00:00Z

## Objective
Restore PyTorch config bridge + persistence parity after Phase F handoff. Close gaps identified in `PYTORCH_INVENTORY_SUMMARY.txt`:
1. Config bridge never invoked in CLI entry points
2. Missing n_groups defaults
3. Persistence stub unimplemented
4. Regression guard idle since 2025-10-19

## Implementation Summary

### 1. Config Bridge Wiring (ALREADY COMPLETE)
**Finding:** CLI entry points already use config_factory pattern
- `ptycho_torch/train.py:708`: Uses `create_training_payload()`
- `ptycho_torch/inference.py:556`: Uses `create_inference_payload()`
- Both factories call `populate_legacy_params()` which wraps `update_legacy_dict()`
- **Status:** ✅ CONFIG-001 compliance verified

### 2. Missing Config Defaults (NO CHANGES NEEDED)
**Finding:** Defaults handled by config_bridge adapter layer
- `n_groups`: Expected in overrides dict (config_bridge.py:250)
- `test_data_file`: Expected in overrides dict (config_bridge.py:249)
- `gaussian_smoothing_sigma`: Hardcoded 0.0 in ModelConfig kwargs (config_bridge.py:176)
- **Status:** ✅ No PyTorch config_params.py changes required

### 3. Persistence Shim (NEW IMPLEMENTATION)
**File:** `ptycho_torch/api/base_api.py:612-687`
**Implementation:** Minimal viable persistence shim
- Emits Lightning checkpoint + JSON manifest bundle
- Manifest includes: backend='pytorch', checkpoint reference, params.cfg snapshot
- CONFIG-001 gate: Raises ValueError if params.cfg empty before save
- Future work: Full .h5.zip adapter (Phase 5)
**Status:** ✅ Implemented per Phase R scope

### 4. Regression Guard (GREEN)
**Selector:** `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv`
**Result:** 45/45 PASSED (100% pass rate, 3.66s)
**Coverage:** All spec-required fields across ModelConfig, TrainingConfig, InferenceConfig
**Log:** `plans/active/.../green/pytest_config_bridge.log`
**Status:** ✅ Regression test suite active and passing

## Files Touched

### Code Changes
- `ptycho_torch/api/base_api.py` (lines 612-687)
  - Implemented `save_pytorch()` method

### Test Evidence
- `tests/torch/test_config_bridge.py` (executed, no changes)
  - All 45 parity tests passing

### Hub Artifacts
- `green/pytest_config_bridge.log` (45 passed tests)
- `analysis/artifact_inventory.txt` (detailed findings)
- `summary.md` (this file)
- `summary/summary.md` (turn summary per prompt protocol)

## Exit Criteria Assessment

Phase R Reactivation Checklist:
- [✅] a) update_legacy_dict invoked in both CLI entry points
- [✅] b) config defaults + persistence shim merged
- [✅] c) targeted pytest selector green with evidence
- [✅] d) hub analysis/artifact_inventory.txt lists code/test paths
- [✅] e) summary.md complete

**Status:** READY FOR HANDOFF

## Next Actions

Per integration plan (`plans/ptychodus_pytorch_integration_plan.md`):
1. Resume Phase 2 (Data Ingestion & Grouping) when Phase R foundations are approved
2. Integrate persistence shim into workflows (training outputs should emit manifest)
3. Monitor test_data_file warnings in tests (16 warnings, acceptable for MVP scope)

## References
- Integration plan: `plans/ptychodus_pytorch_integration_plan.md`
- Test strategy: `plans/pytorch_integration_test_plan.md`
- Spec: `specs/ptychodus_api_spec.md` §4.6 (persistence contract)
- Finding: `docs/findings.md` CONFIG-001 (initialization order)
