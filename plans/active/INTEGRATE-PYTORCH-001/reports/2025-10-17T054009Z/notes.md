# Phase B.B5.B2+B4 — probe_mask and nphotons Parity Coverage

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** B.B5 (Parity Green — probe_mask/nphotons extension)
**Attempt:** #21

## Summary

Extended config bridge test suite with explicit coverage for `probe_mask` translation logic and `nphotons` override validation. Both test additions implement Phase B.B5 tasks B2 (probe_mask parity) and B4 (nphotons override error regression).

## Changes Made

### 1. probe_mask Translation Tests (Phase B.B5.B2)

Added two test functions to `tests/torch/test_config_bridge.py`:

#### Test 1: `test_model_config_probe_mask_translation`
- **Location:** lines 411-435
- **Purpose:** Validates default translation behavior (None → False)
- **Parameterization:** Single case `probe_mask-default`
- **Note:** Tensor→True case documented but untestable without torch runtime in fallback mode
- **Coverage:** §5.1:8 (probe_mask field)

#### Test 2: `test_model_config_probe_mask_override`
- **Location:** lines 437-459
- **Purpose:** Validates explicit override pattern (overrides={'probe_mask': True})
- **Confirms:** External callers can force True even when PyTorch config has None
- **Coverage:** §5.1:8 (probe_mask override pattern)

### 2. nphotons Override Validation Tests (Phase B.B5.B4)

Added two paired test functions implementing error regression and green path:

#### Test 1: `test_nphotons_default_divergence_error`
- **Location:** lines 607-643
- **Purpose:** Validates that PyTorch default (1e5) without override raises ValueError
- **Error message assertions:**
  - Contains `'nphotons default divergence'`
  - Contains `'overrides'` parameter guidance
  - Contains `'nphotons='` syntax example
- **Coverage:** §5.2:9 (nphotons HIGH risk divergence)

#### Test 2: `test_nphotons_override_passes_validation`
- **Location:** lines 645-677
- **Purpose:** Green path test confirming explicit override passes validation
- **Validates:** `nphotons=1e9` override applied successfully
- **Coverage:** §5.2:9 (nphotons override pattern)

## Test Execution Results

### Targeted Selector Run

**Command:**
```bash
pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -vv
```

**Results:**
- **5 tests collected** (4 new + 1 existing nphotons-divergence)
- **5 PASSED**
- **34 deselected**
- **Execution time:** 3.71s

**Test breakdown:**
1. ✅ `test_model_config_probe_mask_translation[probe_mask-default]` — PASSED
2. ✅ `test_model_config_probe_mask_override` — PASSED
3. ✅ `test_default_divergence_detection[nphotons-divergence]` — PASSED (existing)
4. ✅ `test_nphotons_default_divergence_error` — PASSED (new)
5. ✅ `test_nphotons_override_passes_validation` — PASSED (new)

### MVP Regression Check

**Command:**
```bash
pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv
```

**Results:**
- **1 test collected**
- **1 PASSED**
- **Execution time:** 3.25s

Confirms broader parity unaffected by new test additions.

## Adapter Implementation (Already in Place)

The following adapter logic was implemented in prior attempts:

### probe_mask Translation (Attempt #17)
```python
# ptycho_torch/config_bridge.py:144-150
probe_mask_value = False  # Default when None
if TORCH_AVAILABLE and model.probe_mask is not None:
    # If torch available and probe_mask is a tensor, enable masking
    probe_mask_value = True
```

### nphotons Override Validation (Attempt #17)
```python
# ptycho_torch/config_bridge.py:253-263
pytorch_default_nphotons = 1e5
tensorflow_default_nphotons = 1e9
if 'nphotons' not in overrides and data.nphotons == pytorch_default_nphotons:
    raise ValueError(
        f"nphotons default divergence detected: PyTorch default ({pytorch_default_nphotons}) "
        f"differs from TensorFlow default ({tensorflow_default_nphotons}). "
        f"Provide explicit nphotons override to resolve: "
        f"overrides=dict(..., nphotons={tensorflow_default_nphotons})"
    )
```

Both implementations are now covered by explicit parity tests.

## Remaining Gaps (Not Blocking)

1. **Tensor→True probe_mask case:** Cannot test without torch tensor in runtime-agnostic harness; logic implemented but not testable in fallback mode
2. **Phase D baseline comparison:** Deferred params.cfg snapshot diffing (not required for probe_mask/nphotons coverage)

## Next Steps

Per Phase B.B5 green plan:
- ✅ Phase B.B5.B2 complete — probe_mask parity cases added and green
- ✅ Phase B.B5.B4 complete — nphotons override error regression with message validation green
- **Next:** Phase C tasks (n_subsample semantics, error handling) or Phase D (params.cfg baseline comparison)

## Artifacts

- `pytest_probe_mask.log` — Targeted selector output (5 tests PASSED)
- `pytest_mvp.log` — MVP regression check (1 test PASSED)
- `notes.md` — This summary document

## References

- Parity green plan: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md`
- Field matrix: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md`
- Adapter implementation: `ptycho_torch/config_bridge.py:144-150, 253-263`
- Test file: `tests/torch/test_config_bridge.py:411-677`
