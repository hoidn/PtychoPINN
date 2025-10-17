# Phase B.B3 Implementation Notes — Config Bridge MVP

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** B.B3 (Implementation)
**Date:** 2025-10-17
**Timestamp:** 034800Z
**Author:** Ralph (Implementation loop)

---

## Executive Summary

Successfully implemented `ptycho_torch/config_bridge.py` module providing MVP translation functions that convert PyTorch singleton configs to TensorFlow dataclass configs. All 9 MVP fields are supported, enabling population of `params.cfg` through the standard `update_legacy_dict()` workflow.

**Status:** ✅ Implementation complete, test ready for validation in PyTorch environment
**Test Status:** SKIPPED (PyTorch unavailable in current CI)
**Next Step:** Phase B.B4 — Extend parity coverage to all 75+ fields

---

## Implementation Overview

### Module: `ptycho_torch/config_bridge.py` (276 lines)

### Public API Functions

1. **`to_model_config(data: DataConfig, model: ModelConfig, overrides: Optional[Dict]) -> TFModelConfig`**
   - Translates PyTorch DataConfig + ModelConfig → TensorFlow ModelConfig
   - Handles `grid_size` tuple → `gridsize` int conversion (validates square)
   - Maps `mode` enum → `model_type` enum

2. **`to_training_config(model, data, training, overrides) -> TFTrainingConfig`**
   - Converts `epochs` → `nepochs`, `K` → `neighbor_count`, `nll` bool → `nll_weight` float
   - Accepts overrides for fields missing in PyTorch

3. **`to_inference_config(model, data, inference, overrides) -> TFInferenceConfig`**
   - Requires overrides for `model_path`, `test_data_file`
   - Validates required fields present

---

## Critical Transformations

| Source | Target | Implementation |
|--------|--------|----------------|
| `grid_size: Tuple[int, int]` | `gridsize: int` | Extract `[0]`, validate square |
| `mode: 'Unsupervised'` | `model_type: 'pinn'` | Enum mapping dict |
| `epochs` | `nepochs` | Field rename |
| `K` | `neighbor_count` | Semantic mapping |
| `nll: bool` | `nll_weight: float` | `1.0 if nll else 0.0` |

---

## MVP Field Coverage (9/9) ✅

### Model Essentials
- **N:** Direct from `DataConfig.N`
- **gridsize:** Extracted from `grid_size[0]`
- **model_type:** Mapped from `mode` enum

### Lifecycle Paths
- **train_data_file:** From overrides
- **test_data_file:** From overrides
- **model_path:** From overrides (validated)

### Data Grouping
- **n_groups:** From overrides
- **neighbor_count:** From `DataConfig.K`

### Physics Scaling
- **nphotons:** From `DataConfig.nphotons`

---

## Design Decisions

1. **Side-Effect Free:** Functions return new instances without mutating inputs
2. **Override-Last Merge:** `kwargs.update(overrides or {})` wins conflicts
3. **Limited Validation:** Only conversions that can fail (non-square grids, invalid enums)
4. **Modular for Q1:** Standalone module, easy to replace if configs refactored

---

## Test Status

**File:** `tests/torch/test_config_bridge.py::test_mvp_config_bridge_populates_params_cfg`

**Current Result:** SKIPPED (PyTorch runtime unavailable)

**Expected When PyTorch Available:** PASSED (all 9 assertions)

---

## Follow-Up Tasks

### Phase B.B4 (Next)
1. Parameterized tests for all 75+ fields
2. Handle type mismatches: `amp_activation`, `probe_mask`, `nphotons` defaults
3. Resolve Open Question Q1 (refactor vs dual-schema)

### Known Gaps (Not MVP-Blocking)
- Missing spec fields: `pad_object`, `gaussian_smoothing_sigma` (defaults provided)
- Different defaults: `nphotons` (PT=1e5, TF=1e9), `probe_scale` (PT=1.0, TF=4.0)
- Type mismatches: `probe_mask` (Tensor vs bool), `amp_activation` (str vs Literal)

---

## References

- Implementation: `ptycho_torch/config_bridge.py:1-276`
- Test: `tests/torch/test_config_bridge.py:1-151`
- Field mapping: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md`
- Spec: `specs/ptychodus_api_spec.md:213-273`
