# Phase D3.B Implementation Summary — PyTorch Persistence Shim

**Initiative:** INTEGRATE-PYTORCH-001 | **Phase:** D3.B (Archive Writer) | **Date:** 2025-10-17
**Status:** ✅ COMPLETE — Tests green, full regression passed

---

## Implementation Overview

Successfully implemented `save_torch_bundle` persistence shim producing wts.h5.zip-compatible archives
that maintain format parity with TensorFlow `ModelManager.save_multiple_models` baseline while
supporting torch-optional execution.

**Key Deliverables:**
1. `ptycho_torch/model_manager.py` (280 lines) — torch-optional persistence layer
2. `tests/torch/test_model_manager.py` (352 lines) — comprehensive bundle structure + params snapshot tests
3. Full test suite regression: 193 passed, 13 skipped, 0 new failures

---

## Archive Format Compliance

### Implemented Structure

```
{base_path}.zip/
├── manifest.dill         # {'models': [...], 'version': '2.0-pytorch'}
├── autoencoder/
│   ├── model.pth        # PyTorch state_dict (replaces model.keras)
│   └── params.dill      # Full params.cfg snapshot (CONFIG-001 gate)
└── diffraction_to_obj/
    ├── model.pth
    └── params.dill
```

### Parity Validation

| Requirement | TensorFlow Baseline | PyTorch Implementation | Status |
|-------------|---------------------|------------------------|--------|
| Dual-model bundle | ptycho/model_manager.py:346-378 | save_torch_bundle lines 70-190 | ✅ PASS |
| manifest.dill at root | ModelManager lines 361-364 | save_torch_bundle lines 130-139 | ✅ PASS |
| Per-model subdirs | ModelManager lines 366-370 | save_torch_bundle lines 141-173 | ✅ PASS |
| params.dill snapshot | ModelManager lines 73-78 | save_torch_bundle lines 107-121, 166-170 | ✅ PASS |
| Version tag '2.0-pytorch' | ModelManager line 76 (_version: '1.0') | save_torch_bundle line 121, 133 | ✅ PASS |
| CONFIG-001 compliance | update via dataclass_to_legacy_dict | save_torch_bundle lines 107-121 | ✅ PASS |
| Torch-optional imports | N/A (TensorFlow-only) | model_manager.py lines 43-51 | ✅ PASS |

**All spec §4.6 requirements satisfied.**

---

## Test Coverage

### Red Phase (TDD Validation)

**Targeted tests authored:**
1. `test_archive_structure` — validates zip structure compliance per spec §4.6
2. `test_params_snapshot` — validates CONFIG-001 params.cfg snapshot correctness

**Red-phase pytest logs:**
- `pytest_archive_structure_red.log` — SKIPPED (module not yet implemented) ✅
- `pytest_params_snapshot_red.log` — SKIPPED (module not yet implemented) ✅

### Green Phase (Implementation)

**Test Results:**
```
pytest tests/torch/test_model_manager.py -k bundle -vv
tests/torch/test_model_manager.py::TestSaveTorchBundle::test_archive_structure PASSED
tests/torch/test_model_manager.py::TestSaveTorchBundle::test_params_snapshot PASSED
============================== 2 passed in 7.71s ==============================
```

**Green-phase pytest log:** `pytest_green.log` ✅

**Validated Behaviors:**
- ✅ Archive creation at `{base_path}.zip`
- ✅ manifest.dill structure: `{'models': [...], 'version': '2.0-pytorch'}`
- ✅ Per-model subdirectories: `autoencoder/`, `diffraction_to_obj/`
- ✅ params.dill presence in each model directory
- ✅ model.pth presence (PyTorch state_dict)
- ✅ CONFIG-001 fields in params snapshot: N, gridsize, model_type, nphotons
- ✅ Version tag '2.0-pytorch' for backend detection

### Full Regression Suite

**Command:** `pytest tests/ --ignore=tests/test_benchmark_throughput.py --ignore=tests/test_run_baseline.py -v`

**Results:**
- **193 passed** (no new failures)
- **13 skipped** (pre-existing; documented in fix_plan.md LEGACY-TESTS-001)
- **17 warnings** (UserWarnings from test_config_bridge test_data_file; expected)

**Regression Status:** ✅ CLEAN — No new test failures introduced

---

## Implementation Details

### Core Functions

#### `save_torch_bundle(models_dict, base_path, config, intensity_scale=None)`

**Signature:** `Dict[str, nn.Module], str, TrainingConfig, Optional[float] → None`

**Key Design Decisions:**
1. **Dual-model validation:** Warns when `models_dict` keys ≠ `{'autoencoder', 'diffraction_to_obj'}` (spec §4.6 preference)
2. **CONFIG-001 bridge:** Uses `dataclass_to_legacy_dict(config)` to generate params snapshot (lines 107-121)
3. **intensity_scale handling:** 3-tier fallback — explicit arg → params.cfg → default 1.0 (lines 110-121)
4. **Version tagging:** Adds `_version: '2.0-pytorch'` to params snapshot + manifest (lines 121, 133)
5. **Torch-optional weights:** Handles both `nn.Module.state_dict()` and sentinel dicts for testing (lines 149-163)
6. **Temporary directory pattern:** Uses `tempfile.TemporaryDirectory` + `zipfile.ZipFile` (TF baseline parity)

#### `load_torch_bundle(base_path, model_name='diffraction_to_obj')`

**Signature:** `str, str → Tuple[nn.Module, dict]`

**Implementation Status:** 🔴 STUB — Raises `NotImplementedError` (Phase D3.C scope)

**Implemented Logic:**
- Archive extraction + manifest loading ✅
- params.dill loading ✅
- CONFIG-001 params.cfg restoration via `params.cfg.update(params_dict)` ✅
- Model reconstruction — ❌ **BLOCKED** (requires `create_torch_model_with_gridsize` helper from Phase D3.C)

**Stub Behavior:**
```python
raise NotImplementedError(
    "load_torch_bundle model reconstruction not yet implemented. "
    "Requires create_torch_model_with_gridsize helper from Phase D3.C. "
    f"params.cfg successfully restored: N={params_dict['N']}, gridsize={params_dict['gridsize']}"
)
```

**Phase D3.C TODO:** Implement model architecture reconstruction + weight loading (see lines 214-225 placeholder)

---

## CONFIG-001 Compliance

### Params Snapshot Fields Captured

| Field | Source | Value (test fixture) | Location |
|-------|--------|----------------------|----------|
| N | ModelConfig.N | 64 | params.dill via dataclass_to_legacy_dict |
| gridsize | ModelConfig.gridsize | 2 | params.dill via dataclass_to_legacy_dict |
| model_type | ModelConfig.model_type | 'pinn' | params.dill via dataclass_to_legacy_dict |
| nphotons | TrainingConfig.nphotons | 1e9 | params.dill via dataclass_to_legacy_dict |
| n_groups | TrainingConfig.n_groups | 10 | params.dill via dataclass_to_legacy_dict |
| neighbor_count | TrainingConfig.neighbor_count | 4 | params.dill via dataclass_to_legacy_dict |
| batch_size | TrainingConfig.batch_size | 4 | params.dill via dataclass_to_legacy_dict |
| nepochs | TrainingConfig.nepochs | 5 | params.dill via dataclass_to_legacy_dict |
| intensity_scale | arg / params.cfg / default | 1.0 (fallback) | params.dill manual addition |
| _version | hardcoded | '2.0-pytorch' | params.dill version tag |

**All CONFIG-001 critical fields present and validated by `test_params_snapshot`.**

---

## Torch-Optional Behavior

### Import Guard Pattern

```python
# ptycho_torch/model_manager.py:43-51
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
```

**Fallback Behavior:**
- When torch unavailable: sentinel dict models allowed for structure testing
- When torch available: full `nn.Module.state_dict()` serialization

**Test Harness Support:**
- Added `test_model_manager` to `tests/conftest.py:42` `TORCH_OPTIONAL_MODULES` whitelist
- Tests execute successfully without torch runtime (validates structure, skips weight loading)

---

## Deferred Work (Phase D3.C)

### Loader Implementation Blockers

**Q1: Model Architecture Reconstruction**
- **Issue:** PyTorch loader requires `create_torch_model_with_gridsize(gridsize, N)` helper
- **Current status:** Function does not exist; placeholder at `load_torch_bundle` lines 214-225
- **Resolution:** Phase D3.C must implement this factory function in `ptycho_torch/model.py` or equivalent

**Q2: Cross-Backend Loading**
- **Issue:** Can PyTorch load TensorFlow wts.h5.zip archives?
- **Current status:** Format compatible (manifest.dill + params.dill shared), but weights differ (model.pth vs model.keras)
- **Resolution:** Phase D3.C must decide: (a) separate loaders, or (b) format detection + dual-loader shim

**Q3: custom_objects.dill Requirement**
- **Issue:** TensorFlow saves ~25 Lambda layers in custom_objects.dill; PyTorch uses standard nn.Module
- **Current status:** Omitted from PyTorch archives (assumed unnecessary per Phase D3.A open question Q1)
- **Resolution:** Phase D3.C must validate no custom_objects needed or add minimal custom layer registry

---

## Artifacts Generated

### Test Files
- `tests/torch/test_model_manager.py` (352 lines) — 2 bundle structure tests + fixtures
- `tests/conftest.py` — updated torch-optional whitelist (line 42)

### Implementation Files
- `ptycho_torch/model_manager.py` (280 lines) — `save_torch_bundle` + `load_torch_bundle` stub

### Logs
- `pytest_archive_structure_red.log` — Red-phase SKIPPED (module missing)
- `pytest_params_snapshot_red.log` — Red-phase SKIPPED (module missing)
- `pytest_green.log` — Green-phase 2 PASSED (7.71s)

### Documentation
- `phase_d3b_summary.md` (this file)

---

## Next Steps (Phase D3.C)

### Immediate Actions

1. **Implement model factory:** Create `create_torch_model_with_gridsize(gridsize, N)` in `ptycho_torch/model.py`
2. **Complete loader:** Finish `load_torch_bundle` implementation (lines 214-225 placeholder)
3. **Add loader tests:** Extend `test_model_manager.py` with `TestLoadTorchBundle` class covering:
   - Archive extraction ✅ (already implemented)
   - params.cfg restoration ✅ (already implemented)
   - Model reconstruction ❌ (Phase D3.C scope)
   - Weight loading ❌ (Phase D3.C scope)
   - Shape mismatch prevention (CONFIG-001 validation test)

### Integration Tasks (Phase D3.D)

4. **Wire into training workflow:** Integrate `save_torch_bundle` call in `ptycho_torch/workflows/components.py::train_cdi_model_torch` after Lightning training completes
5. **Update `load_inference_bundle_torch`:** Replace Phase D2.C stub with `load_torch_bundle` delegation
6. **Document intensity_scale capture:** Resolve Phase D3.A open question Q2 (where is intensity_scale computed in PyTorch training?)

---

## Exit Criteria Status

### Phase D3.B Checklist

- ✅ **D3.B.1:** `save_torch_bundle` function implemented (torch-optional)
- ✅ **D3.B.2:** Dual-model bundle support (autoencoder + diffraction_to_obj)
- ✅ **D3.B.3:** params.dill snapshot via `dataclass_to_legacy_dict` (CONFIG-001 compliant)
- ✅ **D3.B.4:** manifest.dill with version='2.0-pytorch'
- ✅ **D3.B.5:** Test coverage (test_archive_structure + test_params_snapshot)
- ✅ **D3.B.6:** Full regression suite clean (193 passed, 0 new failures)
- ⚠️ **D3.B.7:** Integration into training workflow — **DEFERRED** to Phase D3.D (implementation complete, wiring pending)

**Phase D3.B: ✅ COMPLETE** — All core persistence shim requirements satisfied; loader implementation deferred to D3.C per plan.

---

## Summary One-Liner

**PyTorch save_torch_bundle produces spec-compliant wts.h5.zip archives with dual-model structure, CONFIG-001 params snapshots, and torch-optional execution; loader stub requires Phase D3.C model factory for completion.**
