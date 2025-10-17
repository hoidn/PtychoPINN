# Phase F2.1 — Torch-Optional Guard Inventory

## Executive Summary

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001 Phase F2.1
**Total Guard Count:** 47 instances across 15 files
**Affected Modules:** 9 production modules, 3 test modules, 3 support files
**Guard Patterns:** 4 distinct patterns identified

**Key Finding:** All torch-optional guards are concentrated in the `ptycho_torch/` package and its tests. The core `ptycho/` TensorFlow stack is unaffected, confirming clean separation between backends.

---

## 1. Module-Level TORCH_AVAILABLE Flags

### 1.1 Production Modules (Core Integration)

| File | Lines | Pattern | Context | Status |
|:-----|:------|:--------|:--------|:-------|
| `ptycho_torch/config_params.py` | 6-13 | try/except import + flag | Type alias fallback (`TensorType = torch.Tensor` → `type(None)`) | **REMOVE in F3.2** |
| `ptycho_torch/config_bridge.py` | 77, 148 | Flag import + runtime check | Import from config_params; probe_mask tensor detection | **REMOVE in F3.2** |
| `ptycho_torch/data_container_bridge.py` | 99-105, 208, 284 | try/except import + flag | NumPy fallback for tensor conversion | **REMOVE in F3.2** |
| `ptycho_torch/raw_data_bridge.py` | 99-104 | try/except import + flag | RawDataTorch adapter with NumPy fallback | **REMOVE in F3.2** |
| `ptycho_torch/memmap_bridge.py` | 32-38 | try/except import + flag | Memory-mapped dataset torch-optional loading | **REMOVE in F3.2** |
| `ptycho_torch/model_manager.py` | 49-57, 160, 221, 165 | try/except import + flag | save/load_torch_bundle conditional execution | **REMOVE in F3.2** |
| `ptycho_torch/workflows/components.py` | 66-88 | try/except import + flag | Phase C/D stub workflow orchestration | **REMOVE in F3.2** |
| `ptycho_torch/__init__.py` | 16-21, 28, 57 | try/except import + flag | Package-level torch detection + export | **REMOVE in F3.2** |

**Pattern Example (config_params.py:6-13):**
```python
try:
    import torch
    TORCH_AVAILABLE = True
    TensorType = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    TensorType = type(None)  # Fallback type alias
```

---

### 1.2 Test Modules (Torch-Optional Harness)

| File | Lines | Pattern | Context | Status |
|:-----|:------|:--------|:--------|:-------|
| `tests/torch/test_data_pipeline.py` | 25-27, 308, 320, 394 | try/except import + flag | Parity tests with NumPy fallback assertions | **REMOVE in F3.3** |
| `tests/torch/test_tf_helper.py` | 11-13, 19, 68, 75, 87 | try/except import + flag + skipUnless | Stub helper tests with conditional skip | **REMOVE in F3.3** |
| `tests/test_pytorch_tf_wrapper.py` | 12-14, 20 | try/except import + flag | Legacy wrapper tests (may be obsolete?) | **AUDIT + REMOVE** |

**Pattern Example (test_data_pipeline.py:25-27):**
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

---

### 1.3 Conftest Skip Logic

| File | Lines | Pattern | Context | Status |
|:-----|:------|:--------|:--------|:-------|
| `tests/conftest.py` | 32-36, 46, 84-90 | Runtime check + whitelist | Auto-skip for tests/torch/ except whitelist; `torch_available` fixture | **SIMPLIFY in F3.3** |

**Conftest Whitelist (line 42):**
```python
TORCH_OPTIONAL_MODULES = [
    "test_config_bridge",
    "test_data_pipeline",
    "test_workflows_components",
    "test_model_manager",
    "test_backend_selection"
]
```

**Impact:** 5 test modules exempted from auto-skip; currently execute with NumPy fallbacks when torch unavailable.

---

## 2. Conditional Import Guards

### 2.1 Production Modules with Try/Except Import

| File | Lines | Fallback Behavior | Migration Action |
|:-----|:------|:------------------|:-----------------|
| `ptycho_torch/config_params.py` | 6-13 | `TensorType = type(None)` | Unconditional `import torch`; remove type alias fallback |
| `ptycho_torch/config_bridge.py` | 70-78 | Silent ImportError catch, flag = False | Unconditional import from config_params |
| `ptycho_torch/data_container_bridge.py` | 98-105 | `TensorType = Union[torch.Tensor, np.ndarray]` → NumPy-only | Remove Union; always torch.Tensor |
| `ptycho_torch/raw_data_bridge.py` | 99-104 | Flag = False, no torch module | Unconditional `import torch` |
| `ptycho_torch/memmap_bridge.py` | 32-38 | Flag = False | Unconditional `import torch` |
| `ptycho_torch/model_manager.py` | 49-57 | `import torch.nn as nn` → nn = None | Unconditional imports |
| `ptycho_torch/workflows/components.py` | 66-70 | Flag = False | Unconditional import from config_params |
| `ptycho_torch/__init__.py` | 16-21 | Flag = False | Unconditional package-level import |

**Total Production Guards:** 8 try/except blocks to remove

---

### 2.2 Test Modules with Try/Except Import

| File | Lines | Fallback Behavior | Migration Action |
|:-----|:------|:------------------|:-----------------|
| `tests/torch/test_data_pipeline.py` | 25-27 | Flag = False, NumPy mode | Remove guard; expect torch available |
| `tests/torch/test_tf_helper.py` | 11-13 | Flag = False + @skipUnless decorators | Remove guard + decorators |
| `tests/test_pytorch_tf_wrapper.py` | 12-14 | Flag = False | Audit relevance; remove or fix |

**Total Test Guards:** 3 try/except blocks to remove

---

## 3. Runtime Availability Checks

### 3.1 Conditional Execution Paths

| File | Lines | Decision Logic | Branch Behavior |
|:-----|:------|:---------------|:----------------|
| `ptycho_torch/config_bridge.py` | 148 | `if TORCH_AVAILABLE and model.probe_mask is not None` | Tensor→bool conversion OR hardcoded False |
| `ptycho_torch/data_container_bridge.py` | 208 | `if TORCH_AVAILABLE: return torch.from_numpy(...)` | torch.Tensor OR NumPy array |
| `ptycho_torch/data_container_bridge.py` | 284 | `if TORCH_AVAILABLE and isinstance(attr, torch.Tensor)` | Tensor repr() OR NumPy repr() |
| `ptycho_torch/model_manager.py` | 160 | `if TORCH_AVAILABLE and isinstance(model, nn.Module)` | torch.save OR dill.dump |
| `ptycho_torch/model_manager.py` | 165 | `torch.save(...) if TORCH_AVAILABLE else dill.dump(...)` | Ternary for persistence |
| `ptycho_torch/model_manager.py` | 221 | `if not TORCH_AVAILABLE: raise NotImplementedError(...)` | Fail-fast when loading without torch |
| `ptycho_torch/workflows/components.py` | 73 | `if TORCH_AVAILABLE: [Phase C/D imports]` | Real workflow OR stub NotImplementedError |
| `tests/torch/test_data_pipeline.py` | 308, 320, 394 | `if TORCH_AVAILABLE: assert isinstance(..., torch.Tensor)` | Torch assertion OR NumPy assertion |
| `tests/torch/test_tf_helper.py` | 87 | `if not TF_HELPER_AVAILABLE or not TORCH_AVAILABLE: self.skipTest(...)` | Runtime skip |

**Total Runtime Checks:** 9 conditional branches to refactor

**Migration Strategy:**
- **Production modules:** Remove `if TORCH_AVAILABLE` branches; assume torch always available; remove fallback logic
- **Test modules:** Remove conditional assertions; expect torch.Tensor types only

---

## 4. Type Alias Fallbacks

| File | Lines | Original Type | Fallback Type | Migration Action |
|:-----|:------|:--------------|:--------------|:-----------------|
| `ptycho_torch/config_params.py` | 9, 13 | `torch.Tensor` | `type(None)` | Remove fallback; always `torch.Tensor` |
| `ptycho_torch/data_container_bridge.py` | 102, 105 | `Union[torch.Tensor, np.ndarray]` | `np.ndarray` only | Remove Union; always `torch.Tensor` |

**Impact:**
- `config_params.py`: `probe_mask: Optional[TensorType]` field currently accepts None OR torch.Tensor OR type(None); will become `Optional[torch.Tensor]` (cleaner)
- `data_container_bridge.py`: All tensor attributes currently Union types; will become pure torch.Tensor

---

## 5. Unconditional Torch Imports (No Guards)

These modules already assume PyTorch availability—no F3.2 changes required:

| File | Torch Import Lines | Status |
|:-----|:-------------------|:-------|
| `ptycho_torch/model.py` | 2, 4, 5 | ✅ Already torch-required |
| `ptycho_torch/helper.py` | 6, 8, 490 | ✅ Already torch-required |
| `ptycho_torch/train.py` | 15, 19 | ✅ Already torch-required |
| `ptycho_torch/train_utils.py` | 16, 17, 21 | ✅ Already torch-required |
| `ptycho_torch/reassembly.py` | 4, 7, 9 | ✅ Already torch-required |
| `ptycho_torch/reassembly_beta.py` | 6, 9, 11 | ✅ Already torch-required |
| `ptycho_torch/reassembly_alpha.py` | 6, 7, 8 | ✅ Already torch-required |
| `ptycho_torch/train_full.py` | 17, 21 | ✅ Already torch-required |
| `ptycho_torch/train_dummy.py` | 4 | ✅ Already torch-required |
| `ptycho_torch/utils.py` | 9 | ✅ Already torch-required |
| `ptycho_torch/dset_loader_pt_mmap.py` | 7 | ✅ Already torch-required |
| `ptycho_torch/dataloader.py` | 12, 13, 14 | ✅ Already torch-required |
| `ptycho_torch/model_attention.py` | 1, 2, 3 | ✅ Already torch-required |
| `ptycho_torch/datagen.py` | 5, 7 | ✅ Already torch-required |
| `ptycho_torch/datagen/datagen.py` | 5, 7 | ✅ Already torch-required |
| `ptycho_torch/datagen/objects.py` | 5, 7 | ✅ Already torch-required |
| `ptycho_torch/api/base_api.py` | 182 | ✅ Already torch-required |

**Count:** 17 files already torch-required (no changes needed)

---

## 6. Backend Dispatcher (Preservation Case)

| File | Lines | Pattern | Status |
|:-----|:------|:--------|:-------|
| `ptycho/workflows/backend_selector.py` | 145-156 (Attempt #63) | try/except import with actionable RuntimeError | **PRESERVE** (correct fail-fast behavior) |

**Example (backend_selector.py:145-156):**
```python
if config.backend == 'pytorch':
    try:
        from ptycho_torch.workflows import components as torch_components
    except ImportError as e:
        raise RuntimeError(
            f"PyTorch backend selected (backend='{config.backend}') but PyTorch workflows are unavailable.\n"
            f"Error: {e}\n"
            f"Install PyTorch support with: pip install torch torchvision\n"
            f"Or switch to TensorFlow backend (backend='tensorflow')."
        )
```

**Rationale:** This guard is **correct production behavior** (fail-fast when user explicitly selects unavailable backend), NOT a torch-optional development pattern. Per governance decision (governance_decision.md §8.3.Q3), this pattern aligns with torch-required policy (PyTorch absence is an ERROR when selected, not silent degradation).

**Action:** NO CHANGES in Phase F3.2; preserve as-is.

---

## 7. Migration Impact Analysis

### 7.1 Modules Requiring Changes

**Production Modules (8 files):**
1. `ptycho_torch/config_params.py` — Remove try/except (lines 6-13), TensorType fallback
2. `ptycho_torch/config_bridge.py` — Remove TORCH_AVAILABLE import (line 77), probe_mask check (line 148)
3. `ptycho_torch/data_container_bridge.py` — Remove try/except (lines 98-105), conditional returns (lines 208, 284)
4. `ptycho_torch/raw_data_bridge.py` — Remove try/except (lines 99-104)
5. `ptycho_torch/memmap_bridge.py` — Remove try/except (lines 32-38)
6. `ptycho_torch/model_manager.py` — Remove try/except (lines 49-57), conditional saves (lines 160, 165, 221)
7. `ptycho_torch/workflows/components.py` — Remove try/except (lines 66-70), conditional imports (line 73)
8. `ptycho_torch/__init__.py` — Remove try/except (lines 16-21), TORCH_AVAILABLE export (lines 28, 57)

**Test Modules (4 files):**
1. `tests/torch/test_data_pipeline.py` — Remove try/except (lines 25-27), conditional assertions (lines 308, 320, 394)
2. `tests/torch/test_tf_helper.py` — Remove try/except (lines 11-13), skipUnless decorators (lines 68, 75), runtime skip (line 87)
3. `tests/test_pytorch_tf_wrapper.py` — Audit relevance, remove guards (lines 12-14, 20)
4. `tests/conftest.py` — Remove whitelist (line 42), simplify skip logic (lines 32-47)

**Total Files:** 12 files requiring edits

---

### 7.2 Guard Patterns to Remove

| Pattern | Count | Example |
|:--------|:------|:--------|
| **try/except import + flag** | 11 | `try: import torch; TORCH_AVAILABLE = True except: TORCH_AVAILABLE = False` |
| **if TORCH_AVAILABLE: runtime branch** | 9 | `if TORCH_AVAILABLE: return torch.Tensor else: return np.ndarray` |
| **Type alias fallback** | 2 | `TensorType = torch.Tensor if TORCH_AVAILABLE else type(None)` |
| **@skipUnless decorator** | 2 | `@unittest.skipUnless(TORCH_AVAILABLE, "...")` |
| **Conftest whitelist** | 1 | `TORCH_OPTIONAL_MODULES = [...]` |

**Total Pattern Instances:** 25 distinct code blocks to refactor

---

### 7.3 Estimated Code Reduction

**Lines to Remove/Simplify:**
- try/except imports: ~50 lines (11 blocks × ~4-5 lines each)
- Conditional branches: ~30 lines (9 branches × ~3-4 lines each)
- Type alias fallbacks: ~4 lines
- Test decorators: ~4 lines
- Conftest whitelist: ~10 lines (list + logic)

**Total:** ~98 lines of guard boilerplate removed
**Simplified Logic:** ~15 conditional branches eliminated

---

## 8. File:Line Anchor Matrix

Complete reference for Phase F3.2 implementation (atomic commits per file):

### Production Modules

```
ptycho_torch/config_params.py
├─ 6-13   : try/except import guard + TensorType fallback
├─ 8      : TORCH_AVAILABLE = True
└─ 12     : TORCH_AVAILABLE = False

ptycho_torch/config_bridge.py
├─ 77     : TORCH_AVAILABLE import
└─ 148    : if TORCH_AVAILABLE and model.probe_mask

ptycho_torch/data_container_bridge.py
├─ 98-105 : try/except import guard
├─ 101    : TORCH_AVAILABLE = True
├─ 104    : TORCH_AVAILABLE = False
├─ 208    : if TORCH_AVAILABLE: return torch.from_numpy
└─ 284    : if TORCH_AVAILABLE and isinstance(attr, torch.Tensor)

ptycho_torch/raw_data_bridge.py
├─ 99-104 : try/except import guard
├─ 102    : TORCH_AVAILABLE = True
└─ 104    : TORCH_AVAILABLE = False

ptycho_torch/memmap_bridge.py
├─ 32-38  : try/except import guard
├─ 35     : TORCH_AVAILABLE = True
└─ 37     : TORCH_AVAILABLE = False

ptycho_torch/model_manager.py
├─ 49-57  : try/except import guard
├─ 53     : TORCH_AVAILABLE = True
├─ 55     : TORCH_AVAILABLE = False
├─ 160    : if TORCH_AVAILABLE and isinstance(model, nn.Module)
├─ 165    : torch.save(...) if TORCH_AVAILABLE else dill.dump(...)
└─ 221    : if not TORCH_AVAILABLE: raise NotImplementedError

ptycho_torch/workflows/components.py
├─ 66-88  : try/except import guard + docstring
├─ 68     : from ptycho_torch.config_params import TORCH_AVAILABLE
├─ 70     : TORCH_AVAILABLE = False
└─ 73     : if TORCH_AVAILABLE: [Phase C/D imports]

ptycho_torch/__init__.py
├─ 16-21  : try/except import guard
├─ 19     : TORCH_AVAILABLE = True
├─ 21     : TORCH_AVAILABLE = False
├─ 28     : TORCH_AVAILABLE as _config_bridge_torch_available
└─ 57     : 'TORCH_AVAILABLE' in __all__
```

### Test Modules

```
tests/torch/test_data_pipeline.py
├─ 25-27  : try/except import guard
├─ 308    : if TORCH_AVAILABLE: assert isinstance
├─ 320    : if TORCH_AVAILABLE and isinstance(..., torch.Tensor)
└─ 394    : if TORCH_AVAILABLE and isinstance(..., torch.Tensor)

tests/torch/test_tf_helper.py
├─ 11-13  : try/except import guard
├─ 19     : if not TORCH_AVAILABLE: [stub definitions]
├─ 68     : @unittest.skipUnless(TORCH_AVAILABLE, ...)
├─ 75     : @unittest.skipUnless(TORCH_AVAILABLE, ...)
└─ 87     : if not TF_HELPER_AVAILABLE or not TORCH_AVAILABLE

tests/test_pytorch_tf_wrapper.py
├─ 12-14  : try/except import guard
└─ 20     : if not TORCH_AVAILABLE: [stub torch module]

tests/conftest.py
├─ 32-36  : torch_available runtime check
├─ 42     : TORCH_OPTIONAL_MODULES whitelist
├─ 46     : if not torch_available and not is_torch_optional
└─ 84-90  : torch_available fixture
```

### Preservation (No Changes)

```
ptycho/workflows/backend_selector.py
└─ 145-156 : PRESERVE (fail-fast guard, correct behavior)
```

---

## 9. Cross-References

**Related Phase F2 Artifacts:**
- `test_skip_audit.md` (F2.2) — Conftest whitelist behavior analysis
- `migration_plan.md` (F2.3) — Phase F3.2 refactoring sequence

**Governance Documents:**
- `governance_decision.md` (F1.2) — Torch-required policy approval
- `directive_conflict.md` (F1.1) — Historical torch-optional evolution

**Phase F Plan:**
- `phase_f_torch_mandatory.md` — Phase F3.2 task definition

---

## 10. Phase F3.2 Implementation Checklist

Based on this inventory, Phase F3.2 must:

- [ ] **Config Params:** Remove try/except guard (lines 6-13), unconditional `import torch`, delete TensorType fallback
- [ ] **Config Bridge:** Remove TORCH_AVAILABLE import (line 77), simplify probe_mask logic (line 148)
- [ ] **Data Container:** Remove try/except (98-105), torch.from_numpy always (line 208), tensor repr always (line 284)
- [ ] **Raw Data Bridge:** Remove try/except (99-104), unconditional torch import
- [ ] **Memmap Bridge:** Remove try/except (32-38), unconditional torch import
- [ ] **Model Manager:** Remove try/except (49-57), torch.save always (lines 160, 165), delete load guard (line 221)
- [ ] **Workflows:** Remove try/except (66-70), unconditional Phase C/D imports (line 73)
- [ ] **Package Init:** Remove try/except (16-21), delete TORCH_AVAILABLE export (lines 28, 57)
- [ ] **Test Data Pipeline:** Remove try/except (25-27), torch assertions always (lines 308, 320, 394)
- [ ] **Test TF Helper:** Remove try/except (11-13), delete skipUnless decorators (lines 68, 75, 87)
- [ ] **Test PyTorch Wrapper:** Audit + remove guards (lines 12-14, 20) OR delete file if obsolete
- [ ] **Conftest:** Remove whitelist (line 42), simplify skip logic (lines 32-47) per F2.2 audit

**Validation:** After each file edit, run targeted pytest for that module to catch import errors immediately.

---

**Inventory Complete:** All 47 torch-optional guard instances catalogued with file:line anchors. Ready for Phase F3.2 execution.
