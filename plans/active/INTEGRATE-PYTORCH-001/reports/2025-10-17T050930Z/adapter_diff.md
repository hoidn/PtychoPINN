# Phase A Adapter Diff — Optional Torch Import Implementation

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** B.B5 Phase A
**Timestamp:** 2025-10-17T050930Z
**Loop:** Attempt #14

---

## Summary
Implemented optional torch import guard to allow config bridge tests to execute without PyTorch runtime available. This enables parity testing in CI environments and development setups where torch may be unavailable or broken (e.g., CUDA symbol errors).

---

## File 1: `ptycho_torch/config_params.py`

### Change 1.1: Optional Torch Import Guard
**Location:** Lines 1-13
**Type:** Enhancement (backward compatible)

```diff
 from dataclasses import dataclass, field
-from typing import Tuple, Optional, Literal, Dict, Union, List
-import torch
+from typing import Tuple, Optional, Literal, Dict, Union, List, Any, TYPE_CHECKING
+
+# Optional torch import for environments where PyTorch may not be available
+# This allows config_bridge tests to run without PyTorch installed
+try:
+    import torch
+    TORCH_AVAILABLE = True
+    TensorType = torch.Tensor
+except ImportError:
+    torch = None
+    TORCH_AVAILABLE = False
+    TensorType = Any  # Fallback type for environments without torch
```

**Rationale:**
- Guarded import prevents ImportError when torch unavailable
- `TORCH_AVAILABLE` flag exported for consumers to check torch status
- `TensorType` alias provides type hint compatibility (torch.Tensor when available, Any when not)

**Impact:**
- No behavioral change when torch available
- Module becomes importable when torch missing (was previously hard failure)
- Enables dataclass instantiation with stub types

### Change 1.2: Update probe_mask Type Annotation
**Location:** Line 62
**Type:** Refactor (type annotation only)

```diff
-    probe_mask: Optional[torch.Tensor] = None # Optional probe mask tensor
+    probe_mask: Optional[TensorType] = None # Optional probe mask tensor
```

**Rationale:**
- Use `TensorType` alias instead of direct `torch.Tensor` reference
- Prevents NameError when torch unavailable
- Maintains same type hint when torch available

**Impact:**
- Type checkers see same type when torch available
- Runtime behavior unchanged (field still accepts None or tensor)
- Fallback to Any when torch missing (permissive typing)

---

## File 2: `ptycho_torch/config_bridge.py`

### Change 2.1: Remove Hard Import Error
**Location:** Lines 70-78
**Type:** Refactor (error handling removed)

```diff
-# Import PyTorch configs using try/except for environments where PyTorch may not be available
-try:
-    from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
-except ImportError as e:
-    raise ImportError(
-        "PyTorch config_params module not available. Ensure ptycho_torch package is installed."
-    ) from e
+# Import PyTorch configs - module should be importable even without torch
+# thanks to optional import guard in config_params.py
+from ptycho_torch.config_params import (
+    DataConfig,
+    ModelConfig,
+    TrainingConfig,
+    InferenceConfig,
+    TORCH_AVAILABLE
+)
```

**Rationale:**
- `config_params` module now importable without torch (Change 1.1)
- Hard error no longer necessary or desired
- Import `TORCH_AVAILABLE` flag for potential future runtime checks

**Impact:**
- Module imports successfully when torch unavailable
- Enables config bridge tests to collect and execute
- No change to behavior when torch available

---

## File 3: `tests/conftest.py`

### Change 3.1: Exemption for Config Bridge Tests
**Location:** Lines 38-46
**Type:** Enhancement (skip logic refinement)

```diff
     # Add skip markers for tests requiring unavailable dependencies
     for item in items:
-        # Skip torch tests if torch is not available
-        if "torch" in str(item.fspath).lower() or item.get_closest_marker("torch"):
-            if not torch_available:
-                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
+        # Skip torch tests if torch is not available
+        # EXCEPTION: config_bridge tests can run without torch (use stub types)
+        is_config_bridge_test = "test_config_bridge" in str(item.fspath)
+
+        if ("torch" in str(item.fspath).lower() or item.get_closest_marker("torch")):
+            if not torch_available and not is_config_bridge_test:
+                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

**Rationale:**
- Config bridge tests validate dataclass field mapping, not torch ops
- Tests should execute to verify translation logic even when torch unavailable
- Other `tests/torch/` tests still appropriately skipped (e.g., model training tests)

**Impact:**
- `tests/torch/test_config_bridge.py` now executes when torch unavailable
- Other torch tests retain skip behavior
- Enables parity test validation in CI without GPU/CUDA

---

## Verification

### Import Test
```bash
$ python3 -c "from ptycho_torch import config_bridge; print('Import successful')"
Import successful
```
✅ Module importable without torch

### Pytest Execution Test
```bash
$ pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP -v
...
tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg FAILED [100%]
...
```
✅ Test executes (not SKIPPED), reaches assertion stage

### Other Torch Tests Still Skip
```bash
$ pytest tests/torch/test_tf_helper.py -v
...
tests/torch/test_tf_helper.py::TestTorchTFHelperFunctions::test_placeholder_torch_functions SKIPPED [100%]
SKIPPED [1] tests/conftest.py:46: PyTorch not available
```
✅ Non-config-bridge torch tests still skip appropriately

---

## Risk Assessment

### Low Risk
1. **Backward Compatibility:** Changes are additive/permissive when torch available
2. **Type Safety:** `TensorType` alias preserves type hints for torch.Tensor
3. **Test Isolation:** Config bridge exemption scoped to single test file

### Medium Risk
1. **Parametrized Test Compatibility:** Discovered `@pytest.mark.parametrize` incompatibility with `unittest.TestCase` (requires test refactor)
2. **False Negatives:** Tests may pass with stub types but fail with real torch (addressed in Phase B validation)

### Mitigation
- Phase B will fix parametrized test issue (convert to pytest-style)
- Full parity suite with torch available planned for green phase validation

---

## Open Questions
1. Should `config_bridge` functions raise NotImplementedError when `TORCH_AVAILABLE=False`?
   - **Current:** Functions execute with stub types (permissive)
   - **Alternative:** Explicit runtime check + error (strict)
   - **Decision:** Defer to Phase B if issues arise

2. Should tests emit warning when running in fallback mode?
   - **Current:** Silent fallback
   - **Alternative:** pytest warning fixture
   - **Decision:** Not needed for Phase A (can add in Phase E if diagnostic value)

---

## Next Actions
1. Convert `TestConfigBridgeParity` to pytest-style (remove unittest inheritance)
2. Validate parity tests execute with all parameterizations
3. Proceed to Phase B P0 blockers (probe_mask, nphotons)
