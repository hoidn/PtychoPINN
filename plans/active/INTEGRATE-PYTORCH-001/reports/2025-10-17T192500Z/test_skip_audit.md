# Phase F2.2 — Pytest Skip Logic Audit

## Executive Summary

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001 Phase F2.2
**Current Skip Mechanism:** Conftest-based auto-skip with 5-module whitelist exception
**Whitelist Coverage:** 56+ tests execute without torch; others skip in torch-free environments
**Torch-Required Impact:** Whitelist removal + directory-based skip (TF-only CI preserves skip, Ptychodus CI fails on import error)

**Key Finding:** The current torch-optional harness creates a **bifurcated test behavior**:
- **Whitelist tests:** Execute with NumPy fallbacks when torch unavailable (validates adapter logic, NOT production workflows)
- **Non-whitelist tests:** Skip entirely when torch unavailable (hides integration failures)

Torch-required transition eliminates this bifurcation: ALL tests in `tests/torch/` assume PyTorch installed and FAIL (not skip) when unavailable.

---

## 1. Conftest Skip Mechanism Analysis

### 1.1 Current Implementation (tests/conftest.py:25-47)

```python
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle optional dependencies.
    This runs after test collection and can modify the collected items.
    """

    # Check what optional dependencies are available
    torch_available = True
    try:
        import torch
    except ImportError:
        torch_available = False

    # Add skip markers for tests requiring unavailable dependencies
    for item in items:
        # Skip torch tests if torch is not available
        # EXCEPTIONS: Some torch/ tests can run without torch (use fallback/stub types)
        TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline", "test_workflows_components", "test_model_manager", "test_backend_selection"]
        is_torch_optional = any(module in str(item.fspath) for module in TORCH_OPTIONAL_MODULES)

        if ("torch" in str(item.fspath).lower() or item.get_closest_marker("torch")):
            if not torch_available and not is_torch_optional:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

**Mechanism:**
1. **Runtime torch detection:** `try: import torch` → sets `torch_available` flag (lines 32-36)
2. **Whitelist exemption:** `TORCH_OPTIONAL_MODULES` list (line 42) defines exceptions
3. **Conditional skip:** Tests in `tests/torch/` skip UNLESS module name in whitelist (lines 45-47)

**Logic Table:**

| Test Location | Torch Available? | In Whitelist? | Behavior |
|:--------------|:-----------------|:--------------|:---------|
| `tests/torch/test_config_bridge.py` | YES | YES | EXECUTE |
| `tests/torch/test_config_bridge.py` | NO | YES | EXECUTE (NumPy fallback) |
| `tests/torch/test_model.py` | YES | NO | EXECUTE |
| `tests/torch/test_model.py` | NO | NO | SKIP |
| `tests/test_misc.py` (non-torch) | YES | N/A | EXECUTE |
| `tests/test_misc.py` (non-torch) | NO | N/A | EXECUTE |

---

### 1.2 Whitelist Modules (Details)

| Module | Test Count (Approx) | Current Behavior (No Torch) | Torch-Required Behavior (No Torch) |
|:-------|:--------------------|:----------------------------|:-----------------------------------|
| `test_config_bridge` | 39 | EXECUTES (NumPy mode, TORCH_AVAILABLE=False) | FAILS (ImportError in adapter modules) |
| `test_data_pipeline` | 12 | EXECUTES (NumPy arrays, conditional assertions) | FAILS (ImportError in ptycho_torch.data_container_bridge) |
| `test_workflows_components` | 8 | EXECUTES (stub workflow spies, no real PyTorch calls) | FAILS (ImportError in ptycho_torch.workflows.components) |
| `test_model_manager` | 4 | EXECUTES (save/load_torch_bundle stubs) | FAILS (ImportError in ptycho_torch.model_manager) |
| `test_backend_selection` | 5 | EXECUTES (dispatcher fail-fast validation) | FAILS (ImportError when backend='pytorch' selected) |

**Total Whitelisted Tests:** ~68 tests across 5 modules

**Historical Context (from docs/fix_plan.md):**
- Whitelist grew organically: 1 module (Attempt #15, test_config_bridge) → 5 modules (Attempt #63, full stack)
- Purpose: Enable TDD cycles without requiring PyTorch installation in early development phases
- Side effect: Created torch-optional execution pattern that conflicts with production backend goals

---

### 1.3 Non-Whitelist Tests (Currently Skip)

**Files in `tests/torch/` NOT in whitelist:**

1. `tests/torch/test_tf_helper.py` — PyTorch tensor helper shims
   - **Current:** Skips when torch unavailable (has own @skipUnless decorators as backup)
   - **Count:** ~6 tests
   - **Torch-Required:** Will FAIL on import without torch

2. *(Future tests)* — Any new torch/ tests not explicitly whitelisted
   - **Current:** Would auto-skip
   - **Torch-Required:** Will FAIL on import

**Total Non-Whitelist Tests:** ~6 tests + future additions

---

## 2. Test Marker Usage

### 2.1 `@pytest.mark.torch` Decorator

**Usage:** Currently UNUSED in the codebase (registered in conftest.py:21 but not applied to any tests)

**Intent:** Explicit marker for tests requiring PyTorch (alternative to directory-based detection)

**Torch-Required Impact:** Marker becomes **informational only** (all tests in `tests/torch/` assumed to need PyTorch by directory location)

**Recommendation:** RETAIN marker registration for documentation purposes; do NOT require explicit marking (directory-based skip sufficient)

---

### 2.2 `torch_available` Fixture (conftest.py:84-90)

```python
@pytest.fixture(scope="session")
def torch_available():
    """Fixture to check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False
```

**Current Usage:** ZERO tests currently use this fixture (grep found no references)

**Intent:** Allow tests to conditionally execute different logic based on torch availability

**Torch-Required Impact:** Fixture becomes **obsolete** (always returns True in torch-required environment)

**Recommendation:** REMOVE fixture in Phase F3.3 (no consumers to break)

---

## 3. Module-Level Skip Decorators

**Search Results:** FOUND in `tests/torch/test_tf_helper.py`

```python
# Line 68-70
@unittest.skipUnless(TF_HELPER_AVAILABLE and TORCH_AVAILABLE, "torch tf_helper module or torch not available")
def test_combine_complex(self):
    ...

# Line 75-77
@unittest.skipUnless(TF_HELPER_AVAILABLE and TORCH_AVAILABLE, "torch tf_helper module or torch not available")
def test_get_mask(self):
    ...
```

**Pattern:** `@unittest.skipUnless(TORCH_AVAILABLE, ...)` decorators as **secondary skip layer** (conftest auto-skip already handles this)

**Redundancy:** Decorators are defensive programming (ensure skip even if conftest fails)

**Torch-Required Impact:** Decorators become **counterproductive** (prevent tests from executing in torch-required environment)

**Recommendation:** REMOVE decorators in Phase F3.3 (conftest directory-based skip sufficient; tests should FAIL not skip in torch-required CI)

---

## 4. Expected Behavior Changes

### 4.1 Tests That Will Change from SKIP → FAIL (Non-Whitelist)

| Test File | Test Name | Current (No Torch) | Torch-Required (No Torch) |
|:----------|:----------|:-------------------|:--------------------------|
| `tests/torch/test_tf_helper.py` | `test_combine_complex` | SKIPPED (conftest + decorator) | FAILED (ImportError: no module named 'torch') |
| `tests/torch/test_tf_helper.py` | `test_get_mask` | SKIPPED (conftest + decorator) | FAILED (ImportError) |
| `tests/torch/test_tf_helper.py` | `test_placeholder_torch_functions` | SKIPPED (conftest) | FAILED (ImportError) |

**Count:** ~6 tests (all in test_tf_helper.py)

**Impact:** These tests currently hide torch unavailability via skip; torch-required exposes import failures as explicit test failures.

---

### 4.2 Tests That Will Change from EXECUTE → FAIL (Whitelist)

| Test File | Test Name (Example) | Current (No Torch) | Torch-Required (No Torch) | Reason |
|:----------|:--------------------|:-------------------|:--------------------------|:-------|
| `test_config_bridge.py` | `test_mvp_config_bridge_populates_params_cfg` | PASSED (NumPy mode) | FAILED (ImportError) | Loses fallback import guard in config_params.py |
| `test_data_pipeline.py` | `test_data_container_shapes_and_dtypes` | PASSED (NumPy arrays) | FAILED (ImportError) | Loses NumPy fallback in data_container_bridge.py |
| `test_workflows_components.py` | `test_run_cdi_example_stub` | PASSED (monkeypatch spies) | FAILED (ImportError) | Loses stub workflows in components.py |
| `test_model_manager.py` | `test_archive_structure` | PASSED (stub save) | FAILED (ImportError) | Loses torch.save fallback in model_manager.py |
| `test_backend_selection.py` | `test_pytorch_backend_selection` | PASSED (dispatcher validation) | FAILED (ImportError) | Backend dispatcher imports ptycho_torch.workflows (fails without torch) |

**Count:** ~68 tests across 5 modules

**Impact:** Whitelisted tests currently validate adapter logic in torch-free environments; torch-required removes this capability (tests validate production workflows only, requiring PyTorch installed).

---

## 5. Conftest Migration Plan

### 5.1 Changes Required

1. **Remove whitelist:** Delete `TORCH_OPTIONAL_MODULES` list (line 42)
2. **Simplify skip logic:** Keep directory-based skip for TF-only CI, remove per-module exceptions
3. **Update markers:** `@pytest.mark.torch` remains informational (no functional change)
4. **Remove fixture:** Delete `torch_available` fixture (lines 84-90, zero consumers)

---

### 5.2 Before/After Comparison

**Before (torch-optional, conftest.py:25-47):**

```python
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle optional dependencies.
    This runs after test collection and can modify the collected items.
    """

    # Check what optional dependencies are available
    torch_available = True
    try:
        import torch
    except ImportError:
        torch_available = False

    # Add skip markers for tests requiring unavailable dependencies
    for item in items:
        # Skip torch tests if torch is not available
        # EXCEPTIONS: Some torch/ tests can run without torch (use fallback/stub types)
        TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline", "test_workflows_components", "test_model_manager", "test_backend_selection"]
        is_torch_optional = any(module in str(item.fspath) for module in TORCH_OPTIONAL_MODULES)

        if ("torch" in str(item.fspath).lower() or item.get_closest_marker("torch")):
            if not torch_available and not is_torch_optional:
                item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

**After (torch-required, PROPOSED):**

```python
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to handle optional dependencies.

    In torch-required environments (Ptychodus CI), PyTorch must be installed;
    tests in tests/torch/ will FAIL (not skip) if torch unavailable.

    In TF-only CI environments, tests/torch/ directory is skipped entirely.
    """

    # Check if PyTorch is available
    torch_available = True
    try:
        import torch
    except ImportError:
        torch_available = False

    # Skip torch tests only in TF-only CI environments (directory-based)
    for item in items:
        if "torch" in str(item.fspath).lower() or item.get_closest_marker("torch"):
            if not torch_available:
                # No whitelist exceptions: ALL torch tests skip in TF-only CI
                item.add_marker(pytest.mark.skip(reason="PyTorch not available (TF-only CI)"))
```

**Key Differences:**
1. **Whitelist removed:** No `TORCH_OPTIONAL_MODULES` exceptions
2. **Comment updated:** Clarifies torch-required vs TF-only CI behavior
3. **Skip reason updated:** Explicit "(TF-only CI)" suffix for clarity
4. **Simplified logic:** No per-module inspection, uniform skip behavior

**Lines of Code:** 23 lines (before) → 19 lines (after), ~4 lines removed

---

### 5.3 Additional Conftest Changes

**Delete `torch_available` fixture (lines 84-90):**

```diff
-@pytest.fixture(scope="session")
-def torch_available():
-    """Fixture to check if PyTorch is available."""
-    try:
-        import torch
-        return True
-    except ImportError:
-        return False
```

**Rationale:** Zero consumers; fixture obsolete in torch-required environment.

**Impact:** None (no tests use it)

---

## 6. Test Suite Impact Matrix

| Test Category | Total Tests | Currently Skip (No Torch) | Will Fail (Torch-Required, No Torch) | Require Code Changes |
|:--------------|:------------|:--------------------------|:-------------------------------------|:---------------------|
| Whitelisted torch/ tests | ~68 | 0 | ~68 | YES (remove fallbacks in production modules) |
| Non-whitelisted torch/ tests | ~6 | ~6 | ~6 | NO (already fail to import) |
| TensorFlow-only tests (tests/) | ~100 | 0 | 0 | NO |
| **TOTAL** | **~174** | **~6** | **~74** | **8 production + 4 test files** |

**Key Insight:** Torch-required transition exposes 68 additional test failures (whitelisted tests) when PyTorch unavailable. This is **intentional**—tests should fail (not skip) to prevent silent production breakage.

---

### 6.1 CI Runner Matrix (Post-Migration)

| CI Job | PyTorch Installed? | tests/torch/ Behavior | Expected Pass Rate |
|:-------|:-------------------|:----------------------|:-------------------|
| **Ptychodus Integration CI** | YES (required) | EXECUTE, expect 0 FAILED | 174/174 PASSED |
| **Ptychodus Integration CI** | NO (misconfigured) | FAIL (import errors) | 74 FAILED, 100 PASSED |
| **TF-only CI** | NO (intentional) | SKIP (directory-based) | 100 PASSED, 74 SKIPPED |

**Governance Decision (governance_decision.md Q1):** TF-only CI runners continue to exist; torch tests skip at directory level (no whitelist exceptions).

---

## 7. Rollback Safety

### 7.1 Reversion Steps

**If Phase F3.3 conftest changes cause unexpected failures:**

1. **Git revert conftest.py changes:**
   ```bash
   git checkout HEAD~1 -- tests/conftest.py
   ```

2. **Restore whitelist (if partially migrated):**
   ```python
   TORCH_OPTIONAL_MODULES = ["test_config_bridge", "test_data_pipeline", "test_workflows_components", "test_model_manager", "test_backend_selection"]
   ```

3. **Revert production module guards (Phase F3.2 rollback):**
   - See `torch_optional_inventory.md` §7 for file-by-file reversion

4. **Document blocker in docs/fix_plan.md:**
   - Add new attempt entry describing failure mode
   - Tag as BLOCKED pending resolution

**Rollback Time Estimate:** ~10 minutes (git revert + validation run)

---

### 7.2 Test Validation Commands

**Verify torch-required behavior (PyTorch installed):**

```bash
# All torch tests execute
pytest tests/torch/ -v

# Expected: ~74 tests PASSED, 0 SKIPPED, 0 FAILED
```

**Verify TF-only CI behavior (PyTorch NOT installed):**

```bash
# Simulate torch unavailability
pip uninstall torch -y

# Run full suite
pytest tests/ -v

# Expected: ~100 tests PASSED, ~74 SKIPPED (all in tests/torch/), 0 FAILED
```

**Verify conftest skip logic directly:**

```bash
# Dry-run collection (no execution)
pytest --collect-only tests/torch/ -v

# Expected (no torch): 74 tests collected, all marked SKIPPED
# Expected (with torch): 74 tests collected, 0 marked SKIPPED
```

---

## 8. Downstream Impact

### 8.1 CI/CD Runners

**Current Assumption:** CI runners MAY lack PyTorch (evidenced by conftest skip logic)

**Torch-Required Requirement:** Ptychodus integration CI MUST have PyTorch installed

**Action Required (Phase F3.1 BLOCKING):**

1. **Update CI configuration:**
   - Add `pip install torch>=2.2` to test setup step
   - Validate installation: `pytest --collect-only tests/torch/` (expect 0 skipped)

2. **TF-only CI (separate job):**
   - Continue without PyTorch
   - tests/torch/ skipped at directory level (conftest preserves this)

3. **Environment variables (optional):**
   - `PYTORCH_REQUIRED=1` flag to assert torch installation in Ptychodus CI
   - Fail fast in conftest if flag set but torch unavailable

**Example CI YAML (GitHub Actions):**

```yaml
# Ptychodus Integration CI (torch-required)
test-ptychodus:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Install dependencies
      run: |
        pip install -e .
        pip install torch>=2.2 torchvision>=0.17
    - name: Validate PyTorch
      run: pytest --collect-only tests/torch/ -q
    - name: Run tests
      run: pytest tests/ -v

# TensorFlow-only CI (torch-optional)
test-tensorflow:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Install dependencies
      run: pip install -e .
    - name: Run tests (torch/ skipped)
      run: pytest tests/ -v
```

---

### 8.2 Developer Environments

**Current State:** Developers can work on TensorFlow-only features without PyTorch

**Torch-Required State:** PyTorch becomes mandatory for full test suite execution

**Updated Developer Workflow:**

1. **Installation (README.md update required):**
   ```bash
   # Full installation (torch-required)
   pip install -e .
   pip install torch>=2.2
   ```

2. **Test execution expectations:**
   - `pytest tests/` → Requires PyTorch (tests/torch/ will fail without it)
   - `pytest tests/ --ignore=tests/torch/` → TensorFlow-only validation (workaround)

3. **Pre-commit hooks (optional):**
   - Add PyTorch availability check: `python -c "import torch"`
   - Fail if torch unavailable with actionable error message

**Mitigation (CLAUDE.md §5 update):**
- Document PyTorch as mandatory in "Key Commands" section
- Provide one-line install command
- Clarify TF-only workflow (use --ignore flag for torch/)

---

## 9. Open Questions

### Q1: Should we preserve `@pytest.mark.torch` marker functionality?

**Options:**
- **A (Recommended):** Keep marker registered but informational-only (no behavioral change)
- **B:** Remove marker registration entirely (cleanup)

**Decision:** Option A (preserve for future explicit marking if needed)

---

### Q2: Should conftest raise an ERROR (not skip) when torch unavailable in Ptychodus CI?

**Current behavior:** Skip tests/torch/ when torch unavailable
**Alternative:** Raise `RuntimeError` if environment variable `PYTORCH_REQUIRED=1` set but torch unavailable

**Pros:** Prevents silent CI misconfiguration (tests skip instead of fail)
**Cons:** Adds environment variable dependency; complicates rollback

**Decision:** DEFER to Phase F3.4 (validate green test suite first, then add env var assertion if needed)

---

### Q3: What happens to `tests/test_pytorch_tf_wrapper.py`?

**Current status:** Whitelisted test with torch-optional guards
**Relevance:** UNCLEAR (appears to be legacy wrapper; may be obsolete)

**Action Required:** AUDIT in Phase F3.3
- If still relevant: Remove guards, expect torch available
- If obsolete: DELETE file entirely

**Decision:** Add audit task to Phase F3.3 checklist

---

### Q4: Should we add a CI job that INTENTIONALLY runs without torch to verify skip behavior?

**Rationale:** Validate TF-only CI skip logic continues to work correctly

**Recommendation:** YES (add to Phase F3.4 validation)

**Example:**
```bash
# Simulate torch unavailability
pip uninstall torch -y

# Expect: tests/torch/ skipped, TF tests passed
pytest tests/ -v

# Validate skip count
pytest tests/ --collect-only -q | grep "74 skipped"
```

---

## 10. Phase F3.3 Implementation Checklist

Based on this audit, Phase F3.3 must:

- [ ] **Conftest:** Remove `TORCH_OPTIONAL_MODULES` whitelist (line 42)
- [ ] **Conftest:** Simplify `pytest_collection_modifyitems` skip logic (lines 32-47)
- [ ] **Conftest:** Update skip reason to "(TF-only CI)" for clarity
- [ ] **Conftest:** Delete `torch_available` fixture (lines 84-90)
- [ ] **Test TF Helper:** Remove `@unittest.skipUnless(TORCH_AVAILABLE, ...)` decorators (lines 68, 75)
- [ ] **Test TF Helper:** Remove try/except import guard (lines 11-13)
- [ ] **Test TF Helper:** Remove runtime skip check (line 87)
- [ ] **Test Data Pipeline:** Remove try/except import guard (lines 25-27)
- [ ] **Test Data Pipeline:** Remove conditional assertions (lines 308, 320, 394)
- [ ] **Test PyTorch Wrapper:** AUDIT relevance → remove guards OR delete file
- [ ] **Validation:** Run `pytest tests/torch/ -v` (expect 74 PASSED, 0 SKIPPED)
- [ ] **Validation:** Simulate torch unavailable, run `pytest tests/ -v` (expect 74 SKIPPED, 100 PASSED)

**Artifacts:** `pytest_update.md` with before/after conftest comparison + validation logs

---

## 11. Success Metrics

**Conftest Simplification:**
- Whitelist removed: 0 exceptions
- Skip logic: directory-based only
- Lines of code: ~4 lines removed

**Test Suite Behavior:**
- Ptychodus CI (torch installed): 174 PASSED, 0 SKIPPED, 0 FAILED
- TF-only CI (no torch): 100 PASSED, 74 SKIPPED, 0 FAILED
- Ptychodus CI (misconfigured, no torch): 74 FAILED (ImportError), 100 PASSED

**Developer Experience:**
- Clear error messages when torch unavailable
- README documents torch installation
- --ignore=tests/torch/ workaround for TF-only work

---

**Audit Complete:** All pytest skip logic documented with behavioral transition matrix. Ready for Phase F3.3 execution.
