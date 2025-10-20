# Test Infrastructure Strategy
**Initiative:** <INITIATIVE-NAME>
**Created:** <YYYY-MM-DD>
**Status:** <DRAFT | APPROVED | IMPLEMENTED>
**Reviewer:** <Supervisor/Lead>

---

## Purpose

This document defines the test infrastructure design for <INITIATIVE-NAME> before implementation begins. It ensures test framework compatibility, CI constraint handling, and execution proof requirements are planned upfront.

**Why This Matters:**
- Prevents framework incompatibility (unittest + pytest.parametrize)
- Addresses CI constraints proactively (PyTorch availability, GPU access)
- Defines execution proof requirements (PASSED vs SKIPPED)
- Avoids test refactoring iterations

---

## 1. Framework Selection & Compatibility

### 1.1 Test Framework Decision

**Primary Framework:**
- [ ] pytest (recommended for new tests)
- [ ] unittest (legacy compatibility only)
- [ ] Other: ___________

**Rationale:**
<!-- Example: pytest chosen for parametrization support, fixture flexibility, and CI compatibility -->

### 1.2 Compatibility Checklist

**Framework Mixing:**
- [ ] ✅ All new tests use pure pytest (no unittest.TestCase)
- [ ] ✅ pytest.mark.parametrize compatible (no unittest.TestCase base)
- [ ] ✅ Fixtures used instead of setUp/tearDown
- [ ] ⚠️ Legacy tests remain unittest (document migration plan)

**Parametrization Needs:**
- [ ] N/A - no parametrization needed
- [ ] Simple - single parameter per test
- [ ] Complex - multiple parameters, field matrix
- [ ] Strategy: ___________ (e.g., @pytest.mark.parametrize, pytest.fixture, etc.)

**Test Discovery:**
- [ ] pytest auto-discovery (test_*.py, *_test.py)
- [ ] unittest discovery (test*.py)
- [ ] Custom test runner: ___________

### 1.3 Framework Validation

**Proof Required:**
- [ ] Sample test written in chosen framework
- [ ] Parametrization tested (if applicable)
- [ ] Fixtures validated
- [ ] No framework warnings/errors in test run

**Command:**
```bash
pytest tests/<module>/test_sample.py -vv
# Expected: PASSED (not SKIPPED, not TypeError)
```

---

## 2. CI/CD Constraint Analysis

### 2.1 Environment Inventory

**Development Environment:**
```
Python: <version>
Frameworks: torch <version>, tensorflow <version>, etc.
Hardware: CPU / GPU / TPU
OS: Linux / macOS / Windows
```

**CI Environment:**
```
Python: <version>
Frameworks: <list available> (NOTE: torch may be UNAVAILABLE)
Hardware: CPU-only / GPU / TPU
OS: <typically Linux>
```

### 2.2 Constraint Matrix

| Resource | Dev | CI | Gap? | Mitigation |
|----------|-----|----|----|------------|
| PyTorch | ✅ v2.0 | ❌ Not installed | YES | Torch-optional harness |
| TensorFlow | ✅ v2.15 | ✅ v2.15 | NO | N/A |
| GPU | ✅ CUDA 12 | ❌ CPU-only | YES | Skip GPU tests in CI |
| Test Data | ✅ 10GB | ⚠️ 100MB limit | YES | Use small test fixtures |

### 2.3 Optional Dependency Strategy

**For unavailable frameworks (e.g., PyTorch in CI):**

**Approach:**
- [ ] Torch-Optional Harness (recommended)
  - Optional imports with try/except
  - Tests RUN with stubs when torch unavailable
  - Graceful degradation for type checking

- [ ] Skip Tests (acceptable for hardware constraints)
  - @pytest.mark.skipif(not torch_available)
  - Document why skipping is acceptable
  - Ensure local validation before merge

- [ ] Mock/Stub (for external services)
  - unittest.mock for API calls
  - Fixtures for external data

**Implementation Example:**
```python
# For torch-optional harness
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define stubs or skip

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_with_torch():
    ...

# OR use torch-optional pattern
def test_with_stubs():
    if TORCH_AVAILABLE:
        # Real test with torch
    else:
        # Stub test validating behavior without torch
```

### 2.4 Skip Policy

**Acceptable Reasons to SKIP:**
- ✅ Hardware unavailable (GPU tests on CPU-only CI)
- ✅ External service unavailable (integration tests)
- ✅ Long-running tests (performance benchmarks with @pytest.mark.slow)
- ✅ Platform-specific (Windows-only tests on Linux CI)

**UNACCEPTABLE Reasons:**
- ❌ Missing optional dependency (use torch-optional pattern)
- ❌ Framework incompatibility (design error, fix before merging)
- ❌ "Tests don't work in CI" (fix CI or test design)

---

## 3. Test Tier Definitions

### 3.1 Test Pyramid

```
        /\
       /E2E\      End-to-End (5%)
      /------\
     /Integration\ Integration (15%)
    /------------\
   /    Unit      \ Unit (80%)
  /----------------\
```

### 3.2 Unit Tests

**Scope:** Individual functions, classes, modules
**Coverage Goal:** 80%+ for new code
**Execution Time:** <5 seconds total
**Dependencies:** Minimal (mocks/stubs for external)

**Examples:**
- `test_config_bridge_field_translation.py` - Individual field conversions
- `test_adapter_type_conversions.py` - Type system translations
- `test_validation_logic.py` - Input validation rules

**Framework:** pytest with parametrization
**Fixtures:** Lightweight config objects, sample data

### 3.3 Integration Tests

**Scope:** Multiple modules working together
**Coverage Goal:** Critical paths covered
**Execution Time:** <30 seconds total
**Dependencies:** Real objects, minimal mocking

**Examples:**
- `test_config_pipeline_integration.py` - PyTorch config → adapter → TF dataclass → params.cfg
- `test_torch_optional_integration.py` - Full harness with/without torch
- `test_key_mappings_integration.py` - Adapter + update_legacy_dict

**Framework:** pytest
**Fixtures:** Full config objects, integration harness

### 3.4 End-to-End Tests (if applicable)

**Scope:** Full system workflow
**Coverage Goal:** Happy path + critical error paths
**Execution Time:** <5 minutes
**Dependencies:** Real components, real data (small samples)

**Examples:**
- `test_full_training_pipeline.py` - Config → data loading → model init → training step
- (Not needed for config bridge MVP, plan for Phase C/D)

**Framework:** pytest with @pytest.mark.e2e
**Fixtures:** Integration environment, sample datasets

### 3.5 Smoke Tests

**Scope:** Quick validation that nothing is broken
**Coverage Goal:** Core functionality only
**Execution Time:** <10 seconds
**Dependencies:** Minimal

**Examples:**
- `test_imports_smoke.py` - All modules import without error
- `test_config_bridge_smoke.py` - MVP fields translate successfully

**Framework:** pytest
**Run Frequency:** Every commit (pre-commit hook)

---

## 4. Execution Proof Requirements

### 4.1 Quality Gate Definition

**A test is NOT validated if:**
- ❌ Status is SKIPPED without justification
- ❌ No pytest execution log in artifacts
- ❌ Test only validates imports (no assertions)
- ❌ Test passes due to mocking everything

**A test IS validated if:**
- ✅ Status is PASSED (assertions executed)
- ✅ pytest.log shows test execution details
- ✅ Meaningful assertions validated
- ✅ SKIPs are justified and documented

### 4.2 Artifact Requirements

**Every test run MUST produce:**

```
plans/active/<initiative>/reports/<timestamp>/
├── pytest.log           # REQUIRED: Full pytest -vv output
├── summary.md           # REQUIRED: Test execution summary
├── coverage.xml         # Optional: Coverage report
└── [other artifacts]    # Test-specific outputs
```

**pytest.log format:**
```bash
# Generate with:
pytest <path> -vv 2>&1 | tee pytest.log

# Must show:
- Test discovery (X items collected)
- Test execution (PASSED/FAILED/SKIPPED with reasons)
- Summary statistics
- No framework errors
```

**summary.md format:**
```markdown
# Test Execution Summary

**Date:** YYYY-MM-DD HH:MM
**Command:** pytest tests/torch/test_config_bridge.py -vv
**Duration:** X seconds

## Results
- Passed: X
- Failed: X
- Skipped: X (with justifications)
- Coverage: X%

## Skip Justifications
- test_gpu_optimization: GPU unavailable in CI (acceptable)
- test_torch_tensor_conversion: PyTorch not installed (NOT acceptable - fix with torch-optional)

## Issues Found
[List any test failures, warnings, or concerns]

## Next Steps
[Action items based on results]
```

### 4.3 Validation Commands

**Run tests and capture proof:**
```bash
# Unit tests
pytest tests/unit/ -vv --cov=ptycho_torch 2>&1 | tee reports/<timestamp>/pytest_unit.log

# Integration tests
pytest tests/integration/ -vv 2>&1 | tee reports/<timestamp>/pytest_integration.log

# Specific test selector
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_field -vv 2>&1 | tee reports/<timestamp>/pytest_parity.log

# Smoke test
pytest tests/ -k smoke -vv 2>&1 | tee reports/<timestamp>/pytest_smoke.log
```

---

## 5. Mock/Stub Strategy

### 5.1 What to Mock

**External Services:**
- [ ] API calls (use responses or httpretty)
- [ ] Database connections (use SQLite in-memory)
- [ ] File system operations (use tmp_path fixture)

**Expensive Operations:**
- [ ] Model training (mock with simple forward pass)
- [ ] Large data loading (use small fixtures)
- [ ] GPU operations (use CPU tensors)

**Unavailable Resources:**
- [ ] PyTorch (torch-optional harness, not mocking)
- [ ] GPU (skip tests, document in CI)

### 5.2 What NOT to Mock

**Core Logic Under Test:**
- ❌ Don't mock the config bridge (test the real thing)
- ❌ Don't mock type conversions (test real behavior)
- ❌ Don't mock validation logic (test real checks)

**Simple Dependencies:**
- ❌ Don't mock dataclasses (use real instances)
- ❌ Don't mock Python stdlib (os, pathlib, etc.)

### 5.3 Mocking Patterns

**Acceptable:**
```python
# Mock external API
@pytest.fixture
def mock_api(monkeypatch):
    monkeypatch.setattr('module.api_call', lambda x: {'status': 'ok'})
    return {'status': 'ok'}
```

**NOT Acceptable:**
```python
# Don't mock the thing you're testing!
def test_config_bridge():
    with mock.patch('ptycho_torch.config_bridge.to_model_config'):
        # This doesn't test anything!
        result = to_model_config(...)
```

---

## 6. Test Data Strategy

### 6.1 Fixtures

**Location:** `tests/fixtures/` or `tests/conftest.py`

**Types:**
- **Minimal:** Single field, smallest valid input
- **Typical:** Realistic values, common use case
- **Edge Cases:** Boundary values, unusual inputs
- **Invalid:** For error handling tests

**Example:**
```python
# tests/conftest.py
@pytest.fixture
def minimal_pytorch_config():
    """Minimal valid PyTorch config for testing."""
    return ModelConfig(
        grid_size=(64, 64),
        mode=TrainingMode.Supervised,
        # ... minimal fields only
    )

@pytest.fixture
def typical_pytorch_config():
    """Realistic PyTorch config with common values."""
    return ModelConfig(
        grid_size=(128, 128),
        mode=TrainingMode.Supervised,
        n_filters_scale=4,
        # ... typical production values
    )
```

### 6.2 Test Data Files

**Small Fixtures (<1MB):** Include in repo
- `tests/data/sample_config.json`
- `tests/data/minimal_dataset.npz`

**Large Fixtures (>1MB):** Download on demand
- Use pytest-datafiles or manual download
- Document in `tests/README.md`
- CI may skip if unavailable

---

## 7. Coverage Requirements

### 7.1 Coverage Targets

**Minimum Acceptable:**
- New code: 80% line coverage
- Critical paths: 100% branch coverage
- MVP scope: 90% coverage

**Measure With:**
```bash
pytest --cov=ptycho_torch --cov-report=html --cov-report=term
```

### 7.2 Coverage Exemptions

**Acceptable to Skip:**
- Debug code (if __debug__:)
- Type stubs (if TYPE_CHECKING:)
- Abstract base classes (ABC methods)
- Deprecated code (marked for removal)

**NOT Acceptable:**
- "Too hard to test" (refactor for testability)
- "Works in production" (write the test)

---

## 8. Continuous Integration

### 8.1 CI Pipeline

```yaml
# .github/workflows/test.yml (example)
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ -vv --cov=ptycho_torch --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 8.2 Pre-commit Hooks

**Smoke Tests:**
```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-smoke
      name: pytest-smoke
      entry: pytest tests/ -k smoke -vv
      language: system
      pass_filenames: false
      always_run: true
```

---

## 9. Approval Checklist

**Before approving this strategy:**

- [ ] Framework compatibility validated (no unittest + pytest.parametrize)
- [ ] CI constraints documented and mitigations planned
- [ ] Test tiers defined (unit/integration/e2e)
- [ ] Execution proof requirements clear (PASSED vs SKIPPED)
- [ ] Mock strategy defined (what to mock, what not to)
- [ ] Coverage targets set and achievable
- [ ] Artifact requirements documented
- [ ] Sample test written and validated

**Approved By:** _____________
**Date:** _____________

---

## 10. References

**Project Docs:**
- `docs/TESTING_GUIDE.md` - Testing methodology
- `docs/DEVELOPER_GUIDE.md` - Development workflow
- `specs/data_contracts.md` - Data format requirements

**Related Plans:**
- `plans/active/<initiative>/implementation.md` - References this strategy in Phase B
- `plans/active/<initiative>/constraint_analysis.md` - Environment constraints

**Templates:**
- This document serves as template for future initiatives

---

**Last Updated:** <YYYY-MM-DD>
**Next Review:** After Phase B completion
