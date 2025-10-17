# Torch-Optional Test Blueprint (Phase C.B1)

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** C.B — Torch-Optional Test Harness (Red Phase)
**Date:** 2025-10-17
**Purpose:** Define pytest structure for RawDataTorch and PtychoDataContainerTorch parity tests.

---

## 1. Torch-Optional Test Patterns

The project uses **auto-skip by directory** as the primary torch-optional pattern, with whitelist exceptions for adapter/bridge tests.

### Pattern Hierarchy (Use in Order)

#### Pattern A: Auto-Skip by Directory (Primary)
**Source:** `tests/conftest.py:25-46`

```python
def pytest_collection_modifyitems(config, items):
    """Skip tests in tests/torch/ when PyTorch unavailable."""
    for item in items:
        if "tests/torch/" in str(item.fspath):
            # Whitelist exceptions (e.g., test_config_bridge.py runs without torch)
            if item.fspath.basename not in TORCH_OPTIONAL_MODULES:
                item.add_marker(pytest.mark.skip(...))
```

**Usage:**
- Place test in `tests/torch/test_data_pipeline.py`
- Auto-skipped when PyTorch unavailable
- No per-test boilerplate required

**When to Use:** All tests that require torch tensors or PyTorch-specific behavior.

#### Pattern B: Whitelist Exception (Bridge/Adapter Tests)
**Source:** `tests/conftest.py:38-46`

```python
TORCH_OPTIONAL_MODULES = [
    "test_config_bridge.py",  # Runs without torch (fallback mode)
    "test_data_pipeline.py",  # NEW: Add for RawDataTorch adapter tests
]
```

**Usage:**
- Add module to `TORCH_OPTIONAL_MODULES` in `tests/conftest.py`
- Tests run even when PyTorch unavailable (using fallback logic)
- Import guards required inside test functions

**When to Use:** Adapter tests that validate torch-agnostic wrappers.

#### Pattern C: Import Guard (Within Whitelist Modules)
**Source:** `tests/torch/test_config_bridge.py:1-20`

```python
# Top of module (torch-optional modules only)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Inside test
def test_with_torch_fallback(self):
    """Test runs with or without torch."""
    if TORCH_AVAILABLE:
        import torch
        tensor = torch.randn(10, 10)
    else:
        # Fallback to NumPy
        tensor = np.random.randn(10, 10)
    # Assertions work on both
```

**When to Use:** Tests validating adapters that delegate to NumPy when torch unavailable.

---

## 2. Fixture Strategy

### 2.1 Essential Fixtures

**Required for All Data Tests:**

```python
@pytest.fixture
def params_cfg_snapshot():
    """Save/restore params.cfg state (CRITICAL)."""
    import ptycho.params as params
    snapshot = dict(params.cfg)
    yield
    params.cfg.clear()
    params.cfg.update(snapshot)
```

**Source:** `tests/torch/test_config_bridge.py:151-160`

**Why Critical:** Every data test MUST preserve `params.cfg` state because:
1. `RawData.generate_grouped_data()` reads `params.cfg['gridsize']`
2. Failure to restore causes test pollution (CONFIG-001)
3. Subsequent tests inherit corrupted gridsize values

### 2.2 Synthetic RawData Fixture

**Recommended Pattern:**

```python
@pytest.fixture
def minimal_raw_data(params_cfg_snapshot):
    """Create synthetic RawData for testing."""
    from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
    from ptycho import params as p
    from ptycho.raw_data import RawData
    import numpy as np

    # 1. Initialize params.cfg (MANDATORY before RawData)
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        n_groups=64,
        neighbor_count=4
    )
    update_legacy_dict(p.cfg, config)

    # 2. Create deterministic synthetic data
    n_points = 100
    x = np.linspace(0, 10, int(np.sqrt(n_points)))
    y = np.linspace(0, 10, int(np.sqrt(n_points)))
    xv, yv = np.meshgrid(x, y)
    xcoords = xv.flatten()[:n_points].astype(np.float64)
    ycoords = yv.flatten()[:n_points].astype(np.float64)

    np.random.seed(42)
    diff3d = np.random.rand(n_points, 64, 64).astype(np.float32)
    probe = np.ones((64, 64), dtype=np.complex64)
    obj = np.ones((128, 128), dtype=np.complex64)

    return RawData(xcoords, ycoords, diff3d, probe, obj, scan_index=None)
```

**Source Pattern:** `tests/test_coordinate_grouping.py` setUp() methods

**Advantages:**
- ✅ Deterministic (seed=42)
- ✅ Fast (<100ms creation time)
- ✅ Respects data contracts (dtype, normalization)
- ✅ No I/O dependencies

### 2.3 Real Dataset Fixture

**Status:** `datasets/fly/fly001_transposed.npz` **DOES NOT EXIST**

**Alternative:** Use existing integration test dataset

```python
@pytest.fixture(scope="session")
def integration_dataset():
    """Real dataset for integration tests (slow)."""
    path = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")
    if not path.exists():
        pytest.skip("Integration dataset not available")
    return path
```

**When to Use:** Integration tests only (slow, ~30s load time).

---

## 3. Recommended Test Structure

### 3.1 Module Organization

```
tests/
├── conftest.py                      # Global fixtures + torch skip logic
├── test_data_pipeline.py            # Unit tests (no torch required)
│   ├── TestRawDataCreation
│   ├── TestGroupingAlgorithm
│   └── TestNormalization
├── torch/
│   ├── test_config_bridge.py        # Config adapter (existing, torch-optional)
│   └── test_data_pipeline.py        # NEW: Data pipeline parity (torch-optional)
│       ├── TestRawDataTorchAdapter  # Wrapper delegation
│       ├── TestDataContainerParity  # TensorFlow vs PyTorch tensor comparison
│       └── TestGroundTruthLoading   # Y patch validation
└── test_integration_pytorch.py      # END-TO-END (future, requires torch)
```

### 3.2 Naming Conventions

| Test Type | Module Location | Naming Pattern | Example |
|-----------|----------------|----------------|---------|
| Unit (no torch) | `tests/` | `test_*.py` | `test_data_pipeline.py` |
| Parity (torch-optional) | `tests/torch/` | `test_*_parity.py` or `test_*_bridge.py` | `test_data_pipeline.py` |
| Integration (requires torch) | `tests/` | `test_integration_*.py` | `test_integration_pytorch.py` |

**File Naming Rule:** Use `test_data_pipeline.py` in **both** `tests/` and `tests/torch/`:
- `tests/test_data_pipeline.py` — Unit tests (NumPy-only, no torch)
- `tests/torch/test_data_pipeline.py` — Parity tests (torch-optional adapters)

### 3.3 Parametrization Pattern

**Source:** `tests/torch/test_config_bridge.py:163-435`

```python
@pytest.mark.parametrize('gridsize,expected_channels', [
    pytest.param(1, 1, id='gridsize-1'),
    pytest.param(2, 4, id='gridsize-2'),
    pytest.param(3, 9, id='gridsize-3'),
])
def test_grouping_parity(self, params_cfg_snapshot, minimal_raw_data,
                         gridsize, expected_channels):
    """Test RawData grouping produces correct channel counts."""
    # Update config with test gridsize
    from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
    from ptycho import params as p

    config = TrainingConfig(model=ModelConfig(N=64, gridsize=gridsize))
    update_legacy_dict(p.cfg, config)

    # Generate grouped data
    grouped = minimal_raw_data.generate_grouped_data(
        N=64, K=4, nsamples=10, gridsize=gridsize
    )

    # Validate shape
    assert grouped['diffraction'].shape == (10, 64, 64, expected_channels)
```

**Advantages:**
- ✅ Single test covers multiple gridsize values
- ✅ Clear test IDs for debugging (`-k gridsize-2`)
- ✅ Reuses fixture across parametrized cases

---

## 4. Minimal ROI for Fast Tests

| Parameter | Unit Test | Integration | Rationale |
|-----------|-----------|-------------|-----------|
| N (grid size) | 32-64 | 64 | Smaller = faster forward pass |
| gridsize | 1-2 | 2 | Standard is 2×2 patches |
| n_groups (nsamples) | 64-128 | 512 | Fewer = faster grouping |
| n_points | 10-100 | 200-1000 | Minimal for K-NN (K=4) |
| batch_size | 4 | 16-32 | CPU limit ~4 for unit tests |

**Expected Runtimes:**
- **Unit test (synthetic):** <100ms
- **Integration (1 epoch):** <30s CPU

**Validation:**
```python
def test_unit_test_performance():
    """Ensure unit tests complete quickly."""
    import time
    start = time.time()

    # Run unit test logic
    fixture = minimal_raw_data()
    grouped = fixture.generate_grouped_data(N=64, K=4, nsamples=64)

    elapsed = time.time() - start
    assert elapsed < 0.1, f"Unit test too slow: {elapsed:.2f}s"
```

---

## 5. Critical Data Contract Checks

**Source:** `specs/data_contracts.md:23-70`

### 5.1 Amplitude vs Intensity

```python
def test_diffraction_is_amplitude_not_intensity(self, minimal_raw_data):
    """Validate diffraction contains amplitude per data contract."""
    grouped = minimal_raw_data.generate_grouped_data(N=64, K=4, nsamples=10)
    diffraction = grouped['diffraction']

    # Amplitude has smaller dynamic range than intensity
    ratio = np.max(diffraction) / np.mean(diffraction)
    assert ratio < 100, f"May be intensity instead of amplitude (ratio={ratio})"

    # Amplitude should be normalized
    assert np.max(diffraction) < 10.0, "Data appears unnormalized"
    assert np.min(diffraction) >= 0.0, "Amplitude should be non-negative"
```

### 5.2 Dtype Validation

```python
def test_dtypes_match_contract(self, minimal_raw_data):
    """Validate dtypes per specs/data_contracts.md."""
    grouped = minimal_raw_data.generate_grouped_data(N=64, K=4, nsamples=10)

    assert grouped['diffraction'].dtype == np.float32, "diffraction must be float32"
    assert grouped['X_full'].dtype == np.float32, "X_full must be float32"

    if grouped['Y'] is not None:
        assert grouped['Y'].dtype == np.complex64, "Y must be complex64 (not float64)"
```

### 5.3 Shape Validation

```python
@pytest.mark.parametrize('gridsize,C', [(1, 1), (2, 4), (3, 9)])
def test_shapes_match_contract(self, params_cfg_snapshot, minimal_raw_data, gridsize, C):
    """Validate shapes per data_contract.md."""
    # ... initialization ...

    grouped = minimal_raw_data.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=gridsize)

    assert grouped['diffraction'].shape == (10, 64, 64, C)
    assert grouped['coords_offsets'].shape == (10, 1, 2, 1)
    assert grouped['coords_relative'].shape == (10, 1, 2, C)
    assert grouped['nn_indices'].shape == (10, C)
```

---

## 6. Torch-Optional Adapter Test Template

**New File:** `tests/torch/test_data_pipeline.py`

```python
"""
Torch-optional parity tests for RawDataTorch and PtychoDataContainerTorch adapters.

This module is whitelisted in conftest.py to run without PyTorch installed,
validating that adapters correctly delegate to TensorFlow RawData when torch unavailable.
"""

import pytest
import numpy as np
from pathlib import Path

# Torch-optional import guard
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestRawDataTorchAdapter:
    """Test RawDataTorch wrapper delegates correctly to TensorFlow RawData."""

    def test_wrapper_delegates_to_tensorflow(self, params_cfg_snapshot, minimal_raw_data):
        """RawDataTorch should produce identical outputs to RawData."""
        # This test runs without torch (validates delegation)
        from ptycho.raw_data import RawData

        # Create TensorFlow baseline
        tf_grouped = minimal_raw_data.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)

        # TODO (Phase C.C1): Import and test RawDataTorch
        # from ptycho_torch.raw_data_bridge import RawDataTorch
        # pt_grouped = RawDataTorch(...).generate_grouped_data(...)
        # np.testing.assert_array_equal(tf_grouped['nn_indices'], pt_grouped['nn_indices'])

        pytest.skip("RawDataTorch not yet implemented (Phase C.C1)")


    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_outputs_convert_to_torch_tensors(self, params_cfg_snapshot, minimal_raw_data):
        """RawDataTorch outputs should be torch.Tensor when torch available."""
        pytest.skip("RawDataTorch not yet implemented (Phase C.C1)")


class TestDataContainerParity:
    """Test PtychoDataContainerTorch matches TensorFlow PtychoDataContainer API."""

    def test_container_has_required_attributes(self):
        """Container must expose X, Y, coords_nominal, probe, etc."""
        pytest.skip("PtychoDataContainerTorch not yet implemented (Phase C.C2)")

    def test_tensor_shapes_match_tensorflow(self, params_cfg_snapshot, minimal_raw_data):
        """Tensor shapes must match TensorFlow contract."""
        pytest.skip("PtychoDataContainerTorch not yet implemented (Phase C.C2)")


class TestGroundTruthLoading:
    """Test Y patch loading and dtype validation."""

    def test_y_patches_are_complex64(self, params_cfg_snapshot):
        """Y patches MUST be complex64 per DATA-001 finding."""
        pytest.skip("Ground truth loading not yet implemented (Phase C.C3)")

    def test_y_patches_extracted_via_nn_indices(self, params_cfg_snapshot):
        """Y patches should be extracted using nn_indices."""
        pytest.skip("Ground truth loading not yet implemented (Phase C.C3)")
```

**Whitelist Registration:** Add to `tests/conftest.py`:

```python
TORCH_OPTIONAL_MODULES = [
    "test_config_bridge.py",
    "test_data_pipeline.py",  # NEW: torch-optional data adapter tests
]
```

---

## 7. Test Execution Commands

### Run Targeted Selectors

```bash
# Run only RawDataTorch adapter tests (torch-optional)
pytest tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter -vv

# Run data container parity tests
pytest tests/torch/test_data_pipeline.py::TestDataContainerParity -vv

# Run all parity tests (requires torch)
pytest tests/torch/test_data_pipeline.py -vv

# Run parity tests WITHOUT torch (validate fallback)
pytest tests/torch/test_data_pipeline.py --ignore-torch -vv
```

### Show Test Collection

```bash
# Dry run to see which tests will execute
pytest tests/torch/test_data_pipeline.py --collect-only

# Check torch skip behavior
pytest tests/torch/ --collect-only -v
```

### Performance Validation

```bash
# Ensure unit tests complete quickly (<1s)
pytest tests/test_data_pipeline.py -vv --durations=10
```

---

## 8. Critical Gotchas

### Gotcha 1: params.cfg Initialization (The #1 Bug)
**Source:** `docs/debugging/QUICK_REFERENCE_PARAMS.md`, `docs/findings.md:CONFIG-001`

**Problem:** Silent shape mismatch if `update_legacy_dict()` not called before `RawData`

**Solution:** ALWAYS use `params_cfg_snapshot` fixture and initialize params before data operations:

```python
def test_with_raw_data(self, params_cfg_snapshot):  # ← Fixture REQUIRED
    from ptycho.config.config import update_legacy_dict, TrainingConfig, ModelConfig
    from ptycho import params as p

    config = TrainingConfig(model=ModelConfig(gridsize=2, N=64))
    update_legacy_dict(p.cfg, config)  # ← MANDATORY

    # Now safe to use RawData
    from ptycho.raw_data import RawData
```

### Gotcha 2: Mixed unittest/pytest
**Source:** `CLAUDE.md:102-105`, `tests/torch/test_config_bridge.py` (refactored)

**Problem:** `unittest.TestCase` + `@pytest.mark.parametrize` causes TypeError

**Solution:** Use native pytest style (plain functions or pytest-managed classes):

```python
# ❌ WRONG
class TestParity(unittest.TestCase):
    @pytest.mark.parametrize('value', [1, 2, 3])
    def test_something(self, value):
        pass

# ✅ CORRECT
class TestParity:  # No unittest.TestCase inheritance
    @pytest.mark.parametrize('value', [1, 2, 3])
    def test_something(self, value):
        pass
```

### Gotcha 3: Data Contract Violations
**Source:** `specs/data_contracts.md:23-70`, `docs/findings.md:DATA-001`

**Problem:** Silent bugs from dtype/normalization violations

**Solution:** Validate in every test:

```python
def test_data_contract_compliance(self, grouped_data):
    """Validate specs/data_contracts.md requirements."""
    assert grouped_data['diffraction'].dtype == np.float32
    assert grouped_data['Y'].dtype == np.complex64  # NOT float64
    assert np.max(grouped_data['diffraction']) < 10.0  # Normalized
```

---

## 9. Test Phases for Phase C Implementation

### Phase C.B2: Red Phase (Failing Tests)

**Goal:** Capture baseline pytest failure logs

**Commands:**
```bash
pytest tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter::test_wrapper_delegates_to_tensorflow -vv \
    > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/pytest_red_rawdata.log 2>&1

pytest tests/torch/test_data_pipeline.py::TestDataContainerParity -vv \
    > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/pytest_red_container.log 2>&1
```

**Expected Status:** SKIPPED (not yet implemented) or FAILED (xfail markers)

### Phase C.C (Green Phase)

**Goal:** Implement adapters to make tests pass

**Validation:**
```bash
# After RawDataTorch implementation
pytest tests/torch/test_data_pipeline.py::TestRawDataTorchAdapter -vv \
    > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T<timestamp>Z/pytest_green_rawdata.log 2>&1

# After PtychoDataContainerTorch implementation
pytest tests/torch/test_data_pipeline.py -vv \
    > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T<timestamp>Z/pytest_green_full.log 2>&1
```

**Success Criteria:** All tests PASSED (no SKIPPED/FAILED)

### Phase C.D (Regression)

**Goal:** Validate no breakage in existing tests

**Commands:**
```bash
# Full suite regression
pytest tests/ -vv > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T<timestamp>Z/pytest_regression.log 2>&1

# Config bridge parity (existing)
pytest tests/torch/test_config_bridge.py -k parity -vv
```

**Success Criteria:** Same pass/fail count as baseline (no new failures)

---

## 10. Summary: Test Blueprint Checklist

Before implementing Phase C.C adapters:

- [ ] Create `tests/torch/test_data_pipeline.py` with template classes
- [ ] Add to `TORCH_OPTIONAL_MODULES` in `tests/conftest.py`
- [ ] Add `minimal_raw_data` fixture to `tests/conftest.py`
- [ ] Write 3 failing tests:
  - [ ] `test_wrapper_delegates_to_tensorflow` (RawDataTorch adapter)
  - [ ] `test_container_has_required_attributes` (PtychoDataContainerTorch)
  - [ ] `test_y_patches_are_complex64` (ground truth loading)
- [ ] Capture red-phase pytest logs under timestamped reports directory
- [ ] Document expected shapes/dtypes in test docstrings (reference data_contract.md)

After Phase C.C implementation:

- [ ] Remove `pytest.skip()` calls from implemented tests
- [ ] Run targeted selectors to capture green logs
- [ ] Run full pytest suite to validate no regressions
- [ ] Update this blueprint with actual test counts and selectors

---

## File References

**Testing Infrastructure:**
- `/tests/conftest.py` — Global pytest configuration, torch skip logic
- `/tests/torch/test_config_bridge.py` — Reference for torch-optional pattern
- `/docs/TESTING_GUIDE.md` — TDD methodology, fixture strategy

**Data Contracts:**
- `/specs/data_contracts.md` — NPZ schema, dtype requirements
- `/plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` — TensorFlow baseline

**Gap Analysis:**
- `/plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/torch_gap_matrix.md` — PyTorch implementation gaps

**Findings:**
- `/docs/findings.md` — CONFIG-001, DATA-001, BUG-TF-001 gotchas

---

**Status:** Blueprint ready for Phase C.B2 (red phase test authoring) and C.C (green phase implementation).
