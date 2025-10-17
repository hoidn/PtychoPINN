# Parity Test Case Design

**Initiative:** INTEGRATE-PYTORCH-001 Phase B.B4
**Timestamp:** 2025-10-17T041908Z
**Purpose:** pytest parameterization strategy for configuration bridge parity tests

---

## Test Architecture

### Test Class Organization

```python
class TestConfigBridgeParity(unittest.TestCase):
    """
    Comprehensive parity tests for PyTorch → TensorFlow config bridge adapter.

    Tests are organized by transformation class and use pytest.mark.parametrize
    for field-level coverage. Each test validates that the adapter correctly
    translates PyTorch config values to TensorFlow dataclass attributes.
    """
```

### Parameterization Strategy

Tests will use `pytest.mark.parametrize` with three dimensions:
1. **Field name**: Specific config attribute being tested
2. **PyTorch value**: Input value from PyTorch config
3. **Expected TF value**: Expected output in TensorFlow dataclass

Grouping by transformation class (direct/transform/override_required) provides clear failure diagnostics.

---

## Test Case 1: Direct Field Translation

**Purpose**: Validate fields that pass through without transformation

### ModelConfig Direct Fields

```python
@pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
    pytest.param('N', 128, 128, id='N-direct'),
    pytest.param('n_filters_scale', 2, 2, id='n_filters_scale-direct'),
    pytest.param('object_big', False, False, id='object_big-direct'),
    pytest.param('probe_big', False, False, id='probe_big-direct'),
])
def test_model_config_direct_fields(self, field_name, pytorch_value, expected_tf_value):
    """Test ModelConfig fields that translate directly without transformation."""
```

**Spec mapping**:
- `N`: §5.1:1
- `n_filters_scale`: §5.1:3
- `object_big`: §5.1:6
- `probe_big`: §5.1:7

### TrainingConfig Direct Fields

```python
@pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
    pytest.param('batch_size', 32, 32, id='batch_size-direct'),
    pytest.param('intensity_scale_trainable', True, True, id='intensity_scale_trainable-direct'),
])
def test_training_config_direct_fields(self, field_name, pytorch_value, expected_tf_value):
    """Test TrainingConfig fields that translate directly without transformation."""
```

**Spec mapping**:
- `batch_size`: §5.2:3
- `intensity_scale_trainable`: §5.2:17

---

## Test Case 2: Transformed Field Translation

**Purpose**: Validate fields requiring name/type conversion

### ModelConfig Transform Fields

```python
@pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
    # grid_size tuple → gridsize int (extract first element, validate square)
    pytest.param('grid_size', (3, 3), 'gridsize', 3, id='gridsize-tuple-to-int'),

    # mode enum → model_type enum (categorical mapping)
    pytest.param('mode', 'Unsupervised', 'model_type', 'pinn', id='model_type-unsupervised-to-pinn'),
    pytest.param('mode', 'Supervised', 'model_type', 'supervised', id='model_type-supervised-to-supervised'),

    # amp_activation string normalization
    pytest.param('amp_activation', 'silu', 'amp_activation', 'swish', id='amp_activation-silu-to-swish'),
    pytest.param('amp_activation', 'SiLU', 'amp_activation', 'swish', id='amp_activation-SiLU-to-swish'),
    pytest.param('amp_activation', 'sigmoid', 'amp_activation', 'sigmoid', id='amp_activation-sigmoid-passthrough'),
])
def test_model_config_transform_fields(self, pt_field, pt_value, tf_field, tf_value):
    """Test ModelConfig fields that require transformation during translation."""
```

**Spec mapping**:
- `gridsize`: §5.1:2 (CRITICAL transformation)
- `model_type`: §5.1:4 (CRITICAL transformation)
- `amp_activation`: §5.1:5

### TrainingConfig Transform Fields

```python
@pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
    # epochs → nepochs (field rename)
    pytest.param('epochs', 100, 'nepochs', 100, id='nepochs-rename'),

    # nll bool → nll_weight float (type conversion)
    pytest.param('nll', True, 'nll_weight', 1.0, id='nll_weight-true-to-1.0'),
    pytest.param('nll', False, 'nll_weight', 0.0, id='nll_weight-false-to-0.0'),

    # K → neighbor_count (semantic rename from DataConfig)
    pytest.param('K', 5, 'neighbor_count', 5, id='neighbor_count-from-K'),
])
def test_training_config_transform_fields(self, pt_field, pt_value, tf_field, tf_value):
    """Test TrainingConfig fields that require transformation during translation."""
```

**Spec mapping**:
- `nepochs`: §5.2:4
- `nll_weight`: §5.2:6 (CRITICAL transformation)
- `neighbor_count`: §5.2:14 (CRITICAL transformation)

### InferenceConfig Transform Fields

```python
@pytest.mark.parametrize('pt_field,pt_value,tf_field,tf_value', [
    # K → neighbor_count (same as TrainingConfig)
    pytest.param('K', 6, 'neighbor_count', 6, id='neighbor_count-from-K-inference'),
])
def test_inference_config_transform_fields(self, pt_field, pt_value, tf_field, tf_value):
    """Test InferenceConfig fields that require transformation during translation."""
```

**Spec mapping**:
- `neighbor_count`: §5.3:7

---

## Test Case 3: Override-Required Fields

**Purpose**: Validate fields missing from PyTorch that must come from overrides dict

### ModelConfig Override Fields

```python
@pytest.mark.parametrize('field_name,override_value,expected_default', [
    # Fields missing from PyTorch, using defaults when not overridden
    pytest.param('pad_object', None, True, id='pad_object-default'),
    pytest.param('pad_object', False, False, id='pad_object-override'),

    pytest.param('gaussian_smoothing_sigma', None, 0.0, id='gaussian_smoothing_sigma-default'),
    pytest.param('gaussian_smoothing_sigma', 0.5, 0.5, id='gaussian_smoothing_sigma-override'),
])
def test_model_config_override_fields(self, field_name, override_value, expected_default):
    """Test ModelConfig fields missing from PyTorch that use defaults or overrides."""
```

**Spec mapping**:
- `pad_object`: §5.1:9
- `gaussian_smoothing_sigma`: §5.1:11

### TrainingConfig Override Fields

```python
@pytest.mark.parametrize('field_name,override_value,expected_value', [
    # Critical lifecycle paths (must come from overrides)
    pytest.param('train_data_file', Path('/train.npz'), '/train.npz', id='train_data_file-override', marks=pytest.mark.mvp),
    pytest.param('test_data_file', Path('/test.npz'), '/test.npz', id='test_data_file-override'),
    pytest.param('output_dir', Path('/outputs'), '/outputs', id='output_dir-override'),

    # Grouping parameters (must come from overrides)
    pytest.param('n_groups', 1024, 1024, id='n_groups-override', marks=pytest.mark.mvp),
    pytest.param('n_subsample', 2048, 2048, id='n_subsample-override'),
    pytest.param('subsample_seed', 42, 42, id='subsample_seed-override'),

    # Loss weight defaults (missing from PyTorch)
    pytest.param('mae_weight', None, 0.0, id='mae_weight-default'),
    pytest.param('mae_weight', 0.3, 0.3, id='mae_weight-override'),
    pytest.param('realspace_mae_weight', None, 0.0, id='realspace_mae_weight-default'),
    pytest.param('realspace_weight', None, 0.0, id='realspace_weight-default'),

    # Training flags (missing from PyTorch)
    pytest.param('positions_provided', None, True, id='positions_provided-default'),
    pytest.param('probe_trainable', None, False, id='probe_trainable-default'),
    pytest.param('sequential_sampling', None, False, id='sequential_sampling-default'),
])
def test_training_config_override_fields(self, field_name, override_value, expected_value):
    """Test TrainingConfig fields missing from PyTorch that require overrides."""
```

**Spec mapping**:
- `train_data_file`: §5.2:1 (CRITICAL - MVP)
- `test_data_file`: §5.2:2
- `mae_weight`: §5.2:5
- `realspace_mae_weight`: §5.2:7
- `realspace_weight`: §5.2:8
- `n_groups`: §5.2:10 (CRITICAL - MVP)
- `n_subsample`: §5.2:12
- `subsample_seed`: §5.2:13
- `positions_provided`: §5.2:15
- `probe_trainable`: §5.2:16
- `output_dir`: §5.2:18
- `sequential_sampling`: §5.2:19

### InferenceConfig Override Fields

```python
@pytest.mark.parametrize('field_name,override_value,expected_value', [
    # Critical lifecycle paths (must come from overrides)
    pytest.param('model_path', Path('/model'), '/model', id='model_path-override', marks=pytest.mark.mvp),
    pytest.param('test_data_file', Path('/test.npz'), '/test.npz', id='test_data_file-override', marks=pytest.mark.mvp),
    pytest.param('output_dir', Path('/outputs'), '/outputs', id='output_dir-override'),

    # Grouping parameters (must come from overrides)
    pytest.param('n_groups', 512, 512, id='n_groups-override', marks=pytest.mark.mvp),
    pytest.param('n_subsample', 1024, 1024, id='n_subsample-override'),
    pytest.param('subsample_seed', 99, 99, id='subsample_seed-override'),

    # Debug flag (missing from PyTorch)
    pytest.param('debug', None, False, id='debug-default'),
    pytest.param('debug', True, True, id='debug-override'),
])
def test_inference_config_override_fields(self, field_name, override_value, expected_value):
    """Test InferenceConfig fields missing from PyTorch that require overrides."""
```

**Spec mapping**:
- `model_path`: §5.3:1 (CRITICAL - MVP)
- `test_data_file`: §5.3:2 (CRITICAL - MVP)
- `n_groups`: §5.3:3 (CRITICAL - MVP)
- `n_subsample`: §5.3:5
- `subsample_seed`: §5.3:6
- `debug`: §5.3:8
- `output_dir`: §5.3:9

---

## Test Case 4: Default Divergence Detection

**Purpose**: Catch silent fallback bugs when PyTorch/TensorFlow defaults differ

```python
@pytest.mark.parametrize('field_name,pytorch_default,tf_default,test_value', [
    # CRITICAL: 4 orders of magnitude difference
    pytest.param('nphotons', 1e5, 1e9, 5e8,
                 id='nphotons-default-divergence', marks=pytest.mark.mvp),

    # MEDIUM: 4x difference in probe normalization
    pytest.param('probe_scale', 1.0, 4.0, 2.0,
                 id='probe_scale-default-divergence'),

    # MEDIUM: Different activation function
    pytest.param('amp_activation', 'silu', 'sigmoid', 'swish',
                 id='amp_activation-default-divergence'),
])
def test_default_divergence_detection(self, field_name, pytorch_default, tf_default, test_value):
    """
    Test that fields with different PyTorch/TensorFlow defaults are explicitly set.

    This test ensures the adapter uses the provided value instead of silently
    falling back to incompatible defaults.
    """
```

**Spec mapping**:
- `nphotons`: §5.2:9 (HIGH risk - physics scaling affected)
- `probe_scale`: §5.1:10 (MEDIUM risk - probe normalization affected)
- `amp_activation`: §5.1:5 (MEDIUM risk - reconstruction quality affected)

---

## Test Case 5: Error Handling & Validation

**Purpose**: Validate adapter raises actionable errors for invalid inputs

```python
@pytest.mark.parametrize('invalid_input,expected_error,error_message_fragment', [
    # Non-square grid_size should raise ValueError
    pytest.param(
        {'grid_size': (2, 3)},
        ValueError,
        'Non-square grids not supported',
        id='gridsize-non-square-error'
    ),

    # Invalid mode should raise ValueError
    pytest.param(
        {'mode': 'InvalidMode'},
        ValueError,
        'Invalid mode',
        id='model_type-invalid-enum-error'
    ),

    # Unknown activation should raise ValueError
    pytest.param(
        {'amp_activation': 'unknown_activation'},
        ValueError,
        'Unknown activation',
        id='amp_activation-unknown-error'
    ),

    # Missing required override (train_data_file) should raise ValueError
    pytest.param(
        {'train_data_file': None},
        ValueError,
        'train_data_file is required',
        id='train_data_file-missing-error',
        marks=pytest.mark.mvp
    ),

    # Missing required override (model_path) should raise ValueError
    pytest.param(
        {'model_path': None},
        ValueError,
        'model_path is required',
        id='model_path-missing-error',
        marks=pytest.mark.mvp
    ),
])
def test_adapter_error_handling(self, invalid_input, expected_error, error_message_fragment):
    """Test that adapter raises actionable errors for invalid configurations."""
```

---

## Test Case 6: XFail Markers for Known Gaps

**Purpose**: Document fields not yet implemented (TDD red phase placeholders)

```python
@pytest.mark.parametrize('field_name,pytorch_value,expected_tf_value', [
    # probe_mask: Complex type mismatch (Tensor → bool conversion)
    pytest.param(
        'probe_mask',
        torch.ones(64, 64),  # PyTorch Tensor
        True,  # TensorFlow bool
        id='probe_mask-tensor-to-bool',
        marks=pytest.mark.xfail(
            reason='Complex type mismatch: probe_mask Tensor→bool conversion not implemented',
            strict=True
        )
    ),
])
def test_config_bridge_known_gaps(self, field_name, pytorch_value, expected_tf_value):
    """Test fields with known implementation gaps (xfail until implemented)."""
```

---

## Test Execution Strategy

### Selector Patterns

```bash
# Run all parity tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v

# Run only MVP fields (9 fields from original test)
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -m mvp

# Run only direct field tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -k direct

# Run only transform field tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -k transform

# Run only override field tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -k override

# Run only default divergence tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -k divergence

# Run only error handling tests
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v -k error
```

### Expected Red-Phase Output

Capture to `reports/2025-10-17T041908Z/pytest_red.log`:
```bash
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v 2>&1 | \
  tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/pytest_red.log
```

Expected outcomes:
- **SKIPPED**: All tests (PyTorch runtime unavailable in CI)
- **PASSED** (when PyTorch available): MVP fields (9 tests)
- **FAILED/XFAIL** (when PyTorch available): Phase B.B4 extension fields (29 tests)

---

## Coverage Summary

| Test Case | Field Count | Status | Priority |
|-----------|-------------|--------|----------|
| Direct fields | 6 | Implemented in adapter | MVP + Extension |
| Transform fields | 8 | Implemented in adapter | MVP + Extension |
| Override-required | 21 | Partially implemented (MVP: 9, Extension: 12) | MVP + Extension |
| Default divergence | 2 | Not yet tested | Extension |
| Error handling | 5 | Partially tested (MVP: 2, Extension: 3) | MVP + Extension |
| Known gaps (xfail) | 1 | Documented, not implemented | Extension |
| **TOTAL** | **43** | MVP: 9 implemented, Extension: 34 pending | |

---

## Implementation Checklist

- [ ] Import canonical fixtures from `fixtures.py`
- [ ] Create `TestConfigBridgeParity` class extending `unittest.TestCase`
- [ ] Implement `setUp`/`tearDown` for `params.cfg` snapshot/restore
- [ ] Add pytest.mark.parametrize decorators for each test case
- [ ] Mark MVP fields with `pytest.mark.mvp` custom marker
- [ ] Mark known gaps with `pytest.mark.xfail(strict=True)`
- [ ] Add docstrings linking to spec sections (e.g., "§5.1:1")
- [ ] Capture red-phase pytest output to `pytest_red.log`
- [ ] Document any newly discovered transformation requirements

---

## References

- Canonical baseline: `fixtures.py`
- Field mapping: `field_matrix.md`
- Spec tables: `specs/ptychodus_api_spec.md:220-273`
- Existing MVP test: `tests/torch/test_config_bridge.py:34-162`
- TDD methodology: `docs/TESTING_GUIDE.md:153-161`
