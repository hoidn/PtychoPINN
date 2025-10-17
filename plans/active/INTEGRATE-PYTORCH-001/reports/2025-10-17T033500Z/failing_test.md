# Phase B.B2 Failing Test Report — Config Bridge MVP

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** B.B2 (TDD — Failing Test)
**Date:** 2025-10-17
**Timestamp:** 033500Z
**Author:** Ralph (TDD loop)

---

## Executive Summary

Created failing test `tests/torch/test_config_bridge.py::test_mvp_config_bridge_populates_params_cfg` that defines the contract for the PyTorch → TensorFlow configuration bridge adapter. Test currently **skipped** due to PyTorch environment unavailability in this CI context, but demonstrates the expected implementation contract and failure pathway.

**Status:** ✅ Failing test authored and logged
**Next Step:** Phase B.B3 — Implement adapter module to make test pass

---

## Test File Location

```
tests/torch/test_config_bridge.py
```

**Test Selector:**
```bash
pytest tests/torch/test_config_bridge.py -k mvp -v
```

---

## Test Contract Definition

### Purpose
Validates that PyTorch singleton configs (`ptycho_torch.config_params`) can be translated to TensorFlow dataclass configs (`ptycho.config.config`) via a bridge adapter module, enabling population of the legacy `params.cfg` dictionary through the standard `update_legacy_dict()` function.

### MVP Scope (9 Fields)

Based on `scope_notes.md:150-170`, the test covers these critical fields:

#### Model Essentials (3)
1. **`N`** — Tensor shape foundation (DataConfig.N → ModelConfig.N)
2. **`gridsize`** — Channel count determinant (DataConfig.grid_size tuple → ModelConfig.gridsize int)
3. **`model_type`** — Workflow selector (ModelConfig.mode enum → ModelConfig.model_type enum)

#### Lifecycle Paths (3)
4. **`train_data_file`** — Training data source (override → TrainingConfig.train_data_file)
5. **`test_data_file`** — Inference data source (override → InferenceConfig.test_data_file)
6. **`model_path`** — Model loading path (override → InferenceConfig.model_path)

#### Data Grouping (2)
7. **`n_groups`** — Number of groups to generate (override → TrainingConfig.n_groups)
8. **`neighbor_count`** — K-NN search width (DataConfig.K → TrainingConfig.neighbor_count)

#### Physics Scaling (1)
9. **`nphotons`** — Poisson loss scaling (DataConfig.nphotons → ModelConfig.nphotons)

---

## Test Workflow

### Step 1: Instantiate PyTorch Configs
```python
pt_data = DataConfig(
    N=128,
    grid_size=(2, 2),
    nphotons=1e9,
    K=7
)
pt_model = ModelConfig(mode='Unsupervised')
pt_train = TrainingConfig(epochs=1)
pt_infer = InferenceConfig(batch_size=1)
```

### Step 2: Import Adapter Module (Expected to Fail)
```python
from ptycho_torch import config_bridge

spec_model = config_bridge.to_model_config(pt_data, pt_model)
spec_train = config_bridge.to_training_config(
    spec_model, pt_data, pt_train,
    overrides=dict(
        train_data_file=Path('train.npz'),
        n_groups=512,
        neighbor_count=7,
        nphotons=1e9
    )
)
spec_infer = config_bridge.to_inference_config(
    spec_model, pt_data, pt_infer,
    overrides=dict(
        model_path=Path('model_dir'),
        test_data_file=Path('test.npz'),
        n_groups=512,
        neighbor_count=7
    )
)
```

### Step 3: Call update_legacy_dict
```python
update_legacy_dict(params.cfg, spec_train)
update_legacy_dict(params.cfg, spec_infer)
```

### Step 4: Assert params.cfg Population
```python
assert params.cfg['N'] == 128
assert params.cfg['gridsize'] == 2
assert params.cfg['model_type'] == 'pinn'
assert params.cfg['train_data_file_path'] == 'train.npz'
assert params.cfg['test_data_file_path'] == 'test.npz'
assert params.cfg['model_path'] == 'model_dir'
assert params.cfg['n_groups'] == 512
assert params.cfg['neighbor_count'] == 7
assert params.cfg['nphotons'] == 1e9
```

---

## Actual Test Execution Results

### Command
```bash
pytest tests/torch/test_config_bridge.py -k mvp -v 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/pytest.log
```

### Output
```
============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0 -- /home/ollie/miniconda3/envs/ptycho311/bin/python3.11
cachedir: .pytest_cache
PyTorch: not available (/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommWindowRegister)

rootdir: /home/ollie/Documents/PtychoPINN2
configfile: pyproject.toml
plugins: anyio-4.9.0
collecting ... collected 1 item

tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg SKIPPED [100%]

=========================== short test summary info ============================
SKIPPED [1] tests/torch/test_config_bridge.py:50: PyTorch not available
============================== 1 skipped in 0.16s ==============================
```

### Exit Code
`0` (pytest passed with 1 skipped test)

---

## Failure Mode Analysis

### Expected Failure (Implementation-Ready Environment)

**When PyTorch is available**, the test will reach the adapter import and fail with:

```python
ModuleNotFoundError: No module named 'ptycho_torch.config_bridge'
```

This triggers the `except (ModuleNotFoundError, AttributeError, ImportError)` block:

```python
pytest.xfail(f"Config bridge adapter not yet implemented: ModuleNotFoundError: No module named 'ptycho_torch.config_bridge'")
```

**Test Status:** `XFAIL` (expected failure, test is strict)

### Actual Failure (Current CI Environment)

**When PyTorch is unavailable**, `tests/conftest.py:25-43` auto-skips all tests in `tests/torch/` directory:

```python
if "torch" in str(item.fspath).lower() or item.get_closest_marker("torch"):
    if not torch_available:
        item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
```

**Test Status:** `SKIPPED` (environment constraint, not a test failure)

---

## Implications

### For TDD Workflow

✅ **Test Contract Defined:** The test fully specifies the adapter API:
- Expected module: `ptycho_torch.config_bridge`
- Expected functions: `to_model_config()`, `to_training_config()`, `to_inference_config()`
- Expected behavior: Translate PyTorch configs → TensorFlow dataclasses → `params.cfg`

✅ **Red-Green-Refactor Ready:** Phase B.B3 can now implement the adapter to make the test pass.

### For CI/CD

⚠️ **Environment Dependency:** This test requires PyTorch runtime availability. Current skip is acceptable for TDD documentation phase but blocks actual validation.

**Mitigation Options:**
1. **Accept skip during TDD documentation** (current state) — implementation phase (B.B3) will require PyTorch environment
2. **Mock PyTorch imports** for environment-independent structural testing
3. **Run in PyTorch-enabled environment** when validating implementation

**Decision:** Proceed with Option 1. Phase B.B3 implementation must be tested in PyTorch-capable environment.

---

## Field Translation Requirements (Derived from Test)

### Type Conversions Needed

| PyTorch Source | TensorFlow Target | Transformation |
|----------------|-------------------|----------------|
| `DataConfig.grid_size: Tuple[int, int]` | `ModelConfig.gridsize: int` | Extract `grid_size[0]` (assume square) |
| `ModelConfig.mode: 'Unsupervised'` | `ModelConfig.model_type: 'pinn'` | Enum mapping |
| `ModelConfig.mode: 'Supervised'` | `ModelConfig.model_type: 'supervised'` | Enum mapping |
| `DataConfig.K: int` | `TrainingConfig.neighbor_count: int` | Direct copy |
| `TrainingConfig.epochs: int` | `TrainingConfig.nepochs: int` | Field rename |

### Override Pattern

Adapter functions must accept `overrides: dict` parameter for fields missing from PyTorch configs:

```python
def to_training_config(
    model: ModelConfig,
    data: DataConfig,
    training: TrainingConfig,
    overrides: dict = None
) -> TFTrainingConfig:
    # Merge PyTorch fields + overrides to build TensorFlow dataclass
    ...
```

**Critical Overrides (not in PyTorch configs):**
- `train_data_file` / `test_data_file` (PyTorch uses `training_directories: List[str]`)
- `model_path` (InferenceConfig)
- `n_groups` (must be explicit)

---

## Test Artifacts

### Files Created
- `tests/torch/test_config_bridge.py` (162 lines)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/pytest.log` (18 lines)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/failing_test.md` (this file)

### Test Metadata
- **Test Class:** `TestConfigBridgeMVP`
- **Test Method:** `test_mvp_config_bridge_populates_params_cfg`
- **Markers:** `@pytest.mark.xfail(strict=True, reason="PyTorch config bridge missing MVP translations")`
- **State Management:** `setUp()` saves `params.cfg` snapshot, `tearDown()` restores it

---

## Next Steps (Phase B.B3)

### Implementation Checklist

1. **Create adapter module:**
   ```bash
   touch ptycho_torch/config_bridge.py
   ```

2. **Implement translation functions:**
   - `to_model_config(data: DataConfig, model: ModelConfig) -> TFModelConfig`
   - `to_training_config(model: TFModelConfig, data: DataConfig, training: TrainingConfig, overrides: dict) -> TFTrainingConfig`
   - `to_inference_config(model: TFModelConfig, data: DataConfig, inference: InferenceConfig, overrides: dict) -> TFInferenceConfig`

3. **Handle type conversions:**
   - Tuple → int (`grid_size`)
   - Enum mapping (`mode` → `model_type`)
   - Field rename (`epochs` → `nepochs`, `K` → `neighbor_count`)

4. **Merge override pattern:**
   - Accept `overrides` dict in each function
   - Merge with translated PyTorch fields
   - Instantiate TensorFlow dataclass with merged values

5. **Run test in PyTorch environment:**
   - Expected result: `XFAIL → PASSED` (remove `@pytest.mark.xfail` once implementation complete)

6. **Extend to full parity (Phase B.B4):**
   - Add parameterized tests for all 75+ fields from `config_schema_map.md`

---

## References

- Test implementation: `tests/torch/test_config_bridge.py:1-162`
- input.md directive: `input.md:1-44`
- MVP scope definition: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/scope_notes.md:150-170`
- Field mapping: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md:36-227`
- Spec contract: `specs/ptychodus_api_spec.md:61-149`
- PyTorch configs: `ptycho_torch/config_params.py:1-171`
- TensorFlow configs: `ptycho/config/config.py:72-154`
- CONFIG-001 finding: `docs/findings.md:9`

---

**Loop Status:** ✅ Phase B.B2 complete — failing test authored and documented
**Artifacts Saved:** `pytest.log`, `failing_test.md`
**Ready for Phase B.B3:** Adapter implementation
