# Phase F4.3 — Verification Command Reference

## Purpose

This document provides expanded context and rationale for the authoritative pytest selectors defined in `handoff_notes.md §3`. Use this as a reference when setting up CI validation or debugging test failures.

---

## 1. Collection Health Check

**Command:**
```bash
pytest --collect-only tests/torch/ -q
```

**Purpose:** Validates that all torch tests can be discovered without ImportError or collection failures.

**Expected Outcome:**
```
tests/torch/test_backend_selection.py::TestBackendSelection::test_pytorch_backend_available
tests/torch/test_backend_selection.py::TestBackendSelection::test_pytorch_unavailable_raises_error
tests/torch/test_config_bridge.py::test_mvp_config_bridge_populates_params_cfg
tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_N
[... 60+ more items ...]

66 tests collected
```

**Failure Modes:**
- **`ImportError: No module named 'torch'`** → PyTorch not installed; CI provisioning failed
- **`ModuleNotFoundError: ptycho_torch.config_bridge`** → Module missing or import guard broken
- **`collection failure`** → Test syntax error or fixture dependency issue

**Troubleshooting:**
1. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
2. Check test file syntax: `python -m py_compile tests/torch/test_config_bridge.py`
3. Review conftest skip logic: ensure `tests/conftest.py` does NOT skip torch tests when PyTorch available

---

## 2. Backend Selection Validation

**Command:**
```bash
pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv
```

**Purpose:** Confirms fail-fast behavior when PyTorch import fails (validates governance §3.3 requirement).

**Expected Outcome:**
```
tests/torch/test_backend_selection.py::TestBackendSelection::test_pytorch_unavailable_raises_error PASSED
```

**Test Logic:**
- Mocks `TORCH_AVAILABLE=False` in backend dispatcher
- Attempts to select `backend='pytorch'`
- Asserts `RuntimeError` raised with actionable message: "Install PyTorch support with: pip install torch..."

**Failure Modes:**
- **FAILED (no exception raised)** → Backend dispatcher silently falls back instead of failing fast
- **FAILED (wrong exception type)** → Dispatcher raises ImportError instead of RuntimeError
- **FAILED (unclear error message)** → User guidance missing from exception text

**Spec Reference:** `specs/ptychodus_api_spec.md:140-162` (§4.2 fail-fast imports)

---

## 3. Configuration Bridge Parity

**Command:**
```bash
pytest tests/torch/test_config_bridge.py -k parity -vv
```

**Purpose:** Validates PyTorch→TensorFlow config translation for 38+ spec-required fields.

**Expected Outcome:**
```
tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_N PASSED
tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_gridsize PASSED
tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_n_filters_scale PASSED
[... 35+ more parameterized tests ...]

38 passed
```

**Test Coverage:**
- **ModelConfig:** 11 fields (N, gridsize, n_filters_scale, model_type, amp_activation, object_big, probe_big, probe_mask, pad_object, probe_scale, gaussian_smoothing_sigma)
- **TrainingConfig:** 18 fields (train_data_file, test_data_file, batch_size, nepochs, mae_weight, nll_weight, realspace_mae_weight, realspace_weight, nphotons, n_groups, n_subsample, subsample_seed, neighbor_count, positions_provided, probe_trainable, intensity_scale_trainable, output_dir, sequential_sampling)
- **InferenceConfig:** 9 fields (model_path, test_data_file, n_groups, n_subsample, subsample_seed, neighbor_count, debug, output_dir)

**Failure Modes:**
- **FAILED (assertion error)** → Field missing from translated TensorFlow config
- **FAILED (TypeError)** → Type conversion failed (e.g., tuple → int)
- **FAILED (ValueError)** → Enum mapping incorrect (e.g., 'Unsupervised' → 'pinn')
- **FAILED (KeyError)** → Override required but not provided

**Debugging:**
1. Run single field test: `pytest tests/torch/test_config_bridge.py -k test_model_config_N -vv`
2. Check field matrix: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md`
3. Review adapter logic: `ptycho_torch/config_bridge.py:70-380`

---

## 4. Full Torch Suite Regression

**Command:**
```bash
pytest tests/torch/ -v
```

**Purpose:** Comprehensive validation of entire PyTorch test suite (backend selection, config bridge, data pipelines, workflows).

**Expected Baseline (Phase F3.4):**
```
66 passed, 3 skipped, 1 xfailed in 15.65s
```

**Skipped Tests (Expected):**
- `tests/torch/test_tf_helper.py::test_combine_complex` — Helper not implemented
- `tests/torch/test_tf_helper.py::test_get_mask` — Helper not implemented
- `tests/torch/test_tf_helper.py::test_placeholder_torch_functions` — Module-level skip

**Xfailed Tests (Expected):**
- (Known issue tracked in docs/fix_plan.md; does not block Phase F)

**Failure Modes:**
- **New failures (baseline: 0)** → Regression introduced
- **New skips (baseline: 3)** → Test guard added incorrectly
- **Collection errors** → Import failures or syntax errors

**Regression Protocol:**
1. Compare against Phase F3.4 baseline: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T201922Z/pytest_torch_green.log`
2. If new failures: isolate failing test (`pytest tests/torch/test_X.py::test_Y -vv`)
3. Bisect commits to identify regression source
4. Log failure in docs/fix_plan.md with artifact path

---

## 5. Full Project Regression

**Command:**
```bash
pytest tests/ -v
```

**Purpose:** Validates no regressions in TensorFlow stack or shared utilities due to PyTorch changes.

**Expected Baseline (Phase F3.4):**
```
207 passed, 13 skipped, 1 xfailed
```

**Skipped Distribution:**
- 3 skipped in `tests/torch/` (expected per §4)
- 10 skipped elsewhere (pre-existing, unrelated to Phase F)

**Failure Modes:**
- **New failures in `tests/` (non-torch)** → Cross-contamination from torch changes
- **New failures in `tests/torch/`** → See §4 debugging protocol
- **Increased skip count** → Unintended test guard activation

**Validation Checklist:**
- [ ] No new failures in `tests/test_integration_workflow.py` (TensorFlow persistence)
- [ ] No new failures in `tests/test_coordinate_grouping.py` (shared data pipeline)
- [ ] No new failures in `tests/image/` (shared utilities)
- [ ] Torch suite matches §4 baseline (66 passed, 3 skipped, 1 xfailed)

**Bisection Strategy (if regressions detected):**
1. Run TensorFlow-only subset: `pytest tests/ --ignore=tests/torch/ -v`
2. Run shared utilities: `pytest tests/test_tf_helper.py tests/image/ -v`
3. Isolate to Phase F commits: `git bisect start HEAD <phase_f_start_commit>`
4. Document findings in `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/regression_investigation.md`

---

## 6. CI Integration Examples

### 6.1 GitHub Actions Snippet

```yaml
name: PyTorch Test Suite

on: [push, pull_request]

jobs:
  test-torch-required:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install torch>=2.2 torchvision --index-url https://download.pytorch.org/whl/cpu

      - name: Verify PyTorch installation
        run: python -c "import torch; print(f'PyTorch {torch.__version__} installed')"

      - name: Collection health check
        run: pytest --collect-only tests/torch/ -q

      - name: Backend selection validation
        run: pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv

      - name: Config bridge parity
        run: pytest tests/torch/test_config_bridge.py -k parity -vv

      - name: Full torch suite
        run: pytest tests/torch/ -v

      - name: Full project regression
        run: pytest tests/ -v
```

### 6.2 GitLab CI Snippet

```yaml
torch-required-tests:
  image: python:3.9
  before_script:
    - pip install -e .
    - pip install torch>=2.2 torchvision --index-url https://download.pytorch.org/whl/cpu
    - python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
  script:
    - pytest --collect-only tests/torch/ -q
    - pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv
    - pytest tests/torch/test_config_bridge.py -k parity -vv
    - pytest tests/torch/ -v
    - pytest tests/ -v
  artifacts:
    when: on_failure
    paths:
      - pytest.log
    expire_in: 1 week
```

---

## 7. Troubleshooting Decision Tree

```
Test failure detected
│
├─ Collection failure?
│  ├─ YES → Check PyTorch installation, module imports, test syntax
│  └─ NO → Continue
│
├─ Backend selection test failed?
│  ├─ YES → Review backend dispatcher fail-fast logic (ptycho/workflows/backend_selector.py)
│  └─ NO → Continue
│
├─ Config bridge parity test failed?
│  ├─ YES → Check field matrix, adapter logic, override requirements
│  └─ NO → Continue
│
├─ Torch suite regression?
│  ├─ YES → Compare against Phase F3.4 baseline, bisect commits
│  └─ NO → Continue
│
└─ Full project regression?
   ├─ YES → Isolate TensorFlow-only subset, check cross-contamination
   └─ NO → All tests passing! 🎉
```

---

## 8. Archive Template

When executing verification commands, save logs using this naming convention:

```
plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/verification_<type>.log
```

**Examples:**
- `verification_collection.log` (§1 output)
- `verification_backend_selection.log` (§2 output)
- `verification_config_parity.log` (§3 output)
- `verification_torch_suite.log` (§4 output)
- `verification_full_regression.log` (§5 output)

**Cross-reference template for docs/fix_plan.md:**
```markdown
* [YYYY-MM-DD] Attempt #N — Verification: Executed torch suite validation per handoff guidance.
  Artifacts: `reports/<ISO8601>/verification_*.log`.
  Results: [66 passed, 3 skipped, 1 xfailed | <describe any deviations>].
  Next: [Continue with next phase | Investigate regression at file:line].
```

---

**Summary:**
This reference expands the 5 authoritative commands from `handoff_notes.md §3` with detailed rationale, expected outcomes, failure modes, debugging protocols, CI integration examples, and log archival conventions. Use this document when onboarding new CI maintainers or troubleshooting test failures.
