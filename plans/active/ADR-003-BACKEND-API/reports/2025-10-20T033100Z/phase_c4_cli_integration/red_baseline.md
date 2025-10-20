# RED Baseline Summary — ADR-003 Phase C4.B4

**Date:** 2025-10-20
**Purpose:** Document RED phase test results and establish acceptance criteria for GREEN phase.

---

## RED Test Execution Summary

### Training CLI Tests

**Selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv
```

**Results:**
- **Total Tests:** 6
- **Passed:** 0
- **Failed:** 6
- **Runtime:** 5.03s
- **Log Location:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_train_red.log`

**Test Breakdown:**
1. `test_accelerator_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --accelerator cpu)
2. `test_deterministic_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --deterministic)
3. `test_no_deterministic_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --no-deterministic)
4. `test_num_workers_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --num-workers 4)
5. `test_learning_rate_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --learning-rate 5e-4)
6. `test_multiple_execution_config_flags` — FAILED (argparse.ArgumentError: unrecognized arguments: --accelerator gpu --no-deterministic --num-workers 8 --learning-rate 1e-3)

---

### Inference CLI Tests

**Selector:**
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
```

**Results:**
- **Total Tests:** 4
- **Passed:** 0
- **Failed:** 4
- **Runtime:** 3.77s
- **Log Location:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_inference_red.log`

**Test Breakdown:**
1. `test_accelerator_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --accelerator cpu)
2. `test_num_workers_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --num-workers 4)
3. `test_inference_batch_size_flag_roundtrip` — FAILED (argparse.ArgumentError: unrecognized arguments: --inference-batch-size 32)
4. `test_multiple_execution_config_flags` — FAILED (argparse.ArgumentError: unrecognized arguments: --accelerator gpu --num-workers 8 --inference-batch-size 64)

---

## Failure Signature Analysis

### Primary Failure Mode: Unrecognized Arguments

**All 10 tests** failed with the same root cause:
```
argparse.ArgumentError: unrecognized arguments: <new_flag> <value>
```

**Diagnosis:**
- The CLI argparse parsers in `ptycho_torch/train.py` and `ptycho_torch/inference.py` do **not** define the execution config flags
- Tests correctly invoke CLI with new flags, but argparse rejects them before reaching factory
- Factory mocks were **not called** because CLI exited early due to argparse errors

**Expected Behavior (RED Phase):**
- This is the **correct RED failure signature** per TDD methodology
- Tests establish acceptance criteria: argparse must accept new flags
- Mocks verify end-to-end roundtrip (CLI → factory → execution_config)

---

### Secondary Failure Mode: Factory Not Called

**Assertion Pattern:**
```python
AssertionError: Factory was not called
assert False
 +  where False = <MagicMock id='...'>.called
```

**Diagnosis:**
- Tests use `unittest.mock.patch` to intercept factory calls
- Factory mocks are correctly configured with expected return values
- Mocks never trigger because argparse exits before factory invocation

**Implication:**
- Phase C4.C must add argparse flags **before** tests can validate factory wiring
- Once flags exist, tests will validate execution_config field mapping

---

## GREEN Phase Acceptance Criteria

### Training CLI (6 tests → GREEN)

**Required Implementation (Phase C4.C1-C4.C4):**

1. **Argparse Flags Added:**
   - `--accelerator` (type=str, choices=[auto, cpu, gpu, cuda, tpu, mps], default='auto')
   - `--deterministic` (action='store_true', default=True)
   - `--no-deterministic` (action='store_false', dest='deterministic')
   - `--num-workers` (type=int, default=0)
   - `--learning-rate` (type=float, default=1e-3)

2. **Execution Config Instantiation:**
   ```python
   from ptycho.config.config import PyTorchExecutionConfig

   execution_config = PyTorchExecutionConfig(
       accelerator=args.accelerator,
       deterministic=args.deterministic,
       num_workers=args.num_workers,
       learning_rate=args.learning_rate,
   )
   ```

3. **Factory Integration:**
   ```python
   from ptycho_torch.config_factory import create_training_payload

   payload = create_training_payload(
       train_data_file=args.train_data_file,
       output_dir=args.output_dir,
       overrides=dict(n_groups=args.n_images),
       execution_config=execution_config,  # ← NEW PARAMETER
   )
   ```

4. **Workflow Dispatch:**
   - Execution config forwarded from payload to workflow helpers
   - Already wired in Phase C3 (`_train_with_lightning` accepts execution_config)

**GREEN Criteria:**
- `pytest tests/torch/test_cli_train_torch.py -vv` → **6 PASSED, 0 FAILED**
- Factory mocks capture `execution_config` argument
- Execution config fields match CLI arg values

---

### Inference CLI (4 tests → GREEN)

**Required Implementation (Phase C4.C5-C4.C7):**

1. **Argparse Flags Added:**
   - `--accelerator` (same spec as training)
   - `--num-workers` (same spec as training)
   - `--inference-batch-size` (type=int, default=None)

2. **Execution Config Instantiation:**
   ```python
   execution_config = PyTorchExecutionConfig(
       accelerator=args.accelerator,
       num_workers=args.num_workers,
       inference_batch_size=args.inference_batch_size,
   )
   ```

3. **Factory Integration:**
   ```python
   from ptycho_torch.config_factory import create_inference_payload

   payload = create_inference_payload(
       model_path=args.model_path,
       test_data_file=args.test_data,
       output_dir=args.output_dir,
       overrides=dict(n_groups=args.n_images),
       execution_config=execution_config,  # ← NEW PARAMETER
   )
   ```

**GREEN Criteria:**
- `pytest tests/torch/test_cli_inference_torch.py -vv` → **4 PASSED, 0 FAILED**
- Factory mocks capture `execution_config` argument
- Execution config fields match CLI arg values

---

## Implementation Checklist (Phase C4.C)

### Training CLI Updates (`ptycho_torch/train.py`)

- [ ] Add 5 argparse arguments (lines ~370-395):
  - [ ] `--accelerator`
  - [ ] `--deterministic`
  - [ ] `--no-deterministic`
  - [ ] `--num-workers`
  - [ ] `--learning-rate`
- [ ] Import `PyTorchExecutionConfig` (line ~15)
- [ ] Instantiate execution config from parsed args (after line ~459)
- [ ] Pass `execution_config` to `create_training_payload()` (line ~520-532)
- [ ] Remove hardcoded `learning_rate=1e-3` in workflow (components.py:538 replacement handled via factory)

### Inference CLI Updates (`ptycho_torch/inference.py`)

- [ ] Add 3 argparse arguments (lines ~330-350):
  - [ ] `--accelerator`
  - [ ] `--num-workers`
  - [ ] `--inference-batch-size`
- [ ] Import `PyTorchExecutionConfig`
- [ ] Instantiate execution config from parsed args
- [ ] Pass `execution_config` to `create_inference_payload()`

### Validation (Phase C4.D)

- [ ] Run RED → GREEN transition: `pytest tests/torch/test_cli_train_torch.py -vv` (expect 6/6 PASSED)
- [ ] Run RED → GREEN transition: `pytest tests/torch/test_cli_inference_torch.py -vv` (expect 4/4 PASSED)
- [ ] Run factory smoke: `pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv` (expect no regressions)
- [ ] Run full suite: `CUDA_VISIBLE_DEVICES="" pytest tests/ -v` (expect 271 passed, 0 new failures)

---

## RED Phase Exit Criteria Validation

| Criterion | Status | Evidence |
|-----------|--------|----------|
| **C4.B1:** Training CLI test scaffold authored | ✅ COMPLETE | `tests/torch/test_cli_train_torch.py` (241 lines, 6 test methods) |
| **C4.B2:** Inference CLI test scaffold authored | ✅ COMPLETE | `tests/torch/test_cli_inference_torch.py` (199 lines, 4 test methods) |
| **C4.B3:** RED logs captured | ✅ COMPLETE | Training: 6 FAILED in 5.03s; Inference: 4 FAILED in 3.77s |
| **C4.B4:** RED baseline documented | ✅ COMPLETE | This document (comprehensive failure analysis + acceptance criteria) |
| **Failure Signature:** Clear argparse errors | ✅ VERIFIED | All tests fail with "unrecognized arguments: <flag>" |
| **Test Count:** 10 total tests (6 training, 4 inference) | ✅ VERIFIED | As planned in Phase C4 plan §C4.B1-C4.B2 |
| **Runtime Budget:** RED tests < 10s each | ✅ VERIFIED | Training 5.03s, Inference 3.77s (well within budget) |

---

## Next Phase: C4.C Implementation

**Objective:** Turn RED tests GREEN by implementing CLI flag exposure + factory integration.

**Estimated Effort:** 2 hours (argparse 30min, execution config instantiation 30min, factory integration 30min, validation 30min)

**Deliverables:**
- Modified `ptycho_torch/train.py` with 5 new flags + execution config wiring
- Modified `ptycho_torch/inference.py` with 3 new flags + execution config wiring
- GREEN logs: `pytest_cli_train_green.log` (6 PASSED), `pytest_cli_inference_green.log` (4 PASSED)
- Factory smoke log: `pytest_factory_smoke.log` (no regressions)
- Full suite log: `pytest_full_suite_c4.log` (271 passed, 0 new failures)

**References:**
- Phase C4 Plan: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` §C4.C
- Argparse Schema: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/argparse_schema.md`
- Factory Design: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`

---

**RED Phase COMPLETE:** All scaffolds authored, logs captured, acceptance criteria documented. Ready for Phase C4.C GREEN implementation.
