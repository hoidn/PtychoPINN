# Phase B.B4 Lightning Regression Completion (Attempt #21)

**Date:** 2025-10-19
**Loop ID:** Attempt #21
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** D2.B4 (Turn Lightning regression suite fully green)

## Executive Summary

Successfully completed Phase B.B4 by fixing the `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` fixture. All 3 Lightning regression tests now pass (3/3 ✅), unblocking Phase C stitching implementation.

## Problem Statement

**Quoted from phase_d2_completion.md B4 guidance:**
> Fix the `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` fixture so the monkeypatched stub inherits from `lightning.pytorch.core.LightningModule` (minimal `training_step` + `configure_optimizers`).

**Root cause (from supervisor notes at `reports/2025-10-18T171500Z/phase_d2_completion/summary.md`):**
> `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` still fails because the monkeypatched stub returned by the test is not a subclass of `LightningModule`. Lightning validates module types before `trainer.fit`, causing the RuntimeError.

**Original error (from baseline failure log):**
```
RuntimeError: Lightning training failed. See logs for details.
...
ERROR: Lightning training failed: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`,
       got `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module.<locals>.mock_lightning_init.<locals>.StubLightningModule`
```

## Implementation Changes

**File modified:** `tests/torch/test_workflows_components.py:820-845`

**Change summary:**
Updated the `mock_lightning_init` function's `StubLightningModule` class to:
1. Inherit from `lightning.pytorch.core.LightningModule` (was: plain class)
2. Implement required Lightning interface:
   - `training_step(batch, batch_idx)` — returns deterministic zero loss
   - `configure_optimizers()` — returns Adam optimizer
3. Add minimal `dummy_param` to satisfy Lightning's parameter requirements

**Rationale:**
Lightning's `Trainer.fit` performs `isinstance(module, LightningModule)` validation before executing training. The original stub was a plain Python class, causing this check to fail. The updated stub now inherits from `LightningModule` while preserving the spy functionality to validate constructor arguments (all four config objects).

**Key design decision:**
The stub remains minimal (zero loss, no-op training) to keep test execution fast (≤6s) while satisfying Lightning's type contract.

## Test Results

### Targeted Selector
**Command:**
```bash
pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv \
  | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log
```

**Outcome:** ✅ 3 passed in 5.27s
- `test_train_with_lightning_instantiates_module` ✅ (was FAILED before fix)
- `test_train_with_lightning_runs_trainer_fit` ✅ (passing from Attempt #10)
- `test_train_with_lightning_returns_models_dict` ✅ (passing from Attempt #10)

### Full Regression Suite
**Command:** `pytest tests/ -v`

**Outcome:** ✅ 220 passed, 16 skipped, 1 xfailed, 1 failed (235.32s)

**Comparison to baseline (Attempt #12/#14):**
- **Passed:** 220 (was 219) — **+1 test fixed** ✅
- **Skipped:** 16 (was 16) — unchanged
- **xfailed:** 1 (was 1) — unchanged
- **Failed:** 1 (was 1) — pre-existing `test_pytorch_train_save_load_infer_cycle` (Phase D checkpoint loading, separate from B4 scope)

**Verdict:** ZERO new failures introduced. Net improvement: +1 passing test.

## Exit Criteria Validation

Per `phase_d2_completion.md` B4 row:
- [x] `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` fixture adjusted to inherit from `LightningModule`
- [x] All 3 TestTrainWithLightningRed tests passing
- [x] Green log captured at `reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log`
- [x] Full regression suite passed with ZERO new failures
- [x] Plan checklist B4 updated to `[x]` with artifact references

## Artifacts

| Artifact | Location | Size | Description |
|----------|----------|------|-------------|
| Green log | `reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log` | 1.5KB | Targeted selector output (3 passed) |
| Summary | `reports/2025-10-18T171500Z/phase_d2_completion/b4_completion_summary.md` | (this file) | Loop execution evidence |
| Plan update | `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` | — | B4 row marked `[x]` with Attempt #21 reference |
| Code change | `tests/torch/test_workflows_components.py:820-845` | 25 lines | StubLightningModule fix |

## Next Steps

Per `phase_d2_completion.md` Phase C guidance:
- **C1:** Design inference data flow for `_reassemble_cdi_image_torch`
- **C2:** Add failing pytest coverage for stitching path
- **C3:** Implement `_reassemble_cdi_image_torch` with PyTorch helper parity
- **C4:** Validate stitching tests + integration subset with `do_stitching=True`

**Recommendation:** Proceed to Phase C.C1 (inference design) now that Lightning regression suite is fully green.

## Observations

1. **Test execution time:** 5.27s for all 3 Lightning tests — fast enough for TDD red-green cycles.
2. **Monkeypatch pattern:** The updated fixture preserves spy functionality while satisfying Lightning's type requirements; pattern can be reused for future Lightning interface tests.
3. **Full suite stability:** 220/221 tests passing (99.5%); single failing test (`test_pytorch_train_save_load_infer_cycle`) is documented Phase D checkpoint loading issue unrelated to B4 work.

---

**Sign-off:** Phase B.B4 COMPLETE per plan exit criteria. Lightning regression suite now 3/3 passing with full regression verification.
