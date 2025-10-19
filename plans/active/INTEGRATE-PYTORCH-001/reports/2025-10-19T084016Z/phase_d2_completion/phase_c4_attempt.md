# Phase C4 Attempt Summary

Date: 2025-10-19
Loop: Ralph Attempt #26
Status: Blocked on TensorFlow reassembly shape contract
Artifact Hub: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/

## Work Completed

### 1. Test Modernization
- Renamed `TestReassembleCdiImageTorchRed` → `TestReassembleCdiImageTorchGreen`
- Updated docstrings to reflect GREEN phase expectations
- Added `mock_lightning_module` fixture returning deterministic complex tensors (shape: batch×N×N)
- Added `stitch_train_results` fixture with mock Lightning module handle
- Updated 4 tests to supply `train_results` and assert actual outputs instead of NotImplementedError
- Kept 1 regression test (`test_reassemble_cdi_image_torch_guard_without_train_results`) for `train_results=None` guard

### 2. Implementation Fix
- Fixed `_ensure_container` missing `probe` argument (ptycho_torch/workflows/components.py:256-257)
- Issue: `PtychoDataContainerTorch(grouped_data)` was missing required `probe` parameter
- Fix: Extract `probe` from `data.probeGuess` and pass to constructor

### 3. Test Execution
Command: `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv`
Result: 1/8 passing (guard test ✅), 7/8 failing with TensorFlow shape error

## Blocking Issue

### TensorFlow `reassemble_position` Shape Mismatch

**Error Message:**
```
ValueError: Input tensor `cond/zeros:0` enters the loop with shape (None, None, 1), 
but has shape (None, None, 64) after one iteration. 
To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape.
```

**Error Location:** `ptycho/tf_helper.py:1180` in `_streaming` function

**Root Cause Analysis:**
1. Mock Lightning module returns shape `(batch, 64, 64)` complex tensors
2. Implementation adds channel dimension: `obj_tensor_full.unsqueeze(1)` → `(n_samples, 1, 64, 64)`
3. TensorFlow `reassemble_position` expects `(n_samples, 1, N, N)` but is seeing channel dimension 64 instead of 1
4. This triggers shape invariant violation in TensorFlow's tf.while_loop

**Hypotheses:**
- The tensor shape conversion from PyTorch→NumPy may be transposing dimensions incorrectly
- The `reassemble_position` function expects a different memory layout (NCHW vs NHWC)
- The mock's output shape doesn't match real PtychoPINN_Lightning output contract

## Analysis Required

To unblock Phase C4, need to:
1. Investigate actual PtychoPINN_Lightning output shape from real training
2. Check if TensorFlow `reassemble_position` has channel-first vs channel-last assumptions
3. Validate PyTorch→NumPy conversion preserves expected layout
4. Consider whether MVP should use simplified mock with explicit shape handling

## Next Actions

**Option A (Debug Shape Contract):**
- Add shape logging to `_reassemble_cdi_image_torch` before `reassemble_position` call
- Compare with TensorFlow workflow's tensor shapes at equivalent point
- Adjust mock or add transpose/reshape before delegating to TensorFlow helper

**Option B (Defer to Phase E):**
- Mark Phase C4 as "implementation complete, test validation deferred"
- Add finding to docs/findings.md: "REASSEMBLY-001: PyTorch→TF reassembly shape contract TBD"
- Prioritize native PyTorch reassembly in Phase E to avoid TensorFlow interop issues

**Recommendation:** Option A (quick debug) worth 1 more attempt; if >30min, switch to Option B and document as Phase E prerequisite.

## Artifacts

- Test file: `tests/torch/test_workflows_components.py:1076-1499` (TestReassembleCdiImageTorchGreen class)
- Implementation fix: `ptycho_torch/workflows/components.py:256-257` (probe argument)
- Test log: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log` (7 failures, 1 pass)

## Exit Criteria Not Met

❌ Phase C4 checklist C4 ("Modernize stitching tests + capture green log") blocked pending shape contract resolution.

Next loop: Debug shape issue or defer to Phase E with documented finding.
