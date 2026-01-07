# Implementation Plan - Implement Lazy Tensor Allocation

- **Initiative:** FEAT-LAZY-LOADING-001
- **Status:** in_progress
- **Owner:** Ralph
- **Priority:** High
- **Working Plan:** `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- **Reports Hub:** `plans/active/FEAT-LAZY-LOADING-001/reports/`

## Context Priming
- `docs/specs/spec-ptycho-workflow.md` (Normative): See "Resource Constraints" section.
- `docs/findings.md` (Knowledge Base): See `PINN-CHUNKED-001` (OOM error details).
- `ptycho/loader.py` (Target): The `PtychoDataContainer` class.

## Problem Statement
`PtychoDataContainer.__init__` eagerly converts all input arrays (diffraction patterns, probe, object) into TensorFlow tensors using `tf.convert_to_tensor`. This forces the entire dataset into GPU memory immediately, causing OOM errors for large datasets (like the dense fly64 study) even if downstream processing attempts to chunk the data.

## Objectives
1.  **Lazy Storage:** `PtychoDataContainer` stores inputs as NumPy arrays or memory-mapped files.
2.  **On-Demand Tensorification:** Conversion to TF Tensors happens only when data is requested for a training step or inference batch.
3.  **API Compatibility:** Maintain the property interface (`.X`, `.Y`) but make them return streamed/batched results or warn on full access.

## Phases

### Phase A: Reproduction & Baseline ✅ COMPLETE
**Goal:** Create a failing test case that OOMs with the current architecture.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| A1 | Create OOM reproduction test | [x] | Created `tests/test_lazy_loading.py` with `test_memory_usage_scales_with_dataset_size` (3 param cases) and `test_oom_with_eager_loading` (skipped by default). Tests verify eager loading behavior. Commit 1d4b09f4. |

**Evidence:** 3 passed, 3 skipped (5.13s). See `reports/2026-01-07T210000Z/pytest_memory_scaling.log`.

### Phase B: Refactor Container Architecture ← CURRENT
**Goal:** Implement the Lazy Container.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| B1 | Modify `__init__` signature | [ ] | Store as `self._X_np`, `self._Y_I_np`, etc. using `X.numpy() if tf.is_tensor(X) else X` pattern. Add `self._tensor_cache = {}`. |
| B2 | Add lazy property accessors | [ ] | Convert `.X`, `.Y_I`, `.Y_phi`, `.coords_nominal`, `.coords_true`, `.probe` to `@property` with caching. |
| B3 | Add `as_tf_dataset()` method | [ ] | Returns `tf.data.Dataset` for memory-efficient batched access. Uses generator pattern. |
| B4 | Update `load()` function | [ ] | Remove eager `tf.convert_to_tensor()` calls at lines 309-311, 325. Pass NumPy arrays to container. |
| B5 | Update tests | [ ] | Activate `test_lazy_loading_avoids_oom` and `test_lazy_container_backward_compatible`. |
| B6 | Run regression tests | [ ] | Verify `test_model_factory.py` still passes. |

### Phase C: Integration
**Goal:** Connect new container to Training/Inference loops.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| C1 | Update `train_pinn.py` | [ ] | Optionally use `.as_tf_dataset()` for large datasets. |
| C2 | Update `compare_models.py` | [ ] | Enable chunked PINN inference via lazy loading. |

## Exit Criteria
1. `tests/test_lazy_loading.py::test_lazy_loading_avoids_oom` passes
2. `tests/test_lazy_loading.py::test_lazy_container_backward_compatible` passes
3. `tests/test_model_factory.py` continues to pass (no regression)
4. STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 can proceed with dense dataset (OOM resolved)
