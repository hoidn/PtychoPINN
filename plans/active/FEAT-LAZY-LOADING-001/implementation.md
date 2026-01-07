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

### Phase B: Refactor Container Architecture ✅ COMPLETE
**Goal:** Implement the Lazy Container.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| B1 | Modify `__init__` signature | [x] | Store as `self._X_np`, `self._Y_I_np`, etc. using `X.numpy() if tf.is_tensor(X) else X` pattern. Add `self._tensor_cache = {}`. Commit 37985157. |
| B2 | Add lazy property accessors | [x] | Converted `.X`, `.Y_I`, `.Y_phi`, `.coords_nominal`, `.coords_true`, `.probe` to `@property` with caching. |
| B3 | Add `as_tf_dataset()` method | [x] | Returns `tf.data.Dataset` for memory-efficient batched access. Uses generator pattern (ptycho/loader.py:255-321). |
| B4 | Update `load()` function | [x] | Removed eager `tf.convert_to_tensor()` calls. Pass NumPy arrays with `.astype(np.float32)` (ptycho/loader.py:474-478). |
| B5 | Update tests | [x] | All 5 Phase B tests active and passing: `test_lazy_loading_avoids_oom`, `test_lazy_container_backward_compatible`, `test_lazy_caching`, `test_tensor_input_handled`, `test_len_method`. |
| B6 | Run regression tests | [x] | `test_model_factory.py` passes (3/3, 25.67s). No regressions. |

**Evidence:** 8 passed, 1 skipped (6.44s). See `reports/2026-01-07T220000Z/pytest_phase_b.log`.

### Phase C: Integration ← CURRENT
**Goal:** Connect new container to Training/Inference loops.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| C1 | Add `train_with_dataset()` to model.py | [ ] | New function that accepts `tf.data.Dataset` from `as_tf_dataset()`. Use `model.fit(dataset, ...)` directly. Add heuristic threshold (e.g., n_samples > 10000) to auto-select streaming. |
| C2 | Update `ptycho.model.train()` | [ ] | Add optional `use_streaming: bool = None` parameter. When True or auto-detected, use `as_tf_dataset()`. Otherwise use current `prepare_inputs/prepare_outputs` pattern. |
| C3 | Update `compare_models.py` | [ ] | Enable chunked PINN inference by iterating over container without loading full `.X` tensor. Use `_X_np` slicing with chunk-wise tensorification. |
| C4 | Add integration tests | [ ] | Test that training works with `use_streaming=True` on small datasets. Verify metrics are equivalent to current path. |

## Exit Criteria
1. `tests/test_lazy_loading.py::test_lazy_loading_avoids_oom` passes
2. `tests/test_lazy_loading.py::test_lazy_container_backward_compatible` passes
3. `tests/test_model_factory.py` continues to pass (no regression)
4. STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 can proceed with dense dataset (OOM resolved)
