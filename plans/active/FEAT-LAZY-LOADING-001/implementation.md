# Implementation Plan - Implement Lazy Tensor Allocation

- **Initiative:** FEAT-LAZY-LOADING-001
- **Status:** pending
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

### Phase A: Reproduction & Baseline
**Goal:** Create a failing test case that OOMs with the current architecture.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| A1 | Create OOM reproduction script | [ ] | Create `tests/repro_oom.py`. Attempt to load a synthetic dataset larger than available VRAM (e.g., 20k images). Assert it fails. |

### Phase B: Refactor Container Architecture
**Goal:** Implement the Lazy Container.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| B1 | Modify `__init__` signature | [ ] | Allow `PtychoDataContainer` to accept/store numpy arrays without immediate conversion. |
| B2 | Implement Batch Generator | [ ] | Add `.as_dataset(batch_size)` method returning a `tf.data.Dataset` generator. |
| B3 | Implement Lazy Properties | [ ] | Change `.X` to a property that returns the full tensor (for legacy compat) but logs a warning about memory usage. |

### Phase C: Integration
**Goal:** Connect new container to Training/Inference loops.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| C1 | Update `train_pinn.py` | [ ] | Modify training loop to consume the `.as_dataset()` generator. |
| C2 | Update `compare_models.py` | [ ] | Ensure chunked inference utilizes the lazy loading capabilities. |
