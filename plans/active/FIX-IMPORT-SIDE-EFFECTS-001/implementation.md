# Implementation Plan - Remove Global State Side-Effects in Model

- **Initiative:** FIX-IMPORT-SIDE-EFFECTS-001
- **Status:** superseded
- **Superseded By:** REFACTOR-MODEL-SINGLETON-001
- **Owner:** Ralph
- **Priority:** N/A (see superseding initiative)
- **Working Plan:** `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/implementation.md`
- **Reports Hub:** `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/reports/`

> **Note (2026-01-06):** This initiative has been superseded by REFACTOR-MODEL-SINGLETON-001,
> which provides a more comprehensive analysis including the non-XLA translation bug fix,
> detailed module-level variable inventory, and phased implementation approach.
> See `plans/active/REFACTOR-MODEL-SINGLETON-001/implementation.md` for the active plan.

## Context Priming
- `docs/specs/spec-ptycho-config-bridge.md` (Normative): Defines "Compliance & Prohibitions" regarding import-time side effects.
- `docs/debugging/QUICK_REFERENCE_PARAMS.md` (Guide): Explains the correct `update_legacy_dict` lifecycle.
- `ptycho/model.py` (Target): The module currently violating the spec.

## Problem Statement
The `ptycho/model.py` module executes `params.get()` calls at the top level (import time) to initialize global variables (`tprobe`, `N`, `gridsize`). This creates a race condition where the module latches onto default parameter values before the configuration bridge (`update_legacy_dict`) runs, causing "Schrödinger's Config" bugs and blocking proper PyTorch parity.

## Objectives
1.  **Compliance:** Eliminate all top-level `params.get()` calls in `ptycho/model.py`.
2.  **Explicit Initialization:** Ensure `autoencoder` and `diffraction_to_obj` are only instantiated inside factory functions or after explicit configuration.
3.  **Safety:** Add a regression test ensuring `import ptycho.model` does not access `ptycho.params.cfg`.

## Phases

### Phase A: Audit & Safety Guard
**Goal:** Prove the defect exists and prevent regression.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| A1 | Create import-safety regression test | [ ] | Create `tests/test_model_import_safety.py`. It should mock `ptycho.params.get` and assert it is NOT called during `import ptycho.model`. |
| A2 | Audit consumers of module-level models | [ ] | Search repo for usages of `ptycho.model.autoencoder` (the global instance). List them in summary. |

### Phase B: Refactor Core Model
**Goal:** Move global state into local scopes.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| B1 | Move `tprobe` / `probe_mask` initialization | [ ] | Move these inside `ProbeIllumination.__init__` or the model factory. |
| B2 | Move global model instantiation | [ ] | Deprecate the global `autoencoder` variable. Replace with a lazy property or force users to use `create_model_with_gridsize`. |
| B3 | Verify Factory Pattern | [ ] | Ensure `create_model_with_gridsize` passes parameters explicitly down to layers. |

### Phase C: Legacy Compatibility & Verification
**Goal:** Ensure existing scripts don't break.
| ID | Task Description | State | How/Why & Guidance |
|---|---|---|---|
| C1 | Update Legacy Training Script | [ ] | Check `scripts/training/train.py`. Ensure it uses the factory or triggers the lazy global after config sync. |
| C2 | Verify Parity | [ ] | Run `tests/test_integration_workflow.py` to ensure end-to-end flows still work. |
