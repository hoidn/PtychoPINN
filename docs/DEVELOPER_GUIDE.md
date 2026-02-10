# Unified Developer Guide for PtychoPINN

## Document Purpose

This guide explains how to work in this repository without breaking data contracts, backend parity, or legacy compatibility.

Use this document for:
- day-to-day development workflow
- architecture orientation
- config/data/testing guardrails
- code review expectations

For command recipes, see `docs/COMMANDS_REFERENCE.md`.
For full troubleshooting playbooks, see `docs/debugging/TROUBLESHOOTING.md`.

## Quick Start Checklist

Before editing code:
1. Read `docs/index.md`.
2. Read `docs/findings.md`.
3. Identify whether your change touches TensorFlow, PyTorch, or both.
4. Confirm required contracts in `specs/data_contracts.md` and relevant specs.
5. Choose a focused pytest selector and keep logs as evidence.

---

## 1. The Core Concept: A "Two-System" Architecture

This codebase still has two overlapping systems:

- Legacy system:
  - centered around global `ptycho.params.cfg`
  - older code paths and modules still read from this state directly
- Modern system:
  - dataclass configs (`TrainingConfig`, `InferenceConfig`, `ModelConfig`)
  - backend-dispatched workflows and clearer interfaces

Most regressions come from mixing these systems without syncing config.

Practical rule:
- Decide first which system your target code is in.
- For legacy-touching paths, run `update_legacy_dict(params.cfg, config)` before data loading/model work.

---

## 2. Architectural Rules and Anti-Patterns

### 2.1. Anti-Pattern: Side Effects on Import

Do not run stateful logic at import time (data loading, model creation, file I/O, hidden env setup).

Bad pattern:
- importing a module triggers work based on current global state
- behavior changes based on import order

Preferred pattern:
- keep module scope lightweight
- pass required inputs explicitly into functions
- create models/data in runtime functions, not module globals

### 2.2. Anti-Pattern: Implicit Dependencies via Global State

`params.cfg` still exists for compatibility, but new code should not add hidden dependencies.

If code must interact with legacy modules:
1. Build dataclass config.
2. Call `update_legacy_dict(params.cfg, config)`.
3. Then call legacy-dependent code.

When possible:
- pass `N`, `gridsize`, and related values as explicit arguments
- document any unavoidable `params.cfg` dependency in docstrings

### 2.3. Interpreter & Subprocess Policy (PYTHON-ENV-001)

Use `python` from PATH consistently:

- shell commands: `python -m ...`
- subprocess calls: `subprocess.run(["python", "-m", ...])`

Do not introduce repository-specific interpreter wrappers in new code/docs.

### 2.4. Stable-Module Caution

Treat core physics/model internals as stable unless your task explicitly targets them:
- `ptycho/model.py`
- `ptycho/diffsim.py`
- `ptycho/tf_helper.py`

If edits are required, add focused regression tests and document the rationale.

---

## 3. Data Pipeline Contracts

### 3.1. Data Contracts Are API

NPZ/HDF5 structures are external contracts, not implementation details.

Authoritative sources:
- `specs/data_contracts.md`
- `specs/ptychodus_api_spec.md`

Do not "fix" malformed inputs deep in the pipeline unless the contract explicitly allows adaptation.

### 3.2. Keep Shapes and Dtypes Explicit

Common regression pattern:
- implicit NumPy dtype defaults (for example, `np.zeros` creating `float64` and dropping complex parts)

Rules:
- set dtype explicitly for arrays holding complex values
- assert expected ranks/channels before model calls
- fail early with shape/dtype context in error messages

### 3.3. Tensor Format Expectations (gridsize > 1)

Key formats used throughout workflows:
- channel format: `(B, N, N, C)`
- flat format: `(B*C, N, N, 1)`

Before physics operations that expect flat patches, convert explicitly (for TensorFlow path, use helper conversions from `ptycho.tf_helper`).

### 3.4. Configuration Sync Rule (CONFIG-001)

When legacy modules are involved, call:

```python
update_legacy_dict(params.cfg, config)
```

before loading/grouping data or instantiating legacy-dependent model paths.

This is required for both TensorFlow and PyTorch workflows that bridge into shared legacy components.

### 3.5. Normalization Architecture: Three Distinct Systems

Keep these separate:

1. Physics normalization
- tied to photon-count/physics modeling
- applied where physics loss/modeling expects it

2. Statistical normalization
- standard ML preprocessing for training stability

3. Display/evaluation scaling
- visualization or report formatting only

Do not mix these in a single transform step. Most "scaling bugs" are boundary violations between these systems.

---

## 4. Workflow Entry Points

Use these primary entrypoints:

- Training:
  - `scripts/training/train.py`
- Inference:
  - `scripts/inference/inference.py`

Backend selection is explicit via config/CLI (`tensorflow` or `pytorch`) and routed through backend selectors/workflow components.

For grid-lines study workflows:
- orchestrator: `scripts/studies/grid_lines_compare_wrapper.py`
- torch runner: `scripts/studies/grid_lines_torch_runner.py`

Prefer modern flags and paths; keep legacy flags only for compatibility.

---

## 5. Testing and Verification Expectations

Project standard:
- run tests with pytest selectors
- include at least one targeted selector for each change
- run integration marker for workflow-impacting changes

Primary references:
- `docs/TESTING_GUIDE.md`
- `docs/development/TEST_SUITE_INDEX.md`

Baseline expectations for production-path changes:
1. targeted unit/regression tests
2. integration selector (`-m integration` or repo alias)
3. saved logs/artifacts in the active plan/report location

Do not claim success without command evidence.

---

## 6. Logging and Output Hygiene

Use centralized logging helpers where available (`ptycho.log_config`) instead of ad-hoc `logging.basicConfig` setups.

Expected behavior:
- logs are written under the run output directory
- console verbosity is controlled by CLI flags
- debug records are reproducible and tied to run artifacts

Avoid writing root-level transient logs in new code.

---

## 7. Code Review Checklist

Use this checklist for PRs and self-review:

1. Config flow
- Are config values explicit?
- Is CONFIG-001 applied before legacy access?

2. Data contracts
- Do new/changed files still meet contract shapes/dtypes/keys?

3. Backend parity
- If a shared workflow changed, were both TF and Torch paths considered/tested?

4. Error quality
- Are failures actionable (include expected vs actual shape/value/path)?

5. Tests
- Is there at least one focused regression test?
- Were relevant integration selectors run?

6. Docs
- Are command/behavior changes reflected in docs?
- Are deprecated flags/modes clearly labeled?

---

## 8. Configuration Migration

Long-term direction:
- reduce direct `params.cfg` reads
- push explicit config arguments through module boundaries
- keep `update_legacy_dict` as a compatibility bridge, not a design target

Migration guidance:
1. Do not add new `params.get()` dependencies.
2. When touching legacy code, isolate and document remaining global reads.
3. Add typed config plumbing at call boundaries first, then remove internal globals.

---

## 9. Troubleshooting Map

Use these docs directly instead of reinventing diagnostics:

- Common failures:
  - `docs/debugging/TROUBLESHOOTING.md`
- Params/config quick checks:
  - `docs/debugging/QUICK_REFERENCE_PARAMS.md`
- Known historical issues and policy IDs:
  - `docs/findings.md`
- Data generation behavior:
  - `docs/DATA_GENERATION_GUIDE.md`
- PyTorch workflow behavior:
  - `docs/workflows/pytorch.md`

If you discover a recurring issue, add it to `docs/findings.md` with evidence.

