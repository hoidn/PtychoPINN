# Implementation Plan (Phased)

## Initiative
- ID: FIX-DEVICE-TOGGLE-001
- Title: Remove CPU/GPU toggles (GPU-only PyTorch execution)
- Owner: Codex
- Spec Owner: specs/ptychodus_api_spec.md
- Status: pending

## Goals
- Remove CPU/GPU toggles from CLI and execution config so GPU is the only supported runtime path.
- Enforce GPU-only execution with clear, actionable errors when CUDA is unavailable.
- Align specs, docs, and tests with the GPU-only contract while preserving CONFIG-001 compliance.

## Phases Overview
- Phase A — Discovery + policy alignment: inventory toggles, define GPU-only contract, capture test strategy.
- Phase B — Implementation: remove toggles, enforce GPU requirement, update execution config + tests.
- Phase C — Docs + validation: update specs/docs, run tests, archive evidence.

## Exit Criteria
1. CLI interfaces no longer expose `--device`, `--accelerator`, or `--torch-accelerator` toggles; GPU-only behavior is enforced with a clear error on missing CUDA.
2. Execution config defaults to GPU-only values with no CPU fallback path.
3. Specs/docs updated so no user-facing CPU toggle remains.
4. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - No regression in existing test suite (full suite green or known-skip documented)
   - Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` updated
   - Logs saved to `plans/active/FIX-DEVICE-TOGGLE-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `specs/ptychodus_api_spec.md §4.9` (PyTorch Execution Configuration Contract) + §7 CLI flags
- [ ] **Fix-Plan Link:** `docs/fix_plan.md — [FIX-DEVICE-TOGGLE-001]`
- [ ] **Finding/Policy ID:** `POLICY-001`, `CONFIG-002`
- [ ] **Test Strategy:** `plans/active/FIX-DEVICE-TOGGLE-001/test_strategy.md`

## Spec Alignment
- **Normative Spec:** `specs/ptychodus_api_spec.md`
- **Key Clauses:** §4.9 execution config fields + validation; §7 CLI execution flag contract

## Testing Integration

**Principle:** Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

**Format for checklist items:**
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

**Complex testing needs:** Use `plans/active/FIX-DEVICE-TOGGLE-001/test_strategy.md`.

## Context Priming (read before edits)
- Primary docs/specs to re-read: `docs/index.md`, `docs/workflows/pytorch.md`, `specs/ptychodus_api_spec.md`, `docs/cli_flags_quick_reference.md`, `docs/cli_config_dataflow.md`, `docs/pytorch_cli_inventory.md`, `docs/PYTORCH_CLI_ANALYSIS_README.md`, `docs/CLI_FLAGS_MAPPING.md`
- Required findings/case law: `POLICY-001`, `CONFIG-002` (docs/findings.md)
- Related telemetry/attempts: `plans/active/ADR-003-BACKEND-API/` (CLI execution config history)
- Data dependencies to verify: None (behavioral + CLI-only change)

## Phase A — Discovery + Contract Alignment
### Checklist
- [ ] A0: Inventory GPU/CPU toggle touchpoints in code/docs/tests; capture file list in plan notes.
      Test: N/A: discovery task
- [ ] A1: Define GPU-only contract + failure behavior (no CUDA -> actionable error); confirm spec updates required.
      Test: N/A: policy/contract definition
- [ ] A2: Draft test strategy and link it in `docs/fix_plan.md`.
      Test: N/A: planning artifact

### Notes & Risks
- Risk: CPU-only environments will fail unless tests gate on CUDA availability or mock it.

## Phase B — Implementation
### Checklist
- [ ] B1: Remove CLI toggles from `ptycho_torch/train.py`, `ptycho_torch/inference.py`, `scripts/training/train.py`, and `scripts/inference/inference.py`; enforce GPU-only execution with explicit CUDA checks.
      Test: `tests/torch/test_cli_train_torch.py`, `tests/torch/test_cli_inference_torch.py`, `tests/scripts/test_training_backend_selector.py`, `tests/scripts/test_inference_backend_selector.py`
- [ ] B2: Remove accelerator resolution helpers + CPU fallback in `ptycho_torch/cli/shared.py`; update `PyTorchExecutionConfig` and any callers to default to GPU-only values.
      Test: `tests/torch/test_cli_shared.py`, `tests/torch/test_execution_config_defaults.py`, `tests/torch/test_integration_workflow_torch.py`
- [ ] B3: Update backend selection + runtime device mapping to assume GPU (no CPU fallback); ensure error messages mention GPU requirement.
      Test: `tests/scripts/test_inference_backend_selector.py::TestPyTorchExecutionConfig` (or updated selector)

### Notes & Risks
- Avoid touching stable physics files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Risk: Removing flags may break downstream scripts; document migration guidance.

## Phase C — Docs + Validation
### Checklist
- [ ] C1: Update specs/docs to remove CPU toggle references and note GPU-only requirement.
      Test: N/A: documentation-only
- [ ] C2: Update test registry docs (if tests move/change) and capture pytest evidence with archived logs.
      Test: `pytest --collect-only <selectors>` + `pytest <selectors>` (see test strategy)

### Notes & Risks
- Ensure examples remove `--device`/`--accelerator` flags and use `CUDA_VISIBLE_DEVICES` for GPU selection if needed.

## Artifacts Index
- Reports root: `plans/active/FIX-DEVICE-TOGGLE-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
