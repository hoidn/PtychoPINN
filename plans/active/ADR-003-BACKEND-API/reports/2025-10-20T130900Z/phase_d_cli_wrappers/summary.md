# Phase D Planning Summary — CLI Thin Wrappers

## Why this plan exists
- Phase C4 finished exposing execution-config CLI flags and consolidating documentation, but `ptycho_torch/train.py` and `ptycho_torch/inference.py` still contain substantial orchestration logic (RawData loading, workflow calls, validation) that duplicates behaviour covered by factories/workflow helpers.
- ADR-003 Phase D requires collapsing the CLIs into thin shims so ongoing maintenance (e.g., new execution knobs, backend governance) happens in one place.
- The new factories introduced in Phases B–C3 provide all primitives needed for a delegation-based CLI, but we need a structured TDD approach to migrate safely while honouring CONFIG-001, POLICY-001, and FORMAT-001 findings.

## Planning highlights
- **Dependencies:** Spec §4.8/§7 and workflow guide §§11–13 anchor behaviour for CLI surface, backend selection, and runtime expectations. Findings `CONFIG-001`, `POLICY-001`, `FORMAT-001` remain non-negotiable gates.
- **Decomposition:** The plan splits work into four phases (A–D) mirroring fix-plan rows D1–D3:
  - Phase A inventories current behaviour and locks decisions for legacy flags (`--device`, `--disable_mlflow`) before code edits.
  - Phase B refactors the training CLI with explicit TDD steps (blueprint → RED tests → implementation → documentation).
  - Phase C performs the same treatment for inference, with additional checks against bundle loading and integration workflow parity.
  - Phase D provides smoke evidence, ledger hygiene, and a structured handoff into Phase E governance tasks.
- **TDD Discipline:** Every implementation phase mandates RED selectors (`tests/torch/test_cli_train_torch.py`, `tests/torch/test_cli_inference_torch.py`) followed by GREEN logs and regression selectors (`tests/torch/test_workflows_components.py`, integration workflow test).
- **Artifact Hygiene:** All evidence, design notes, and logs will live under the timestamped `phase_d_cli_wrappers/` directory, maintaining traceability from docs/fix_plan Attempts history.

## Outstanding decisions / risks
- **Legacy interface sunset:** The plan records a decision checkpoint to either wrap or formally deprecate the `--ptycho_dir/--config` legacy path. Final outcome may require stakeholder sign-off (Phase E).
- **Helper module placement:** The blueprint tasks should decide whether new helpers live under `ptycho_torch/cli/` or `ptycho_torch/workflows/` to avoid circular imports.
- **MLflow + deterministic defaults:** Need to ensure CLI wrappers preserve current behaviour of `--disable_mlflow` toggles and deterministic warnings when `num_workers > 0`.
- **Integration runtime:** Smoke commands in Phase D must reuse the minimal dataset fixture to stay within CI budgets documented in TEST-PYTORCH-001 reports.

## B3/B4 status checkpoint (2025-10-20)
- B3.a–B3.d complete: helper package, execution-config validation, and training CLI thin wrapper merged with GREEN evidence (`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/`).
- B3.e complete: plan checklist updated, this summary refreshed, and docs/fix_plan Attempt #43 logged for traceability.
- B4 complete: documentation refresh, test docstring update, and artifact hygiene captured under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/`. Implementation plan D1 now `[x]`; docs/fix_plan Attempt #44 documents deliverables.

## C1 status checkpoint (2025-10-20)
- C1 complete: inference blueprint authored (`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md`, 51 KB comprehensive spec). Captured helper reuse strategy, RawData ownership decision (Option A), inference orchestration extraction (Option 2), and RED test plan (5 delegation tests + 3 inference-mode shared helper tests). Design mirrors training CLI refactor for consistency.

## Next steps for Ralph
1. Phase D.C C2 — stage RED coverage in `tests/torch/test_cli_inference_torch.py` (or companion module) capturing expected helper delegation failures before implementation; archive RED log alongside the blueprint.
2. Prepare for Phase D.C C3 by enumerating required refactors in the blueprint (delegation flow, helper reuse, warning semantics) so implementation can follow the same helper pattern as training CLI.

**Artifacts Created:**  
- `plan.md` — phased implementation roadmap with checklist IDs and guidance  
- `summary.md` (this file) — rationale, risks, and execution notes
