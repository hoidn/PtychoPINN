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

## C2 status checkpoint (2025-10-20)
- C2 complete: Extended test suites with RED tests for inference thin wrapper delegation. 5/5 thin wrapper tests FAILED as expected (validate_paths, _run_inference_and_reconstruct, RawData delegation not yet implemented). 7/7 shared helper tests PASSED (inference mode already supported). Artifacts: `reports/2025-10-20T115252Z/phase_d_cli_wrappers_inference_red/`.

## C3 status checkpoint (2025-10-20)
- C3 complete: Implemented inference thin wrapper with helper delegation. Applied test fixes for keyword invocation and bundle loader contract. CLI inference selector: 9/9 PASSED in 4.59s. Integration selector: 1/1 PASSED in 16.75s. Artifacts: `reports/2025-10-20T122425Z/phase_d_cli_wrappers_inference_followup/`.

## C4 status checkpoint (2025-10-20, Attempt #52)
- **C4 COMPLETE:** Updated `docs/workflows/pytorch.md:354-393` with inference thin wrapper documentation per `docs_gap_analysis.md` guidance. Changes: (i) corrected flag defaults table (`--accelerator='auto'`, `--inference-batch-size=None`), (ii) added "Helper-Based Configuration Flow" subsection documenting delegation to `cli/shared.py` helpers + `_run_inference_and_reconstruct()` orchestration with CONFIG-001 compliance guarantee, (iii) added example CLI command using minimal dataset fixture (`tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`) demonstrating current flag syntax (`--accelerator cpu --quiet`), (iv) documented expected output artifacts (amplitude/phase PNGs per `save_individual_reconstructions()` contract), (v) aligned deprecation notice with training section wording ("will be removed in Phase E (post-ADR acceptance)"), (vi) cited Phase D.C C3 GREEN evidence (9/9 tests). All five gaps identified in `docs_gap_analysis.md` addressed. No tests run (docs-only loop per input.md Mode: Docs). Artifacts: `reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/{docs_gap_analysis.md,docs_update_summary.md}`. Plan row C4 marked `[x]`.
- **Phase D.C COMPLETE:** All inference CLI thin wrapper tasks (C1 blueprint, C2 RED, C3 GREEN, C4 docs) delivered. Implementation mirrors training refactor architecture per ADR-003 Phase D blueprint. Ready for Phase D.D (smoke evidence + hygiene) or Phase E (governance/ADR formalization).

## Next steps for Ralph
1. Phase D.D (smoke evidence) — Execute documented train + inference CLI commands with minimal dataset fixture, capture runtime metrics, store logs under `reports/<timestamp>/phase_d_cli_wrappers/`.
2. Phase D.D (ledger update) — Append Attempt #52 entry to docs/fix_plan.md documenting C4 completion with artifact references, then update implementation plan Phase D rows (D1–D3) with completion state.
3. Phase D.D (hygiene + handoff) — Record hygiene commands (`git status`, cleanup checks), author `handoff_summary.md` listing remaining execution knobs (checkpoint callbacks, logger backend, scheduler) and Phase E governance inputs.

**Artifacts Created:**  
- `plan.md` — phased implementation roadmap with checklist IDs and guidance  
- `summary.md` (this file) — rationale, risks, and execution notes
