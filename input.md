Summary: Draft Phase C4 CLI integration blueprint (ADR-003-BACKEND-API)
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/{plan.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API C4-planning @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — author Phase C4 CLI integration plan (flag mapping, workflow entry points, reproduction selectors, artifact checklist); tests: none.
2. ADR-003-BACKEND-API C4-crossref @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/summary.md — capture narrative summary, update implementation/phase plan checklists with new references, and log Attempt in docs/fix_plan.md (tests: none).

If Blocked: Document incomplete flag inventory or conflicting CLI behaviour in summary.md, note blockers in phase_c_execution/summary.md and docs/fix_plan.md, leave C4 rows at [P].

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — C4 checklist awaiting detailed plan.
- ptycho_torch/train.py:420-560 & ptycho_torch/inference.py:150-260 — authoritative CLI surfaces requiring mapping.
- specs/ptychodus_api_spec.md §4.8 & docs/workflows/pytorch.md §13 — contract references for backend/CLI alignment.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md — capture deferred knobs that must appear in C4 plan.

How-To Map:
- Create directory `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/` with `plan.md` (phased checklist per prompts/plan template) and `summary.md`.
- Enumerate CLI flags / config knobs (train & inference) with file:line citations; map each to execution config fields and override precedence (reference `override_matrix.md`).
- Define targeted pytest selectors for future execution: e.g., `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k execution_config -vv` (stub if test missing); include in plan checklist.
- Update `phase_c_execution/plan.md` C4 rows to reference the new plan directory; mark planning subtasks `[x]` only when artefacts saved.
- Append new Attempt entry in docs/fix_plan.md noting plan artefacts and outstanding decisions.

Pitfalls To Avoid:
- Do not modify production code or tests in this loop; planning artifacts only.
- Keep new plan ASCII; avoid copying large logs into docs.
- Ensure artifact paths use ISO timestamps and live under the specified directory.
- Reference authoritative docs (specs/ptychodus_api_spec.md, docs/workflows/pytorch.md) when making decisions.
- Do not promise CLI flag behaviour that contradicts CONFIG-001 or POLICY-001.
- Avoid inventing pytest selectors without verifying target files; cite source modules.
- Maintain CONFIG-001 ordering in proposed workflows (update_legacy_dict before torched imports).
- Record deferred knobs explicitly rather than silently dropping them.
- Keep notes about existing train_debug.log hygiene—no new root-level files.
- Follow TDD policy even in planning: identify RED selectors to be authored later.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md:56
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md:1
- ptycho_torch/train.py:400
- ptycho_torch/inference.py:149
- specs/ptychodus_api_spec.md:226

Next Up:
1. Execute Phase C4 implementation tasks (wiring CLI flags + tests) once plan approved.
