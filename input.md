Summary: Capture ADR-003 Phase B1 factory design artifacts before any code edits
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase B1 factory blueprint
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/{factory_design.md,override_matrix.md,open_questions.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API B1.a @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — draft `factory_design.md` with module layout, function signatures, integration call sites (CLI + workflows); tests: none.
2. ADR-003-BACKEND-API B1.b @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — build `override_matrix.md` mapping each config/execution field to its data source and default; tests: none.
3. ADR-003-BACKEND-API B1.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md — record open questions + required spec/ADR updates in `open_questions.md`; summarise outcomes + blockers in `summary.md`; tests: none.
4. Implementation sync — update `plans/active/ADR-003-BACKEND-API/implementation.md` B1 row to `[x]`, note new artifacts in summary, and append docs/fix_plan Attempt if any surprises emerge; tests: none.

If Blocked: Capture the ambiguous detail (flag name, override source, spec conflict) in `open_questions.md`, leave B1 tasks `[P]`, and note the blocker + artifact path in docs/fix_plan.md before stopping.

Priorities & Rationale:
- ptycho_torch/config_bridge.py:1 — Bridge already defines canonical translation; factories must build atop it without duplication.
- ptycho_torch/train.py:366 — Current CLI wires configs manually; design needs to replace this with factory calls.
- ptycho_torch/workflows/components.py:459 — `_train_with_lightning` constructs PyTorch configs inline; capture integration touchpoints for factory adoption.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/execution_knobs.md — Source list of execution-only knobs for override matrix.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — Reference prior CLI commitments to maintain parity.

How-To Map:
- Export authoritative commands doc: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before recording any command guidance.
- Reuse Phase A grep for flag inventory if needed (`rg --no-heading --line-number "add_argument" ptycho_torch/train.py ptycho_torch/inference.py`).
- For workflow touchpoints, inspect `_train_with_lightning` + `_reassemble_cdi_image_torch` (`sed -n '459,620p' ptycho_torch/workflows/components.py`).
- When building override matrix, cross-reference TensorFlow dataclasses (`rg "class .*Config" -n ptycho/config/config.py`) to confirm canonical field names.
- Record any unresolved spec impacts in `open_questions.md` with explicit follow-up owner (spec vs ADR team). Update `summary.md` with bullet list of deliverables + outstanding questions.

Pitfalls To Avoid:
- Do not edit production code or tests during this docs loop.
- Keep artefacts under the specified timestamp directory; avoid renaming existing Phase A files.
- Use ASCII tables in `override_matrix.md`; include file:line citations for each entry.
- Distinguish canonical config fields from execution-only knobs—avoid merging them without rationale.
- Reflect POLICY-001 and FORMAT-001 constraints explicitly where relevant.
- Do not invent CLI flags; document gaps relative to TensorFlow interface instead.
- Capture RED/Green expectations for upcoming phases but do not pre-create test files yet.
- Maintain consistency with TEST-PYTORCH-001 runtime guardrails (<90s integration) when proposing future validation steps.
- If unsure where `PyTorchExecutionConfig` should live, log options rather than choosing unilaterally.
- Update implementation plan and fix_plan in the same loop when task states change.

Pointers:
- ptycho_torch/train.py:366
- ptycho_torch/inference.py:293
- ptycho_torch/workflows/components.py:459
- ptycho_torch/config_bridge.py:1
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/plan.md

Next Up: 1. Execute B2 RED scaffold (factory skeleton + failing pytest) once design artefacts are complete.
