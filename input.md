Summary: Redline the backend API spec to document PyTorch execution config semantics and align CLI defaults with the current implementation.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E (Governance dossier E.A2)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/{spec_delta_notes.md,spec_redline.md}

Do Now:
1. ADR-003-BACKEND-API E.A2 (spec redline) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A2 — update `specs/ptychodus_api_spec.md` §§4.7–4.9 and §7 per `spec_delta_notes.md`, then capture the change summary in `spec_redline.md`; tests: none.

If Blocked: Log the blocker in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/blocker.md`, leave E.A2 `[P]`, and append the issue to docs/fix_plan Attempts History before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A2 — spec update is the gating deliverable before Phase E.B implementation work.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_delta_notes.md — provides the vetted field inventory and proposed structure for the redline.
- specs/ptychodus_api_spec.md:215-340 — current text is TensorFlow-only and lists stale CLI defaults; must align with PyTorch ExecutionConfig behavior.
- ptycho/config/config.py:178-258 — authoritative defaults/validation rules for `PyTorchExecutionConfig` to cite in the spec.
- ptycho_torch/train.py:360-460 & ptycho_torch/inference.py:420-520 — confirm CLI defaults and flag names when refreshing §7 tables.

How-To Map:
- Edit `specs/ptychodus_api_spec.md` adding a PyTorch runtime subsection (Lightning/factory requirements), a new §4.9 documenting `PyTorchExecutionConfig` fields/validation, and refreshed §7 tables (correct `--accelerator` default `auto`, mention `--no-deterministic`, `--quiet`, `--inference-batch-size`).
- Generate a concise change log in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md` (before/after bullet list + rationale for each subsection touched).
- Update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md` row E.A2 to `[x]` with the new artifact link, and append a docs/fix_plan Attempt entry once spec changes are staged.
- Keep edits within the Docs/Diff scope; no code or tests need to run. Validate Markdown tables render (pipe alignment) and ensure new section numbering matches the spec style.

Pitfalls To Avoid:
- Don’t regress TensorFlow requirements while adding PyTorch language; keep legacy guarantees intact.
- Avoid promising future knobs not yet implemented—mark execution-config backlog as forthcoming Phase E.B work instead.
- Ensure CONFIG-001 and POLICY-001 statements remain explicit (no vague references).
- Do not leave raw diffs or temp files outside the artifact directory; keep documentation under git control.
- Keep CLI default values sourced directly from code, not memory; double-check `--accelerator` default and deterministic toggles.
- Maintain consistent terminology (`PyTorchExecutionConfig`, `execution_config`, `factory`) across sections.
- Preserve spec numbering/heading hierarchy; adjust cross-references if sections shift.
- No changes to production modules or tests in this loop.
- Keep `summary.md` from prior attempt untouched; new report is `spec_redline.md`.
- When updating plan/fix_plan, retain table formatting and attempt numbering.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_delta_notes.md
- specs/ptychodus_api_spec.md:215
- ptycho/config/config.py:178
- ptycho_torch/train.py:360
- ptycho_torch/inference.py:420

Next Up: Queue E.A3 (workflow guide refresh) after spec redline lands.
