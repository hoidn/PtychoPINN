Summary: Apply the Phase B redline to refresh `plans/ptychodus_pytorch_integration_plan.md` for the rebased PyTorch backend.
Mode: Docs
Focus: INTEGRATE-PYTORCH-000 — Pre-refresh Planning for PyTorch Backend Integration
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/{plan_redline.md,summary.md}
Do Now: INTEGRATE-PYTORCH-000 — Phase B.B2 canonical plan update; edit `plans/ptychodus_pytorch_integration_plan.md` to implement the redline items (sections 0–5, deliverables, risks) and cite the refreshed sources.
If Blocked: Capture unresolved decisions in `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/summary.md`, add a note under Attempts History in docs/fix_plan.md, and stop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-000/implementation.md:24 — Phase B exit requires the canonical plan edits before governance handoff.
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/plan_redline.md:1 — Outlines the exact changes demanded by Critical Deltas 1‑5.
- plans/ptychodus_pytorch_integration_plan.md:1 — Source document to align with rebased PyTorch modules and parity map.
- specs/ptychodus_api_spec.md:20 — Contract citations that must stay referenced when updating configuration and workflow sections.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/summary.md:18 — Ensures the refreshed plan speaks to the Phase B blockers identified by the execution initiative.
How-To Map:
- Review `plan_redline.md` and list its five revision bullets; keep it open while editing the canonical plan.
- Update Phase 0–4 subsections in `plans/ptychodus_pytorch_integration_plan.md` to document the API layer choice, config bridge work, data pipeline shims, reassembly parity, and persistence strategy; add cross-references to specs and reports.
- Refresh the deliverables list and risks section to mention Lightning/MLflow policy, RawDataTorch shim, and persistence adapter decisions.
- Run `git diff plans/ptychodus_pytorch_integration_plan.md` to verify edits are scoped to documentation and capture key decisions in commit message notes for supervisor review.
Pitfalls To Avoid:
- Do not modify any production Python modules this loop.
- Keep the plan in phased form; retain tables and headings while updating content.
- Avoid removing existing TensorFlow references—add PyTorch context alongside them.
- Cite spec sections (`specs/ptychodus_api_spec.md`) when describing contracts to prevent drift.
- Do not invent new directory structures for artifacts; reference existing report paths.
- Leave `plan_redline.md` untouched; only consume it.
- Note open questions rather than resolving them silently.
- Maintain ASCII characters only when editing docs.
- Ensure parity risks mention testing coordination with TEST-PYTORCH-001.
- Verify links and file paths remain accurate after edits.
Pointers:
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/plan_redline.md:1
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025633Z/summary.md:1
- plans/ptychodus_pytorch_integration_plan.md:1
- specs/ptychodus_api_spec.md:20
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:24
Next Up: 1) Phase B.B3 stakeholder brief (summarize decisions + open questions); 2) Phase C governance updates once canonical plan lands.
