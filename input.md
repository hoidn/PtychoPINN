
Summary: Capture Phase E3 gap inventory for docs/spec handoff before editing artifacts.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2 — Phase E3 inventory
Branch: feature/torchapi
Mapped tests: none — documentation inventory
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/{phase_e3_docs_inventory.md,summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS A.A1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Build documentation gap inventory covering workflow/architecture/onboarding docs (tests: none).
2. INTEGRATE-PYTORCH-001-STUBS A.A2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Capture spec & findings deltas with file:line anchors (tests: none).
3. INTEGRATE-PYTORCH-001-STUBS A.A3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md — Note TEST-PYTORCH-001 handoff requirements and owners (tests: none).

If Blocked: Record the blocker inside phase_e3_docs_inventory.md, mark the affected checklist row `[P]` in phase_e3_docs_plan.md, and summarize the issue in docs/fix_plan.md before stopping.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md: new phased checklist defines Phase A inventory scope.
- docs/fix_plan.md: latest Attempt #12 keeps focus on finishing Phase E deliverables before code changes.
- docs/workflows/pytorch.md: section 11 shows current messaging; inventory must flag remaining TensorFlow-only language.
- specs/ptychodus_api_spec.md §4.1–§4.6: normative contract needs annotations for backend selection.
- plans/active/TEST-PYTORCH-001/implementation.md §Phase D: informs handoff expectations for CI guidance.

How-To Map:
- Create or append `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_inventory.md` with three headings: Developer Docs, Spec & Findings, TEST-PYTORCH-001 Handoff.
- Run `rg -n "backend" docs/workflows/pytorch.md README.md docs/architecture.md` and log the relevant lines with notes on whether text already reflects PyTorch availability.
- Use `rg -n "NotImplementedError" docs/workflows/pytorch.md docs/architecture.md` to confirm legacy warnings; capture any remaining hits.
- Skim `specs/ptychodus_api_spec.md` lines 143-213 and note missing guidance on backend flags, fail-fast behaviour, or persistence parity.
- Review `plans/active/TEST-PYTORCH-001/implementation.md` Phase D table (lines 55-66) and list selectors/guardrails TEST-PYTORCH-001 needs surfaced in the eventual handoff.
- Summaries should reference files with file:line anchors and cite POLICY-001 / FORMAT-001 where relevant.

Pitfalls To Avoid:
- Do not edit documentation/spec files yet; this loop is inventory only.
- Avoid running tests; Mode=Docs.
- Keep new notes inside the 2025-10-19T205832Z report directory; do not create additional timestamp folders.
- Do not advance plan rows beyond Phase A until the inventory is complete and linked.
- Maintain neutral tone—document facts, not proposed fixes.
- Ensure inventory cites both the stale text and the intended remediation hint.
- Leave existing parity docs untouched; reference them instead of duplicating content.
- Preserve markdown tables when adding notes to the plan.
- Keep commands reproducible; record full command lines in the inventory file.
- Remember to update docs/fix_plan.md in a future loop after actual edits, not during this inventory pass.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md:1
- docs/workflows/pytorch.md:246
- specs/ptychodus_api_spec.md:143
- plans/active/TEST-PYTORCH-001/implementation.md:55
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md:46

Next Up: Phase E3 — execute Phase B documentation updates once inventory lands.
