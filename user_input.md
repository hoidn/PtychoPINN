# Reviewer Findings — 2026-01-20T083025Z

## Summary
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md` ends with quick-links to `CONFIGURATION_GUIDE.md` and `data_contracts.md`, but those files do not exist (the real docs live at `docs/CONFIGURATION.md` and `../specs/data_contracts.md`).
- `prompts/arch_writer.md` instructs authors to cite `/specs/spec-ptycho-workflow.md#data-loading` and `/specs/spec-ptycho-interfaces.md#data-contracts`, yet those anchors are not defined in the specs, so the links 404.
- `plans/active/DEBUG-SIM-LINES-DOSE-001` shows Phase A (A0/A1b/A2) and Phase C4d unchecked even though later phases are underway and analyzer evidence already exists under `reports/2026-01-20T143000Z/`, so the plan no longer captures reality.

## Evidence
- Missing files referenced in `docs/GRIDSIZE_N_GROUPS_GUIDE.md:159-160`; confirmed only `docs/CONFIGURATION.md` and `specs/data_contracts.md` exist.
- `prompts/arch_writer.md:257-258,632-633` contain the broken anchors; `specs/spec-ptycho-workflow.md` and `specs/spec-ptycho-interfaces.md` have no `## Data Loading` or `## Data Contracts` headings.
- Plan misalignment shown at `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:91-114` (Phase A boxes unchecked) and line 194 (C4d unchecked) while analyzer artifacts live in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/`.

## Plan / Tracking Updates Needed
1. Document fixes: adjust `docs/GRIDSIZE_N_GROUPS_GUIDE.md` to link to actual files (`docs/CONFIGURATION.md`, `../specs/data_contracts.md`) and add regression tests or linting to catch missing targets.
2. Prompt maintenance: update `prompts/arch_writer.md` to reference real anchors (or add the missing sections/anchors to the specs) so instructions stay actionable.
3. Plan synchronization: either close or explicitly waive Phase A items in `plans/active/DEBUG-SIM-LINES-DOSE-001`, and mark C4d complete (with evidence pointers) so the plan reflects the current artifact state before new work begins.

## Requested Next Steps
1. Supervisor to log/assign a quick documentation task (could fold into existing planning) covering the broken links and prompt anchors.
2. DEBUG-SIM-LINES-DOSE-001 owner to update the implementation plan + summary to acknowledge skipped Phase A work, close C4d, and describe the next concrete implementation step beyond analyzer instrumentation.
