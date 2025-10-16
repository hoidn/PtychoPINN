# Plan — TEST-SUITE-TRIAGE: Restore Test Suite Signal

## Context
- Initiative: TEST-SUITE-TRIAGE (Full pytest sweep, failure triage, remediation sequencing)
- Phase Goal: Deliver reliable guidance for Ralph to run the authoritative pytest suite, classify every failure, and sequence fixes.
- Dependencies: `docs/TESTING_GUIDE.md` (authoritative commands), `docs/debugging/debugging.md` (triage SOP), `docs/findings.md#BUG-TF-001` (historical gridsize failure), `CLAUDE.md` (critical directives), `specs/data_contracts.md` (data validation prerequisite).

## Phase A — Establish Baseline Failure Ledger
Goal: Capture the current pytest failure surface with reproducible artifacts.
Prerqs: Verify environment sync (`git status` clean, conda env ready); acknowledge CLAUDE directives on data handling.
Exit Criteria: `pytest tests/` run captured under `plans/active/TEST-SUITE-TRIAGE/reports/<timestamp>/pytest.log`; failures enumerated in `summary.md` with module::test identifiers and trace snippets.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Prep environment + document configuration | [ ] | Record python version, env activation commands in `reports/<ts>/env.md`. Reference `docs/TESTING_GUIDE.md#Running the Full Test Suite`. |
| A2 | Execute pytest sweep | [ ] | Run `pytest tests/ -vv` from repo root; tee output to `reports/<ts>/pytest.log`. |
| A3 | Extract failure manifest | [ ] | Parse failing selectors into bullet list within `reports/<ts>/summary.md`; include last failing stack frame + message for each. |

## Phase B — Failure Classification & Ownership
Goal: Map each failure to bug/deprecation buckets and identify owning component.
Prerqs: Phase A artifacts published; verify specs/data contracts for data-related tests.
Exit Criteria: `summary.md` updated with table capturing classification, suspected root cause, and recommended owner; each failure cross-referenced to spec/plan item or marked "legacy / remove" for deprecation proposals.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Cross-reference knowledge base | [ ] | For each failure, check `docs/findings.md` for prior art (e.g., BUG-TF-001) before deeper triage. |
| B2 | Classify failure type | [ ] | Annotate in `summary.md` whether failure == implementation bug, flaky, or deprecated test; cite evidence (spec section, log excerpt). |
| B3 | Identify remediation pathway | [ ] | Record whether fix demands new TDD cycle, data correction, or test retirement request; note follow-up plan item IDs when created. |

## Phase C — Remediation Sequencing & Delegation
Goal: Produce actionable backlog updates (docs/fix_plan.md + child plans) for prioritized fixes.
Prerqs: Phase B classification complete and validated.
Exit Criteria: docs/fix_plan.md updated with new items for each actionable failure cluster; plan cross-links recorded; blocking issues flagged; ready-to-run guidance placed into `input.md` for Ralph.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Prioritize fixes | [ ] | Rank failure clusters by severity (blocking CI, critical functionality, etc.) referencing specs/arch. |
| C2 | Draft fix_plan entries | [ ] | Add new entries under `docs/fix_plan.md` with spec references + reproduction commands. |
| C3 | Prepare delegation packet | [ ] | Summarize next actionable step for Ralph (e.g., targeted pytest command) in `input.md`, ensuring artifacts path logged. |

## Reporting & Artifacts
- Artifact root: `plans/active/TEST-SUITE-TRIAGE/reports/`
- Each Phase A/B/C loop uses ISO timestamp subdirectories (e.g., `2025-10-16T000000Z`).
- Always capture environment metadata (`python -m pip freeze > requirements_snapshot.txt`) when new failures appear.

## Risks & Mitigations
- High failure volume may exceed single loop → Mitigate via per-cluster follow-up plans.
- Deprecated tests without clear spec reference → escalate via fix_plan notes before removal.
- Data regressions masquerading as code bugs → enforce spec validation first per `specs/data_contracts.md`.

## Decision Rules
- Treat tests touching stable modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) as sacred—if they fail, default to data/config causes before modifying core logic.
- Remove/skip tests only after documenting rationale in fix_plan and ensuring docs point to replacement coverage.

## Verification Checklist
- [ ] Phase A artifacts committed with timestamped directory.
- [ ] Summary ledger maintained across loops (append-only per timestamp).
- [ ] fix_plan entries synced with latest classification results.

