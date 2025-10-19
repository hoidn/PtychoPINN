# Phase E3 — Documentation & Handoff Planning (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration
- Phase Goal: Align project documentation, normative specs, and downstream initiative handoffs with the newly operational PyTorch backend so Ptychodus can surface backend selection without ambiguity.
- Dependencies:
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` (Phase E checklist and historical guidance)
  - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md` (latest parity summary + metrics)
  - `docs/workflows/pytorch.md` §5–§11 (current workflow doc after Phase D2 updates)
  - `specs/ptychodus_api_spec.md` §4 (reconstructor contract expectations)
  - `plans/pytorch_integration_test_plan.md` (TEST-PYTORCH-001 charter linkage)
  - Findings: POLICY-001 (PyTorch mandatory), FORMAT-001 (legacy NPZ transpose guard)
- Artifact Discipline: Capture all Phase E3 artifacts under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/`. Each subtask should add a short `summary.md` (or equivalent) and reference generated diffs/logs. Link every artifact from docs/fix_plan.md attempts.

### Phase A — Gap Assessment & Scope Confirmation
Goal: Confirm which docs/specs still describe TensorFlow-only behaviour and enumerate the exact updates required for dual-backend parity messaging.
Prereqs: Review parity_update.md to understand the current PyTorch feature set and visit policy findings (POLICY-001, FORMAT-001).
Exit Criteria: Inventory document capturing deltas, target file anchors, and ownership so downstream edits stay focused.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Catalogue documentation gaps | [ ] | Read `docs/workflows/pytorch.md`, `README.md`, `docs/ARCHITECTURE.md`, and `docs/DEVELOPER_GUIDE.md` sections that mention backend selection or workflow status. Record outdated statements (e.g., TensorFlow-only assumptions, references to NotImplementedError) in `reports/2025-10-19T205832Z/phase_e3_docs_inventory.md`. |
| A2 | Audit spec & findings coverage | [ ] | Compare `specs/ptychodus_api_spec.md` §4 and `docs/findings.md` entries against Phase E expectations (backend flag, fail-fast policy, TEST-PYTORCH-001 requirements). Document necessary spec amendments in the same inventory file with line anchors. |
| A3 | Map downstream handoff needs | [ ] | Coordinate with TEST-PYTORCH-001 plan (`plans/active/TEST-PYTORCH-001/implementation.md`) and note which selectors, fixtures, or governance decisions must be surfaced in the eventual handoff (`phase_e_handoff.md`). Capture notes + owners in the inventory. |

### Phase B — Documentation Updates (Developer-Facing)
Goal: Update developer-facing documentation so engineers can enable the PyTorch backend through Ptychodus with clear instructions and warnings.
Prereqs: Phase A inventory complete with prioritized doc edits.
Exit Criteria: Documentation updated, change log summarized, and references added to inventory + fix_plan.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Update workflow & architecture docs | [ ] | Apply edits to `docs/workflows/pytorch.md` (new subsection outlining backend selection flow, pointer to Phase E tests) and `docs/architecture.md` (component diagram note for backend switch). Summarize edits in `reports/2025-10-19T205832Z/phase_e3_docs_update.md`. |
| B2 | Refresh agent guidance & onboarding docs | [ ] | Ensure `CLAUDE.md` and `README.md` mention backend selection flag, PyTorch requirement, and where parity evidence lives. Include explicit CONFIG-001 reminder for PyTorch path. Capture diff anchors + rationale in the same update report. |
| B3 | Verify documentation cross-links | [ ] | Run `rg "NotImplementedError" docs/workflows/pytorch.md docs/architecture.md` to confirm legacy warnings removed or updated. Update the report with command output and confirmation notes. |

### Phase C — Specification & Knowledge Base Synchronization
Goal: Amend normative specs and findings so backend selection, fail-fast behaviour, and parity expectations are codified.
Prereqs: Phase A inventory ready; confirm no conflicting ADRs.
Exit Criteria: Spec patch drafted, reviewed, and referenced from fix_plan; findings ledger updated if new policy IDs required.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Draft spec amendments | [ ] | Update `specs/ptychodus_api_spec.md` §4.1–§4.6 with PyTorch backend selection details (config flag, dispatch behaviour, persistence expectations). Stage working copy in `reports/2025-10-19T205832Z/phase_e3_spec_patch.md` before editing the spec. |
| C2 | Align knowledge base | [ ] | If new policies arise (e.g., backend flag usage), append entries to `docs/findings.md` with evidence paths. Document outcome in the spec patch file. |
| C3 | Review with governance log | [ ] | Cross-check `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` and `phase_f_torch_mandatory.md` to verify no contradictions. Note confirmations (or discrepancies) in the spec patch report. |

### Phase D — TEST-PYTORCH-001 Handoff Package
Goal: Produce a concise handoff describing how TEST-PYTORCH-001 should validate PyTorch integration inside Ptychodus and track ongoing maintenance.
Prereqs: Phases B/C updates drafted so references are stable.
Exit Criteria: Handoff document authored, plan checklist updated, and fix_plan attempts logged with artifact links.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Author handoff brief | [ ] | Write `reports/2025-10-19T205832Z/phase_e3_handoff.md` summarizing backend selection flag defaults, pytest selectors, coordination points with TEST-PYTORCH-001 (especially Phase D3 CI guidance), and ownership matrix. |
| D2 | Update plan & ledger references | [ ] | Mark `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` E3 rows with `[x]` once documents are updated; append docs/fix_plan.md Attempt entry with artifact links (`phase_e3_docs_update.md`, `phase_e3_spec_patch.md`, `phase_e3_handoff.md`). |
| D3 | Define follow-up checks | [ ] | List verification steps (e.g., rerun `pytest tests/torch/test_backend_selection.py -vv`) and scheduling cadence in the handoff doc. Note whether TEST-PYTORCH-001 should extend its plan (e.g., new Phase D4). |

## References
- `plans/active/INTEGRATE-PYTORCH-001/implementation.md`
- `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md`
- `plans/pytorch_integration_test_plan.md`
- `plans/active/TEST-PYTORCH-001/implementation.md`
- `specs/ptychodus_api_spec.md`
- `docs/workflows/pytorch.md`
- `docs/findings.md#POLICY-001`
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md`

