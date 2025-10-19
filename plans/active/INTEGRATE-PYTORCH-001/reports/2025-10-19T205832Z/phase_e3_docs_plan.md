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
| A1 | Catalogue documentation gaps | [x] | ✅ 2025-10-19 Attempt #13: Surveyed 4 files (pytorch.md, README.md, architecture.md, CLAUDE.md). Found 1 HIGH gap (missing backend selection API in pytorch.md), 3 LOW/MEDIUM enhancements. Zero NotImplementedError warnings. Evidence: `phase_e3_docs_inventory.md` §A.A1. |
| A2 | Audit spec & findings coverage | [x] | ✅ 2025-10-19 Attempt #13: Reviewed ptychodus_api_spec.md §1/§2.2/§4 and findings.md (POLICY-001, FORMAT-001). **BLOCKING:** §4.8 backend selection spec MISSING (how PtychoPINNTrainableReconstructor chooses backend). POLICY-002 placeholder recommended. Evidence: `phase_e3_docs_inventory.md` §A.A2. |
| A3 | Map downstream handoff needs | [x] | ✅ 2025-10-19 Attempt #13: Identified 4 handoff items for TEST-PYTORCH-001 Phase D3: CI execution guidance (markers/timeout/skip policy), parity selectors (6 critical tests), fixture coordination (canonical dataset approved), ownership matrix. Evidence: `phase_e3_docs_inventory.md` §A.A3. |

### Phase B — Documentation Updates (Developer-Facing)
Goal: Update developer-facing documentation so engineers can enable the PyTorch backend through Ptychodus with clear instructions and warnings.
Prereqs: Phase A inventory complete with prioritized doc edits.
Exit Criteria: Documentation updated, change log summarized, and references added to inventory + fix_plan.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Update workflow & architecture docs | [x] | ✅ 2025-10-19: Added §12 "Backend Selection in Ptychodus Integration" to `docs/workflows/pytorch.md` (108 lines) with configuration API, dispatcher routing (spec §4.8), error handling (POLICY-001), checkpoint compatibility, test selectors, and Ptychodus integration example. Added backend selector paragraph to `docs/architecture.md:13` explaining dual-backend routing and CONFIG-001 compliance. Artifacts: `reports/2025-10-19T210000Z/phase_e3_docs_update/{diff_notes.md,summary.md}`. |
| B2 | Refresh agent guidance & onboarding docs | [x] | ✅ 2025-10-19: Added PyTorch Backend Selection directive to `CLAUDE.md:61-63` (CONFIG-001 reminder + spec §4.8/workflow §12 references). Inserted `### Dual-Backend Architecture` subsection into `README.md:17-26` (TensorFlow default, PyTorch production-ready, configuration API, runtime evidence). Artifacts: `reports/2025-10-19T213900Z/phase_e3_docs_b2/summary.md`. |
| B3 | Verify documentation cross-links | [x] | ✅ 2025-10-19: Executed `rg "NotImplementedError"` verification command, captured empty output to `rg_notimplemented.log` (0 matches = PASS). Confirmed no stray stub warnings remain in workflow/architecture docs. Cross-references validated in summary.md. Artifacts: `reports/2025-10-19T213900Z/phase_e3_docs_b2/{summary.md,rg_notimplemented.log}`. |

### Phase C — Specification & Knowledge Base Synchronization
Goal: Amend normative specs and findings so backend selection, fail-fast behaviour, and parity expectations are codified.
Prereqs: Phase A inventory ready; confirm no conflicting ADRs.
Exit Criteria: Spec patch drafted, reviewed, and referenced from fix_plan; findings ledger updated if new policy IDs required.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Draft spec amendments | [x] | ✅ 2025-10-19 Attempt #15: Inserted §4.8 text into `specs/ptychodus_api_spec.md` after §4.7 (lines 224-235). Backend selection specification now normative. Source: `phase_e3_spec_patch.md`. Evidence: `phase_e3_spec_update.md`. |
| C2 | Align knowledge base | [x] | ✅ 2025-10-19 Attempt #15: Reviewed `docs/findings.md` — POLICY-002 NOT REQUIRED. Backend selection behavior fully specified in §4.8 (normative spec) and covered by existing POLICY-001 (PyTorch fail-fast) + CONFIG-001 (params.cfg sync). Decision documented in `phase_e3_spec_patch.md` and `phase_e3_spec_update.md`. |
| C3 | Review with governance log | [x] | ✅ 2025-10-19 Attempt #16: Cross-checked `phase_e_integration.md` and `phase_f_torch_mandatory.md`; no conflicts with §4.8 spec addition. Documented results in `reports/2025-10-19T202600Z/phase_e3_governance_review.md` and updated spec patch notes. |

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
