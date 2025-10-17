# Phase F4 — Documentation, Spec Sync, and Initiative Handoff

## Context
- Initiative: INTEGRATE-PYTORCH-001 (PyTorch backend integration)
- Phase Goal: Publish authoritative guidance for the torch-required policy, align normative specs, and coordinate downstream initiatives so the new baseline is unambiguous.
- Dependencies: Phase F1–F3 exit criteria complete (governance approval, migration implementation, regression green). Review `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md` and `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T184624Z/{governance_decision.md,guidance_updates.md}` before modifying docs or specs.
- Artifact Storage: Capture Phase F4 evidence under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/` (see checklist guidance for filenames). Cross-reference each artifact from docs/fix_plan.md.

---

### Phase F4.1 — Developer-Facing Documentation Updates
Goal: Ensure every developer-facing document reflects that PyTorch is now mandatory and torch-optional code paths are retired.
Prereqs: Confirm governance decision (Phase F1) and migration completion (Phase F3) are recorded.
Exit Criteria: CLAUDE.md, docs/workflows/pytorch.md, README (PyTorch usage section), and any other doc mentioning torch-optional behavior are updated; summary of edits stored under `reports/<timestamp>/doc_updates.md` with diffs/anchors.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F4.1.A | Inventory torch-optional language across docs | [x] | ✅ 2025-10-17 — Searched docs/ and README.md; zero active torch-optional references (all in fix_plan.md history). Identified CLAUDE.md:57, pytorch.md:17-22, README.md:23-28 requiring updates. Inventory documented in `reports/2025-10-17T203640Z/doc_updates.md`. |
| F4.1.B | Update CLAUDE.md directives | [x] | ✅ 2025-10-17 — Replaced §2 directive (lines 57-59) from "Keep PyTorch parity tests torch-optional" to "Enforce PyTorch Requirement". New text references Phase F governance, setup.py specification, and conftest.py skip behavior. |
| F4.1.C | Refresh workflow guides & README | [x] | ✅ 2025-10-17 — Updated docs/workflows/pytorch.md prerequisites (bold "REQUIRED", setup.py reference, PyTorch install link) and README.md installation (auto-install note, GPU/CUDA guidance). All changes logged in `doc_updates.md`. |

---

### Phase F4.2 — Spec & Findings Synchronization
Goal: Align normative specifications and knowledge base with the torch-required baseline.
Prereqs: F4.1 doc edits drafted.
Exit Criteria: `specs/ptychodus_api_spec.md` explicitly lists PyTorch as required dependency; new finding logged describing the policy change and rationale; evidence recorded under `reports/<timestamp>/spec_sync.md`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F4.2.A | Update spec prerequisites | [x] | ✅ 2025-10-17 — Updated `specs/ptychodus_api_spec.md` Section 1 (line 14, PyTorch requirement paragraph), Section 2.3 (lines 75-81, config bridge adapters), Section 4.2 (line 162, fail-fast imports). Summarized in `reports/2025-10-17T205413Z/spec_sync.md`. |
| F4.2.B | Add knowledge-base entry | [x] | ✅ 2025-10-17 — Added POLICY-001 row to `docs/findings.md` (line 8) capturing torch-required transition with governance link. Documented in `spec_sync.md`. |
| F4.2.C | Verify cross-references | [x] | ✅ 2025-10-17 — Updated CLAUDE.md directive (line 58) to reference POLICY-001 via doc-ref tag. Verification checklist recorded in `spec_sync.md`. |

**Prep Note:** See `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T204818Z/spec_sync_brief.md` for supervisor guidance before editing.

---

### Phase F4.3 — Initiative Handoff & Follow-Up Actions
Goal: Notify dependent initiatives and CI maintainers, document required follow-up work, and mark residual TODOs.
Prereqs: F4.1–F4.2 drafts ready for review.
Exit Criteria: Handoff notes committed under `reports/<timestamp>/handoff_notes.md` covering TEST-PYTORCH-001, CI configuration, and outstanding prerequisites (e.g., cloning Ptychodus repo). docs/fix_plan.md updated with clear next steps.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F4.3.A | Map impacted initiatives & CI tasks | [ ] | Enumerate follow-up actions for TEST-PYTORCH-001, CI/CD, and repository prerequisites (submodule addition). Include owners or TBD notes in `handoff_notes.md`. |
| F4.3.B | Define verification cadence | [ ] | Specify how/when to re-run targeted suites post-doc/spec changes (likely `pytest --collect-only tests/torch/` to validate skip messaging). Log command guidance in `handoff_notes.md`. |
| F4.3.C | Update plan & ledger cross-references | [ ] | After edits land, update `phase_f_torch_mandatory.md` F4 row and docs/fix_plan.md attempt history with artifact paths + remaining risks. Summarize completion criteria in `handoff_notes.md`. |

---

### Exit Checklist
- [ ] `reports/<timestamp>/doc_updates.md` summarizes documentation changes with anchors.
- [ ] `reports/<timestamp>/spec_sync.md` records spec edits and finding entry.
- [ ] `reports/<timestamp>/handoff_notes.md` lists follow-up owners, commands, and residual risks.
- [ ] `phase_f_torch_mandatory.md` F4 row reflects latest state and links to this plan.

---

*Last updated: 2025-10-17*
