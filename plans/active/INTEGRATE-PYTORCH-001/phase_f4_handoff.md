# Phase F4.3 — Initiative Handoff Coordination

## Context
- Initiative: INTEGRATE-PYTORCH-001 (PyTorch backend integration)
- Phase Goal: Notify downstream owners about the torch-required baseline, capture required follow-up actions, and define ongoing verification cadence before closing Phase F.
- Dependencies: Phase F4.1 documentation updates and Phase F4.2 spec/finding synchronization must be complete. Reference artifacts: `phase_f4_doc_sync.md` (F4 tasks), `phase_f_torch_mandatory.md` F4 row, governance decision (`reports/2025-10-17T184624Z/governance_decision.md`).
- Artifact Storage: Save handoff evidence under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/` alongside existing F4 reports. Primary deliverable is `handoff_notes.md`; supplement with matrices or command logs as needed (e.g., `owner_matrix.md`, `verification_commands.md`).

---

### Handoff Phase — Coordinate Downstream Execution
Goal: Produce a reusable handoff packet that downstream initiatives (tests, CI, release management) can execute without re-reading Phase F history.
Prereqs: Confirm artifacts `doc_updates.md` (F4.1) and `spec_sync.md` (F4.2) exist and are cited in docs/fix_plan.md Attempt #75.
Exit Criteria: `handoff_notes.md` summarizes initiative owners, CI updates, versioning actions, and verification cadence with concrete commands and artifact destinations; `phase_f4_doc_sync.md` F4.3 table updated; docs/fix_plan.md Attempts history references the new report.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| H1 | Map impacted initiatives and owners | [ ] | Capture a table in `handoff_notes.md` listing TEST-PYTORCH-001, CI/CD maintainers, Ptychodus integration team, and release/versioning owners. For each, record required action, blocking dependencies, and reference documents (e.g., `plans/pytorch_integration_test_plan.md`, `governance_decision.md §7`, `phase_f_torch_mandatory.md` F4 row). Identify responsible contact if available; otherwise mark TBD. |
| H2 | Document CI and environment updates | [ ] | Summarize required CI changes (torch install step, runner matrix) per governance decision §4.2 and Phase F3.1 artifacts. Specify the script/config files to touch (e.g., CI workflow YAMLs once located) and the validation step (command to verify torch import before pytest). Include any temporary mitigations if CI lacks GPU support. |
| H3 | Define ongoing verification cadence | [ ] | List authoritative verification commands in `handoff_notes.md`, drawing from Phase F3.4 logs and `docs/TESTING_GUIDE.md`. Include at minimum `pytest --collect-only tests/torch/ -q`, `pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv`, and `pytest tests/torch/test_config_bridge.py -k parity -vv`. Note expected outcomes (pass/fail semantics) and where to archive future logs. |
| H4 | Record ledger and plan synchronization steps | [ ] | Outline the exact updates needed once handoff is complete: mark `phase_f_torch_mandatory.md` F4 row `[x]`, note docs/fix_plan.md Attempt with artifact path, and trigger version bump to v2.0.0 (per governance §5.4) including CHANGELOG entry. Provide reminder to create release notes referencing POLICY-001. |

---

## Verification Checklist
- [ ] `handoff_notes.md` created under new timestamped directory with initiative+CI+verification sections.
- [ ] Command checklist references precise pytest selectors and expected results.
- [ ] Plan cross-references (`phase_f4_doc_sync.md`, `phase_f_torch_mandatory.md`) updated to link the new report.
- [ ] docs/fix_plan.md Attempts history logs the handoff planning/execution with artifact path.
- [ ] Versioning guidance (v2.0.0 bump + CHANGELOG) captured for release owner.

---

*Last updated: 2025-10-17*
