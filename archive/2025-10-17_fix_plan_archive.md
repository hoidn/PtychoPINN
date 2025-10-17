# Fix Plan Archive — 2025-10-17

Archived entries trimmed from docs/fix_plan.md for length hygiene. Each section carries its latest Attempts history and exit criteria for future reference.

## [INTEGRATE-PYTORCH-000] Pre-refresh Planning for PyTorch Backend Integration
*(Status: done)*

- Spec/AT: `plans/ptychodus_pytorch_integration_plan.md`, commit bfc22e7, `specs/ptychodus_api_spec.md`
- Attempts History (excerpt):
  * [2025-10-17] Attempt #0 — Authored phased rebaseline plan `plans/active/INTEGRATE-PYTORCH-000/implementation.md`.
  * [2025-10-17] Attempt #1 — Phase A module inventory & delta analysis (artifacts under `reports/2025-10-17T025000Z/`).
  * [2025-10-17] Attempt #2 — Phase B redline outline + summary (`reports/2025-10-17T025633Z/`).
  * [2025-10-17] Attempt #3 — Applied canonical plan updates; refreshed risk register.
  * [2025-10-17] Attempt #4 — Governance prep and stakeholder brief kickoff (see plan doc for detail).
- Exit Criteria: Alignment of integration plan with PyTorch tree, harmonized configuration schema, updated risk register. ✅
- Notes: All deliverables landed; keeping in archive for audit trail.

## [INTEGRATE-PYTORCH-001-PROBE-SIZE] Resolve PyTorch probe size mismatch in integration test
*(Status: pending — archived for space; reopen if investigation resumes)*

- Depends on: [INTEGRATE-PYTORCH-001-DATALOADER]
- Spec/AT: `specs/data_contracts.md` §1 (probe/object dimension), `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md`
- Attempts History:
  * [2025-10-17] Attempt #0 — Supervisor analysis captured in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/parity_summary.md` (PyTorch run now fails on probe mismatch). No implementation yet.
- Exit Criteria (deferred): train/inference CLI derive probe shape consistent with dataset; targeted tests proving parity with TensorFlow probe dimensions; parity summary updated with green run.
- Notes: Archived to reduce active ledger length; re-promote if work resumes.

## [LEGACY-TESTS-001] Restore throughput/baseline pytest modules
*(Status: pending — deprioritized)*

- Original Goal: Reenable skipped legacy throughput/baseline tests.
- Attempts History: None beyond initial identification.
- Exit Criteria: Legacy modules adapted to new dependency posture, pytest suite collecting without manual skips.
- Notes: Work postponed indefinitely during PyTorch integration focus.

