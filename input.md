Summary: Close ADR-003 Phase E by syncing governance ledgers and packaging the deprecation evidence for archive.
Mode: Docs
Focus: ADR-003-BACKEND-API.EC — governance close-out (E.C2–E.C3)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/torch/test_api_deprecation.py -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/

Do Now — ADR-003-BACKEND-API.EC:
  - Implement: plans/active/ADR-003-BACKEND-API/implementation.md::Phase E — mark governance rows complete, link the 2025-10-24T070500Z deprecation artifacts, and note pending archival decisions.
  - Ledger: docs/fix_plan.md::[ADR-003-BACKEND-API] — append the Phase E.C1 completion summary and reference the new close-out hub.
  - Archive: plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/docs/summary.md — capture E.C2/E.C3 wrap-up, list relocated artifacts, and document remaining backlog (e.g., neighbor-count CLI follow-up).
  - Validate: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/torch/test_api_deprecation.py -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/logs/pytest_api_deprecation.log
  - Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:38-40 keeps E.C2–E.C3 open until ledger/doc archival finishes.
- docs/fix_plan.md:234 highlights outstanding governance wrap-up for ADR-003 after the deprecation warning landed.
- docs/TESTING_GUIDE.md:163-179 now documents the API deprecation tests and must stay aligned with post-closeout evidence.
- docs/development/TEST_SUITE_INDEX.md:84 depends on finalized artifact pointers once we relocate logs under the new 2025-11-04 hub.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md before running any pytest selectors.
- Edit plans/active/ADR-003-BACKEND-API/implementation.md and docs/fix_plan.md with consistent `[x]` states and artifact links to 2025-10-24T070500Z and 2025-11-04T093500Z hubs.
- Capture the governance close-out narrative in plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/docs/summary.md (reference E.C1 warning text, spec/workflow redlines, and remaining backlog).
- pytest tests/torch/test_api_deprecation.py -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/logs/pytest_api_deprecation.log

Pitfalls To Avoid:
- Do not relocate artifacts outside timestamped hubs; move stray logs into the new close-out directory instead.
- Keep `_warn_legacy_api_import` messaging unchanged; only documentation/ledger updates are allowed this loop.
- Do not mark Phase E fully complete unless the implementation plan and fix plan both cite the deprecation evidence path.
- Avoid editing PyTorch backend production modules beyond the existing deprecation shim.
- Ensure the pytest command runs with AUTHORITATIVE_CMDS_DOC exported; rerun under CPU-only assumptions (`CUDA_VISIBLE_DEVICES=""` inherited).

If Blocked:
- Store failing pytest output at plans/active/ADR-003-BACKEND-API/reports/2025-11-04T093500Z/phase_e_governance_closeout/logs/pytest_api_deprecation_error.log, note the blocker in docs/summary and docs/fix_plan.md, then pause for supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — Maintain PyTorch-required workflow guidance while documenting the deprecation (docs/findings.md:8).
- CONFIG-001 — Ensure ledger updates reaffirm that canonical configs bridge params.cfg; execution config stays separate (docs/findings.md:10).
- CONFIG-002 — Document execution-config isolation and accelerator auto-default within the governance close-out (docs/findings.md:11).

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:38
- docs/fix_plan.md:234
- docs/TESTING_GUIDE.md:163
- docs/development/TEST_SUITE_INDEX.md:84
- ptycho_torch/api/__init__.py:1

Next Up (optional):
- Draft follow-up backlog item for exposing `--neighbor-count` in CLI smokes once governance closure is logged.
