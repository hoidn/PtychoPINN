# Phase F — Torch Mandatory Transition

## Context
- Initiative: INTEGRATE-PYTORCH-001 (PyTorch backend integration)
- Phase Goal: Retire torch-optional execution pathways and require PyTorch availability across config, adapters, workflows, and persistence so Ptychodus can rely on a single backend implementation path.
- Dependencies: CLAUDE.md directive (torch-optional parity) must be revised; verify CI/build environments include PyTorch extras; coordinate with TEST-PYTORCH-001 initiative for downstream coverage changes.
- Artifact Storage: Capture Phase F artifacts under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/` (F1 assets: `2025-10-17T184624Z/`; F2 use `2025-10-17T192500Z/`).

---

### Phase F1 — Directive Alignment & Governance Sign-off
Goal: Confirm stakeholder consensus to deprecate torch-optional behavior and update authoritative guidance (CLAUDE.md, docs/findings.md) before touching code.
Prereqs: Review existing directives, gather conflict evidence, engage initiative owners.
Exit Criteria: Documented approval plus updated guidance removing torch-optional requirement.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F1.1 | Catalogue conflicting directives | [x] | ✅ 2025-10-17 — Conflict inventory captured in `reports/2025-10-17T184624Z/directive_conflict.md` (covers CLAUDE.md §2, plan footprints, and skip whitelist history). |
| F1.2 | Secure governance decision | [x] | ✅ 2025-10-17 — Approval + risk assessment recorded in `reports/2025-10-17T184624Z/governance_decision.md`; stakeholders aligned on torch-required transition. |
| F1.3 | Update authoritative docs | [x] | ✅ 2025-10-17 — Redline + rollout plan documented in `reports/2025-10-17T184624Z/guidance_updates.md`; apply edits during Phase F3 doc sync. |

---

### Phase F2 — Impact Inventory & Migration Blueprint
Goal: Produce a comprehensive inventory of torch-optional code paths and tests, then map the step-by-step migration approach.
Prereqs: F1 exit (policy green-light).
Exit Criteria: Inventory and migration blueprint checked into reports and referenced from plan.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F2.1 | Enumerate guarded imports & flags | [x] | ✅ 2025-10-17 — Inventory captured in `reports/2025-10-17T192500Z/torch_optional_inventory.md` (47 instances across 15 files with file:line anchors + F3.2 checklist). |
| F2.2 | Audit test skip logic | [x] | ✅ 2025-10-17 — Skip behavior documented in `reports/2025-10-17T192500Z/test_skip_audit.md` (whitelist matrix, behavioral transitions, F3.3 validation checklist). |
| F2.3 | Draft migration sequence | [x] | ✅ 2025-10-17 — Migration plan finalized in `reports/2025-10-17T192500Z/migration_plan.md` (F3–F4 phased sequencing, gating checks, rollback strategies). |

---

### Phase F3 — Implementation & Test Realignment
Goal: Execute the migration by removing torch-optional fallbacks, enforcing imports, and updating tests to assume PyTorch is installed.
Prereqs: F2 blueprint complete; CI ready for PyTorch-on-by-default.
Exit Criteria: Codebase free of torch-optional guards, tests updated, CI green under mandatory PyTorch.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F3.1 | Update dependency management | [ ] | Promote PyTorch extras to core requirement in packaging (setup.cfg/pyproject), ensure development tooling installs PyTorch by default. Document environment changes in `reports/.../dependency_update.md`. |
| F3.2 | Remove guarded imports & flags | [ ] | Refactor modules to drop `TORCH_AVAILABLE`, unconditionalize imports, and simplify code paths. Ensure config adapters, data bridges, and workflows now import torch directly. Capture diffs + rationale in `reports/.../code_changes.md`. |
| F3.3 | Rewrite pytest skip logic | [ ] | Simplify `tests/conftest.py` (remove torch skip whitelist) and adjust tests depending on fallbacks. Record new selectors + expected runtime in `reports/.../pytest_update.md`. |
| F3.4 | Regression verification | [ ] | Run targeted parity + integration suites (`pytest tests/torch`, `pytest tests/test_integration_workflow.py -k torch`). Archive logs under `reports/.../pytest_green.log` with summary metrics. |

---

### Phase F4 — Documentation, Spec Sync, and Initiative Handoff
Goal: Communicate the policy shift and ensure related initiatives adapt to the torch-required baseline.
Prereqs: F3 complete with tests green.
Exit Criteria: Documentation, specs, and downstream plans updated; open risks logged.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| F4.1 | Update developer-facing docs | [ ] | Revise `docs/workflows/pytorch.md`, README, and training guides to state PyTorch is mandatory. Store summary in `reports/.../doc_updates.md`. |
| F4.2 | Sync specs & findings | [ ] | Amend `specs/ptychodus_api_spec.md` (installation/prereq sections) and add new finding capturing the policy change consequences. Capture diff in `reports/.../spec_sync.md`. |
| F4.3 | Coordinate initiative handoffs | [ ] | Notify TEST-PYTORCH-001 / CI maintainers; document follow-up tasks in `reports/.../handoff_notes.md`. |

---

### Exit Checklist
- [ ] Governance approval documented and CLAUDE.md updated to reflect torch-required policy.
- [ ] Comprehensive inventory and migration blueprint committed to reports.
- [ ] Codebase updated with PyTorch-required dependencies and simplified imports; regression suite green.
- [ ] Documentation/specs/handoff notes published and referenced from docs/fix_plan.md.
