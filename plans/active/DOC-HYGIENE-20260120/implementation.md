# Implementation Plan (Phased)

## Initiative
- ID: DOC-HYGIENE-20260120
- Title: Reviewer orchestration config + prompt hygiene
- Owner: Codex
- Spec Owner: scripts/orchestration/README.md
- Status: **in_progress**

## Goals
- Restore a real, versioned `orchestration.yaml` so router cadence, state paths, doc hygiene knobs, and spec bootstrap directories come from a single source of truth.
- Update reviewer/architecture prompt templates to cite documentation that actually exists (`docs/index.md`, `docs/architecture*.md`, `specs/spec-ptycho-*.md`) instead of dead paths.
- Make supervisor sync loops honor `--no-git` (skip branch guards, pulls/pushes, auto-commits) and prove the guard via targeted pytest coverage so offline/spec bootstrap runs stop failing.

## Phases Overview
- Phase A — Reality Audit & Doc Alignment: confirm invalid references, update prompts/docs to cite current anchors, and capture the mandate/test strategy.
- Phase B — Orchestration Config Source of Truth: land root-level `orchestration.yaml`, document spec_bootstrap paths, and update docs/index/test registries.
- Phase C — Supervisor `--no-git` Enforcement: code + tests to gate git operations, plus README/test index updates.

## Exit Criteria
1. `orchestration.yaml` exists at repo root recording prompts_dir, router.review_every_n, state/log paths, doc/test whitelists, agent defaults, and spec_bootstrap dirs that match current usage.
2. `prompts/arch_writer.md`, `prompts/spec_reviewer.md`, and related documentation reference live files/sections under `docs/` and `specs/`, with instructions pointing to `docs/index.md` as the canonical map.
3. Supervisor sync mode respects `--no-git` end-to-end (no branch guard, pull/push, auto-commit, or git scrub operations), and pytest regressions cover the guard + doc/test registry updates.
4. **Test coverage verified:**
   - New/updated selectors collect (`pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q`, etc.).
   - Targeted suites (`pytest scripts/orchestration/tests/test_supervisor.py::TestNoGit`, `pytest scripts/orchestration/tests/test_orchestrator.py -v`) pass.
   - Docs/test registries updated per `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md`.
   - Logs and docs saved under `plans/active/DOC-HYGIENE-20260120/reports/<timestamp>/`.

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** scripts/orchestration/README.md — router cadence + sync via git semantics
- [ ] **Fix-Plan Link:** docs/fix_plan.md — DOC-HYGIENE-20260120
- [ ] **Finding/Policy ID:** TEST-CLI-001 (orchestration test gating)
- [ ] **Finding/Policy ID:** PYTHON-ENV-001 (invoke `python` via PATH when wiring tests/scripts)
- [ ] **Test Strategy:** plans/active/DOC-HYGIENE-20260120/test_strategy.md

## Spec Alignment
- **Normative Spec:** scripts/orchestration/README.md (router cadence, sync state, orchestration config contract)
- **Key Clauses:** Router review cadence, sync `state.json` fields, doc/meta auto-commit policy, spec bootstrap configuration requirements documented in README + docs/index.md.

## Testing Integration

Principle: Every checklist item changing behavior references its pytest selector; doc-only steps call out N/A with justification.

## Architecture / Interfaces
- **Key Data Types / Protocols:** `OrchConfig` (YAML loader), `OrchestrationState` fields (`last_prompt`, `last_prompt_actor`), supervisor sync loops (branch guard + git bus), prompt templates referencing doc anchors.
- **Boundary Definitions:** Prompt templates ↔ docs/index/specs; supervisor CLI ↔ git bus; YAML config ↔ config loader + router/orchestrator tests.
- **Sequence Sketch (Happy Path):** `orchestration.yaml` → `load_config()` → supervisor/orchestrator CLI parse → router cadence selection/reporting → doc/test registries.
- **Data-Flow Notes:** YAML (checked in) provides config; prompts reference docs/specs; supervisor runs call git bus unless `--no-git`; tests stub git ops to prove gating.

## Context Priming (read before edits)
- docs/index.md (documentation map + references that prompts should cite)
- docs/findings.md — TEST-CLI-001, PYTHON-ENV-001
- scripts/orchestration/README.md (config contract, router cadence)
- scripts/orchestration/{supervisor.py,loop.py,orchestrator.py,config.py}
- prompts/{arch_writer.md,spec_reviewer.md,router.md}
- docs/TESTING_GUIDE.md §2 + docs/development/TEST_SUITE_INDEX.md (test registry updates)
- Existing artifacts under `plans/active/ORCH-ROUTER-001/` and `plans/active/ORCH-ORCHESTRATOR-001/` (sync cadence context)
- User_input.md overrides archived via docs/fix_plan.md and galph_memory (revoked but referenced)
- Data dependencies: sync/state.json, orchestration.yaml (to be created), prompts directory, tests under scripts/orchestration/tests/

## Phase A — Reality Audit & Doc Alignment
### Checklist
- [ ] A0: Capture refreshed test strategy + context (this plan + `test_strategy.md`) and record the reviewer override in docs/fix_plan.md.
      Test: N/A — documentation/planning artifact
- [ ] A1: Update `prompts/arch_writer.md` to reference real docs/specs (e.g., `docs/architecture.md`, `specs/spec-ptycho-workflow.md`) and remove dead `docs/architecture/data-pipeline.md` references.
      Test: N/A — documentation-only change
- [ ] A2: Update `prompts/spec_reviewer.md` required-reading + spec_bootstrap sections to reference `specs/` and `docs/index.md`, removing `docs/spec-shards/*.md`.
      Test: N/A — documentation-only change
- [ ] A3: Note the doc updates + new prompt targets in docs/index.md (if needed) so future prompts stay aligned.
      Test: N/A — documentation-only change

### Notes & Risks
- Prompt edits must keep XML-like schema intact; only URLs/paths should change.

## Phase B — Orchestration Config Source of Truth
### Checklist
- [ ] B1: Author root-level `orchestration.yaml` matching current defaults (prompts, router review cadence=3, state/log dirs, doc whitelist, agent mapping, spec_bootstrap dirs/files) and check it in.
      Test: pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v
- [ ] B2: Document the new config in scripts/orchestration/README.md + docs/index.md (link to root file + key fields) so reviewers know where to look.
      Test: N/A — documentation-only change
- [ ] B3: Ensure spec bootstrap instructions (`sync/spec_bootstrap_state.json`, templates dir, impl dirs) in orchestration.yaml align with prompts/spec reviewer instructions; update docs/test registries if references change.
      Test: pytest scripts/orchestration/tests/test_agent_dispatch.py -v (sanity run after config addition)

### Notes & Risks
- Config must avoid secrets; keep repo-safe defaults only.
- Ensure YAML loads even when PyYAML absent (file still present but loader defaults should match).

## Phase C — Supervisor `--no-git` Enforcement
### Checklist
- [ ] C1: Update `scripts/orchestration/supervisor.py` to treat `--no-git` as a hard guard (skip branch assertions, pulls/pushes, auto-commit helpers, git scrubs) for both sync and legacy modes; share helpers with loop/orchestrator implementation when possible.
      Test: pytest scripts/orchestration/tests/test_supervisor.py::TestNoGit::test_sync_loop_skips_git_ops -v
- [ ] C2: Add pytest coverage (new `scripts/orchestration/tests/test_supervisor.py`) stubbing git bus calls to assert `--no-git` suppresses them while still running prompts/logging. Update docs/TESTING_GUIDE.md + TEST_SUITE_INDEX entries with the new selector.
      Test: 
        - pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q
        - pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_no_git -v (regression)
- [ ] C3: Document the new behavior in scripts/orchestration/README.md and confirm `prompts/router.md`/galph instructions mention orchestration.yaml + no-git guard. Capture evidence/logs under `plans/active/DOC-HYGIENE-20260120/reports/<timestamp>/`.
      Test: N/A — documentation-only change

### Notes & Risks
- Tests must avoid calling real git; use monkeypatch/mocks.
- Combined mode already respects `no_git`; ensure supervisor behavior matches orchestrator expectations to avoid drift.

## Artifacts Index
- Reports root: `plans/active/DOC-HYGIENE-20260120/reports/`
- Latest run: `2026-01-20T235033Z/`
