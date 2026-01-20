# Implementation Plan (Phased)

## Initiative
- ID: ORCH-AGENT-DISPATCH-001
- Title: Per-role / per-prompt agent dispatch (codex vs claude)
- Owner: user + Codex
- Spec Owner: scripts/orchestration/README.md
- Status: pending

## Goals
- Allow selecting different model CLIs per role or per prompt (for example, codex for supervisor.md, claude for main.md).
- Preserve existing behavior when no agent mapping is configured (global --agent still works).
- Make combined mode resolve agent by selected prompt (router-aware) with clear logging.

## Phases Overview
- Phase A — Design & Tests: define resolution precedence and seed failing tests.
- Phase B — Implementation: add config/CLI plumbing + resolver and wire into supervisor/loop/orchestrator.
- Phase C — Verification & Docs: finalize tests, update orchestration docs and test registry.

## Exit Criteria
1. Agent resolution precedence implemented and documented (CLI prompt map > CLI role map > YAML prompt map > YAML role map > default).
2. Combined mode selects agent per prompt after router selection; supervisor/loop resolve agent for their turn only.
3. Logging includes selected agent and resolved command per turn.
4. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - No regression in existing test suite (full suite green or known-skip documented)
   - Test registry synchronized: `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` updated
   - Logs saved to `plans/active/ORCH-AGENT-DISPATCH-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `scripts/orchestration/README.md` (routing + combined orchestration behavior)
- [ ] **Fix-Plan Link:** `docs/fix_plan.md` — ORCH-AGENT-DISPATCH-001
- [ ] **Finding/Policy ID:** PYTHON-ENV-001 (use PATH `python` for subprocesses/docs)
- [ ] **Test Strategy:** `plans/active/ORCH-AGENT-DISPATCH-001/test_strategy.md`

## Spec Alignment
- **Normative Spec:** `scripts/orchestration/README.md`
- **Key Clauses:** combined orchestration, router selection flow, CLI flags/behavior, logging conventions.

## Testing Integration

**Principle:** Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

**Format for checklist items:**
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

**Complex testing needs:** See `plans/active/ORCH-AGENT-DISPATCH-001/test_strategy.md`.

## Architecture / Interfaces (optional)
- **Key Data Types / Protocols:** `AgentResolver`, `AgentSelection` (agent name + cmd list), `AgentConfig` (default, role map, prompt map).
- **Boundary Definitions:** `[orchestrator/supervisor/loop] -> [runner] -> [agent resolver] -> [prompt executor]`.
- **Sequence Sketch (Happy Path):** select prompt -> resolve agent -> build cmd -> execute prompt -> log agent choice.
- **Data-Flow Notes:** YAML/CLI values -> resolver -> subprocess command string; no external state changes.

## Context Priming (read before edits)
- Primary docs/specs to re-read: `scripts/orchestration/README.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`, `docs/DEVELOPER_GUIDE.md` (PYTHON-ENV-001)
- Required findings/case law: PYTHON-ENV-001
- Related telemetry/attempts: ORCH-ORCHESTRATOR-001 and ORCH-ROUTER-001 plans
- Data dependencies to verify: `orchestration.yaml`, prompt files under `prompts/`

## Phase A — Design & Tests
### Checklist
- [ ] A0: Define agent resolution precedence and config shape (YAML + CLI) and add a failing resolver test.
      Test: `scripts/orchestration/tests/test_agent_dispatch.py::test_agent_prompt_override_precedence`
- [ ] A1: Add test strategy doc and link in plan/fix_plan.
      Test: N/A — planning artifact
- [ ] A2: Seed tests for router-selected prompt using a different agent in combined mode.
      Test: `scripts/orchestration/tests/test_agent_dispatch.py::test_router_prompt_drives_agent`

### Dependency Analysis (Required for Refactors)
- **Touched Modules:** `scripts/orchestration/config.py`, `scripts/orchestration/runner.py`, `scripts/orchestration/orchestrator.py`, `scripts/orchestration/supervisor.py`, `scripts/orchestration/loop.py`
- **Circular Import Risks:** avoid importing supervisor/loop inside resolver; keep resolver in runner or new module.
- **State Migration:** none (config-only additions).

### Notes & Risks
- Avoid breaking existing `--agent` flows; resolver must default to current behavior when no mappings are configured.

## Phase B — Implementation
### Checklist
- [ ] B1: Extend config loader for `agent` block (default/roles/prompts) and CLI overrides.
      Test: `scripts/orchestration/tests/test_agent_dispatch.py::test_agent_config_parsing`
- [ ] B2: Implement `AgentResolver` and cmd builder; log selected agent per turn.
      Test: `scripts/orchestration/tests/test_agent_dispatch.py::test_agent_role_fallback`
- [ ] B3: Wire resolver into combined mode (per prompt) and supervisor/loop (per role).
      Test: `scripts/orchestration/tests/test_agent_dispatch.py::test_combined_uses_prompt_agent`

### Notes & Risks
- Ensure agent resolution occurs after router selection in combined mode so router overrides can change agent.

## Phase C — Verification & Docs
### Checklist
- [ ] C1: Run collect-only and module tests for agent dispatch.
      Test: `pytest --collect-only scripts/orchestration/tests/test_agent_dispatch.py -v`
- [ ] C2: Run agent dispatch test module.
      Test: `pytest scripts/orchestration/tests/test_agent_dispatch.py -v`
- [ ] C3: Update docs (`scripts/orchestration/README.md`, `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`).
      Test: N/A — documentation-only

### Notes & Risks
- Keep tests stubbed (no external CLI calls).

## Artifacts Index
- Reports root: `plans/active/ORCH-AGENT-DISPATCH-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
