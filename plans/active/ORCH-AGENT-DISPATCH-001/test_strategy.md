# Test Strategy: Agent Dispatch

**Initiative:** ORCH-AGENT-DISPATCH-001
**Phase:** Phase A / Planning
**Date:** 2026-01-20
**Status:** Draft

---

## 1. Framework Selection & Compatibility
- **Primary Framework:** pytest
- **Scope:** orchestration submodule tests under `scripts/orchestration/tests/`

## 2. Test Coverage Targets
- **Unit Tests:** agent resolution precedence (prompt vs role vs default) and CLI/YAML parsing.
- **Integration (light):** combined mode uses router-selected prompt to resolve agent.
- **No external process calls:** use stubbed executors to avoid invoking real CLIs.

## 3. Target Selectors
- `pytest --collect-only scripts/orchestration/tests/test_agent_dispatch.py -v`
- `pytest scripts/orchestration/tests/test_agent_dispatch.py -v`

## 4. Mock/Stub Strategy
- Stub prompt execution and CLI builders; assert selected agent strings/commands only.

## 5. Success Criteria
- All agent dispatch tests pass on CPU-only environments.
- No external CLI calls in tests.
- Logs captured under `plans/active/ORCH-AGENT-DISPATCH-001/reports/<timestamp>/`.
