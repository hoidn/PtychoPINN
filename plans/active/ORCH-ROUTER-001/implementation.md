# Implementation Plan (Phased)

## Initiative
- ID: ORCH-ROUTER-001
- Title: Router prompt + orchestration dispatch layer
- Owner: user + Codex
- Spec Owner: scripts/orchestration/README.md
- Status: done

## Goals
- Add a router layer that deterministically selects prompts based on sync state + iteration, with optional router-prompt override.
- Preserve existing sync semantics while enforcing allowlist validation and clear failure modes.

## Phases Overview
- Phase A -- Design: routing contract, config schema, and state.json extension decision.
- Phase B -- Implementation: router loop + prompt, dispatch wiring, and logging.
- Phase C -- Verification: tests and documentation updates.
- Phase D -- Router-First/Only: enable router prompt control mode with safety guards.

## Exit Criteria
1. Deterministic routing maps `state.json` + iteration to a prompt path with actor gating.
2. Router prompt override wins when valid; invalid output crashes with a descriptive error.
3. Allowlist + file existence checks prevent unsafe dispatch.
4. Router decisions are logged and `state.json` persists **only** `last_prompt` (no additional metadata).
5. Test coverage verified (targeted selectors + collect-only as needed), logs archived under `.artifacts/orch-router-001/`.
6. Test registry updated when tests are added/renamed (docs/TESTING_GUIDE.md §2, docs/development/TEST_SUITE_INDEX.md).
7. Router-only mode (if enabled) is documented and tested with allowlist + expected_actor enforcement.

## Compliance Matrix (Mandatory)
- [x] **Spec Constraint:** scripts/orchestration/README.md -- sync state + dispatch contract
- [x] **Fix-Plan Link:** docs/fix_plan.md -- ORCH-ROUTER-001
- [x] **Finding/Policy ID:** PYTHON-ENV-001 (docs/DEVELOPER_GUIDE.md)
- [x] **Test Strategy:** plans/active/ORCH-ROUTER-001/test_strategy.md

## Spec Alignment
- **Normative Spec:** scripts/orchestration/README.md
- **Key Clauses:** State file fields; sync via git handshake; branch safety and prompt dispatch flow.

## Testing Integration

Principle: Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

Format for checklist items:
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

## Architecture / Interfaces
- **Key Data Types / Protocols:**
  - OrchestrationState (iteration, expected_actor, status, commits)
  - RouterDecision (selected_prompt, source, reason)
  - RoutingPolicy (review_every_n, prompt_map, allowlist)
- **Boundary Definitions:** Router loop -> prompt runner (supervisor/main/reviewer). YAML supplies parameters only; routing function lives in code.
- **Sequence Sketch (Happy Path):** read state.json -> choose deterministic prompt -> optional router override -> validate -> dispatch -> log
- **Data-Flow Notes:** state.json (authoritative) + config params -> routing function -> prompt name/path -> runner invocation; router override comes only from the router prompt (config does not override). state.json stores only last selected prompt (no additional metadata).

## Context Priming (read before edits)
- scripts/orchestration/README.md (sync contract)
- scripts/orchestration/config.py (config loading)
- scripts/orchestration/state.py (state schema)
- scripts/orchestration/router.py (new routing function, to be added)
- scripts/orchestration/supervisor.py, loop.py (current dispatch)
- prompts/supervisor.md, prompts/main.md (existing prompt entrypoints)
- docs/DEVELOPER_GUIDE.md (PYTHON-ENV-001)
- docs/TESTING_GUIDE.md (testing gates)
- docs/INITIATIVE_WORKFLOW_GUIDE.md (plan hygiene)
- Required findings/case law: PYTHON-ENV-001 (use `python` for subprocesses); no other relevant findings recorded yet.
- Related telemetry/attempts: docs/fix_plan.md Attempts History for ORCH-ROUTER-001; plans/active/ORCH-ROUTER-001/routing_contract.md.
- Data dependencies to verify: sync/state.json; orchestration.yaml (optional); prompts/ (router, supervisor, main, reviewer).

## Phase A -- Design
### Checklist
- [x] A0: Define routing contract (deterministic rules, actor gating, router override precedence), including review cadence (every N iterations).
      Test: N/A -- design artifact (`plans/active/ORCH-ROUTER-001/routing_contract.md`)
- [x] A1: Decide state.json extensions (optional fields only) and document the schema rules; persist only the last selected prompt.
      Test: N/A -- design artifact (`plans/active/ORCH-ROUTER-001/routing_contract.md`)
- [x] A2: Create test strategy doc and link it in docs/fix_plan.md.
      Test: N/A -- planning artifact (`plans/active/ORCH-ROUTER-001/test_strategy.md`)
- [x] A3: Update orchestration README with router contract + failure behavior and explicit function location.
      Test: N/A -- documentation-only
- [x] A4: Update docs/index.md to reference the router contract and orchestration updates once drafted.
      Test: N/A -- documentation-only

### Notes & Risks
- Ensure reviewer prompt routing stays compatible with expected_actor semantics.

## Phase B -- Implementation
### Checklist
- [x] B1: Add router loop entrypoint + wrapper; deterministic routing function in scripts/orchestration/router.py using state.json + iteration.
      Test: scripts/orchestration/tests/test_router.py::test_router_deterministic
- [x] B1.1: Create prompts/router.md template with a strict single-line prompt selection contract.
      Test: N/A -- documentation-only
- [x] B2: Add router prompt override, allowlist enforcement, and invalid-output crash path.
      Test: scripts/orchestration/tests/test_router.py::test_router_prompt_override
- [x] B3: Extend orchestration config for router settings (prompt path, allowlist, review cadence, mode).
      Test: scripts/orchestration/tests/test_router.py::test_router_config_loads
- [x] B4: Logging + optional state.json annotation of routing decisions.
      Test: scripts/orchestration/tests/test_router.py::test_router_logs_decision

### Notes & Risks
- Preserve backward-compatible defaults when orchestration.yaml is absent.

## Phase C -- Verification
### Checklist
- [x] C1: Run router test module and verify collection.
      Test: pytest scripts/orchestration/tests/test_router.py -v
- [x] C2: Update docs/TESTING_GUIDE.md + TEST_SUITE_INDEX if new tests are added.
      Test: pytest --collect-only scripts/orchestration/tests/test_router.py -v
- [x] C3: Update docs/index.md entry metadata if a new orchestration doc is added.
      Test: N/A -- documentation-only

### Notes & Risks
- Keep tests CPU-only; avoid invoking external LLM binaries.

## Phase D -- Router-First/Only
### Checklist
- [x] D1: Add routing mode config (`router_first` / `router_only`) and document precedence rules.
      Test: scripts/orchestration/tests/test_router.py::test_router_mode_selection
- [x] D2: Implement router-only dispatch with allowlist + expected_actor enforcement.
      Test: scripts/orchestration/tests/test_router.py::test_router_only_enforces_actor
- [x] D3: Update orchestration README and docs/index.md with router-only mode behavior.
      Test: N/A -- documentation-only

### Notes & Risks
- Router-only mode must still fail fast on invalid prompt output.

## Artifacts Index
- Reports root: .artifacts/orch-router-001/
- Latest run: <YYYY-MM-DDTHHMMSSZ>/
