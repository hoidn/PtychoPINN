# Agent-Orchestration Artifact Contract Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Initiative
- ID: `ORCH-ARTIFACT-CONTRACT-V0`
- Title: `Artifact-Contract Orchestration Prototype`
- Owner: `ollie + coding agent`
- Spec Owner: `~/Documents/agent-orchestration/specs/index.md`
- Status: `in_progress`

## Goals
- Prototype deterministic file-based handoff between multi-step agent calls.
- Support simple review/test-fix loops with explicit artifact contracts and deterministic gates.
- Keep v0 changes backward compatible with existing `agent-orchestration` workflows.

## Phases Overview
- Phase A — Contract Core: DSL + validator + executor integration (`Task 1-3`).
- Phase B — Prompted Contracts: provider prompt contract injection (`Task 4`).
- Phase C — Pattern Workflows + Verification: example workflows, runtime smoke tests, runbook (`Task 5-6`).

## Exit Criteria
1. Deterministic handoff uses explicit artifact keys (`name`) and file-based contracts.
2. Contract validation preserves original execution failures and only evaluates successful step executions.
3. Provider prompt contract suffix is deterministic and opt-out capable via `inject_output_contract: false`.
4. Example workflows for backlog plan/execute, test-fix loop, and unit-of-work + test-fix run with deterministic termination.
5. Targeted selectors for Tasks 1-6 pass in `~/Documents/agent-orchestration`.

## Compliance Matrix (Mandatory)
- [x] **Spec Constraint:** `~/Documents/agent-orchestration/specs/dsl.md` (step schema + strict validation behavior).
- [x] **Spec Constraint:** `~/Documents/agent-orchestration/specs/providers.md` (prompt composition and provider semantics).
- [x] **Spec Constraint:** `~/Documents/agent-orchestration/specs/state.md` (state persistence shape for step outputs).
- [ ] **Fix-Plan Link:** `N/A` (external prototype repo; no `docs/fix_plan.md` row in this workspace).
- [x] **Finding/Policy ID:** `POLICY-001` is not impacted; no changes to PyTorch/model code in `PtychoPINN`.

## Spec Alignment
- **Normative Spec:** `~/Documents/agent-orchestration/specs/index.md`
- **Key Clauses:**
  - `specs/dsl.md`: strict loader validation and step schema safety.
  - `specs/providers.md`: provider prompt composition and input-mode semantics.
  - `specs/state.md`: deterministic step result persistence.

## Context Priming (read before edits)
- `~/Documents/agent-orchestration/specs/index.md`
- `~/Documents/agent-orchestration/specs/dsl.md`
- `~/Documents/agent-orchestration/specs/providers.md`
- `~/Documents/agent-orchestration/specs/state.md`
- `~/Documents/agent-orchestration/specs/examples/patterns.md`
- `~/Documents/agent-orchestration/tests/README.md`

**Goal:** Prototype deterministic, file-based handoff contracts for multi-step Codex/Claude workflows in `~/Documents/agent-orchestration`, including simple review/test-fix loops.

**Architecture:** Build on `agent-orchestration` (do not modify `scripts/orchestration` yet). Add step-level `expected_outputs` contracts, runtime output validation, and optional prompt contract injection so each agent step has deterministic output artifacts (files) and downstream steps consume validated values. Use existing control-flow (`on.*.goto`, retries, conditions) for loop-until semantics.

**Tech Stack:** Python 3.11, YAML workflow DSL (`orchestrator/loader.py`), workflow executor (`orchestrator/workflow/executor.py`), state persistence (`orchestrator/state.py`), pytest.

---

## Feedback Incorporation (2026-02-20)

- Deterministic artifact keys are now explicit: `expected_outputs[].name` is required and downstream references use `steps.<step>.artifacts.<name>`.
- `inject_output_contract` is explicitly part of the step schema as optional boolean (default `true`), with validation coverage planned in Task 1.
- Contract failures do not mask execution failures: contract validation runs only when step `exit_code == 0`.
- `required` semantics are explicit: default `true`; when `false`, missing artifact files are allowed and omitted from `artifacts`.
- Path-security checks require canonical path resolution with symlink-safe containment checks for both workspace and optional `under` roots.

---

## Scope and Design Decisions (Clarified)

- Use **artifact-first handoff** (state files), not stdout JSON as the primary channel.
- Add step field `expected_outputs` to encode deterministic contracts in workflow YAML.
- Runtime contract enforcement:
  - inject a deterministic “Output Contract” block into provider prompt text (default on, opt-out flag)
  - validate output artifacts immediately after each command/provider step
  - fail fast with structured `contract_violation` errors
- Persist validated output values under step result (e.g., `steps.<name>.artifacts.*`) so downstream substitution is deterministic.
- Use explicit artifact keys (`name`) in every `expected_outputs` entry; never infer keys from filenames.
- Implement loop-until using existing control flow first; no new general control-flow primitives in v0.
- Keep v0 simple: support core output types only (`enum`, `integer`, `float`, `bool`, `relpath`).

## Constraints

- Primary implementation repo for this initiative: `~/Documents/agent-orchestration`.
- No edits to `scripts/orchestration` in this initiative.
- Keep DSL additions minimal and backward compatible.
- Test-first for each behavior change.

## Contract Conventions (v0)

- `expected_outputs[]` required fields: `name`, `path`, `type`.
- `expected_outputs[]` optional fields: `allowed`, `under`, `must_exist_target`, `required` (default `true`).
- `must_exist_target` behavior: for `type: relpath`, resolve the emitted path canonically and require the resolved target path to exist when `must_exist_target: true`.
- Step field `inject_output_contract` is optional boolean, consumed only by provider steps; command steps accept it as a no-op for compatibility.
- Substitution contract: downstream consumers reference only `steps.<step_name>.artifacts.<name>`.
- Validation sequencing:
  - if step execution fails (`exit_code != 0`), preserve the original failure and skip contract validation
  - if step execution succeeds and contract validation fails, emit `contract_violation`
- Path security is two-stage:
  - loader-time: syntactic path safety checks (`absolute`, `..`) for fast feedback
  - runtime: canonical/real path containment checks (symlink-safe) for workspace and optional `under` roots

## Task 1: Add `expected_outputs` DSL Validation

**Files:**
- Modify: `~/Documents/agent-orchestration/orchestrator/loader.py`
- Modify: `~/Documents/agent-orchestration/specs/dsl.md`
- Test: `~/Documents/agent-orchestration/tests/test_loader_validation.py`

**Step 1: Write failing tests**

Add tests for:
- valid `expected_outputs` list on provider/command step
- invalid missing required keys (`name`, `path`, `type`)
- invalid `type` value
- invalid `under` syntactic path safety (`absolute`, `..`)
- invalid `inject_output_contract` type

```python
def test_step_expected_outputs_validates_required_fields():
    workflow = {
        "version": "1.1.1",
        "steps": [{
            "name": "DraftPlan",
            "provider": "codex",
            "expected_outputs": [{"name": "plan_path", "path": "state/plan_path.txt", "type": "relpath"}],
            "inject_output_contract": True,
        }],
    }
    # expect no validation error for this shape
```

**Step 2: Run test to verify failure**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_loader_validation.py -k expected_outputs -v`

Expected: FAIL because loader does not yet support `expected_outputs`.

**Step 3: Implement minimal validation**

- Extend known step fields in loader.
- Add `_validate_expected_outputs(...)`.
- Add `_validate_provider_contract_options(...)`.
- Support fields:
  - required: `name`, `path`, `type`
  - optional: `allowed`, `under`, `must_exist_target`, `required`
- `inject_output_contract`: optional boolean, default true, provider-aware but command-safe no-op.
- Keep canonical/symlink checks out of loader and in runtime validator.

**Step 4: Run test to verify pass**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_loader_validation.py -k expected_outputs -v`

Expected: PASS.

**Step 5: Commit**

```bash
cd ~/Documents/agent-orchestration
git add orchestrator/loader.py specs/dsl.md tests/test_loader_validation.py
git commit -m "feat(dsl): add expected_outputs schema validation"
```

## Task 2: Implement Output Contract Validator (Artifact Parsing + Checks)

**Files:**
- Create: `~/Documents/agent-orchestration/orchestrator/contracts/__init__.py`
- Create: `~/Documents/agent-orchestration/orchestrator/contracts/output_contract.py`
- Test: `~/Documents/agent-orchestration/tests/test_output_contract.py`

**Step 1: Write failing tests**

Cover:
- successful parsing/validation for `enum`, `integer`, `float`, `bool`, `relpath`
- missing file -> violation
- missing file with `required: false` -> no violation, artifact omitted
- invalid enum value -> violation
- `relpath` escaping workspace or outside `under` root -> violation
- symlink escape outside workspace/`under` -> violation

```python
def test_validate_enum_contract_passes(tmp_path):
    p = tmp_path / "decision.txt"
    p.write_text("APPROVE\n")
    spec = {"name": "review_decision", "path": str(p), "type": "enum", "allowed": ["APPROVE", "REVISE"]}
    artifacts = validate_expected_outputs([spec], workspace=tmp_path)
    assert artifacts["review_decision"] == "APPROVE"
```

**Step 2: Run test to verify failure**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_output_contract.py -v`

Expected: FAIL (module missing).

**Step 3: Implement validator**

- Add a typed validation result and error helpers.
- Parse from files only.
- Return a normalized artifacts map for state/substitution.
- Enforce canonical path containment checks (workspace and optional `under`) after resolving symlinks.
- Define normalization rules in code/docs (`strip` newline, strict bool tokens, numeric parsing).
- If `required: false` and file is missing, do not raise; omit the artifact key.

**Step 4: Run test to verify pass**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_output_contract.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
cd ~/Documents/agent-orchestration
git add orchestrator/contracts/__init__.py orchestrator/contracts/output_contract.py tests/test_output_contract.py
git commit -m "feat(contracts): add artifact output contract validator"
```

## Task 3: Integrate Contract Validation into Step Execution

**Files:**
- Modify: `~/Documents/agent-orchestration/orchestrator/workflow/executor.py`
- Modify: `~/Documents/agent-orchestration/orchestrator/state.py`
- Test: `~/Documents/agent-orchestration/tests/test_workflow_output_contract_integration.py`

**Step 1: Write failing integration tests**

Cases:
- command step succeeds but contract file missing -> step exits with contract violation
- command/provider step with valid contract -> `steps.<name>.artifacts` exists
- command/provider execution failure without contract issue -> preserves original error (no override)

```python
def test_command_step_fails_when_expected_output_missing(tmp_path):
    # run workflow with expected_outputs pointing to absent file
    result = run_workflow(...)
    assert result["steps"]["DraftPlan"]["error"]["type"] == "contract_violation"
    assert result["steps"]["DraftPlan"]["exit_code"] != 0
```

**Step 2: Run test to verify failure**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_output_contract_integration.py -v`

Expected: FAIL.

**Step 3: Implement minimal integration**

- After each command/provider execution result, call output contract validator only when `expected_outputs` exists and `exit_code == 0`.
- On violation: emit deterministic `contract_violation` failure for that otherwise-successful step.
- If execution already failed, preserve original exit/error and skip contract validation.
- On success: persist parsed values in step result as `artifacts`.
- Do not normatively pin a numeric exit code in this plan; use non-zero + `error.type == "contract_violation"` as compatibility contract.
- Extend `StepResult` dataclass to include optional `artifacts` field.

**Step 4: Run tests to verify pass**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_output_contract_integration.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
cd ~/Documents/agent-orchestration
git add orchestrator/workflow/executor.py orchestrator/state.py tests/test_workflow_output_contract_integration.py
git commit -m "feat(executor): enforce expected_outputs and persist artifacts"
```

## Task 4: Add Deterministic Prompt Contract Injection for Provider Steps

**Files:**
- Create: `~/Documents/agent-orchestration/orchestrator/contracts/prompt_contract.py`
- Modify: `~/Documents/agent-orchestration/orchestrator/workflow/executor.py`
- Test: `~/Documents/agent-orchestration/tests/test_prompt_contract_injection.py`
- Modify: `~/Documents/agent-orchestration/specs/providers.md`

**Step 1: Write failing tests**

Cases:
- provider step with `expected_outputs` appends generated contract block to composed prompt
- `inject_output_contract: false` disables appending
- command steps accept `inject_output_contract` but ignore it (no prompt path, no behavior change)

**Step 2: Run test to verify failure**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_prompt_contract_injection.py -v`

Expected: FAIL.

**Step 3: Implement injection**

- Build renderer that emits stable contract block text from `expected_outputs`.
- Append block in `_execute_provider_with_context` after input/dependency composition and before provider execution.
- Keep existing literal input file semantics unchanged except for additive suffix.
- Use explicit `name` keys in rendered contract text to match substitution contract.

**Step 4: Run tests to verify pass**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_prompt_contract_injection.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
cd ~/Documents/agent-orchestration
git add orchestrator/contracts/prompt_contract.py orchestrator/workflow/executor.py tests/test_prompt_contract_injection.py specs/providers.md
git commit -m "feat(providers): inject deterministic output contract block into prompts"
```

## Task 5: Add Prototype Workflows for Requested Patterns

**Files:**
- Create: `~/Documents/agent-orchestration/workflows/examples/backlog_plan_execute_v0.yaml`
- Create: `~/Documents/agent-orchestration/workflows/examples/test_fix_loop_v0.yaml`
- Create: `~/Documents/agent-orchestration/workflows/examples/unit_of_work_plus_test_fix_v0.yaml`
- Modify: `~/Documents/agent-orchestration/specs/examples/patterns.md`
- Test: `~/Documents/agent-orchestration/tests/test_workflow_examples_v0.py`

**Step 1: Write failing tests**

- validate workflow files load
- validate control-flow paths are valid
- smoke-run with mocked commands/providers to verify loop behavior and handoff files

**Step 2: Run test to verify failure**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_examples_v0.py -v`

Expected: FAIL (examples/tests missing).

**Step 3: Implement example workflows**

- Pattern A: `docs/backlog/ -> pick item -> draft plan -> execute plan -> optional review loop`.
- Pattern B: `unit_of_work -> call test_fix loop until pass/threshold or max cycles`.
- Use artifact files under `state/` for handoff (`backlog_item_path.txt`, `plan_path.txt`, `review_decision.txt`, `review_score.txt`).

**Step 4: Run tests to verify pass**

Run: `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_examples_v0.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
cd ~/Documents/agent-orchestration
git add workflows/examples/backlog_plan_execute_v0.yaml workflows/examples/test_fix_loop_v0.yaml workflows/examples/unit_of_work_plus_test_fix_v0.yaml specs/examples/patterns.md tests/test_workflow_examples_v0.py
git commit -m "feat(examples): add artifact-contract backlog and test-fix prototype workflows"
```

## Task 6: Verification Sweep and Prototype Runbook

**Files:**
- Modify: `~/Documents/agent-orchestration/tests/README.md`
- Create: `~/Documents/agent-orchestration/workflows/examples/README_v0_artifact_contract.md`

**Step 1: Run targeted selectors**

Run:
- `cd ~/Documents/agent-orchestration && pytest tests/test_loader_validation.py -k expected_outputs -v`
- `cd ~/Documents/agent-orchestration && pytest tests/test_output_contract.py -v`
- `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_output_contract_integration.py -v`
- `cd ~/Documents/agent-orchestration && pytest tests/test_prompt_contract_injection.py -v`
- `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_examples_v0.py -v`

Expected: all PASS.

**Step 2: Run dry-run + runtime integration proof**

Run:
- `cd ~/Documents/agent-orchestration && orchestrate run workflows/examples/backlog_plan_execute_v0.yaml --dry-run`
- `cd ~/Documents/agent-orchestration && pytest tests/test_workflow_examples_v0.py -k runtime -v`

Expected:
- dry-run passes schema/control-flow validation
- runtime selector proves artifact files are produced/validated and loop exits by gate or max-cycle guard.

**Step 3: Document runbook**

- Add minimal “how to run prototype” and known limitations.

**Step 4: Commit**

```bash
cd ~/Documents/agent-orchestration
git add tests/README.md workflows/examples/README_v0_artifact_contract.md
git commit -m "docs: add v0 artifact-contract workflow runbook and test map"
```

## Out-of-Scope for v0

- New general step type for `workflow_call` (use command-based call for now).
- Complex expressions or advanced branching DSL.
- Parallel execution blocks.
- Replacing existing `scripts/orchestration` runtime.

## Suggested Branch

- `feature/artifact-contract-prototype-v0`

## Success Criteria

- Deterministic file-based handoff works without parsing stdout JSON, using explicit artifact keys (`name`).
- Prototype workflows demonstrate both requested patterns.
- Review/test-fix loop terminates deterministically via explicit gates and max-cycle guard.
- Contract enforcement is symlink-safe and prevents workspace/`under` path escapes.
- Execution failures remain debuggable (original failures are not masked by contract checks).
- All new selectors, including at least one runtime integration selector, pass in `~/Documents/agent-orchestration`.

## Artifacts Index
- Plan file: `docs/plans/2026-02-20-agent-orchestration-artifact-contract-prototype.md`
- Reports root: `docs/plans/2026-02-20-agent-orchestration-artifact-contract-prototype/reports/`
- Prototype implementation repo: `~/Documents/agent-orchestration`
