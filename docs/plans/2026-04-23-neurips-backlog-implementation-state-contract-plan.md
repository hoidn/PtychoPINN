# NeurIPS Backlog Implementation-State Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the NeurIPS steered backlog drain to distinguish between a selected item that is still running, one that is semantically blocked, and a true workflow failure, so long-running experiments do not get incorrectly recorded as blocked.

**Architecture:** Add an explicit implementation-state contract to the repo-local implementation phase, with separate progress and final-report surfaces. Route selected-item outcomes through `COMPLETED | RUNNING | BLOCKED` instead of inferring item state from missing final artifacts. Keep deterministic workflow control in YAML and limit provider prompts to local execution judgment plus status reporting.

**Tech Stack:** agent-orchestration DSL v2.7, PtychoPINN NeurIPS backlog workflows, Python helper scripts, pytest workflow/runtime tests

---

## Scope and intent

This plan is narrowly about the bug exposed by the Markov `history_len=1` item:

- the implementation provider completed a partial execution pass
- the required final execution report did not exist yet
- the workflow treated that as an implementation-phase failure
- the selected item was recorded as `BLOCKED`

That behavior is wrong for long-running experiment work. The fix is not to weaken final-report requirements. The fix is to introduce an explicit non-terminal execution state and route it cleanly.

This plan does **not** redesign selection, roadmap sync, or plan drafting. It only touches the selected-item / implementation-state contract and the top-level drain status needed to carry it.

## Files and boundaries

### Modify

- `workflows/library/neurips_backlog_implementation_phase.yaml`
  - Own the implementation-state contract and route provider output into deterministic phase outcomes.
- `workflows/library/prompts/neurips_backlog_implementation_phase/implement_implementation.md`
  - Tell the provider to emit explicit state plus the right report surface for that state.
- `workflows/library/prompts/neurips_backlog_implementation_phase/fix_implementation.md`
  - Same contract for revise passes.
- `workflows/library/neurips_selected_backlog_item.yaml`
  - Stop collapsing implementation call failure and semantic item blocking into the same route.
  - Add a clean `WAITING` selected-item outcome.
- `workflows/examples/neurips_steered_backlog_drain.yaml`
  - Accept and propagate `WAITING` as a top-level drain outcome instead of forcing `CONTINUE | DONE | BLOCKED`.
- `tests/studies/test_neurips_steered_backlog_workflow.py`
  - Assert the new workflow contracts and routes.
- `tests/studies/test_neurips_steered_backlog_runtime.py`
  - Add a smoke/runtime case covering `RUNNING -> WAITING`.
- `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`
  - Update the design to reflect the principled execution-state model.
- `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md`
  - Update the earlier implementation plan so it no longer describes the broken two-state assumption.
- `docs/index.md`
  - Index this follow-on plan if the repo convention is to index current implementation plans.

### Keep unchanged unless a task proves otherwise

- selector workflow and prompts
- roadmap-sync phase
- seeded plan phase
- backlog queue helper scripts unrelated to implementation-state routing

## Design decisions locked by this plan

- A selected backlog item must have three semantic implementation outcomes:
  - `COMPLETED`
  - `RUNNING`
  - `BLOCKED`
- Workflow/infra failure is **not** a semantic item outcome.
  - A bad pointer, contract mismatch, helper crash, or malformed runtime state should fail the run, not mark the backlog item blocked.
- Final execution reports remain required for `COMPLETED`.
- Progress reports are the required artifact for `RUNNING`.
- `BLOCKED` requires a reason-bearing summary artifact, not merely a missing final report.
- A selected item in `RUNNING` stays in `docs/backlog/in_progress/`.
- The drain must be able to finish cleanly in a non-error `WAITING` state when the current selected item is still executing.

## Target contract shape

### Implementation phase outputs

Replace the current implicit contract with an explicit bundle that includes:

- `execution_state`: enum `COMPLETED | RUNNING | BLOCKED`
- `progress_report_path`: optional relpath under `artifacts/work`
- `final_execution_report_path`: optional relpath under `artifacts/work`
- `checks_report_path`: optional relpath under `artifacts/checks`
- `implementation_review_report_path`: optional relpath under `artifacts/review`
- `implementation_review_decision`: optional enum `APPROVE | REVISE`
- `block_reason`: optional string

Validation rules:

- `COMPLETED`
  - requires `final_execution_report_path`
  - enters checks/review flow
- `RUNNING`
  - requires `progress_report_path`
  - must not require final review/check artifacts yet
- `BLOCKED`
  - requires `block_reason`
  - may optionally include a progress report, but does not require final review/check artifacts

### Selected-item workflow outputs

Selected-item outcome becomes:

- `CONTINUE` when the item completed and was approved
- `WAITING` when the item is still running
- `BLOCKED` only when the item is semantically blocked

### Top-level drain outputs

Drain output becomes:

- `CONTINUE`
- `DONE`
- `WAITING`
- `BLOCKED`

`WAITING` means:

- the workflow did not fail
- the selected item remains owned by the run and stays in `in_progress`
- the operator should resume later or the workflow should be resumed after external completion

## Implementation tasks

### Task 1: Update the design and implementation docs

**Files:**
- Modify: `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`
- Modify: `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Add the execution-state model to the design doc**

Document:
- semantic `COMPLETED | RUNNING | BLOCKED`
- non-semantic workflow failure
- separate progress vs final report surfaces
- `WAITING` top-level drain outcome

- [ ] **Step 2: Update the older implementation plan**

Replace the old assumption that implementation either yields a final report or fails. Explicitly document:
- output-bundle-based execution-state contract
- `WAITING` route
- “raw call failure does not auto-block the item”

- [ ] **Step 3: Index this follow-on plan if needed**

Run:
```bash
rg -n "implementation-state contract|WAITING|RUNNING" docs/index.md docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md
```

- [ ] **Step 4: Verify docs formatting**

Run:
```bash
git diff --check -- docs/index.md docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md
```

### Task 2: Add failing workflow-structure tests for the new contract

**Files:**
- Modify: `tests/studies/test_neurips_steered_backlog_workflow.py`

- [ ] **Step 1: Add a failing test for implementation-phase contract shape**

Cover:
- implementation phase publishes `execution_state`
- `RUNNING` path does not require final execution report publication
- `COMPLETED` path still requires the final report route

- [ ] **Step 2: Add a failing test for selected-item routing**

Cover:
- `RunImplementationPhase` semantic `RUNNING` routes to a non-blocking outcome
- `RunImplementationPhase` semantic `BLOCKED` routes to `RecordImplementationBlocked`
- raw `call` failure is not used as the normal semantic block signal

- [ ] **Step 3: Add a failing test for top-level drain outputs**

Cover:
- top-level drain contract includes `WAITING`
- `repeat_until` terminates on `DONE`, `BLOCKED`, or `WAITING`

- [ ] **Step 4: Collect only**

Run:
```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py --collect-only -q
```

Expected:
- new test names appear

### Task 3: Add failing runtime smoke for the `RUNNING -> WAITING` path

**Files:**
- Modify: `tests/studies/test_neurips_steered_backlog_runtime.py`
- Reuse/add fixture data only if strictly needed under `tests/fixtures/neurips_steered_backlog/`

- [ ] **Step 1: Add a failing runtime test where implementation returns `RUNNING`**

Test shape:
- selector chooses one item
- roadmap sync returns `NO_CHANGE`
- plan phase approves
- implementation phase returns a progress report and `execution_state=RUNNING`

Assertions:
- run completes successfully
- top-level output is `WAITING`
- item remains in `docs/backlog/in_progress/`
- run state does not add the item to `blocked_items`

- [ ] **Step 2: Add a failing runtime test for real semantic `BLOCKED`**

Assertions:
- `execution_state=BLOCKED` produces drain status `BLOCKED`
- block reason is preserved in the item summary / run-state ledger

- [ ] **Step 3: Run the targeted runtime selector**

Run:
```bash
pytest tests/studies/test_neurips_steered_backlog_runtime.py -q
```

Expected:
- fail on the new tests before implementation

### Task 4: Refactor the implementation phase to emit explicit execution state

**Files:**
- Modify: `workflows/library/neurips_backlog_implementation_phase.yaml`
- Modify: `workflows/library/prompts/neurips_backlog_implementation_phase/implement_implementation.md`
- Modify: `workflows/library/prompts/neurips_backlog_implementation_phase/fix_implementation.md`

- [ ] **Step 1: Replace the implicit final-report assumption with an `output_bundle`**

Refactor the phase so the provider writes a deterministic JSON status bundle, not just prose plus an expected final report side effect.

Suggested bundle fields:
- `execution_state`
- `progress_report_path`
- `final_execution_report_path`
- `block_reason`

- [ ] **Step 2: Add deterministic publication steps per semantic branch**

Use `match` on `execution_state`:
- `COMPLETED`
  - publish final execution report
  - continue to checks / review loop
- `RUNNING`
  - publish progress report
  - finalize phase outputs without checks/review
- `BLOCKED`
  - publish blocked summary / reason
  - finalize phase outputs without checks/review

- [ ] **Step 3: Update prompt instructions**

`implement_implementation.md` and `fix_implementation.md` must explicitly tell the provider:
- emit one of `COMPLETED | RUNNING | BLOCKED`
- write a progress report when the work is still running
- write the final execution report only when the implementation pass is actually complete
- do not fake completion just to satisfy the workflow

- [ ] **Step 4: Keep final-report strictness for `COMPLETED`**

Do not weaken:
- final execution report existence check
- checks report and review requirements once the phase claims completion

- [ ] **Step 5: Run the narrow workflow-structure test**

Run:
```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py -q
```

Expected:
- structure tests now pass

### Task 5: Route selected-item outcomes through `CONTINUE | WAITING | BLOCKED`

**Files:**
- Modify: `workflows/library/neurips_selected_backlog_item.yaml`

- [ ] **Step 1: Stop treating any implementation-phase failure as semantic blocking**

Change the current route so:
- semantic `BLOCKED` comes from the implementation phase output
- raw `call` failure remains a run failure unless there is an explicit deterministic translation step with evidence

- [ ] **Step 2: Add `WAITING` outcome handling**

When implementation phase returns `RUNNING`:
- write an item summary with `item_outcome: WAITING`
- keep the selected item in `docs/backlog/in_progress/`
- do not update `blocked_items`
- write the selected-item outcome bundle with `drain_status: WAITING`

- [ ] **Step 3: Keep `CONTINUE` path unchanged for approved completions**

Make sure:
- completed item still moves to `done`
- run-state ledger still records completion

- [ ] **Step 4: Keep true semantic `BLOCKED` path explicit**

Require a real block reason and write it to the summary artifact before routing to `BLOCKED`.

- [ ] **Step 5: Run selected workflow tests**

Run:
```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py -q
```

Expected:
- selected-item routing tests pass

### Task 6: Teach the top-level drain about `WAITING`

**Files:**
- Modify: `workflows/examples/neurips_steered_backlog_drain.yaml`

- [ ] **Step 1: Add `WAITING` to top-level output contract**

Update:
- workflow outputs
- top-level drain artifact contract
- `repeat_until.outputs.drain_status`
- termination condition

- [ ] **Step 2: Make `WAITING` a clean terminal outcome**

The run should finish successfully in `WAITING` with:
- current selected item still in `in_progress`
- durable run-state path
- drain summary path

- [ ] **Step 3: Make sure `BLOCKED` still means actual blocked work**

Do not change:
- `DONE` when no eligible work remains
- `BLOCKED` when the selected item is semantically blocked or nothing runnable remains

- [ ] **Step 4: Run dry-run validation**

Run:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/examples/neurips_steered_backlog_drain.yaml --dry-run --stream-output
```

Expected:
- workflow validation successful

### Task 7: Runtime verification and regression coverage

**Files:**
- Modify only if test/debug output requires it

- [ ] **Step 1: Run focused workflow + runtime tests**

Run:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration pytest \
  tests/studies/test_neurips_steered_backlog_workflow.py \
  tests/studies/test_neurips_steered_backlog_runtime.py -q
```

- [ ] **Step 2: Run collect-only if any tests were added or renamed**

Run:
```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py tests/studies/test_neurips_steered_backlog_runtime.py --collect-only -q
```

- [ ] **Step 3: Run a real orchestrator smoke from repo root**

Use a bounded fixture-backed smoke or mocked runtime path that exercises:
- `SELECTED`
- `NO_CHANGE`
- `APPROVE`
- implementation `RUNNING`
- top-level `WAITING`

If a second smoke is cheap, also exercise semantic `BLOCKED`.

- [ ] **Step 4: Verify no prompt-text assertion creep**

Run:
```bash
rg -n "select_next_item|implement_implementation|fix_implementation|review_or_update" tests/studies
```

Review results manually to ensure tests assert contracts/behavior, not literal prompt text.

### Task 8: Recovery check on the real backlog item

**Files:**
- No new source files unless the runtime surface proves another gap

- [ ] **Step 1: Inspect the current blocked Markov item state**

Confirm:
- item remains in `docs/backlog/in_progress/`
- run-state ledger still records the old blocked event

- [ ] **Step 2: Define the operator recovery path**

The operator path after this fix should be:
- clear stale blocked state for the Markov item
- relaunch or resume in a way that routes the item to `WAITING` instead of `BLOCKED` if the long run is still incomplete

- [ ] **Step 3: Document exact recovery commands in the operator runbook or execution notes**

Do not leave the recovery method implicit.

## Verification checklist

- `git diff --check`
- `pytest tests/studies/test_neurips_steered_backlog_workflow.py -q`
- `pytest tests/studies/test_neurips_steered_backlog_runtime.py -q`
- `pytest tests/studies/test_neurips_steered_backlog_workflow.py tests/studies/test_neurips_steered_backlog_runtime.py --collect-only -q`
- `PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/examples/neurips_steered_backlog_drain.yaml --dry-run --stream-output`
- at least one real fixture-backed or mocked orchestrator smoke that yields `WAITING`

## Risks and notes

- The biggest risk is accidentally weakening final completion gates while adding `RUNNING`. Avoid that by keeping separate progress and final report paths.
- The second risk is still conflating workflow failure and semantic blocking. Keep that distinction explicit in YAML routing and item-summary writing.
- Do not solve this by letting missing final reports pass silently.
- Do not solve this by teaching prompts to talk about workflow routing. Prompts should only emit local execution state and required report paths.
