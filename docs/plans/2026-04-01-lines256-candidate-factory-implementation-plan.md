# Lines 256 Candidate Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable `candidate_factory` boundary to the `lines_256` v2 controller so queued broad redesign ideas can be produced by a subordinate design/plan/implement/review workflow while the controller keeps ownership of smoke, scoring, keep/discard, queue movement, and resume.

**Architecture:** Keep the existing outer controller loop and thin workflow intact. Extract the current prompt-driven proposal path into a `direct` factory, add a queue-only `redesign` factory that calls a subordinate workflow, and force both paths to emit the same `candidate_metadata.json` / `proposal_result.json` contract so the downstream controller path stays unchanged.

**Tech Stack:** Python controller (`scripts/studies/lines_256_session_controller.py`), agent-orchestration DSL YAML workflows, provider prompt files under `prompts/workflows/lines_256_arch_improvement/`, pytest, orchestrator dry-run validation.

---

## File Map

**Primary controller/runtime files**
- Modify: `scripts/studies/lines_256_session_controller.py`
  - Add queue frontmatter parsing for `candidate_factory`
  - Extract current proposal path behind a `direct` factory function
  - Add queue-only `redesign` factory dispatch
  - Preserve existing smoke/scored/acceptance behavior
- Modify: `workflows/agent_orchestration/lines_256_session_controller.yaml`
  - Add validation inputs needed by the redesign workflow path if necessary

**New workflow files**
- Create: `workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml`
  - Subordinate candidate-production workflow
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_design.md`
  - Design-step prompt
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_plan.md`
  - Plan-step prompt
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_review.md`
  - Review/fix-step prompt

**Docs**
- Modify: `docs/studies/lines_256_controller_loop.md`
  - Document factory selection, redesign workflow, and queue-only phase-1 policy
- Modify: `docs/workflow_queue/README.md`
  - Document `candidate_factory: redesign` frontmatter
- Optionally modify: `docs/findings.md`
  - Record the architectural boundary change if the repo treats it as a notable workflow finding

**Tests**
- Modify: `tests/studies/test_lines_256_session_controller.py`
  - Factory selection tests
  - Proposal/resume behavior tests
  - Redesign output-contract acceptance tests
- Create if needed: `tests/studies/test_lines_256_redesign_candidate_workflow.py`
  - Narrow workflow-shape / artifact-contract tests

## Task 1: Lock Down The Factory Selection Contract

**Files:**
- Modify: `tests/studies/test_lines_256_session_controller.py`
- Reference: `docs/plans/2026-04-01-lines256-candidate-factory-redesign-spec.md`

- [ ] **Step 1: Add failing tests for queue frontmatter factory selection**

Cover at least:
- queued item without `candidate_factory` selects `direct`
- queued item with `candidate_factory: redesign` selects `redesign`
- queued item with unknown `candidate_factory` produces a retryable proposal failure without moving the queue item

- [ ] **Step 2: Run the narrow selector tests to verify they fail**

Run:
```bash
pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
pytest tests/studies/test_lines_256_session_controller.py -k "candidate_factory or redesign_factory or workflow_queue" -v
```

Expected:
- collection succeeds
- new tests fail because controller selection logic does not exist yet

- [ ] **Step 3: Implement the smallest controller-side frontmatter parsing helpers**

In `scripts/studies/lines_256_session_controller.py`:
- add queue-frontmatter parsing for optional `candidate_factory`
- normalize missing values to `direct`
- reject unknown values deterministically

- [ ] **Step 4: Re-run the narrow selector tests**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "candidate_factory or redesign_factory or workflow_queue" -v
```

Expected:
- selector tests pass

- [ ] **Step 5: Commit the factory-selection slice**

```bash
git add tests/studies/test_lines_256_session_controller.py scripts/studies/lines_256_session_controller.py
git commit -m "feat: add lines256 candidate factory selection"
```

## Task 2: Extract The Direct Candidate Factory Without Changing Behavior

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add a failing regression test that existing direct proposal behavior is still accepted through a factory wrapper**

Test should prove:
- the `direct` factory still writes `candidate_metadata.json`
- controller proceeds into the same smoke/scored path contract
- no downstream special-casing is introduced

- [ ] **Step 2: Run the focused direct-factory regression test**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "direct_factory or proposal_context or proposal_resume" -v
```

Expected:
- new direct-factory regression fails before extraction

- [ ] **Step 3: Refactor current proposal generation into `direct_candidate_factory`**

In `scripts/studies/lines_256_session_controller.py`:
- extract current exploit/explore prompt path into a named helper
- keep existing prompt files and proposal context behavior unchanged
- ensure the extracted helper returns/writes the same authoritative files as before

- [ ] **Step 4: Re-run direct-factory and proposal regression tests**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "direct_factory or proposal_context or proposal_resume" -v
```

Expected:
- all selected tests pass

- [ ] **Step 5: Commit the direct-factory extraction**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "refactor: extract lines256 direct candidate factory"
```

## Task 3: Add The Redesign Candidate Workflow Skeleton

**Files:**
- Create: `workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml`
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_design.md`
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_plan.md`
- Create: `prompts/workflows/lines_256_arch_improvement/redesign_candidate_review.md`
- Modify: `workflows/agent_orchestration/lines_256_session_controller.yaml` only if the top-level validation step must include redesign assets

- [ ] **Step 1: Write the redesign workflow YAML with a minimal artifact contract**

The first version should:
- accept explicit relpath inputs for all controller-owned output paths
- prepare a redesign brief from queued idea + proposal context
- run design, plan, implement, checks, and bounded review/fix phases
- finish by writing `candidate_metadata.json` and `proposal_result.json`

- [ ] **Step 2: Write minimal prompts for redesign design/plan/review**

Each prompt should be task-local:
- design prompt chooses a coherent redesign
- plan prompt writes a candidate-local plan
- review prompt reviews/fixes the candidate package

Do not put controller mechanics in these prompts.

- [ ] **Step 3: Validate the new workflow structure before controller integration**

Run:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml --dry-run --stream-output
```

Expected:
- workflow validates successfully

- [ ] **Step 4: Add workflow-level tests or contract checks if needed**

If a dedicated test module is added:
```bash
pytest --collect-only tests/studies/test_lines_256_redesign_candidate_workflow.py -q
pytest tests/studies/test_lines_256_redesign_candidate_workflow.py -v
```

Expected:
- new workflow-contract tests pass

- [ ] **Step 5: Commit the redesign workflow skeleton**

```bash
git add workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml \
  prompts/workflows/lines_256_arch_improvement/redesign_candidate_design.md \
  prompts/workflows/lines_256_arch_improvement/redesign_candidate_plan.md \
  prompts/workflows/lines_256_arch_improvement/redesign_candidate_review.md \
  workflows/agent_orchestration/lines_256_session_controller.yaml \
  tests/studies/test_lines_256_redesign_candidate_workflow.py
git commit -m "feat: add lines256 redesign candidate workflow"
```

## Task 4: Integrate The Redesign Factory Into The Controller

**Files:**
- Modify: `scripts/studies/lines_256_session_controller.py`
- Modify: `tests/studies/test_lines_256_session_controller.py`

- [ ] **Step 1: Add a failing controller test for redesign-factory invocation**

Cover:
- queued redesign item causes controller to call the subordinate workflow
- subordinate workflow success with valid metadata lets controller continue
- subordinate workflow failure before metadata leaves proposal phase retryable

- [ ] **Step 2: Run the narrow redesign integration tests**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "redesign_factory or proposal_retry or proposal_running" -v
```

Expected:
- new redesign-factory tests fail before integration

- [ ] **Step 3: Implement `redesign_candidate_factory` in the controller**

In `scripts/studies/lines_256_session_controller.py`:
- build the redesign workflow input bundle from controller-owned state
- invoke the subordinate workflow deterministically
- reuse the same candidate-package validation path used by `direct`
- do not add special-case downstream smoke/scored logic

- [ ] **Step 4: Re-run redesign integration tests**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py -k "redesign_factory or proposal_retry or proposal_running" -v
```

Expected:
- redesign integration tests pass

- [ ] **Step 5: Commit controller integration**

```bash
git add scripts/studies/lines_256_session_controller.py tests/studies/test_lines_256_session_controller.py
git commit -m "feat: integrate lines256 redesign candidate factory"
```

## Task 5: Document The New Boundary And Queue Frontmatter

**Files:**
- Modify: `docs/studies/lines_256_controller_loop.md`
- Modify: `docs/workflow_queue/README.md`
- Optionally modify: `docs/findings.md`

- [ ] **Step 1: Update the controller loop doc**

Document:
- `candidate_factory`
- queue-only redesign policy in phase 1
- redesign workflow ownership boundaries
- unchanged controller ownership of smoke/scored/acceptance

- [ ] **Step 2: Update the workflow queue README**

Document:
- optional `candidate_factory: redesign` frontmatter
- queue-only redesign semantics
- reminder that queue movement remains controller-owned

- [ ] **Step 3: Run a doc/mechanical sanity check**

Run:
```bash
git diff --check -- docs/studies/lines_256_controller_loop.md docs/workflow_queue/README.md docs/findings.md
```

Expected:
- no diff-check failures on touched docs

- [ ] **Step 4: Commit the docs slice**

```bash
git add docs/studies/lines_256_controller_loop.md docs/workflow_queue/README.md docs/findings.md
git commit -m "docs: describe lines256 candidate factory workflow"
```

## Task 6: Full Verification

**Files:**
- Modify as needed from earlier tasks

- [ ] **Step 1: Run collection on any new or renamed test modules**

Run:
```bash
pytest --collect-only tests/studies/test_lines_256_session_controller.py -q
pytest --collect-only tests/studies/test_lines_256_redesign_candidate_workflow.py -q
```

Expected:
- collection succeeds for all changed/new modules

- [ ] **Step 2: Run the targeted test modules**

Run:
```bash
pytest tests/studies/test_lines_256_session_controller.py tests/studies/test_lines_256_redesign_candidate_workflow.py -v
```

Expected:
- all targeted tests pass

- [ ] **Step 3: Validate both workflows with orchestrator dry-run**

Run:
```bash
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_redesign_candidate_impl_review.yaml --dry-run --stream-output
```

Expected:
- both workflows validate successfully

- [ ] **Step 4: Run cached-diff check on all touched files**

Run:
```bash
git diff --check
```

Expected:
- no whitespace or patch-format issues

- [ ] **Step 5: Final commit if verification required cleanup**

```bash
git add -A
git commit -m "chore: finalize lines256 candidate factory rollout"
```

## Notes For The Implementer

- Keep the outer study loop deterministic and controller-owned.
- Do not move keep/discard, queue movement, or accepted-state mutation into prompts or the redesign workflow.
- Prefer typed workflow inputs/outputs for the redesign workflow boundary instead of smuggling controller state through prompt prose.
- Keep the first redesign workflow tranche queue-only. Do not add heuristic controller escalation to redesign mode in this implementation.
- If a redesign workflow run succeeds and `candidate_metadata.json` exists, resume should reuse it rather than re-running redesign production blindly.
