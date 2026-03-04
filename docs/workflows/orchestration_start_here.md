# Orchestration Start Here (Backlog Loop)

This document explains the backlog-loop orchestration system in concrete terms.
It separates:

- orchestration policy
- workflow DSL authoring
- runtime step execution

Scope:
- `workflows/agent_orchestration/backlog_plan_slice_impl_dual_review_loop.yaml`
- `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

## Read Order

1. This file (`docs/workflows/orchestration_start_here.md`)
2. Runtime runbook: `docs/workflows/agent_orchestration_backlog_loop.md`
3. Provider prompt rules: `docs/PROMPT_DRAFTING_GUIDE.md`
4. Plan contract template: `docs/plans/templates/workflow_contract_plan.md`

## One-Screen Mental Model

```text
Design-time (authoring)                                  Runtime (execution)
--------------------------------------------------------------------------------
Edit workflow DSL YAML  ------------------------------->  Orchestrator loads step graph
Edit prompt markdown files ---------------------------->  Provider steps run with composed prompts
Edit plan docs ---------------------------------------->  Plan content is consumed as inputs
Edit backlog items (priority/plan_path/check_commands)->  Queue selects next item and drives loop
```

Writing a DSL workflow is authoring.
Running a DSL workflow is runtime.
Orchestration includes both, plus queue and policy conventions around them.

## Glossary (Concrete, Repo-Scoped)

| Term | What it means here | Concrete location |
| --- | --- | --- |
| Queue | Ordered set of pending work items | `docs/backlog/active/*.md` |
| Policy | Rules for selection, review loops, gating, transitions, commits | Encoded in workflow YAML + documented in runbook/docs |
| Runbook | Human instructions for launch/monitor/resume/recovery | `docs/workflows/agent_orchestration_backlog_loop.md` |
| Workflow | Executable orchestration definition | `workflows/agent_orchestration/*.yaml` |
| DSL | YAML schema used to describe workflow graph/contracts | `steps`, `on.goto`, `artifacts`, `publishes`, `consumes` |
| Step | Single workflow node (`command` or provider) | One `steps[]` entry in workflow YAML |
| Step execution | One runtime invocation of a step | Recorded under `.orchestrate/runs/<run_id>/` |
| Prompt | Instructions for one provider step invocation | `prompts/workflows/backlog_plan_loop/*.md` |
| Plan | Human intent/scope/acceptance doc for work | `docs/plans/*.md` |
| Authoring | Editing workflow/prompt/plan/backlog files | Git changes before running workflow |
| Runtime | Orchestrator executing steps and enforcing contracts | `python -m orchestrator.cli.main run|resume ...` |

## Relationship Diagram (Control + Data)

```text
                 +----------------------------------------------+
                 | Workflow DSL YAML                            |
                 | (step graph, routing, artifact contracts)    |
                 +----------------------+-----------------------+
                                        |
                                        v
                            +-------------------------+
                            | Orchestrator Runtime    |
                            | executes steps in order |
                            +------+------------------+
                                   |
                +------------------+-------------------+
                |                                      |
                v                                      v
    +----------------------------+         +----------------------------+
    | Command steps              |         | Provider steps             |
    | deterministic shell/python |         | prompts + model execution  |
    +-------------+--------------+         +-------------+--------------+
                  |                                        ^
                  v                                        |
    +----------------------------+              +----------+-----------+
    | Artifacts / state pointers |<-------------| Prompt files         |
    | (publishes/consumes)       |   consumed   | (invocation text)    |
    +-------------+--------------+   artifacts  +----------------------+
                  ^
                  |
    +-------------+--------------+
    | Backlog queue item         |
    | (priority, plan_path, ...) |
    +-------------+--------------+
                  |
                  v
    +----------------------------+
    | Plan document(s)           |
    | implementation intent      |
    +----------------------------+
```

## Execution Timeline (What Actually Happens)

```text
1) Runtime starts with a workflow YAML.
2) Queue selection step picks one backlog item from docs/backlog/active.
3) Selected item provides plan_path + check_commands.
4) Provider step prompt is composed from:
   - prompt file text
   - consumed artifacts injection
   - output contract injection
5) Step runs and writes declared outputs.
6) Runtime validates outputs and updates artifact lineage.
7) Gate steps read artifacts and choose goto path.
8) Loop repeats (review/fix) or transitions item active->done + commit path.
```

## What Belongs Where (Boundary Rules)

Change workflow YAML when you need to change:
- step ordering
- `goto` routing
- retry/review loop behavior
- producer/consumer contract semantics
- fail-open vs fail-closed gates

Change prompt files when you need to change:
- invocation-local provider instructions
- how a provider step analyzes/edits using consumed artifacts
- output content guidance (not routing)

Change plan docs when you need to change:
- goal/scope/acceptance criteria
- evidence expectations
- high-level contract intent for in-workflow artifacts

Change backlog item docs when you need to change:
- queue priority
- selected plan (`plan_path`)
- targeted checks (`check_commands`)

## Plan-to-Workflow Contract Rule

For plans executed by this orchestration flow:
- expose at most 2 workflow-level artifacts (`< 3`)
- name producer and consumer for each artifact
- resolve inter-stage artifacts from current producer outputs only
- do not hardcode timestamped historical roots
- fail closed when required producer artifacts are missing/stale

Template:
- `docs/plans/templates/workflow_contract_plan.md`

## Common Confusions (and Correct Placement)

Confusion: "Prompt should decide whether to loop/retry."
- Correct: loop/retry belongs in workflow gates and `goto` paths.

Confusion: "Plan prose is enough to enforce producer/consumer behavior."
- Correct: producer/consumer behavior must be encoded in workflow DSL contracts.

Confusion: "Hardcoded historical output path is acceptable if it exists."
- Correct: canonical inter-stage inputs must resolve from current producer artifacts.

Confusion: "Runbook defines execution logic."
- Correct: runbook explains operations; workflow YAML defines executable logic.

## If You Are About to Edit

Before editing, classify the change:
- behavior/routing/contract change -> workflow YAML
- invocation instruction change -> prompt file
- scope/acceptance change -> plan doc
- queue selection change -> backlog item

If a change spans multiple categories, update each corresponding layer explicitly.
