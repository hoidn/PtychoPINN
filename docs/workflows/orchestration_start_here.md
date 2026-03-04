# Orchestration Start Here

This is the entry point for the agent-orchestration backlog loop in PtychoPINN.
Use this doc to understand ownership boundaries before editing workflow files, prompts, or plans.

## Read Order

1. This doc (`docs/workflows/orchestration_start_here.md`)
2. Runbook: `docs/workflows/agent_orchestration_backlog_loop.md`
3. Prompt rules: `docs/PROMPT_DRAFTING_GUIDE.md`
4. Plan template: `docs/plans/templates/workflow_contract_plan.md`

## Core Definitions

- Workflow: a DSL file that defines step graph, routing, retries/loops, and deterministic artifact contracts.
- Step: one executable unit in a workflow (`command` or provider call), with declared inputs/outputs.
- Prompt: instructions for a provider step invocation; prompt text is not control-flow logic.
- Plan: human-authored scope and intent document under `docs/plans/`; source of implementation intent, not runtime routing.
- Backlog item: queue metadata (`priority`, `plan_path`, `check_commands`) that selects what plan gets executed.

## Ownership Boundaries (Do Not Mix)

- Workflow owns:
  - Step ordering and `goto` routing
  - Retry/review loop policy
  - Producer/consumer artifact contracts
  - Fail-open vs fail-closed gate behavior
- Step owns:
  - Local execution of one operation
  - Producing declared outputs only
- Prompt owns:
  - Invocation-local instructions
  - How to perform the current task against consumed artifacts
  - Clear constraints for in-scope edits
- Plan owns:
  - What should be built/fixed and why
  - Acceptance criteria and evidence expectations
  - At most a small workflow-level contract surface (see template)

If a statement decides routing, it belongs in workflow DSL, not prompt text.
If a statement defines deterministic handoff, it belongs in artifact contracts, not prose-only plan sections.

## Relationship Map

Backlog item -> `plan_path` -> workflow `SelectBacklogItem` step -> consumed artifacts injected into provider prompt -> provider writes declared outputs -> workflow gates consume those outputs -> loop/commit decisions.

The workflow executes plans; plans do not directly execute or route workflow steps.

## Plan-to-Workflow Contract Rule

For in-workflow plans, expose only a small contract surface:
- Max 2 workflow-level artifacts (`< 3`).
- Each artifact must have a named producer and consumer.
- No hardcoded timestamped/historical output roots for inter-stage inputs.
- Consumer must fail closed when required producer artifact is missing/stale.

Template:
- `docs/plans/templates/workflow_contract_plan.md`

## Common Failure Modes

- Prompt text encodes workflow control semantics ("if review fails, go fix ...").
- Plan prose implies producer/consumer behavior not represented in workflow DSL.
- Hardcoded historical output paths used as canonical inter-stage inputs.
- Queue/process semantics duplicated in prompts instead of enforced in steps/gates.

## Practical Rule of Thumb

- Change workflow YAML when behavior/routing/contracts should change.
- Change prompt files when invocation instructions should change.
- Change plan docs when goals/scope/acceptance should change.

If you are touching more than one of these layers, update all linked docs in the read order above.
