# Prompts Index

This index maps prompt files in `prompts/` to their purpose and primary usage.

## Backlog Plan Loop Prompts

These are used by agent-orchestration workflows in `workflows/agent_orchestration/`.

| Prompt | Purpose | Used By |
|---|---|---|
| `workflows/backlog_plan_loop/execute_plan.md` | Execute selected backlog item's plan. | dual-review + legacy single-review |
| `workflows/backlog_plan_loop/review_impl_vs_plan.md` | Review implementation vs plan execution evidence. | dual-review + legacy single-review |
| `workflows/backlog_plan_loop/review_plan_level_issues.md` | Review plan-level issues and required plan revisions. | dual-review |
| `workflows/backlog_plan_loop/fix_issues.md` | Re-run/fix after review requests revision. | dual-review + legacy single-review |
| `workflows/backlog_plan_loop/commit_on_approve.md` | Commit approved cycle changes and record commit SHA. | dual-review + legacy single-review |
| `workflows/backlog_plan_loop/assess_execution_completion.md` | Generic assess/gate prompt artifact (reference). | not currently wired in active backlog workflows |
| `workflows/backlog_plan_loop/review_and_revise_plan_issues.md` | Combined plan review+revision prompt (single pass). | available; not currently wired in active backlog workflows |

Workflow references:
- recommended: `workflows/agent_orchestration/backlog_plan_slice_impl_dual_review_loop.yaml`
- legacy: `workflows/agent_orchestration/backlog_plan_slice_impl_review_loop.yaml`

## Lines 256 Experiment Prompt

| Prompt | Purpose | Used By |
|---|---|---|
| `workflows/lines_256_arch_improvement/experiment_step.md` | Prepare exactly one candidate `lines_256` architecture-improvement attempt without running the experiment or owning ledger/keep-discard logic. | `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml` |
| `workflows/lines_256_arch_improvement/debug_crash.md` | Diagnose one concrete candidate crash, prepare a focused crash-fix candidate if warranted, or block cleanly. | `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml` |

## Core Agent Prompts

| Prompt | Purpose |
|---|---|
| `main.md`, `main2.md` | Main agent prompt variants. |
| `supervisor.md`, `supervisor2.md` | Supervisor loop behavior and policy variants. |
| `taskctx.md` | Task context framing for focused execution. |
| `retrospective.md` | Retrospective/postmortem writing prompt. |

## Spec and Architecture Prompts

| Prompt | Purpose |
|---|---|
| `spec_writer.md` | Draft or revise specifications. |
| `spec_reviewer.md` | Review specs for correctness/completeness. |
| `arch_writer.md` | Draft architecture documentation. |
| `arch_reviewer.md` | Review architecture docs/design quality. |

## Debugging and Analysis Prompts

| Prompt | Purpose |
|---|---|
| `debug.md`, `debug2.md` | Debugging workflows and failure triage. |
| `fsm_analysis.md` | State-machine/workflow behavior analysis. |
| `callchain.md` | Trace call paths and control/data flow. |
| `full_suite.md` | Full validation/test sweep orchestration. |
| `pyrefly.md` | Python refactor/analysis support prompt. |

## Operational / Process Prompts

| Prompt | Purpose |
|---|---|
| `git_setup_agent.md` | Git setup runbook for automation contexts. |
| `git_hygiene.md` | Ongoing git hygiene guidelines. |
| `doc_sync_sop.md` | Documentation sync standard operating procedure. |
| `housekeep_fix_plan.md` | Housekeeping-oriented fix planning. |
| `update_fix_plan.md` | Update/refresh existing fix plans. |
| `ARRP_PROTOCOL.md` | Process/protocol reference prompt. |

## Migration

| Prompt | Purpose |
|---|---|
| `migration/agent-prompt-migration-guide.md` | Guide for migrating older prompt sets/contracts. |
