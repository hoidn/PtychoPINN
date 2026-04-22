# NeurIPS Steered Backlog Drain Workflow Design

## Design Metadata

- ID: `NEURIPS-STEERED-BACKLOG-DRAIN-001`
- Status: draft
- Date: 2026-04-22
- Owner: Codex session draft for user review
- Scope: PtychoPINN-local orchestration design for the NeurIPS paper effort

## Problem and Goal

The repo has three partially overlapping coordination surfaces for the NeurIPS effort:

- strategic intent that will live in `docs/steering.md`
- the NeurIPS roadmap and progress ledger
- backlog items under `docs/backlog/`

Today those surfaces are not tied together by a single drain workflow. Existing backlog workflows are deterministic and priority-driven, while the NeurIPS roadmap workflow is adaptive but tranche-based rather than backlog-item-based.

The goal of this design is a workflow that:

1. reads high-level strategic intent from `docs/steering.md`
2. selects the next backlog item using steering, roadmap, backlog contents, and current repo/project state
3. moves the selected item to `docs/backlog/in_progress/`
4. updates the roadmap when the selected item exposes real roadmap drift
5. always drafts and reviews a fresh plan for the selected backlog item before implementation
6. runs implementation and review for that item
7. repeats until the active backlog is exhausted or no item is currently runnable

This is a design-only artifact. It does not author the YAML or prompts yet.

## Consumed Inputs and Authority

- `docs/index.md`
- `CLAUDE.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/backlog/templates/backlog_item_workflow.md`
- existing workflow precedents:
  - `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`
  - `workflows/library/roadmap_tranche_selector.yaml`
  - `workflows/library/roadmap_seeded_plan_phase.yaml`
  - `workflows/library/design_plan_impl_implementation_phase.yaml`
  - `workflows/library/backlog_item_design_plan_impl_stack.yaml`

Authority order:

1. user direction in the current session
2. steering document intent
3. approved NeurIPS design and roadmap
4. repo guidance, findings, and current state
5. existing backlog item text

The steering document is strategic intent, not a queue manifest. Backlog items are candidate work units, not authoritative execution plans.

## Decision Summary

- Build this first as a PtychoPINN-local workflow family, not as a generic library workflow.
- Reuse the existing NeurIPS selector/drain pattern and the generic implementation-phase structure as a reference.
- Preserve the current backlog verification contract: `check_commands` remain required backlog-item inputs and must run before each implementation review pass.
- Add a provider-backed backlog selector instead of using deterministic backlog ordering.
- Add an explicit `in_progress` backlog state.
- Always draft and review a fresh plan for the selected backlog item before implementation, even if the item already names a `plan_path`.
- Keep roadmap updates narrow and judgment-driven. Do not rewrite the roadmap every cycle.
- Treat `docs/steering.md` as human-authored and workflow-consumed. The workflow may read it but should not rewrite it.
- Reuse the generic implementation phase as a structural reference, not as an unchanged interface. This workflow needs a repo-local implementation phase that adds backlog `check_commands` and a checks report.
- Consider library extraction only after this shape succeeds in a real repo-local use.

## Why Existing Workflows Are Not Enough

### Good reuse candidates

- `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`
  - already models a selector-driven drain over the NeurIPS program
- `workflows/library/roadmap_tranche_selector.yaml`
  - already models provider-backed “next work unit” selection
- `workflows/library/roadmap_seeded_plan_phase.yaml`
  - already models plan drafting with roadmap/design context
- `workflows/library/design_plan_impl_implementation_phase.yaml`
  - already provides the implementation review/fix loop

### Gaps

- backlog execution today is not agent-driven enough
- no workflow currently consumes a separate steering document as a first-class authority
- no backlog workflow today always redrafts the plan before implementation
- no queue contract today includes `docs/backlog/in_progress/`
- no existing step performs narrow roadmap maintenance before backlog execution

## Proposed Workflow Family

### Top-level workflow

Proposed file:

- `workflows/examples/neurips_steered_backlog_drain.yaml`

Purpose:

- drain the NeurIPS backlog in an adaptive, steering-aware order
- keep roadmap and backlog state synchronized
- always run plan -> implementation per selected item

### Proposed supporting local workflows

- `workflows/library/neurips_backlog_selector.yaml`
- `workflows/library/neurips_backlog_roadmap_sync_phase.yaml`
- `workflows/library/neurips_backlog_seeded_plan_phase.yaml`
- `workflows/library/neurips_backlog_implementation_phase.yaml`

### Proposed deterministic helper scripts

- `workflows/library/scripts/build_neurips_backlog_manifest.py`
- `workflows/library/scripts/move_neurips_backlog_item.py`
- `workflows/library/scripts/update_neurips_backlog_run_state.py`
- `workflows/library/scripts/run_neurips_backlog_checks.py`

These helpers are deterministic data-movers/parsers. Selection and roadmap judgment remain provider-backed.

## Workflow Shape

The intended control flow is:

1. Build backlog manifest from `docs/backlog/active/*.md`
   - include any recoverable `docs/backlog/in_progress/*.md` items whose owning run is no longer active and whose recorded blocker is now satisfied
2. Select next backlog item using:
   - `docs/steering.md`
   - NeurIPS design and roadmap
   - backlog manifest
   - progress ledger
   - selected current state and relevant findings
3. If no item is selectable, exit with `DONE` or `BLOCKED`
4. Move the selected item from `active/` to `in_progress/`
   - or reclaim an eligible `in_progress/` item when recovery conditions are met
5. Run a narrow roadmap-sync phase
6. Draft and review a fresh plan for the selected item
7. Run a repo-local implementation phase that executes the work, runs backlog `check_commands`, and reviews/fixes against the design, plan, and checks report
8. Update backlog location and progress state
9. Repeat

## Backlog Queue Contract

This workflow requires a new backlog subdirectory:

- `docs/backlog/in_progress/`

Queue semantics:

- `docs/backlog/active/`: eligible for selection
- `docs/backlog/in_progress/`: currently owned by an active workflow run
- `docs/backlog/done/`: completed and accepted
- `docs/backlog/paused/`: explicitly excluded from selection

Lifecycle:

1. selector chooses from `active/` plus recoverable `in_progress/` items
2. move selected item to `in_progress/`
3. if implementation is approved, move item to `done/`
4. if blocked or failed, keep item in `in_progress/` and write durable blocker state
5. `in_progress/` is exclusive while its owning workflow run is still active
6. resume the owning run when possible
7. if the owning run is no longer active and the recorded blocker predicate is now satisfied, a later drain pass may reclaim the `in_progress/` item instead of leaving it stranded

## Steering Document Contract

The workflow depends on a stable, human-authored `docs/steering.md`. It should have short stable sections rather than free-form narrative.

Recommended sections:

- `# Current Objective`
- `## Ordered Near-Term Priorities`
- `## Required Comparison Standards`
- `## Fairness / Apples-to-Apples Constraints`
- `## Known Blockers / Deferred Work`
- `## Non-Goals / Not Now`
- `## Initiative Guidance`

The selector should treat `docs/steering.md` as strategic direction. It may resolve ambiguity with roadmap and backlog state, but it should not rewrite steering text.

## Backlog Manifest Builder

The selector should not crawl raw Markdown ad hoc. It should consume a deterministic manifest built from `docs/backlog/active/*.md`.

Proposed manifest fields:

- `item_id`
- `title`
- `path`
- `status`
- `plan_path` if present
- `check_commands`
- `summary`
- `prerequisites`
- `related_roadmap_phases`
- `blocking_signals`
- `signals_for_selection`

The manifest builder should also reject malformed backlog items early so the selector sees a typed candidate list instead of arbitrary Markdown.

## Selector Contract

Proposed workflow:

- `workflows/library/neurips_backlog_selector.yaml`

Purpose:

- choose the next backlog item to execute based on strategy, dependencies, and current project state

Consumed inputs:

- `docs/steering.md`
- NeurIPS design
- the authoritative roadmap pointer from state
- backlog manifest
- progress ledger
- current backlog run-state ledger
- optional recent execution summaries

Output bundle:

- `selection_status`: `SELECTED | DONE | BLOCKED`
- `selected_item_id`
- `selected_item_path`
- `selection_rationale`
- `roadmap_sync_hint`: `NO_CHANGE | REVIEW_RECOMMENDED`
- `blocking_reasons` when not `SELECTED`

Selector judgment standard:

- prefer items that directly advance the current steering objective
- enforce explicit prerequisites
- prefer items whose inputs are already available
- avoid items blocked by in-flight roadmap phases, missing evidence, or missing upstream artifacts
- avoid deterministic “first item wins” behavior unless the evidence genuinely makes the choice obvious

## Roadmap Sync Phase

Proposed workflow:

- `workflows/library/neurips_backlog_roadmap_sync_phase.yaml`

Purpose:

- keep the roadmap accurate before executing the selected backlog item

This is intentionally narrow. It is not a full roadmap rewrite phase.

Allowed outcomes:

- `NO_CHANGE`
- `UPDATED`
- `BLOCKED`

Roadmap updates are justified only when the selected backlog item reveals one of:

- a missing prerequisite that changes execution order
- a stale roadmap claim
- a new required work item or tranche
- a now-invalid comparison or experimental direction

Roadmap sync should not rewrite the roadmap just to restate existing priorities or polish prose.

If roadmap sync discovers net-new required executable work that is not already represented by the selected item or an existing backlog item, it must not create the backlog item itself. It should:

- write the roadmap update or gap note
- return `BLOCKED`
- record that a human must add or revise backlog items before the drain can continue

The top-level workflow must maintain an authoritative roadmap pointer under the state root. On `UPDATED`, roadmap sync updates that pointer; on `NO_CHANGE`, the pointer is preserved. Later plan drafting and later selection rounds must consume that pointer, not the original workflow input path.

## Fresh Plan Phase

Proposed workflow:

- `workflows/library/neurips_backlog_seeded_plan_phase.yaml`

Purpose:

- produce an execution-ready plan for the selected backlog item using current steering and roadmap authority

Required rule:

- always draft and review a fresh plan before implementation

Consumed inputs:

- selected backlog item
- `docs/steering.md`
- approved NeurIPS design
- current roadmap
- progress ledger
- selected relevant docs via `docs/index.md`
- prior related execution summaries when present

Important boundary:

- any `plan_path` named in the backlog item is background context only
- the fresh approved plan produced by this phase is authoritative for implementation
- the fresh approved plan must be self-contained enough for implementation and review, restating the selected backlog objective/scope, the relevant roadmap and steering constraints, and the required `check_commands` / verification expectations

## Implementation Phase

Proposed workflow:

- `workflows/library/neurips_backlog_implementation_phase.yaml`

Reuse strategy:

- derive this phase from `workflows/library/design_plan_impl_implementation_phase.yaml`
- do not reuse the generic phase unchanged, because this workflow must preserve the backlog `check_commands` contract and must review against the resulting checks report

Implementation should consume:

- approved fresh self-contained plan
- design
- `check_commands`

Implementation-phase behavior:

1. execute the approved plan
2. run the selected backlog item's `check_commands`
3. write a checks log/report artifact
4. review against design, self-contained plan, execution report, and checks report
5. if revised, fix the implementation and rerun checks before the next review pass

Expected outputs:

- implementation execution report
- checks report / log
- implementation review report
- item summary
- updated backlog state

## State and Artifact Layout

Proposed state root:

- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/`

Suggested structure:

- `manifest.json`
- `current_roadmap_path.txt`
- `selector/`
- `items/<item_id>/`
- `items/<item_id>/roadmap_sync/`
- `items/<item_id>/plan_phase/`
- `items/<item_id>/implementation_phase/`
- `items/<item_id>/implementation_phase/check_commands.json`
- `items/<item_id>/implementation_phase/checks_report.*`
- `run_state.json`

The selector and backlog move helper should also maintain a durable run-state record that captures the owning run id for each `in_progress/` item, the blocker state, and whether an `in_progress/` item is recoverable by a later drain pass.

## Resume and Failure Behavior

- If a selected item fails planning or implementation, keep it in `docs/backlog/in_progress/`.
- Record the blocker in state rather than silently returning it to `active/`.
- Prefer `orchestrator resume <run_id>` when a downstream step fails after earlier approvals.
- Resume-first remains the normal recovery path.
- If the owning run is gone and the blocker predicate later becomes satisfied, a later drain pass may reclaim the `in_progress/` item instead of requiring a manual reset.
- If no item is runnable because every remaining item is blocked by upstream prerequisites, the workflow should exit `BLOCKED`, not churn.

## Repo-Local vs Library Boundary

### Keep repo-local for now

- the top-level steered backlog drain workflow
- the steering contract
- the backlog manifest builder for current PtychoPINN backlog conventions
- the roadmap sync phase
- the backlog-aware implementation phase
- any NeurIPS-specific selection criteria

### Possible later library candidates

- a generic provider-backed backlog selector skeleton
- a generic “drain from backlog manifest” loop shell

Do not extract the roadmap sync phase into the library yet. It is too strategy-specific.

## Implementation Notes for the Future YAML

- model the top-level loop on the existing selector-driven NeurIPS workflow rather than the old deterministic backlog loop
- keep deterministic file moves in command steps, not provider prompts
- keep selector and roadmap-update judgment in prompts, with typed output contracts
- do not let the plan phase assume the backlog item's old `plan_path` is authoritative
- keep steering as consumed context, not a mutable workflow artifact
- preserve the existing backlog `check_commands` contract rather than silently replacing it
- make the fresh approved plan self-contained enough that implementation and review do not need separate provider-time reads of backlog item, roadmap, or steering files

## Recommended Rollout

1. Add `docs/steering.md` and define its stable section contract
2. Add `docs/backlog/in_progress/` and document queue semantics
3. Implement the manifest builder and move helper
4. Implement the selector workflow
5. Implement the narrow roadmap sync phase
6. Implement the fresh plan phase
7. Implement the backlog-aware implementation phase
8. Compose the top-level drain workflow
9. Run the workflow on a small subset of backlog items before trusting full-drain behavior

## Resolved Operational Decisions

- `docs/backlog/in_progress/` is exclusive only while its owning workflow run remains active.
- Recovery of an `in_progress/` item is resume-first; later drain passes may reclaim it when the owner run is gone and the recorded blocker is now satisfied.
- Roadmap sync does not create backlog items directly; if it finds missing required work outside the current queue, it returns `BLOCKED` and records the gap.
- The selector chooses from `docs/backlog/active/` plus recoverable `docs/backlog/in_progress/` items.

## Recommended Decision

Approve this as the design basis for implementation. Then write a concrete implementation plan before authoring YAML, prompts, or helper scripts.
