# NeurIPS Steered Backlog Drain Workflow Implementation Plan

## Initiative
- ID: `NEURIPS-STEERED-BACKLOG-DRAIN-001`
- Title: NeurIPS steered backlog drain workflow
- Status: pending
- Owner: Codex session draft for user review
- Spec/Source: `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`

## Compliance Matrix (Mandatory)
> List the specific Spec constraints, Fix-Plan ledger rows, and Findings/Policies this initiative must honor.
- [ ] **Spec Constraint:** `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md` — workflow must read `docs/steering.md`, select backlog items adaptively, move items into `docs/backlog/in_progress/`, optionally sync the roadmap, always draft/review a fresh plan, preserve backlog `check_commands`, then implement, check, review, and repeat.
- [ ] **Spec Constraint:** `docs/INITIATIVE_WORKFLOW_GUIDE.md` — backlog items are candidate work units, plans are authoritative execution artifacts, and workflow outputs must stay discoverable and stateful.
- [ ] **Spec Constraint:** `docs/backlog/templates/backlog_item_workflow.md` — backlog items keep frontmatter under `docs/backlog/`; any new selection fields must remain parseable and backwards-compatible.
- [ ] **Finding/Policy ID:** `CLAUDE.md` and `docs/findings.md` — read `docs/index.md` first, consult findings before workflow changes, keep changes scoped, and verify with targeted checks plus at least one real orchestrator smoke.

## Spec Alignment
- **Normative Spec:** `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`
- **Key Clauses:**
  - `docs/steering.md` is human-authored strategic intent and is workflow-consumed, not workflow-authored.
  - selection is provider-backed and must not degrade into deterministic manifest order unless evidence makes the choice obvious.
  - `docs/backlog/in_progress/` becomes a first-class queue state.
  - roadmap sync is narrow and may revise roadmap text/order, but should not perform a fresh roadmap rewrite every cycle.
  - every selected backlog item gets a fresh reviewed plan before implementation, even if it already names a `plan_path`.
  - the fresh approved plan must be self-contained enough for implementation and review.
  - backlog `check_commands` remain part of the authoritative execution contract.

## Architecture / Interfaces
- Components/boundaries touched:
  - backlog queue contract under `docs/backlog/`
  - NeurIPS strategy authority in `docs/steering.md`
  - deterministic helper scripts under `workflows/library/scripts/`
  - provider-backed workflow phases under `workflows/library/`
  - top-level drain workflow under `workflows/examples/`
- Primary data flow:
  - build typed manifest from `docs/backlog/active/` -> select next item from steering/roadmap/state -> move item to `in_progress/` -> optionally revise roadmap -> draft/review fresh self-contained plan -> update backlog `plan_path` -> implement -> run `check_commands` -> review/fix -> move item to `done/` or leave blocked in `in_progress/`.
- External interfaces/contracts impacted:
  - backlog item frontmatter
  - roadmap markdown path
  - authoritative roadmap pointer under state
  - state root under `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/`
  - workflow prompts and output bundles

## Planned File Map

### Create
- `docs/steering.md`
- `docs/backlog/in_progress/.gitkeep`
- `docs/workflows/neurips_steered_backlog_drain.md`
- `workflows/examples/neurips_steered_backlog_drain.yaml`
- `workflows/library/neurips_backlog_selector.yaml`
- `workflows/library/neurips_backlog_roadmap_sync_phase.yaml`
- `workflows/library/neurips_backlog_seeded_plan_phase.yaml`
- `workflows/library/neurips_backlog_implementation_phase.yaml`
- `workflows/library/scripts/build_neurips_backlog_manifest.py`
- `workflows/library/scripts/materialize_neurips_selected_item_inputs.py`
- `workflows/library/scripts/move_neurips_backlog_item.py`
- `workflows/library/scripts/update_neurips_backlog_run_state.py`
- `workflows/library/scripts/run_neurips_backlog_checks.py`
- `workflows/library/prompts/neurips_backlog_selector/select_next_item.md`
- `workflows/library/prompts/neurips_backlog_roadmap_sync/review_or_update.md`
- `workflows/library/prompts/neurips_backlog_seeded_plan_phase/draft_plan.md`
- `workflows/library/prompts/neurips_backlog_seeded_plan_phase/review_plan.md`
- `workflows/library/prompts/neurips_backlog_seeded_plan_phase/revise_plan.md`
- `workflows/library/prompts/neurips_backlog_implementation_phase/review_implementation.md`
- `workflows/library/prompts/neurips_backlog_implementation_phase/fix_implementation.md`
- `tests/studies/test_neurips_steered_backlog_helpers.py`
- `tests/studies/test_neurips_steered_backlog_workflow.py`
- `tests/studies/test_neurips_steered_backlog_runtime.py`
- `tests/fixtures/neurips_steered_backlog/steering.md`
- `tests/fixtures/neurips_steered_backlog/design.md`
- `tests/fixtures/neurips_steered_backlog/roadmap.md`
- `tests/fixtures/neurips_steered_backlog/progress_ledger.json`
- `tests/fixtures/neurips_steered_backlog/backlog/active/*.md`

### Modify
- `docs/index.md`
- `docs/backlog/templates/backlog_item_workflow.md`
- `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml` only if a small shared helper extraction is necessary; otherwise leave it untouched

### Reuse As Reference / Base
- `workflows/library/design_plan_impl_implementation_phase.yaml` as the base loop shape for the repo-local backlog-aware implementation phase, not as a drop-in interface
- `workflows/library/roadmap_tranche_selector.yaml` as the control-flow reference
- `workflows/library/roadmap_seeded_plan_phase.yaml` as the fresh-plan wrapper reference
- `workflows/library/tracked_plan_phase.yaml` if the seeded plan phase is implemented as a thin adapter

## Context Priming (read before edits)
- Primary docs/specs to re-read:
  - `docs/index.md`
  - `docs/findings.md`
  - `docs/INITIATIVE_WORKFLOW_GUIDE.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`
- Required findings/case law:
  - queue/workflow state should be explicit and resumable
  - avoid deterministic priority shortcuts when the selection is supposed to be judgment-driven
  - do not encode brittle prompt-phrase assertions in tests
- Related telemetry/attempts:
  - `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`
  - legacy deterministic backlog loop docs under `docs/workflows/agent_orchestration_backlog_loop.md`
- Data dependencies to verify:
  - actual backlog markdown structure in `docs/backlog/active/`
  - existing roadmap path and progress ledger path
  - presence and shape of steering doc before smoke runs

## Implementation Decisions Locked By This Plan

- `docs/backlog/in_progress/` is exclusive while its owning run remains active. When the owner run is gone and the recorded blocker is cleared, a later drain pass may reclaim the item.
- roadmap sync may revise roadmap prose, ordering, and phase notes, but must not create new backlog items directly. If it discovers required executable work that is missing from the queue, it must emit `BLOCKED` plus a gap artifact for human backlog maintenance.
- the top-level workflow must maintain `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/current_roadmap_path.txt` as the single authoritative roadmap pointer after each roadmap-sync decision.
- after fresh plan approval, the selected backlog item's `plan_path` is rewritten to the fresh approved plan before implementation begins.
- the fresh approved plan must carry forward the selected backlog objective/scope, relevant roadmap and steering constraints, and the authoritative `check_commands` / verification expectations so the implementation phase can stay narrow.
- backlog `check_commands` are preserved and must run before each implementation review pass.
- blocked or failed items remain in `docs/backlog/in_progress/` with durable blocker state. Recovery is resume-first; later drain passes may reclaim the item when the owner run is gone and the recorded blocker is cleared.

## Phases

### Phase A — Authority and Queue Contract
- [ ] Create `docs/steering.md` with stable sections:
  - `# Current Objective`
  - `## Ordered Near-Term Priorities`
  - `## Required Comparison Standards`
  - `## Fairness / Apples-to-Apples Constraints`
  - `## Known Blockers / Deferred Work`
  - `## Non-Goals / Not Now`
  - `## Initiative Guidance`
- [ ] Add `docs/backlog/in_progress/.gitkeep`.
- [ ] Update `docs/backlog/templates/backlog_item_workflow.md` to document:
  - `in_progress` queue state
  - optional selector-facing frontmatter such as `prerequisites` and `related_roadmap_phases`
  - `check_commands` remain required and are run by the new workflow before review
  - rule that `plan_path` may be rewritten by the workflow after plan approval
- [ ] Add a short operator doc at `docs/workflows/neurips_steered_backlog_drain.md` with:
  - what the workflow consumes
  - the queue lifecycle
  - the run command
  - the resume command
- [ ] Update `docs/index.md` to index the new steering doc, plan, and workflow runbook.

### Phase B — Deterministic Backlog Helpers
- [ ] Implement `workflows/library/scripts/build_neurips_backlog_manifest.py`.
  - Inputs: backlog root, roadmap path, optional steering path.
  - Output: typed JSON manifest of active items.
  - Must reject malformed frontmatter with useful errors.
  - Must preserve `check_commands` and remain backwards-compatible with existing backlog items that only have `priority`, `plan_path`, and `check_commands`.
- [ ] Implement `workflows/library/scripts/materialize_neurips_selected_item_inputs.py`.
  - Input: selector output bundle plus the typed manifest.
  - Output: a deterministic selected-item bundle under the item state root, including:
    - selected backlog item path
    - `check_commands.json`
    - `check_commands_path.txt`
    - queue transition hint (`ACTIVE_SELECTION` vs `RECOVERED_IN_PROGRESS`)
  - This step is the authoritative handoff from selection into implementation inputs.
- [ ] Implement `workflows/library/scripts/move_neurips_backlog_item.py`.
  - Supported transitions: `active -> in_progress`, `in_progress -> done`, reclaim/refresh ownership for recoverable `in_progress`, and explicit reset back to `active` only when requested.
  - Must reject illegal transitions and unsafe paths.
- [ ] Implement `workflows/library/scripts/update_neurips_backlog_run_state.py`.
  - Records current selection, blocker state, recoverability, run id, selected plan path, authoritative roadmap path, and final item outcome under `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/`.
  - Provides a stable ledger for selector exclusion and operator recovery.
- [ ] Add focused helper tests in `tests/studies/test_neurips_steered_backlog_helpers.py`.
  - valid manifest build
  - malformed frontmatter rejection
  - `check_commands` preservation
  - selected-item materialization writes `check_commands_path`
  - prerequisite field parsing
  - legal/illegal queue transitions
  - blocker-state persistence

### Phase C — Backlog Selector
- [ ] Implement `workflows/library/neurips_backlog_selector.yaml`.
  - Reuse `roadmap_tranche_selector.yaml` shape where helpful.
  - Consume steering, design, the authoritative roadmap pointer, manifest, progress ledger, and run-state ledger.
  - Emit:
    - `selection_status`
    - `selected_item_id`
    - `selected_item_path`
    - `selected_check_commands`
    - `selection_rationale`
    - `roadmap_sync_hint`
    - `blocking_reasons`
    - `selection_mode` (`ACTIVE_SELECTION` or `RECOVERED_IN_PROGRESS`)
- [ ] Write prompt `workflows/library/prompts/neurips_backlog_selector/select_next_item.md`.
  - Must instruct the selector to read `docs/index.md` first when architectural/project docs matter.
  - Must explicitly avoid deterministic first-item behavior.
  - Must respect prerequisites and in-flight blockers.
- [ ] Add workflow-structure tests in `tests/studies/test_neurips_steered_backlog_workflow.py`.
  - selector consumes steering + authoritative roadmap pointer + progress ledger + manifest
  - selector output contract fields are typed and routed
  - selector output carries the selected `check_commands`
  - no tests may assert literal prompt wording

### Phase D — Narrow Roadmap Sync Phase
- [ ] Implement `workflows/library/neurips_backlog_roadmap_sync_phase.yaml`.
  - Consume selected backlog item, steering, design, roadmap, and progress ledger.
  - Use one provider-backed review/update step plus deterministic validation.
  - Emit:
    - `NO_CHANGE | UPDATED | BLOCKED`
    - `roadmap_path`
  - On `NO_CHANGE`, it must republish the incoming authoritative roadmap path.
  - On `UPDATED`, it must publish the new authoritative roadmap path.
- [ ] Write prompt `workflows/library/prompts/neurips_backlog_roadmap_sync/review_or_update.md`.
  - It should update the roadmap only when the selected item reveals a missing prerequisite, stale claim, invalid sequence, or now-required phase/tranche.
  - It should not use the selected item as an excuse to restyle the roadmap.
- [ ] Add tests that assert:
  - roadmap sync can pass through `NO_CHANGE`
  - roadmap sync republishes the incoming roadmap path on `NO_CHANGE`
  - roadmap sync can publish an updated roadmap path on `UPDATED`
  - net-new required executable work without a corresponding backlog item yields `BLOCKED` plus a gap artifact
  - blocked outcomes route cleanly

### Phase E — Fresh Plan Phase
- [ ] Implement `workflows/library/neurips_backlog_seeded_plan_phase.yaml`.
  - Prefer adapting `roadmap_seeded_plan_phase.yaml` or wrapping `tracked_plan_phase.yaml`.
  - Inputs must include steering, selected backlog item, design, the authoritative roadmap path, progress ledger, and optional prior item summaries.
  - The selected backlog item's previous `plan_path` is consumed as background context only.
  - The approved plan it writes must be self-contained enough for implementation and review, including the selected backlog scope, relevant roadmap/steering constraints, and the authoritative verification/check contract.
- [ ] Write local plan prompts:
  - `workflows/library/prompts/neurips_backlog_seeded_plan_phase/draft_plan.md`
  - `workflows/library/prompts/neurips_backlog_seeded_plan_phase/review_plan.md`
  - `workflows/library/prompts/neurips_backlog_seeded_plan_phase/revise_plan.md`
- [ ] Add deterministic command step(s) that rewrite the selected backlog item's `plan_path` to the newly approved plan and persist that change before implementation begins.
- [ ] Add tests that assert:
  - plan phase consumes steering + selected item + authoritative roadmap path
  - approved plan path is published
  - approved plan carries forward the verification/check contract
  - no later implementation provider step needs direct backlog-item, steering, or roadmap prompt consumption because the approved plan is self-contained
  - backlog item `plan_path` rewrite occurs before the implementation call

### Phase F — Backlog-Aware Implementation Phase
- [ ] Implement `workflows/library/neurips_backlog_implementation_phase.yaml`.
  - Derive its control-flow structure from `workflows/library/design_plan_impl_implementation_phase.yaml`.
  - This is a repo-local phase, not a semantic no-op wrapper around the generic implementation phase.
  - Inputs should include:
    - `design_path`
    - `plan_path`
    - `check_commands_path`
    - report target paths
  - It must run:
    1. implementation execution
    2. deterministic `check_commands`
    3. implementation review against design + plan + execution report + checks report
    4. fix pass and re-check on `REVISE`
- [ ] Implement `workflows/library/scripts/run_neurips_backlog_checks.py`.
  - Reads `check_commands` as a JSON list.
  - Executes them from the repo root.
  - Writes a durable checks log/report artifact and a machine-readable pass/fail outcome.
- [ ] Write repo-local prompts:
  - `workflows/library/prompts/neurips_backlog_implementation_phase/review_implementation.md`
  - `workflows/library/prompts/neurips_backlog_implementation_phase/fix_implementation.md`
- [ ] Add tests that assert:
  - the implementation phase publishes a checks report artifact
  - the implementation phase consumes `check_commands_path`
  - review consumes the checks report
  - failed checks are visible to the review/fix loop rather than bypassed

### Phase G — Top-Level Drain Workflow
- [ ] Implement `workflows/examples/neurips_steered_backlog_drain.yaml`.
  - Initialize state root.
  - Initialize and publish `current_roadmap_path.txt` from the workflow input roadmap path.
  - Build manifest.
  - Select next item.
  - Exit `DONE` when no eligible items remain.
  - Exit `BLOCKED` when active items remain but none are runnable.
  - Materialize selected item inputs, including `check_commands_path`.
  - Move selected item to `in_progress/` or reclaim a recoverable `in_progress/` item based on `selection_mode`.
  - Run roadmap sync.
  - Update `current_roadmap_path.txt` from the roadmap-sync outputs before any later plan drafting or selector iteration.
  - Run fresh plan phase.
  - Run `neurips_backlog_implementation_phase.yaml`.
  - Move item to `done/` on approval; otherwise leave it in `in_progress/` and record blocker state.
  - Repeat.
- [ ] Keep deterministic path manipulations in command steps, not provider prompts.
- [ ] Update the workflow runbook with a concrete launch command from the PtychoPINN repo root using `ptycho311`.

### Phase H — Verification and Smoke Coverage
- [ ] Add fixture data under `tests/fixtures/neurips_steered_backlog/` for a tiny backlog with:
  - one immediately runnable item
  - one blocked item that depends on a future prerequisite
- [ ] Add or extend `tests/studies/test_neurips_steered_backlog_workflow.py` to validate:
  - workflow topology
  - selector contracts
  - roadmap-sync routing
  - authoritative roadmap pointer update routing
  - plan rewrite contract
  - check-command preservation and routing
  - done/blocked terminal behavior
- [ ] Add one runtime smoke test modeled on the `WorkflowExecutor` / `ProviderExecutor` patch tests used in `agent-orchestration/tests/test_major_project_workflows.py`.
  - Run the new workflow in a temporary workspace copied from the fixture tree.
  - Patch provider execution to return deterministic selector, roadmap-sync, plan-review, and implementation-review outputs.
  - Prove that the workflow performs manifest build, item move, roadmap sync, plan publication, `check_commands` wiring, implementation call wiring, and root output export.
- [ ] Run one orchestrator CLI smoke from the PtychoPINN repo root after implementation.
  - Dry-run is acceptable for schema/control-flow proof.
  - The patched runtime smoke above is required for post-execution contract proof.

## Dependency Analysis
- Touched modules:
  - workflow YAML in `workflows/`
  - deterministic Python helpers in `workflows/library/scripts/`
  - backlog docs and queue layout in `docs/backlog/`
  - tests under `tests/studies/`
- Circular import risks:
  - keep helper scripts standalone and argparse-based; do not create a new shared Python package unless the helpers naturally demand it.
- State migration:
  - no migration of existing run state is required
- backlog queue semantics extend existing directories by adding `in_progress/`
- existing backlog items must remain readable even before they are backfilled with optional selector fields

## Workflow Compatibility Contract

When this plan is executed by a backlog workflow:

  - selected backlog item must end the plan phase with `plan_path` pointing at the freshly approved plan
- selected backlog item must retain authoritative `check_commands`, and those commands must run before each implementation review pass
- the authoritative roadmap for each cycle is whatever `current_roadmap_path.txt` points to after roadmap sync
- execution unit is the full selected backlog item, not an ad hoc slice
- blocked items must remain in `docs/backlog/in_progress/` with durable state
- completion requires:
  - helper tests pass
  - workflow structure tests pass
  - runtime smoke passes
  - orchestrator CLI dry-run passes

## Verification Commands
Run from `/home/ollie/Documents/PtychoPINN` in the `ptycho311` environment.

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py -q
pytest tests/studies/test_neurips_steered_backlog_workflow.py -q
pytest tests/studies/test_neurips_steered_backlog_workflow.py --collect-only -q
pytest tests/studies/test_neurips_steered_backlog_runtime.py -q
PYTHONPATH=/home/ollie/Documents/agent-orchestration pytest tests/studies/test_neurips_steered_backlog_workflow.py -q
PYTHONPATH=/home/ollie/Documents/agent-orchestration pytest tests/studies/test_neurips_steered_backlog_runtime.py -q
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/examples/neurips_steered_backlog_drain.yaml --dry-run --stream-output
```

## Completion Criteria
- [ ] `docs/steering.md`, `docs/backlog/in_progress/`, and the backlog template document the new queue/authority contract clearly enough that a human can prepare inputs without reading code.
- [ ] The new helper scripts deterministically build a manifest, move items through legal queue transitions, and persist blocker/run state.
- [ ] The selector, roadmap-sync phase, fresh-plan phase, backlog-aware implementation phase, and top-level drain workflow exist and wire together with typed outputs and path-safe artifacts.
- [ ] A selected backlog item gets a newly approved plan written into its `plan_path` before implementation starts.
- [ ] Backlog `check_commands` are preserved, executed, and surfaced to review/fix rather than silently dropped.
- [ ] Blocked items remain in `docs/backlog/in_progress/` and are not silently reselected.
- [ ] Focused workflow tests, a patched runtime smoke, and an orchestrator CLI dry-run all pass.
- [ ] The workflow and operator docs are indexed from `docs/index.md`.

## Artifacts Index
- Reports root: `docs/plans/NEURIPS-STEERED-BACKLOG-DRAIN-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
