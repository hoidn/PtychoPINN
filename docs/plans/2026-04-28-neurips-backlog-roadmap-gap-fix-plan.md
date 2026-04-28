# NeurIPS Backlog Roadmap Gap Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `workflows/examples/neurips_steered_backlog_drain.yaml` continue autonomously when the active queue lacks a roadmap-consistent item, without selecting out-of-phase CDI work or requiring a relaunch.

**Architecture:** Add a deterministic roadmap/backlog eligibility gate before provider selection, route missing authorized roadmap work through a controlled backlog-gap drafter, and move selected items to `in_progress` only after roadmap sync has accepted them. Provider judgment ranks eligible work or drafts a missing item; workflow scripts own phase legality, routing, validation, queue movement, and resume-safe state.

**Tech Stack:** Agent-orchestration DSL v2.7, Python helper scripts under `workflows/library/scripts/`, Codex provider workflow steps, Pytest workflow/runtime fixtures, PtychoPINN NeurIPS backlog/docs state.

---

## Documents Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/workflows/neurips_steered_backlog_drain.md`
- `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md`
- `docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-implementation-plan.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/backlog/index.md`
- `/home/ollie/Documents/agent-orchestration/docs/workflow_drafting_guide.md`
- `/home/ollie/Documents/agent-orchestration/specs/dsl.md`
- `/home/ollie/Documents/agent-orchestration/specs/variables.md`
- `/home/ollie/Documents/agent-orchestration/specs/dependencies.md`
- `/home/ollie/Documents/agent-orchestration/specs/providers.md`
- `/home/ollie/Documents/agent-orchestration/workflows/README.md`

## Current Failure Mode

At drain iteration 20, the manifest had no runnable Phase 2 backlog item left, but still had a Phase 3 CDI item. The provider selector chose the CDI item with `roadmap_sync_hint: REVIEW_RECOMMENDED`; the narrow roadmap-sync phase correctly rejected it because the roadmap still required Phase 2 PDEBench evidence before Phase 3 CDI work. The workflow then blocked instead of either selecting a better item or drafting the missing Phase 2 item.

The principled bug is that phase legality lived in selector judgment and post-selection roadmap sync. It must live in deterministic workflow control before selection.

## File Structure

- Modify `docs/backlog/templates/backlog_item_workflow.md`
  - Document workflow-authored gap items and required frontmatter for machine validation.
- Create `docs/backlog/roadmap_gate.json`
  - Human-readable, machine-readable gate policy for this drain: current allowed roadmap phase prefixes, disallowed future prefixes, gap drafting policy, and draft target directories.
- Modify `docs/workflows/neurips_steered_backlog_drain.md`
  - Update lifecycle: build manifest -> reconcile gate -> select eligible item or draft missing item -> sync roadmap -> move to `in_progress`.
- Modify `workflows/examples/neurips_steered_backlog_drain.yaml`
  - Add roadmap gate input/artifact.
  - Add gate reconciliation before selector.
  - Route `ELIGIBLE | BACKLOG_GAP | DONE | BLOCKED`.
  - Loop after gap draft instead of terminating.
- Modify `workflows/library/neurips_backlog_selector.yaml`
  - Accept an already-filtered eligible manifest.
  - Keep output statuses to ranking outcomes, not phase legality decisions.
- Create `workflows/library/neurips_backlog_gap_drafter.yaml`
  - Provider-backed drafter for missing authorized backlog work.
- Create `workflows/library/prompts/neurips_backlog_gap_drafter/draft_missing_item.md`
  - Draft only already-authorized roadmap work; do not edit the roadmap or invent a new science scope.
- Create `workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py`
  - Deterministically filters the manifest by roadmap gate and emits gate status plus eligible manifest/gap request paths.
- Create `workflows/library/scripts/validate_neurips_backlog_gap_draft.py`
  - Validates drafted backlog item and seed plan before the loop continues.
- Modify `workflows/library/neurips_selected_backlog_item.yaml`
  - Delay `active -> in_progress` and current-selection recording until after roadmap sync accepts the item.
  - For active selections rejected by roadmap sync, write a selection-rejection summary and leave the item in `active`.
- Modify `workflows/library/scripts/materialize_neurips_selected_item_inputs.py`
  - Support pre-move selected item context while still producing the intended in-progress path for later phases.
- Modify `tests/studies/test_neurips_steered_backlog_helpers.py`
  - Add helper tests for gate filtering, gap detection, and draft validation.
- Modify `tests/studies/test_neurips_steered_backlog_workflow.py`
  - Add structure tests for gate routing and delayed queue movement.
- Modify `tests/studies/test_neurips_steered_backlog_runtime.py`
  - Add mocked-provider runtime tests for the exact CDI-misselection regression and gap-draft continuation.

## Task 1: Add Red Tests For Roadmap Gate Classification

**Files:**
- Modify: `tests/studies/test_neurips_steered_backlog_helpers.py`
- Test fixtures under: `tests/fixtures/neurips_steered_backlog/`

- [ ] **Step 1: Add fixture gate policy**

Create `tests/fixtures/neurips_steered_backlog/docs/backlog/roadmap_gate.json`:

```json
{
  "gate_version": 1,
  "current_gate_id": "phase-2-pdebench-full-training-evidence",
  "allowed_roadmap_phase_prefixes": ["phase-2-pdebench-"],
  "disallowed_roadmap_phase_prefixes": ["phase-3-", "phase-4-", "phase-5-"],
  "gap_policy": "draft_backlog_item",
  "gap_item_target_dir": "docs/backlog/active",
  "gap_plan_target_root": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps",
  "required_scope_summary": "Remaining Phase 2 PDEBench full-training evidence gate"
}
```

- [ ] **Step 2: Write failing helper tests**

Add tests that call the new script through `_run_script`:

```python
def test_roadmap_gate_filters_out_phase3_cdi_when_phase2_required(tmp_path):
    workspace = _workspace(tmp_path)
    manifest_path = workspace / "state/manifest.json"
    _run_script(workspace, "workflows/library/scripts/build_neurips_backlog_manifest.py",
                "--backlog-root", "docs/backlog/active", "--output", str(manifest_path))

    output_path = workspace / "state/gate.json"
    _run_script(workspace, "workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py",
                "--manifest-path", str(manifest_path),
                "--gate-policy-path", "docs/backlog/roadmap_gate.json",
                "--progress-ledger-path", "state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json",
                "--run-state-path", "state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json",
                "--output", str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert all("phase-3-" not in " ".join(item["related_roadmap_phases"])
               for item in payload["eligible_items"])
```

Also add:

- active manifest contains only Phase 3 CDI plus unrelated no-phase support item -> `gate_status == "BACKLOG_GAP"`
- empty active manifest -> `gate_status == "DONE"`
- malformed gate policy with no allowed prefixes -> non-zero process exit

- [ ] **Step 3: Run the red selector**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py -k 'roadmap_gate' -q
```

Expected: fail because `reconcile_neurips_backlog_roadmap_gate.py` does not exist.

## Task 2: Implement Deterministic Roadmap Gate Reconciliation

**Files:**
- Create: `workflows/library/scripts/reconcile_neurips_backlog_roadmap_gate.py`
- Create: `docs/backlog/roadmap_gate.json`
- Modify: `tests/fixtures/neurips_steered_backlog/docs/backlog/roadmap_gate.json`

- [ ] **Step 1: Implement the script**

The script must:

- load manifest JSON from `build_neurips_backlog_manifest.py`
- load gate policy JSON
- reject unsafe paths and invalid gate policy
- exclude completed or blocked item ids from run state
- classify active items:
  - eligible when any `related_roadmap_phases` value starts with an allowed prefix and no value starts with a disallowed prefix
  - ineligible when it is future-phase, no-phase support work, blocked, completed, or prerequisite-blocked
- emit an output bundle:

```json
{
  "gate_status": "ELIGIBLE",
  "eligible_manifest_path": "state/.../eligible_manifest.json",
  "gap_request_path": "state/.../gap_request.json",
  "eligible_count": 1,
  "ineligible_count": 2,
  "eligible_items": [],
  "ineligible_items": []
}
```

Allowed statuses:

- `ELIGIBLE`
- `BACKLOG_GAP`
- `DONE`
- `BLOCKED`

Use `BACKLOG_GAP` only when active items remain but none are eligible and `gap_policy == "draft_backlog_item"`.

- [ ] **Step 2: Write the eligible manifest and gap request**

Always write:

- `eligible_manifest_path`: same manifest shape as the original, but `items` contains only eligible rows.
- `gap_request_path`: JSON containing `current_gate_id`, `required_scope_summary`, ineligible item summaries, allowed prefixes, disallowed prefixes, target directories, and source manifest path.

- [ ] **Step 3: Add production gate policy**

Create `docs/backlog/roadmap_gate.json` with the same schema as the fixture. The first production policy should keep Phase 2 PDEBench allowed and Phase 3+ disallowed until the roadmap is explicitly advanced.

- [ ] **Step 4: Verify helper tests**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py -k 'roadmap_gate' -q
```

Expected: all roadmap-gate helper tests pass.

## Task 3: Add Controlled Backlog Gap Drafting

**Files:**
- Create: `workflows/library/neurips_backlog_gap_drafter.yaml`
- Create: `workflows/library/prompts/neurips_backlog_gap_drafter/draft_missing_item.md`
- Create: `workflows/library/scripts/validate_neurips_backlog_gap_draft.py`
- Modify: `tests/studies/test_neurips_steered_backlog_helpers.py`

- [ ] **Step 1: Add red validation tests**

Add tests for `validate_neurips_backlog_gap_draft.py`:

- accepts one new backlog item under `docs/backlog/active/` plus an existing seed plan under `docs/plans/`
- rejects a draft under `docs/backlog/in_progress/`
- rejects future-phase `related_roadmap_phases`
- rejects missing `check_commands`
- rejects a `plan_path` whose target does not exist
- rejects attempts to edit the roadmap path

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py -k 'gap_draft' -q
```

Expected: fail because validator does not exist.

- [ ] **Step 2: Implement the validator**

The validator should read:

- `--gap-request-path`
- `--draft-bundle-path`
- `--gate-policy-path`
- `--output`

The draft bundle shape:

```json
{
  "draft_status": "DRAFTED",
  "backlog_item_path": "docs/backlog/active/YYYY-MM-DD-pdebench-full-training-evidence-gate.md",
  "seed_plan_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/YYYY-MM-DD-pdebench-full-training-evidence-gate.md",
  "summary": "..."
}
```

The output bundle shape:

```json
{
  "draft_validation_status": "VALID",
  "backlog_item_path": "...",
  "seed_plan_path": "...",
  "reason": "..."
}
```

- [ ] **Step 3: Implement the drafter workflow**

`workflows/library/neurips_backlog_gap_drafter.yaml` should:

- accept `state_root`, `steering_path`, `design_path`, `roadmap_path`, `progress_ledger_path`, `gate_policy_path`, and `gap_request_path`
- run one provider step to write the backlog item and seed plan
- run `validate_neurips_backlog_gap_draft.py`
- export `draft_status`, `backlog_item_path`, and `seed_plan_path`

- [ ] **Step 4: Write the prompt**

Prompt constraints:

- draft only work already authorized by the gate request and roadmap
- do not edit `docs/steering.md`
- do not edit the roadmap
- write a minimal seed plan because `plan_path` must point at an existing file
- include `related_roadmap_phases` with an allowed Phase 2 PDEBench value
- include fast, targeted `check_commands`
- if no safe draft is possible, emit `BLOCKED` with a precise reason

- [ ] **Step 5: Verify draft tests**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py -k 'gap_draft' -q
```

Expected: pass.

## Task 4: Integrate Gate Routing Into The Top-Level Drain

**Files:**
- Modify: `workflows/examples/neurips_steered_backlog_drain.yaml`
- Modify: `workflows/library/neurips_backlog_selector.yaml`
- Modify: `workflows/library/prompts/neurips_backlog_selector/select_next_item.md`
- Modify: `tests/studies/test_neurips_steered_backlog_workflow.py`

- [ ] **Step 1: Add red workflow-structure tests**

Assert:

- top-level workflow imports `gap_drafter`
- top-level inputs include `roadmap_gate_path`
- `BuildBacklogManifest` is followed by a gate reconciliation step when no current in-progress item is recovered
- selector receives the eligible manifest, not the raw manifest
- route statuses include `ELIGIBLE`, `BACKLOG_GAP`, `DONE`, `BLOCKED`
- `BACKLOG_GAP` calls the drafter and writes drain status `CONTINUE`
- selector prompt does not own phase legality; it ranks eligible candidates

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py -k 'roadmap_gate or gap' -q
```

Expected: fail before YAML changes.

- [ ] **Step 2: Add workflow input and artifact**

In `workflows/examples/neurips_steered_backlog_drain.yaml`, add:

```yaml
roadmap_gate_path:
  type: relpath
  under: docs/backlog
  must_exist_target: true
  default: docs/backlog/roadmap_gate.json
```

Publish it during `InitializeBacklogDrain`.

- [ ] **Step 3: Add `ReconcileBacklogRoadmapGate`**

After `RecoverCurrentInProgressSelection`, run the gate script only when recovery status is `NONE`.

Its output bundle must expose:

- `gate_status`: `ELIGIBLE | BACKLOG_GAP | DONE | BLOCKED`
- `eligible_manifest_path`
- `gap_request_path`

- [ ] **Step 4: Route by gate status**

Use a structured `match` before provider selection:

- `ELIGIBLE`: call selector with `eligible_manifest_path`
- `BACKLOG_GAP`: call gap drafter, then set `drain_status` to `CONTINUE`
- `DONE`: set `drain_status` to `DONE`
- `BLOCKED`: set `drain_status` to `BLOCKED`

Recovered in-progress selections should bypass gate routing and go straight to selected-item execution.

- [ ] **Step 5: Narrow the selector contract**

Update `neurips_backlog_selector.yaml` and prompt:

- manifest input remains named `manifest_path`, but the caller passes the eligible manifest
- prompt says the manifest has already been filtered by deterministic roadmap gate
- selector must not pick outside the manifest
- selector returns `BLOCKED` only for eligible items whose prerequisites or inputs are still not runnable, not for phase legality

- [ ] **Step 6: Verify workflow structure**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py -k 'roadmap_gate or gap' -q
```

Expected: pass.

## Task 5: Delay Queue Movement Until After Roadmap Sync Acceptance

**Files:**
- Modify: `workflows/library/neurips_selected_backlog_item.yaml`
- Modify: `workflows/library/scripts/materialize_neurips_selected_item_inputs.py`
- Modify: `tests/studies/test_neurips_steered_backlog_workflow.py`
- Modify: `tests/studies/test_neurips_steered_backlog_runtime.py`

- [ ] **Step 1: Add red tests for movement order**

Structure test assertions:

- `RunRoadmapSyncPhase` appears before `MoveSelectedItemToInProgress`
- `RecordCurrentSelection` appears after `MoveSelectedItemToInProgress`
- active-selection roadmap rejection routes to `RecordSelectionRejected`, not `RecordRoadmapBlocked`

Runtime test:

- selector selects a Phase 3 CDI item
- roadmap sync emits `BLOCKED`
- item remains under `docs/backlog/active/`
- run state records a selection rejection or workflow block reason, not a completed/blocked in-progress item

- [ ] **Step 2: Reorder selected-item workflow**

New order for `ACTIVE_SELECTION`:

1. `MaterializeSelectedItemInputs`
2. `RunRoadmapSyncPhase`
3. `ApplyRoadmapSyncDecision`
4. `MoveSelectedItemToInProgress`
5. `RecordCurrentSelection`
6. `RunFreshPlanPhase`

For `RECOVERED_IN_PROGRESS`, skip the move but keep `RecordCurrentSelection` compatible with the recovered path.

- [ ] **Step 3: Add selection-rejection recording**

Add `RecordSelectionRejected` for active selections that roadmap sync rejects before movement. It should:

- leave the backlog file in `docs/backlog/active/`
- write an item summary with `item_outcome: "SELECTION_REJECTED"`
- write `drain_status: "BLOCKED"` unless the caller later adds a reselect loop
- avoid adding the item to `blocked_items` as an in-progress semantic blocker

- [ ] **Step 4: Verify movement tests**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_workflow.py -k 'roadmap_sync or in_progress or rejected' -q
pytest tests/studies/test_neurips_steered_backlog_runtime.py -k 'roadmap_rejects_active_selection' -q
```

Expected: pass.

## Task 6: Add End-To-End Runtime Coverage For Gap Draft Continuation

**Files:**
- Modify: `tests/studies/test_neurips_steered_backlog_runtime.py`
- Modify fixture files under `tests/fixtures/neurips_steered_backlog/`

- [ ] **Step 1: Add a mocked-provider test for the original regression**

Fixture setup:

- only Phase 3 CDI and unrelated no-phase support items are active
- gate policy allows only Phase 2 PDEBench
- mocked `DraftMissingBacklogItem` writes a valid Phase 2 backlog item and seed plan
- next loop builds manifest again
- selector then chooses the newly drafted Phase 2 item

Assertions:

- provider selector is not called on the raw Phase 3-only manifest
- CDI item remains active and unmodified
- drafted Phase 2 item exists under `docs/backlog/active/`
- loop continues after draft without relaunch

- [ ] **Step 2: Run runtime test**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_runtime.py -k 'gap_draft_continues' -q
```

Expected: pass.

- [ ] **Step 3: Run all NeurIPS workflow tests**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py tests/studies/test_neurips_steered_backlog_workflow.py tests/studies/test_neurips_steered_backlog_runtime.py -q
```

Expected: pass.

## Task 7: Update Docs And Run Validation

**Files:**
- Modify: `docs/workflows/neurips_steered_backlog_drain.md`
- Modify: `docs/index.md`
- Modify: `docs/backlog/index.md`
- Modify: `docs/backlog/templates/backlog_item_workflow.md`

- [ ] **Step 1: Update docs**

Document:

- roadmap gate file
- eligible-item filtering before selection
- backlog-gap drafting behavior
- delayed `in_progress` movement
- resume-first behavior for already in-progress items
- limitation: drafter may materialize already-authorized roadmap work, but may not edit the roadmap or advance phases

- [ ] **Step 2: Run collect-only for changed tests**

Run:

```bash
pytest --collect-only tests/studies/test_neurips_steered_backlog_helpers.py tests/studies/test_neurips_steered_backlog_workflow.py tests/studies/test_neurips_steered_backlog_runtime.py -q
```

Expected: collection succeeds.

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_helpers.py tests/studies/test_neurips_steered_backlog_workflow.py tests/studies/test_neurips_steered_backlog_runtime.py -q
```

Expected: pass.

- [ ] **Step 4: Run orchestrator dry-run validation**

Run from `/home/ollie/Documents/PtychoPINN`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
export PYTHONPATH=/home/ollie/Documents/agent-orchestration
python -m orchestrator run workflows/examples/neurips_steered_backlog_drain.yaml --dry-run
```

Expected: workflow validates and dry-run exits `0`.

- [ ] **Step 5: Run a mocked runtime smoke**

Run:

```bash
pytest tests/studies/test_neurips_steered_backlog_runtime.py::test_neurips_steered_backlog_runtime_smoke -q
```

Expected: pass.

## Completion Criteria

- The selector never receives future-phase-only raw manifests when the roadmap gate allows only Phase 2.
- When no active eligible item exists and the gate authorizes drafting, the workflow drafts a valid Phase 2 backlog item and loops.
- A roadmap-sync rejection of an active item does not move that item to `in_progress`.
- Existing `WAITING` recovery still works.
- The workflow dry-run validates under `ptycho311` with `PYTHONPATH=/home/ollie/Documents/agent-orchestration`.

## Notes For Implementer

- Do not create a worktree; this repo explicitly forbids it.
- Do not use prompt text as the control gate. The gate script and workflow `match` own legality and routing.
- Keep `roadmap_sync` narrow. It may reject inconsistency, but it is not the first line of defense against phase drift.
- Preserve existing dirty worktree changes. This plan touches workflow files that already have local edits from the active drain work; read current content before patching.
