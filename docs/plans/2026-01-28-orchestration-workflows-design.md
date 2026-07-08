# Orchestration Workflow Registry Design (Roleless Sequencing)

Date: 2026-01-28
Status: Draft (validated with Ollie)

## Summary
Replace the galph/ralph role distinction with a workflow-driven prompt sequencer. A workflow is a small, extensible registry entry that emits a step list and cadence rules. The orchestration stack resolves the current step by index and runs the associated prompt, making the system compatible with a future YAML DSL without adopting the full DSL now.

## Goals
- Remove role-specific branching (galph vs ralph) while preserving intent.
- Support multiple registered workflows (e.g., standard and review cadence).
- Keep router override functionality.
- Align internal step representation with the future DSL for easy migration.

## Non-goals (for this phase)
- Full YAML DSL parsing/execution.
- Conditional branching, for-each, or complex control flow.
- Provider abstraction beyond existing prompt execution.

## Current Behavior (Intent)
- Standard cadence: supervisor → main repeat.
- Review cadence: every N cycles, use reviewer instead of supervisor+main.
- Sync mode uses state.json for turn-taking; combined mode runs both steps sequentially.

## Proposed Workflow Registry
Add a lightweight registry with minimal metadata:

```python
@dataclass(frozen=True)
class WorkflowStep:
    name: str          # "supervisor", "main", "reviewer" (or any label)
    prompt: str        # "supervisor.md", "main.md", "reviewer.md"

@dataclass(frozen=True)
class Workflow:
    name: str
    steps: list[WorkflowStep]
    cycle_len: int
    review_every_n_cycles: int | None = None
    review_prompt: str | None = None
```

Registry entries:
- `standard`: steps = [supervisor, main], cycle_len=2
- `review_cadence`: steps = [supervisor, main], cycle_len=2, review_every_n_cycles=N, review_prompt=reviewer.md

Resolution logic:
- `base_index = step_index % cycle_len`
- `cycle_index = step_index // cycle_len`
- If `review_every_n_cycles` is set and `cycle_index % review_every_n_cycles == 0`, replace BOTH steps in the cycle with reviewer (so step_index 2 and 3 are reviewer for N=2 with cycle_len=2).

This preserves the required sequence: supervisor, main, reviewer, reviewer, supervisor, main, ...

## State Model Changes
Replace role-specific fields in `sync/state.json` with workflow and step tracking.

New/updated fields:
- `workflow_name`: string
- `step_index`: int (0-based; increments on success)
- `expected_step`: string (optional, for clarity/logging)
- `last_prompt`: string (unchanged, still useful)
- `iteration`: keep as legacy alias for `step_index + 1` for log naming compatibility

Removed:
- `expected_actor`
- `last_prompt_actor`

## Prompt Selection + Router
- Deterministic prompt selection resolves from `(workflow_name, step_index)`.
- Router override is still optional and runs after deterministic selection (same semantics as today) with allowlist checks.
- Actor gating is removed; allowlists are enforced against workflow prompts.

## Sequencing & Modes
- Sync mode: each step execution stamps `status=running` then `status=waiting-next` (success) or `status=failed` (failure). Step index increments only on success.
- Combined mode: run the next step, advance state, then run the next step in the workflow (if configured to do multiple steps per loop).
- Async mode: uses step_index overrides for local runs.

## Logging
- Replace role-based log paths with step-based naming, for example:
  - `logs/<branch>/steps/iter-00012_supervisor.log`
  - `logs/<branch>/steps/iter-00013_main.log`
- Keep log format and content unchanged otherwise.
- Update helper scripts that assume galph/ralph folders (e.g., interleaving) in a follow-up task.

## Compatibility & Migration
- Provide a default workflow name in `orchestration.yaml` (e.g., `workflow: standard`).
- CLI should allow `--workflow` override.
- The registry step list is intentionally close to the future DSL `steps[]` schema; later a YAML loader can compile DSL → the same step list + cadence metadata.

## Testing Strategy
- Update existing tests to assert step-based sequencing instead of role-based.
- Add new tests for:
  - Review cadence cycle replacement (both steps become reviewer).
  - State progression: `step_index` increments on success, unchanged on failure.
  - Router override still works with allowlist validation.
  - Log path format uses step-based naming.

## Open Questions
- Whether to keep `iteration` as a derived field or fully replace it with `step_index`.
- Whether to retain separate log folders per workflow or keep a shared `steps/` directory.
