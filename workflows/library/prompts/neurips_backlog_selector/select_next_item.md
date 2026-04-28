Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `steering`, `design`, `roadmap`, `manifest`, `progress_ledger`, and `run_state` artifacts before acting.

If `docs/index.md` is present, read it first and then consult only the specific indexed docs that are needed to understand the selected backlog candidates, roadmap gates, or architectural constraints.

Select the next backlog item to draft, review, implement, and verify.

Use the steering document as strategic intent, the roadmap as ordered execution authority, the design as scope authority, the manifest as the active candidate list, the progress ledger as durable project progress, and the run-state ledger as durable backlog-run state.
The manifest has already been filtered by deterministic roadmap-gate checks. Rank only the items in this manifest; do not select work outside it and do not decide broader phase legality here. If an item is still active in the manifest, progress-ledger or summary evidence about related completed work is context or queue drift, not automatic closure.

Choose from active backlog items only. Do not reselect `in_progress` items automatically.

Decision rules:
- Return `DONE` only when the manifest shows no active items.
- Return `BLOCKED` only when active items remain but none can be selected without violating prerequisites, roadmap gates, blocking state, or steering constraints.
- Otherwise return `SELECTED` for exactly one item.
- Do not return `BLOCKED` merely because broader roadmap work remains unfinished or because the roadmap also names non-backlog work. If the current roadmap and steering still allow at least one active backlog item as bounded follow-up or decision-support work, choose among the active items.
- Do not degrade into deterministic first-item order unless the steering, roadmap, prerequisites, and current state genuinely make one item the only defensible choice.
- Treat prerequisite strings as satisfied when they are clearly present in `run_state.completed_items` or `progress_ledger.completed_tranches`.
- If manifest state conflicts with progress-ledger item-completion notes, treat that as state drift to mention in the rationale or `roadmap_sync_hint`, not as a standalone reason to return `BLOCKED`.
- Do not select an item whose required inputs, fairness constraints, or upstream evidence are not actually available.
- Prefer the item that most directly advances the current steering objective without skipping required roadmap order or safety gates.

For the selected item, provide:
- `selected_item_id`
- `selected_item_path`
- `selection_rationale`
- `roadmap_sync_hint` as `NO_CHANGE` or `REVIEW_RECOMMENDED`

Use `REVIEW_RECOMMENDED` only when the selected item likely exposes real roadmap drift such as a missing prerequisite, stale claim, invalid sequence, or a newly required tranche or comparison rule.

For `BLOCKED`, provide concise `blocking_reasons`.

Write the output bundle JSON to the path specified by the output contract using this shape:

```json
{
  "selection_status": "SELECTED",
  "selected_item_id": "2026-04-21-example-item",
  "selected_item_path": "docs/backlog/active/2026-04-21-example-item.md",
  "selection_rationale": "short reason",
  "roadmap_sync_hint": "NO_CHANGE"
}
```

For `DONE`:

```json
{
  "selection_status": "DONE",
  "selection_rationale": "No active backlog items remain."
}
```

For `BLOCKED`:

```json
{
  "selection_status": "BLOCKED",
  "selection_rationale": "No active backlog item is runnable under the current prerequisites and roadmap gates.",
  "blocking_reasons": ["short reason"]
}
```
