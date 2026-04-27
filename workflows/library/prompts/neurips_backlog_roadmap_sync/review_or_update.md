Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `steering`, `design`, `roadmap`, `selected_item_context`, and `progress_ledger` artifacts before acting.
Treat `selected_item_context` as the authoritative backlog-item content and queue location; use any `selection_source_path` there only as provenance.

Review the authoritative roadmap in light of the selected backlog item.

This is a single-pass narrow roadmap sync step. Operate on the consumed roadmap file in place. Do not rewrite the roadmap for style, restate unchanged priorities, or broaden the work into a fresh roadmap exercise.

Use `UPDATED` only when the selected backlog item reveals a real roadmap problem such as:
- a missing prerequisite that changes execution order
- a stale roadmap claim
- a now-invalid experimental or comparison direction
- a newly required tranche or phase note that must exist before the selected item can be executed honestly

Use `NO_CHANGE` when the selected item is already consistent with the current roadmap.
Use `BLOCKED` when the selected item exposes a real contradiction or missing authority that cannot be resolved safely in this narrow sync step.

For the output contract's `roadmap_sync_report_path`, read the path recorded in that file and write JSON there using this shape:

```json
{
  "status": "NO_CHANGE",
  "summary": "short summary",
  "changed": false,
  "blocking_reason": null,
  "touched_sections": []
}
```

Also write `NO_CHANGE`, `UPDATED`, or `BLOCKED` to the `roadmap_sync_status` path specified in the output contract.

If status is `UPDATED`, edit the consumed roadmap file in place before writing the report.
