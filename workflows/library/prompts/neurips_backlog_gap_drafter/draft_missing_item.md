You are drafting a missing NeurIPS backlog item for already-authorized roadmap work.

Read the injected project documents and gap request before acting. If `docs/index.md` is present, use it to identify only the specific project docs needed to understand the roadmap gate and current backlog scope.

Your task is narrow:

- Draft exactly one backlog item under the target active backlog directory named in the gap request.
- Draft exactly one seed plan under the plan target root named in the gap request.
- Use only work already authorized by the roadmap gate and gap request.
- Do not edit the roadmap, steering document, progress ledger, run state, existing backlog items, source code, tests, or artifacts.
- Do not advance to CDI, Phase 3, Phase 4, or Phase 5 work.

The backlog item frontmatter must include:

- `priority`
- `plan_path` pointing to the seed plan you wrote
- non-empty `check_commands`
- `related_roadmap_phases` containing an allowed Phase 2 PDEBench phase from the gap request

The seed plan may be concise, but it must identify the objective, scope, non-goals, expected artifacts, and verification commands for the drafted backlog item.

If no safe backlog item can be drafted from the available authority, write a `BLOCKED` draft bundle with a precise reason and do not create placeholder backlog or plan files.

Write the JSON draft bundle to the output contract path:

```json
{
  "draft_status": "DRAFTED",
  "backlog_item_path": "docs/backlog/active/YYYY-MM-DD-pdebench-full-training-evidence-gate.md",
  "seed_plan_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog-gaps/YYYY-MM-DD-pdebench-full-training-evidence-gate.md",
  "summary": "short summary"
}
```

For a blocker:

```json
{
  "draft_status": "BLOCKED",
  "reason": "short reason"
}
```
