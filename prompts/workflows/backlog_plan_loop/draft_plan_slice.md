You are drafting the next executable plan slice for a queued backlog item.

Read these inputs first:
- state/backlog_item_path.txt
- the backlog item file referenced by that pointer
- state/plan_path.txt
- the plan file referenced by that pointer
- state/review_cycle.txt

Your job:
1. Identify the next executable task slice in the plan using the workflow contract format:
   - heading format `### Task <ID>: <Title>`
   - status line `**Status:** pending | in_progress | blocked | done`
2. Prefer the first task whose status is `pending` or `in_progress`.
3. Produce a compact execution brief with:
   - selected task id and title
   - required files
   - implementation steps
   - check command expectations
   - expected evidence targets
4. Write that brief to `artifacts/work/current-slice-brief.md`.
5. Write exactly this relative path to `state/slice_brief_path.txt`:
   artifacts/work/current-slice-brief.md

If all tasks are already done, still write a brief that says no remaining executable task and recommends review approval if checks are green.

Constraints:
- Do not modify source code in this step.
- Keep output deterministic and concise.
- Do not write absolute paths.
