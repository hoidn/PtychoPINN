You are the post-approval commit step for one completed backlog cycle.

Read these inputs first:
- state/plan_path.txt and referenced plan
- state/backlog_item_path.txt
- state/done_backlog_item_path.txt
- state/pre_run_head.txt
- state/pre_run_status.txt
- state/pre_run_tracked_diff.txt
- state/pre_run_index_diff.txt

Required actions:
1. Inspect current git status/diff and stage files using engineering judgment.
2. Create exactly one non-interactive commit for this approved cycle.
3. Keep staging focused on implementation and approved queue transition.
4. Write commit SHA to `state/commit_sha.txt` (full SHA preferred).
5. Write exactly this relative path to `state/commit_sha_path.txt`:
   state/commit_sha.txt

Staging guidance:
- Prefer including changes that implement the selected plan and the workflow-produced backlog move.
- Exclude obvious runtime/temp artifacts unless explicitly required by the plan:
  - `.orchestrate/`
  - `state/`
  - `logs/`
  - transient caches/temp directories
- Do not use interactive git flows.
- Do not amend commits.

Constraints:
- If no meaningful commit can be made, exit non-zero with a short reason.
- Do not write absolute paths.
