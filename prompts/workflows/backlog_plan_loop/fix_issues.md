Re-run the selected backlog item's full plan after a `REVISE` decision.

Use the `Consumed Artifacts` section as the authoritative input list, and read those files before acting.

Do:
1. Execute the full plan again, incorporating required fixes from the code review report.
2. Use the consumed check log to prioritize concrete failing checks first.
3. If the consumed plan-level review requires plan/backlog contract edits, apply those edits first and create a dedicated non-interactive commit for the plan revision before continuing execution.
   - Commit subject pattern: `plan-revision(<backlog-item-stem>): <short reason>`
   - Example: `plan-revision(2026-02-26-hybrid-resnet-skip-mode-search-stage-d-execution): clarify N=256 baseline coverage contract`
   - Include only the plan/backlog revision files in that commit.
4. Keep changes focused on plan completion.
5. Do not fabricate or backfill run results.
6. Read destination path from `state/execution_session_log_path.txt`, then write the execution session log to that path, with:
   - plan path used
   - fixes implemented
   - commands executed and outcomes
   - plan revision commit(s), if created (subject + SHA)
   - files changed
   - blockers or follow-ups

Constraints:
- No unrelated refactors.
- No queue movement.
- No absolute paths.
- Do not modify `state/execution_session_log_path.txt` in this step.
- Do not create or use git worktrees; execute in the current checked-out workspace and write artifacts in that same workspace.

<long-running-work-policy>
  This policy applies to any required long-running process in the selected plan.

  1. Blocking semantics:
     - Do not mark the step complete when only launch succeeded.
     - A long-running process is complete only after an explicit terminal state is observed.
  2. Tracking requirements:
     - Record PID, start time (UTC), status transitions, and final exit code.
     - Persist this evidence in a machine-readable status artifact (not only terminal output).
  3. Polling requirements:
     - Poll status evidence at ~30s cadence (or bounded backoff for very long runs) until terminal state or timeout.
     - Do not rely on pane/capture text as the sole completion signal.
  4. Completion gate (all required):
     - required process(es) exited,
     - final exit code is 0,
     - required completion marker(s) exist,
     - required output artifacts exist and are fresh for this run,
     - required verification checks pass.
  5. Failure handling:
     - If any completion gate fails or evidence is missing, do not claim completion.
     - Mark the step as failed using the status/decision mechanism defined by this step's output contract.
     - State the exact failed gate and include the evidence path.

  Default timeout: 12 hours unless the plan defines a different limit.
</long-running-work-policy>
