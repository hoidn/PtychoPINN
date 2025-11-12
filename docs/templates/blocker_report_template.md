# Blocker Report: [FOCUS-ID]

**Date:** [ISO8601Z timestamp]
**Dwell at escalation:** [N]
**Trigger:** [Tier 2 | Tier 3 | manual]

## Summary
2–3 sentences: what was being attempted, and why it is blocked.

## Attempts History
List iterations and commits where this blocker appeared. Include evidence pointers.

- Iteration NNN (commit XXXXXXXX)
  - Attempt: [what was tried]
  - Result: [minimal error signature or partial success]
  - Evidence: `path/to/log:line`

## Root Cause Analysis
Best current understanding of why this keeps failing.

## Prerequisites
List missing dependencies or infrastructure fixes needed before resuming.

- [ ] [Prerequisite 1]
- [ ] [Prerequisite 2]

## Required Citations
- Do Now (quoted) from `input.md` that failed.
- Exact command(s) and/or pytest selector.
- Hub artifact paths (green/red logs, analysis files) and minimal error signature.

## Recommended Action
What should happen next (manual intervention, scope reduction, dependency focus, etc.).

## Return Condition
Specific condition that would unblock this focus (e.g., "Once FIX-GIT-LOCK-001 is complete and `git status` is clean, retry the Do Now").

## Links
- Hub: <plans/active/.../reports/.../>
- Plan: <plans/active/.../implementation.md>
- Fix Plan Attempt ID: <docs/fix_plan.md#L...>

