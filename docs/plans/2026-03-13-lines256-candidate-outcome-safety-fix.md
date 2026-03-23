## Summary

Narrow the `lines_256` candidate outcome safety check so discard/crash handling preserves preexisting protected edits, cleans candidate edits, and tolerates new unrelated tracked dirt that appears after the session starts.

## Problem

The current `HandleInitialCandidateOutcome` and `HandleDebuggedCandidateOutcome` steps compare the entire tracked-dirty set against the session-start `protected_local_paths` snapshot. That makes the session brittle: if unrelated tracked edits appear later in the repo, a discard or crash recovery fails even when the candidate files were cleaned up correctly.

## Desired Behavior

- Preserve session-start protected tracked edits.
- Reject candidate commits that touch protected paths.
- On `DISCARD` or `CRASH`, restore candidate paths cleanly.
- Tolerate new unrelated tracked dirt outside the candidate path set.
- Keep the same behavior for both the monolithic and encapsulated iteration workflows.

## Plan

1. Add a focused regression test for candidate outcome handling in a temporary git repo.
2. Introduce a small helper script that owns keep/discard/crash outcome handling.
3. Replace duplicated inline Python in both workflow variants with calls to that helper.
4. Update the authoritative study loop doc so it describes the narrower invariant.
5. Run the new targeted tests, existing workflow tests, and an orchestrator dry-run.
6. Resume the failed `lines_256` run instead of relaunching from scratch.

## Verification

- `pytest --collect-only tests/studies/test_lines_256_handle_candidate_outcome.py -q`
- `pytest tests/studies/test_lines_256_handle_candidate_outcome.py -v`
- `pytest tests/studies/test_lines_256_arch_improvement_workflow.py -v`
- `PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml --dry-run --stream-output`
