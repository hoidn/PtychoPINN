# lines_256 Crash Recovery Workflow Plan

## Goal

Make the `lines_256` architecture-improvement workflow robust to candidate crashes by:
- removing the wrapper-only post-check that can fail after an otherwise usable run,
- letting crash harvest outputs omit unavailable metric/path fields, and
- routing `CRASH` outcomes into a debug step instead of killing the whole session.

## Scope

- `scripts/studies/run_lines_256_arch_experiment.py`
- `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- `prompts/workflows/lines_256_arch_improvement/`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `tests/studies/test_lines_256_arch_improvement_workflow.py`

## Planned changes

1. Remove the wrapper-level `ensure_probe_inclusive_comparison_png(...)` hard failure so successful runs with metrics and the standard comparison PNG do not crash during post-processing.
2. Change `HarvestCandidateOutputs` so `CRASH` assessments omit `candidate_amp_ssim`, `delta_amp_ssim`, and `comparison_png_path` instead of writing explicit `null` values that violate the output contract.
3. Restructure the workflow after harvest:
   - `KEEP` and `DISCARD` continue through deterministic ledger/update/reset behavior.
   - `CRASH` writes the ledger row, resets the failed candidate, runs a focused provider debug step, and decides whether to continue or stop based on that step’s output.
4. Add a dedicated crash-debug prompt that stays task-local: inspect the crash log and candidate context, decide whether a clean next bugfix attempt is warranted, and emit structured `READY|BLOCKED` metadata.
5. Update the study loop doc and workflow tests to match the new crash semantics.

## Verification

- Targeted pytest for the workflow module and any new crash-routing behavior.
- `pytest --collect-only` on the touched test module if test names/files change.
- Orchestrator dry-run from the PtychoPINN repo root using the real workflow path and `PYTHONPATH=/home/ollie/Documents/agent-orchestration`.
