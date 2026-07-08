# lines_256 Iteration Subworkflow Extraction Plan

## Goal

Draft a cleaner additive variant of the `lines_256` architecture-improvement workflow by extracting one whole experiment iteration into a reusable called subworkflow, while leaving the currently running session workflow unchanged.

## Scope

- `workflows/agent_orchestration/lines_256_arch_improvement_session_loop_v2_call.yaml`
- `workflows/library/lines_256_arch_improvement_iteration.yaml`
- `tests/studies/test_lines_256_arch_improvement_workflow.py`
- `docs/studies/index.md`

## Planned changes

1. Keep the current `lines_256_arch_improvement_session_loop.yaml` in place for the live resumed run.
2. Create a new `v2_call` top-level workflow that owns:
   - study validation
   - protected-local-path capture
   - ledger/session setup
   - baseline run + harvest + ledger append
   - the outer `repeat_until`
3. Move one whole candidate iteration into a library workflow that owns:
   - candidate-context preparation
   - candidate proposal
   - full run + harvest + ledger append
   - keep/discard handling
   - crash debug attempt
   - final per-iteration `loop_decision`
4. Reuse the existing experiment/debug prompts rather than inventing new prompt surfaces.
5. Update workflow tests to assert the new encapsulated topology and run an orchestrator dry-run against the new top-level workflow.

## Verification

- `pytest --collect-only tests/studies/test_lines_256_arch_improvement_workflow.py -q`
- `pytest tests/studies/test_lines_256_arch_improvement_workflow.py -v`
- `PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_arch_improvement_session_loop_v2_call.yaml --dry-run --stream-output`
