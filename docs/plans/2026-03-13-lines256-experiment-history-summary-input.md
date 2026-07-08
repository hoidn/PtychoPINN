## Summary

Add a small, deterministic experiment-history summary input for the `lines_256` architecture-improvement agent so it can avoid repeating recently discarded or crashing ideas without injecting the full raw TSV ledger into the prompt.

## Problem

The current experiment agent reads:

- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `protected_local_paths.json`
- `accepted_state.json`
- `candidate_context.json`

That is enough to know the current accepted reference, but not enough to understand recent failed attempts. The full ledger in `state/lines_256_arch_improvement/results.tsv` has that history, but injecting the raw TSV directly will grow prompt context, mix durable signal with noise, and become harder to maintain as the session grows.

## Goal

Produce a compact, deterministic summary artifact derived from `results.tsv` and inject that summary into the experiment and crash-debug steps.

## Proposed Shape

1. Add a deterministic pre-step that reads `results.tsv` and writes a small JSON summary such as:
   - latest accepted commit + metric
   - recent discarded hypotheses
   - recent crash families
   - maybe the last 10-20 attempts only
2. Inject that summary artifact into:
   - `experiment_step.md`
   - `debug_crash.md`
3. Keep `accepted_state.json` as the source of truth for optimization target.
4. Do not inject the full raw TSV directly unless the summary path proves inadequate.

## Constraints

- Keep the summary bounded and deterministic.
- Prefer stable, machine-generated structure over prose.
- Preserve current workflow behavior when the ledger is empty or very short.
- Apply the same pattern to both the monolithic and encapsulated `lines_256` workflows.

## Verification

- Add targeted tests for the summary-building helper/step.
- Update workflow structure tests to assert the summary artifact is wired into the relevant provider steps.
- Run dry-run validation for both workflow variants.
