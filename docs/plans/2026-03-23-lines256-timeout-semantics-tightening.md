# `lines_256` Timeout Semantics Tightening

## Goal

Treat over-budget candidates as a first-class `TIMEOUT` outcome instead of a fatal workflow failure or a crash-debug target.

## Changes

1. Make the inner experiment launcher timeout slightly shorter than the workflow step timeout so the step can persist timeout artifacts cleanly.
2. Classify timed-out candidate runs as `TIMEOUT` during harvest in both workflow variants.
3. Route `TIMEOUT` through the normal discard/reset path, not the crash-debug path.
4. Update the candidate outcome helper script to treat `TIMEOUT` like `DISCARD`.
5. Update the study loop doc and workflow structure tests to pin the new timeout semantics.

## Verification

1. Run the focused workflow-structure and helper-script tests.
2. Run `pytest --collect-only` for any touched test modules.
3. Run dry-run validation for both workflow variants.
