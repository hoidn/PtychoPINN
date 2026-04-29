# Execution Report

## Completed In This Pass

- Fixed the bundle-completion contract bug in
  `scripts/studies/metrics_tables.py`:
  - `write_paper_benchmark_bundle(...)` now downgrades to
    `benchmark_incomplete` when a required row is present in the bundle but its
    supplied `row_statuses` entry is anything other than
    `supported_for_harness`
  - the helper preserves its older generic behavior when no `row_statuses`
    surface is supplied, so existing non-harness call sites do not regress
- Fixed the delegated-preflight fail-open bug in
  `scripts/studies/lines128_paper_benchmark.py`:
  - the harness now validates that delegated compare-wrapper
    `selected_models` exactly match the decision-artifact
    `supported_for_harness` rows
  - the harness now validates that delegated `row_plan` coverage and per-row
    statuses also exactly match that supported-row surface before readiness
    artifacts are written
- Added focused regressions for the two blocking review findings:
  - required-row `row_blocker` state forcing `benchmark_incomplete`
  - delegated `selected_models` omission rejection
  - delegated `row_plan` omission rejection

## Completed Current-Scope Work

- The blocking implementation-review defects are closed:
  - the harness can no longer publish `paper_complete` when a required row has
    already failed at the row-status layer
  - the readiness path now fails closed if delegated compare-wrapper routing
    silently drops or remaps any supported decision-artifact row
- Refreshed readiness artifacts against the current decision artifact:
  - readiness bundle root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - delegated wrapper preflight:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/compare_wrapper_preflight.json`
  - refreshed harness CLI log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T140612Z_harness_preflight.log`
  - the live decision artifact currently marks the minimum subset plus
    `spectral_resnet_bottleneck_net` and `pinn_ffno` as
    `supported_for_harness`; the refreshed readiness bundle therefore remains
    `benchmark_incomplete` because row metrics are still intentionally absent,
    not because any row is currently blocked
- Verification from this pass:
  - focused regression selectors:
    `python -m pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py`
    -> `65 passed, 23 warnings in 7.96s`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T140612Z_focused_review_fix_pytest.log`
  - required deterministic repo gate:
    `python -m pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `164 passed, 43 warnings in 281.84s (0:04:41)`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T140612Z_required_pytest.log`
  - required compile gate:
    `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T140612Z_compileall.log`

## Follow-Up Work

- The later backlog item
  `2026-04-29-cdi-lines128-paper-benchmark-execution` still owns the full
  multi-row paper benchmark launch, fresh per-row artifacts, and any
  `paper_complete` claim.
- If a later item changes the decision-artifact supported-row surface,
  comparator choice, or fixed CDI contract, it must rerun the readiness
  preflight so the exact delegated row coverage remains proven against the
  updated manifest.

## Residual Risks

- The readiness bundle now proves manifest-driven routing, delegated row-plan
  coverage, and required-row status downgrade behavior, but it still does not
  prove benchmark performance.
- `metrics.json` intentionally records missing required row fields and metric
  families for every supported row; the later benchmark execution item must
  replace those placeholders with real row metadata and measured metrics before
  the merged status can become `paper_complete`.
- The required-row status downgrade depends on callers passing the authoritative
  `row_statuses` surface into `write_paper_benchmark_bundle(...)`; the harness
  does this now, and later paper-facing callers must continue to do so.
