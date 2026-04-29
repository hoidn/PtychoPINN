# Execution Report

## Completed In This Pass

- Closed the remaining review-scoped verification gap in
  `tests/studies/test_lines128_paper_benchmark.py`:
  - the readiness-bundle regression now asserts both
    `visual_collation.fixed_sample_ids` and
    `visual_collation.shared_visual_scales`
  - this satisfies the plan’s explicit Task 3 / Task 4 requirement to prove
    fixed-sample visual collation preserves sample identity and shared-scale
    metadata
- Refreshed the checked-in readiness bundle on current `HEAD`
  `e9cf7c70b18fcd57fa7aafbd4daf88255c8c9717`:
  - reran
    `python scripts/studies/lines128_paper_benchmark.py --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - `validation/readiness_only_preflight/invocation.json` now records the
    current git commit instead of the stale earlier SHA
- Refreshed durable summary pointers in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md`
  so the checked-in audit trail points at the latest focused pytest, required
  pytest, compile, and harness-preflight logs

## Completed Current-Scope Work

- The blocking review findings are now fully closed for current scope:
  - the row-status downgrade and delegated preflight coverage fixes remain in
    place from the prior pass
  - the missing shared visual-scale coverage required by the plan is now
    present
  - the persisted readiness provenance is synchronized to the delivered
    revision
- Refreshed readiness artifacts against the current decision artifact:
  - readiness bundle root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - delegated wrapper preflight:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/compare_wrapper_preflight.json`
  - refreshed harness CLI log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T143800Z_harness_preflight.log`
  - the live decision artifact still marks the minimum subset plus
    `spectral_resnet_bottleneck_net` and `pinn_ffno` as
    `supported_for_harness`; the refreshed readiness bundle therefore remains
    `benchmark_incomplete` because row metrics are intentionally absent in
    readiness mode, not because any supported row is currently blocked
- Verification from this pass:
  - focused regression selectors:
    `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py`
    -> `65 passed, 23 warnings in 7.86s`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T143000Z_focused_visual_collation_pytest.log`
  - required deterministic repo gate:
    `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `164 passed, 43 warnings in 282.08s (0:04:42)`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T143200Z_required_pytest.log`
  - required compile gate:
    `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T143800Z_compileall.log`
  - refreshed readiness preflight:
    `python scripts/studies/lines128_paper_benchmark.py --decision-artifact .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
    -> exit `0`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T143800Z_harness_preflight.log`

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
  coverage, required-row status downgrade behavior, and fixed-sample visual
  collation metadata, but it still does not prove benchmark performance.
- `metrics.json` intentionally records missing required row fields and metric
  families for every supported row; the later benchmark execution item must
  replace those placeholders with real row metadata and measured metrics before
  the merged status can become `paper_complete`.
- The required-row status downgrade depends on callers passing the authoritative
  `row_statuses` surface into `write_paper_benchmark_bundle(...)`; the harness
  does this now, and later paper-facing callers must continue to do so.
