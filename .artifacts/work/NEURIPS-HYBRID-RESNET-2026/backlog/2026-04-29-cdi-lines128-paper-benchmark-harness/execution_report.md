# Execution Report

## Completed In This Pass

- Replaced the synthetic readiness-only harness path with a real delegated
  compare-wrapper preflight:
  - `scripts/studies/lines128_paper_benchmark.py` now calls
    `run_grid_lines_compare(..., preflight_only=True)` against the fixed
    `lines128` contract, writes
    `validation/readiness_only_preflight/compare_wrapper_preflight.json`, and
    emits a readiness bundle from that delegated row plan instead of mocked
    metrics.
- Added a narrow compare-wrapper preflight mode in
  `scripts/studies/grid_lines_compare_wrapper.py`:
  - validates the custom probe path
  - validates explicit external dataset paths when applicable
  - validates supported Torch rows through
    `grid_lines_torch_runner.setup_torch_configs(...)`
  - preserves TF/Torch backend routing in a machine-readable `row_plan`
  - avoids launching backend training or inference during readiness preflight
- Hardened the harness go/no-go guard:
  - `scripts/studies/lines128_paper_benchmark.py` now rejects decision artifacts
    whose resolved `contract_note.path` does not exist, even if the decision
    JSON incorrectly claims `status: resolved`.
- Completed the paper metric-schema contract in
  `scripts/studies/metrics_tables.py`:
  - `metric_schema.json` now includes `field_definitions`
  - per-metric definitions now carry explicit units and nullability metadata
  - the bundle still downgrades to `benchmark_incomplete` when required fields
    or metric families are absent
- Added regression coverage for:
  - missing resolved contract-note file rejection
  - harness delegation to the compare-wrapper preflight path
  - compare-wrapper `preflight_only` validation without backend execution
  - metric-schema units/nullability metadata

## Completed Current-Scope Work

- The current-scope harness/preflight item is approveable with the review fixes
  applied:
  - the harness is now an execution-ready preflight surface that consumes the
    decision artifact, validates the fixed CDI contract through the existing
    compare-wrapper/runner ownership boundary, and preserves row-level status
    for the later benchmark launch
  - the readiness bundle remains intentionally
    `readiness_only_not_benchmark_performance` and
    `benchmark_incomplete`; that is now due to deliberately missing real
    paper-grade row metrics, not because the harness bypassed the real wrapper
    path
- Refreshed readiness artifacts:
  - readiness bundle root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - delegated wrapper preflight:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/compare_wrapper_preflight.json`
  - harness CLI log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T131846Z_harness_preflight.log`
- Verification from this pass:
  - focused regression selectors:
    `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py`
    -> `58 passed, 23 warnings in 7.91s`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T131846Z_focused_review_fix_pytest.log`
  - required deterministic repo gate:
    `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `164 passed, 43 warnings in 281.91s (0:04:41)`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T131846Z_required_pytest.log`
  - required compile gate:
    `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T131846Z_compileall.log`

## Follow-Up Work

- The later backlog item
  `2026-04-29-cdi-lines128-paper-benchmark-execution` still owns the full
  multi-row paper benchmark launch, fresh per-row artifacts, and any
  `paper_complete` claim.
- Schema versioning remains a genuine future enhancement if later execution
  items need explicit incompatible-bundle detection across harness revisions.

## Residual Risks

- The readiness bundle now proves routing, contract validation, and schema
  emission through the real wrapper path, but it still does not prove benchmark
  performance.
- `metrics.json` intentionally records missing required row fields and metric
  families for every supported row; this is correct for preflight, but later
  benchmark execution must replace those placeholders with real row metadata and
  measured metrics before the merged status can become `paper_complete`.
