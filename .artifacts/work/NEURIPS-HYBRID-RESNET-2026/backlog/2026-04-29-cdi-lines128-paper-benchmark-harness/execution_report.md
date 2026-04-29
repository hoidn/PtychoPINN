# Execution Report

## Completed In This Pass

- Removed the hardcoded harness contract split across code and prose:
  - `scripts/studies/lines128_paper_benchmark.py` now loads the fixed CDI
    contract, seed policy, go/no-go state, and approved-deviation surface from
    `preflight/benchmark_decisions.json`
  - the harness now validates fixed-contract provenance metadata and fails
    closed if the delegated compare-wrapper preflight drifts from the decision
    artifact
- Expanded the delegated preflight contract surface in
  `scripts/studies/grid_lines_compare_wrapper.py` so readiness validation
  returns the full machine-readable contract needed for drift checks.
- Closed the review gap in the durable contract-freeze artifacts:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
    now records the historical sources, per-field confidence, comparator
    rationale, fixed-seed policy, explicit no-deviation state, and go/no-go
    boundary
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
    now carries the full fixed contract, per-field provenance/confidence, seed
    policy rationale, structured comparator decision, and explicit
    `approved_deviations: []`
- Added regression coverage for the review findings:
  - missing fixed-contract metadata
  - seed-policy mismatch against the frozen contract
  - invalid go/no-go state that would authorize the full benchmark
  - delegated compare-wrapper contract drift at readiness time

## Completed Current-Scope Work

- The current-scope harness/preflight item is approveable with the contract
  authority repairs applied:
  - the checked-in preflight note and machine-readable manifest now freeze the
    reconstructed contract fields, historical sources, confidence, comparator
    choice, seed policy, deviation surface, and go/no-go scope
  - the harness consumes that decision surface directly instead of silently
    falling back to hardcoded defaults
  - the readiness preflight still remains intentionally
    `readiness_only_not_benchmark_performance` and `benchmark_incomplete`
- Refreshed readiness artifacts:
  - readiness bundle root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - delegated wrapper preflight:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/compare_wrapper_preflight.json`
  - refreshed harness CLI log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_harness_preflight.log`
- Verification from this pass:
  - focused regression selectors:
    `python -m pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_metrics_tables.py tests/test_grid_lines_compare_wrapper.py`
    -> `62 passed, 23 warnings in 7.94s`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_focused_review_fix_pytest.log`
  - required deterministic repo gate:
    `python -m pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `164 passed, 43 warnings in 281.86s (0:04:41)`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_required_pytest.log`
  - required compile gate:
    `python -m compileall -q ptycho_torch scripts/studies`
    -> exit `0`
    log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_compileall.log`

## Follow-Up Work

- The later backlog item
  `2026-04-29-cdi-lines128-paper-benchmark-execution` still owns the full
  multi-row paper benchmark launch, fresh per-row artifacts, and any
  `paper_complete` claim.
- If a later item intentionally changes the fixed CDI contract or seed policy,
  it should update both the checked-in preflight note and the decision manifest
  together before execution so the harness drift guard remains meaningful.

## Residual Risks

- The readiness bundle now proves manifest-driven routing and delegated
  contract validation, but it still does not prove benchmark performance.
- `metrics.json` intentionally records missing required row fields and metric
  families for every supported row; this remains correct for preflight, but the
  later benchmark execution item must replace those placeholders with real row
  metadata and measured metrics before the merged status can become
  `paper_complete`.
