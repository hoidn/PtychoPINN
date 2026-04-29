# NeurIPS Lines128 Paper Benchmark Harness Summary

- Date: `2026-04-29`
- Backlog item: `2026-04-29-cdi-lines128-paper-benchmark-harness`
- State: `harness_preflight_complete`

## Completed In This Pass

- froze the durable paper-harness decision surface in:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
- corrected the readiness harness entry point:
  - `scripts/studies/lines128_paper_benchmark.py`
  - the harness now consumes the machine-readable fixed contract, seed policy,
    and go/no-go state from `benchmark_decisions.json` instead of relying on a
    hardcoded Python contract
  - the harness now fails closed if the delegated compare-wrapper preflight
    reports contract drift relative to the decision artifact
- extended the compare wrapper / table helpers so the harness can:
  - route `pinn_spectral_resnet_bottleneck_net`
  - preserve an explicit FNO comparator decision
  - delegate readiness validation through `run_grid_lines_compare(..., preflight_only=True)`
    instead of synthesizing mock row metrics
  - validate that the resolved contract note path exists before preflight starts
  - emit the full delegated preflight contract surface needed for drift checks
  - emit paper-schema validation artifacts with `paper_complete` versus
    `benchmark_incomplete` status plus explicit field definitions, units, and
    nullability metadata
- produced a bounded readiness-only validation bundle under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
  - includes delegated wrapper evidence at:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/compare_wrapper_preflight.json`

## Supported Minimum Subset

- `pinn_hybrid_resnet`
- `pinn`
- `pinn_fno_vanilla`

Selected FNO comparator:

- `fno_vanilla`

Seed policy:

- fixed `seed=3`

Additional row states:

- `pinn_spectral_resnet_bottleneck_net`: `supported_for_harness`
- `pinn_ffno`: `supported_for_harness`

## Boundary

The full multi-row paper benchmark was not launched in this pass.

The readiness bundle is intentionally labeled
`readiness_only_not_benchmark_performance`, and its merged `metrics.json`
remains `benchmark_incomplete` because the later execution item still owes:

- fresh multi-row benchmark outputs
- real per-row parameter/runtime/train-history metadata
- fixed-sample paper visuals from actual row artifacts
- full paper-metric payloads instead of the deliberate preflight-only missing
  fields emitted by the readiness bundle

## Artifact Pointers

- decision manifest:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
- readiness bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
- verification logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_focused_review_fix_pytest.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_required_pytest.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/verification/20260429T134542Z_compileall.log`
