# NeurIPS Lines128 Complete Paper Benchmark Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-paper-benchmark-execution`
- State: `paper_complete`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T141325Z_repair`
- Selected FNO comparator: `fno_vanilla`
- Fixed seed policy: `seed=3`
- Fixed sample ids: `0`, `1`

## Completed In This Pass

- extended `scripts/studies/lines128_paper_benchmark.py` with a complete-table execution surface that:
  - freezes the six-row roster under a machine-readable execution manifest
  - promotes contract-complete prerequisite rows into a fresh output root
  - reruns only rows still marked `rerun_required`
  - finalizes merged metrics, manifests, tables, and visuals for the complete paper bundle
- added complete-table coverage in:
  - `tests/studies/test_lines128_paper_benchmark.py`
  - `tests/studies/test_paper_provenance.py`
- emitted deterministic pre-launch audit artifacts:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/execution/row_audit_manifest.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/execution/complete_execution_manifest.json`
- launched the initial complete-table root at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T134500Z`
  - this root completed the fresh `pinn_spectral_resnet_bottleneck_net` rerun and produced the expected bundle artifacts
  - it remained `benchmark_incomplete` because promoted-row recovery still omitted TensorFlow parameter-count source files and FFNO completion-proof recovery
- repaired the promotion/finalization path without another expensive rerun:
  - preserved the TensorFlow legacy model artifacts required for parameter-count recovery
  - normalized promoted completed invocations when backfilled rows lacked explicit `exit_code`
  - added stdout-marker completion evidence support for reused promoted Torch rows
  - overrode promoted-row payloads in the complete-table finalizer with source-root recovery rather than the wrapper’s generic reused-row reconstruction
- published the final accepted repaired root at:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T141325Z_repair`
  - this repaired root was assembled entirely from promoted existing rows, including the spectral row promoted from the finished initial complete-table root
  - `metrics.json`, `model_manifest.json`, and `paper_benchmark_manifest.json` all report `paper_complete` with no missing bundle artifacts and no row-level missing fields

## Accepted Six-Row Roster

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_spectral_resnet_bottleneck_net` -> `Spectral ResNet Bottleneck + PINN`
- `pinn_ffno` -> `FFNO + PINN`

## Row Provenance

- `baseline`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- `pinn`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- `pinn_hybrid_resnet`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- `pinn_fno_vanilla`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- `pinn_spectral_resnet_bottleneck_net`: fresh rerun completed in
  `complete_table_20260430T134500Z`, then promoted into the repaired authoritative root
- `pinn_ffno`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  - row-local invocation provenance remains backfilled-from-wrapper-contract evidence, consistent with the prerequisite repair summary

## Benchmark Status And Boundary

- benchmark status: `paper_complete`
- claim boundary: `complete_lines128_cdi_benchmark`
- evidence scope: one coherent six-row `N=128` CDI paper bundle under the frozen comparator, seed, sample, and shared-visual-scale contract
- preserved prerequisite distinction:
  - the FFNO-vs-Hybrid root remains prerequisite pair evidence
  - the four-row minimum subset root remains preserved minimum-table evidence
  - the repaired complete-table root is the first authoritative six-row paper bundle

## Table And Visual Artifacts

- metrics bundle:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
- visual bundle:
  - `visuals/amp_phase_gt.png`
  - `visuals/compare_amp_phase.png`
  - `visuals/frc_curves.png`
  - per-row `visuals/amp_phase_<row>.png`
  - per-row `visuals/amp_phase_error_<row>.png`

## Verification

- focused execution-surface regression:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py tests/studies/test_paper_provenance.py tests/test_grid_lines_compare_wrapper.py`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_focused_20260430T141325Z.log`
- required deterministic gate:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_required_20260430T141325Z.log`
- compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/compileall_required_20260430T141325Z.log`
- repair and bundle-regeneration logs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_bundle_regen_20260430T070700Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T140954Z.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T141325Z.log`

## Residual Risks

- the authoritative root is the repaired complete-table bundle, not the earlier `complete_table_20260430T134500Z` root. Consumers should use only `complete_table_20260430T141325Z_repair` for paper-facing tables and figures.
- FFNO row provenance still depends on backfilled row invocation records from the prerequisite repair pass, but the repaired complete-table root now reattaches the row-local completion proof required by the current paper bundle contract.
