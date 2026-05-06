# NeurIPS Lines128 Complete Paper Benchmark Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-paper-benchmark-execution`
- State: `paper_complete`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Selected FNO comparator: `fno_vanilla`
- Fixed seed policy: `seed=3`
- Fixed sample ids: `0`, `1`

## Completed In This Pass

- Repaired promoted-row completion-proof recovery so complete-table finalization can rebuild row-local `launcher_completion.json` from durable promoted-source wrapper logs, including `live_*.log` and `tmux.log` fallback when source roots do not retain `launcher_stdout.log` / `launcher_stderr.log`.
- Tightened the complete-table validation contract so recovered PyTorch rows without launcher completion proof force `benchmark_incomplete` rather than silently leaving the merged bundle `paper_complete`.
- Added mixed-source complete-table regressions covering:
  - successful FFNO proof recovery from promoted-source logs
  - mandatory downgrade when recovered proof is still unavailable
- Rebuilt the repaired complete-table root through tmux using the checked-in repair execution manifest, yielding a new authoritative `paper_complete` bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Accepted Six-Row Roster

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_spectral_resnet_bottleneck_net` -> `Spectral ResNet Bottleneck + PINN`
- `pinn_ffno` -> `FFNO-local proxy + PINN`

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
  `complete_table_20260430T134500Z`, then promoted into the repaired authoritative root via
  `complete_execution_manifest_repair.json`
- `pinn_ffno`: promoted from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  - row-local completion proof in the final bundle is rebuilt from the prerequisite root’s durable wrapper log lineage, including `tmux.log` fallback
  - row-local invocation provenance remains backfilled-from-wrapper-contract evidence, consistent with the prerequisite repair summary
  - post-hoc 2026-05-06 caveat: this row used `fno_cnn_blocks=2` and is
    historical FFNO-local-refiner proxy evidence; pure FFNO table use requires
    `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`

## Benchmark Status And Boundary

- benchmark status: `paper_complete`
- claim boundary: `complete_lines128_cdi_benchmark`
- evidence scope: one coherent six-row `N=128` CDI paper bundle under the frozen comparator, seed, sample, and shared-visual-scale contract
- preserved prerequisite distinction:
  - the FFNO-vs-Hybrid root remains prerequisite pair evidence
  - the four-row minimum subset root remains preserved minimum-table evidence
  - the tmux-backed repaired complete-table root is the authoritative six-row paper bundle
- superseded repaired roots that must not be used for paper-facing claims:
  - `complete_table_20260430T140954Z_repair`
  - `complete_table_20260430T141325Z_repair`
  - `complete_table_20260430T150642Z_repair`

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
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_focused_20260430T150921Z.log`
- required deterministic gate:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/pytest_required_20260430T150921Z.log`
- compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies`
  - log:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/compileall_required_20260430T150921Z.log`
- repair rebuild log:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/verification/lines128_complete_table_repair_20260430T150757Z.log`

## Residual Risks

- The authoritative root repairs the missing row-local completion proof, but the FFNO row still inherits the earlier prerequisite pass’s recovered invocation provenance caveat.
- Future promoted-source roots must retain some durable wrapper-log evidence (`launcher_*`, `live_*`, or `tmux.log`) or they will now remain correctly `benchmark_incomplete` instead of being overstated as `paper_complete`.
