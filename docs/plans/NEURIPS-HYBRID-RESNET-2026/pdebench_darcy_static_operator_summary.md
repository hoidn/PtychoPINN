# PDEBench Darcy Static Operator Summary

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Tranche: `phase-2-pdebench-darcy-static-operator-benchmark`
- Updated: `2026-04-20`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`

## Status

Darcy static-operator support is implemented for the PDEBench `128x128` image-suite path. The adapter treats the official beta `1.0` file as `nu[i] -> tensor[i]`, with sample-level deterministic splits, separate train-only input/target normalization, denormalized target-space RMSE/nRMSE metrics, supervised real-channel Hybrid ResNet/FNO/U-Net profiles, literature-context reporting, and CLI readiness/benchmark modes.

The readiness run completed on a tiny cap of the staged official file:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/readiness-cap-20260420T222155Z`
- Profile: `unet_tiny_smoke`
- Evidence scope: `smoke_feasibility_only`
- Metric interpretation: `sanity_only_not_benchmark_performance`
- Performance assessment complete: `false`

This readiness result proves load/train/eval/write plumbing only. It is not a benchmark-performance row and does not satisfy the strong-baseline comparison gate.

## Implemented Contracts

- Data: lazy HDF5 `DarcyStaticOperatorDataset` reading `nu` `(N,H,W)` and `tensor` `(N,1,H,W)` as channel-first float tensors.
- Splits: deterministic sample-level split builder; default official split is `8000/1000/1000` with seed `20260420`.
- Normalization: input stats are fit on train-split `nu`; target stats are fit on train-split `tensor`; predictions are denormalized with target stats before metrics.
- Metrics: denormalized RMSE, nRMSE/relative L2, per-channel metrics, and static-operator horizon metadata.
- Models: `hybrid_resnet_base`, `fno_base`, `unet_strong`, and readiness-only `unet_tiny_smoke`; missing `neuralop` becomes an explicit FNO blocker.
- Reporting: `literature_context.json`, `comparison_summary.json`, and `comparison_summary.csv` include PDEBench/HAMLET/FNO calibration context with protocol caveats.
- CLI: `scripts/studies/run_pdebench_image128_suite.py --task darcy --mode inspect|readiness|benchmark`.

## Full Benchmark State

The full benchmark budget is written but the expensive full run has not been claimed as complete:

- Budget: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`
- Required split: `8000/1000/1000`
- Required profiles: `hybrid_resnet_base`, `fno_base`, `unet_strong`
- Loss/optimizer recipe: MAE, Adam, `lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`

No Darcy full benchmark metrics exist yet. Until all required primary profiles complete on the full training split, the Darcy comparison remains readiness-complete but benchmark-incomplete.

## Literature Calibration

The reporting payload records these context values with protocol caveats:

- PDEBench beta `1.0` U-Net: RMSE `6.4e-3`, nRMSE `3.3e-2`
- PDEBench beta `1.0` FNO: RMSE `1.2e-2`, nRMSE `6.4e-2`
- OFormer context: nRMSE `2.05e-2`
- HAMLET context: nRMSE `1.40e-2`
- FNO/JMLR Darcy context: relative error range near `1.08e-2` to `9.8e-3`

These are calibration bands, not exact same-protocol reproduction targets for the local native-`128x128` run.

## Blockers And Boundaries

- `/home/ollie/Documents/neurips/` does not exist locally, so no manuscript evidence-map update was possible. This tranche did not write paper-facing artifacts there.
- The broader three-task PDEBench image-suite remains blocked on missing `2d_reacdiff` data.
- Tiny smoke U-Net remains readiness-only and cannot be reported as a strong U-Net baseline.

## Documents Read

- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/studies/index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

## Verification

- `python -m pytest tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q` -> `15 passed`
- `python -m pytest tests/studies/test_pdebench_image128_preflight.py tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q` -> `25 passed`
