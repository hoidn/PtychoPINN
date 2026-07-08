# PDEBench Darcy Static Operator Summary

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Tranche: `phase-2-pdebench-darcy-static-operator-benchmark`
- Updated: `2026-05-04`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- Backlog item: `2026-05-04-pdebench-darcy-full-training-benchmark`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`

## Status

Darcy static-operator benchmark is now full-training complete. All three required primary profiles (`hybrid_resnet_base`, `fno_base`, `unet_strong`) trained and evaluated under one fixed contract on the official PDEBench `2D_DarcyFlow_beta1.0_Train.hdf5` file with the locked `8000 / 1000 / 1000` sample-level split, relative-L2 loss, Adam `lr=2e-4`, and `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)` for `50` epochs at batch size `8` on a single RTX 3090.

Earlier readiness evidence is preserved as smoke feasibility only; only the rows below carry the `benchmark_performance` evidence scope.

## Earlier Readiness Evidence (Preserved Provenance, Not Benchmark)

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/readiness-cap-20260420T222155Z`
- Profile: `unet_tiny_smoke`
- Evidence scope: `smoke_feasibility_only`
- Metric interpretation: `sanity_only_not_benchmark_performance`
- Performance assessment complete: `false`

This readiness result remains usable as load/train/eval/write plumbing proof only. It is not a strong-baseline row.

## Full-Training Benchmark Run

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/full_benchmark_20260504T182832Z`
- Run id: `20260504T182921.503043Z`
- Mode: `benchmark`
- Evidence scope: `benchmark_performance`
- Performance assessment complete: `true`
- Run budget: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`
- Split (sample-level, no overlap): `train=8000`, `val=1000`, `test=1000`, seed `20260420`
- Loss: relative L2 (`||pred-target||_2 / ||target||_2` per sample, then mean over batch)
- Optimizer: Adam, `lr=2e-4`
- Scheduler: `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`
- Epochs: `50`; batch size `8`; precision `float32`; device `cuda:0` (RTX 3090)
- Metrics units: denormalized target units; per-channel and aggregate `err_RMSE`, `err_nRMSE`/`relative_l2`

Same-contract test-set headline metrics (denormalized):

| Profile | Status | err_RMSE | err_nRMSE | relative_l2 | Parameter count |
|---|---|---:|---:|---:|---:|
| `hybrid_resnet_base` | completed | 0.018767 | 0.085735 | 0.085735 | 7,786,178 |
| `fno_base` | completed | 0.018109 | 0.082727 | 0.082727 | 357,217 |
| `unet_strong` | completed | 0.020415 | 0.093264 | 0.093264 | 7,762,465 |

Same-contract ranking by `err_nRMSE`/`relative_l2` (lower is better): `fno_base < hybrid_resnet_base < unet_strong`. The same ordering holds for `err_RMSE`. The Hybrid ResNet baseline does not displace FNO under this fixed contract; the ordering is reported as observed and is not relabeled.

Headline mandatory artifacts under the run root:

- `invocation.json`, `invocation.sh`
- `dataset_manifest.json`, `hdf5_metadata.json`
- `split_manifest.json`
- `normalization_stats_input.json`, `normalization_stats_target.json`
- `model_profile_hybrid_resnet_base.json`, `model_profile_fno_base.json`, `model_profile_unet_strong.json`
- `metrics_hybrid_resnet_base.json`, `metrics_fno_base.json`, `metrics_unet_strong.json`
- `comparison_summary.json`, `comparison_summary.csv`
- `literature_context.json`

Optional comparison packaging present for all three profiles:

- `comparison_<profile>_sample0.png`
- `comparison_<profile>_sample0.npz`

## Implemented Contracts

- Data: lazy HDF5 `DarcyStaticOperatorDataset` reading `nu` `(N,H,W)` and `tensor` `(N,1,H,W)` as channel-first float tensors.
- Splits: deterministic sample-level split builder; benchmark split is `8000/1000/1000` with seed `20260420`.
- Normalization: input stats are fit on train-split `nu`; target stats are fit on train-split `tensor`; predictions are denormalized with target stats before metrics.
- Metrics: denormalized RMSE, nRMSE/relative L2, per-channel metrics, and static-operator horizon metadata.
- Models: `hybrid_resnet_base`, `fno_base`, `unet_strong` are the locked benchmark profiles; `unet_tiny_smoke` remains readiness-only.
- Reporting: `literature_context.json`, `comparison_summary.json`, and `comparison_summary.csv` include PDEBench/HAMLET/FNO calibration context with protocol caveats.
- CLI: `scripts/studies/run_pdebench_image128_suite.py --task darcy --mode inspect|readiness|benchmark`.

## Literature Calibration

Calibration bands recorded in `literature_context.json` (different protocols; not exact same-contract reproduction targets for the local native `128x128` runs):

- PDEBench beta `1.0` U-Net: RMSE `6.4e-3`, nRMSE `3.3e-2`
- PDEBench beta `1.0` FNO: RMSE `1.2e-2`, nRMSE `6.4e-2`
- OFormer context: nRMSE `2.05e-2`
- HAMLET context: nRMSE `1.40e-2`
- FNO/JMLR Darcy context: relative error range near `1.08e-2` to `9.8e-3`

PDEBench-published values commonly use `2x` spatial subsampling and different training protocols; the local runs above use the official native `128x128` field. The literature numbers are calibration context only.

## Claim Boundary

- Allowed claim under this run: same-contract Darcy static-operator full-training benchmark for `hybrid_resnet_base`, `fno_base`, and `unet_strong` on the official PDEBench beta `1.0` file under the locked `8000/1000/1000` sample-level split, relative-L2 loss, Adam `lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`, `50` epochs, batch size `8`, single RTX 3090.
- Disallowed: relabeling these rows as same-protocol PDEBench/HAMLET reproduction; making generalized Darcy SOTA claims; attributing the `fno_base < hybrid_resnet_base` gap to anything beyond what the recorded provenance supports.

## Documents Read

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark/execution_plan.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

## Verification

- `python -m pytest -q tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `108 passed in 67.99s`
- Run-budget contract: `python -c "from scripts.studies.pdebench_image128.run_config import validate_darcy_run_budget; ..."` -> `darcy run budget valid`
- Tracked launched PID `2254398` exited `0` (`exit.code = 0`).
- All sixteen mandatory benchmark artifacts present under the run root (validated by the plan's blocking artifact check).
