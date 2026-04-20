# PDEBench SWE Longer Execution Summary

## Scope

This is the Roadmap Phase 2 longer-execution gate for the selected PDEBench 2D Shallow Water Equations (`swe`) primary benchmark. It covers one-step next-state prediction for Hybrid ResNet, FNO, and U-Net under one shared data, split, normalization, metric, and evaluation contract.

Post-review seed caveat: the selected run below was launched before the longer runner recorded a model/training RNG seed. Its metrics are retained only as unseeded observed evidence for the SWE pivot screen, not as fixed-seed paper-facing evidence. The current runner and run-budget contract now require `training_seed=20260420` for reruns.

Non-goals respected: no CDI Phase 3 anchor regeneration, no Phase 4 `256x256` CDI scaling, no OpenFWI execution, no rollout evaluation, no manuscript prose, and no paper-facing artifact assembly.

## Documents And Artifacts Used

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- Smoke gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- Longer execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-longer-execution/execution_plan.md`
- Raw artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`
- Selected run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/`

## Dataset Identity

- Source: PDEBench repository, `https://github.com/pdebench/PDEBench`.
- DaRUS datafile: `133021`.
- Official file: `2D_rdb_NA_NA.h5`.
- Local path: `/home/ollie/Documents/pdebench-data/swe/2D_rdb_NA_NA.h5`.
- Size: `6626098972` bytes.
- SHA256: `28f0c33723d70eebb420fc170e94b675c18e032fb697dcef080e114ca9645e3a`.
- Source manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/dataset_manifest.json`.

## License And Access Status

The preflight note at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/preflight/license_access.md` records the PDEBench repository license and DaRUS dataset terms from primary sources.

Summary: the PDEBench repository code is MIT licensed except where otherwise stated; this tranche does not import PDEBench source code. The DaRUS dataset `doi:10.18419/DARUS-2986` is released as `CC BY 4.0`; the file metadata for datafile `133021` reports `restricted: false`.

## HDF5 Layout

- Selected state dataset: grouped virtual path `*/data`.
- Path pattern: `{trajectory_id:04d}/data`.
- Axis order: `NTHWC`.
- Shape: `[1000, 101, 128, 128, 1]`.
- Dtype: `float32`.
- HDF5 metadata: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/hdf5_metadata.json`.

## Split And Subset

- Split seed: `20260420`.
- Full split: 800 train, 100 validation, 100 test trajectories.
- Full one-step pair counts: train `80000`, validation `10000`, test `10000`.
- Run subset: same 800/100/100 trajectory split, capped to `10` one-step pairs per trajectory.
- Run one-step pair counts: train `8000`, validation `1000`, test `1000`.
- Full split manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/split_manifest_full.json`.
- Run subset manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/split_manifest_run.json`.

## Normalization And Metric Contract

- Normalization source: train split inputs only.
- Limit semantics: `normalization_max_samples=8000` with `limit_kind=samples`.
- Samples used: `8000`.
- Mean: `1.0326519775603884`.
- Std: `0.16561010415232227`.
- Primary metric: one-step `err_nRMSE`.
- Secondary metric: one-step `err_RMSE`.
- Metric units: denormalized physical units.
- Normalization stats: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/normalization_stats.json`.

## Run Budget

- Epochs: `15`.
- Batch size: `16`.
- Learning rate: `1e-3`.
- Training/model seed: not recorded by selected run `20260420T115509.961336393Z`; post-review runner contract requires `training_seed=20260420` in the budget or CLI before constructing each model profile.
- Max train trajectories: `800`.
- Max validation trajectories: `100`.
- Max test trajectories: `100`.
- Max pairs per trajectory: `10`.
- Device: `cuda`.
- Num workers: `2`.
- Budget record: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/run_budget.json`.

## Long-Run Command

Selected run ID: `20260420T115509.961336393Z`.

Tracked child PID: `3052378`.

The tmux launch wrote selected identity markers before launch:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/selected_longer.run_id`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/selected_longer.run_root`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/selected_longer.tmux_session`

The run invocation is recorded at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/invocation.sh` and `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/invocation.json`.

## Local Baseline Results

These rows are unseeded observed results because the selected run did not record a model/training RNG seed. The data split is deterministic, but model initialization and optimizer RNG state cannot be reconstructed from the recorded artifacts.

| Profile | Parameters | Train batches | Val err_nRMSE | Test err_nRMSE | Test err_RMSE | Runtime sec | Peak CUDA bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fno_base` | 44465 | 7500 | 0.0026630464 | 0.0026049435 | 0.0027217311 | 102.8992 | 512434176 |
| `unet_base` | 18465 | 7500 | 0.0014447536 | 0.0013982666 | 0.0014609551 | 29.5648 | 206197248 |

Both required local baselines completed with metrics and provenance under the same one-step contract.

## Hybrid ResNet Result

This row is subject to the same unseeded-evidence caveat as the local baselines.

| Profile | Parameters | Train batches | Val err_nRMSE | Test err_nRMSE | Test err_RMSE | Runtime sec | Peak CUDA bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_resnet_base` | 291586 | 7500 | 0.0026856791 | 0.0026427819 | 0.0027612660 | 80.6694 | 423643136 |

The best local baseline was `unet_base` with test `err_nRMSE=0.0013982666`. Hybrid ResNet test `err_nRMSE=0.0026427819`, a relative gap of `0.8900414694` versus the best baseline. This fails the plan's operational viability rule: Hybrid ResNet must be no worse than `10%` relative above the best local baseline before ablations are run.

## Ablations

Ablations were skipped. The machine-readable gate input in `comparison_summary.json` records `recommended_decision_input=primary_noncompetitive` and `ablation_skip_reason=primary_noncompetitive`.

No `hybrid_resnet_spectral_reduced` or `hybrid_resnet_local_reduced` result is promoted from this run.

## Published SOTA Caveats

No published SOTA number is used as a same-protocol result in this summary. Any PDEBench published SOTA or paper row remains protocol-dependent context unless the same task, split, preprocessing, model code or accepted reimplementation, metric code, and one-step horizon are locally reproduced.

## Raw Artifact Links

- Comparison JSON: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/comparison_summary.json`
- Comparison CSV: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/comparison_summary.csv`
- Hybrid ResNet metrics: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/runs/hybrid_resnet_base/metrics.json`
- FNO metrics: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/runs/fno_base/metrics.json`
- U-Net metrics: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/runs/unet_base/metrics.json`
- Stdout log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_20260420T115509.961336393Z.stdout.log`
- Stderr log: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/logs/longer_20260420T115509.961336393Z.stderr.log`
- Exit code: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/runs/20260420T115509.961336393Z/logs/longer.exit_code`

## Gate Checks

- Official SWE identity, SHA256, grouped HDF5 layout, and source/access terms are recorded.
- Full and run-subset split manifests are separate and deterministic.
- Normalization limits are recorded as samples, not batches.
- CUDA peak memory is reset per profile and reported as `per_profile_cuda_peak`.
- Hybrid ResNet, FNO, and U-Net completed under one data/split/horizon/normalization/metric contract, but the selected run lacks fixed model/training seed provenance and is downgraded to unseeded observed evidence.
- Post-review runner behavior now rejects live `logs/longer.pid` roots without `logs/longer.exit_code`, requires `training_seed` in the run budget, records the seed in invocation/profile artifacts, and requires both FNO and U-Net for local-baseline completeness.
- The long-run completion is bound to the persisted selected run root and tracked child PID; `logs/longer.exit_code` contains `0`.
- No CDI, OpenFWI execution, `256x256` scaling, paper-facing artifact assembly, or stable core physics/model module edit was performed.

## Residual Risks

- This is one unseeded observed run and one one-step budget. It is sufficient only as a conservative Phase 2 pivot signal, not as fixed-seed reproducible or broad robustness evidence.
- The local U-Net baseline is much stronger on this one-step subset than Hybrid ResNet and FNO; this may reflect task fit, capacity, optimization, or the simplicity of the short-horizon SWE subset.
- The OpenFWI fallback still requires a separate approved execution tranche before it can replace this failed SWE primary for the PDE pillar.

Decision: pivot to OpenFWI FlatVel-A
