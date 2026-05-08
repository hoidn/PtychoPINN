---
priority: 38
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_rollout_video.py
  - pytest -q tests/studies/test_pdebench_image128_runner.py -k "cfd_cns or model_state or matched_condition"
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-05-04-cns-matched-condition-table-refresh
  - 2026-05-04-cns-history5-comparator-gap-fill
  - 2026-05-01-cns-author-ffno-history-length-study
  - 2026-04-29-cns-spectral-history-len4plus-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The current CNS rollout-video exporter requires saved model checkpoints, but the authoritative h5 CNS runs predate checkpoint persistence.
  - The operator wants rollout GIFs from actually trained CNS models, not smoke checkpoints.
  - Checkpoints are needed for FNO, SRU-Net, and FFNO under the current manuscript CNS matched-condition contract.
---

# Backlog Item: Refresh CNS Matched-Condition Checkpoints For Rollout GIFs

## Objective

- Reproduce the current manuscript-facing PDEBench CNS matched-condition rows
  for `fno_base`, `spectral_resnet_bottleneck_base`, and
  `author_ffno_cns_base` with the updated CNS runner so each row emits a
  trained checkpoint usable by the autoregressive rollout GIF exporter.

## Scope

- Use the current CNS matched-condition contract:
  - task: PDEBench `2d_cfd_cns`;
  - dataset:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`;
  - spatial shape: `128x128`;
  - `history_len=5`;
  - train/validation/test trajectories: `512 / 64 / 64`;
  - `max_windows_per_trajectory=8`;
  - epochs: `40`;
  - batch size: `4`;
  - loss: normalized-field MSE;
  - optimizer/scheduler: the existing task-local CNS recipe.
- Rerun only the checkpoint-needed rows:
  - `fno_base` (manuscript label `FNO`);
  - `spectral_resnet_bottleneck_base` (manuscript label `SRU-Net`);
  - `author_ffno_cns_base` (manuscript label `FFNO`).
- Preserve the existing table metrics as the current authority unless the
  implementation plan explicitly promotes the rerun metrics after an audit.
  The primary purpose of this item is checkpoint recovery for rollout visuals,
  not metric replacement.
- Do not rerun U-Net or U-NO unless a later operator request expands the row
  set.
- Do not use 1-epoch smoke checkpoints, one-step comparison NPZ files, or
  older `history_len=2` / `2048 / 256 / 256` artifacts for manuscript rollout
  videos.

## Required Outputs

- One item-local run root per row under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/`,
  each containing:
  - `model_state_<row_id>.pt`;
  - `model_state_<row_id>.json`;
  - `model_profile_<row_id>.json`;
  - `metrics_<row_id>.json`;
  - `split_manifest.json`;
  - `normalization_stats_state.json`;
  - `invocation.json` and `invocation.sh`.
- Rollout GIFs generated from the trained checkpoints for at least the density
  field on a fixed test trajectory:
  - initial history;
  - true future;
  - predicted future;
  - absolute error.
- A rollout manifest that records:
  - source run root;
  - checkpoint path;
  - row id and manuscript label;
  - split, trajectory/sample id, start time, number of rollout steps;
  - field order;
  - visual scales.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cns_rollout_checkpoint_refresh_summary.md`
- Optional index updates if downstream tools need to discover the new
  checkpoint roots:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`.

## Completion Gate

- All three required rows must have real trained checkpoints produced by the
  `40`-epoch h5 matched-condition runner.
- Each checkpoint must load through
  `scripts/studies/pdebench_image128/render_cns_rollout_video.py` without
  substituting another model or another history/cap contract.
- At least one rollout GIF per row must be generated and validated for frame
  count, nonzero image content, and manifest consistency.
- Rerun metrics must be compared against the current table authority. If they
  differ materially, keep the existing table metrics unchanged and record the
  checkpoint rerun as visualization lineage only.

## Notes For Reviewer

- Reject any implementation that uses smoke checkpoints or randomly initialized
  weights for rollout visuals.
- Reject any implementation that mixes `history_len=2` checkpoints with the
  current `history_len=5` table.
- Reject any implementation that overwrites the current manuscript CNS table
  just because checkpoint-producing reruns exist.
- Reject any implementation that generates GIFs from one-step NPZ comparison
  outputs; autoregressive rollout requires a live model checkpoint.
