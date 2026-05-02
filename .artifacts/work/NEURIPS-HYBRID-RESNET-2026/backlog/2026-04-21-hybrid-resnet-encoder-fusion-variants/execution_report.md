# Execution Report: 2026-04-21 Hybrid ResNet Encoder Fusion Variants

- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Implementation state: `COMPLETED`
- Ablation root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Run root: `runs/encoder_fusion_rerun_20260502T121829Z`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`

## Completed In This Pass

- Fixed the checked-in helper `scripts/studies/lines128_hybrid_resnet_encoder_fusion_variants.py` so the current-scope rerun could satisfy the plan instead of reusing the historical repaired root:
  - added `run_id` support to the execution scaffold so a fresh unique ablation run root can be created for a compliant rerun
  - switched canonical dataset input sourcing from the recovered baseline-row invocation to the authoritative complete-table `paper_benchmark_manifest.json`
  - fixed the relaunch guard so precreated row log directories do not masquerade as completed rows
- Extended `tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` to cover:
  - unique `run_id` propagation through the execution scaffold
  - canonical dataset-path sourcing from the authoritative baseline manifest
  - fresh-row launches that start with only precreated `stdout.log` / `stderr.log`
- Materialized a fresh scaffold under `runs/encoder_fusion_rerun_20260502T121829Z` and reran all three mandatory fresh rows in tmux from repo root with the `ptycho311` environment active:
  - `pinn_hybrid_resnet_encoder_layerscale` — PID `1924774`, `2026-05-02T12:31:41+00:00` to `2026-05-02T12:49:56+00:00`
  - `pinn_hybrid_resnet_encoder_branch_gated` — PID `1925654`, `2026-05-02T12:50:57+00:00` to `2026-05-02T13:09:31+00:00`
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale` — PID `1927137`, `2026-05-02T13:10:08+00:00` to `2026-05-02T13:29:00+00:00`
- For each fresh rerun row, wrote true row-local launch evidence under the fresh run root:
  - `runs/<row_id>/launcher_completion.json`
  - `training_runs/<row_id>/training_output_manifest.json`
  - row-local `stdout.log` and `stderr.log`
  - row-local Lightning logs and checkpoints under `training_runs/<row_id>/`
- Rebuilt the append-only bundle against the fresh run root:
  - refreshed `metrics.json`, `comparison_summary.json`, `metrics_table.csv`, `metrics_table.tex`, `model_manifest.json`, and root `visuals/`
  - refreshed the durable summary to point at the fresh rerun rather than the historical repaired launch

## Completed Current-Scope Work

- Blocking review item 1 is fixed with a true compliant rerun. The three mandatory fresh rows were re-executed under regenerated row-local `training_runs/<row_id>/` output roots inside a fresh unique ablation run root, instead of relying on repaired shared-output provenance.
- Blocking review item 2 remains satisfied and is strengthened. The reviewed work still uses the checked-in repo-local helper, and that helper now owns the unique-run-root and canonical-dataset-path parts of the contract as well.
- Medium review item 1 is fixed with fresh proof. The final archived integration log now has a complete footer and passing summary in `verification/final_integration.log`, so the execution report no longer relies on the previously truncated review-fix archive.
- The published numerical read remains decision-support only and does not reopen the optional normalized-fusion lane:
  - `pinn_hybrid_resnet_encoder_layerscale`: not an improvement
  - `pinn_hybrid_resnet_encoder_branch_gated`: closest to a constructive change, with a small amplitude win and small phase MAE trade
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`: stronger phase FRC but not a constructive overall default

## Follow-Up Work

- Add an out-of-process checkpoint-rebuild probe for the encoder-fusion overrides if a future task needs to resume these row-local checkpoints independently of the in-process runner path.
- Keep `pinn_hybrid_resnet_encoder_fusion_norm` deferred unless a later bounded extension explicitly reopens the normalization axis after a clearer stitched-metric win.
- Keep shared-across-encoder-block scalar placement deferred as a separate architecture axis rather than folding it into this per-block first pass.

## Verification

Final closeout evidence archived under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/verification/`:

- `final_test_fno_generators.log`
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"`
  - Result: `34 passed, 59 deselected`
- `final_test_grid_lines_torch_runner.log`
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"`
  - Result: `23 passed, 118 deselected`
- `final_test_encoder_fusion_helper.log`
  - `pytest -q tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py`
  - Result: `5 passed`
- `final_integration.log`
  - `pytest -v -m integration --timeout=900`
  - Result: `5 passed, 4 skipped`
- `final_compileall.log`
  - `python -m compileall -q ptycho_torch scripts/studies`

Supporting recovery logs retained for the mid-pass helper fixes:

- `rerun_prelaunch_test_fno_generators.log`
- `rerun_prelaunch_test_grid_lines_torch_runner.log`
- `rerun_prelaunch_test_encoder_fusion_helper.log`
- `rerun_prelaunch_integration.log`
- `rerun_after_launch_guard_fix_test_encoder_fusion_helper.log`
- `rerun_after_launch_guard_fix_compileall.log`

## Residual Risks

- The ablation is still single-seed (`seed=3`) and bounded to the current backlog-item budget. Small stitched SSIM/MAE deltas at the `1e-4` level remain decision-support evidence, not promotion-grade proof.
- The helper now guarantees a fresh unique run root and canonical dataset sourcing, but the offline checkpoint-rebuild path for these encoder-fusion overrides was not separately exercised beyond the normal in-process row execution.
- `tensorflow-addons` still emits the known unsupported-version warnings in the integration gate under the current environment. They are pre-existing warnings rather than a regression from this item.
- `pinn_hybrid_resnet_encoder_fusion_norm` and shared-scope encoder scalar placement remain deliberately out of scope for this completed pass.
