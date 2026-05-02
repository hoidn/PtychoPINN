# Execution Report: 2026-04-21 Hybrid ResNet Encoder Fusion Variants

- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Implementation state: `COMPLETED`
- Ablation root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Run root: `runs/encoder_fusion_rerun_20260502T121829Z`
- Durable summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`

## Completed In This Pass

- Fixed the checked-in helper `scripts/studies/lines128_hybrid_resnet_encoder_fusion_variants.py` so a fresh `prepare` invocation no longer defaults to the historical repaired run root:
  - `prepare_execution_scaffold()` now generates a unique `encoder_fusion_<timestamp>` run id when no explicit `--run-id` or existing manifest run id is present
  - the CLI `prepare --run-id` flag is now optional instead of silently binding every fresh launch to `encoder_fusion_20260502T104230Z`
- Repaired the checked-in Tranche 1 contract surfaces under `.artifacts/work/.../2026-04-21-hybrid-resnet-encoder-fusion-variants/`:
  - `execution_manifest.json` now strips denylisted recovered-row path keys from `baseline_invocation_args`, records the full canonical dataset-input provenance surface (`train_npz`, `test_npz`, `gt_recon`, `dataset_identity_manifest`), and includes the required claim-boundary plus resume-condition-clearance sections
  - `row_contract_audit.json` now records the required `frozen_semantic_model_fields`, accepted non-path baseline-row invocation fields, canonical dataset provenance, regenerated row-local output-path template, denylisted historical path keys, and the same claim-boundary plus resume-condition-clearance rationale
- Extended `tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` with review-specific coverage for:
  - default fresh `run_id` generation
  - required manifest/audit contract sections
  - the existing unique-run-id, canonical-dataset-path, relaunch-guard, and repair paths after the run-id behavior change

## Completed Current-Scope Work

- Blocking review item 1 is fixed. The checked-in Tranche 1 contract artifacts now match the approved plan instead of carrying stale recovered-row path-bearing keys or an incomplete dataset/semantic-field audit surface.
- Blocking review item 2 is fixed. The checked-in helper now owns the unique-run-root contract by default rather than relying on the already-existing rerun manifest to avoid the historical repaired root.
- The previously completed fresh rerun rows, row-local launch evidence, rebuilt ablation bundle, durable summary, and decision-support numerical read remain the current-scope execution authority:
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
  - Result: `7 passed`
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
- `final_review_fix_test_encoder_fusion_helper.log`
- `final_review_fix_test_fno_generators.log`
- `final_review_fix_test_grid_lines_torch_runner.log`
- `final_review_fix_compileall.log`
- `final_review_fix_integration.log`

## Residual Risks

- The ablation is still single-seed (`seed=3`) and bounded to the current backlog-item budget. Small stitched SSIM/MAE deltas at the `1e-4` level remain decision-support evidence, not promotion-grade proof.
- The helper now guarantees a fresh unique run root and canonical dataset sourcing, but the offline checkpoint-rebuild path for these encoder-fusion overrides was not separately exercised beyond the normal in-process row execution.
- `tensorflow-addons` still emits the known unsupported-version warnings in the integration gate under the current environment. They are pre-existing warnings rather than a regression from this item.
- `pinn_hybrid_resnet_encoder_fusion_norm` and shared-scope encoder scalar placement remain deliberately out of scope for this completed pass.
