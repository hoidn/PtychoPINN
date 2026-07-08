# Lines128 Hybrid ResNet Encoder Fusion Variants Summary

- Status: `complete`
- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Authoritative artifact root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Authoritative ablation run root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/runs/encoder_fusion_rerun_20260502T121829Z`
- Fixed baseline source root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Reused baseline row id: `pinn_hybrid_resnet`
- Mandatory fresh row roster:
  - `pinn_hybrid_resnet_encoder_layerscale`
  - `pinn_hybrid_resnet_encoder_branch_gated`
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
- Optional deferred row:
  - `pinn_hybrid_resnet_encoder_fusion_norm`
- Scalar-scope decision: first scored pass uses per-block learned scalars only; shared-across-encoder-block placement remains a distinct future architecture axis.
- Claim boundary: append-only same-contract CDI ablation, decision-support only. The completed six-row `lines128` CDI paper bundle remains the unchanged headline authority.

## Resume-Condition Clearance Note

This backlog item is selectable in the current window because:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md` records the complete six-row `lines128` CDI bundle as the current `paper_grade` headline authority and marks the CDI pillar draftable.
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json` records `2026-04-29-cdi-lines128-paper-benchmark-execution` and `2026-04-29-paper-evidence-package-audit` as completed before this item was selected.
- `docs/steering.md` preserves the current Phase 2 plus Phase 3 selection window, so one bounded `N=128` CDI-strengthening follow-on is allowed without reopening roadmap order.

## Row Roster

- Reused baseline: `pinn_hybrid_resnet`
  - promoted under `runs/encoder_fusion_rerun_20260502T121829Z/promoted_baseline/`
- Fresh per-block LayerScale row: `pinn_hybrid_resnet_encoder_layerscale`
  - launch evidence: `runs/encoder_fusion_rerun_20260502T121829Z/runs/pinn_hybrid_resnet_encoder_layerscale/launcher_completion.json`
- Fresh per-block branch-gated row: `pinn_hybrid_resnet_encoder_branch_gated`
  - launch evidence: `runs/encoder_fusion_rerun_20260502T121829Z/runs/pinn_hybrid_resnet_encoder_branch_gated/launcher_completion.json`
- Fresh per-block combined row: `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
  - launch evidence: `runs/encoder_fusion_rerun_20260502T121829Z/runs/pinn_hybrid_resnet_encoder_branch_gated_layerscale/launcher_completion.json`

Each fresh row also has row-local `training_output_manifest.json`, Lightning logs, checkpoints, reconstructions, invocation artifacts, and stitched metrics under the same run root.

## Main CDI Findings

- `pinn_hybrid_resnet_encoder_layerscale`
  - ΔSSIM amp `-0.0012`, ΔMAE amp `+0.0017`, ΔSSIM phase `-0.0004`, ΔMAE phase `+0.0042`, FRC50 amp `+1.7`, FRC50 phase `+0.0`
  - Read: per-block outer LayerScale alone slightly hurts the dominant stitched channels and is not an improvement.
- `pinn_hybrid_resnet_encoder_branch_gated`
  - ΔSSIM amp `+0.0009`, ΔMAE amp `-0.0008`, ΔSSIM phase `+0.0001`, ΔMAE phase `+0.0021`, FRC50 amp `+1.3`, FRC50 phase `+0.1`
  - Read: branch-level gating is the closest thing to a constructive change, improving amplitude slightly while still paying a small phase MAE trade.
- `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
  - ΔSSIM amp `-0.0017`, ΔMAE amp `+0.0019`, ΔSSIM phase `-0.0002`, ΔMAE phase `+0.0014`, FRC50 amp `-2.8`, FRC50 phase `+30.0`
  - Read: the combined row improves phase-side high-frequency FRC sharply but regresses the dominant amplitude/SSIM channels, so it is not a constructive default over the simpler branch-gated row.

Final stitched metrics, not training loss, are the decision standard for every row.

## Deferred Work

- Shared-across-encoder-block scalar placement remains an explicit future architecture axis and is not represented in this first-pass result.
- `pinn_hybrid_resnet_encoder_fusion_norm` stays omitted because none of the three mandatory rows produced a strong enough stitched-metric win to justify reopening the normalization follow-up within this bounded backlog item.
- Any paper-grade promotion, headline replacement, or residual-path change remains out of scope.

## Verification Logs

Archived under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/verification/`:

- `final_test_fno_generators.log` — `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"` → `34 passed, 59 deselected`
- `final_test_grid_lines_torch_runner.log` — `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"` → `23 passed, 118 deselected`
- `final_test_encoder_fusion_helper.log` — `pytest -q tests/studies/test_lines128_hybrid_resnet_encoder_fusion_variants.py` → `5 passed`
- `final_integration.log` — `pytest -v -m integration --timeout=900` → `5 passed, 4 skipped`
- `final_compileall.log` — `python -m compileall -q ptycho_torch scripts/studies`

Supporting recovery evidence retained for this pass:

- `rerun_prelaunch_test_fno_generators.log`
- `rerun_prelaunch_test_grid_lines_torch_runner.log`
- `rerun_prelaunch_test_encoder_fusion_helper.log`
- `rerun_prelaunch_integration.log`
- `rerun_after_launch_guard_fix_test_encoder_fusion_helper.log`
- `rerun_after_launch_guard_fix_compileall.log`
