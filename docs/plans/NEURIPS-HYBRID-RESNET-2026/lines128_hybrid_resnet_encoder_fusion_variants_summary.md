# Lines128 Hybrid ResNet Encoder Fusion Variants Summary

- Status: `complete`
- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Authoritative artifact root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Authoritative ablation run root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/runs/encoder_fusion_20260502T104230Z`
- Fixed baseline source root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Reused baseline row id: `pinn_hybrid_resnet`
- Mandatory fresh row roster (per-block scalars):
  - `pinn_hybrid_resnet_encoder_layerscale`
  - `pinn_hybrid_resnet_encoder_branch_gated`
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
- Optional follow-up row (omitted in this pass; see "Deferred work"):
  - `pinn_hybrid_resnet_encoder_fusion_norm`
- Scalar-scope decision (first scored pass): per-block learned scalars; shared-across-encoder-block scalars remain a distinct architecture axis and are not part of this ablation.
- Claim boundary: append-only same-contract CDI ablation, decision-support only. The completed six-row Lines128 paper bundle remains the unchanged headline authority.

## Provenance Repair

The original fresh-row launch preserved the scored row artifacts and shared-root Lightning outputs, but not row-local training roots. In the review-fix pass, the checked-in helper `scripts/studies/lines128_hybrid_resnet_encoder_fusion_variants.py` repaired that provenance deterministically from preserved fusion-mode metadata:

- `runs/encoder_fusion_20260502T104230Z/training_runs/<row_id>/lightning_logs/version_0/` now contains the row-attributed Lightning log copy for each fresh row.
- `runs/encoder_fusion_20260502T104230Z/training_runs/<row_id>/checkpoints/{last,best}.ckpt` now contains the row-attributed checkpoint copy for each fresh row.
- `runs/encoder_fusion_20260502T104230Z/runs/<row_id>/{stdout.log,stderr.log,launcher_completion.json}` now contains row-local launch evidence for each fresh row.
- `row_contract_audit.json` and `model_manifest.json` were updated to point at the repaired row-local paths rather than the historical shared `output_dir`.

## Resume-Condition Clearance Note

This backlog item is selectable on this iteration because:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md` records the complete six-row `lines128` CDI bundle as the current `paper_grade` headline authority and the CDI pillar as draftable now.
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json` lists `2026-04-29-cdi-lines128-paper-benchmark-execution` and `2026-04-29-paper-evidence-package-audit` in `completed_items` prior to this iteration's selection on `2026-05-02T09:56:54Z`.
- `docs/steering.md` preserves the current Phase 2 plus Phase 3 selection window; one bounded `N=128` architecture follow-on for CDI evidence strengthening is allowed without reopening roadmap order or displacing remaining PDE authority.

## Same-Contract Row Roster

- **pinn_hybrid_resnet** (Hybrid ResNet + PINN (reused baseline))
  - status: `reused_baseline`
  - changed factor: none (reused baseline)
  - SSIM amp/phase: 0.9881 / 0.9947
  - MAE amp/phase:  0.0269 / 0.0721
  - FRC50 amp/phase: 135.46 / 106.80

- **pinn_hybrid_resnet_encoder_layerscale** (Hybrid ResNet + per-block encoder LayerScale + PINN)
  - status: `fresh`
  - changed factor: encoder_fusion_mode=layerscale; per-block outer LayerScale on the fused update
  - SSIM amp/phase: 0.9870 / 0.9943
  - MAE amp/phase:  0.0286 / 0.0762
  - FRC50 amp/phase: 137.20 / 106.80

- **pinn_hybrid_resnet_encoder_branch_gated** (Hybrid ResNet + per-block encoder branch gates + PINN)
  - status: `fresh`
  - changed factor: encoder_fusion_mode=branch_gated; per-block spectral and local branch gates
  - SSIM amp/phase: 0.9890 / 0.9948
  - MAE amp/phase:  0.0262 / 0.0742
  - FRC50 amp/phase: 136.74 / 106.87

- **pinn_hybrid_resnet_encoder_branch_gated_layerscale** (Hybrid ResNet + per-block encoder branch gates + LayerScale + PINN)
  - status: `fresh`
  - changed factor: encoder_fusion_mode=branch_gated_layerscale; per-block branch gates + per-block LayerScale
  - SSIM amp/phase: 0.9864 / 0.9946
  - MAE amp/phase:  0.0289 / 0.0734
  - FRC50 amp/phase: 132.64 / 136.83


## Per-row Deltas vs Reused Baseline

### pinn_hybrid_resnet_encoder_layerscale
  - amp: ΔSSIM=-0.0012, ΔMAE=+0.0017
  - phase: ΔSSIM=-0.0004, ΔMAE=+0.0042

### pinn_hybrid_resnet_encoder_branch_gated
  - amp: ΔSSIM=+0.0009, ΔMAE=-0.0008
  - phase: ΔSSIM=+0.0001, ΔMAE=+0.0021

### pinn_hybrid_resnet_encoder_branch_gated_layerscale
  - amp: ΔSSIM=-0.0017, ΔMAE=+0.0019
  - phase: ΔSSIM=-0.0002, ΔMAE=+0.0014

## Main CDI Findings (final stitched metrics)

- **per-block encoder LayerScale alone** (`pinn_hybrid_resnet_encoder_layerscale`): did not improve over the baseline on the dominant SSIM/MAE/PSNR channels (slight regression of `-0.0012` SSIM amp and `+0.0017` MAE amp; phase channels also regress). FRC50 amp picks up by `+1.7` while phase FRC50 is unchanged. Net read: per-block outer LayerScale alone, initialized at `0.1`, suppresses the encoder update enough to slightly hurt the dominant stitched-metric channels at this contract.
- **per-block encoder branch gates alone** (`pinn_hybrid_resnet_encoder_branch_gated`): mixed but the closest to a real improvement among the three. Both SSIM channels improve (`+0.0009` amp, `+0.0001` phase) and amp MAE improves (`-0.0008`). Phase MAE worsens by `+0.0021`. Net read: branch-level gating helps amplitude reconstruction without breaking phase, but the trade is small enough that this row is not a paper-grade improvement.
- **per-block branch gates combined with per-block LayerScale** (`pinn_hybrid_resnet_encoder_branch_gated_layerscale`): does not improve the dominant SSIM/MAE channels (`-0.0017` SSIM amp, `+0.0019` MAE amp), but produces the most striking single-channel result: phase `FRC50` jumps from `~106.80` (baseline) to `~136.83` (combined row). Amp `FRC50` regresses from `135.46` to `132.64`. Net read: combining encoder LayerScale with branch gates redistributes capacity into phase-side high-frequency reconstruction (FRC) but at the cost of dominant amplitude/SSIM channels, so this row is not a constructive replacement for the simpler single-factor variants.

The reused baseline's stitched metrics (final, not training loss) are the decision standard. Training-loss-only impressions are not used to decide whether a variant helped.

The branch-gated row is the only fresh row that is mildly competitive with the baseline on the dominant SSIM/MAE channels; even there the trade is too small to displace the paper-grade `pinn_hybrid_resnet` anchor or to motivate opening the optional `encoder_fusion_norm` follow-up under the bounded budget.

## Claim Boundary

This is an append-only same-contract decision-support ablation. It does not promote any fresh row to paper-grade headline evidence. The reused `pinn_hybrid_resnet` baseline retains its original paper-grade status from the completed six-row Lines128 paper bundle and that bundle is unchanged by this work.

## Deferred Work

- Shared-across-encoder-block scalar placement: explicit follow-up architecture axis. Not a substitution for the per-block first-pass rows here.
- Optional normalized-fusion row (`pinn_hybrid_resnet_encoder_fusion_norm`): omitted in this pass because the primary matrix did not produce a clear improvement that justified opening the follow-up lane within the bounded budget for this backlog item. If a future ablation chooses to revisit fusion normalization, it should reuse this same fixed contract.
- Variant promotion or any change to the encoder residual path identity rule.

## Index Updates

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/studies/index.md`

## Verification Logs

Archived under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/verification/`:

- `preflight_test_fno_generators.log` — `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"` (preflight, passing)
- `preflight_test_grid_lines_torch_runner.log` — `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"` (preflight, passing)
- `preflight_integration.log` — `pytest -v -m integration --timeout=900` (preflight, 5 passed, 4 skipped long-runners)
- `final_test_fno_generators.log` and `final_test_grid_lines_torch_runner.log` — final closeout reruns of the backlog-required gates after row publication
- `final_compileall.log` — `python -m compileall -q ptycho_torch scripts/studies` after code changes
- `review_fix_test_fno_generators.log`, `review_fix_test_grid_lines_torch_runner.log`, `review_fix_test_encoder_fusion_helper.log`, `review_fix_integration.log`, and `review_fix_compileall.log` — review-fix reruns after the row-local provenance repair
