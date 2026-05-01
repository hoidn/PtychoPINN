# Lines128 Hybrid ResNet Skip/Residual Ablation Summary

- Date: `2026-05-01`
- Backlog item: `2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation`
- State: `completed`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/execution_plan.md`
- Baseline source root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Ablation artifact root: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation`

## Intended Fresh Row Roster

- Reused baseline: `pinn_hybrid_resnet`
- Fresh: `pinn_hybrid_resnet_skip_add`
- Fresh: `pinn_hybrid_resnet_residual_fixed`
- Fresh: `pinn_hybrid_resnet_skip_add_residual_fixed`
- Optional after mandatory rows only: `pinn_hybrid_resnet_skip_gated_add`

## Results

Same-contract comparison against the reused `pinn_hybrid_resnet` anchor:

| Row | Changed factor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp FRC50 | Phase FRC50 |
|---|---|---:|---:|---:|---:|---:|---:|
| `pinn_hybrid_resnet` | reused baseline | 0.026939 | 0.072063 | 0.988114 | 0.994740 | 135.464222 | 106.800609 |
| `pinn_hybrid_resnet_skip_add` | enable decoder skip fusion with `add` style | 0.026447 | 0.061022 | 0.988681 | 0.993895 | 135.389826 | 135.960640 |
| `pinn_hybrid_resnet_residual_fixed` | bottleneck residual scale `learned -> fixed` | 0.024611 | 0.077322 | 0.990003 | 0.994298 | 135.916886 | 106.679719 |
| `pinn_hybrid_resnet_skip_add_residual_fixed` | skip-add plus fixed residual scale | 0.028890 | 0.063259 | 0.986797 | 0.992850 | 135.413852 | 106.884035 |

Current read:

- `pinn_hybrid_resnet_skip_add` is the clearest phase-oriented variant. It improved phase MAE by `0.011041` and phase FRC50 by `29.160031` versus the anchor, while slightly improving amplitude MAE and amplitude SSIM. The only notable loss was a small phase SSIM drop (`0.000845`).
- `pinn_hybrid_resnet_residual_fixed` is the clearest amplitude-oriented variant. It achieved the best amplitude MAE (`0.024611`) and amplitude SSIM (`0.990003`), but it worsened phase MAE by `0.005258` and slightly regressed phase SSIM/FRC50 relative to the anchor.
- `pinn_hybrid_resnet_skip_add_residual_fixed` did not show constructive interaction. It preserved some phase-MAE improvement over baseline, but it was clearly worse than the simpler single-factor variants on amplitude MAE and amplitude SSIM.
- No fresh row displaced the current paper-grade `pinn_hybrid_resnet` baseline as the headline CDI Hybrid authority. This bundle remains decision-support evidence only.

Bundle notes:

- `metrics.json` and `model_manifest.json` intentionally report `benchmark_status: "benchmark_incomplete"` because the bundle is append-only same-contract ablation evidence and all rows remain `decision_support`, not `paper_grade`.
- The optional `pinn_hybrid_resnet_skip_gated_add` row was intentionally deferred to keep the run budget bounded after the three required fresh rows.
- The fresh-row training outputs are now isolated under `training_runs/<row_id>/...` and recorded in `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/training_output_recovery.json`. Those row-local roots were recovered post-review from the original shared Lightning output root without rewriting the completed row metrics or run-level invocation history.

## Cross-References

- Legacy skip/mode study: `docs/studies/index.md#hybrid-resnet-mode-skip-sweep`
- CNS skip-add context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Encoder-fusion follow-up: `docs/backlog/active/2026-04-21-hybrid-resnet-encoder-fusion-variants.md`

## Claim Boundary

Append-only same-contract CDI ablation. This does not replace the completed six-row CDI headline bundle and does not promote any fresh row into paper-facing headline evidence.

## Residual Risks

- The skip-add read is based on a two-sample test split under the frozen `lines128` contract and should not be overgeneralized beyond decision-support without a later promotion plan.
- The fixed residual-scale knob is only validated here as a same-contract hybrid-shell ablation. Broader transfer to other datasets or model families remains untested.
- The recovered `training_runs/<row_id>/` roots are faithful copies of the original per-version Lightning logs and checkpoints, but the original shared-root launch history remains visible in the preserved invocation artifacts and legacy shared `lightning_logs/` directory.
