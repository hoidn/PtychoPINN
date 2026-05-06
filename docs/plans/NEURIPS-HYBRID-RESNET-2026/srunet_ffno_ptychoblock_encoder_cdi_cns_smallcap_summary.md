# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap`
- State: `completed`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/execution_plan.md`
- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`
- Fresh CDI run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_20260506T183959Z/`
- Fresh CNS run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`

## Claim Boundary

- CDI evidence remains `decision_support_append_only`. The fresh row does not
  replace the immutable six-row `lines128` authority or the append-only U-NO
  extension.
- CNS evidence remains `bounded_capped_decision_support_only`. The fresh row is
  constrained to the matched-condition `history_len=5`, `512 / 64 / 64`,
  `40`-epoch capped lane and cannot reopen full-training or mixed-condition CNS
  claims.
- This item is an encoder-mechanism probe only. It tests whether replacing the
  current SRU-Net encoder with `FFNO -> 2x(PtychoBlock + downsample)` helps or
  hurts while the rest of the SRU-Net shell stays fixed.

## Fixed Contracts

### CDI

- Contract id: `cdi_lines128_seed3`
- `N=128`, `gridsize=1`, `seed=3`
- `40` epochs, batch `16`, Adam `2e-4`
- `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-4, threshold=0.0)`
- `torch_loss_mode=mae`, `torch_output_mode=real_imag`
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`
- probe lineage: `Run1084_recon3_postPC_shrunk_3.npz`, `probe_source=custom`,
  `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`
- fixed sample ids: `0`, `1`

### CNS

- Contract id: matched capped h5 lane
- task: PDEBench `2d_cfd_cns`
- `history_len=5`
- split caps `512 / 64 / 64`
- `40` epochs, batch `4`, Adam `2e-4`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`,
  `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`

## Encoder Recipe

- architecture id: `hybrid_resnet_ffno_ptychoblock_encoder`
- `encoder_variant=ffno_ptychoblock_encoder`
- `ffno_encoder_blocks=2`
- `ffno_encoder_modes=12`
- `ffno_encoder_share_weights=true`
- `ffno_encoder_gate_init=0.1`
- `ffno_encoder_norm=instance`
- `ffno_encoder_mlp_ratio=2.0`
- `ptychoblock_stage_count=2`
- `downsample_steps=2`
- `downsample_op=stride_conv`

## Reused Baseline Lineage

### CDI

- `pinn_hybrid_resnet`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_hybrid_resnet/`
- `pinn_hybrid_resnet_encoder_spectral_only`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/runs/ablation_20260505T010316Z/runs/pinn_hybrid_resnet_encoder_spectral_only/`
- `pinn_ffno`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_ffno/`
- `pinn_hybrid_resnet_ffno_bottleneck`:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi/runs/pinn_hybrid_resnet_ffno_bottleneck/`

### CNS

- matched-condition headline summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- `author_ffno_cns_base` h5 capped `40`-epoch row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-pilot-40ep-20260502T074500Z/`
- `spectral_resnet_bottleneck_base` h5 capped `40`-epoch row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-40ep-20260501T101147Z/`
- `fno_base` and `unet_strong` h5 capped `40`-epoch rows:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-gap-fill-40ep-20260504T214614Z/`

## CDI Result

The fresh CDI row completed successfully under the locked contract. Root and
row invocation payloads both report `status=completed` and `exit_code=0`, the
tracked tmux wait for PID `2982934` returned `EXIT_CODE:0`, and both row-local
and run-root completion proof artifacts are present.

| Row | amp_mae | phase_mae | amp_ssim | phase_ssim | amp_psnr | phase_psnr | amp_frc50 | phase_frc50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pinn_hybrid_resnet_ffno_ptychoblock_encoder` | 0.0317 | 0.0687 | 0.9858 | 0.9943 | 76.15 | 69.51 | 132.72 | 135.05 |
| `pinn_hybrid_resnet` | 0.0269 | 0.0721 | 0.9881 | 0.9947 | 77.52 | 69.22 | 135.46 | 106.80 |
| `pinn_hybrid_resnet_encoder_spectral_only` | 0.0252 | 0.0615 | 0.9899 | 0.9958 | 78.14 | 70.55 | 137.69 | 135.71 |
| `pinn_hybrid_resnet_ffno_bottleneck` | 0.0312 | 0.0876 | 0.9836 | 0.9910 | 76.27 | 67.37 | 133.97 | 95.65 |

Current read from the completed CDI row alone:

- Versus the baseline `pinn_hybrid_resnet`, the encoder swap worsens amplitude
  MAE by `+0.0048` and lowers amp SSIM / amp PSNR / amp FRC50, but improves
  phase MAE by `-0.0033`, phase PSNR by `+0.29`, and phase FRC50 by `+28.25`.
- Versus the append-only `pinn_hybrid_resnet_encoder_spectral_only` row, the
  new encoder is worse on both amplitude and phase aggregate metrics and is
  slightly worse on phase FRC50 as well (`-0.65`).
- Versus the earlier `pinn_hybrid_resnet_ffno_bottleneck` bridge row, the new
  encoder is materially better on phase (`phase_mae 0.0687 vs 0.0876`,
  `phase_psnr 69.51 vs 67.37`, `phase_frc50 135.05 vs 95.65`) while leaving
  amplitude metrics roughly similar.

## CNS Result

The fresh matched-condition CNS row completed successfully. The tmux shell
tracked PID `2991458` to `EXIT_CODE:0`, the required run-root artifacts are
fresh, and the emitted model profile carries the fixed encoder recipe under
`profile_config.*`.

| Row | err_nRMSE | err_RMSE | relative_l2 | fRMSE_low | fRMSE_mid | fRMSE_high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_resnet_ffno_ptychoblock_encoder_cns` | 0.0433 | 1.0462 | 0.0433 | 2.435 | 0.174 | 0.366 |
| `author_ffno_cns_base` | 0.0198 | 0.4773 | 0.0198 | 1.130 | 0.042 | 0.102 |
| `spectral_resnet_bottleneck_base` | 0.0331 | 0.7988 | 0.0331 | 1.854 | 0.171 | 0.262 |
| `fno_base` | 0.0384 | 0.9282 | 0.0384 | 2.134 | 0.122 | 0.433 |
| `unet_strong` | 0.5386 | 13.0111 | 0.5386 | 30.988 | 0.645 | 1.743 |

Current read from the completed CNS row:

- The new encoder row does not beat the matched-condition spectral SRU-Net
  headline. Versus `spectral_resnet_bottleneck_base`, it is worse on aggregate
  error (`err_nRMSE +0.0102`, `err_RMSE +0.2474`, `relative_l2 +0.0102`) and
  slightly worse in the mid/high Fourier bands (`fRMSE_mid +0.0030`,
  `fRMSE_high +0.1033`).
- The new encoder row also trails `fno_base` on aggregate and low/mid-band
  metrics (`err_nRMSE +0.0049`, `fRMSE_low +0.3014`, `fRMSE_mid +0.0519`), but
  it does improve the high-band error (`fRMSE_high -0.0673`).
- The new encoder row is far stronger than `unet_strong` on every recorded
  metric but remains materially weaker than the authored FFNO row.
- Same-contract ranking therefore depends on which signal is emphasized:
  by aggregate error the row lands between `fno_base` and `unet_strong`,
  while by `fRMSE_high` it lands between the spectral SRU-Net row and FNO.

Supporting final artifacts in the run root now include:

- `comparison_summary.json`
- `comparison_summary.csv`
- `metrics_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`
- `model_profile_hybrid_resnet_ffno_ptychoblock_encoder_cns.json`
- `comparison_hybrid_resnet_ffno_ptychoblock_encoder_cns_sample0.png`
- `comparison_hybrid_resnet_ffno_ptychoblock_encoder_cns_sample0.npz`
- `artifact_freshness_check.json`
- `exit_code_proof.json`

## Cross-Pillar Interpretation

- The FFNO-to-PtychoBlock encoder is not a clean SRU-Net upgrade across the two
  governed tasks.
- On CDI, the encoder swap trades away amplitude fidelity versus the locked
  `pinn_hybrid_resnet` anchor while improving phase-side metrics, especially
  phase FRC50.
- On capped CNS, the same encoder is not competitive with the best existing
  SRU-Net or authored-FFNO rows on aggregate error, although it does recover a
  better high-band score than `fno_base`.
- The net result is domain-dependent and phase/high-frequency-skewed rather than
  broadly positive: useful as mechanism evidence, not as a promotion candidate
  for the current SRU-Net defaults.

## Verification

- Deterministic verification logs for the implementation pass live under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/verification/`
- Archived counts:
  - `test_fno_generators.log`: `61 passed, 62 deselected`
  - `test_grid_lines_torch_runner.log`: `12 passed, 143 deselected`
  - `test_pdebench_image128_models.log`: `21 passed, 25 deselected`
  - `test_grid_lines_compare_wrapper.log`: `13 passed, 66 deselected`
  - `test_generator_registry.log`: `11 passed`
  - `test_lightning_checkpoint.log`: `4 passed, 11 deselected`
  - `compileall.log`: exit `0`
- CDI run-root validation:
  `artifact_freshness_check.json` reports
  `all_required_artifacts_present_and_fresh=true`.
- CNS run-root validation:
  `artifact_freshness_check.json` reports
  `all_required_artifacts_present_and_fresh=true`.
- CDI completion proof:
  tmux tracked PID `2982934` to `EXIT_CODE:0`; run-root
  `exit_code_proof.json` and row-local
  `runs/pinn_hybrid_resnet_ffno_ptychoblock_encoder/exit_code_proof.json`
  both exist.
- CNS completion proof:
  tmux tracked PID `2991458` to `EXIT_CODE:0`; run-root
  `exit_code_proof.json` records the tracked tmux wait result because the
  PDEBench runner's `invocation.json` is launch-only and does not carry a final
  `status` / `exit_code` field.
- Machine-readable item-local comparison bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/comparison_bundle.json`

## Residual Risks

- The CDI row is already clearly append-only mechanism evidence, not a new
  headline replacement.
- The CNS profile schema exposes the fixed encoder recipe under
  `profile_config.*` rather than flattening those fields at the top level of
  `model_profile_*.json`. The information is present and discoverable, but
  downstream readers must inspect the nested config payload rather than only the
  top-level keys.
