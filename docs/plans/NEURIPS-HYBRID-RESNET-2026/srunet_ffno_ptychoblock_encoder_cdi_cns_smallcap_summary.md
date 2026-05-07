# SRU-Net FFNO-To-PtychoBlock Encoder CDI/CNS Small-Cap Summary

- Date: `2026-05-07`
- Backlog item: `2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap`
- State: `completed`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/execution_plan.md`
- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`

## Claim Boundary

- CDI row `pinn_hybrid_resnet_ffno_ptychoblock_encoder` is
  `decision_support_append_only`.
- CNS row `hybrid_resnet_ffno_ptychoblock_encoder_cns` is
  `bounded_capped_decision_support_only`.
- Both corrected rows are `20`-epoch mechanism probes. They do not replace the
  six-row CDI authority or the matched-condition `40`-epoch CNS headline lane.

## Approved Fixed Recipe

- architecture id: `hybrid_resnet_ffno_ptychoblock_encoder`
- `encoder_variant=ffno_ptychoblock_encoder`
- `ffno_encoder_blocks=24`
- `ffno_encoder_modes=12`
- `ffno_encoder_share_weights=true`
- `ffno_encoder_gate_init=0.1`
- `ffno_encoder_norm=instance`
- `ffno_encoder_mlp_ratio=2.0`
- `ptychoblock_stage_count=2`
- `downsample_steps=2`
- `downsample_op=stride_conv`
- corrected rerun epoch budget: `20` epochs for both CDI and CNS rows

## Corrected Authority Roots

- Corrected CDI root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_20260507T073814Z/`
- Corrected CNS root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_20260507T082701Z/`
- Historical superseded CDI root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cdi_ffno_ptychoblock_encoder_20260506T183959Z/`
- Historical superseded CNS root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/runs/cns_ffno_ptychoblock_encoder_20260506T190421Z/`

The historical `2`-block roots remain debugging lineage only and must not be
cited as current authority.

## Results

### CDI

Corrected `20`-epoch row metrics:

- `amp_mae=0.036440`
- `phase_mae=0.080485`
- `amp_ssim=0.978317`
- `phase_ssim=0.990516`
- `amp_psnr=74.914807`
- `phase_psnr=68.102480`
- `amp_frc50=79.948351`
- `phase_frc50=79.675771`
- `parameter_count=3058278`
- `train_wall_time_sec=2660.467`
- `inference_time_sec=7.283`

Current read:

- versus `pinn_hybrid_resnet`, the encoder swap regresses every tracked
  aggregate metric, including `amp_mae +0.0095`, `phase_mae +0.0084`,
  `amp_ssim -0.0098`, `phase_ssim -0.0042`, `amp_frc50 -55.52`, and
  `phase_frc50 -27.12`
- versus `pinn_hybrid_resnet_encoder_spectral_only`, the corrected row is also
  worse on every tracked aggregate metric, including `amp_mae +0.0112`,
  `phase_mae +0.0190`, `amp_frc50 -57.75`, and `phase_frc50 -56.03`
- versus the historical `pinn_ffno` local-proxy row, the corrected encoder row
  is better across all tracked scalar metrics
- versus `pinn_hybrid_resnet_ffno_bottleneck`, phase MAE and phase PSNR improve
  slightly, but amplitude quality and both FRC50 scores regress sharply
- versus the historical `2`-block same-item CDI root, the corrected `24`-block
  rerun is worse on every tracked metric

### CNS

Corrected `20`-epoch row metrics:

- `relative_l2=0.049338`
- `err_nRMSE=0.049338`
- `err_RMSE=1.191816`
- `fRMSE_low=2.731857`
- `fRMSE_mid=0.201231`
- `fRMSE_high=0.553375`
- `parameter_count=3115014`
- `runtime_sec=1574.845`
- `peak_cuda_memory_bytes=2016402944`

Current read:

- versus `spectral_resnet_bottleneck_base`, the corrected row is worse on every
  tracked metric, including `relative_l2 +0.0163`, `fRMSE_mid +0.0299`, and
  `fRMSE_high +0.2912`
- versus `author_ffno_cns_base`, the corrected row is substantially worse on
  every tracked metric, including `relative_l2 +0.0296` and
  `fRMSE_high +0.4516`
- versus `fno_base`, the corrected row is also worse on every tracked metric,
  including `relative_l2 +0.0109` and `fRMSE_high +0.1205`
- the corrected row still beats `unet_strong` on every tracked metric
- versus the historical `2`-block same-item CNS root, the corrected `24`-block
  rerun is worse on every tracked metric

## Interpretation

The corrected shared-weight `24`-block FFNO-to-PtychoBlock encoder does not
transfer cleanly across either governed pillar. Under the locked `20`-epoch
mechanism budgets it weakens the SRU-Net shell on both CDI and matched-condition
capped CNS, and it is directionally worse than the superseded `2`-block
implementation that originally motivated the contract correction.

## Verification

- prerequisite presence gate: pass
- `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`:
  `61 passed, 62 deselected`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`:
  `12 passed, 143 deselected`
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`:
  `21 passed, 27 deselected`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno"`:
  `16 passed, 67 deselected`
- `pytest -q tests/torch/test_generator_registry.py`:
  `11 passed`
- `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`:
  `4 passed, 11 deselected`
- `python -m compileall -q ptycho_torch scripts/studies`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 2291 deselected`
- corrected CDI tracked PID `3215263` exited `0`, and the root carries
  `exit_code_proof.json` plus `artifact_freshness_check.json`
- corrected CNS tracked PID `3220360` exited `0`, and the root carries
  `exit_code_proof.json` plus `artifact_freshness_check.json`

## Residual Risks

- These are `20`-epoch mechanism probes only. They must not be promoted into
  the `40`-epoch CDI or CNS headline authorities.
- Comparisons against the current `40`-epoch CNS headline rows are useful for
  direction only because the probe budget is shorter.
- The historical `2`-block roots remain on disk and can still be misread unless
  downstream discovery surfaces keep labeling them as superseded lineage only.
