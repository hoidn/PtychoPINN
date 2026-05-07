# SRU-Net Encoder-Order FFNO-Vs-PtychoBlock Summary

- Date: `2026-05-07`
- Backlog item: `2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension`
- State: `completed`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/execution_plan.md`
- Item artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/`

## Claim Boundary

- CDI row `pinn_hybrid_resnet_ptychoblock_ffno_encoder` is
  `decision_support_append_only`.
- CNS row `hybrid_resnet_ptychoblock_ffno_encoder_cns` is
  `bounded_capped_decision_support_only`.
- Both fresh rows are `20`-epoch mechanism probes only. They do not replace the
  six-row CDI authority or the matched-condition `40`-epoch CNS headline lane.

## Fixed Encoder Recipe

- regular SRU-Net comparison anchor:
  - CDI: `pinn_hybrid_resnet`
  - CNS: `spectral_resnet_bottleneck_base`
- FFNO-first companion:
  - CDI: `pinn_hybrid_resnet_ffno_ptychoblock_encoder`
  - CNS: `hybrid_resnet_ffno_ptychoblock_encoder_cns`
- new reversed-order row:
  - CDI architecture id: `hybrid_resnet_ptychoblock_ffno_encoder`
  - CDI model id: `pinn_hybrid_resnet_ptychoblock_ffno_encoder`
  - CNS profile id: `hybrid_resnet_ptychoblock_ffno_encoder_cns`
- shared recipe fields held fixed across both mechanism rows:
  - `ffno_encoder_blocks=24`
  - `ffno_encoder_modes=12`
  - `ffno_encoder_share_weights=true`
  - `ffno_encoder_gate_init=0.1`
  - `ffno_encoder_norm=instance`
  - `ffno_encoder_mlp_ratio=2.0`
  - `ptychoblock_stage_count=2`
  - `downsample_steps=2`
  - `downsample_op=stride_conv`

## Fresh Authority Roots

- fresh CDI root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cdi_ptychoblock_ffno_encoder_20260507T094629Z/`
- fresh CNS root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/runs/cns_ptychoblock_ffno_encoder_20260507T100829Z/`
- machine-readable comparison bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/comparison_bundle.json`

## Results

### CDI

Fixed contract: same `lines128` CDI contract as the complete six-row bundle;
all mechanism rows use the bounded `20`-epoch probe budget.

| Row | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp FRC50 | Phase FRC50 |
|---|---:|---:|---:|---:|---:|---:|
| regular SRU-Net `pinn_hybrid_resnet` | 0.026939 | 0.072063 | 0.988114 | 0.994740 | 135.464 | 106.801 |
| `FFNO -> PtychoBlock` | 0.036440 | 0.080485 | 0.978317 | 0.990516 | 79.948 | 79.676 |
| `PtychoBlock -> FFNO` | 0.037627 | 0.078883 | 0.978619 | 0.990768 | 133.998 | 104.968 |

Current read:

- versus regular SRU-Net, the reversed-order row still regresses the main CDI
  aggregate metrics, including `amp_mae +0.0107`, `phase_mae +0.0068`,
  `amp_ssim -0.0095`, and `phase_ssim -0.0040`
- the reversed-order row nearly recovers the regular SRU-Net frequency-side
  behavior: `amp_frc50 -1.47` and `phase_frc50 -1.83` versus the regular
  anchor, compared with the FFNO-first companion's much larger
  `amp_frc50 -55.52` and `phase_frc50 -27.12`
- versus the FFNO-first companion, the reversed-order row is mixed on pure
  image-space scalars (`amp_mae +0.00119`, `amp_psnr -0.309`) but improves
  phase MAE (`-0.00160`), both SSIMs, and both FRC50 scores
  (`amp_frc50 +54.05`, `phase_frc50 +25.29`)
- runtime shifts are directionally favorable for the reversed-order row even
  though parameter count rises (`3,058,278 -> 4,912,998`): moving the FFNO
  stack onto the post-downsample feature map cuts train wall time
  (`2660.467s -> 1182.404s`) and inference time (`7.283s -> 3.055s`)

### CNS

Fixed contract: matched-condition PDEBench `2d_cfd_cns` capped lane
`history_len=5`, `512 / 64 / 64`; the mechanism rows remain bounded `20`-epoch
probes against the existing `40`-epoch headline authority.

| Row | relative_l2 | fRMSE_low | fRMSE_mid | fRMSE_high |
|---|---:|---:|---:|---:|
| regular SRU-Net `spectral_resnet_bottleneck_base` | 0.033069 | 1.854178 | 0.171350 | 0.262218 |
| `FFNO -> PtychoBlock` | 0.049338 | 2.731857 | 0.201231 | 0.553375 |
| `PtychoBlock -> FFNO` | 0.037419 | 2.040936 | 0.215718 | 0.457957 |

Current read:

- versus regular SRU-Net, the reversed-order row remains worse on every tracked
  CNS metric, including `relative_l2 +0.00435`, `fRMSE_low +0.187`,
  `fRMSE_mid +0.0444`, and `fRMSE_high +0.1957`
- versus the FFNO-first companion, the reversed-order row materially improves
  aggregate error and the low/high frequency bands:
  `relative_l2 -0.0119`, `fRMSE_low -0.6909`, and `fRMSE_high -0.0954`
- the only regression versus the FFNO-first companion is the mid band:
  `fRMSE_mid +0.0145`
- the reversed-order row is also materially lighter operationally than the
  FFNO-first companion on this lane: runtime drops
  (`1574.845s -> 1242.525s`) and peak CUDA memory drops
  (`2016402944 -> 838141440` bytes), despite a higher parameter count
  (`3,115,014 -> 4,969,734`)

## Interpretation

Encoder order clearly matters within this mechanism family. Moving the FFNO
stack after the two `PtychoBlock + downsample` stages is directionally better
than the completed `FFNO -> PtychoBlock` companion on both governed pillars:
it nearly restores CDI frequency fidelity and materially closes the CNS gap to
the regular SRU-Net anchor. That said, neither `20`-epoch mechanism row beats
the regular SRU-Net baseline overall, so the correct read remains bounded
mechanism evidence rather than a promotion candidate.

## Verification

- required presence gate: pass
- `pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"`:
  `62 passed, 62 deselected`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"`:
  `13 passed, 143 deselected`
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"`:
  `22 passed, 27 deselected`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"`:
  `18 passed, 67 deselected`
- `pytest -q tests/torch/test_generator_registry.py`:
  `12 passed`
- `pytest -q tests/torch/test_lightning_checkpoint.py -k "hybrid_resnet or ffno"`:
  `5 passed, 11 deselected`
- `python -m compileall -q ptycho_torch scripts/studies`: exit `0`
- `pytest -v -m integration`:
  `5 passed, 4 skipped, 2298 deselected, 2 warnings`
- fresh CDI tracked PID `3232115` exited `0`, and the root carries
  `exit_code_proof.json` plus `artifact_freshness_check.json`
- fresh CNS tracked PID `3234001` exited `0`, and the root carries
  `exit_code_proof.json` plus `artifact_freshness_check.json`

## Residual Risks

- The regular SRU-Net comparison authorities remain `40` epochs on CNS and the
  completed six-row CDI authority on `lines128`; these encoder-order rows are
  shorter bounded probes and must not be relabeled as headline replacements.
- The reversed-order implementation changes FFNO placement, so post-downsample
  channel width increases parameter count. That is an architectural consequence
  of the approved order swap, not a post-hoc tuning change.
- The encoder-order family is still under single-seed evidence on both pillars.
  If a later plan needs promotion-grade confidence, it must authorize longer or
  replicated reruns explicitly.
