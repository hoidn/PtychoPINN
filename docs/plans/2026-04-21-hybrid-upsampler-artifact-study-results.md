# Hybrid Upsampler Artifact Study Results

## Scope

- Date: 2026-04-21
- Task: PDEBench `2d_cfd_cns`
- Shell: canonical post-skip-add CNS Hybrid shell
- Run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z`
- Evidence scope: `smoke_feasibility_only`
- Interpretation boundary: capped readiness evidence only, not benchmark
  evidence

This study reran the earlier upsampler compare after skip-add had already been
promoted into the canonical CNS shell. The fairness boundary was strict: same
`hybrid_resnet_cns` shell, same `2` downsamples, same skip-add topology, same
local bottleneck, same data slice, same `10`-epoch MSE training budget. Only
`hybrid_upsampler` changed.

Compared rows:

- `hybrid_resnet_cns_transpose`
- `hybrid_resnet_cns_interp_bilinear_conv`
- `hybrid_resnet_cns_pixelshuffle`

## Run Contract

- trajectories: `512 / 64 / 64` train/val/test
- max windows per trajectory: `8`
- windows: `4096 / 512 / 512`
- epochs: `10`
- batch size: `4`
- loss: `mse`

## Metrics

From
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z/comparison_summary.json`:

- `hybrid_resnet_cns_transpose`
  - final train loss: `0.0338358228`
  - `err_RMSE=3.1529943943`
  - `err_nRMSE=0.1304672956`
  - `relative_l2=0.1304672956`
  - `fRMSE_high=0.7634432316`

- `hybrid_resnet_cns_interp_bilinear_conv`
  - final train loss: `0.0350025710`
  - `err_RMSE=2.3944778442`
  - `err_nRMSE=0.0990807489`
  - `relative_l2=0.0990807489`
  - `fRMSE_high=1.1363334656`

- `hybrid_resnet_cns_pixelshuffle`
  - final train loss: `0.0325912039`
  - `err_RMSE=2.3277606964`
  - `err_nRMSE=0.0963200703`
  - `relative_l2=0.0963200703`
  - `fRMSE_high=0.8639594913`

## Visual Read

Rendered galleries:

- prediction gallery:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z/gallery_sample0_full_compare.png`
- error gallery:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z/gallery_sample0_full_compare_error.png`

Observed qualitative pattern:

- transpose still had the best high-frequency metric, but retained more obvious
  checkerboard-like texture
- bilinear reduced the classic transpose checkerboard, but the resulting errors
  looked directionally biased and stripe-like rather than clean
- pixelshuffle gave the best aggregate error and avoided the worst bilinear
  high-frequency penalty

## Decision

Promotion decision:

- promote `pixelshuffle` into the canonical CNS Hybrid row
- keep bilinear manual-only
- keep the old transpose decoder available only as
  `hybrid_resnet_cns_transpose` for study reproducibility and future ablations

Reason:

- `pixelshuffle` was best on aggregate capped CNS metrics
- its high-frequency penalty was materially smaller than bilinear
- it is the only post-skip-add upsampler variant from this study that looks
  worth carrying forward as the default CNS Hybrid decoder

## Boundary

This study does not justify any manuscript or benchmark claim by itself. It is
decision-support evidence for the local CNS default profile only.
