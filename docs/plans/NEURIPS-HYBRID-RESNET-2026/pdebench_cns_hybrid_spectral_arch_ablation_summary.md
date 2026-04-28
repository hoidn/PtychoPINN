# PDEBench CNS Hybrid-Spectral Architecture Ablation Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation`
- Date: `2026-04-28`
- Status: implementation complete; capped CNS hybrid-spectral architecture ablation complete
- Scope: keep the canonical CNS shell and capped contract fixed, rerun fresh `10`/`40`-epoch sharing pilots plus a fresh `40`-epoch shared-depth pilot, then confirm the unique tranche finalists on the larger `1024 / 128 / 128` cap
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation`

This summary records capped decision-support evidence only. It does not create a benchmark-complete CNS claim or justify promoting any manual spectral profile into a default bundle.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- history contract: `concat u[t-2:t] -> u[t]`
- training loss: `mse`
- batch size: `4`
- `max_windows_per_trajectory=8`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- pilot split: `512 / 64 / 64` trajectories
- finalist-confirmation split: `1024 / 128 / 128` trajectories
- budgets: `10` and `40` epochs

The shell stayed fixed for every fresh row:

- `base_model="spectral_resnet_bottleneck_net"`
- `hidden_channels=32`
- `fno_modes=12`
- `fno_blocks=4`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`
- `spectral_bottleneck_modes=12`
- `spectral_bottleneck_gate_init=0.1`
- `spectral_bottleneck_gate_mode="shared"`

The only intended ablation axes were:

- weight sharing:
  - `spectral_resnet_bottleneck_base`: `spectral_bottleneck_share_weights=True`
  - `spectral_resnet_bottleneck_noshare`: `spectral_bottleneck_share_weights=False`
- shared bottleneck depth:
  - `spectral_resnet_bottleneck_base`: `spectral_bottleneck_blocks=6`
  - `spectral_resnet_bottleneck_shared_blocks8`: `spectral_bottleneck_blocks=8`
  - `spectral_resnet_bottleneck_shared_blocks10`: `spectral_bottleneck_blocks=10`

No production-code patching was needed after the required deterministic preflight checks.

## Fresh Run Roots

- inspect snapshot:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/inspect-20260428T054400Z`
- sharing `10` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
- sharing `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
- shared-depth `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- finalist confirmation `1024 / 128 / 128`, `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`

Reference manifests:

- `10` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/reference_runs_10ep.json`
- `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/reference_runs_40ep.json`

## Sharing Outcomes

Anchored sidecars:

- `10` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_sharing_10ep_against_existing.json`
- `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_sharing_40ep_against_existing.json`

The sharing sidecars were regenerated from the frozen `reference_runs_10ep.json`
and `reference_runs_40ep.json` manifests during implementation-review repair, so
they now keep the fresh spectral rows separate from the frozen FNO/U-Net context
rows and the optional `10`-epoch hybrid context row. This bookkeeping repair did
not change the study interpretation below.

Observed `10`-epoch sharing outcome:

- `spectral_resnet_bottleneck_noshare`:
  `err_nRMSE=0.0795788`, `err_RMSE=1.92317`, `fRMSE_low=4.4364`, `fRMSE_mid=0.3549`, `fRMSE_high=0.7859`
- `spectral_resnet_bottleneck_base`:
  `err_nRMSE=0.0830028`, `err_RMSE=2.00592`, `fRMSE_low=4.6498`, `fRMSE_mid=0.3594`, `fRMSE_high=0.7512`

Interpretation at `10` epochs:

- disabling weight sharing improved aggregate denormalized error and low/mid-band Fourier error
- the non-shared row did not improve the full metric family because `fRMSE_high` worsened slightly (`0.7859` vs `0.7512`)
- the bounded ranking rule still picked `spectral_resnet_bottleneck_noshare` because `relative_l2` and `err_nRMSE` are the primary ordering keys

Observed `40`-epoch sharing outcome:

- `spectral_resnet_bottleneck_base`:
  `err_nRMSE=0.0607818`, `err_RMSE=1.46891`, `fRMSE_low=3.4477`, `fRMSE_mid=0.2428`, `fRMSE_high=0.3878`
- `spectral_resnet_bottleneck_noshare`:
  `err_nRMSE=0.0645084`, `err_RMSE=1.55897`, `fRMSE_low=3.6568`, `fRMSE_mid=0.2468`, `fRMSE_high=0.4344`

Interpretation at `40` epochs:

- the sharing result flipped relative to `10` epochs
- the shared base row beat the non-shared row on every tracked eval metric
- the fixed-shell longer capped lane therefore does not support promoting the earlier non-shared `10`-epoch result into a stronger default claim

## Shared-Depth Outcome

Anchored sidecar:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_depth_40ep_against_existing.json`

Observed `40`-epoch shared-depth outcome:

- `spectral_resnet_bottleneck_shared_blocks10`:
  `err_nRMSE=0.0543920`, `err_RMSE=1.31449`, `fRMSE_low=3.0832`, `fRMSE_mid=0.2326`, `fRMSE_high=0.3401`
- `spectral_resnet_bottleneck_base`:
  `err_nRMSE=0.0552611`, `err_RMSE=1.33549`, `fRMSE_low=3.1300`, `fRMSE_mid=0.2411`, `fRMSE_high=0.3532`
- `spectral_resnet_bottleneck_shared_blocks8`:
  `err_nRMSE=0.0767923`, `err_RMSE=1.85584`, `fRMSE_low=4.3862`, `fRMSE_mid=0.2668`, `fRMSE_high=0.3550`

Interpretation:

- increasing the shared bottleneck depth from `6` to `10` blocks improved every tracked eval metric over the shared base row on the `512 / 64 / 64` capped lane
- the `8`-block row was clearly worse than both `6` and `10`
- `spectral_resnet_bottleneck_shared_blocks10` became the fresh shared-depth finalist even though it added parameters and runtime

## Larger-Cap Finalist Confirmation

Selection payload:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/selected_finalists_1024cap.json`

The fresh `40`-epoch sharing and shared-depth tranches nominated different winners, so both unique finalists were rerun together on the larger `1024 / 128 / 128` cap:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_shared_blocks10`

Within-run confirmation sidecar:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_finalists_1024cap_40ep_within_run.json`

Observed `1024 / 128 / 128` outcome:

- `spectral_resnet_bottleneck_base`:
  `err_nRMSE=0.0435010`, `err_RMSE=1.03746`, `fRMSE_low=2.4177`, `fRMSE_mid=0.2224`, `fRMSE_high=0.2983`
- `spectral_resnet_bottleneck_shared_blocks10`:
  `err_nRMSE=0.0445733`, `err_RMSE=1.06304`, `fRMSE_low=2.4827`, `fRMSE_mid=0.2168`, `fRMSE_high=0.2940`

Interpretation on the larger cap:

- the shared base row recovered the aggregate lead on the stronger slice:
  - lower `relative_l2` / `err_nRMSE`
  - lower `err_RMSE`
  - lower `fRMSE_low`
- the `10`-block row retained only a narrow higher-frequency edge:
  - lower `fRMSE_mid`
  - lower `fRMSE_high`
- the depth extension also remained materially slower:
  `2703.72s` vs `2162.70s`

Finalist delta payload:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/finalist_delta_1024cap.json`

From the finalist delta payload:

- `spectral_resnet_bottleneck_base` improved from the `512 / 64 / 64` sharing tranche to the `1024 / 128 / 128` confirmation tranche by:
  - `err_nRMSE: 0.0607818 -> 0.0435010`
  - `fRMSE_high: 0.3878 -> 0.2983`
- `spectral_resnet_bottleneck_shared_blocks10` improved from the `512 / 64 / 64` depth tranche to the `1024 / 128 / 128` confirmation tranche by:
  - `err_nRMSE: 0.0543920 -> 0.0445733`
  - `fRMSE_high: 0.3401 -> 0.2940`

The confirmation pass therefore narrows the claim:

- deeper shared spectral bottlenecks can help on the capped `512 / 64 / 64` lane
- but the larger capped confirmation does not support replacing the shared base row as the aggregate local reference
- the deeper row remains a manual follow-up for users who care more about `fRMSE_mid/high` than aggregate or low-band error, and who accept the extra runtime

## Verification Artifacts

Preflight verification:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/preflight_pytest.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/preflight_compileall.log`

Fresh final verification for the repo state that publishes this summary:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/final_pytest.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/verification/final_compileall.log`

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify changing the canonical CNS shell
- it does not justify promoting `spectral_resnet_bottleneck_shared_blocks10` into a default profile
- the current bounded take is:
  - the shared base spectral row is the better aggregate larger-cap local reference
  - the deeper shared row is a manual higher-frequency trade-off
  - the non-shared row did not hold its `10`-epoch aggregate advantage at `40` epochs
