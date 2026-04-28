# PDEBench CNS Hybrid-Spectral 2048-Cap Scaling Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap`
- Date: `2026-04-28`
- Status: implementation complete; capped `2048 / 256 / 256`, `40`-epoch scaling follow-up complete
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap`

This summary records capped decision-support evidence only. It does not create a benchmark-complete CNS claim or justify promoting either spectral profile into a default bundle.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- history contract: `concat u[t-2:t] -> u[t]`
- training loss: `mse`
- batch size: `4`
- `max_windows_per_trajectory=8`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- cap sequence:
  - `512 / 64 / 64`
  - `1024 / 128 / 128`
  - `2048 / 256 / 256`
- budget: `40` epochs at every cap

The shell stayed fixed for every row:

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

The only intended difference remained the finalist-defining shared spectral depth:

- `spectral_resnet_bottleneck_base`: `spectral_bottleneck_blocks=6`
- `spectral_resnet_bottleneck_shared_blocks10`: `spectral_bottleneck_blocks=10`

The comparison standard stayed the same as the prerequisite architecture-ablation tranche:

- lower `relative_l2`
- then lower `err_nRMSE`
- then lower `fRMSE_high`
- then lower parameter count

## Frozen References And Fresh Run

Frozen reference roots:

- `512cap`, `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
- `512cap`, `spectral_resnet_bottleneck_shared_blocks10`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- `1024cap`, both finalists:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`

Item-local reference manifests:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`

Inspect contract proof:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-20260428T190521Z/inspection_summary.json`

Fresh `2048cap` run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`

Scaling payloads:

- JSON:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.json`
- CSV:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/finalist_scaling_trend_512_1024_2048.csv`

The generated scaling payload preserves the fixed contract and records `allowed_contract_delta.delta_kind="split_counts_only"`.

## Scaling Outcomes

### `spectral_resnet_bottleneck_base`

Absolute metrics:

- `512cap`:
  `err_nRMSE=0.0607818365`, `relative_l2=0.0607818365`, `err_RMSE=1.4689105749`, `fRMSE_low=3.4476962090`, `fRMSE_mid=0.2428373098`, `fRMSE_high=0.3878487051`, `runtime_sec=1085.3778`
- `1024cap`:
  `err_nRMSE=0.0435009934`, `relative_l2=0.0435009934`, `err_RMSE=1.0374627113`, `fRMSE_low=2.4177434444`, `fRMSE_mid=0.2224166840`, `fRMSE_high=0.2982962728`, `runtime_sec=2162.7014`
- `2048cap`:
  `err_nRMSE=0.0421656668`, `relative_l2=0.0421656668`, `err_RMSE=1.0198297501`, `fRMSE_low=2.3713548183`, `fRMSE_mid=0.2230527103`, `fRMSE_high=0.3117601573`, `runtime_sec=4311.1880`

Delta summary:

- `512 -> 1024`:
  `err_nRMSE=-0.0172808431`, `relative_l2=-0.0172808431`, `err_RMSE=-0.4314478636`, `fRMSE_low=-1.0299527645`, `fRMSE_mid=-0.0204206258`, `fRMSE_high=-0.0895524323`, `runtime_sec=+1077.3236`
- `1024 -> 2048`:
  `err_nRMSE=-0.0013353266`, `relative_l2=-0.0013353266`, `err_RMSE=-0.0176329613`, `fRMSE_low=-0.0463886261`, `fRMSE_mid=+0.0006360263`, `fRMSE_high=+0.0134638846`, `runtime_sec=+2148.4866`

Improvement per added training trajectory:

- `512 -> 1024`:
  `err_nRMSE=3.3751646697e-05`, `relative_l2=3.3751646697e-05`, `fRMSE_high=1.7490709433e-04`
- `1024 -> 2048`:
  `err_nRMSE=1.3040298650e-06`, `relative_l2=1.3040298650e-06`, `fRMSE_high=-1.3148324797e-05`

Interpretation:

- the base row improved strongly from `512cap` to `1024cap`
- the `2048cap` extension kept a small aggregate improvement over `1024cap`
- the added `2048cap` data no longer improved `fRMSE_high`; it regressed slightly while runtime nearly doubled

### `spectral_resnet_bottleneck_shared_blocks10`

Absolute metrics:

- `512cap`:
  `err_nRMSE=0.0543919504`, `relative_l2=0.0543919504`, `err_RMSE=1.3144866228`, `fRMSE_low=3.0832421780`, `fRMSE_mid=0.2326273471`, `fRMSE_high=0.3401203752`, `runtime_sec=1358.9588`
- `1024cap`:
  `err_nRMSE=0.0445732661`, `relative_l2=0.0445732661`, `err_RMSE=1.0630357265`, `fRMSE_low=2.4827439785`, `fRMSE_mid=0.2167904228`, `fRMSE_high=0.2939873338`, `runtime_sec=2703.7176`
- `2048cap`:
  `err_nRMSE=0.0504393019`, `relative_l2=0.0504393019`, `err_RMSE=1.2199381590`, `fRMSE_low=2.8665962219`, `fRMSE_mid=0.2082921267`, `fRMSE_high=0.2969700098`, `runtime_sec=5419.3110`

Delta summary:

- `512 -> 1024`:
  `err_nRMSE=-0.0098186843`, `relative_l2=-0.0098186843`, `err_RMSE=-0.2514508963`, `fRMSE_low=-0.6004981995`, `fRMSE_mid=-0.0158369243`, `fRMSE_high=-0.0461330414`, `runtime_sec=+1344.7588`
- `1024 -> 2048`:
  `err_nRMSE=+0.0058660358`, `relative_l2=+0.0058660358`, `err_RMSE=+0.1569024324`, `fRMSE_low=+0.3838522434`, `fRMSE_mid=-0.0084982961`, `fRMSE_high=+0.0029826760`, `runtime_sec=+2715.5933`

Improvement per added training trajectory:

- `512 -> 1024`:
  `err_nRMSE=1.9177117792e-05`, `relative_l2=1.9177117792e-05`, `fRMSE_high=9.0103596449e-05`
- `1024 -> 2048`:
  `err_nRMSE=-5.7285506050e-06`, `relative_l2=-5.7285506050e-06`, `fRMSE_high=-2.9127695600e-06`

Interpretation:

- the `10`-block shared row also improved from `512cap` to `1024cap`
- the `2048cap` extension reversed that trend on aggregate metrics and low-band error
- only `fRMSE_mid` continued improving at `2048cap`, while `fRMSE_high` also drifted slightly worse and runtime again nearly doubled

## Bounded Interpretation

- The larger `2048cap` follow-up does not overturn the prior `1024cap` directional result.
- By the preserved ranking rule, `spectral_resnet_bottleneck_base` remains the stronger aggregate local reference at `2048cap`:
  - lower `relative_l2`
  - lower `err_nRMSE`
  - lower `err_RMSE`
  - lower `fRMSE_low`
  - lower runtime
- `spectral_resnet_bottleneck_shared_blocks10` still keeps narrower higher-frequency advantages at `2048cap`:
  - lower `fRMSE_mid`
  - lower `fRMSE_high`
- The scaling separation is not ambiguous in the aggregate direction:
  - the base row improved slightly beyond `1024cap`
  - the deeper shared row regressed beyond `1024cap`
- The fresh helper-generated improvement-per-added-trajectory values strengthen the same bounded take:
  - base stays positive for aggregate improvement from `1024 -> 2048`
  - shared-blocks10 turns negative on `err_nRMSE` and `relative_l2` over the same transition

## Reporting Warning

- The required scaling payloads were written successfully.
- The optional cross-run gallery was skipped with:
  `reason="target_mismatch"` and `message="target mismatch between spectral_resnet_bottleneck_base@512cap and spectral_resnet_bottleneck_base@1024cap"`
- Per `REPORTING-ARTIFACT-BOUNDARY-001`, this warning does not invalidate the completed metrics/manifests/scaling payloads.

## Verification Artifacts

Preflight and helper evidence:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_pytest.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/preflight_compileall.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/runner_scaling_trend.log`

Fresh completion verification:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_pytest.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_compileall.log`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/verification/final_artifact_validation.log`

Run-completion proof:

- launch sidecar:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z.launch/exit_code.txt`
- required fresh metrics:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z/metrics_spectral_resnet_bottleneck_base.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z/metrics_spectral_resnet_bottleneck_shared_blocks10.json`

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify changing the canonical CNS shell
- it does not justify promoting `spectral_resnet_bottleneck_shared_blocks10` into a default profile
- it does not justify any paper-facing competitiveness claim on its own
- the bounded take is:
  - `spectral_resnet_bottleneck_base` remains the better aggregate larger-cap local reference
  - `spectral_resnet_bottleneck_shared_blocks10` remains a manual higher-frequency trade-off profile
  - the `2048cap` follow-up does not strengthen a case for deeper shared spectral promotion
